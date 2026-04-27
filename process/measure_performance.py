"""
RQ4 + RQ5 性能测量脚本。

直接跑一遍完整的 ATHENA pipeline，分模块计时：
  - Table VIII: End-to-end (train time, inference time, peak memory)
  - Table IX: Module-level overhead
  - Table X: Detection only vs Detection + Interpretation (ablation)

用法:
  python -m process.measure_performance
"""
from __future__ import annotations
import os
import sys
import time
import json
import pickle
import threading
import numpy as np
import torch
import yaml
from pathlib import Path
from collections import OrderedDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# 配置
# ============================================================

DATASET_NAME = "theia"
SCENE_NAME = "theia311"
GLOBAL_ID = "perf"  # 性能测量专用 ID
EMBEDDER_NAME = "gcc_dev"
CLASSIFY_NAME = "topk"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 内存采样
# ============================================================

class MemSampler:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.peak_mb = 0.0
        self.samples = []
        self._stop = threading.Event()

    def start(self):
        self._stop.clear()
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        return self

    def _run(self):
        try:
            import psutil
            proc = psutil.Process(os.getpid())
            while not self._stop.is_set():
                rss = proc.memory_info().rss / (1024 * 1024)
                self.samples.append(rss)
                if rss > self.peak_mb:
                    self.peak_mb = rss
                self._stop.wait(self.interval)
        except Exception:
            pass

    def stop(self):
        self._stop.set()
        time.sleep(0.1)
        return self.peak_mb

    @property
    def avg_mb(self):
        return sum(self.samples) / len(self.samples) if self.samples else 0


def calc_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return {"acc": round(acc * 100, 2), "prec": round(prec * 100, 2),
            "rec": round(rec * 100, 2), "f1": round(f1 * 100, 2),
            "fpr": round(fpr * 100, 3), "tp": tp, "fp": fp, "tn": tn, "fn": fn}


# ============================================================
# 主流程：完整 ATHENA pipeline
# ============================================================

def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    path_map = config["remote"]["path_map"]

    from process.datahandlers import get_handler
    from process.embedders import get_embedder_by_name
    from process.classfy import get_classfy

    timing = OrderedDict()
    sampler = MemSampler(0.5).start()

    print("=" * 70)
    print(f"ATHENA Performance Measurement ({DATASET_NAME}/{SCENE_NAME})")
    print("=" * 70)

    # ========== 训练阶段 ==========
    print("\n[Phase 1] Snapshot Construction ...")
    t0 = time.perf_counter()
    handler = get_handler(DATASET_NAME, True, path_map, scene_name=SCENE_NAME)
    handler.load()
    handler.build_graph(GLOBAL_ID)
    timing["Snapshot Construction"] = round(time.perf_counter() - t0, 2)
    mem_snapshot = sampler.peak_mb
    print(f"  → {timing['Snapshot Construction']:.2f}s, peak_mem={mem_snapshot:.0f}MB")

    print("\n[Phase 2] Contrastive Learning ...")
    t0 = time.perf_counter()
    embedder_cls = get_embedder_by_name(EMBEDDER_NAME)
    default_model = getattr(embedder_cls, "_default_path", "embedder_model.pth")
    _p = Path(default_model)
    model_path = f"{_p.stem}_{GLOBAL_ID}{_p.suffix}"
    try:
        embedder = embedder_cls(handler.snapshots, model_path=model_path)
    except TypeError:
        embedder = embedder_cls(handler.snapshots)
    embedder.train()
    snapshot_embeddings = embedder.get_snapshot_embeddings()
    timing["Contrastive Learning"] = round(time.perf_counter() - t0, 2)
    mem_cl = sampler.peak_mb
    print(f"  → {timing['Contrastive Learning']:.2f}s, peak_mem={mem_cl:.0f}MB")

    print("\n[Phase 3] MLP Train ...")
    t0 = time.perf_counter()
    benign = snapshot_embeddings[handler.benign_idx_start:handler.benign_idx_end + 1]
    clf = get_classfy(CLASSIFY_NAME, gid=GLOBAL_ID)
    clf.train(benign)
    timing["MLP Train"] = round(time.perf_counter() - t0, 2)
    print(f"  → {timing['MLP Train']:.2f}s")

    total_train = timing["Snapshot Construction"] + timing["Contrastive Learning"] + timing["MLP Train"]

    # ========== 推理阶段 ==========
    # 加载快照
    snapshot_file = f"snapshot_data_{GLOBAL_ID}.pkl"
    with open(snapshot_file, 'rb') as f:
        snap_data = pickle.load(f)
    all_snapshots = snap_data['all_snapshots']
    mal_start = snap_data['malicious_idx_start']
    mal_end = snap_data['malicious_idx_end']
    mal_snaps = all_snapshots[mal_start:mal_end + 1]

    # 构造 true labels
    true_labels = np.zeros(len(mal_snaps), dtype=int)
    for i, snap in enumerate(mal_snaps):
        has_mal = any(int(v.attributes().get("label", 0)) == 1
                      for v in snap.vs)
        if has_mal:
            true_labels[i] = 1

    print(f"\n[推理] 恶意快照段: {mal_start}~{mal_end}, 共 {len(mal_snaps)} 个, 其中 {true_labels.sum()} 个含恶意节点")

    # 4. Anomaly Detection（嵌入 + 分类）
    print("\n[Phase 4] Anomaly Detection ...")
    t0 = time.perf_counter()
    # 重新加载模型做推理
    embedder_infer = embedder_cls.load(snapshot_sequence=all_snapshots, path=model_path)
    snap_embs = embedder_infer.get_snapshot_embeddings()
    mal_embs = snap_embs[mal_start:mal_end + 1]

    clf_infer = get_classfy(CLASSIFY_NAME, gid=GLOBAL_ID)
    clf_infer.load()
    pred_labels, _ = clf_infer.predict(mal_embs)
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()
    pred_labels = np.asarray(pred_labels, dtype=int)
    timing["Anomaly Detection"] = round(time.perf_counter() - t0, 3)
    print(f"  → {timing['Anomaly Detection']:.3f}s")

    # Detection only 指标
    metrics_detect = calc_metrics(true_labels, pred_labels)
    print(f"  Detection only: Acc={metrics_detect['acc']:.2f}% Prec={metrics_detect['prec']:.2f}% "
          f"Rec={metrics_detect['rec']:.2f}% F1={metrics_detect['f1']:.2f}% FPR={metrics_detect['fpr']:.3f}%")

    # 5. Anomaly Interpretation（语义映射 + LCS 过滤）
    print("\n[Phase 5] Anomaly Interpretation ...")
    t0 = time.perf_counter()
    pred_with_interp = pred_labels.copy()
    try:
        from process.technique_semantic_mapper import TechniqueSemanticMapper, snapshot_to_query
        mapper = TechniqueSemanticMapper(
            triples_path=os.path.join(os.path.dirname(__file__), "data/technique_triples_raw.json"),
            top_k=5, threshold=0.0,
        )
        # 映射每个阳性快照到 ATT&CK 技术
        idx_pos = np.where(pred_labels == 1)[0]
        tech_codes = []
        for i in idx_pos:
            query = snapshot_to_query(mal_snaps[i])
            if query:
                result = mapper.predict_top(query)
                tech_codes.append(result[0] if result else "UNKNOWN")
            else:
                tech_codes.append("UNKNOWN")
        print(f"  映射 {len(idx_pos)} 个阳性快照 → {len([c for c in tech_codes if c != 'UNKNOWN'])} 个技术")

        # LCS 序列过滤
        seq_lib_path = os.path.join(os.path.dirname(__file__), "technique_sequences.txt")
        if os.path.exists(seq_lib_path):
            with open(seq_lib_path) as f:
                lib_seqs = [l.strip().split(",") for l in f if l.strip()]
            pred_seq = [c for c in tech_codes if c != "UNKNOWN"]
            if pred_seq and lib_seqs:
                from difflib import SequenceMatcher
                best_ratio = max(
                    SequenceMatcher(None, pred_seq, lib_s).ratio()
                    for lib_s in lib_seqs
                )
                print(f"  LCS 最佳匹配率: {best_ratio:.3f}")
                if best_ratio < 0.3:
                    pred_with_interp[idx_pos] = 0
                    print(f"  → 低于阈值 0.3, 清零 {len(idx_pos)} 个阳性")
    except Exception as e:
        print(f"  [WARN] 解释模块异常: {e}")
        import traceback
        traceback.print_exc()

    timing["Anomaly Interpretation"] = round(time.perf_counter() - t0, 3)
    print(f"  → {timing['Anomaly Interpretation']:.3f}s")

    # Detection + Interpretation 指标
    metrics_full = calc_metrics(true_labels, pred_with_interp)
    print(f"  Detection + Interpretation: Acc={metrics_full['acc']:.2f}% Prec={metrics_full['prec']:.2f}% "
          f"Rec={metrics_full['rec']:.2f}% F1={metrics_full['f1']:.2f}% FPR={metrics_full['fpr']:.3f}%")

    total_inference = timing["Anomaly Detection"] + timing["Anomaly Interpretation"]
    peak_mem = sampler.stop()

    # ============================================================
    # 输出结果
    # ============================================================

    print("\n\n" + "=" * 70)
    print("Table VIII: End-to-End Performance (ATHENA)")
    print("=" * 70)
    print(f"  Train:       {total_train:.1f}s")
    print(f"  Inference:   {total_inference:.3f}s")
    print(f"  Peak Memory: {peak_mem:.0f} MB")

    print("\n" + "=" * 70)
    print("Table IX: Module-Level Overhead of ATHENA")
    print("=" * 70)
    print(f"{'Module':<28} {'Time (s)':>10} {'Memory (MB)':>12}")
    print("-" * 54)
    for mod, t in timing.items():
        print(f"{mod:<28} {t:>10.2f}")
    print("-" * 54)
    print(f"{'Total Train':<28} {total_train:>10.2f}")
    print(f"{'Total Inference':<28} {total_inference:>10.3f}")
    print(f"{'Peak Memory':<28} {'':>10} {peak_mem:>11.0f}")

    print("\n" + "=" * 70)
    print("Table X: Impact of Global Anomaly Interpretation")
    print("=" * 70)
    print(f"{'Configuration':<32} {'Acc':>7} {'Prec':>7} {'F1':>7} {'Rec':>7} {'FPR':>7}")
    print("-" * 72)
    m = metrics_detect
    print(f"{'Detection only':<32} {m['acc']:6.2f}% {m['prec']:6.2f}% {m['f1']:6.2f}% {m['rec']:6.2f}% {m['fpr']:6.3f}%")
    m = metrics_full
    print(f"{'Detection + Interpretation':<32} {m['acc']:6.2f}% {m['prec']:6.2f}% {m['f1']:6.2f}% {m['rec']:6.2f}% {m['fpr']:6.3f}%")

    # 保存
    output = {
        "dataset": DATASET_NAME, "scene": SCENE_NAME, "device": str(DEVICE),
        "table_viii": {
            "train_s": round(total_train, 1),
            "inference_s": round(total_inference, 3),
            "peak_mem_mb": round(peak_mem, 0),
        },
        "table_ix": {mod: {"time_s": t} for mod, t in timing.items()},
        "table_x": {
            "detection_only": metrics_detect,
            "detection_interpretation": metrics_full,
        },
        "timing_detail": dict(timing),
        "n_snapshots": len(all_snapshots),
        "n_mal_snapshots": len(mal_snaps),
        "n_true_positive": int(true_labels.sum()),
    }

    out_path = os.path.join(os.path.dirname(__file__), "..", "rq4_rq5_performance.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
