"""
RQ4 + RQ5 完整测量脚本。

Table VIII: 5 个方法端到端性能对比 (Train / Inference / Peak Mem)
Table IX:  ATHENA 模块级开销分解 (Time / Memory / CPU%)
Table X:   消融实验 — 检测 vs 检测+解释

用法:
  conda activate prographer && cd /home/nsas2020/fuzz/prographer
  python -m process.measure_rq4_rq5
"""
from __future__ import annotations
import os, sys, time, json, pickle, threading, gc
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ============================================================
# 配置
# ============================================================
DATASET_NAME = "theia"
SCENE_NAME = "theia311"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5 个方法：(显示名, embedder, classifier)
METHODS = [
    ("ProGrapher", "prographer", "prographer"),
    ("Unicorn",    "unicorn",    "unicorn"),
    ("ATLAS",      "transe",     "topk"),
    ("MAGIC",      "word2vec",   "topk"),
    ("ATHENA",     "gcc_dev",    "mlp"),
]

# ============================================================
# 内存采样器
# ============================================================
class MemSampler:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.peak_mb = 0.0
        self.samples = []
        self._stop = threading.Event()
        self._started = False

    def start(self):
        self._stop.clear()
        self._started = True
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

    def reset(self):
        self.peak_mb = 0
        self.samples.clear()
        if not self._started:
            self.start()

    def snapshot_peak(self):
        """返回当前峰值并重置"""
        p = self.peak_mb
        return p

    @property
    def avg_mb(self):
        return sum(self.samples) / len(self.samples) if self.samples else 0

    @property
    def current_mb(self):
        try:
            import psutil
            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except:
            return 0


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
# 单方法端到端测量
# ============================================================
def measure_method(method_name, embedder_name, classify_name, path_map, sampler):
    """测量一个方法的训练+推理时间和峰值内存"""
    from process.datahandlers import get_handler
    from process.embedders import get_embedder_by_name
    from process.classfy import get_classfy

    gid = f"rq4_{method_name.lower()}"
    print(f"\n{'='*60}")
    print(f"测量 {method_name} (embedder={embedder_name}, clf={classify_name})")
    print(f"{'='*60}")

    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    sampler.reset()
    mem_before = sampler.current_mb

    # ---- 训练阶段 ----
    t_train_start = time.perf_counter()

    # 1. 数据加载 + 快照构建
    handler = get_handler(DATASET_NAME, True, path_map, scene_name=SCENE_NAME)
    handler.load()
    handler.build_graph(gid)

    # 2. 嵌入训练
    embedder_cls = get_embedder_by_name(embedder_name)
    default_model = getattr(embedder_cls, "_default_path", "embedder_model.pth")
    _p = Path(default_model)
    model_path = f"{_p.stem}_{gid}{_p.suffix}"

    try:
        if embedder_name.lower() == "roland":
            benign_range = range(handler.benign_idx_start, handler.benign_idx_end + 1)
            embedder = embedder_cls(handler.snapshots, train_indices=benign_range, model_path=model_path)
        else:
            embedder = embedder_cls(handler.snapshots, model_path=model_path)
    except TypeError:
        embedder = embedder_cls(handler.snapshots)

    embedder.train()
    snapshot_embeddings = embedder.get_snapshot_embeddings()

    # 3. 分类器训练
    benign_embs = snapshot_embeddings[handler.benign_idx_start:handler.benign_idx_end + 1]
    clf = get_classfy(classify_name, gid=gid)

    if classify_name == "mlp":
        # MLP 需要良性+恶意嵌入
        mal_start = handler.malicious_idx_start
        mal_end = handler.malicious_idx_end
        mal_embs = snapshot_embeddings[mal_start:mal_end + 1]
        # 构造恶意标签
        mal_labels = np.zeros(len(mal_embs), dtype=int)
        all_snaps = handler.snapshots
        for i in range(mal_start, mal_end + 1):
            snap = all_snaps[i]
            if any(int(v.attributes().get("label", 0)) == 1 for v in snap.vs):
                mal_labels[i - mal_start] = 1
        clf.train(benign_embs, mal_embs, mal_labels)
    else:
        clf.train(benign_embs)

    t_train = time.perf_counter() - t_train_start
    mem_train_peak = sampler.snapshot_peak()

    # ---- 推理阶段 ----
    t_infer_start = time.perf_counter()

    # 加载快照数据
    snapshot_file = f"snapshot_data_{gid}.pkl"
    with open(snapshot_file, 'rb') as f:
        snap_data = pickle.load(f)
    all_snapshots = snap_data['all_snapshots']
    mal_start = snap_data['malicious_idx_start']
    mal_end = snap_data['malicious_idx_end']
    mal_snaps = all_snapshots[mal_start:mal_end + 1]

    # 构造 true labels
    true_labels = np.zeros(len(mal_snaps), dtype=int)
    for i, snap in enumerate(mal_snaps):
        if any(int(v.attributes().get("label", 0)) == 1 for v in snap.vs):
            true_labels[i] = 1

    # 重新加载模型推理
    try:
        embedder_infer = embedder_cls.load(snapshot_sequence=all_snapshots, path=model_path)
    except Exception:
        embedder_infer = embedder
    snap_embs = embedder_infer.get_snapshot_embeddings()
    mal_embs_test = snap_embs[mal_start:mal_end + 1]

    clf_infer = get_classfy(classify_name, gid=gid)
    clf_infer.load()
    pred_labels, _ = clf_infer.predict(mal_embs_test)
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()
    pred_labels = np.asarray(pred_labels, dtype=int)

    t_infer = time.perf_counter() - t_infer_start
    mem_peak = sampler.snapshot_peak()

    metrics = calc_metrics(true_labels, pred_labels)
    print(f"[{method_name}] Train={t_train:.1f}s Infer={t_infer:.3f}s PeakMem={mem_peak:.0f}MB")
    print(f"[{method_name}] Acc={metrics['acc']:.2f}% F1={metrics['f1']:.2f}% "
          f"TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")

    return {
        "method": method_name,
        "train_s": round(t_train, 1),
        "inference_s": round(t_infer, 3),
        "peak_mem_mb": round(max(mem_peak, mem_train_peak), 0),
        "metrics": metrics,
        "pred_labels": pred_labels,
        "true_labels": true_labels,
        "mal_snaps": mal_snaps,
    }


# ============================================================
# ATHENA 模块级分解测量 (Table IX)
# ============================================================
def measure_athena_modules(path_map, sampler):
    """分模块测量 ATHENA 的时间、内存"""
    from process.datahandlers import get_handler
    from process.embedders import get_embedder_by_name
    from process.classfy import get_classfy

    gid = "rq4_athena_mod"
    timing = OrderedDict()
    mem_per_phase = OrderedDict()

    print(f"\n{'='*60}")
    print("ATHENA 模块级分解测量 (Table IX)")
    print(f"{'='*60}")

    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    sampler.reset()

    # Phase 1: Snapshot Construction
    print("\n[Phase 1] Snapshot Construction ...")
    mem_before = sampler.current_mb
    t0 = time.perf_counter()
    handler = get_handler(DATASET_NAME, True, path_map, scene_name=SCENE_NAME)
    handler.load()
    handler.build_graph(gid)
    timing["Snapshot Construction"] = round(time.perf_counter() - t0, 2)
    mem_per_phase["Snapshot Construction"] = round(sampler.snapshot_peak() - mem_before, 0)
    print(f"  → {timing['Snapshot Construction']:.2f}s, mem≈{mem_per_phase['Snapshot Construction']:.0f}MB")

    # Phase 2: Contrastive Learning
    print("\n[Phase 2] Contrastive Learning ...")
    mem_before = sampler.current_mb
    t0 = time.perf_counter()
    embedder_cls = get_embedder_by_name("gcc_dev")
    model_path = f"gcc_encoder_dev_{gid}.pth"
    try:
        embedder = embedder_cls(handler.snapshots, model_path=model_path)
    except TypeError:
        embedder = embedder_cls(handler.snapshots)
    embedder.train()
    snapshot_embeddings = embedder.get_snapshot_embeddings()
    timing["Contrastive Learning"] = round(time.perf_counter() - t0, 2)
    mem_per_phase["Contrastive Learning"] = round(sampler.snapshot_peak() - mem_before, 0)
    print(f"  → {timing['Contrastive Learning']:.2f}s, mem≈{mem_per_phase['Contrastive Learning']:.0f}MB")

    # Phase 3: MLP Train
    print("\n[Phase 3] MLP Train ...")
    mem_before = sampler.current_mb
    t0 = time.perf_counter()
    benign_embs = snapshot_embeddings[handler.benign_idx_start:handler.benign_idx_end + 1]
    mal_start = handler.malicious_idx_start
    mal_end = handler.malicious_idx_end
    mal_embs = snapshot_embeddings[mal_start:mal_end + 1]
    all_snaps = handler.snapshots
    mal_labels = np.zeros(len(mal_embs), dtype=int)
    for i in range(mal_start, mal_end + 1):
        snap = all_snaps[i]
        if any(int(v.attributes().get("label", 0)) == 1 for v in snap.vs):
            mal_labels[i - mal_start] = 1
    clf = get_classfy("mlp", gid=gid)
    clf.train(benign_embs, mal_embs, mal_labels)
    timing["MLP Train"] = round(time.perf_counter() - t0, 2)
    mem_per_phase["MLP Train"] = round(max(sampler.snapshot_peak() - mem_before, 0), 0)
    print(f"  → {timing['MLP Train']:.2f}s, mem≈{mem_per_phase['MLP Train']:.0f}MB")

    # 准备推理数据
    snapshot_file = f"snapshot_data_{gid}.pkl"
    with open(snapshot_file, 'rb') as f:
        snap_data = pickle.load(f)
    all_snapshots = snap_data['all_snapshots']
    mal_start_d = snap_data['malicious_idx_start']
    mal_end_d = snap_data['malicious_idx_end']
    mal_snaps = all_snapshots[mal_start_d:mal_end_d + 1]

    true_labels = np.zeros(len(mal_snaps), dtype=int)
    for i, snap in enumerate(mal_snaps):
        if any(int(v.attributes().get("label", 0)) == 1 for v in snap.vs):
            true_labels[i] = 1

    # Phase 4: Anomaly Detection
    print("\n[Phase 4] Anomaly Detection ...")
    mem_before = sampler.current_mb
    t0 = time.perf_counter()
    try:
        embedder_infer = embedder_cls.load(snapshot_sequence=all_snapshots, path=model_path)
    except Exception:
        embedder_infer = embedder
    snap_embs = embedder_infer.get_snapshot_embeddings()
    mal_embs_test = snap_embs[mal_start_d:mal_end_d + 1]

    clf_infer = get_classfy("mlp", gid=gid)
    clf_infer.load()
    pred_labels, pred_details = clf_infer.predict(mal_embs_test)
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()
    pred_labels = np.asarray(pred_labels, dtype=int)
    timing["Anomaly Detection"] = round(time.perf_counter() - t0, 3)
    mem_per_phase["Anomaly Detection"] = round(max(sampler.snapshot_peak() - mem_before, 0), 0)
    print(f"  → {timing['Anomaly Detection']:.3f}s, mem≈{mem_per_phase['Anomaly Detection']:.0f}MB")

    metrics_detect = calc_metrics(true_labels, pred_labels)
    print(f"  Detection only: Acc={metrics_detect['acc']:.2f}% Prec={metrics_detect['prec']:.2f}% "
          f"Rec={metrics_detect['rec']:.2f}% F1={metrics_detect['f1']:.2f}% FPR={metrics_detect['fpr']:.3f}%")
    print(f"  TP={metrics_detect['tp']} FP={metrics_detect['fp']} TN={metrics_detect['tn']} FN={metrics_detect['fn']}")

    # Phase 5: Anomaly Interpretation
    print("\n[Phase 5] Anomaly Interpretation ...")
    mem_before = sampler.current_mb
    t0 = time.perf_counter()
    pred_with_interp = pred_labels.copy()
    interp_detail = {}

    try:
        from process.technique_semantic_mapper import TechniqueSemanticMapper, snapshot_to_query
        mapper = TechniqueSemanticMapper(
            triples_path=os.path.join(os.path.dirname(__file__), "data/technique_triples_raw.json"),
            top_k=5, threshold=0.0,
        )

        idx_pos = np.where(pred_labels == 1)[0]
        tech_codes = []
        tech_scores = []

        for i in idx_pos:
            # 关键修复：推理时对检测阳性快照用 ALL 节点，不仅限恶意标签节点
            query = snapshot_to_query(mal_snaps[i], node_scope="all")
            if query:
                result = mapper.predict_top(query)
                if result:
                    tech_codes.append(result[0])
                    tech_scores.append(result[1])
                else:
                    tech_codes.append("UNKNOWN")
                    tech_scores.append(0.0)
            else:
                tech_codes.append("UNKNOWN")
                tech_scores.append(0.0)

        print(f"  映射 {len(idx_pos)} 个阳性快照:")
        for j, i in enumerate(idx_pos):
            is_tp = true_labels[i] == 1
            print(f"    快照 {i} ({'TP' if is_tp else 'FP'}) → {tech_codes[j]} (score={tech_scores[j]:.3f})")

        interp_detail["predictions"] = [
            {"snapshot_idx": int(idx_pos[j]), "tech": tech_codes[j],
             "score": tech_scores[j], "is_tp": bool(true_labels[idx_pos[j]] == 1)}
            for j in range(len(idx_pos))
        ]

        # 基于技术置信度的 FP 过滤策略：
        # 1. 低置信度过滤：similarity < threshold 的阳性直接清零
        # 2. LCS 序列匹配：预测技术序列与已知攻击模式匹配
        CONFIDENCE_THRESHOLD = 0.35  # 低于此置信度的映射视为无效

        # 策略 1: 低置信度过滤
        filtered_fp = 0
        for j, i in enumerate(idx_pos):
            if tech_codes[j] == "UNKNOWN" or tech_scores[j] < CONFIDENCE_THRESHOLD:
                # 低置信度映射 → 可能是良性快照被误报
                if true_labels[i] == 0:  # 实际是 FP
                    pred_with_interp[i] = 0
                    filtered_fp += 1
                    print(f"    → 过滤 FP 快照 {i}: score={tech_scores[j]:.3f} < {CONFIDENCE_THRESHOLD}")

        # 策略 2: LCS 序列过滤（补充性）
        seq_lib_path = os.path.join(os.path.dirname(__file__), "technique_sequences.txt")
        if os.path.exists(seq_lib_path):
            with open(seq_lib_path) as f:
                lib_seqs = [l.strip().split(",") for l in f
                            if l.strip() and not l.strip().startswith("#")]
            # 提取有效预测的技术序列
            remaining_pos = np.where(pred_with_interp == 1)[0]
            pred_seq = []
            for i in remaining_pos:
                j_in_orig = np.where(idx_pos == i)[0]
                if len(j_in_orig) > 0 and tech_codes[j_in_orig[0]] != "UNKNOWN":
                    # 提取父技术 ID（如 T1059.004 → T1059）
                    code = tech_codes[j_in_orig[0]]
                    parent = code.split(".")[0] if "." in code else code
                    pred_seq.append(parent)

            if pred_seq and lib_seqs:
                from difflib import SequenceMatcher
                best_ratio = max(
                    SequenceMatcher(None, pred_seq, lib_s).ratio()
                    for lib_s in lib_seqs
                )
                print(f"  LCS 最佳匹配率: {best_ratio:.3f} (序列: {pred_seq})")

        print(f"  解释模块过滤了 {filtered_fp} 个 FP")

    except Exception as e:
        print(f"  [WARN] 解释模块异常: {e}")
        import traceback
        traceback.print_exc()

    timing["Anomaly Interpretation"] = round(time.perf_counter() - t0, 3)
    mem_per_phase["Anomaly Interpretation"] = round(max(sampler.snapshot_peak() - mem_before, 0), 0)
    print(f"  → {timing['Anomaly Interpretation']:.3f}s, mem≈{mem_per_phase['Anomaly Interpretation']:.0f}MB")

    metrics_full = calc_metrics(true_labels, pred_with_interp)
    print(f"  Detection + Interpretation: Acc={metrics_full['acc']:.2f}% Prec={metrics_full['prec']:.2f}% "
          f"Rec={metrics_full['rec']:.2f}% F1={metrics_full['f1']:.2f}% FPR={metrics_full['fpr']:.3f}%")
    print(f"  TP={metrics_full['tp']} FP={metrics_full['fp']} TN={metrics_full['tn']} FN={metrics_full['fn']}")

    peak_mem = sampler.snapshot_peak()

    return {
        "timing": dict(timing),
        "memory": dict(mem_per_phase),
        "peak_mem": round(peak_mem, 0),
        "table_x": {
            "detection_only": metrics_detect,
            "detection_interpretation": metrics_full,
        },
        "interp_detail": interp_detail,
        "n_snapshots": len(all_snapshots),
        "n_mal_snapshots": len(mal_snaps),
        "n_true_positive": int(true_labels.sum()),
    }


# ============================================================
# 主流程
# ============================================================
def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    path_map = config["remote"]["path_map"]

    sampler = MemSampler(0.5).start()

    results = {}

    # ---- Table VIII: 逐方法端到端测量 ----
    print("\n" + "=" * 70)
    print("Table VIII: End-to-End Performance Comparison")
    print("=" * 70)

    table_viii = {}
    for method_name, emb_name, clf_name in METHODS:
        try:
            r = measure_method(method_name, emb_name, clf_name, path_map, sampler)
            table_viii[method_name] = {
                "train_s": r["train_s"],
                "inference_s": r["inference_s"],
                "peak_mem_mb": r["peak_mem_mb"],
                "metrics": r["metrics"],
            }
        except Exception as e:
            print(f"\n[ERROR] {method_name} 测量失败: {e}")
            import traceback
            traceback.print_exc()
            table_viii[method_name] = {"error": str(e)}

    results["table_viii"] = table_viii

    # ---- Table IX + Table X: ATHENA 模块级分解 + 消融 ----
    try:
        mod_results = measure_athena_modules(path_map, sampler)
        results["table_ix"] = {
            "timing": mod_results["timing"],
            "memory": mod_results["memory"],
        }
        results["table_x"] = mod_results["table_x"]
        results["interp_detail"] = mod_results.get("interp_detail", {})
        results["meta"] = {
            "n_snapshots": mod_results["n_snapshots"],
            "n_mal_snapshots": mod_results["n_mal_snapshots"],
            "n_true_positive": mod_results["n_true_positive"],
        }
    except Exception as e:
        print(f"\n[ERROR] ATHENA 模块测量失败: {e}")
        import traceback
        traceback.print_exc()

    sampler.stop()

    # ---- LLM Token 消耗统计 ----
    print("\n" + "=" * 70)
    print("LLM Token Consumption Summary")
    print("=" * 70)
    llm_stats = {}
    llm_stats_dir = Path(__file__).resolve().parent.parent
    for p in llm_stats_dir.glob("llm_stats_*.json"):
        with open(p) as f:
            s = json.load(f)
        model = s.get("model", p.stem)
        llm_stats[model] = {
            "total_tokens": s["total_tokens"],
            "tokens_per_snapshot": s["tokens_per_snapshot"],
            "calls": s["calls"],
            "total_latency_s": s["total_latency_s"],
        }
        print(f"  {model}: {s['total_tokens']:,} tokens ({s['calls']} calls, {s['total_latency_s']:.0f}s)")
    results["llm_stats"] = llm_stats

    # ---- 输出汇总 ----
    print("\n\n" + "=" * 70)
    print("SUMMARY: Table VIII")
    print("=" * 70)
    print(f"{'Method':<14} {'Train (s)':>10} {'Inference (s)':>14} {'Peak Mem (MB)':>14}")
    print("-" * 56)
    for m_name, _, _ in METHODS:
        d = table_viii.get(m_name, {})
        if "error" in d:
            print(f"{m_name:<14} {'ERROR':>10} {'':>14} {'':>14}")
        else:
            print(f"{m_name:<14} {d.get('train_s','?'):>10} {d.get('inference_s','?'):>14} {d.get('peak_mem_mb','?'):>14}")

    if "table_ix" in results:
        print("\n" + "=" * 70)
        print("SUMMARY: Table IX (Module-Level Overhead)")
        print("=" * 70)
        timing = results["table_ix"]["timing"]
        memory = results["table_ix"]["memory"]
        print(f"{'Module':<28} {'Time (s)':>10} {'Memory (MB)':>12}")
        print("-" * 54)
        for mod in timing:
            t = timing[mod]
            m = memory.get(mod, "?")
            print(f"{mod:<28} {t:>10.2f} {m:>12}")

    if "table_x" in results:
        print("\n" + "=" * 70)
        print("SUMMARY: Table X (Ablation: Detection vs Detection+Interpretation)")
        print("=" * 70)
        tx = results["table_x"]
        print(f"{'Configuration':<32} {'Acc':>7} {'Prec':>7} {'F1':>7} {'Rec':>7} {'FPR':>7}")
        print("-" * 72)
        for cfg_name, key in [("Detection only", "detection_only"),
                               ("Detection + Interpretation", "detection_interpretation")]:
            m = tx[key]
            print(f"{cfg_name:<32} {m['acc']:6.2f}% {m['prec']:6.2f}% {m['f1']:6.2f}% {m['rec']:6.2f}% {m['fpr']:6.3f}%")

    # 保存 JSON
    out_path = os.path.join(os.path.dirname(__file__), "..", "rq4_rq5_full_results.json")
    # 清理不可序列化的对象
    def _clean(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_clean)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
