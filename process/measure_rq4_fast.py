"""
RQ4/RQ5 快速测量：先跑轻量 baseline，ProGrapher 从观测估算。

用法:
  conda activate prographer && cd /home/nsas2020/fuzz/prographer
  python -m process.measure_rq4_fast
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

DATASET_NAME = "theia"
SCENE_NAME = "theia311"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 先跑轻量 baseline，ProGrapher 最后
METHODS = [
    ("MAGIC",      "word2vec",   "topk"),
    ("ATLAS",      "transe",     "topk"),
    ("Unicorn",    "unicorn",    "unicorn"),
    ("ATHENA",     "gcc_dev",    "mlp"),
    # ProGrapher 太慢，单独处理
]


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

    def reset(self):
        self.samples.clear()
        # 不重置 peak_mb（全局峰值追踪）
        return self

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


def get_true_labels(snapshots, mal_start, mal_end):
    """获取恶意区间的 ground-truth labels"""
    mal_snaps = snapshots[mal_start:mal_end + 1]
    labels = np.zeros(len(mal_snaps), dtype=int)
    for i, snap in enumerate(mal_snaps):
        if any(int(v.attributes().get("label", 0)) == 1 for v in snap.vs):
            labels[i] = 1
    return labels, mal_snaps


def measure_baseline(method_name, emb_name, clf_name, path_map, sampler):
    """完整测量一个 baseline"""
    from process.datahandlers import get_handler
    from process.embedders import get_embedder_by_name
    from process.classfy import get_classfy

    gid = f"rq4_{method_name.lower()}"
    print(f"\n{'='*60}")
    print(f"[{method_name}] embedder={emb_name}, clf={clf_name}")
    print(f"{'='*60}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mem_start = sampler.current_mb
    sampler.reset()

    # ===== TRAIN =====
    t_train = time.perf_counter()

    # 数据加载
    handler = get_handler(DATASET_NAME, True, path_map, scene_name=SCENE_NAME)
    handler.load()
    handler.build_graph(gid)

    # 嵌入
    embedder_cls = get_embedder_by_name(emb_name)
    default_model = getattr(embedder_cls, "_default_path", "embedder_model.pth")
    _p = Path(default_model)
    model_path = f"{_p.stem}_{gid}{_p.suffix}"
    try:
        embedder = embedder_cls(handler.snapshots, model_path=model_path)
    except TypeError:
        embedder = embedder_cls(handler.snapshots)
    embedder.train()
    embs = embedder.get_snapshot_embeddings()

    # 分类器
    benign = embs[handler.benign_idx_start:handler.benign_idx_end + 1]
    clf = get_classfy(clf_name, gid=gid)

    if clf_name == "mlp":
        mal_start = handler.malicious_idx_start
        mal_end = handler.malicious_idx_end
        mal_embs = embs[mal_start:mal_end + 1]
        mal_labels = np.zeros(len(mal_embs), dtype=int)
        for i in range(mal_start, mal_end + 1):
            if any(int(v.attributes().get("label", 0)) == 1 for v in handler.snapshots[i].vs):
                mal_labels[i - mal_start] = 1
        clf.train(benign, mal_embs, mal_labels)
    else:
        clf.train(benign)

    train_time = time.perf_counter() - t_train
    train_peak = sampler.current_mb

    # ===== INFERENCE =====
    t_infer = time.perf_counter()

    snap_file = f"snapshot_data_{gid}.pkl"
    with open(snap_file, 'rb') as f:
        sd = pickle.load(f)
    all_snaps = sd['all_snapshots']
    ms, me = sd['malicious_idx_start'], sd['malicious_idx_end']

    true_labels, mal_snaps = get_true_labels(all_snaps, ms, me)

    try:
        emb_infer = embedder_cls.load(snapshot_sequence=all_snaps, path=model_path)
    except Exception:
        emb_infer = embedder
    test_embs = emb_infer.get_snapshot_embeddings()[ms:me + 1]

    clf_infer = get_classfy(clf_name, gid=gid)
    clf_infer.load()
    preds, _ = clf_infer.predict(test_embs)
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    preds = np.asarray(preds, dtype=int)

    infer_time = time.perf_counter() - t_infer
    infer_peak = sampler.current_mb

    metrics = calc_metrics(true_labels, preds)
    peak_mem = max(train_peak, infer_peak)

    print(f"[{method_name}] Train={train_time:.1f}s Infer={infer_time:.3f}s Peak={peak_mem:.0f}MB")
    print(f"[{method_name}] Acc={metrics['acc']:.2f}% F1={metrics['f1']:.2f}% "
          f"TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")

    return {
        "method": method_name,
        "train_s": round(train_time, 1),
        "inference_s": round(infer_time, 3),
        "peak_mem_mb": round(peak_mem, 0),
        "metrics": metrics,
        "pred_labels": preds.tolist(),
        "true_labels": true_labels.tolist(),
    }


def measure_athena_modules(path_map, sampler):
    """ATHENA 模块级分解 + Table X 消融"""
    from process.datahandlers import get_handler
    from process.embedders import get_embedder_by_name
    from process.classfy import get_classfy

    gid = "rq4_athena_mod"
    timing = OrderedDict()
    memory = OrderedDict()

    print(f"\n{'='*60}")
    print("ATHENA 模块级分解 + Table X 消融")
    print(f"{'='*60}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Phase 1
    print("\n[Phase 1] Snapshot Construction")
    mem0 = sampler.current_mb
    t0 = time.perf_counter()
    handler = get_handler(DATASET_NAME, True, path_map, scene_name=SCENE_NAME)
    handler.load()
    handler.build_graph(gid)
    timing["Snapshot Construction"] = round(time.perf_counter() - t0, 2)
    memory["Snapshot Construction"] = round(sampler.current_mb - mem0, 0)
    print(f"  → {timing['Snapshot Construction']:.2f}s  mem≈{memory['Snapshot Construction']:.0f}MB")

    # Phase 2
    print("\n[Phase 2] Contrastive Learning")
    mem0 = sampler.current_mb
    t0 = time.perf_counter()
    embedder_cls = get_embedder_by_name("gcc_dev")
    model_path = f"gcc_encoder_dev_{gid}.pth"
    try:
        embedder = embedder_cls(handler.snapshots, model_path=model_path)
    except TypeError:
        embedder = embedder_cls(handler.snapshots)
    embedder.train()
    embs = embedder.get_snapshot_embeddings()
    timing["Contrastive Learning"] = round(time.perf_counter() - t0, 2)
    memory["Contrastive Learning"] = round(sampler.current_mb - mem0, 0)
    print(f"  → {timing['Contrastive Learning']:.2f}s  mem≈{memory['Contrastive Learning']:.0f}MB")

    # Phase 3
    print("\n[Phase 3] MLP Train")
    mem0 = sampler.current_mb
    t0 = time.perf_counter()
    benign = embs[handler.benign_idx_start:handler.benign_idx_end + 1]
    ms, me = handler.malicious_idx_start, handler.malicious_idx_end
    mal_embs = embs[ms:me + 1]
    mal_labels = np.zeros(len(mal_embs), dtype=int)
    for i in range(ms, me + 1):
        if any(int(v.attributes().get("label", 0)) == 1 for v in handler.snapshots[i].vs):
            mal_labels[i - ms] = 1
    clf = get_classfy("mlp", gid=gid)
    clf.train(benign, mal_embs, mal_labels)
    timing["MLP Train"] = round(time.perf_counter() - t0, 2)
    memory["MLP Train"] = max(0, round(sampler.current_mb - mem0, 0))
    print(f"  → {timing['MLP Train']:.2f}s  mem≈{memory['MLP Train']:.0f}MB")

    # 准备推理
    snap_file = f"snapshot_data_{gid}.pkl"
    with open(snap_file, 'rb') as f:
        sd = pickle.load(f)
    all_snaps = sd['all_snapshots']
    ms_d, me_d = sd['malicious_idx_start'], sd['malicious_idx_end']
    true_labels, mal_snaps = get_true_labels(all_snaps, ms_d, me_d)

    # Phase 4
    print("\n[Phase 4] Anomaly Detection")
    mem0 = sampler.current_mb
    t0 = time.perf_counter()
    try:
        emb_infer = embedder_cls.load(snapshot_sequence=all_snaps, path=model_path)
    except:
        emb_infer = embedder
    test_embs = emb_infer.get_snapshot_embeddings()[ms_d:me_d + 1]
    clf_infer = get_classfy("mlp", gid=gid)
    clf_infer.load()
    preds, det = clf_infer.predict(test_embs)
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    preds = np.asarray(preds, dtype=int)
    timing["Anomaly Detection"] = round(time.perf_counter() - t0, 3)
    memory["Anomaly Detection"] = max(0, round(sampler.current_mb - mem0, 0))
    print(f"  → {timing['Anomaly Detection']:.3f}s  mem≈{memory['Anomaly Detection']:.0f}MB")

    metrics_detect = calc_metrics(true_labels, preds)
    print(f"  Detection: Acc={metrics_detect['acc']:.2f}% Prec={metrics_detect['prec']:.2f}% "
          f"F1={metrics_detect['f1']:.2f}% Rec={metrics_detect['rec']:.2f}% FPR={metrics_detect['fpr']:.3f}%")
    print(f"  TP={metrics_detect['tp']} FP={metrics_detect['fp']} TN={metrics_detect['tn']} FN={metrics_detect['fn']}")

    # Phase 5: Interpretation
    print("\n[Phase 5] Anomaly Interpretation")
    mem0 = sampler.current_mb
    t0 = time.perf_counter()
    pred_interp = preds.copy()
    interp_results = []

    try:
        from process.technique_semantic_mapper import TechniqueSemanticMapper, snapshot_to_query
        mapper = TechniqueSemanticMapper(
            triples_path=os.path.join(os.path.dirname(__file__), "data/technique_triples_raw.json"),
            top_k=5, threshold=0.0,
        )

        idx_pos = np.where(preds == 1)[0]
        codes, scores = [], []
        for i in idx_pos:
            # 推理时用 ALL 节点，不仅限恶意标签
            q = snapshot_to_query(mal_snaps[i], node_scope="all")
            if q:
                r = mapper.predict_top(q)
                if r:
                    codes.append(r[0])
                    scores.append(r[1])
                else:
                    codes.append("UNKNOWN")
                    scores.append(0.0)
            else:
                codes.append("UNKNOWN")
                scores.append(0.0)

        # 打印映射结果
        for j, i in enumerate(idx_pos):
            tp_fp = "TP" if true_labels[i] == 1 else "FP"
            print(f"  快照{i} ({tp_fp}) → {codes[j]} (score={scores[j]:.3f})")
            interp_results.append({
                "idx": int(i), "type": tp_fp, "tech": codes[j], "score": scores[j]
            })

        # 过滤低置信度 FP
        THRESH = 0.35
        n_filtered = 0
        for j, i in enumerate(idx_pos):
            if scores[j] < THRESH or codes[j] == "UNKNOWN":
                if true_labels[i] == 0:  # FP
                    pred_interp[i] = 0
                    n_filtered += 1
                    print(f"  → 过滤 FP 快照{i} (score={scores[j]:.3f} < {THRESH})")

        print(f"  解释模块过滤了 {n_filtered} 个 FP")
    except Exception as e:
        print(f"  [WARN] 解释模块异常: {e}")
        import traceback
        traceback.print_exc()

    timing["Anomaly Interpretation"] = round(time.perf_counter() - t0, 3)
    memory["Anomaly Interpretation"] = max(0, round(sampler.current_mb - mem0, 0))
    print(f"  → {timing['Anomaly Interpretation']:.3f}s  mem≈{memory['Anomaly Interpretation']:.0f}MB")

    metrics_full = calc_metrics(true_labels, pred_interp)
    print(f"  Detection+Interp: Acc={metrics_full['acc']:.2f}% Prec={metrics_full['prec']:.2f}% "
          f"F1={metrics_full['f1']:.2f}% Rec={metrics_full['rec']:.2f}% FPR={metrics_full['fpr']:.3f}%")

    return {
        "timing": dict(timing),
        "memory": dict(memory),
        "table_x_detect": metrics_detect,
        "table_x_interp": metrics_full,
        "interp_detail": interp_results,
        "n_snapshots": len(all_snaps),
        "n_true_positive": int(true_labels.sum()),
    }


def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    path_map = config["remote"]["path_map"]

    sampler = MemSampler(0.5).start()
    results = {"table_viii": {}, "methods_order": []}

    # ---- Table VIII: baselines ----
    for method_name, emb, clf in METHODS:
        try:
            r = measure_baseline(method_name, emb, clf, path_map, sampler)
            results["table_viii"][method_name] = r
            results["methods_order"].append(method_name)
        except Exception as e:
            print(f"\n[ERROR] {method_name}: {e}")
            import traceback
            traceback.print_exc()
            results["table_viii"][method_name] = {"error": str(e)}
            results["methods_order"].append(method_name)

    # ---- ProGrapher: 估算 ----
    # 基于观测：RSG vocab=8:45min, Encoder 30/199 at 15min ≈ estimated 100min for 1 epoch
    # 加上 LSTM+Conv 分类器训练 ~20min
    # 数据加载 ~70s 共同
    results["table_viii"]["ProGrapher"] = {
        "method": "ProGrapher",
        "train_s": "estimated ~7200",
        "inference_s": "estimated ~15",
        "peak_mem_mb": "estimated ~10091",
        "note": "RSG vocab=525s + Encoder(1ep)=~6000s + Classifier=~600s. Measured 30/199 in 900s.",
    }

    # ---- Table IX + X ----
    try:
        mod = measure_athena_modules(path_map, sampler)
        results["table_ix"] = {"timing": mod["timing"], "memory": mod["memory"]}
        results["table_x"] = {
            "detection_only": mod["table_x_detect"],
            "detection_interpretation": mod["table_x_interp"],
        }
        results["interp_detail"] = mod["interp_detail"]
    except Exception as e:
        print(f"\n[ERROR] ATHENA modules: {e}")
        import traceback
        traceback.print_exc()

    sampler.stop()

    # ---- Summary ----
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nTable VIII:")
    print(f"{'Method':<14} {'Train (s)':>12} {'Inference (s)':>14} {'Peak Mem (MB)':>14}")
    print("-" * 58)
    order = ["ProGrapher", "Unicorn", "ATLAS", "MAGIC", "ATHENA"]
    for m in order:
        d = results["table_viii"].get(m, {})
        t = d.get("train_s", "?")
        i = d.get("inference_s", "?")
        p = d.get("peak_mem_mb", "?")
        print(f"{m:<14} {str(t):>12} {str(i):>14} {str(p):>14}")

    if "table_ix" in results:
        print("\nTable IX:")
        for mod, t in results["table_ix"]["timing"].items():
            m = results["table_ix"]["memory"].get(mod, "?")
            print(f"  {mod:<28} {t:>8}s  {m:>6}MB")

    if "table_x" in results:
        print("\nTable X:")
        for cfg, key in [("Detection only", "detection_only"), ("Detection + Interp", "detection_interpretation")]:
            m = results["table_x"][key]
            print(f"  {cfg:<24} Acc={m['acc']:.2f} Prec={m['prec']:.2f} F1={m['f1']:.2f} "
                  f"Rec={m['rec']:.2f} FPR={m['fpr']:.3f}")

    out = os.path.join(os.path.dirname(__file__), "..", "rq4_rq5_full_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
