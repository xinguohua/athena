"""
分析分类预测结果：哪些快照被误判，为什么
"""
import os, sys, pickle, time, platform, json
import numpy as np
import torch
import yaml
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from process.datahandlers import get_handler
from process.embedders import get_embedder_by_name
from process.classfy import get_classfy
from process.benchmark_augmentation import (
    AUGMENT_PRESETS, DEVICE, GLOBAL_ID, CLASSIFY_NAME,
    _make_llm_fn, _run_mutation_pipeline, _set_seed, load_config, prepare_data,
)
from process.mutation.semantic import _get_properties, _clean_set_str


def main():
    cache_file = "snapshot_data_bench_theia_theia311.pkl"
    with open(cache_file, "rb") as f:
        cache = pickle.load(f)
    handler = type('H', (), {
        'snapshots': cache['all_snapshots'],
        'benign_idx_start': cache['benign_idx_start'],
        'benign_idx_end': cache['benign_idx_end'],
        'malicious_idx_start': cache['malicious_idx_start'],
        'malicious_idx_end': cache['malicious_idx_end'],
    })()

    preset = AUGMENT_PRESETS["llm_guided"]
    seed = 42
    _set_seed(seed)
    tag = f"{GLOBAL_ID}_llm_guided_{seed}"

    # 训练编码器
    embedder_cls = get_embedder_by_name("gcc_dev")
    _p = Path(getattr(embedder_cls, "_default_path", "embedder_model.pth"))
    model_path = f"{_p.stem}_{tag}{_p.suffix}"

    train_indices = list(range(handler.benign_idx_start, handler.benign_idx_end + 1))
    embedder = embedder_cls(handler.snapshots, **{
        "model_path": model_path,
        "drop_edge_p": preset["drop_edge_p"],
        "feat_mask_p": preset["feat_mask_p"],
        "use_degree_coop_augment": preset["use_degree_coop_augment"],
        "use_malicious_snapshots": preset["use_malicious_snapshots"],
        "use_malicious_negatives": preset["use_malicious_negatives"],
        "combine": preset["combine"],
        "train_indices": train_indices,
        "num_epochs": preset.get("num_epochs", 3),
        "attr_weight_alpha": preset.get("attr_weight_alpha", 0.3),
    })

    _run_mutation_pipeline(embedder, handler, preset)
    embedder.train()

    snapshot_embeddings = embedder.get_snapshot_embeddings()

    # 分类器
    benign_emb = snapshot_embeddings[handler.benign_idx_start:handler.benign_idx_end + 1]
    mal_start = handler.malicious_idx_start
    mal_end = handler.malicious_idx_end
    mal_snapshots = handler.snapshots[mal_start:mal_end + 1]
    mal_emb = snapshot_embeddings[mal_start:mal_end + 1]
    mal_labels = np.array([
        int(any(v["label"] == 1 for v in g.vs))
        for g in mal_snapshots
    ], dtype=int)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.7, random_state=42)
    train_idx, test_idx = next(sss.split(mal_emb, mal_labels))

    classify = get_classfy(CLASSIFY_NAME, gid=tag)
    classify.train(benign_emb, mal_emb[train_idx], mal_labels[train_idx])

    # 预测
    result = classify.predict(mal_emb[test_idx])
    pred = result[0] if isinstance(result, tuple) else result
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    pred = np.asarray(pred, dtype=int)
    true = mal_labels[test_idx]

    # 分析每个测试快照
    print("\n" + "=" * 80)
    print("预测分析")
    print("=" * 80)

    fp_list, fn_list, tp_list, tn_list = [], [], [], []
    fn_ids, tp_ids = [], []

    for i, tidx in enumerate(test_idx):
        sidx = mal_start + tidx  # 实际快照索引
        g = handler.snapshots[sidx]
        y_true = true[i]
        y_pred = pred[i]
        n_nodes = g.vcount()
        n_attack = sum(1 for v in range(n_nodes) if g.vs[v]['label'] == 1)

        # 节点类型分布
        types = {}
        for v in range(n_nodes):
            t = str(g.vs[v]['type'])
            types[t] = types.get(t, 0) + 1

        # 攻击节点的 properties
        attack_props = []
        for v in range(n_nodes):
            if g.vs[v]['label'] == 1:
                attack_props.append(_get_properties(g, v)[:50])

        if y_true == 0 and y_pred == 1:
            fp_list.append((sidx, n_nodes, types, attack_props))
        elif y_true == 1 and y_pred == 0:
            fn_list.append((sidx, n_nodes, n_attack, types, attack_props))
            fn_ids.append(sidx)
        elif y_true == 1 and y_pred == 1:
            tp_list.append((sidx, n_nodes, n_attack, types, attack_props))
            tp_ids.append(sidx)

    # 打印 FP
    print(f"\n--- FP (误报): {len(fp_list)} 个良性快照被判为攻击 ---")
    for sidx, n, types, _ in fp_list:
        top_types = sorted(types.items(), key=lambda x: -x[1])[:3]
        type_str = ", ".join(f"{t}={c}" for t, c in top_types)
        print(f"  A[{sidx}] {n} 节点 | {type_str}")

    # 打印 FN
    print(f"\n--- FN (漏检): {len(fn_list)} 个攻击快照未被检出 ---")
    for sidx, n, n_att, types, props in fn_list:
        print(f"  A[{sidx}] {n} 节点, {n_att} 攻击 | props: {props}")

    # 打印 TP
    print(f"\n--- TP (正确检出): {len(tp_list)} 个 ---")
    for sidx, n, n_att, types, props in tp_list:
        print(f"  A[{sidx}] {n} 节点, {n_att} 攻击")

    # FP 快照的嵌入 vs 良性嵌入的距离
    print(f"\n--- 嵌入空间分析 ---")
    benign_mean = benign_emb.mean(axis=0)
    mal_attack_embs = mal_emb[test_idx][true == 1]
    mal_benign_embs = mal_emb[test_idx][true == 0]
    fp_embs = mal_emb[test_idx][(true == 0) & (pred == 1)]
    tn_embs = mal_emb[test_idx][(true == 0) & (pred == 0)]

    # ====== 节点级分析：攻击节点嵌入是否被推远 ======
    print(f"\n--- 节点级嵌入分析 ---")

    # 良性中心：所有良性快照所有节点嵌入的均值
    benign_node_vecs = []
    for i in range(handler.benign_idx_start, handler.benign_idx_end + 1):
        emb_dict = embedder.snapshot_node_embeddings[i]
        if emb_dict:
            benign_node_vecs.extend(emb_dict.values())
    benign_node_center = np.mean(benign_node_vecs, axis=0)
    benign_node_dists = [np.linalg.norm(v - benign_node_center) for v in benign_node_vecs]
    benign_p95 = np.percentile(benign_node_dists, 95)
    benign_p99 = np.percentile(benign_node_dists, 99)
    print(f"  良性节点到中心距离: mean={np.mean(benign_node_dists):.2f}, "
          f"p95={benign_p95:.2f}, p99={benign_p99:.2f}")

    for label, sidx_list in [("FN (漏检)", fn_ids), ("TP (检出)", tp_ids)]:
        print(f"\n  === {label} ===")
        for sid in sidx_list:
            g = handler.snapshots[sid]
            emb_dict = embedder.snapshot_node_embeddings[sid]
            if not emb_dict:
                continue

            # 攻击节点距离
            attack_dists = []
            benign_dists_in_snap = []
            for v in range(g.vcount()):
                nid = g.vs[v]['name']
                vec = emb_dict.get(nid)
                if vec is None:
                    continue
                d = np.linalg.norm(vec - benign_node_center)
                if g.vs[v]['label'] == 1:
                    attack_dists.append(d)
                else:
                    benign_dists_in_snap.append(d)

            if attack_dists:
                max_att = max(attack_dists)
                mean_att = np.mean(attack_dists)
                mean_ben = np.mean(benign_dists_in_snap) if benign_dists_in_snap else 0
                max_ben = max(benign_dists_in_snap) if benign_dists_in_snap else 0
                # 攻击节点超过良性 p99 的数量
                n_above = sum(1 for d in attack_dists if d > benign_p99)
                print(f"    A[{sid}] ({g.vcount()}节点, {len(attack_dists)}攻击)")
                print(f"      攻击节点: mean_dist={mean_att:.2f}, max_dist={max_att:.2f}")
                print(f"      良性节点: mean_dist={mean_ben:.2f}, max_dist={max_ben:.2f}")
                print(f"      攻击超p99: {n_above}/{len(attack_dists)}")


if __name__ == "__main__":
    main()
