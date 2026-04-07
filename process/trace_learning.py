"""
追踪对比学习过程：每个 epoch 后评估节点级和快照级的区分效果
"""
import pickle, numpy as np, torch, sys, os, time
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from process.embedders import get_embedder_by_name
from process.classfy import get_classfy
from process.benchmark_augmentation import (
    AUGMENT_PRESETS, _make_llm_fn, _set_seed,
)
from process.mutation import MutationPipeline


def main():
    _set_seed(42)

    # 加载数据
    with open('snapshot_data_bench_theia_theia311.pkl', 'rb') as f:
        d = pickle.load(f)

    class Handler:
        pass
    handler = Handler()
    handler.snapshots = d['all_snapshots']
    handler.benign_idx_start = d['benign_idx_start']
    handler.benign_idx_end = d['benign_idx_end']
    handler.malicious_idx_start = d['malicious_idx_start']
    handler.malicious_idx_end = d['malicious_idx_end']

    preset = AUGMENT_PRESETS["llm_guided"]

    # 创建编码器
    embedder_cls = get_embedder_by_name("gcc_dev")
    train_indices = list(range(handler.benign_idx_start, handler.benign_idx_end + 1))
    embedder = embedder_cls(handler.snapshots, **{
        "model_path": "trace_model.pth",
        "drop_edge_p": preset["drop_edge_p"],
        "feat_mask_p": preset["feat_mask_p"],
        "use_degree_coop_augment": preset["use_degree_coop_augment"],
        "use_malicious_snapshots": preset["use_malicious_snapshots"],
        "use_malicious_negatives": preset["use_malicious_negatives"],
        "combine": preset["combine"],
        "train_indices": train_indices,
        "num_epochs": 1,  # 手动控每个 epoch
        "attr_weight_alpha": preset.get("attr_weight_alpha", 0.3),
    })

    # 生成变异图
    pipeline = MutationPipeline(
        snapshots=handler.snapshots,
        benign_range=(handler.benign_idx_start, handler.benign_idx_end),
        attack_range=(handler.malicious_idx_start, handler.malicious_idx_end),
        delta_h=preset.get("delta_h", 0.3),
        delta_h_upper=preset.get("delta_h_upper", 0.95),
    )
    mutation_map = pipeline.generate(llm_fn=None, skip_verification=False)
    embedder.mutation_map = mutation_map

    # 攻击快照信息
    a_s, a_e = handler.malicious_idx_start, handler.malicious_idx_end
    mal_labels = np.array([
        int(any(handler.snapshots[i].vs[v]['label'] == 1 for v in range(handler.snapshots[i].vcount())))
        for i in range(a_s, a_e + 1)
    ])
    attack_sids = [a_s + i for i in range(len(mal_labels)) if mal_labels[i] == 1]

    # MLP 划分
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.7, random_state=42)
    train_idx, test_idx = next(sss.split(np.zeros((len(mal_labels), 1)), mal_labels))
    test_attack_sids = [a_s + t for t in test_idx if mal_labels[t] == 1]

    print(f"\n追踪 {len(attack_sids)} 个攻击快照的学习过程")
    print(f"测试集攻击: {test_attack_sids}")
    print()

    # 每个 epoch 训练 + 评估
    NUM_EPOCHS = 6
    header = f"{'epoch':>5} {'loss':>8} | {'mean_AUC':>8} {'min_AUC':>8} | {'Acc':>6} {'F1':>6} {'Rec':>6} | FN"
    print(header)
    print("-" * len(header))

    for epoch in range(NUM_EPOCHS):
        # 训练一个 epoch
        embedder.num_epochs = 1
        embedder.train()

        # 生成嵌入
        snapshot_embs = embedder.get_snapshot_embeddings()
        node_embs = embedder.snapshot_node_embeddings

        # 计算良性中心
        benign_vecs = []
        for i in range(handler.benign_idx_start, handler.benign_idx_end + 1):
            emb = node_embs[i] if i < len(node_embs) else {}
            benign_vecs.extend(emb.values())
        benign_center = np.mean(benign_vecs, axis=0)

        # 节点级 AUC
        aucs = []
        for sid in attack_sids:
            g = handler.snapshots[sid]
            emb = node_embs[sid] if sid < len(node_embs) else {}
            if not emb:
                continue
            dists, labels = [], []
            for v in range(g.vcount()):
                nid = g.vs[v]['name']
                vec = emb.get(nid)
                if vec is None:
                    continue
                dists.append(np.linalg.norm(vec - benign_center))
                labels.append(int(g.vs[v]['label']))
            if sum(labels) > 0 and sum(labels) < len(labels):
                aucs.append(roc_auc_score(labels, dists))

        # 快照级分类
        benign_emb = snapshot_embs[handler.benign_idx_start:handler.benign_idx_end + 1]
        mal_emb = snapshot_embs[a_s:a_e + 1]

        classify = get_classfy("mlp", gid=f"trace_{epoch}")
        classify.train(benign_emb, mal_emb[train_idx], mal_labels[train_idx])

        result = classify.predict(mal_emb[test_idx])
        pred = result[0] if isinstance(result, tuple) else result
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        pred = np.asarray(pred, dtype=int)
        true = mal_labels[test_idx]

        tp = int(np.sum((true == 1) & (pred == 1)))
        fn = int(np.sum((true == 1) & (pred == 0)))
        fp = int(np.sum((true == 0) & (pred == 1)))
        acc = (tp + int(np.sum((true == 0) & (pred == 0)))) / len(true) * 100
        prec = tp / max(tp + fp, 1) * 100
        rec = tp / max(tp + fn, 1) * 100
        f1 = 2 * prec * rec / max(prec + rec, 1)

        # FN 列表
        fn_list = [a_s + test_idx[i] for i in range(len(test_idx)) if true[i] == 1 and pred[i] == 0]

        # Loss
        loss = getattr(embedder, '_last_epoch_loss', 0)

        mean_auc = np.mean(aucs) if aucs else 0
        min_auc = min(aucs) if aucs else 0

        fn_str = ",".join(str(s) for s in fn_list) if fn_list else "无"
        print(f"{epoch+1:>5} {loss:>8.4f} | {mean_auc:>8.3f} {min_auc:>8.3f} | {acc:>5.1f}% {f1:>5.1f}% {rec:>5.1f}% | {fn_str}")


if __name__ == "__main__":
    main()
