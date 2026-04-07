"""
benchmark_robustness.py — 对抗规避鲁棒性测试

用 LLM-guided mutation（反向）对测试攻击快照做隐蔽化变异，
然后用各策略训练的 encoder+MLP 检测，比较鲁棒性差异。

变异方式：MutationPipeline 反向使用
  - 对每个攻击快照，找相似的良性快照
  - 用良性子图替换攻击快照中的子图（结构变异）→ 稀释恶意拓扑信号
  - 可选语义变异（LLM/规则 fallback）→ 伪装操作标签

用法:
    conda activate prographer && cd /home/nsas2020/fuzz/prographer
    python -m process.benchmark_robustness --dataset theia --scene theia311
"""
import argparse
import json
import os
import sys
import time
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from process.benchmark_augmentation import (
    AUGMENT_PRESETS, DEVICE, GLOBAL_ID,
    load_config, prepare_data, _set_seed,
    _train_single_encoder,
)

# 只跑 5 种基础策略
BASE_STRATEGIES = ["no_aug", "graphcl", "gca", "mimicry", "llm_guided"]


# ========================================================================
# 正向变异：良性图 + 攻击子图 → 生成攻击变体（测试用）
# ========================================================================

def generate_attack_variants(handler):
    """用 EgoMutationPipeline 生成 ego 级攻击变体。

    和训练时一样的粒度：
    - 良性 ego + 攻击 ego → ego 级子图替换 → 变异 ego
    - 这些变体是"伪装成良性的攻击 ego"，用来测试各策略的检测能力

    llm_guided 训练时见过同类变体 → 应该检测更好
    其他策略没见过 → 检测更差
    """
    from process.mutation.pipeline import EgoMutationPipeline

    print("\n[AttackVariant] 生成 ego 级攻击变体...")
    print(f"  良性快照范围: {handler.benign_idx_start}-{handler.benign_idx_end}")
    print(f"  攻击快照范围: {handler.malicious_idx_start}-{handler.malicious_idx_end}")

    pipeline = EgoMutationPipeline(
        snapshots=handler.snapshots,
        benign_range=(handler.benign_idx_start, handler.benign_idx_end),
        attack_range=(handler.malicious_idx_start, handler.malicious_idx_end),
        r_hop=2,
        ego_max_nodes=32,
        top_k=5,
        max_region_size=16,
    )

    mutation_map = pipeline.generate(llm_fn=None, egos_per_snapshot=5)

    n_egos = sum(len(v) for v in mutation_map.values())
    print(f"[AttackVariant] 生成 {n_egos} 个 ego 级攻击变体，"
          f"覆盖 {len(mutation_map)} 个快照")

    return mutation_map


# ========================================================================
# 训练 + 评估
# ========================================================================

def _build_variant_test_cache(embedder, mutation_map):
    """从 ego 级攻击变体构建测试 cache。

    每个变异 ego 已是 ~32 节点，直接编码。
    变异 ego 视为攻击样本（label=1），同时从测试快照采样良性 ego 作为负样本。
    """
    cache = []

    # 攻击变体：每个变异 ego 整体视为攻击
    n_attack_egos = 0
    for sidx, ego_list in mutation_map.items():
        if not isinstance(ego_list, list):
            ego_list = [ego_list]
        for g_ego in ego_list:
            if g_ego is None or g_ego.vcount() == 0:
                continue
            embedder._preheat_snapshot_properties(g_ego)
            x_np = embedder._build_node_features(g_ego)
            ei, ef = embedder._igraph_edges_to_edge_index(g_ego)
            cache.append((x_np, ei.cpu(), ef.cpu(), 1, sidx))
            n_attack_egos += 1

    # 良性负样本：从测试集良性 ego 中采样（保持和标准测试一致的比例）
    n_benign_target = min(n_attack_egos * 10, 3000)
    benign_from_test = [e for e in embedder.test_ego_cache if e[3] == 0]
    import random as _rng
    if len(benign_from_test) > n_benign_target:
        benign_from_test = _rng.sample(benign_from_test, n_benign_target)
    for entry in benign_from_test:
        x_np, ei, ef, lab = entry[0], entry[1], entry[2], entry[3]
        sidx = entry[4] if len(entry) > 4 else -1
        cache.append((x_np, ei, ef, 0, sidx))

    n_benign = sum(1 for e in cache if e[3] == 0)
    n_attack = sum(1 for e in cache if e[3] == 1)
    print(f"  变体测试 ego: {len(cache)} (良性={n_benign}, 攻击={n_attack})")
    return cache


def evaluate_on_cache(embedder, classifier, test_cache):
    """在给定 test_cache 上评估，返回 ego 级 + 快照级指标"""
    classifier.eval()
    embedder.encoder.eval()

    ego_preds = []
    with torch.no_grad():
        for entry in test_cache:
            x_np, ei, ef, lab = entry[0], entry[1], entry[2], entry[3]
            sidx = entry[4] if len(entry) > 4 else -1

            x = torch.from_numpy(x_np).to(DEVICE)
            ei_d, ef_d = ei.to(DEVICE), ef.to(DEVICE)
            h = embedder.encoder(x, ei_d, edge_feat=ef_d)
            feat = torch.cat([h[0], h.mean(dim=0)], dim=0)
            pred = classifier(feat.unsqueeze(0)).argmax(dim=1).item()
            ego_preds.append((lab, pred, sidx))

    true_labels = np.array([r[0] for r in ego_preds], dtype=int)
    pred_labels = np.array([r[1] for r in ego_preds], dtype=int)

    ego_acc = accuracy_score(true_labels, pred_labels) * 100
    ego_prec = precision_score(true_labels, pred_labels, zero_division=0) * 100
    ego_rec = recall_score(true_labels, pred_labels, zero_division=0) * 100
    ego_f1 = f1_score(true_labels, pred_labels, zero_division=0) * 100

    # 快照级
    snap_true, snap_pred = defaultdict(int), defaultdict(int)
    for lab, pred, si in ego_preds:
        if lab == 1: snap_true[si] = 1
        if pred == 1: snap_pred[si] = 1
    all_sids = sorted(set(e[2] for e in ego_preds))
    s_tp = sum(1 for s in all_sids if snap_true.get(s, 0) == 1 and snap_pred.get(s, 0) == 1)
    s_fp = sum(1 for s in all_sids if snap_true.get(s, 0) == 0 and snap_pred.get(s, 0) == 1)
    s_tn = sum(1 for s in all_sids if snap_true.get(s, 0) == 0 and snap_pred.get(s, 0) == 0)
    s_fn = sum(1 for s in all_sids if snap_true.get(s, 0) == 1 and snap_pred.get(s, 0) == 0)
    s_total = s_tp + s_fp + s_tn + s_fn
    snap_acc = 100 * (s_tp + s_tn) / max(s_total, 1)
    snap_rec = 100 * s_tp / max(s_tp + s_fn, 1)
    snap_prec = 100 * s_tp / max(s_tp + s_fp, 1)
    snap_f1 = 100 * 2 * s_tp / max(2 * s_tp + s_fp + s_fn, 1)

    return {
        "ego_acc": ego_acc, "ego_prec": ego_prec, "ego_rec": ego_rec, "ego_f1": ego_f1,
        "snap_acc": snap_acc, "snap_rec": snap_rec, "snap_prec": snap_prec, "snap_f1": snap_f1,
        "snap_tp": s_tp, "snap_fp": s_fp, "snap_tn": s_tn, "snap_fn": s_fn,
    }


def train_and_evaluate(strategy_name, handler, mutation_map, seed=42):
    """训练编码器+MLP，做标准评估 + 攻击变体评估"""
    preset = AUGMENT_PRESETS[strategy_name]

    print(f"\n{'='*60}")
    print(f" 策略: {strategy_name}")
    print(f"{'='*60}")

    # ---- Stage 1: 编码器训练 ----
    _set_seed(seed)
    embedder = _train_single_encoder(strategy_name, handler, preset, seed)

    # ---- Stage 2: MLP 训练 ----
    ego_cache = embedder.train_ego_cache
    feat_dim = embedder.enc_out_dim * 2

    classifier = torch.nn.Sequential(
        torch.nn.Linear(feat_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(128, 2),
    ).to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    embedder.encoder.eval()
    classifier.train()

    BATCH_SIZE = 128
    NUM_EPOCHS = 10

    benign_indices = [i for i, (_, _, _, lab, _) in enumerate(ego_cache) if lab == 0]
    attack_indices = [i for i, (_, _, _, lab, _) in enumerate(ego_cache) if lab == 1]
    # 1:1 平衡：取两边较小的数量，不过采样
    max_benign = min(len(benign_indices), max(len(attack_indices) * 10, 500))
    sampled_benign = random.sample(benign_indices, max_benign)
    balanced_attack = (attack_indices * (max_benign // max(len(attack_indices), 1) + 1))[:max_benign]
    balanced_indices = sampled_benign + balanced_attack
    n_balanced = len(balanced_indices)

    for ep in range(NUM_EPOCHS):
        perm = torch.randperm(n_balanced)
        ep_loss, n_batches = 0.0, 0
        for start in range(0, n_balanced, BATCH_SIZE):
            batch_perm = perm[start:start + BATCH_SIZE]
            batch_idx = [balanced_indices[p] for p in batch_perm]
            batch_feats, batch_labels = [], []
            with torch.no_grad():
                for i in batch_idx:
                    x_np, ei, ef, lab, _ = ego_cache[i]
                    x = torch.from_numpy(x_np).to(DEVICE)
                    ei_d, ef_d = ei.to(DEVICE), ef.to(DEVICE)
                    x_a, ei_a, ef_a = embedder.augment_ego(x, ei_d, ef_d, drop_edge_p=0.2, feat_mask_p=0.2)
                    h = embedder.encoder(x_a, ei_a, edge_feat=ef_a)
                    batch_feats.append(torch.cat([h[0], h.mean(dim=0)], dim=0))
                    batch_labels.append(lab)
            if not batch_feats:
                continue
            feats = torch.stack(batch_feats)
            labels = torch.tensor(batch_labels, dtype=torch.long, device=DEVICE)
            loss = criterion(classifier(feats), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n_batches += 1
        if (ep + 1) % 5 == 0:
            print(f"  [MLP] Epoch {ep+1}/{NUM_EPOCHS} Loss={ep_loss/max(n_batches,1):.4f}")

    # ---- 标准评估（原始测试数据）----
    print(f"\n  [{strategy_name}] 标准评估（原始攻击数据）:")
    std_metrics = evaluate_on_cache(embedder, classifier, embedder.test_ego_cache)
    print(f"    Ego:  Acc={std_metrics['ego_acc']:.2f}% Prec={std_metrics['ego_prec']:.2f}% "
          f"Rec={std_metrics['ego_rec']:.2f}% F1={std_metrics['ego_f1']:.2f}%")
    print(f"    Snap: Rec={std_metrics['snap_rec']:.2f}% Prec={std_metrics['snap_prec']:.2f}% "
          f"F1={std_metrics['snap_f1']:.2f}%")

    # ---- 攻击变体评估（LLM-guided 生成的攻击变体）----
    print(f"  [{strategy_name}] 攻击变体评估（良性图+攻击子图生成）:")
    variant_cache = _build_variant_test_cache(embedder, mutation_map)
    if len(variant_cache) == 0:
        print(f"    [WARN] 无攻击变体可测试")
        eva_metrics = {"ego_acc": 0, "ego_prec": 0, "ego_rec": 0, "ego_f1": 0,
                       "snap_rec": 0, "snap_prec": 0, "snap_f1": 0,
                       "snap_tp": 0, "snap_fp": 0, "snap_tn": 0, "snap_fn": 0}
        rec_drop = 0
    else:
        eva_metrics = evaluate_on_cache(embedder, classifier, variant_cache)
        print(f"    Ego:  Acc={eva_metrics['ego_acc']:.2f}% Prec={eva_metrics['ego_prec']:.2f}% "
              f"Rec={eva_metrics['ego_rec']:.2f}% F1={eva_metrics['ego_f1']:.2f}%")
        print(f"    Snap: Rec={eva_metrics['snap_rec']:.2f}% Prec={eva_metrics['snap_prec']:.2f}% "
              f"F1={eva_metrics['snap_f1']:.2f}%")
        rec_drop = std_metrics['ego_rec'] - eva_metrics['ego_rec']
        print(f"    Recall 差异: {rec_drop:+.2f}%")

    return {
        "strategy": strategy_name,
        "standard": std_metrics,
        "evasion": eva_metrics,
        "recall_drop": rec_drop,
    }


# ========================================================================
# 主入口
# ========================================================================

def main():
    parser = argparse.ArgumentParser(description="对抗规避鲁棒性测试")
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="theia")
    parser.add_argument("--scene", type=str, default="theia311")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env_config = load_config()
    path_map = env_config["path_map"]
    handler = prepare_data(path_map, args.dataset, args.scene)

    # 生成攻击变体（正向：良性图+攻击子图，一次生成，所有策略共用）
    mutation_map = generate_attack_variants(handler)

    strategies = [args.strategy] if args.strategy else BASE_STRATEGIES
    all_results = []

    for strat in strategies:
        try:
            result = train_and_evaluate(strat, handler, mutation_map, seed=args.seed)
            all_results.append(result)
        except Exception as e:
            print(f"\n[ERROR] 策略 {strat} 失败: {e}")
            import traceback
            traceback.print_exc()

    # ---- 汇总表 ----
    print(f"\n{'='*90}")
    print(" 鲁棒性对比 — 标准测试 vs 攻击变体检测")
    print(f"{'='*90}")
    print(f"{'策略':<16} | {'标准Rec':>8} {'标准F1':>8} | {'变体Rec':>8} {'变体F1':>8} | {'Rec差异':>8}")
    print("-" * 78)
    for r in all_results:
        s = r['standard']
        e = r['evasion']
        print(f"{r['strategy']:<16} | {s['ego_rec']:7.2f}% {s['ego_f1']:7.2f}% "
              f"| {e['ego_rec']:7.2f}% {e['ego_f1']:7.2f}% | {r['recall_drop']:+7.2f}%")
    print("=" * 78)

    # 快照级汇总
    print(f"\n{'策略':<16} | {'标准SnapRec':>11} | {'变体SnapRec':>11} | {'SnapRec差异':>11}")
    print("-" * 62)
    for r in all_results:
        s = r['standard']
        e = r['evasion']
        drop = s['snap_rec'] - e['snap_rec']
        print(f"{r['strategy']:<16} | {s['snap_rec']:10.2f}% | {e['snap_rec']:10.2f}% | {drop:+10.2f}%")
    print("=" * 62)

    # 保存
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = f"robustness_results_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    log_name = f"bench_robustness_{time.strftime('%Y%m%d-%H%M%S')}.log"
    logf = open(log_name, 'a', encoding='utf-8')

    class _Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, s):
            for f in self.files:
                f.write(s)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    sys.stdout = _Tee(sys.stdout, logf)
    sys.stderr = _Tee(sys.stderr, logf)
    print(f"[Log] writing to {log_name}")
    main()
