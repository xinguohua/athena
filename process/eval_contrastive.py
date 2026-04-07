"""量化对比学习在节点级的效果"""
import pickle, numpy as np, torch
from sklearn.metrics import roc_auc_score

with open('snapshot_data_bench_theia_theia311.pkl','rb') as f:
    d = pickle.load(f)
snapshots = d['all_snapshots']

state = torch.load('gcc_encoder_dev_bench_llm_guided_42.pth', map_location='cpu')
node_embs = state.get('snapshot_node_embeddings', [])

# 良性中心
benign_vecs = []
for i in range(d['benign_idx_start'], d['benign_idx_end']+1):
    emb = node_embs[i] if i < len(node_embs) else {}
    benign_vecs.extend(emb.values())
benign_center = np.mean(benign_vecs, axis=0)

tp_ids = [130,131,136,137,140,142,147,149,157,159,160]
fn_ids = [129,146,123,155]

header = f"{'snap':>6} {'nodes':>6} {'att':>4} {'ratio':>6} {'AUC':>7} {'att_rank%':>10} {'det':>4}"
print(header)
print("-" * len(header))

for sid in sorted(tp_ids + fn_ids):
    g = snapshots[sid]
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

    dists = np.array(dists)
    labels = np.array(labels)

    if labels.sum() == 0 or labels.sum() == len(labels):
        continue

    auc = roc_auc_score(labels, dists)

    # 攻击节点在距离排名中的中位百分位（越低越好，0%=最远）
    ranks = np.argsort(np.argsort(-dists))  # 降序排名，0=最远
    attack_ranks = ranks[labels == 1]
    median_pct = np.median(attack_ranks) / len(dists) * 100

    detected = "Y" if sid in tp_ids else "N"
    n_att = int(labels.sum())
    ratio = n_att / len(labels) * 100
    print(f"A[{sid:>3}] {len(labels):>6} {n_att:>4} {ratio:>5.1f}% {auc:>7.3f} {median_pct:>9.1f}% {detected:>4}")
