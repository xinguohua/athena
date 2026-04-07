"""对比节点级 vs 快照级的距离，找出漏检原因"""
import pickle, numpy as np, torch

with open('snapshot_data_bench_theia_theia311.pkl', 'rb') as f:
    d = pickle.load(f)
snapshots = d['all_snapshots']

state = torch.load('gcc_encoder_dev_bench_llm_guided_42.pth', map_location='cpu')
node_embs = state.get('snapshot_node_embeddings', [])
snap_embs = state.get('snapshot_embeddings', None)

# 良性节点中心
benign_vecs = []
for i in range(d['benign_idx_start'], d['benign_idx_end'] + 1):
    emb = node_embs[i] if i < len(node_embs) else {}
    benign_vecs.extend(emb.values())
benign_node_center = np.mean(benign_vecs, axis=0)

# 重新算快照嵌入（跟 get_snapshot_embeddings 一致）
from process.embedders.gcc_embedder_dev import GCCEmbedderDev
embedder = GCCEmbedderDev.__new__(GCCEmbedderDev)
embedder.snapshot_node_embeddings = node_embs
embedder.snapshots = snapshots
embedder.train_snapshot_indices = list(range(d['benign_idx_start'], d['benign_idx_end'] + 1))
embedder.attr_weight_alpha = 0.3
embedder.enc_out_dim = len(benign_node_center)
snap_embs = embedder.get_snapshot_embeddings()

# 良性快照中心
benign_snap_center = snap_embs[d['benign_idx_start']:d['benign_idx_end'] + 1].mean(axis=0)
benign_snap_dists = [np.linalg.norm(snap_embs[i] - benign_snap_center)
                     for i in range(d['benign_idx_start'], d['benign_idx_end'] + 1)]
snap_p95 = np.percentile(benign_snap_dists, 95)
snap_p99 = np.percentile(benign_snap_dists, 99)

tp_ids = [130, 131, 136, 137, 140, 142, 147, 149, 157, 159, 160]
fn_ids = [129, 146, 123, 155]

from sklearn.metrics import roc_auc_score

print(f"良性快照距离: mean={np.mean(benign_snap_dists):.2f}, p95={snap_p95:.2f}, p99={snap_p99:.2f}")
print()
print(f"{'snap':>6} {'nodes':>5} {'att':>3} | {'node_AUC':>8} {'att_rank%':>9} | {'snap_dist':>9} {'>p99?':>5} | {'det':>3}")
print("-" * 75)

for sid in sorted(tp_ids + fn_ids):
    g = snapshots[sid]
    emb = node_embs[sid] if sid < len(node_embs) else {}
    if not emb:
        continue

    # 节点级 AUC
    dists, labels = [], []
    for v in range(g.vcount()):
        nid = g.vs[v]['name']
        vec = emb.get(nid)
        if vec is None:
            continue
        dists.append(np.linalg.norm(vec - benign_node_center))
        labels.append(int(g.vs[v]['label']))
    dists = np.array(dists)
    labels = np.array(labels)
    auc = roc_auc_score(labels, dists) if labels.sum() > 0 and labels.sum() < len(labels) else 0
    ranks = np.argsort(np.argsort(-dists))
    att_rank_pct = np.median(ranks[labels == 1]) / len(dists) * 100

    # 快照级距离
    snap_dist = np.linalg.norm(snap_embs[sid] - benign_snap_center)
    above_p99 = "Y" if snap_dist > snap_p99 else "N"

    detected = "Y" if sid in tp_ids else "N"
    n_att = int(labels.sum())

    print(f"A[{sid:>3}] {len(labels):>5} {n_att:>3} | {auc:>8.3f} {att_rank_pct:>8.1f}% | {snap_dist:>9.2f} {above_p99:>5} | {detected:>3}")
