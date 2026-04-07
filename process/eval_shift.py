"""分析攻击节点对快照嵌入的偏移量"""
import pickle, numpy as np, torch

with open('snapshot_data_bench_theia_theia311.pkl', 'rb') as f:
    d = pickle.load(f)
snapshots = d['all_snapshots']

state = torch.load('gcc_encoder_dev_bench_llm_guided_42.pth', map_location='cpu')
node_embs = state.get('snapshot_node_embeddings', [])

tp_ids = [130, 131, 136, 137, 140, 142, 147, 149, 157, 159, 160]
fn_ids = [129, 146, 123, 155]

print(f"{'snap':>6} {'nodes':>5} {'att':>3} {'ratio':>6} | {'shift_norm':>10} {'shift_ratio':>11} | {'det':>3}")
print("-" * 65)

for sid in sorted(tp_ids + fn_ids):
    g = snapshots[sid]
    emb = node_embs[sid] if sid < len(node_embs) else {}
    if not emb:
        continue

    # 所有节点嵌入
    all_vecs = []
    att_vecs = []
    ben_vecs = []
    for v in range(g.vcount()):
        nid = g.vs[v]['name']
        vec = emb.get(nid)
        if vec is None:
            continue
        all_vecs.append(vec)
        if g.vs[v]['label'] == 1:
            att_vecs.append(vec)
        else:
            ben_vecs.append(vec)

    if not att_vecs or not ben_vecs:
        continue

    # 快照嵌入（含攻击节点）= 全部节点均值
    snap_with = np.mean(all_vecs, axis=0)
    # 假设无攻击节点 = 仅良性节点均值
    snap_without = np.mean(ben_vecs, axis=0)

    # 攻击节点造成的偏移
    shift = snap_with - snap_without
    shift_norm = np.linalg.norm(shift)

    # 偏移相对于快照嵌入范数的比例
    snap_norm = np.linalg.norm(snap_with)
    shift_ratio = shift_norm / (snap_norm + 1e-12) * 100

    detected = "Y" if sid in tp_ids else "N"
    n_att = len(att_vecs)
    ratio = n_att / len(all_vecs) * 100

    print(f"A[{sid:>3}] {len(all_vecs):>5} {n_att:>3} {ratio:>5.1f}% | {shift_norm:>10.2f} {shift_ratio:>10.2f}% | {detected:>3}")
