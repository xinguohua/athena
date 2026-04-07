"""分析 ego 级检测结果：FP 和 FN 的详细信息"""
import pickle, numpy as np, torch, random
from collections import Counter

random.seed(42)

with open('snapshot_data_bench_theia_theia311.pkl', 'rb') as f:
    d = pickle.load(f)
snapshots = d['all_snapshots']
a_s, a_e = d['malicious_idx_start'], d['malicious_idx_end']

# 加载模型
state = torch.load('gcc_encoder_dev_bench_llm_guided_42.pth', map_location='cpu')

from process.embedders.gcc_embedder_dev import GCCEmbedderDev
embedder = GCCEmbedderDev.__new__(GCCEmbedderDev)
embedder.snapshots = snapshots
embedder.train_snapshot_indices = list(range(d['benign_idx_start'], d['benign_idx_end'] + 1))
embedder.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedder.r_hop = 2
embedder.ego_max_nodes = 64
embedder.prop_feat_dim = 128
embedder.enc_out_dim = 256
embedder._prop_cache = {}
embedder._w2v_model = None

# 重建 encoder
from process.embedders.gcc_embedder_dev import GINEncoder
embedder.encoder = GINEncoder(128, 128, 256, num_layers=3, dropout=0.1).to(embedder.device)
embedder.encoder.load_state_dict(state['encoder'])
embedder.encoder.eval()

# 重建 MLP (需要从最近的训练中获取，这里重新训练一个快速版本)
# 先构建测试集
SAMPLE = 50
test_egos = []  # (x_np, ei, ef, lab, sidx, vtype, prop)

print("构建测试集...")
for sid in range(a_s, a_e + 1):
    g = snapshots[sid]
    if g is None or g.vcount() == 0:
        continue
    embedder._preheat_snapshot_properties(g)
    attacks = [v for v in range(g.vcount()) if g.vs[v]['label'] == 1]
    benigns = [v for v in range(g.vcount()) if g.vs[v]['label'] == 0]
    if len(benigns) > SAMPLE:
        benigns = random.sample(benigns, SAMPLE)

    for v in attacks + benigns:
        sub = embedder._ego_subgraph(g, v, r=2, max_nodes=64)
        if sub.vcount() == 0:
            continue
        x_np = embedder._build_node_features(sub)
        ei, ef = embedder._igraph_edges_to_edge_index(sub)
        lab = int(g.vs[v]['label'])
        vtype = str(g.vs[v]['type'])
        prop = str(g.vs[v]['properties'])[:60]
        test_egos.append((x_np, ei.cpu(), ef.cpu(), lab, sid, vtype, prop))

n_att = sum(1 for t in test_egos if t[3] == 1)
n_ben = sum(1 for t in test_egos if t[3] == 0)
print(f"测试集: {len(test_egos)} ego (攻击={n_att}, 良性={n_ben})")

# 编码测试集
print("编码测试集...")
test_embs = []
with torch.no_grad():
    for x_np, ei, ef, lab, sid, vtype, prop in test_egos:
        x = torch.from_numpy(x_np).to(embedder.device)
        h = embedder.encoder(x, ei.to(embedder.device), edge_feat=ef.to(embedder.device))
        test_embs.append(h[0].cpu().numpy())

test_X = np.array(test_embs, dtype=np.float32)
test_y = np.array([t[3] for t in test_egos], dtype=int)

# 快速训练 MLP (用攻击池的 ego)
print("训练 MLP...")
# 构建训练集：良性快照 ego + 攻击 ego
train_embs, train_labels = [], []
for sid in range(d['benign_idx_start'], d['benign_idx_end'] + 1):
    g = snapshots[sid]
    if g is None or g.vcount() == 0:
        continue
    embedder._preheat_snapshot_properties(g)
    nodes = list(range(g.vcount()))
    if len(nodes) > 50:
        nodes = random.sample(nodes, 50)
    for v in nodes:
        sub = embedder._ego_subgraph(g, v, r=2, max_nodes=64)
        if sub.vcount() == 0:
            continue
        x = torch.from_numpy(embedder._build_node_features(sub)).to(embedder.device)
        ei, ef = embedder._igraph_edges_to_edge_index(sub)
        with torch.no_grad():
            h = embedder.encoder(x, ei.to(embedder.device), edge_feat=ef.to(embedder.device))
        train_embs.append(h[0].cpu().numpy())
        train_labels.append(0)

# 攻击 ego
for sid in range(a_s, a_e + 1):
    g = snapshots[sid]
    if g is None:
        continue
    for v in range(g.vcount()):
        if g.vs[v]['label'] == 1:
            sub = embedder._ego_subgraph(g, v, r=2, max_nodes=64)
            if sub.vcount() == 0:
                continue
            embedder._preheat_snapshot_properties(g)
            x = torch.from_numpy(embedder._build_node_features(sub)).to(embedder.device)
            ei, ef = embedder._igraph_edges_to_edge_index(sub)
            with torch.no_grad():
                h = embedder.encoder(x, ei.to(embedder.device), edge_feat=ef.to(embedder.device))
            train_embs.append(h[0].cpu().numpy())
            train_labels.append(1)

train_X = np.array(train_embs, dtype=np.float32)
train_y = np.array(train_labels, dtype=int)
n_train_att = int(train_y.sum())
print(f"训练集: {len(train_y)} (良性={len(train_y)-n_train_att}, 攻击={n_train_att})")

# 简单 MLP
import torch.nn as nn
device = embedder.device
classifier = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 2)).to(device)
crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, (len(train_y)-n_train_att)/max(n_train_att,1)], device=device))
opt = torch.optim.Adam(classifier.parameters(), lr=1e-3)

train_Xt = torch.tensor(train_X, device=device)
train_yt = torch.tensor(train_y, dtype=torch.long, device=device)
classifier.train()
for ep in range(20):
    perm = torch.randperm(len(train_y), device=device)
    for s in range(0, len(train_y), 256):
        idx = perm[s:s+256]
        out = classifier(train_Xt[idx])
        loss = crit(out, train_yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()

# 预测测试集
classifier.eval()
test_Xt = torch.tensor(test_X, device=device)
with torch.no_grad():
    preds = classifier(test_Xt).argmax(dim=1).cpu().numpy()

# 分析结果
print("\n" + "=" * 70)
print("FN (攻击漏检):")
fn_props = []
for i, (x_np, ei, ef, lab, sid, vtype, prop) in enumerate(test_egos):
    if lab == 1 and preds[i] == 0:
        print(f"  snap={sid} {vtype} {prop}")
        fn_props.append(prop)

print(f"\nFP (良性误判) 统计:")
fp_types = Counter()
fp_snaps = Counter()
for i, (x_np, ei, ef, lab, sid, vtype, prop) in enumerate(test_egos):
    if lab == 0 and preds[i] == 1:
        fp_types[vtype] += 1
        fp_snaps[sid] += 1

print(f"  总 FP: {sum(fp_types.values())}")
print(f"  按类型:")
for t, c in fp_types.most_common():
    total_of_type = sum(1 for x in test_egos if x[5] == t and x[3] == 0)
    print(f"    {t}: {c}/{total_of_type} ({c/max(total_of_type,1)*100:.1f}%)")

print(f"  FP 最多的快照 (top 10):")
for sid, c in fp_snaps.most_common(10):
    total = sum(1 for x in test_egos if x[4] == sid and x[3] == 0)
    print(f"    A[{sid}] {c}/{total} FP ({snapshots[sid].vcount()}节点)")

# 汇总
tp = int(np.sum((test_y == 1) & (preds == 1)))
fp = int(np.sum((test_y == 0) & (preds == 1)))
tn = int(np.sum((test_y == 0) & (preds == 0)))
fn = int(np.sum((test_y == 1) & (preds == 0)))
print(f"\nego 级: Acc={100*(tp+tn)/len(test_y):.1f}% Prec={100*tp/max(tp+fp,1):.1f}% Rec={100*tp/max(tp+fn,1):.1f}% F1={100*2*tp/max(2*tp+fp+fn,1):.1f}%")
print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
