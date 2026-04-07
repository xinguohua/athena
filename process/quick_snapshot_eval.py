"""快速快照级评估：加载已有编码器 + 重训 MLP + ego/快照两级指标"""
import pickle, numpy as np, torch, random, time
from collections import Counter, defaultdict

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
with open('snapshot_data_bench_theia_theia311.pkl', 'rb') as f:
    d = pickle.load(f)
snapshots = d['all_snapshots']
a_s, a_e = d['malicious_idx_start'], d['malicious_idx_end']
b_s, b_e = d['benign_idx_start'], d['benign_idx_end']

# 重建 embedder（只做推理）
from process.embedders.gcc_embedder_dev import GCCEmbedderDev, GINEncoder

embedder = GCCEmbedderDev.__new__(GCCEmbedderDev)
embedder.snapshots = snapshots
embedder.train_snapshot_indices = list(range(b_s, b_e + 1))
embedder.device = DEVICE
embedder.r_hop = 2
embedder.ego_max_nodes = 64
embedder.prop_feat_dim = 128
embedder.enc_out_dim = 256
embedder.use_degree_coop_augment = True
embedder.drop_edge_p = 0.2
embedder.feat_mask_p = 0.2
embedder._prop_cache = {}
embedder._w2v_model = None
embedder.w2v_pretrained_path = None
embedder.w2v_window = 5
embedder.w2v_min_count = 1
embedder.w2v_sg = 1
embedder.w2v_epochs = 20

state = torch.load('gcc_encoder_dev_bench_llm_guided_42.pth', map_location='cpu')
from process.embedders.gcc_embedder_dev import TypedGINConv
embedder.encoder = GINEncoder(128, 64, 256, num_layers=3, dropout=0.1).to(DEVICE)
embedder.encoder.load_state_dict(state['encoder'])
embedder.encoder.eval()

# 构建训练集 + 测试集
SAMPLE = 50

def build_ego_set(sid_range, sample_benign=True):
    """构建 ego 集合，返回 [(x_np, ei, ef, lab, sidx)]"""
    egos = []
    for sid in sid_range:
        g = snapshots[sid]
        if g is None or g.vcount() == 0:
            continue
        embedder._preheat_snapshot_properties(g)
        attacks = [v for v in range(g.vcount()) if g.vs[v].attributes().get('label', 0) == 1]
        benigns = [v for v in range(g.vcount()) if g.vs[v].attributes().get('label', 0) == 0]
        if sample_benign and len(benigns) > SAMPLE:
            benigns = random.sample(benigns, SAMPLE)
        nodes = attacks + benigns
        for v in nodes:
            sub = embedder._ego_subgraph(g, v, r=2, max_nodes=64)
            if sub.vcount() == 0:
                continue
            x_np = embedder._build_node_features(sub)
            ei, ef = embedder._igraph_edges_to_edge_index(sub)
            lab = int(g.vs[v].attributes().get('label', 0))
            egos.append((x_np, ei.cpu(), ef.cpu(), lab, sid))
    return egos

print("构建训练集（良性快照）...")
train_egos = build_ego_set(range(b_s, b_e + 1), sample_benign=True)
# 攻击 ego 也加入训练
print("构建攻击 ego（恶意快照）...")
for sid in range(a_s, a_e + 1):
    g = snapshots[sid]
    if g is None or g.vcount() == 0:
        continue
    embedder._preheat_snapshot_properties(g)
    for v in range(g.vcount()):
        if g.vs[v].attributes().get('label', 0) == 1:
            sub = embedder._ego_subgraph(g, v, r=2, max_nodes=64)
            if sub.vcount() == 0:
                continue
            x_np = embedder._build_node_features(sub)
            ei, ef = embedder._igraph_edges_to_edge_index(sub)
            train_egos.append((x_np, ei.cpu(), ef.cpu(), 1, sid))

print("构建测试集（恶意快照）...")
test_egos = build_ego_set(range(a_s, a_e + 1), sample_benign=True)

n_train_b = sum(1 for e in train_egos if e[3] == 0)
n_train_a = sum(1 for e in train_egos if e[3] == 1)
n_test_b = sum(1 for e in test_egos if e[3] == 0)
n_test_a = sum(1 for e in test_egos if e[3] == 1)
print(f"训练集: {len(train_egos)} (良性={n_train_b}, 攻击={n_train_a})")
print(f"测试集: {len(test_egos)} (良性={n_test_b}, 攻击={n_test_a})")

# 编码
def encode_egos(egos):
    embs = []
    with torch.no_grad():
        for x_np, ei, ef, lab, sid in egos:
            x = torch.from_numpy(x_np).to(DEVICE)
            h = embedder.encoder(x, ei.to(DEVICE), edge_feat=ef.to(DEVICE))
            embs.append(h.mean(dim=0).cpu().numpy())  # mean pooling，和 Stage 1 一致
    return np.array(embs, dtype=np.float32)

print("编码训练集...")
train_X = encode_egos(train_egos)
train_y = np.array([e[3] for e in train_egos], dtype=int)
print("编码测试集...")
test_X = encode_egos(test_egos)
test_y = np.array([e[3] for e in test_egos], dtype=int)
test_sids = [e[4] for e in test_egos]

# 训练 MLP
import torch.nn as nn
classifier = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 2)).to(DEVICE)

benign_idx = [i for i, y in enumerate(train_y) if y == 0]
attack_idx = [i for i, y in enumerate(train_y) if y == 1]
max_benign = min(len(benign_idx), max(len(attack_idx) * 10, 500))
sampled_b = random.sample(benign_idx, max_benign)
balanced_a = (attack_idx * (max_benign // max(len(attack_idx), 1) + 1))[:max_benign]
balanced = sampled_b + balanced_a
print(f"MLP 训练: {len(sampled_b)} 良性 + {len(balanced_a)} 攻击")

criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
train_Xt = torch.tensor(train_X, device=DEVICE)
train_yt = torch.tensor(train_y, dtype=torch.long, device=DEVICE)

classifier.train()
for ep in range(10):
    perm = torch.randperm(len(balanced))
    for s in range(0, len(balanced), 128):
        idx = [balanced[p] for p in perm[s:s+128]]
        out = classifier(train_Xt[idx])
        loss = criterion(out, train_yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()

# 预测
classifier.eval()
test_Xt = torch.tensor(test_X, device=DEVICE)
with torch.no_grad():
    preds = classifier(test_Xt).argmax(dim=1).cpu().numpy()

# ===== Ego 级 =====
tp = int(np.sum((test_y == 1) & (preds == 1)))
fp = int(np.sum((test_y == 0) & (preds == 1)))
tn = int(np.sum((test_y == 0) & (preds == 0)))
fn = int(np.sum((test_y == 1) & (preds == 0)))
print(f"\n{'='*70}")
print(f"Ego 级: Acc={100*(tp+tn)/len(test_y):.2f}% Prec={100*tp/max(tp+fp,1):.2f}% Rec={100*tp/max(tp+fn,1):.2f}% F1={100*2*tp/max(2*tp+fp+fn,1):.2f}%")
print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")

# ===== 快照级 =====
snap_true = defaultdict(int)
snap_pred = defaultdict(int)
for i, sid in enumerate(test_sids):
    if test_y[i] == 1:
        snap_true[sid] = 1
    if preds[i] == 1:
        snap_pred[sid] = 1

all_sids = sorted(set(test_sids))
s_tp = sum(1 for s in all_sids if snap_true.get(s, 0) == 1 and snap_pred.get(s, 0) == 1)
s_fp = sum(1 for s in all_sids if snap_true.get(s, 0) == 0 and snap_pred.get(s, 0) == 1)
s_tn = sum(1 for s in all_sids if snap_true.get(s, 0) == 0 and snap_pred.get(s, 0) == 0)
s_fn = sum(1 for s in all_sids if snap_true.get(s, 0) == 1 and snap_pred.get(s, 0) == 0)
s_total = s_tp + s_fp + s_tn + s_fn

print(f"\n快照级 (any-ego-positive): Acc={100*(s_tp+s_tn)/max(s_total,1):.2f}% Prec={100*s_tp/max(s_tp+s_fp,1):.2f}% Rec={100*s_tp/max(s_tp+s_fn,1):.2f}% F1={100*2*s_tp/max(2*s_tp+s_fp+s_fn,1):.2f}%")
print(f"  TP={s_tp} FP={s_fp} TN={s_tn} FN={s_fn} (共{s_total}个快照)")

# 快照级详情
if s_fp > 0:
    print(f"\n  FP 快照:")
    for s in all_sids:
        if snap_true.get(s, 0) == 0 and snap_pred.get(s, 0) == 1:
            n_ego = sum(1 for i, sid in enumerate(test_sids) if sid == s)
            n_fp_ego = sum(1 for i, sid in enumerate(test_sids) if sid == s and test_y[i] == 0 and preds[i] == 1)
            print(f"    snap[{s}]: {n_fp_ego}/{n_ego} ego 误判")

if s_fn > 0:
    print(f"\n  FN 快照:")
    for s in all_sids:
        if snap_true.get(s, 0) == 1 and snap_pred.get(s, 0) == 0:
            n_att = sum(1 for i, sid in enumerate(test_sids) if sid == s and test_y[i] == 1)
            print(f"    snap[{s}]: {n_att}个攻击ego全部漏检")

# 恶意快照 TP 详情
if s_tp > 0:
    print(f"\n  TP 快照 ({s_tp}个恶意快照检出):")
    for s in all_sids:
        if snap_true.get(s, 0) == 1 and snap_pred.get(s, 0) == 1:
            n_att = sum(1 for i, sid in enumerate(test_sids) if sid == s and test_y[i] == 1)
            n_att_det = sum(1 for i, sid in enumerate(test_sids) if sid == s and test_y[i] == 1 and preds[i] == 1)
            n_ben_fp = sum(1 for i, sid in enumerate(test_sids) if sid == s and test_y[i] == 0 and preds[i] == 1)
            print(f"    snap[{s}]: 攻击ego {n_att_det}/{n_att} 检出, 良性ego {n_ben_fp}个误判")
