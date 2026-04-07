"""分析 FP ego 子图：检查 BFS 邻域是否包含攻击节点"""
import pickle, numpy as np, torch, random
from collections import Counter, deque

random.seed(42)

with open('snapshot_data_bench_theia_theia311.pkl', 'rb') as f:
    d = pickle.load(f)
snapshots = d['all_snapshots']
a_s, a_e = d['malicious_idx_start'], d['malicious_idx_end']

def ego_subgraph_nodes(g, center, max_nodes=64):
    """BFS 提取 ego 节点列表（不建子图，只返回节点 ID）"""
    visited = [center]
    visited_set = {center}
    queue = deque([center])
    while queue and len(visited) < max_nodes:
        v = queue.popleft()
        for nb in g.neighbors(v, mode="all"):
            if nb not in visited_set and len(visited) < max_nodes:
                visited.append(nb)
                visited_set.add(nb)
                queue.append(nb)
    return visited

# 分析恶意区间每个快照中 benign center 的 ego 是否包含攻击节点
print("=" * 70)
print("分析恶意区间快照中 benign 节点的 BFS ego 邻域是否包含攻击节点")
print("=" * 70)

total_benign = 0
benign_with_attack_in_ego = 0
benign_no_attack = 0
type_with_attack = Counter()
type_no_attack = Counter()

# 统计每个快照的情况
for sid in range(a_s, a_e + 1):
    g = snapshots[sid]
    if g is None or g.vcount() == 0:
        continue

    attack_set = {v for v in range(g.vcount()) if g.vs[v]['label'] == 1}
    benign_nodes = [v for v in range(g.vcount()) if g.vs[v]['label'] == 0]

    if not attack_set:
        continue

    # 采样 benign（和测试集一致：50个）
    if len(benign_nodes) > 50:
        benign_sample = random.sample(benign_nodes, 50)
    else:
        benign_sample = benign_nodes

    snap_fp_risk = 0
    for v in benign_sample:
        total_benign += 1
        ego_nodes = ego_subgraph_nodes(g, v, max_nodes=64)

        # ego 中有多少攻击节点？
        attack_in_ego = attack_set.intersection(ego_nodes)

        vtype = str(g.vs[v]['type'])

        if attack_in_ego:
            benign_with_attack_in_ego += 1
            snap_fp_risk += 1
            type_with_attack[vtype] += 1
        else:
            benign_no_attack += 1
            type_no_attack[vtype] += 1

    if snap_fp_risk > 0:
        print(f"  snap[{sid}]: {g.vcount()}节点, 攻击={len(attack_set)}, "
              f"采样benign={len(benign_sample)}, ego含攻击={snap_fp_risk}")

print(f"\n总 benign ego: {total_benign}")
print(f"  ego 包含攻击节点: {benign_with_attack_in_ego} ({100*benign_with_attack_in_ego/max(total_benign,1):.1f}%)")
print(f"  ego 不含攻击节点: {benign_no_attack} ({100*benign_no_attack/max(total_benign,1):.1f}%)")

print(f"\nego 含攻击节点的 benign 按类型:")
for t, c in type_with_attack.most_common():
    print(f"  {t}: {c}")

print(f"\nego 不含攻击节点的 benign 按类型:")
for t, c in type_no_attack.most_common():
    print(f"  {t}: {c}")

# 更细致：分析 hop 距离
print("\n" + "=" * 70)
print("攻击节点到 benign center 的最短路径 hop 分布")
print("=" * 70)

hop_dist = Counter()
for sid in range(a_s, a_e + 1):
    g = snapshots[sid]
    if g is None or g.vcount() == 0:
        continue
    attack_set = {v for v in range(g.vcount()) if g.vs[v]['label'] == 1}
    if not attack_set:
        continue

    benign_nodes = [v for v in range(g.vcount()) if g.vs[v]['label'] == 0]
    if len(benign_nodes) > 50:
        benign_nodes = random.sample(benign_nodes, 50)

    for v in benign_nodes:
        # BFS 计算到最近攻击节点的距离
        visited = {v}
        queue = deque([(v, 0)])
        min_hop = -1
        while queue:
            node, hop = queue.popleft()
            if node in attack_set and node != v:
                min_hop = hop
                break
            if hop < 3:  # 检查 3 hop 范围
                for nb in g.neighbors(node, mode="all"):
                    if nb not in visited:
                        visited.add(nb)
                        queue.append((nb, hop + 1))
        if min_hop > 0:
            hop_dist[min_hop] += 1

print("到最近攻击节点的 hop 距离:")
for h in sorted(hop_dist.keys()):
    print(f"  {h}-hop: {hop_dist[h]} 个 benign 节点")
no_attack_nearby = total_benign - sum(hop_dist.values())
print(f"  >3-hop 或无连通: {no_attack_nearby} 个")

# 分析：BFS ego 中攻击节点数量分布
print("\n" + "=" * 70)
print("BFS ego (max_nodes=64) 中包含的攻击节点数量分布")
print("=" * 70)

attack_count_dist = Counter()
random.seed(42)
for sid in range(a_s, a_e + 1):
    g = snapshots[sid]
    if g is None or g.vcount() == 0:
        continue
    attack_set = {v for v in range(g.vcount()) if g.vs[v]['label'] == 1}
    if not attack_set:
        continue
    benign_nodes = [v for v in range(g.vcount()) if g.vs[v]['label'] == 0]
    if len(benign_nodes) > 50:
        benign_nodes = random.sample(benign_nodes, 50)

    for v in benign_nodes:
        ego_nodes = ego_subgraph_nodes(g, v, max_nodes=64)
        n_attack = len(attack_set.intersection(ego_nodes))
        if n_attack > 0:
            attack_count_dist[n_attack] += 1

print("ego 中攻击节点数量: 个数")
for n, c in sorted(attack_count_dist.items()):
    print(f"  {n}个攻击节点: {c} 个 benign ego")
