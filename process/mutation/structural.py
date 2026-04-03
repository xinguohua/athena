"""
结构变异 (Structural Mutation) — 论文 Algorithm 1

核心流程：
1. 对每个良性锚图 G_b，用 WL kernel 检索 Top-K 相似攻击图
2. 对每个攻击图，BFS 搜索对齐的子图区域 (S_i, S'_i)
3. LLM 候选选择（按类型匹配率排序 + LLM 打分）
4. 将攻击子图 S' 替换到良性图 G_b 中，生成变异图 G~
"""
from __future__ import annotations
from collections import deque
from typing import List, Tuple, Dict, Optional
import random

try:
    import igraph as ig
except ImportError:
    ig = None


def _token_set(g, v_idx: int) -> set:
    """获取节点的 token 集合（属性值的集合），用于 Jaccard 相似度"""
    attrs = g.vs[v_idx].attributes()
    tokens = set()
    for k, v in attrs.items():
        if k in ("label",):  # 排除标签
            continue
        tokens.add(f"{k}:{v}")
    return tokens


def _jaccard_tok(tokens_a: set, tokens_b: set) -> float:
    """Token-level Jaccard 相似度 J_tok(u, w')"""
    if not tokens_a and not tokens_b:
        return 0.0
    inter = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return inter / union if union > 0 else 0.0


def aligned_region_search(
    g_b, g_a,
    max_region_size: int = 32,
) -> List[Tuple[list, list, dict, float]]:
    """
    BFS 对齐子图区域搜索（Algorithm 1, Lines 3-22）

    对每对跨图 process 节点 (v ∈ G_b, w ∈ G_a)，做 BFS 扩展并
    按 entity type 一致性和 token-level Jaccard 匹配对齐节点。

    Args:
        g_b: 良性图
        g_a: 攻击图
        max_region_size: 对齐区域最大节点数

    Returns:
        List of (S_benign_nodes, S_attack_nodes, mapping, type_match_ratio)
        mapping: attack_node_idx -> benign_node_idx
    """
    # 找到两图中的 process 节点
    def _process_nodes(g):
        result = []
        for i in range(g.vcount()):
            t = str(g.vs[i].get("type", "")).lower()
            if "process" in t or "subject" in t:
                result.append(i)
        return result

    procs_b = _process_nodes(g_b)
    procs_a = _process_nodes(g_a)

    if not procs_b or not procs_a:
        return []

    candidates = []

    for v in procs_b:
        for w in procs_a:
            # 初始化 BFS
            pi = {w: v}  # mapping: attack -> benign
            S_b = {v}
            S_a = {w}
            queue = deque([w])

            while queue and len(S_a) < max_region_size:
                w_c = queue.popleft()
                v_c = pi[w_c]

                # 攻击图中 w_c 的邻居（排除已访问）
                neighbors_a = set(g_a.neighbors(w_c, mode="all")) - S_a
                # 良性图中 v_c 的邻居（排除已映射）
                neighbors_b = set(g_b.neighbors(v_c, mode="all")) - S_b

                for w_prime in neighbors_a:
                    type_a = str(g_a.vs[w_prime].get("type", ""))

                    # 优先找类型一致的邻居
                    type_consistent = [
                        u for u in neighbors_b
                        if str(g_b.vs[u].get("type", "")) == type_a
                    ]

                    if type_consistent:
                        # 在类型一致的邻居中，选 Jaccard 最高的
                        tokens_a = _token_set(g_a, w_prime)
                        best_u = max(type_consistent,
                                     key=lambda u: _jaccard_tok(_token_set(g_b, u), tokens_a))
                        v_prime = best_u
                    elif neighbors_b:
                        # 无类型一致邻居，选任意可用邻居（保持连通性）
                        tokens_a = _token_set(g_a, w_prime)
                        v_prime = max(neighbors_b,
                                      key=lambda u: _jaccard_tok(_token_set(g_b, u), tokens_a))
                    else:
                        continue

                    pi[w_prime] = v_prime
                    S_a.add(w_prime)
                    S_b.add(v_prime)
                    neighbors_b.discard(v_prime)
                    queue.append(w_prime)

            # 计算类型匹配率 ρ
            n_matched = sum(
                1 for w_j, v_j in pi.items()
                if str(g_a.vs[w_j].get("type", "")) == str(g_b.vs[v_j].get("type", ""))
            )
            rho = n_matched / len(pi) if pi else 0.0

            if len(pi) >= 2:  # 至少 2 个节点才有意义
                candidates.append((list(S_b), list(S_a), dict(pi), rho))

    # 按类型匹配率排序，取 top
    candidates.sort(key=lambda x: -x[3])
    return candidates


def subgraph_replacement(
    g_b, g_a,
    S_b_nodes: list,
    S_a_nodes: list,
    pi: dict,
) -> Optional:
    """
    子图替换（Algorithm 1, Lines 27-29）

    将良性图 G_b 中的子图 S 替换为攻击图 G_a 中对齐的子图 S'。
    - S 内部的节点和边被替换为 S' 的
    - S 边界与 G_b 其余部分的连接保持不变
    - 替换边继承攻击图中的操作类型

    Args:
        g_b: 良性图（将被修改的副本）
        g_a: 攻击图（提供攻击子图）
        S_b_nodes: 良性图中要被替换的节点索引列表
        S_a_nodes: 攻击图中的对齐子图节点索引列表
        pi: 映射 attack_node_idx -> benign_node_idx

    Returns:
        igraph.Graph: 变异后的图 G~ (g_b 的副本)，若失败返回 None
    """
    if ig is None:
        return None

    try:
        # 创建 g_b 的副本
        g_mut = g_b.copy()
        S_b_set = set(S_b_nodes)

        # 1. 替换节点属性：将 S 中节点的属性改为对应攻击节点的属性
        for w_idx, v_idx in pi.items():
            if v_idx >= g_mut.vcount() or w_idx >= g_a.vcount():
                continue
            # 复制攻击节点的属性（保留 benign 的 name 用于连接维护）
            attack_attrs = g_a.vs[w_idx].attributes()
            for attr_name, attr_val in attack_attrs.items():
                if attr_name == "name":
                    # 保留良性图原有的 name 以维护边连接
                    continue
                try:
                    g_mut.vs[v_idx][attr_name] = attr_val
                except Exception:
                    pass

        # 2. 替换边：删除 S 内部的边，添加攻击图 S' 内部的边
        # 找到 S 内部的边并删除
        s_internal_edges = []
        for e_idx in range(g_mut.ecount()):
            e = g_mut.es[e_idx]
            if e.source in S_b_set and e.target in S_b_set:
                s_internal_edges.append(e_idx)
        if s_internal_edges:
            g_mut.delete_edges(s_internal_edges)

        # 添加攻击图 S' 内部的边（映射到良性图节点索引）
        inv_pi = {v: w for w, v in pi.items()}  # benign -> attack
        S_a_set = set(S_a_nodes)

        for e_idx in range(g_a.ecount()):
            e = g_a.es[e_idx]
            if e.source in S_a_set and e.target in S_a_set:
                # 两端都在攻击子图中，映射到良性图的节点索引
                v_src = pi.get(e.source)
                v_dst = pi.get(e.target)
                if v_src is not None and v_dst is not None:
                    edge_attrs = e.attributes()
                    try:
                        g_mut.add_edge(v_src, v_dst, **edge_attrs)
                    except Exception:
                        g_mut.add_edge(v_src, v_dst)

        return g_mut

    except Exception as ex:
        print(f"[StructMut] 子图替换失败: {ex}")
        return None


def generate_structural_mutations(
    benign_graphs: list,
    attack_graphs: list,
    top_k: int = 5,
    top_m: int = 3,
    max_region_size: int = 32,
    max_mutations: int = 50,
) -> List[Tuple]:
    """
    批量结构变异：为一批良性图生成变异图。

    Args:
        benign_graphs: [(graph, snapshot_idx), ...]
        attack_graphs: [(graph, snapshot_idx), ...]
        top_k: 每个良性图检索的攻击图数量
        top_m: 每个攻击图选取的候选区域数量
        max_region_size: 对齐区域最大节点数
        max_mutations: 最大生成变异图数

    Returns:
        [(mutated_graph, source_benign_idx, source_attack_idx), ...]
    """
    from .wl_kernel import top_k_similar_attacks

    mutations = []

    for g_b, b_idx in benign_graphs:
        if len(mutations) >= max_mutations:
            break

        # 检索 Top-K 相似攻击图
        similar_attacks = top_k_similar_attacks(
            g_b, attack_graphs, k=top_k, h=3
        )

        for g_a, a_idx, sim in similar_attacks:
            if len(mutations) >= max_mutations:
                break

            # 搜索对齐子图区域
            candidates = aligned_region_search(g_b, g_a, max_region_size=max_region_size)

            # 取 top_m 候选
            for S_b, S_a, pi, rho in candidates[:top_m]:
                if len(mutations) >= max_mutations:
                    break
                if rho < 0.3:  # 类型匹配率过低则跳过
                    continue

                # 执行子图替换
                g_mut = subgraph_replacement(g_b, g_a, S_b, S_a, pi)
                if g_mut is not None:
                    mutations.append((g_mut, b_idx, a_idx))

    print(f"[StructMut] 生成 {len(mutations)} 个结构变异图")
    return mutations
