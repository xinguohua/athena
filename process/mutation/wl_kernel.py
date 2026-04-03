"""
Weisfeiler-Leman subtree kernel 实现

用于计算两个 igraph 图之间的结构相似度（论文公式 4）。
使用实体类型（process/file/socket）、操作类型（read/write/exec/fork/connect）
和节点属性（进程名、命令行参数、文件路径、远程地址）作为初始标签。
"""
from __future__ import annotations
from collections import Counter
from typing import List, Optional
import hashlib
import math

try:
    import igraph as ig
except ImportError:
    ig = None


def _node_initial_label(g, v_idx: int) -> str:
    """构造节点初始标签：entity_type + name（截断）"""
    attrs = g.vs[v_idx].attributes()
    entity_type = str(attrs.get("type", "UNK"))
    name = str(attrs.get("name", ""))
    # 截断长名，避免哈希碰撞过少
    name_short = name[:64] if len(name) > 64 else name
    return f"{entity_type}|{name_short}"


def _edge_label(g, e_idx: int) -> str:
    """构造边标签：操作类型"""
    attrs = g.es[e_idx].attributes()
    return str(attrs.get("actions", "UNK"))


def wl_subtree_labels(g, h: int = 3) -> List[Counter]:
    """
    执行 h 轮 WL 迭代，返回每轮的标签直方图列表。

    Args:
        g: igraph.Graph
        h: WL 迭代轮数

    Returns:
        List[Counter]: 长度 h+1，每个 Counter 是该轮的标签频率分布
    """
    n = g.vcount()
    if n == 0:
        return [Counter() for _ in range(h + 1)]

    # 初始标签
    labels = [_node_initial_label(g, i) for i in range(n)]
    histograms = [Counter(labels)]

    for _ in range(h):
        new_labels = []
        for v in range(n):
            # 收集邻居标签（含边标签）
            neighbor_labels = []
            for e_idx in g.incident(v, mode="all"):
                e = g.es[e_idx]
                src, dst = e.source, e.target
                neighbor = dst if src == v else src
                edge_lbl = _edge_label(g, e_idx)
                neighbor_labels.append(f"{edge_lbl}:{labels[neighbor]}")
            neighbor_labels.sort()
            # 拼接：自身标签 + 排序后邻居标签
            combined = labels[v] + "|" + "|".join(neighbor_labels)
            # 哈希压缩（避免标签无限膨胀）
            new_labels.append(hashlib.md5(combined.encode()).hexdigest()[:16])
        labels = new_labels
        histograms.append(Counter(labels))

    return histograms


def wl_kernel(g1, g2, h: int = 3) -> float:
    """
    计算两个图的 WL subtree kernel 值（归一化内积）。

    K_WL(g1, g2) = sum_i <phi_i(g1), phi_i(g2)> / (||phi(g1)|| * ||phi(g2)||)

    Args:
        g1, g2: igraph.Graph
        h: WL 迭代轮数

    Returns:
        float: 归一化相似度 [0, 1]
    """
    hist1 = wl_subtree_labels(g1, h)
    hist2 = wl_subtree_labels(g2, h)

    # 计算每轮的内积，然后求和
    dot = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0

    for c1, c2 in zip(hist1, hist2):
        all_keys = set(c1.keys()) | set(c2.keys())
        for k in all_keys:
            v1 = c1.get(k, 0)
            v2 = c2.get(k, 0)
            dot += v1 * v2
            norm1_sq += v1 * v1
            norm2_sq += v2 * v2

    norm1 = math.sqrt(norm1_sq) if norm1_sq > 0 else 1e-12
    norm2 = math.sqrt(norm2_sq) if norm2_sq > 0 else 1e-12

    return dot / (norm1 * norm2)


def top_k_similar_attacks(benign_graph, attack_graphs: list, k: int = 5, h: int = 3) -> list:
    """
    从攻击图集合中检索与 benign_graph 结构最相似的 Top-K 攻击图（论文公式 4）。

    Args:
        benign_graph: 锚点良性图
        attack_graphs: 标注的攻击图集合 [(graph, snapshot_idx), ...]
        k: 返回数量
        h: WL 迭代轮数

    Returns:
        [(graph, snapshot_idx, similarity), ...] 按相似度降序排列
    """
    scored = []
    for ag, sidx in attack_graphs:
        sim = wl_kernel(benign_graph, ag, h=h)
        scored.append((ag, sidx, sim))

    scored.sort(key=lambda x: -x[2])
    return scored[:k]
