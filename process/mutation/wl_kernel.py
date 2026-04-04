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
    """构造节点初始标签：entity_type + 粗粒度属性（从 properties 提取）。

    name 是 UUID（跨图不重复），必须用 properties 提取语义信息。
    properties 格式：进程='cmdLine,tgid,path'，文件='filepath'，网络='srcaddr,srcport,dstaddr,dstport'。
    """
    attrs = g.vs[v_idx].attributes()
    entity_type = str(attrs.get("type", "UNK")).lower()
    raw_prop = str(attrs.get("properties", ""))

    # 清理 str(set(...)) 格式：取第一个元素
    prop = raw_prop.strip()
    if prop.startswith("{") and prop.endswith("}"):
        prop = prop[1:-1].strip()
        if prop.startswith("'") or prop.startswith('"'):
            q = prop[0]
            end = prop.find(q, 1)
            if end > 0:
                prop = prop[1:end]

    # 粗粒度：从 properties 提取有区分度的语义标签
    if "process" in entity_type or "subject" in entity_type:
        # 进程：从 properties 的 cmdLine 提取命令名
        cmd_line = prop.split(",")[0] if prop else ""
        parts = cmd_line.split()
        token = parts[0] if parts else "UNK"
        # 取命令名（去掉路径前缀如 /usr/bin/）
        if "/" in token:
            token = token.rstrip("/").rsplit("/", 1)[-1]
        coarse = token[:16]
    elif "file" in entity_type:
        # 文件：properties 就是文件路径，保留目录前 3 层
        fp = prop.strip("{ '\"}")
        segments = fp.split("/")[:4]
        coarse = "/".join(segments) if segments and segments[0] != "" else fp[:24]
    elif "net" in entity_type or "flow" in entity_type or "sock" in entity_type:
        # 网络：properties='srcaddr,srcport,dstaddr,dstport'，提取端口
        parts = prop.strip("{ '\"}")  .split(",")
        if len(parts) >= 4:
            coarse = f"port:{parts[3].strip()}"
        elif len(parts) >= 2:
            coarse = f"port:{parts[1].strip()}"
        else:
            coarse = "net"
    else:
        coarse = prop[:12] if prop else "UNK"

    return f"{entity_type}|{coarse}"


def _edge_label(g, e_idx: int) -> str:
    """构造边标签：操作类型"""
    attrs = g.es[e_idx].attributes()
    return str(attrs.get("actions", "UNK"))


def wl_subtree_labels(g, h: int = 3, max_nodes: int = 5000) -> List[Counter]:
    """
    执行 h 轮 WL 迭代，返回每轮的标签直方图列表。
    大图（>max_nodes）时随机采样节点子集，避免计算爆炸。

    Args:
        g: igraph.Graph
        h: WL 迭代轮数
        max_nodes: 节点数超过此值时采样

    Returns:
        List[Counter]: 长度 h+1，每个 Counter 是该轮的标签频率分布
    """
    import random as _rng
    n = g.vcount()
    if n == 0:
        return [Counter() for _ in range(h + 1)]

    # 大图采样：取子图以降低计算量
    if n > max_nodes:
        sampled = sorted(_rng.sample(range(n), max_nodes))
        g = g.subgraph(sampled)
        n = g.vcount()

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


def _kernel_from_histograms(hist1, hist2) -> float:
    """从预计算的 WL histogram 计算归一化 kernel 值"""
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


def top_k_similar_attacks(benign_graph, attack_graphs: list, k: int = 5, h: int = 3,
                          _attack_hist_cache: dict = None) -> list:
    """
    从攻击图集合中检索与 benign_graph 结构最相似的 Top-K 攻击图（论文公式 4）。
    支持预计算缓存以避免重复计算攻击图的 WL histogram。

    Args:
        benign_graph: 锚点良性图
        attack_graphs: 标注的攻击图集合 [(graph, snapshot_idx), ...]
        k: 返回数量
        h: WL 迭代轮数
        _attack_hist_cache: 可选的攻击图 histogram 缓存 {snapshot_idx: histograms}

    Returns:
        [(graph, snapshot_idx, similarity), ...] 按相似度降序排列
    """
    hist_b = wl_subtree_labels(benign_graph, h)

    scored = []
    for ag, sidx in attack_graphs:
        # 优先使用缓存的攻击图 histogram
        if _attack_hist_cache is not None and sidx in _attack_hist_cache:
            hist_a = _attack_hist_cache[sidx]
        else:
            hist_a = wl_subtree_labels(ag, h)
            if _attack_hist_cache is not None:
                _attack_hist_cache[sidx] = hist_a
        sim = _kernel_from_histograms(hist_b, hist_a)
        scored.append((ag, sidx, sim))

    scored.sort(key=lambda x: -x[2])
    return scored[:k]
