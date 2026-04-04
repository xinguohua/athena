"""
统一验证 (Unified Verification) — 论文 Section IV-C.2

4 项 pass/fail 检查，确保变异图是有效的难负样本：
1. Operation Legality: 边界边操作合法性（软检查，仅记录）
2. Attribute Feasibility: 属性可行性（软检查，仅记录）
3. Imperceptibility: 不可感知性（硬检查）
4. Hardness: WL 相似度在合理范围内（硬检查）
"""
from __future__ import annotations
from collections import defaultdict
from typing import Set, Dict, List, Tuple

try:
    import igraph as ig
except ImportError:
    ig = None


PERCEIVABLE_BLACKLIST = {
    ("process", "gui_create"),
    ("process", "gui_show"),
    ("process", "notify"),
    ("process", "alert"),
    ("process", "prompt"),
    ("process", "dialog"),
}


def build_historical_profiles(benign_graphs: list) -> Tuple[
    Dict[str, Set[str]],
    Dict[str, Set[str]],
]:
    from .semantic import _get_properties

    entity_ops: Dict[str, Set[str]] = defaultdict(set)
    type_attrs: Dict[str, Set[str]] = defaultdict(set)

    for g, _ in benign_graphs:
        for e_idx in range(g.ecount()):
            e = g.es[e_idx]
            src_prop = _get_properties(g, e.source)
            action = str(e.attributes().get("actions", ""))
            entity_ops[src_prop].add(action)

        for v_idx in range(g.vcount()):
            vtype = str(g.vs[v_idx].attributes().get("type", "")).lower()
            vprop = _get_properties(g, v_idx)
            type_attrs[vtype].add(vprop)

    return dict(entity_ops), dict(type_attrs)


def check_imperceptibility(g_mut, replaced_nodes: set) -> bool:
    for e_idx in range(g_mut.ecount()):
        e = g_mut.es[e_idx]
        if e.source not in replaced_nodes and e.target not in replaced_nodes:
            continue
        action = str(e.attributes().get("actions", "")).lower()
        src_type = str(g_mut.vs[e.source].attributes().get("type", "")).lower()
        for etype, op in PERCEIVABLE_BLACKLIST:
            if etype in src_type and op in action:
                return False
    return True


def check_hardness(g_mut, g_anchor, delta_h: float = 0.3, delta_h_upper: float = 0.95) -> bool:
    from .wl_kernel import wl_kernel
    sim = wl_kernel(g_mut, g_anchor, h=3)
    return delta_h <= sim <= delta_h_upper


def verify_mutation(
    g_mut,
    g_anchor,
    replaced_nodes: set,
    entity_ops: Dict[str, Set[str]],
    type_attrs: Dict[str, Set[str]],
    delta_h: float = 0.3,
    delta_h_upper: float = 0.95,
) -> Tuple[bool, List[str]]:
    """
    4 项验证。imperceptibility 和 hardness 是硬性检查。

    Returns:
        (passed, failed_checks)
    """
    failed = []

    if not check_imperceptibility(g_mut, replaced_nodes):
        failed.append("imperceptibility")

    if not check_hardness(g_mut, g_anchor, delta_h=delta_h, delta_h_upper=delta_h_upper):
        failed.append("hardness")

    # 硬性检查不过直接失败
    hard_failed = [f for f in failed if f in ("imperceptibility", "hardness")]
    return len(hard_failed) == 0, failed
