"""
统一验证 (Unified Verification) — 论文 Section IV-C.2

4 项验证检查，确保变异图是有效的难负样本：
1. Operation Legality: 替换边的操作类型必须属于源节点的历史操作集合
2. Attribute Feasibility: 变异后的属性值必须在同类型实体的白名单中
3. Imperceptibility: 不能引入用户可感知的操作组合（GUI/通知等）
4. Hardness: 变异图与锚点的 WL 相似度 >= δ_h
"""
from __future__ import annotations
from collections import defaultdict
from typing import Set, Dict, List, Optional, Tuple
import math

try:
    import igraph as ig
except ImportError:
    ig = None


# 用户可感知操作的黑名单（论文参考 ProvNinja [32]）
PERCEIVABLE_BLACKLIST = {
    # entity_type + operation 组合
    ("process", "gui_create"),
    ("process", "gui_show"),
    ("process", "notify"),
    ("process", "alert"),
    ("process", "prompt"),
    ("process", "dialog"),
}


def build_historical_profiles(benign_graphs: list) -> Tuple[
    Dict[str, Set[str]],  # entity -> allowed_ops
    Dict[str, Set[str]],  # entity_type -> allowed_attrs
]:
    """
    从良性图集合构建历史行为剖面 H_b。

    Returns:
        (entity_ops, type_attrs):
        - entity_ops: {entity_name: {op_types}} 每个实体的历史操作类型集合
        - type_attrs: {entity_type: {attribute_values}} 每类实体的属性白名单
    """
    entity_ops: Dict[str, Set[str]] = defaultdict(set)
    type_attrs: Dict[str, Set[str]] = defaultdict(set)

    for g, _ in benign_graphs:
        for e_idx in range(g.ecount()):
            e = g.es[e_idx]
            src_name = str(g.vs[e.source].get("name", ""))
            action = str(e.attributes().get("actions", ""))
            entity_ops[src_name].add(action)

        for v_idx in range(g.vcount()):
            vtype = str(g.vs[v_idx].get("type", "")).lower()
            vname = str(g.vs[v_idx].get("name", ""))
            type_attrs[vtype].add(vname)

    return dict(entity_ops), dict(type_attrs)


def check_operation_legality(
    g_mut,
    replaced_nodes: set,
    entity_ops: Dict[str, Set[str]],
) -> bool:
    """
    检查 1: Operation Legality
    替换区域边界处的每条新边，其操作类型必须属于源节点的历史操作集合。
    """
    for e_idx in range(g_mut.ecount()):
        e = g_mut.es[e_idx]
        src, dst = e.source, e.target

        # 只检查边界边（一端在替换区域内，一端在区域外）
        src_in = src in replaced_nodes
        dst_in = dst in replaced_nodes
        if src_in == dst_in:
            continue  # 两端都在内或都在外，不是边界边

        action = str(e.attributes().get("actions", ""))
        src_name = str(g_mut.vs[src].get("name", ""))

        if src_name in entity_ops:
            if action not in entity_ops[src_name]:
                return False

    return True


def check_attribute_feasibility(
    g_mut,
    mutated_nodes: set,
    type_attrs: Dict[str, Set[str]],
) -> bool:
    """
    检查 2: Attribute Feasibility
    每个变异节点的属性值必须在同类型实体的白名单中。
    """
    for v_idx in mutated_nodes:
        if v_idx >= g_mut.vcount():
            continue
        vtype = str(g_mut.vs[v_idx].get("type", "")).lower()
        vname = str(g_mut.vs[v_idx].get("name", ""))

        allowed = type_attrs.get(vtype, set())
        if allowed and vname not in allowed:
            # 宽松匹配：检查命令名部分是否在白名单中
            cmd = vname.split()[0] if vname.split() else ""
            cmd_in_any = any(cmd in a for a in allowed)
            if not cmd_in_any:
                return False

    return True


def check_imperceptibility(g_mut, replaced_nodes: set) -> bool:
    """
    检查 3: Imperceptibility
    变异引入的边不能包含用户可感知的操作组合。
    """
    for e_idx in range(g_mut.ecount()):
        e = g_mut.es[e_idx]
        if e.source not in replaced_nodes and e.target not in replaced_nodes:
            continue

        action = str(e.attributes().get("actions", "")).lower()
        src_type = str(g_mut.vs[e.source].get("type", "")).lower()

        for etype, op in PERCEIVABLE_BLACKLIST:
            if etype in src_type and op in action:
                return False

    return True


def check_hardness(g_mut, g_anchor, delta_h: float = 0.5, wl_h: int = 3) -> bool:
    """
    检查 4: Hardness
    变异图与锚点良性图的 WL 相似度 >= δ_h。
    """
    from .wl_kernel import wl_kernel
    sim = wl_kernel(g_mut, g_anchor, h=wl_h)
    return sim >= delta_h


def verify_mutation(
    g_mut,
    g_anchor,
    replaced_nodes: set,
    entity_ops: Dict[str, Set[str]],
    type_attrs: Dict[str, Set[str]],
    delta_h: float = 0.5,
) -> Tuple[bool, List[str]]:
    """
    执行全部 4 项验证。

    Returns:
        (passed, failed_checks): passed 为 True 表示全部通过
    """
    failed = []

    if not check_operation_legality(g_mut, replaced_nodes, entity_ops):
        failed.append("operation_legality")

    if not check_attribute_feasibility(g_mut, replaced_nodes, type_attrs):
        failed.append("attribute_feasibility")

    if not check_imperceptibility(g_mut, replaced_nodes):
        failed.append("imperceptibility")

    if not check_hardness(g_mut, g_anchor, delta_h=delta_h):
        failed.append("hardness")

    return len(failed) == 0, failed
