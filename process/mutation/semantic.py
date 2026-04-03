"""
语义变异 (Semantic Mutation) — 论文 Figure 5

对结构变异后注入的攻击节点，使用 LLM 修改其属性使其融入良性上下文。
三种策略：
- Replacement: 命令名在良性语料中，参数不在 → 替换参数
- Rewriting: 命令名和参数都在良性语料中 → 重写整个命令
- Extension: 命令名和参数都不在良性语料中 → 前后追加良性操作
"""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Set
import json
import os

try:
    import igraph as ig
except ImportError:
    ig = None


def _collect_benign_corpus(benign_graphs: list) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    从历史良性图中收集属性白名单 H_b。

    Returns:
        (benign_commands, benign_args, benign_files):
        - benign_commands: 出现过的进程命令名集合
        - benign_args: 出现过的命令行参数集合
        - benign_files: 出现过的文件路径集合
    """
    commands = set()
    args = set()
    files = set()

    for g, _ in benign_graphs:
        for v in range(g.vcount()):
            attrs = g.vs[v].attributes()
            name = str(attrs.get("name", ""))
            vtype = str(attrs.get("type", "")).lower()

            if "process" in vtype or "subject" in vtype:
                # 进程节点：提取命令名和参数
                parts = name.split()
                if parts:
                    commands.add(parts[0])
                    for p in parts[1:]:
                        args.add(p)
            elif "file" in vtype:
                files.add(name)

    return commands, args, files


def _assign_strategy(
    node_name: str,
    benign_commands: Set[str],
    benign_args: Set[str],
) -> str:
    """
    为进程节点分配变异策略。

    - Replacement: 命令名在 H_b，参数不在 → 保留命令，替换参数
    - Rewriting: 命令名和参数都在 H_b → 重写整个命令
    - Extension: 命令名和参数都不在 H_b → 追加良性操作
    """
    parts = node_name.split()
    if not parts:
        return "extension"

    cmd = parts[0]
    node_args = set(parts[1:]) if len(parts) > 1 else set()

    cmd_in_benign = cmd in benign_commands
    args_in_benign = bool(node_args) and node_args.issubset(benign_args)

    if cmd_in_benign and not args_in_benign:
        return "replacement"
    elif cmd_in_benign and args_in_benign:
        return "rewriting"
    else:
        return "extension"


def _build_context_triples(g, node_idx: int, r_hop: int = 2) -> List[str]:
    """
    提取节点的 r-hop 边界上下文，编码为因果三元组序列。
    格式: <entity_type:attribute, operation_type, entity_type:attribute>
    """
    triples = []
    visited = {node_idx}
    frontier = {node_idx}

    for _ in range(r_hop):
        next_frontier = set()
        for v in frontier:
            for e_idx in g.incident(v, mode="all"):
                e = g.es[e_idx]
                src, dst = e.source, e.target
                neighbor = dst if src == v else src
                if neighbor in visited:
                    continue
                next_frontier.add(neighbor)
                visited.add(neighbor)

                # 构建三元组
                action = str(e.attributes().get("actions", "UNK"))
                src_type = str(g.vs[src].get("type", "UNK"))
                src_name = str(g.vs[src].get("name", ""))[:32]
                dst_type = str(g.vs[dst].get("type", "UNK"))
                dst_name = str(g.vs[dst].get("name", ""))[:32]

                triple = f"<{src_type}:{src_name}, {action}, {dst_type}:{dst_name}>"
                triples.append(triple)
        frontier = next_frontier

    return triples


def build_semantic_mutation_prompt(
    nodes_info: List[Dict],
    context_triples: List[List[str]],
) -> str:
    """
    构建语义变异的 LLM prompt（论文 Figure 5）。

    Args:
        nodes_info: [{"node_id": int, "attributes": str, "associated_nodes": str,
                       "strategy": str}, ...]
        context_triples: 对应每个节点的上下文三元组

    Returns:
        str: LLM prompt
    """
    prompt_parts = [
        "[Context] The following attack process nodes are embedded in a provenance graph "
        "with their associated file/network nodes and r-hop benign context C.\n"
    ]
    prompt_parts.append(f"[Input] {len(nodes_info)} process nodes:\n")

    for i, info in enumerate(nodes_info):
        ctx = "; ".join(context_triples[i][:20]) if i < len(context_triples) else ""
        prompt_parts.append(
            f"  Node {i+1}: attributes={info['attributes']}, "
            f"associated_nodes={info.get('associated_nodes', 'N/A')}, "
            f"strategy={info['strategy']}, C={{{ctx}}}\n"
        )

    prompt_parts.append(
        "\n[Task] Mutate each process node's attributes via its assigned strategy:\n"
        "  A. Replacement: preserve the attack-critical component, replace the other "
        "with benign values guided by C.\n"
        "  B. Rewriting: rewrite the entire command into a different expression guided "
        "by C that fits the surrounding benign context.\n"
        "  C. Extension: preserve entire command, prepend/append benign operations from C.\n"
        "For each mutated process node, also output updated attributes for any associated "
        "file or network nodes whose values change.\n"
        "Constraints: (1) generated values must be legitimate system commands, file names, "
        "or IP addresses; (2) mutated values must be compatible with neighboring operations in C.\n"
        "\n[Output] JSON: [{node_id, new_attributes, associated_updates}].\n"
    )

    return "".join(prompt_parts)


def apply_semantic_mutation_llm(
    g_mut,
    attack_node_indices: List[int],
    benign_commands: Set[str],
    benign_args: Set[str],
    llm_fn=None,
    r_hop: int = 2,
) -> Optional:
    """
    对变异图中的攻击节点执行语义变异。

    Args:
        g_mut: 结构变异后的图（将被修改）
        attack_node_indices: 攻击节点在 g_mut 中的索引
        benign_commands: 良性命令集合
        benign_args: 良性参数集合
        llm_fn: LLM 调用函数 (prompt: str) -> str，若为 None 则使用规则变异
        r_hop: 上下文提取半径

    Returns:
        修改后的图，或 None（失败时）
    """
    if not attack_node_indices:
        return g_mut

    # 收集需要变异的进程节点信息
    process_nodes = []
    for idx in attack_node_indices:
        if idx >= g_mut.vcount():
            continue
        vtype = str(g_mut.vs[idx].get("type", "")).lower()
        if "process" not in vtype and "subject" not in vtype:
            continue
        name = str(g_mut.vs[idx].get("name", ""))
        strategy = _assign_strategy(name, benign_commands, benign_args)
        process_nodes.append({
            "idx": idx,
            "name": name,
            "strategy": strategy,
        })

    if not process_nodes:
        return g_mut

    if llm_fn is not None:
        # LLM 语义变异
        nodes_info = []
        context_triples = []
        for pn in process_nodes:
            ctx = _build_context_triples(g_mut, pn["idx"], r_hop=r_hop)
            context_triples.append(ctx)
            # 收集关联的文件/网络节点
            associated = []
            for nb in g_mut.neighbors(pn["idx"], mode="all"):
                nb_type = str(g_mut.vs[nb].get("type", "")).lower()
                nb_name = str(g_mut.vs[nb].get("name", ""))
                if "file" in nb_type or "net" in nb_type or "flow" in nb_type:
                    associated.append(f"{nb_type}:{nb_name}")
            nodes_info.append({
                "node_id": pn["idx"],
                "attributes": pn["name"],
                "associated_nodes": "; ".join(associated[:5]),
                "strategy": pn["strategy"],
            })

        prompt = build_semantic_mutation_prompt(nodes_info, context_triples)

        try:
            response = llm_fn(prompt)
            mutations = _parse_llm_response(response)
            _apply_mutations(g_mut, mutations, process_nodes)
        except Exception as ex:
            print(f"[SemMut] LLM 变异失败，回退到规则变异: {ex}")
            _apply_rule_based_mutation(g_mut, process_nodes, benign_commands, benign_args)
    else:
        # 规则变异（无 LLM 时的 fallback）
        _apply_rule_based_mutation(g_mut, process_nodes, benign_commands, benign_args)

    return g_mut


def _parse_llm_response(response: str) -> List[Dict]:
    """解析 LLM 返回的 JSON 变异结果"""
    # 尝试提取 JSON
    try:
        # 处理 markdown 代码块
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        result = json.loads(text)
        if isinstance(result, list):
            return result
        return []
    except Exception:
        return []


def _apply_mutations(g_mut, mutations: List[Dict], process_nodes: List[Dict]):
    """将 LLM 输出的变异应用到图上"""
    node_map = {pn["idx"]: pn for pn in process_nodes}

    for mut in mutations:
        node_id = mut.get("node_id")
        new_attrs = mut.get("new_attributes", "")
        if node_id is None or node_id not in node_map:
            continue
        if new_attrs and node_id < g_mut.vcount():
            g_mut.vs[node_id]["name"] = str(new_attrs)

        # 关联节点更新
        for update in mut.get("associated_updates", []):
            if isinstance(update, dict):
                for k, v in update.items():
                    try:
                        # 尝试匹配关联节点并更新
                        for nb in g_mut.neighbors(node_id, mode="all"):
                            if str(g_mut.vs[nb].get("name", "")).startswith(str(k)[:10]):
                                g_mut.vs[nb]["name"] = str(v)
                                break
                    except Exception:
                        pass


def _apply_rule_based_mutation(
    g_mut,
    process_nodes: List[Dict],
    benign_commands: Set[str],
    benign_args: Set[str],
):
    """
    规则变异（无 LLM 时的 fallback）：
    根据策略直接替换/重写/扩展节点名称
    """
    benign_cmd_list = list(benign_commands) if benign_commands else ["bash"]
    benign_arg_list = list(benign_args) if benign_args else ["-l", "--help"]

    import random as _rng

    for pn in process_nodes:
        idx = pn["idx"]
        name = pn["name"]
        strategy = pn["strategy"]
        parts = name.split()

        if idx >= g_mut.vcount():
            continue

        if strategy == "replacement" and len(parts) > 1:
            # 保留命令名，替换参数为良性参数
            cmd = parts[0]
            n_args = min(len(parts) - 1, 3)
            new_args = _rng.sample(benign_arg_list, min(n_args, len(benign_arg_list)))
            new_name = f"{cmd} {' '.join(new_args)}"
        elif strategy == "rewriting":
            # 替换为随机良性命令
            new_cmd = _rng.choice(benign_cmd_list) if benign_cmd_list else "ls"
            n_args = _rng.randint(0, 2)
            new_args = _rng.sample(benign_arg_list, min(n_args, len(benign_arg_list)))
            new_name = f"{new_cmd} {' '.join(new_args)}".strip()
        else:  # extension
            # 在命令前后追加良性操作
            prefix_cmd = _rng.choice(benign_cmd_list) if benign_cmd_list else "echo"
            new_name = f"{prefix_cmd} && {name}"

        g_mut.vs[idx]["name"] = new_name
