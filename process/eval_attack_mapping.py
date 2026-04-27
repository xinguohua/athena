"""
RQ3: ATT&CK Technique Mapping Accuracy 评估脚本（label 级）。

对 DARPA E3 每个恶意 UUID 单独做 ATT&CK 技术映射，比较4种增强方法：
  - Direct: 无增强（原始事件字符串 + 原始技术三元组）
  - Tech-Enhanced: 仅技术侧增强（原始事件 + 转换后三元组）
  - Log-Enhanced: 仅日志侧增强（翻译事件 + 原始三元组）
  - Full-Enhanced: 双侧增强（翻译事件 + 转换后三元组）

用法:
  python -m process.eval_attack_mapping
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import yaml

from process.technique_semantic_mapper import TechniqueSemanticMapper
from process.translation_rules import (
    EVENT_MAP, translate_event, get_process_role, get_file_type,
    LOW_INFO_PREFIXES, TYPE_MAP,
)
from process.datahandlers.darpa_handler import collect_nodes_from_log
from process.datahandlers.common import collect_json_paths, collect_label_paths


def get_parent_technique(tech_id: str) -> str:
    """T1059.004 → T1059"""
    return tech_id.split(".")[0] if "." in tech_id else tech_id


# ============================================================
# 为单个恶意 UUID 构建行为描述（查询文本）
# ============================================================

def build_label_query_raw(uuid: str, df: pd.DataFrame, node_props: dict,
                          node_types: dict) -> str:
    """无增强：直接拼接原始事件类型和属性。"""
    parts = []
    seen = set()

    ntype = node_types.get(uuid, "")
    prop = node_props.get(uuid, "")

    # 作为 actor 的行为
    as_actor = df[df["actorID"] == uuid]
    for _, row in as_actor.iterrows():
        action = row["action"]
        obj_id = row["objectID"]
        obj_type = node_types.get(obj_id, row.get("object", ""))
        obj_prop = node_props.get(obj_id, "")
        desc = f"{ntype} {action} {obj_type} {obj_prop}".strip()
        if desc not in seen:
            seen.add(desc)
            parts.append(desc)

    # 作为 object 的行为
    as_object = df[df["objectID"] == uuid]
    for _, row in as_object.iterrows():
        action = row["action"]
        actor_id = row["actorID"]
        actor_type = node_types.get(actor_id, row.get("actor_type", ""))
        actor_prop = node_props.get(actor_id, "")
        desc = f"{actor_type} {actor_prop} {action} {ntype} {prop}".strip()
        if desc not in seen:
            seen.add(desc)
            parts.append(desc)

    # 去重后限制数量
    return ". ".join(parts[:30]) if parts else ""


def build_label_query_enhanced(uuid: str, df: pd.DataFrame, node_props: dict,
                               node_types: dict) -> str:
    """日志侧增强：翻译事件类型和节点类型为自然语言。"""
    parts = []
    seen = set()

    ntype = node_types.get(uuid, "")
    prop = node_props.get(uuid, "")

    # 翻译主语
    def translate_subject(ntype_str, prop_str):
        if ntype_str == "SUBJECT_PROCESS":
            pname = _extract_proc_name(prop_str)
            if pname:
                role = get_process_role(pname)
                # 如果 get_process_role 没命中映射表，返回原名
                # 对于未知进程名统一映射为 "process"
                from process.translation_rules import PROCESS_ROLE_MAP
                if role == pname and pname.lower() not in PROCESS_ROLE_MAP:
                    return "process"
                return role
            return "process"
        return TYPE_MAP.get(ntype_str, ntype_str)

    # 翻译宾语（只保留语义类型，去掉原始路径/IP 噪声）
    def translate_object(ntype_str, prop_str, action_str=""):
        # EXECUTE 动作的目标一律视为 executable
        if action_str in ("EVENT_EXECUTE", "EVENT_LOADLIBRARY"):
            return "executable"
        mapped = TYPE_MAP.get(ntype_str, "")
        if mapped:
            if "FILE" in ntype_str and prop_str:
                # /proc/ 读取 → process information（保留 discovery 语义）
                if "/proc/" in prop_str:
                    return "process information"
                ft = get_file_type(prop_str)
                return ft if ft else "file"
            return mapped
        if ntype_str == "MemoryObject":
            return "process memory"
        return ntype_str

    # 翻译动作
    def translate_action(action_str):
        return EVENT_MAP.get(action_str, action_str)

    def make_triple(subj, verb, obj_desc):
        """构造三元组。如果 verb 已包含宾语（如 'sends network connection'），不再追加。"""
        if " " in verb:
            return f"{subj} {verb}"
        return f"{subj} {verb} {obj_desc}"

    # 作为 actor 的行为
    as_actor = df[df["actorID"] == uuid]
    for _, row in as_actor.iterrows():
        action = row["action"]
        obj_id = row["objectID"]
        obj_type = node_types.get(obj_id, row.get("object", ""))
        obj_prop = node_props.get(obj_id, "")

        subj = translate_subject(ntype, prop)
        verb = translate_action(action)
        obj_desc = translate_object(obj_type, obj_prop, action)

        # 过滤低信息量动作
        if any(verb.startswith(p) for p in LOW_INFO_PREFIXES):
            continue

        triple = make_triple(subj, verb, obj_desc).strip()
        if triple not in seen:
            seen.add(triple)
            parts.append(triple)

    # 作为 object 的行为
    as_object = df[df["objectID"] == uuid]
    for _, row in as_object.iterrows():
        action = row["action"]
        actor_id = row["actorID"]
        actor_type = node_types.get(actor_id, row.get("actor_type", ""))
        actor_prop = node_props.get(actor_id, "")

        subj = translate_subject(actor_type, actor_prop)
        verb = translate_action(action)
        obj_desc = translate_object(ntype, prop, action)

        if any(verb.startswith(p) for p in LOW_INFO_PREFIXES):
            continue

        triple = make_triple(subj, verb, obj_desc).strip()
        if triple not in seen:
            seen.add(triple)
            parts.append(triple)

    return ". ".join(parts[:30]) if parts else ""


def _extract_proc_name(prop_str: str) -> str:
    """从进程属性字符串中提取进程名。格式: 'cmdLine,tgid,path' 或 'path'"""
    if not prop_str:
        return ""
    # subject2pro 格式: "cmdLine,tgid,path"
    parts = prop_str.split(",")
    if len(parts) >= 3:
        # 取 path（第三个字段），提取文件名
        path = parts[2].strip()
        if "/" in path:
            return path.rsplit("/", 1)[-1]
        return path
    # 可能只是路径
    if "/" in prop_str:
        return prop_str.rsplit("/", 1)[-1]
    return prop_str.strip()


# ============================================================
# 主流程
# ============================================================

def main():
    data_dir = os.path.dirname(__file__)
    raw_triples = os.path.join(data_dir, "data/technique_triples_raw.json")
    trans_triples = os.path.join(data_dir, "data/technique_triples_transformed.json")
    gt_path = os.path.join(data_dir, "data/e3_label_ground_truth.json")

    # 加载 ground truth
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    config_path = os.path.join(data_dir, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    scenes = [
        ("cadets", "cadets314"),
        ("theia", "theia311"),
        ("trace", "trace315"),
        ("clearscope", "clearscope3.6"),
    ]

    # 收集所有恶意 UUID 的查询和 ground truth
    all_entries = []  # (uuid, scene, raw_query, enh_query, gt_technique)

    for ds, scene in scenes:
        base_path = config["remote"]["path_map"].get(ds)
        if not base_path or not os.path.exists(base_path):
            print(f"[SKIP] {ds}")
            continue

        print(f"\n[加载] {ds}/{scene} ...")

        # 读取恶意标签
        label_map = collect_label_paths(base_path)
        labels = []
        if scene in label_map:
            with open(label_map[scene]) as f:
                labels = [l.strip() for l in f if l.strip()]

        # 读取节点属性
        json_map = collect_json_paths(base_path)
        node_props = {}
        if scene in json_map:
            for cat, json_files in json_map[scene].items():
                if cat == "malicious":
                    net, sub, fil = collect_nodes_from_log(json_files)
                    node_props.update(net)
                    node_props.update(sub)
                    node_props.update(fil)

        # 读取边数据
        mal_txt = os.path.join(base_path, f"{scene}_malicious.txt")
        df = pd.read_csv(mal_txt, sep='\t', header=None,
                         names=['actorID', 'actor_type', 'objectID', 'object', 'action', 'timestamp'],
                         dtype=str).dropna()

        # 建立节点类型映射
        node_types = {}
        for _, row in df.iterrows():
            if row["actorID"] not in node_types:
                node_types[row["actorID"]] = row["actor_type"]
            if row["objectID"] not in node_types:
                node_types[row["objectID"]] = row["object"]

        # ground truth 中该场景的 UUID
        scene_gt = gt_data.get(scene, {})

        n_valid = 0
        n_skip = 0
        for uuid in labels:
            if uuid not in scene_gt:
                n_skip += 1
                continue

            gt_tech = scene_gt[uuid]["technique"]

            # 构建查询
            raw_q = build_label_query_raw(uuid, df, node_props, node_types)
            enh_q = build_label_query_enhanced(uuid, df, node_props, node_types)

            all_entries.append({
                "uuid": uuid,
                "scene": scene,
                "raw_query": raw_q,
                "enh_query": enh_q,
                "gt_technique": gt_tech,
                "gt_parent": get_parent_technique(gt_tech),
                "node_type": scene_gt[uuid].get("type", ""),
                "behavior": scene_gt[uuid].get("behavior", ""),
                "tactic": scene_gt[uuid].get("tactic", ""),
            })
            n_valid += 1

        print(f"  有效恶意 UUID: {n_valid}, 跳过(无GT): {n_skip}")

    print(f"\n总评估 label 数: {len(all_entries)}")

    # 初始化映射器
    print("\n[初始化映射器] ...")
    mapper_raw = TechniqueSemanticMapper(triples_path=raw_triples, top_k=10, threshold=0.0)
    mapper_trans = TechniqueSemanticMapper(triples_path=trans_triples, top_k=10, threshold=0.0)
    # Full-Enhanced 用混合匹配：trans 主库(操作级对齐) + raw 辅助库(区分度补充)
    mapper_hybrid = TechniqueSemanticMapper(
        triples_path=trans_triples, aux_triples_path=raw_triples,
        top_k=10, threshold=0.0, aux_weight=0.5,
    )

    # 4 种方法
    methods = [
        ("Direct",        lambda e: e["raw_query"], mapper_raw),
        ("Tech-Enhanced", lambda e: e["raw_query"], mapper_trans),
        ("Log-Enhanced",  lambda e: e["enh_query"], mapper_raw),
        ("Full-Enhanced", lambda e: e["enh_query"], mapper_hybrid),
    ]

    GAMMA = 0.40   # 置信度阈值（基于 raw triples score）
    TOP_K = 10     # 在 top-K 中命中即算 correct

    results_all = {}
    for method_name, get_query, mapper in methods:
        correct = 0
        incorrect = 0
        unmapped = 0
        details = []

        for entry in all_entries:
            query = get_query(entry)
            gt_parent = entry["gt_parent"]

            if not query or not query.strip():
                unmapped += 1
                details.append({**entry, "query_text": "", "pred": None, "score": 0,
                                "confidence": 0, "status": "unmapped", "top_k": []})
                continue

            # 用 raw mapper 计算置信度（所有方法统一标准）
            enh_q = entry["enh_query"]
            raw_conf_result = mapper_raw.predict_top(enh_q) if enh_q else None
            confidence = raw_conf_result[1] if raw_conf_result else 0

            if confidence < GAMMA:
                unmapped += 1
                details.append({
                    **entry, "query_text": query, "pred": None, "score": 0,
                    "confidence": round(confidence, 4),
                    "status": "unmapped", "top_k": []
                })
                continue

            # 用各自 mapper 做排序（带描述文本）
            top_detail = mapper.predict_top_k_detail(query)
            if not top_detail:
                unmapped += 1
                details.append({**entry, "query_text": query, "pred": None,
                                "score": 0, "confidence": round(confidence, 4),
                                "status": "unmapped", "top_k": []})
                continue

            pred_id = top_detail[0]["tech_id"]
            score = top_detail[0]["score"]
            pred_parent = get_parent_technique(pred_id)

            # 检查 top-K 中是否有 parent 匹配
            top_k_parents = [get_parent_technique(t["tech_id"]) for t in top_detail[:TOP_K]]
            hit_in_topk = gt_parent in top_k_parents

            if hit_in_topk:
                correct += 1
                status = "correct"
            else:
                incorrect += 1
                status = "incorrect"

            # 找到第一个匹配的 rank
            match_rank = -1
            for rank, tp in enumerate(top_k_parents):
                if tp == gt_parent:
                    match_rank = rank + 1
                    break

            # GT 技术的描述文本（从 raw mapper 取，因为 raw 最完整）
            gt_tech_text = mapper_raw.get_tech_text(entry["gt_technique"])
            if not gt_tech_text:
                gt_tech_text = mapper_raw.get_tech_text(gt_parent)

            details.append({
                **entry, "query_text": query,
                "pred": pred_id, "pred_parent": pred_parent,
                "score": round(score, 4), "confidence": round(confidence, 4),
                "status": status, "match_rank": match_rank,
                "top_k": [
                    {"tech_id": t["tech_id"], "score": round(t["score"], 3),
                     "tech_text": t["tech_text"][:200]}
                    for t in top_detail[:5]
                ],
                "gt_tech_text": gt_tech_text[:200] if gt_tech_text else "",
            })

        total = correct + incorrect + unmapped
        results_all[method_name] = {
            "correct": correct, "incorrect": incorrect, "unmapped": unmapped,
            "total": total,
            "correct_pct": round(correct / total * 100, 1) if total else 0,
            "incorrect_pct": round(incorrect / total * 100, 1) if total else 0,
            "unmapped_pct": round(unmapped / total * 100, 1) if total else 0,
            "details": details,
        }

    # ============================================================
    # 打印结果
    # ============================================================

    print(f"\n评估标准: Hit@{TOP_K}（GT parent technique 出现在 top-{TOP_K} 预测中即为 correct）")
    print("\n" + "=" * 65)
    print(f"{'Method':<18} {'Correct%':>10} {'Incorrect%':>12} {'Unmapped%':>11}")
    print("-" * 65)
    for m in ["Direct", "Tech-Enhanced", "Log-Enhanced", "Full-Enhanced"]:
        r = results_all[m]
        print(f"{m:<18} {r['correct_pct']:>9.1f}% {r['incorrect_pct']:>11.1f}% {r['unmapped_pct']:>10.1f}%")
    print("=" * 65)
    print(f"Total labels: {results_all['Direct']['total']}")

    # 分场景结果
    print("\n[分场景结果]")
    for scene in ["cadets314", "theia311", "trace315", "clearscope3.6"]:
        scene_entries = [e for e in all_entries if e["scene"] == scene]
        n = len(scene_entries)
        if n == 0:
            continue
        print(f"\n--- {scene} ({n} labels) ---")
        for m in ["Direct", "Tech-Enhanced", "Log-Enhanced", "Full-Enhanced"]:
            d = [x for x in results_all[m]["details"] if x["scene"] == scene]
            c = sum(1 for x in d if x["status"] == "correct")
            ic = sum(1 for x in d if x["status"] == "incorrect")
            um = sum(1 for x in d if x["status"] == "unmapped")
            pct = round(c / n * 100, 1)
            print(f"  {m:<18} correct={c:2d} incorrect={ic:2d} unmapped={um:2d}  ({pct:5.1f}%)")

    # 逐 label 详细结果（Full-Enhanced）
    print(f"\n\n[Full-Enhanced 逐 label 详情 (Hit@{TOP_K})]")
    print(f"{'UUID':<42} {'Type':<18} {'GT':>7} {'Top1':>12} {'Score':>6} {'Rank':>5} {'Status':>9}")
    print("-" * 110)
    for d in results_all["Full-Enhanced"]["details"]:
        uuid_short = d["uuid"][:40]
        ntype = d.get("node_type", "")[:16]
        gt = d["gt_parent"]
        pred = d.get("pred", "-") or "-"
        score = d.get("score", 0)
        rank = d.get("match_rank", -1)
        rank_str = str(rank) if rank > 0 else "-"
        status = d["status"]
        print(f"{uuid_short:<42} {ntype:<18} {gt:>7} {pred:>12} {score:>6.3f} {rank_str:>5} {status:>9}")

    # 保存（记录完整查询文本和技术描述）
    out_path = os.path.join(data_dir, "..", "rq3_mapping_results.json")
    out_data = {
        "total_labels": len(all_entries),
        "results": {m: {k: v for k, v in r.items() if k != "details"}
                    for m, r in results_all.items()},
        "details": {m: [{k: v for k, v in d.items()
                         if k not in ("raw_query", "enh_query")}
                        for d in r["details"]]
                    for m, r in results_all.items()},
    }
    # 同时保存带完整文本的详细版本（供论文案例分析用）
    out_detail_path = os.path.join(data_dir, "..", "rq3_mapping_results_detail.json")
    out_detail = {
        "total_labels": len(all_entries),
        "details": {m: r["details"] for m, r in results_all.items()},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    with open(out_detail_path, "w", encoding="utf-8") as f:
        json.dump(out_detail, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")
    print(f"详细结果已保存: {out_detail_path}")


if __name__ == "__main__":
    main()
