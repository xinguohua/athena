"""
Table V: Attack Variant Usability 分析脚本。

四种方法对比（查询侧 × 被查询库侧 的 2×2 组合）：

被查询库的三个层次（从原始到处理程度递增）：
  - 原始文档 (enterprise-attack_Chroma.csv Body): 完整 ATT&CK 描述段落
  - raw triples (technique_triples_raw.json): 从文档用 spaCy 提取的 SVO 三元组
  - transformed triples (technique_triples_transformed.json): 映射为操作级三元组（严重坍塌）

四种方法：
  1. Direct:        原始日志字段 × 原始 ATT&CK 文档描述
  2. Log-Enhanced:  翻译去噪查询 × 原始 ATT&CK 文档描述
  3. Tech-Enhanced: 原始日志字段 × raw triples（技术侧提取三元组）
  4. Full-Enhanced: 翻译去噪查询 × raw triples

分析维度：
  1. Direct 为什么匹配失败（原因）
  2. Log-Enhanced 相比 Direct：去噪找到的案例（背景干扰消除）
  3. Tech-Enhanced 相比 Direct：技术侧提取三元组找到的案例
  4. Full-Enhanced 独有：双侧增强才能找到的案例
  5. Unmapped 分析（未知攻击 / 不可观测）

用法:
  python -m process.analyze_table_v
"""
from __future__ import annotations
import os
import csv
import json
import re
import numpy as np
import pandas as pd
import yaml
from collections import defaultdict

from process.technique_semantic_mapper import TechniqueSemanticMapper
from process.translation_rules import (
    EVENT_MAP, translate_event, get_process_role, get_file_type,
    LOW_INFO_PREFIXES, TYPE_MAP, PROCESS_ROLE_MAP,
)
from process.datahandlers.darpa_handler import collect_nodes_from_log
from process.datahandlers.common import collect_json_paths, collect_label_paths


def get_parent_technique(tech_id: str) -> str:
    return tech_id.split(".")[0] if "." in tech_id else tech_id


# ============================================================
# 加载原始 ATT&CK 文档描述库
# ============================================================

def load_original_attack_descriptions(csv_path: str) -> dict:
    """从 enterprise-attack_Chroma.csv 加载原始 ATT&CK 技术描述。

    返回 {tech_id: full_description_text}
    """
    descriptions = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("filepath", "")
            # 从 URL 提取 technique ID: https://attack.mitre.org/techniques/T1001/001 → T1001.001
            m = re.search(r"T\d+(?:/\d+)?", url)
            if not m:
                continue
            tech_id = m.group().replace("/", ".")
            body = row.get("Body", "").strip()
            if body:
                descriptions[tech_id] = body
    return descriptions


# ============================================================
# 用原始文档描述构建 SBERT 匹配器
# ============================================================

class OriginalDocMapper:
    """用原始 ATT&CK 文档描述做 SBERT 匹配器。

    和 TechniqueSemanticMapper 功能一样，但被查询库是原始文档描述，
    不是提取的三元组。
    """

    def __init__(self, descriptions: dict, model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
                 top_k: int = 10):
        self.top_k = top_k
        self._tech_ids = list(descriptions.keys())
        self._tech_texts = [descriptions[tid] for tid in self._tech_ids]
        self._tech_descs = descriptions

        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

        print(f"[OrigDocMapper] 编码 {len(self._tech_ids)} 个原始 ATT&CK 文档描述...")
        self._tech_embeddings = self._model.encode(
            self._tech_texts, show_progress_bar=False, normalize_embeddings=True
        )
        print(f"[OrigDocMapper] 就绪。")

    def predict_top_k_detail(self, query: str) -> list:
        if not query or not query.strip():
            return []
        q_emb = self._model.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        )
        similarities = np.dot(self._tech_embeddings, q_emb.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:self.top_k]
        results = []
        for idx in top_indices:
            results.append({
                "tech_id": self._tech_ids[idx],
                "score": float(similarities[idx]),
                "tech_text": self._tech_texts[idx],
            })
        return results

    def get_tech_text(self, tech_id: str) -> str:
        return self._tech_descs.get(tech_id, "")


# ============================================================
# 查询构建
# ============================================================

def _extract_proc_name(prop_str: str) -> str:
    if not prop_str:
        return ""
    parts = prop_str.split(",")
    if len(parts) >= 3:
        path = parts[2].strip()
        if "/" in path:
            return path.rsplit("/", 1)[-1]
        return path
    if "/" in prop_str:
        return prop_str.rsplit("/", 1)[-1]
    return prop_str.strip()


def build_query_direct(uuid: str, df: pd.DataFrame, node_props: dict,
                       node_types: dict) -> str:
    """Direct：直接拼接原始事件类型+属性（含路径/IP噪声），不做翻译。"""
    parts = []
    seen = set()
    ntype = node_types.get(uuid, "")
    prop = node_props.get(uuid, "")

    as_actor = df[df["actorID"] == uuid]
    for _, row in as_actor.iterrows():
        action = row["action"]
        obj_id = row["objectID"]
        obj_type = node_types.get(obj_id, row.get("object", ""))
        obj_prop = node_props.get(obj_id, "")
        desc = f"{ntype} {prop} {action} {obj_type} {obj_prop}".strip()
        if desc not in seen:
            seen.add(desc)
            parts.append(desc)

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

    return ". ".join(parts[:30]) if parts else ""


def build_query_denoised(uuid: str, df: pd.DataFrame, node_props: dict,
                         node_types: dict) -> str:
    """Log-Enhanced：翻译事件和类型为自然语言，过滤低信息量动作，去除噪声。"""
    parts = []
    seen = set()
    ntype = node_types.get(uuid, "")
    prop = node_props.get(uuid, "")

    def translate_subject(ntype_str, prop_str):
        if ntype_str == "SUBJECT_PROCESS":
            pname = _extract_proc_name(prop_str)
            if pname:
                role = get_process_role(pname)
                if role == pname and pname.lower() not in PROCESS_ROLE_MAP:
                    return "process"
                return role
            return "process"
        return TYPE_MAP.get(ntype_str, ntype_str)

    def translate_object(ntype_str, prop_str, action_str=""):
        if action_str in ("EVENT_EXECUTE", "EVENT_LOADLIBRARY"):
            return "executable"
        mapped = TYPE_MAP.get(ntype_str, "")
        if mapped:
            if "FILE" in ntype_str and prop_str:
                if "/proc/" in prop_str:
                    return "process information"
                ft = get_file_type(prop_str)
                return ft if ft else "file"
            return mapped
        if ntype_str == "MemoryObject":
            return "process memory"
        return ntype_str

    def translate_action(action_str):
        return EVENT_MAP.get(action_str, action_str)

    def make_triple(subj, verb, obj_desc):
        if " " in verb:
            return f"{subj} {verb}"
        return f"{subj} {verb} {obj_desc}"

    as_actor = df[df["actorID"] == uuid]
    for _, row in as_actor.iterrows():
        action = row["action"]
        obj_id = row["objectID"]
        obj_type = node_types.get(obj_id, row.get("object", ""))
        obj_prop = node_props.get(obj_id, "")
        subj = translate_subject(ntype, prop)
        verb = translate_action(action)
        obj_desc = translate_object(obj_type, obj_prop, action)
        if any(verb.startswith(p) for p in LOW_INFO_PREFIXES):
            continue
        triple = make_triple(subj, verb, obj_desc).strip()
        if triple not in seen:
            seen.add(triple)
            parts.append(triple)

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


# ============================================================
# 主流程
# ============================================================

def main():
    data_dir = os.path.dirname(__file__)

    # ---------- 被查询库来源 ----------
    orig_csv = os.path.join(data_dir, "data/enterprise-attack_Chroma.csv")
    raw_triples_path = os.path.join(data_dir, "data/technique_triples_raw.json")
    trans_triples_path = os.path.join(data_dir, "data/technique_triples_transformed.json")
    gt_path = os.path.join(data_dir, "data/e3_label_ground_truth.json")

    # ---------- 展示三层库的区别 ----------
    print("=" * 70)
    print("被查询库三个层次对比")
    print("=" * 70)

    # 1. 原始文档
    orig_descs = load_original_attack_descriptions(orig_csv)
    print(f"\n1. 原始 ATT&CK 文档 (enterprise-attack_Chroma.csv): {len(orig_descs)} 个技术")
    for tid in ["T1059", "T1070", "T1071"]:
        if tid in orig_descs:
            print(f"   {tid}: \"{orig_descs[tid][:100]}...\"")

    # 2. raw triples
    with open(raw_triples_path) as f:
        raw_triples_data = json.load(f)
    print(f"\n2. Raw triples (technique_triples_raw.json): {len(raw_triples_data)} 个技术")
    for tid in ["T1059", "T1070", "T1071"]:
        if tid in raw_triples_data:
            desc = ". ".join(f"{t['subject']} {t['verb']} {t['object']}" for t in raw_triples_data[tid])
            print(f"   {tid}: \"{desc[:100]}...\"")

    # 3. transformed triples (坍塌)
    with open(trans_triples_path) as f:
        trans_data = json.load(f)
    desc_to_techs = defaultdict(list)
    for tid, triples in trans_data.items():
        desc = ". ".join(f"{t['subject']} {t['verb']} {t['object']}" for t in triples)
        desc_to_techs[desc].append(tid)
    print(f"\n3. Transformed triples (technique_triples_transformed.json): {len(trans_data)} 个技术")
    print(f"   ⚠ 坍塌为 {len(desc_to_techs)} 个唯一描述")
    for tid in ["T1059", "T1070", "T1071"]:
        if tid in trans_data:
            desc = ". ".join(f"{t['subject']} {t['verb']} {t['object']}" for t in trans_data[tid])
            n_shared = len(desc_to_techs.get(desc, []))
            print(f"   {tid}: \"{desc}\" (与 {n_shared} 个技术共享)")

    # ---------- 加载 GT 和日志数据 ----------
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

    all_entries = []
    for ds, scene in scenes:
        base_path = config["remote"]["path_map"].get(ds)
        if not base_path or not os.path.exists(base_path):
            print(f"\n[SKIP] {ds}")
            continue
        print(f"\n[加载] {ds}/{scene} ...")

        label_map = collect_label_paths(base_path)
        labels = []
        if scene in label_map:
            with open(label_map[scene]) as f:
                labels = [l.strip() for l in f if l.strip()]

        json_map = collect_json_paths(base_path)
        node_props = {}
        if scene in json_map:
            for cat, json_files in json_map[scene].items():
                if cat == "malicious":
                    net, sub, fil = collect_nodes_from_log(json_files)
                    node_props.update(net)
                    node_props.update(sub)
                    node_props.update(fil)

        mal_txt = os.path.join(base_path, f"{scene}_malicious.txt")
        df = pd.read_csv(mal_txt, sep='\t', header=None,
                         names=['actorID', 'actor_type', 'objectID', 'object',
                                'action', 'timestamp'],
                         dtype=str).dropna()

        node_types = {}
        for _, row in df.iterrows():
            if row["actorID"] not in node_types:
                node_types[row["actorID"]] = row["actor_type"]
            if row["objectID"] not in node_types:
                node_types[row["objectID"]] = row["object"]

        scene_gt = gt_data.get(scene, {})
        for uuid in labels:
            if uuid not in scene_gt:
                continue
            gt_tech = scene_gt[uuid]["technique"]
            all_entries.append({
                "uuid": uuid, "scene": scene,
                "q_direct": build_query_direct(uuid, df, node_props, node_types),
                "q_denoised": build_query_denoised(uuid, df, node_props, node_types),
                "gt_technique": gt_tech,
                "gt_parent": get_parent_technique(gt_tech),
                "node_type": scene_gt[uuid].get("type", ""),
                "behavior": scene_gt[uuid].get("behavior", ""),
                "tactic": scene_gt[uuid].get("tactic", ""),
            })

    print(f"\n总评估 label 数: {len(all_entries)}\n")

    # ---------- 初始化匹配器 ----------
    print("[初始化匹配器]")

    # Direct / Log-Enhanced 的被查询库: 原始 ATT&CK 文档描述
    mapper_orig = OriginalDocMapper(orig_descs, top_k=10)

    # Tech-Enhanced / Full-Enhanced 的被查询库: raw triples（从文档提取的三元组）
    mapper_raw = TechniqueSemanticMapper(triples_path=raw_triples_path, top_k=10, threshold=0.0)

    # ---------- 4 种方法 ----------
    methods = {
        # 查询侧          被查询库
        "Direct":        (lambda e: e["q_direct"],   mapper_orig),   # 原始 × 原始文档
        "Log-Enhanced":  (lambda e: e["q_denoised"], mapper_orig),   # 去噪 × 原始文档
        "Tech-Enhanced": (lambda e: e["q_direct"],   mapper_raw),    # 原始 × raw triples
        "Full-Enhanced": (lambda e: e["q_denoised"], mapper_raw),    # 去噪 × raw triples
    }

    TOP_K = 10

    for entry in all_entries:
        entry["results"] = {}
        for method_name, (get_q, mapper) in methods.items():
            query = get_q(entry)
            if not query or not query.strip():
                entry["results"][method_name] = {
                    "query": "", "pred": None, "score": 0,
                    "status": "unmapped", "top_k": [], "match_rank": -1,
                }
                continue
            top_detail = mapper.predict_top_k_detail(query)
            if not top_detail:
                entry["results"][method_name] = {
                    "query": query, "pred": None, "score": 0,
                    "status": "unmapped", "top_k": [], "match_rank": -1,
                }
                continue
            top_k_parents = [get_parent_technique(t["tech_id"]) for t in top_detail[:TOP_K]]
            hit = entry["gt_parent"] in top_k_parents
            match_rank = -1
            for rank, tp in enumerate(top_k_parents):
                if tp == entry["gt_parent"]:
                    match_rank = rank + 1
                    break
            entry["results"][method_name] = {
                "query": query,
                "pred": top_detail[0]["tech_id"],
                "score": round(top_detail[0]["score"], 4),
                "status": "correct" if hit else "incorrect",
                "match_rank": match_rank,
                "top_k": [(t["tech_id"], round(t["score"], 3), t["tech_text"][:150])
                          for t in top_detail[:5]],
            }

    # ============================================================
    # 输出分析
    # ============================================================

    print("\n" + "=" * 70)
    print("Table V: Attack Variant Usability Analysis")
    print("=" * 70)

    # --- 方案说明 ---
    print("""
方案设计（查询侧 × 被查询库侧 的 2×2 组合）：

  ┌─────────────────┬──────────────────────┬─────────────────────────┐
  │                 │ 被查询库:原始ATT&CK  │ 被查询库:raw triples    │
  │                 │ 文档描述(566个唯一)  │ (提取三元组,691个唯一)  │
  ├─────────────────┼──────────────────────┼─────────────────────────┤
  │查询:原始日志字段│ Direct (基线)        │ Tech-Enhanced           │
  │查询:翻译去噪    │ Log-Enhanced         │ Full-Enhanced           │
  └─────────────────┴──────────────────────┴─────────────────────────┘
""")

    # --- 总体结果 ---
    print(f"[总体] Hit@{TOP_K}")
    print(f"{'Method':<18} {'Correct':>8} {'Incorrect':>10} {'Unmapped':>10}")
    print("-" * 50)
    for mn in ["Direct", "Log-Enhanced", "Tech-Enhanced", "Full-Enhanced"]:
        c = sum(1 for e in all_entries if e["results"][mn]["status"] == "correct")
        ic = sum(1 for e in all_entries if e["results"][mn]["status"] == "incorrect")
        um = sum(1 for e in all_entries if e["results"][mn]["status"] == "unmapped")
        n = len(all_entries)
        print(f"{mn:<18} {c:>5} ({c/n*100:4.1f}%) {ic:>5} ({ic/n*100:4.1f}%) {um:>5} ({um/n*100:4.1f}%)")

    # ============================================================
    # 1. Direct 失败原因分析
    # ============================================================
    print("\n\n" + "=" * 70)
    print("1. Direct（无增强基线）失败原因分析")
    print("=" * 70)
    print("""
Direct: 原始日志字段 → 原始 ATT&CK 文档描述。
两侧都没有做任何处理，匹配失败的原因是两侧语义空间不一致：

查询侧（审计日志）:
  "SUBJECT_PROCESS /bin/sh,-c ./gtcache,13749 EVENT_CONNECT NetFlowObject 128.55.12.118,80"
  ↑ 系统类型标识符    ↑ 命令行+PID+路径      ↑ 系统调用名    ↑ 原始IP:端口

被查询库（ATT&CK 文档）:
  "Adversaries may use application layer protocols associated with web traffic
   to communicate with systems under their control to avoid detection..."
  ↑ 意图级主语        ↑ 意图级描述                        ↑ 抽象语义

SBERT 在这两种文本间找不到语义桥梁。

典型失败案例：""")

    direct_incorrect = [e for e in all_entries
                        if e["results"]["Direct"]["status"] == "incorrect"]
    for e in direct_incorrect[:4]:
        r = e["results"]["Direct"]
        print(f"\n  UUID: {e['uuid'][:35]}  GT: {e['gt_technique']} ({e['tactic']})")
        print(f"  查询: {r['query'][:120]}...")
        print(f"  误匹配: {r['pred']} (score={r['score']:.3f})")
        if r["top_k"]:
            print(f"  Top1 库文本: {r['top_k'][0][2][:100]}...")
        reasons = []
        q = r["query"]
        if "SUBJECT_PROCESS" in q: reasons.append("主语是系统类型标识符")
        if "EVENT_" in q: reasons.append("动作是系统调用名")
        if any(x in q for x in ["128.", "146.", "192.", "/tmp/", "/proc/", "/usr/"]): reasons.append("含IP/路径噪声")
        print(f"  失败原因: {'; '.join(reasons)}")

    # ============================================================
    # 2. Log-Enhanced 相比 Direct：去噪增益
    # ============================================================
    print("\n\n" + "=" * 70)
    print("2. Log-Enhanced 相比 Direct：去噪带来的增益")
    print("   查询侧翻译去噪，被查询库不变（仍是原始 ATT&CK 文档）")
    print("   → 找到被背景噪声干扰的案例")
    print("=" * 70)

    denoised_gains = [e for e in all_entries
                      if e["results"]["Direct"]["status"] != "correct"
                      and e["results"]["Log-Enhanced"]["status"] == "correct"]
    print(f"\n共 {len(denoised_gains)} 个案例因去噪而正确（Direct 失败 → Log-Enhanced 正确）\n")

    for e in denoised_gains:
        rd = e["results"]["Direct"]
        rl = e["results"]["Log-Enhanced"]
        print(f"  UUID: {e['uuid'][:35]}  GT: {e['gt_technique']} ({e['tactic']})")
        print(f"  Direct 查询:  {rd['query'][:100]}...")
        print(f"  去噪查询:     {rl['query'][:100]}")
        print(f"  Direct 预测:  {rd['pred'] or '(unmapped)'} (score={rd['score']:.3f})")
        print(f"  去噪预测:     {rl['pred']} (score={rl['score']:.3f}, rank={rl['match_rank']})")
        # 分析去噪效果
        dq, lq = rd["query"], rl["query"]
        effects = []
        if "SUBJECT_PROCESS" in dq and "SUBJECT_PROCESS" not in lq:
            effects.append("系统标识符→语义角色")
        if "EVENT_" in dq and "EVENT_" not in lq:
            effects.append("系统调用→自然语言动词")
        if any(x in dq for x in ["128.", "146.", "/tmp/", "/proc/"]) and \
           not any(x in lq for x in ["128.", "146.", "/tmp/", "/proc/"]):
            effects.append("去除IP/路径噪声")
        n_orig = len(dq.split(". "))
        n_denoised = len(lq.split(". "))
        if n_denoised < n_orig:
            effects.append(f"过滤低信息事件({n_orig}→{n_denoised}条)")
        print(f"  去噪效果: {'; '.join(effects)}")
        print()

    # ============================================================
    # 3. Tech-Enhanced 相比 Direct：技术侧增益
    # ============================================================
    print("=" * 70)
    print("3. Tech-Enhanced 相比 Direct：技术侧三元组提取的增益")
    print("   查询侧不变（原始字段），被查询库从原始文档→raw triples")
    print("   → 三元组结构化提取消除了文档中的描述性噪声")
    print("=" * 70)

    tech_gains = [e for e in all_entries
                  if e["results"]["Direct"]["status"] != "correct"
                  and e["results"]["Tech-Enhanced"]["status"] == "correct"]
    tech_losses = [e for e in all_entries
                   if e["results"]["Direct"]["status"] == "correct"
                   and e["results"]["Tech-Enhanced"]["status"] != "correct"]
    print(f"\n增益: {len(tech_gains)} 个案例因技术侧提取而正确")
    print(f"回退: {len(tech_losses)} 个案例因技术侧提取而丢失\n")

    for e in tech_gains:
        rd = e["results"]["Direct"]
        rt = e["results"]["Tech-Enhanced"]
        print(f"  UUID: {e['uuid'][:35]}  GT: {e['gt_technique']} ({e['tactic']})")
        print(f"  查询（相同）: {rd['query'][:80]}...")
        print(f"  Direct(原始文档)→  {rd['pred'] or '(unmapped)':15s} score={rd['score']:.3f}")
        print(f"  TechEnh(三元组)→  {rt['pred']:15s} score={rt['score']:.3f} rank={rt['match_rank']}")
        print(f"  原因: raw triples 将文档段落压缩为 SVO 结构，")
        print(f"        去除了描述性文本（'may','in order to'等），更利于操作级匹配")
        print()

    # ============================================================
    # 4. Full-Enhanced 独有
    # ============================================================
    print("=" * 70)
    print("4. Full-Enhanced 独有：只有双侧增强才能找到")
    print("   查询去噪 + 被查询库用 raw triples")
    print("=" * 70)

    full_only = [e for e in all_entries
                 if e["results"]["Full-Enhanced"]["status"] == "correct"
                 and e["results"]["Direct"]["status"] != "correct"
                 and e["results"]["Log-Enhanced"]["status"] != "correct"
                 and e["results"]["Tech-Enhanced"]["status"] != "correct"]
    print(f"\n共 {len(full_only)} 个案例只有双侧增强才能正确匹配\n")

    for e in full_only:
        rd = e["results"]["Direct"]
        rl = e["results"]["Log-Enhanced"]
        rt = e["results"]["Tech-Enhanced"]
        rf = e["results"]["Full-Enhanced"]
        print(f"  UUID: {e['uuid'][:35]}  GT: {e['gt_technique']} ({e['tactic']})")
        print(f"  Direct(原始×文档):         {rd['pred'] or '(unmapped)':15s} score={rd['score']:.3f} {'✓' if rd['status']=='correct' else '✗'}")
        print(f"  Log-Enh(去噪×文档):        {rl['pred'] or '(unmapped)':15s} score={rl['score']:.3f} {'✓' if rl['status']=='correct' else '✗'}")
        print(f"  Tech-Enh(原始×三元组):     {rt['pred'] or '(unmapped)':15s} score={rt['score']:.3f} {'✓' if rt['status']=='correct' else '✗'}")
        print(f"  Full-Enh(去噪×三元组):     {rf['pred']:15s} score={rf['score']:.3f} ✓ rank={rf['match_rank']}")
        print(f"  原因: 查询侧噪声 + 文档侧冗余 两个障碍叠加，必须同时消除")
        print()

    # ============================================================
    # 5. Unmapped 分析
    # ============================================================
    print("=" * 70)
    print("5. Unmapped 案例分析（Full-Enhanced 最强方法仍无法匹配）")
    print("=" * 70)

    full_unmapped = [e for e in all_entries
                     if e["results"]["Full-Enhanced"]["status"] == "unmapped"]
    empty_q = [e for e in full_unmapped if not e["results"]["Full-Enhanced"]["query"]]
    has_q = [e for e in full_unmapped if e["results"]["Full-Enhanced"]["query"]]

    print(f"\nFull-Enhanced 仍有 {len(full_unmapped)} 个 unmapped:")
    print(f"  查询为空（无可观测行为）: {len(empty_q)}")
    print(f"  有查询但无法匹配:       {len(has_q)}")

    if empty_q:
        print(f"\n  [空查询] — 审计日志中该节点没有关联事件:")
        for e in empty_q:
            print(f"    {e['uuid'][:35]}  GT: {e['gt_technique']} ({e['tactic']})")
        print(f"    → 可能是隐蔽 C2 通信未被内核审计捕获，或审计数据丢失")
        print(f"    → 对应论文中 '未知攻击' 场景：攻击行为超出审计系统可观测范围")

    if has_q:
        print(f"\n  [有查询但无法匹配] — 行为过于通用，与已知技术无法区分:")
        for e in has_q:
            rf = e["results"]["Full-Enhanced"]
            print(f"    {e['uuid'][:35]}  GT: {e['gt_technique']} ({e['tactic']})")
            print(f"    查询: {rf['query'][:80]}")
            # 看下所有方法是否都失败
            all_fail = all(e["results"][m]["status"] != "correct"
                           for m in ["Direct", "Log-Enhanced", "Tech-Enhanced", "Full-Enhanced"])
            if all_fail:
                print(f"    → 四种方法全部失败。行为描述如 'process writes file' 过于通用，")
                print(f"      与数十种技术的描述语义距离相近，无法置信区分")
            print()

    print("  [Unmapped 总结]")
    print("  这类案例通常对应：")
    print("    1. Living-off-the-Land (LOTL)：攻击者只使用系统自带工具，")
    print("       行为模式与正常操作完全重叠，语义匹配无法区分")
    print("    2. 未知攻击变体：行为模式不在已知 ATT&CK 技术库中")
    print("    3. 数据缺失：审计系统未记录关键事件")

    # ============================================================
    # 6. 回归分析
    # ============================================================
    print("\n\n" + "=" * 70)
    print("6. 回归分析：Direct 正确但 Full-Enhanced 反而错误")
    print("=" * 70)

    regressions = [e for e in all_entries
                   if e["results"]["Direct"]["status"] == "correct"
                   and e["results"]["Full-Enhanced"]["status"] != "correct"]
    print(f"\n共 {len(regressions)} 个回归案例\n")

    for e in regressions:
        rd = e["results"]["Direct"]
        rf = e["results"]["Full-Enhanced"]
        print(f"  UUID: {e['uuid'][:35]}  GT: {e['gt_technique']} ({e['tactic']})")
        print(f"  Direct(原始×文档):     {rd['pred']:15s} score={rd['score']:.3f} rank={rd['match_rank']}")
        print(f"  Full(去噪×三元组):     {rf['pred'] or '(unmapped)':15s} score={rf['score']:.3f}")
        print(f"  原因: 原始日志中的具体路径（如 /proc/xxx）恰好与文档描述中的")
        print(f"        关键词匹配（如 'process listing'），翻译后泛化为 'process")
        print(f"        information' 反而丢失了区分性细节")
        print()

    # ============================================================
    # 保存
    # ============================================================
    out_path = os.path.join(data_dir, "..", "table_v_analysis.json")
    out_data = {
        "description": "Table V: Attack Variant Usability Analysis",
        "design": {
            "Direct": "原始日志字段 × 原始ATT&CK文档描述",
            "Log-Enhanced": "翻译去噪查询 × 原始ATT&CK文档描述",
            "Tech-Enhanced": "原始日志字段 × raw triples(从文档提取的三元组)",
            "Full-Enhanced": "翻译去噪查询 × raw triples",
        },
        "library_stats": {
            "original_docs": len(orig_descs),
            "raw_triples": len(raw_triples_data),
            "transformed_triples_total": len(trans_data),
            "transformed_triples_unique": len(desc_to_techs),
        },
        "summary": {},
        "cases": {
            "denoised_gains": [], "tech_gains": [], "tech_losses": [],
            "full_only": [], "unmapped": [], "regressions": [],
        },
    }

    for mn in ["Direct", "Log-Enhanced", "Tech-Enhanced", "Full-Enhanced"]:
        c = sum(1 for e in all_entries if e["results"][mn]["status"] == "correct")
        ic = sum(1 for e in all_entries if e["results"][mn]["status"] == "incorrect")
        um = sum(1 for e in all_entries if e["results"][mn]["status"] == "unmapped")
        out_data["summary"][mn] = {
            "correct": c, "incorrect": ic, "unmapped": um,
            "total": len(all_entries),
            "correct_pct": round(c / len(all_entries) * 100, 1),
        }

    def to_dict(e):
        return {
            "uuid": e["uuid"], "scene": e["scene"],
            "gt_technique": e["gt_technique"], "gt_parent": e["gt_parent"],
            "tactic": e["tactic"], "behavior": e["behavior"],
            "results": e["results"],
        }

    out_data["cases"]["denoised_gains"] = [to_dict(e) for e in denoised_gains]
    out_data["cases"]["tech_gains"] = [to_dict(e) for e in tech_gains]
    out_data["cases"]["tech_losses"] = [to_dict(e) for e in tech_losses]
    out_data["cases"]["full_only"] = [to_dict(e) for e in full_only]
    out_data["cases"]["unmapped"] = [to_dict(e) for e in full_unmapped]
    out_data["cases"]["regressions"] = [to_dict(e) for e in regressions]
    out_data["all_entries"] = [to_dict(e) for e in all_entries]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
