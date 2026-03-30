"""
从 MITRE ATT&CK 官方 STIX 数据生成向量库 CSV。
用 NLP 句法分析流水线将技术描述翻译为操作级三元组。

远程运行：
    conda activate prographer && cd /home/nsas2020/fuzz/prographer && python -m process.build_attack_csv

产出：process/data/attack_techniques.csv

翻译流水线（三步）：
1. 结构化解析：依赖句法分析提取 (主语, 谓语, 宾语) 三元组
2. 语义层转换：主谓宾分别查映射表，意图动词不在表中的三元组丢弃
3. 操作级实例化：一个意图动词展开为多个操作级三元组，拼接为最终描述
"""
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import spacy
except ImportError:
    print("需要 spacy 库: pip install spacy && python -m spacy download en_core_web_sm")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("需要 requests 库: pip install requests")
    sys.exit(1)

from process.translation_rules import (
    map_intent_subject,
    map_intent_verb,
    map_intent_object,
    INTENT_VERB_MAP,
)

STIX_URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "attack_techniques_nlp.csv")
LOCAL_STIX = os.path.join(OUTPUT_DIR, "enterprise-attack.json")


# ============================================================
# 步骤一：结构化解析 — 依赖句法分析提取三元组
# ============================================================

def extract_triples(doc) -> List[Tuple[str, str, str]]:
    """从 spaCy Doc 中提取 (主语, 谓语动词, 宾语) 三元组。"""
    triples = []
    for sent in doc.sents:
        for token in sent:
            # 找谓语动词（ROOT 或 非助动词的动词）
            if token.pos_ != "VERB" or token.dep_ == "aux":
                continue

            # 提取主语
            subjects = []
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subjects.append(_get_span_text(child))

            # 提取宾语
            objects = []
            for child in token.children:
                if child.dep_ in ("dobj", "pobj", "attr", "oprd"):
                    objects.append(_get_span_text(child))
                # 介词短语中的宾语 (e.g., "inject code into processes")
                if child.dep_ == "prep":
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            objects.append(_get_span_text(grandchild))

            verb = token.lemma_.lower()

            if not subjects:
                subjects = [""]
            if not objects:
                objects = [""]

            for subj in subjects:
                for obj in objects:
                    triples.append((subj, verb, obj))

    return triples


def _get_span_text(token) -> str:
    """获取 token 及其子树的文本（用于提取完整的名词短语）。"""
    # 获取名词短语的完整文本
    subtree = list(token.subtree)
    # 限制长度，避免提取过长的从句
    if len(subtree) > 8:
        # 只取核心名词及其直接修饰
        parts = [token.text]
        for child in token.children:
            if child.dep_ in ("compound", "amod", "det", "poss", "nummod"):
                parts.insert(0, child.text)
        return " ".join(parts)
    return " ".join(t.text for t in subtree)


# ============================================================
# 步骤二：语义层转换 — 主谓宾分别查映射表
# ============================================================

def transform_triple(subj: str, verb: str, obj: str) -> Optional[List[Tuple[str, str, str]]]:
    """对一个三元组执行语义层转换。

    返回展开后的操作级三元组列表，或 None（谓语不在意图动词表中，丢弃）。
    """
    # 谓语：查意图动词映射表
    op_verbs = map_intent_verb(verb)
    if op_verbs is None:
        return None

    # 主语：意图级 → 系统级
    sys_subj = map_intent_subject(subj)

    # 宾语：意图级 → 系统级
    sys_obj = map_intent_object(obj)

    return sys_subj, op_verbs, sys_obj


# ============================================================
# 步骤三：操作级实例化 — 展开为多个三元组
# ============================================================

def instantiate_triples(sys_subj: str, op_verbs: List[str], sys_obj: str) -> List[str]:
    """将一个转换后的结果展开为多个操作级三元组描述。"""
    results = []
    seen = set()
    for verb in op_verbs:
        desc = f"{sys_subj} {verb} {sys_obj}"
        if desc not in seen:
            seen.add(desc)
            results.append(desc)
    return results


# ============================================================
# 完整流水线
# ============================================================

def translate_technique(nlp, name: str, description: str) -> str:
    """对一个 ATT&CK 技术描述执行完整的三步翻译流水线。"""
    # 清理文本
    text = re.sub(r"\(Citation:[^)]*\)", "", description).strip()
    text = re.sub(r"</?code>", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # 步骤一：结构化解析
    doc = nlp(text)
    triples = extract_triples(doc)

    # 步骤二 + 三：语义层转换 + 操作级实例化
    all_descriptions = []
    seen = set()
    for subj, verb, obj in triples:
        result = transform_triple(subj, verb, obj)
        if result is None:
            continue
        sys_subj, op_verbs, sys_obj = result
        descs = instantiate_triples(sys_subj, op_verbs, sys_obj)
        for d in descs:
            if d not in seen:
                seen.add(d)
                all_descriptions.append(d)

    return ", ".join(all_descriptions)


# ============================================================
# 主流程
# ============================================================

def main():
    # 1) 加载 spaCy 模型
    print("加载 spaCy 模型...")
    nlp = spacy.load("en_core_web_sm")

    # 2) 加载 STIX 数据（优先本地）
    if os.path.exists(LOCAL_STIX):
        print(f"从本地加载 STIX 数据: {LOCAL_STIX}")
        with open(LOCAL_STIX, "r", encoding="utf-8") as f:
            stix_data = json.load(f)
    else:
        print("正在下载 ATT&CK STIX 数据...")
        resp = requests.get(STIX_URL, timeout=120)
        resp.raise_for_status()
        stix_data = resp.json()
        # 保存到本地
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(LOCAL_STIX, "w", encoding="utf-8") as f:
            json.dump(stix_data, f)

    print(f"共 {len(stix_data.get('objects', []))} 个 STIX 对象")

    # 3) 提取所有技术信息
    tech_info = {}
    stix_id_to_tech_id = {}
    for obj in stix_data.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue

        name = obj.get("name", "")
        description = obj.get("description", "")

        tech_id = ""
        url = ""
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                tech_id = ref.get("external_id", "")
                url = ref.get("url", "")
                break
        if not tech_id:
            continue

        is_sub = obj.get("x_mitre_is_subtechnique", False)
        stix_id = obj.get("id", "")
        stix_id_to_tech_id[stix_id] = tech_id
        tech_info[tech_id] = {
            "name": name, "description": description, "url": url,
            "is_sub": is_sub, "stix_id": stix_id,
        }

    # 4) 建立子技术 → 父技术映射
    sub_to_parent = {}
    for obj in stix_data.get("objects", []):
        if obj.get("type") == "relationship" and obj.get("relationship_type") == "subtechnique-of":
            src = stix_id_to_tech_id.get(obj.get("source_ref", ""), "")
            tgt = stix_id_to_tech_id.get(obj.get("target_ref", ""), "")
            if src and tgt:
                sub_to_parent[src] = tgt

    print(f"总技术数: {len(tech_info)}, 子技术→父技术映射: {len(sub_to_parent)} 条")

    # 5) 翻译每个技术
    total = len(tech_info)
    translated_cache = {}
    success_count = 0
    empty_count = 0

    for i, (tech_id, info) in enumerate(tech_info.items()):
        result = translate_technique(nlp, info["name"], info["description"])
        translated_cache[tech_id] = result

        if result:
            success_count += 1
        else:
            empty_count += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{total}] 已翻译，成功={success_count}，空={empty_count}")

    print(f"\n翻译完成: 成功={success_count}, 空={empty_count}")

    # 6) 生成 CSV，子技术合并父技术
    rows = []
    enriched_count = 0
    for tech_id, info in tech_info.items():
        sys_behaviors = translated_cache.get(tech_id, "")

        # 子技术合并父技术
        if info["is_sub"] and tech_id in sub_to_parent:
            parent_id = sub_to_parent[tech_id]
            parent_behaviors = translated_cache.get(parent_id, "")
            if parent_behaviors:
                if sys_behaviors:
                    existing = set(b.strip() for b in sys_behaviors.split(","))
                    for b in parent_behaviors.split(","):
                        b = b.strip()
                        if b and b not in existing:
                            existing.add(b)
                            sys_behaviors += ", " + b
                else:
                    sys_behaviors = parent_behaviors

        if sys_behaviors:
            body = f"{info['name']}. {sys_behaviors}"
            enriched_count += 1
        else:
            body = f"{info['name']}. {info['description'][:500]}"

        rows.append({
            "Subject": f"{tech_id}: {info['name']}",
            "filepath": info["url"],
            "Date": "",
            "Body": body,
            "Source": "MITRE-ATT&CK",
        })

    print(f"  有操作级描述: {enriched_count}")
    print(f"  保留原始描述: {len(rows) - enriched_count}")

    # 7) 写 CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Subject", "filepath", "Date", "Body", "Source"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\n已写入: {OUTPUT_FILE}")

    # 8) 保存翻译缓存
    cache_file = os.path.join(OUTPUT_DIR, "translation_cache.json")
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(translated_cache, f, ensure_ascii=False, indent=2)

    # 打印示例
    print("\n=== 翻译示例 ===")
    for tech_id in ["T1055", "T1055.009", "T1070.001", "T1059", "T1003"]:
        if tech_id in translated_cache and translated_cache[tech_id]:
            print(f"\n{tech_id}: {tech_info.get(tech_id, {}).get('name', '')}")
            print(f"  → {translated_cache[tech_id][:200]}")


if __name__ == "__main__":
    main()
