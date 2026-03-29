"""
从 MITRE ATT&CK 官方 STIX 数据生成向量库 CSV。
用翻译规则将技术描述转为系统事件自然语言。

远程运行：
    conda activate prographer && cd /home/nsas2020/fuzz/prographer && python -m process.build_attack_csv

产出：process/data/attack_techniques.csv

翻译规则定义在 process/translation_rules.py 中。
"""
import csv
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import requests
except ImportError:
    print("需要 requests 库: pip install requests")
    sys.exit(1)

from process.translation_rules import generate_system_behaviors

STIX_URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "attack_techniques.csv")

def main():
    # 1) 下载 STIX 数据
    print("正在下载 ATT&CK STIX 数据...")
    resp = requests.get(STIX_URL, timeout=120)
    resp.raise_for_status()
    stix_data = resp.json()
    print(f"下载完成，共 {len(stix_data.get('objects', []))} 个 STIX 对象")

    # 2) 提取 attack-pattern（技术），先收集所有技术信息
    tech_info = {}  # tech_id -> {name, description, stix_id, is_sub}
    stix_id_to_tech_id = {}  # STIX id -> tech_id
    for obj in stix_data.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue

        name = obj.get("name", "")
        description = obj.get("description", "")
        description = re.sub(r"\(Citation:[^)]*\)", "", description).strip()

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

    # 3) 建立子技术 → 父技术的映射
    sub_to_parent = {}  # 子技术 tech_id -> 父技术 tech_id
    for obj in stix_data.get("objects", []):
        if obj.get("type") == "relationship" and obj.get("relationship_type") == "subtechnique-of":
            src = stix_id_to_tech_id.get(obj.get("source_ref", ""), "")
            tgt = stix_id_to_tech_id.get(obj.get("target_ref", ""), "")
            if src and tgt:
                sub_to_parent[src] = tgt

    print(f"子技术→父技术映射: {len(sub_to_parent)} 条")

    # 4) 生成系统事件描述，子技术继承父技术的描述
    rows = []
    enriched_count = 0
    for tech_id, info in tech_info.items():
        # 自己的系统事件描述
        sys_behaviors = generate_system_behaviors(info["description"])

        # 子技术继承父技术的系统事件描述
        if info["is_sub"] and tech_id in sub_to_parent:
            parent_id = sub_to_parent[tech_id]
            if parent_id in tech_info:
                parent_behaviors = generate_system_behaviors(tech_info[parent_id]["description"])
                if parent_behaviors:
                    if sys_behaviors:
                        # 合并去重
                        existing = set(sys_behaviors.split(". "))
                        for b in parent_behaviors.split(". "):
                            if b not in existing:
                                existing.add(b)
                                sys_behaviors += ". " + b
                    else:
                        sys_behaviors = parent_behaviors

        # 有系统事件描述的只用系统事件描述，没有的保留原始描述
        if sys_behaviors:
            body = f"{info['name']}. {sys_behaviors}"
            enriched_count += 1
        else:
            body = f"{info['name']}. {info['description']}"

        rows.append({
            "Subject": f"{tech_id}: {info['name']}",
            "filepath": info["url"],
            "Date": "",
            "Body": body,
            "Source": "MITRE-ATT&CK",
            "tech_id": tech_id,
            "is_subtechnique": info["is_sub"],
        })

    print(f"提取到 {len(rows)} 个有效技术（含子技术）")
    print(f"  有系统事件增强: {enriched_count}")
    print(f"  无系统事件增强: {len(rows) - enriched_count}")

    main_count = sum(1 for r in rows if not r["is_subtechnique"])
    sub_count = sum(1 for r in rows if r["is_subtechnique"])
    print(f"  主技术: {main_count}, 子技术: {sub_count}")

    # 3) 写 CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Subject", "filepath", "Date", "Body", "Source"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "Subject": row["Subject"],
                "filepath": row["filepath"],
                "Date": row["Date"],
                "Body": row["Body"],
                "Source": row["Source"],
            })

    print(f"\n已写入: {OUTPUT_FILE}")

    # 打印示例：有增强的
    print("\n=== 有系统事件增强的示例 ===")
    count = 0
    for r in rows:
        if "Observable system behaviors" in r["Body"]:
            print(f"\n{r['tech_id']}: {r['Subject']}")
            # 只打印追加的部分
            idx = r["Body"].index("Observable system behaviors")
            print(f"  {r['Body'][idx:idx+300]}")
            count += 1
            if count >= 5:
                break

    # 打印示例：无增强的
    print("\n=== 无系统事件增强的示例 ===")
    count = 0
    for r in rows:
        if "Observable system behaviors" not in r["Body"]:
            print(f"  {r['tech_id']}: {r['Subject']}")
            count += 1
            if count >= 5:
                break


if __name__ == "__main__":
    main()
