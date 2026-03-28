"""
从 MITRE ATT&CK 官方 STIX 数据生成向量库 CSV。

远程运行：
    conda activate prographer && cd /home/nsas2020/fuzz/prographer && python -m process.build_attack_csv

产出：process/data/attack_techniques.csv
"""
import json
import os
import re
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("需要 requests 库: pip install requests")
    sys.exit(1)

STIX_URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "attack_techniques.csv")


def main():
    # 1) 下载 STIX 数据
    print(f"正在下载 ATT&CK STIX 数据...")
    resp = requests.get(STIX_URL, timeout=120)
    resp.raise_for_status()
    stix_data = resp.json()
    print(f"下载完成，共 {len(stix_data.get('objects', []))} 个 STIX 对象")

    # 2) 提取 attack-pattern（技术）
    rows = []
    for obj in stix_data.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        # 跳过已废弃的技术
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue

        name = obj.get("name", "")
        description = obj.get("description", "")
        # 清理 description 中的 markdown 引用标记
        description = re.sub(r"\(Citation:[^)]*\)", "", description).strip()

        # 提取技术 ID 和 URL
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

        # Body = 技术描述（用于建向量库）
        # Subject = 技术名称 + ID（用于展示）
        rows.append({
            "Subject": f"{tech_id}: {name}",
            "filepath": url,
            "Date": "",
            "Body": f"{name}. {description}",
            "Source": "MITRE-ATT&CK",
            "tech_id": tech_id,
            "is_subtechnique": is_sub,
        })

    print(f"提取到 {len(rows)} 个有效技术（含子技术）")

    # 统计
    main_count = sum(1 for r in rows if not r["is_subtechnique"])
    sub_count = sum(1 for r in rows if r["is_subtechnique"])
    print(f"  主技术: {main_count}, 子技术: {sub_count}")

    # 3) 写 CSV（保持与原 CSV 相同的列结构：Subject, filepath, Date, Body, Source）
    import csv
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

    print(f"已写入: {OUTPUT_FILE}")

    # 打印几个示例
    print("\n示例：")
    for r in rows[:3]:
        print(f"  {r['tech_id']}: {r['Subject']}")
        print(f"    Body (前200字): {r['Body'][:200]}")
        print()


if __name__ == "__main__":
    main()
