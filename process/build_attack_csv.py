"""
从 MITRE ATT&CK 官方 STIX 数据生成向量库 CSV。
用 LLM 按翻译流水线将技术描述转为系统事件自然语言。

远程运行：
    conda activate prographer && cd /home/nsas2020/fuzz/prographer && python -m process.build_attack_csv

产出：process/data/attack_techniques.csv

翻译流水线：
1. 逐句扫描原始描述
2. 跳过目的性短语（in order to, to evade 等）
3. 将攻击动作和对象翻译为系统操作
4. 用查询侧词汇表（translation_rules.py）中的标准词汇表达
5. 去重，按原文顺序拼接
"""
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import requests
except ImportError:
    print("需要 requests 库: pip install requests")
    sys.exit(1)

STIX_URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "attack_techniques.csv")

# ChatAnywhere API 配置
API_KEY = "sk-doxpoyNwE1kfZtZeYZEwqwd0MyHxYsr5pP8OG3NYcepsbQdM"
API_ENDPOINT = "https://api.chatanywhere.org/v1/chat/completions"
MODEL = "gpt-3.5-turbo"

# LLM 翻译 prompt
SYSTEM_PROMPT = """You are a cybersecurity expert. Your task is to translate ATT&CK technique descriptions into system-level observable behaviors.

Follow this pipeline:
1. Scan the description sentence by sentence
2. Skip purpose phrases (e.g., "in order to evade", "to elevate privileges", "to avoid detection")
3. For each attack action or object, translate it into concrete system operations
4. You MUST use ONLY the following vocabulary:

Actions: writes, reads, opens, executes, creates child process, sends network data, receives network data, modifies process, deletes, maps memory, connects to network, modifies file attributes, binds to port, accepts connection, changes principal, creates object, sends message, receives message, renames

Objects: shared library (.so .dll), temporary directory (/tmp), configuration file, credential file (/etc/passwd /etc/shadow), log file, process filesystem (/proc), certificate file, database file, archive file, executable, shell script, key file, registry key

5. Remove duplicates, keep original order
6. Output ONLY the translated system behaviors, no explanation, no original text

Examples:

Input: "Adversaries may inject malicious code into processes via the /proc filesystem in order to evade process-based defenses as well as possibly elevate privileges. Proc memory injection is a method of executing arbitrary code in the address space of a separate live process."
Output: process writes shared library (.so .dll) to disk in temporary directory, process modifies another process memory, process reads and writes /proc filesystem, process creates or opens temporary files

Input: "Adversaries may clear Windows Event Logs to hide the activity of an intrusion."
Output: process deletes log file, process writes to clear log entries

Input: "Adversaries may execute malicious payloads via loading shared modules. Shared modules are executable files that are loaded into processes to provide access to reusable code."
Output: process loads shared library (.so .dll), process writes shared module to disk, process executes shared library

Input: "Adversaries may attempt to dump credentials to obtain account login and credential material from operating system and software. Credentials can be obtained from OS caches, memory, or structures."
Output: process reads credential file (/etc/passwd /etc/shadow), process reads process memory, process accesses credential storage"""

USER_PROMPT_TEMPLATE = """Translate the following ATT&CK technique into system-level observable behaviors.

Technique: {name}
Description: {description}

Output:"""


def llm_translate(name: str, description: str, retries: int = 3) -> str:
    """调用 GPT-3.5 翻译单个技术描述。"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": MODEL,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                name=name, description=description[:2000]
            )},
        ],
    }

    for attempt in range(retries):
        try:
            resp = requests.post(
                API_ENDPOINT,
                headers=headers,
                data=json.dumps(body),
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content.strip()
        except Exception as ex:
            print(f"    [重试 {attempt+1}/{retries}] {ex}")
            time.sleep(2 * (attempt + 1))

    return ""


def main():
    # 1) 下载 STIX 数据
    print("正在下载 ATT&CK STIX 数据...")
    resp = requests.get(STIX_URL, timeout=120)
    resp.raise_for_status()
    stix_data = resp.json()
    print(f"下载完成，共 {len(stix_data.get('objects', []))} 个 STIX 对象")

    # 2) 提取所有技术信息
    tech_info = {}
    stix_id_to_tech_id = {}
    for obj in stix_data.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue

        name = obj.get("name", "")
        description = obj.get("description", "")
        description = re.sub(r"\(Citation:[^)]*\)", "", description).strip()
        # 清理 markdown 标记
        description = re.sub(r"</?code>", "", description)
        description = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", description)

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

    # 3) 建立子技术 → 父技术映射
    sub_to_parent = {}
    for obj in stix_data.get("objects", []):
        if obj.get("type") == "relationship" and obj.get("relationship_type") == "subtechnique-of":
            src = stix_id_to_tech_id.get(obj.get("source_ref", ""), "")
            tgt = stix_id_to_tech_id.get(obj.get("target_ref", ""), "")
            if src and tgt:
                sub_to_parent[src] = tgt

    print(f"总技术数: {len(tech_info)}, 子技术→父技术映射: {len(sub_to_parent)} 条")

    # 4) 检查是否有已翻译的缓存（支持断点续跑）
    cache_file = os.path.join(OUTPUT_DIR, "translation_cache.json")
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"加载翻译缓存: {len(cache)} 条")

    # 5) LLM 翻译每个技术
    total = len(tech_info)
    translated_count = 0
    failed_count = 0
    skipped_count = 0

    for i, (tech_id, info) in enumerate(tech_info.items()):
        if tech_id in cache and cache[tech_id]:
            skipped_count += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{total}] 跳过已缓存...")
            continue

        print(f"  [{i+1}/{total}] 翻译 {tech_id}: {info['name']}...")
        result = llm_translate(info["name"], info["description"])

        if result:
            cache[tech_id] = result
            translated_count += 1
        else:
            cache[tech_id] = ""
            failed_count += 1
            print(f"    翻译失败: {tech_id}")

        # 每翻译10个保存一次缓存
        if (translated_count + failed_count) % 10 == 0:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)

        # 控制速率
        time.sleep(0.5)

    # 保存最终缓存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"\n翻译完成: 成功={translated_count}, 失败={failed_count}, 跳过(缓存)={skipped_count}")

    # 6) 生成 CSV，子技术继承父技术
    rows = []
    enriched_count = 0
    for tech_id, info in tech_info.items():
        sys_behaviors = cache.get(tech_id, "")

        # 子技术继承父技术的翻译
        if info["is_sub"] and tech_id in sub_to_parent:
            parent_id = sub_to_parent[tech_id]
            parent_behaviors = cache.get(parent_id, "")
            if parent_behaviors:
                if sys_behaviors:
                    # 合并去重
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
            body = f"{info['name']}. {info['description']}"

        rows.append({
            "Subject": f"{tech_id}: {info['name']}",
            "filepath": info["url"],
            "Date": "",
            "Body": body,
            "Source": "MITRE-ATT&CK",
        })

    print(f"  有系统事件描述: {enriched_count}")
    print(f"  保留原始描述: {len(rows) - enriched_count}")

    # 7) 写 CSV
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Subject", "filepath", "Date", "Body", "Source"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\n已写入: {OUTPUT_FILE}")

    # 打印示例
    print("\n=== 翻译示例 ===")
    count = 0
    for tech_id in ["T1055", "T1055.009", "T1070.001", "T1059", "T1003"]:
        if tech_id in cache and cache[tech_id]:
            print(f"\n{tech_id}: {tech_info.get(tech_id, {}).get('name', '')}")
            print(f"  → {cache[tech_id][:200]}")
            count += 1
    if count == 0:
        for tech_id, val in list(cache.items())[:5]:
            if val:
                print(f"\n{tech_id}: {tech_info.get(tech_id, {}).get('name', '')}")
                print(f"  → {val[:200]}")


if __name__ == "__main__":
    main()
