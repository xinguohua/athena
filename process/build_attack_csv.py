"""
从 MITRE ATT&CK 官方 STIX 数据生成向量库 CSV。
技术描述后追加系统事件级别的可观测行为描述。

远程运行：
    conda activate prographer && cd /home/nsas2020/fuzz/prographer && python -m process.build_attack_csv

产出：process/data/attack_techniques.csv
"""
import csv
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

# ============================================================
# 关键词 → 系统事件描述 映射规则
# 从技术描述的高层行为推导出系统调用级别的可观测行为
# ============================================================
# 每条规则: (关键词列表(全部命中才触发), 追加的系统事件描述)
# 关键词匹配忽略大小写
KEYWORD_RULES = [
    # --- 进程注入 ---
    (["inject", "process"],
     "process writes shared library (.so .dll) to disk, process modifies another process memory, process creates or opens temporary files"),
    (["inject", "dll"],
     "process loads shared library, process writes shared library (.dll) to disk"),
    (["inject", "library"],
     "process loads shared library (.so), process writes shared library to temporary directory"),
    (["dynamic linker"],
     "process loads or executes shared library (.so), process modifies dynamic linker configuration"),
    (["process hollowing"],
     "process creates child process in suspended state, process modifies another process memory"),
    (["ptrace"],
     "process attaches to another process, process reads and writes another process memory"),

    # --- 内存操作 ---
    (["overwrite", "memory"],
     "process modifies another process memory, process writes to memory mapped region"),
    (["overwrite", "stack"],
     "process modifies another process memory, process overwrites stack"),
    (["memory", "map"],
     "process reads and writes memory mapped files, process maps memory regions"),
    (["shared memory"],
     "process reads and writes shared memory"),
    (["/proc"],
     "process reads and writes /proc filesystem, process opens /proc/pid/maps or /proc/pid/mem"),
    (["virtual memory"],
     "process allocates and writes virtual memory in another process"),

    # --- 文件操作 ---
    (["create", "file"],
     "process creates file, process writes file to directory"),
    (["modify", "file"],
     "process writes file, process modifies file attributes"),
    (["delete", "file"],
     "process deletes file from directory"),
    (["file", "encrypt"],
     "process reads file, process writes encrypted file"),
    (["hidden file"],
     "process creates hidden file, process writes file with dot prefix"),
    (["temporary", "file"],
     "process creates file in temporary directory (/tmp), process writes temporary file"),
    (["log", "file"],
     "process writes log file, process reads log file"),
    (["configuration", "file"],
     "process reads configuration file, process writes configuration file"),
    (["permission", "file"],
     "process modifies file permissions, process changes file attributes"),
    (["shared library"],
     "process writes shared library (.so .dll) to disk, process loads shared library"),
    (["shared module"],
     "process loads shared library (.so .dll), process writes shared module to disk"),

    # --- 网络 ---
    (["command and control"],
     "process sends and receives network data, process connects to remote server"),
    (["exfiltrat"],
     "process reads file, process sends network data to remote server"),
    (["network", "communicat"],
     "process sends and receives network data, process connects to network"),
    (["remote", "server"],
     "process connects to remote server, process sends and receives network data"),
    (["remote", "service"],
     "process connects to remote service, process sends and receives network data"),
    (["lateral", "movement"],
     "process connects to remote host, process sends and receives network data"),
    (["proxy"],
     "process sends and receives network data through proxy, process connects to network"),
    (["tunnel"],
     "process creates network tunnel, process sends and receives network data"),
    (["dns"],
     "process sends dns query, process resolves domain name"),
    (["download"],
     "process receives network data, process writes file to disk"),
    (["upload"],
     "process reads file, process sends network data"),
    (["http", "request"],
     "process sends http request, process receives network data"),
    (["web", "protocol"],
     "process sends and receives network data via http/https"),
    (["socket"],
     "process creates network socket, process connects to network"),

    # --- 进程操作 ---
    (["execute", "command"],
     "process executes command, process creates child process"),
    (["execute", "script"],
     "process executes script file, process creates child process"),
    (["spawn", "process"],
     "process creates child process, process forks"),
    (["child", "process"],
     "process creates child process, process forks"),
    (["privilege", "escalat"],
     "process modifies process, process changes principal or user context"),
    (["elevat", "privilege"],
     "process modifies process, process changes principal or user context"),
    (["schedul", "task"],
     "process creates scheduled task, process writes task configuration"),
    (["cron"],
     "process writes cron job, process modifies crontab file"),
    (["service", "creat"],
     "process creates system service, process writes service configuration file"),
    (["daemon"],
     "process creates daemon, process forks and detaches from terminal"),
    (["boot", "start"],
     "process writes startup configuration, process modifies boot sequence"),
    (["autostart"],
     "process writes autostart entry, process modifies startup configuration"),
    (["persistence"],
     "process writes file for persistence, process modifies startup or scheduled task configuration"),

    # --- 凭证 ---
    (["credential"],
     "process reads password file, process reads sensitive credential file, process accesses credential storage"),
    (["password"],
     "process reads password file (/etc/passwd /etc/shadow), process accesses credential storage"),
    (["kerberos"],
     "process reads kerberos ticket, process accesses credential cache"),
    (["token"],
     "process reads or creates authentication token, process accesses token storage"),
    (["key", "steal"],
     "process reads private key file, process reads certificate file"),
    (["keylog"],
     "process captures keyboard input, process reads input events"),

    # --- 发现/侦察 ---
    (["discover", "system"],
     "process reads system information, process executes system query command"),
    (["discover", "network"],
     "process sends network scan, process reads network configuration"),
    (["enumerat"],
     "process reads system information, process lists files or processes"),
    (["scan", "network"],
     "process sends network probes, process connects to multiple hosts"),

    # --- 规避 ---
    (["obfuscat"],
     "process reads and writes file with encoded or encrypted content"),
    (["encod", "payload"],
     "process reads and writes file with encoded content"),
    (["pack", "payload"],
     "process reads and writes compressed or packed file"),
    (["steganograph"],
     "process reads and writes image or media file to hide data"),
    (["rootkit"],
     "process modifies kernel module, process hides files or processes"),
    (["clear", "log"],
     "process deletes log file, process writes to clear log entries"),
    (["timestomp"],
     "process modifies file timestamp attributes"),
    (["masquerad"],
     "process renames file to appear legitimate, process modifies file name"),

    # --- 数据收集 ---
    (["screen", "capture"],
     "process captures screen content, process writes image file"),
    (["clipboard"],
     "process reads clipboard data"),
    (["audio", "capture"],
     "process reads audio device, process writes audio file"),
    (["video", "capture"],
     "process reads video device, process writes video file"),
    (["archive", "data"],
     "process reads files, process writes compressed archive file"),

    # --- 注册表 (Windows) ---
    (["registry"],
     "process reads or writes registry key, process modifies registry value"),

    # --- 容器 ---
    (["container", "escape"],
     "process executes system calls to escape container namespace, process creates process on host"),
    (["container"],
     "process interacts with container runtime, process creates or modifies container"),
]


def generate_system_behaviors(description: str) -> str:
    """从技术描述中提取关键词，生成系统事件级别的可观测行为。"""
    desc_lower = description.lower()
    behaviors = []
    seen = set()

    for keywords, behavior in KEYWORD_RULES:
        # 所有关键词都必须出现在描述中
        if all(kw.lower() in desc_lower for kw in keywords):
            # 去重
            if behavior not in seen:
                seen.add(behavior)
                behaviors.append(behavior)

    return ". ".join(behaviors) if behaviors else ""


def main():
    # 1) 下载 STIX 数据
    print("正在下载 ATT&CK STIX 数据...")
    resp = requests.get(STIX_URL, timeout=120)
    resp.raise_for_status()
    stix_data = resp.json()
    print(f"下载完成，共 {len(stix_data.get('objects', []))} 个 STIX 对象")

    # 2) 提取 attack-pattern（技术）
    rows = []
    enriched_count = 0
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

        # 生成系统事件描述，放到 Body 开头（embedding 模型有 token 限制，开头的内容优先被编码）
        sys_behaviors = generate_system_behaviors(description)
        if sys_behaviors:
            body = f"Observable system behaviors: {sys_behaviors}\n\n{name}. {description}"
            enriched_count += 1
        else:
            body = f"{name}. {description}"

        rows.append({
            "Subject": f"{tech_id}: {name}",
            "filepath": url,
            "Date": "",
            "Body": body,
            "Source": "MITRE-ATT&CK",
            "tech_id": tech_id,
            "is_subtechnique": is_sub,
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
