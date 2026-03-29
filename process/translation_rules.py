"""
系统事件翻译规则。

两侧翻译的统一规则定义：
1. 查询侧：系统调用日志 → 系统事件自然语言（供 technique_semantic_mapper.py 使用）
2. 技术侧：ATT&CK 高层描述 → 系统事件自然语言（供 build_attack_csv.py 使用）

匹配思路：
  查询侧（快照系统调用） → 规则翻译 → 系统事件自然语言 ← 规则翻译 ← 技术侧（ATT&CK描述）
  两侧都翻译到同一个中间层，用 embedding 做语义匹配。
"""
import re
from typing import List

# ============================================================
# 查询侧翻译规则：系统调用 → 系统事件自然语言
# ============================================================

# 节点类型翻译
TYPE_MAP = {
    "SUBJECT_PROCESS": "process",
    "FILE_OBJECT_FILE": "file",
    "FILE_OBJECT_UNIX_SOCKET": "unix socket",
    "NetFlowObject": "network connection",
    "UnnamedPipeObject": "pipe",
    "SUBJECT_UNIT": "service unit",
    "FILE_OBJECT_DIR": "directory",
    "FILE_OBJECT_BLOCK": "block device",
    "FILE_OBJECT_CHAR": "character device",
    "RegistryKeyObject": "registry key",
    "SrcSinkObject": "source sink",
}

# 系统调用事件翻译
EVENT_MAP = {
    "EVENT_WRITE": "writes",
    "EVENT_READ": "reads",
    "EVENT_OPEN": "opens",
    "EVENT_CLOSE": "closes",
    "EVENT_EXECUTE": "executes",
    "EVENT_FORK": "creates child process",
    "EVENT_EXIT": "exits",
    "EVENT_CONNECT": "connects to network",
    "EVENT_SENDTO": "sends network data",
    "EVENT_RECVFROM": "receives network data",
    "EVENT_SENDMSG": "sends message",
    "EVENT_RECVMSG": "receives message",
    "EVENT_MODIFY_PROCESS": "modifies process",
    "EVENT_CREATE_OBJECT": "creates object",
    "EVENT_CHANGE_PRINCIPAL": "changes principal",
    "EVENT_LSEEK": "seeks in file",
    "EVENT_MODIFY_FILE_ATTRIBUTES": "modifies file attributes",
    "EVENT_RENAME": "renames",
    "EVENT_UNLINK": "deletes",
    "EVENT_MMAP": "maps memory",
    "EVENT_MPROTECT": "changes memory protection",
    "EVENT_CLONE": "clones process",
    "EVENT_BIND": "binds to port",
    "EVENT_ACCEPT": "accepts connection",
    "EVENT_LOGIN": "logs in",
    "EVENT_LOGOUT": "logs out",
}

# 文件扩展名翻译
EXT_MAP = {
    ".so": "shared library",
    ".dll": "dynamic library",
    ".exe": "executable",
    ".sh": "shell script",
    ".py": "python script",
    ".pl": "perl script",
    ".conf": "configuration file",
    ".cfg": "configuration file",
    ".log": "log file",
    ".txt": "text file",
    ".key": "key file",
    ".pem": "certificate file",
    ".crt": "certificate file",
    ".db": "database file",
    ".sqlite": "database file",
    ".json": "json file",
    ".xml": "xml file",
    ".zip": "archive file",
    ".tar": "archive file",
    ".gz": "compressed file",
}

# 路径关键词翻译
PATH_MAP = {
    "/tmp": "temporary directory",
    "/etc": "system configuration directory",
    "/proc": "process filesystem",
    "/dev": "device directory",
    "/bin": "binary directory",
    "/sbin": "system binary directory",
    "/usr/bin": "user binary directory",
    "/var/log": "log directory",
    "/home": "user home directory",
    "/root": "root home directory",
}

# 低信息量事件，翻译后过滤掉
LOW_INFO_EVENTS = {"closes", "exits"}


# ============================================================
# 技术侧翻译规则：ATT&CK 高层描述关键词 → 系统事件自然语言
# ============================================================
# 每条规则: (关键词列表(全部命中才触发), 生成的系统事件描述)
# 关键词匹配忽略大小写

KEYWORD_RULES = [
    # --- 进程注入 ---
    (["inject", "process"],
     "process writes shared library (.so .dll) to disk, process modifies another process memory, process creates or opens temporary files in temporary directory"),
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


# ============================================================
# 翻译函数
# ============================================================

def translate_event(event_str: str) -> str:
    """查询侧：将单个系统调用事件翻译为自然语言。
    输入: ' EVENT_WRITE memhelp.so'
    输出: 'writes memhelp.so shared library (memhelp) in temporary directory'
    """
    event_str = event_str.strip()
    if not event_str:
        return ""

    parts = event_str.split(None, 1)
    event_type = parts[0]
    obj = parts[1] if len(parts) > 1 else ""

    action = EVENT_MAP.get(event_type, event_type.replace("EVENT_", "").lower())
    if not obj:
        return action

    return f"{action} {describe_object(obj)}"


def describe_object(obj: str) -> str:
    """查询侧：为文件路径/对象名生成自然语言描述。"""
    obj = obj.strip()
    descriptions = []

    for path_prefix, desc in PATH_MAP.items():
        if path_prefix in obj:
            descriptions.append(f"in {desc}")
            break

    for ext, desc in EXT_MAP.items():
        if obj.endswith(ext) or ext in obj:
            descriptions.append(desc)
            break

    # 从文件名中提取有意义的词
    basename = obj.rsplit("/", 1)[-1] if "/" in obj else obj
    name_stem = basename.rsplit(".", 1)[0] if "." in basename else basename
    words = re.findall(r'[a-zA-Z]+', name_stem)
    meaningful = [w.lower() for w in words if len(w) > 2]
    if meaningful:
        descriptions.append("(" + " ".join(meaningful) + ")")

    if descriptions:
        return obj + " " + " ".join(descriptions)
    return obj


def generate_system_behaviors(description: str) -> str:
    """技术侧：从 ATT&CK 技术描述中提取关键词，生成系统事件自然语言。"""
    desc_lower = description.lower()
    behaviors = []
    seen = set()

    for keywords, behavior in KEYWORD_RULES:
        if all(kw.lower() in desc_lower for kw in keywords):
            if behavior not in seen:
                seen.add(behavior)
                behaviors.append(behavior)

    return ". ".join(behaviors) if behaviors else ""
