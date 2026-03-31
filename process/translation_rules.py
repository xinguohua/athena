"""
双侧语义提升规则表。

日志侧规则：
  - PROCESS_ROLE_MAP: 进程名 → 功能角色
  - FILE_TYPE_MAP: 文件扩展名/路径 → 系统级类型
  - NET_TYPE_MAP: 端口 → 协议名
  - EVENT_MAP: 事件类型 → 操作动词
  - TYPE_MAP: 节点类型 → 描述

技术侧规则：
  - INTENT_VERB_MAP: 意图动词 → 操作动词列表
  - INTENT_OBJECT_MAP: 意图级宾语 → 系统级对象类型
  - INTENT_SUBJECT_MAP: 意图级主语 → 系统级主体
"""
import re
from typing import List, Tuple, Optional

# ============================================================
# 日志侧：节点类型翻译
# ============================================================

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

# ============================================================
# 日志侧：事件类型 → 操作动词
# ============================================================

EVENT_MAP = {
    "EVENT_WRITE": "writes",
    "EVENT_READ": "reads",
    "EVENT_OPEN": "opens",
    "EVENT_CLOSE": "closes",
    "EVENT_EXECUTE": "executes",
    "EVENT_FORK": "executes process",
    "EVENT_EXIT": "exits",
    "EVENT_CONNECT": "sends network connection",
    "EVENT_SENDTO": "sends network connection",
    "EVENT_RECVFROM": "receives network connection",
    "EVENT_SENDMSG": "sends network connection",
    "EVENT_RECVMSG": "receives network connection",
    "EVENT_MODIFY_PROCESS": "writes process",
    "EVENT_CREATE_OBJECT": "creates",
    "EVENT_CHANGE_PRINCIPAL": "changes principal",
    "EVENT_LSEEK": "seeks in file",
    "EVENT_MODIFY_FILE_ATTRIBUTES": "writes configuration file",
    "EVENT_RENAME": "writes file",
    "EVENT_UNLINK": "writes file",
    "EVENT_MMAP": "maps memory",
    "EVENT_MPROTECT": "changes memory protection",
    "EVENT_CLONE": "executes process",
    "EVENT_BIND": "sends network connection",
    "EVENT_ACCEPT": "receives network connection",
    "EVENT_LOGIN": "reads credential file",
    "EVENT_LOGOUT": "exits",
}

# 低信息量动作前缀：translate_event 返回以这些词开头的结果会被过滤
LOW_INFO_PREFIXES = {"closes", "exits", "opens", "seeks", "maps memory",
                     "changes memory", "creates", "changes principal"}

# ============================================================
# 日志侧：进程名 → 功能角色
# ============================================================

PROCESS_ROLE_MAP = {
    # command shell
    "bash": "command shell",
    "sh": "command shell",
    "zsh": "command shell",
    "csh": "command shell",
    "tcsh": "command shell",
    "dash": "command shell",
    "fish": "command shell",
    # scripting interpreter
    "python": "scripting interpreter",
    "python2": "scripting interpreter",
    "python3": "scripting interpreter",
    "perl": "scripting interpreter",
    "ruby": "scripting interpreter",
    "node": "scripting interpreter",
    "php": "scripting interpreter",
    # remote access service
    "sshd": "remote access service",
    "ssh": "remote access service",
    "telnetd": "remote access service",
    # task scheduler
    "crond": "task scheduler",
    "cron": "task scheduler",
    "atd": "task scheduler",
    "at": "task scheduler",
    # web server
    "apache": "web server",
    "apache2": "web server",
    "httpd": "web server",
    "nginx": "web server",
    # database service
    "mysqld": "database service",
    "postgres": "database service",
    "mongod": "database service",
    "redis-server": "database service",
    # network utility → process（兜底）
    "curl": "process",
    "wget": "process",
    "nc": "process",
    "ncat": "process",
    "netcat": "process",
    # system service
    "systemd": "system service",
    "init": "system service",
    "launchd": "system service",
    # scripting interpreter (PowerShell)
    "powershell": "scripting interpreter",
    "pwsh": "scripting interpreter",
    "powershell.exe": "scripting interpreter",
    # cmd
    "cmd": "command shell",
    "cmd.exe": "command shell",
}

# ============================================================
# 日志侧：文件扩展名 → 系统级类型
# ============================================================

FILE_EXT_MAP = {
    # shared library
    ".so": "shared library",
    ".dll": "shared library",
    ".dylib": "shared library",
    # configuration file
    ".conf": "configuration file",
    ".cfg": "configuration file",
    ".ini": "configuration file",
    ".yaml": "configuration file",
    ".yml": "configuration file",
    # log file
    ".log": "log file",
    ".evtx": "log file",
    # credential / key file
    ".pem": "authentication key file",
    ".key": "authentication key file",
    ".crt": "authentication key file",
    ".cer": "authentication key file",
    ".pgp": "authentication key file",
    ".gpg": "authentication key file",
    # executable
    ".exe": "executable",
    ".elf": "executable",
    # script → executable
    ".sh": "executable",
    ".py": "executable",
    ".pl": "executable",
    ".rb": "executable",
    ".js": "executable",
    ".vbs": "executable",
    ".ps1": "executable",
    ".bat": "executable",
    ".cmd": "executable",
    # data → file
    ".db": "file",
    ".sqlite": "file",
    ".json": "file",
    ".xml": "file",
    ".csv": "file",
    # archive → file
    ".zip": "file",
    ".tar": "file",
    ".gz": "file",
    ".bz2": "file",
    ".7z": "file",
    ".rar": "file",
}

# ============================================================
# 日志侧：路径前缀 → 系统级类型
# ============================================================

FILE_PATH_MAP = [
    # credential files (具体路径优先匹配)
    ("/etc/shadow", "credential file"),
    ("/etc/passwd", "credential file"),
    ("/etc/master.passwd", "credential file"),
    # authorized keys
    ("authorized_keys", "authentication key file"),
    (".ssh/", "configuration file"),
    # scheduled task → configuration file（17种类型中无单独的 scheduled task 类型）
    ("crontab", "configuration file"),
    ("/etc/cron", "configuration file"),
    ("/etc/init.d/", "configuration file"),
    ("/etc/systemd/", "configuration file"),
    # proc filesystem → process
    ("/proc/", "process"),
    # log
    ("/var/log/", "log file"),
    # general paths → 映射为17种类型中的兜底类型
    ("/tmp/", "file"),
    ("/etc/", "configuration file"),
    ("/dev/", "file"),
    ("/bin/", "executable"),
    ("/sbin/", "executable"),
    ("/usr/bin/", "executable"),
    ("/home/", "file"),
    ("/root/", "file"),
]

# ============================================================
# 日志侧：端口号 → 协议名
# ============================================================

PORT_MAP = {
    "80": "network connection",
    "443": "network connection",
    "22": "network connection",
    "53": "network connection",
    "25": "email",
    "587": "email",
    "110": "network connection",
    "143": "network connection",
    "21": "network connection",
    "23": "network connection",
    "3306": "network connection",
    "5432": "network connection",
    "6379": "network connection",
    "3389": "network connection",
    "445": "network connection",
    "139": "network connection",
    "8080": "network connection",
    "8443": "network connection",
}

# ============================================================
# 技术侧：意图级主语 → 系统级主体
# ============================================================

INTENT_SUBJECT_MAP = {
    "adversaries": "process",
    "adversary": "process",
    "threat actors": "process",
    "threat actor": "process",
    "attackers": "process",
    "attacker": "process",
    "victims": "process",
    "victim": "process",
    "legitimate users": "user",
    "legitimate user": "user",
    "an adversary": "process",
}

# ============================================================
# 技术侧：意图动词 → 操作动词列表
# ============================================================

INTENT_VERB_MAP = {
    "inject": ["writes", "writes", "reads"],
    "exfiltrate": ["reads", "sends"],
    "persist": ["writes"],
    "establish": ["writes"],
    "dump": ["reads", "reads", "writes"],
    "escalate": ["reads", "executes"],
    "elevate": ["reads", "executes"],
    "steal": ["reads", "sends"],
    "hijack": ["reads", "writes"],
    "enumerate": ["reads"],
    "discover": ["reads"],
    "collect": ["reads"],
    "encrypt": ["reads", "writes"],
    "obfuscate": ["reads", "writes"],
    "masquerade": ["writes", "renames"],
    "impersonate": ["reads", "executes"],
    "capture": ["reads"],
    "harvest": ["reads"],
    "compromise": ["reads", "writes", "executes"],
    "exploit": ["reads", "executes"],
    "abuse": ["executes"],
    "leverage": ["executes"],
    "deploy": ["writes", "executes"],
    "deliver": ["writes", "sends"],
    "stage": ["writes"],
    "scan": ["sends", "receives"],
    "sniff": ["reads"],
    "intercept": ["reads"],
    "tamper": ["writes"],
    "modify": ["writes"],
    "create": ["writes"],
    "delete": ["deletes"],
    "disable": ["writes"],
    "clear": ["deletes", "writes"],
}

# ============================================================
# 技术侧：意图级宾语关键词 → 系统级对象类型
# ============================================================

INTENT_OBJECT_MAP = {
    # 进程/代码相关
    "code": "shared library",
    "malicious code": "shared library",
    "arbitrary code": "shared library",
    "processes": "process memory",
    "process memory": "process memory",
    "process": "process memory",
    "dll": "shared library",
    "shared library": "shared library",
    "shared module": "shared library",
    "executable": "executable",
    "binary": "executable",
    "payload": "executable",
    # 凭证相关
    "credentials": "credential file",
    "passwords": "credential file",
    "password": "credential file",
    "credential": "credential file",
    "hashes": "credential file",
    "tokens": "authentication token",
    "token": "authentication token",
    "keys": "authentication key file",
    "key": "authentication key file",
    "kerberos ticket": "authentication token",
    "certificates": "certificate file",
    # 文件相关
    "data": "file",
    "collected data": "file",
    "files": "file",
    "file": "file",
    "documents": "file",
    "registry": "configuration file",
    "registry key": "configuration file",
    "configuration": "configuration file",
    "scheduled task": "scheduled task configuration",
    "cron job": "scheduled task configuration",
    "startup entry": "scheduled task configuration",
    "service": "service configuration",
    "log": "log file",
    "logs": "log file",
    "event logs": "log file",
    # 网络相关
    "network": "network data",
    "network data": "network data",
    "traffic": "network data",
    "c2 channel": "network connection",
    "command and control": "network connection",
    "remote server": "remote connection",
    "remote service": "remote connection",
    "remote services": "remote connection",
    "email": "email",
    "phishing": "email",
}

# ============================================================
# 日志侧翻译函数
# ============================================================

def get_process_role(process_name: str) -> str:
    """进程名 → 功能角色。未命中则保持原名。"""
    name = process_name.strip().lower()
    # 去掉路径前缀
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    return PROCESS_ROLE_MAP.get(name, name)


def get_file_type(filepath: str) -> str:
    """文件路径 → 系统级类型描述。扩展名优先于路径前缀。"""
    fp = filepath.strip()

    # 先匹配扩展名（优先，因为 .so/.dll 比 /tmp/ 更有信息量）
    for ext, desc in FILE_EXT_MAP.items():
        if fp.endswith(ext):
            return desc

    # 再匹配具体路径
    for prefix, desc in FILE_PATH_MAP:
        if prefix in fp:
            return desc

    return ""


def get_port_protocol(port: str) -> str:
    """端口号 → 协议名。"""
    return PORT_MAP.get(port.strip(), "")


def is_internal_ip(ip: str) -> bool:
    """判断 IP 是否为内网地址（RFC 1918）。"""
    ip = ip.strip()
    return (ip.startswith("10.") or
            ip.startswith("192.168.") or
            ip.startswith("172.16.") or ip.startswith("172.17.") or
            ip.startswith("172.18.") or ip.startswith("172.19.") or
            ip.startswith("172.2") or ip.startswith("172.30.") or
            ip.startswith("172.31.") or
            ip.startswith("127.") or
            ip == "localhost")


def translate_event(event_str: str) -> str:
    """日志侧：将单个系统调用事件翻译为系统事件描述。

    输入: 'EVENT_WRITE /tmp/memhelp.so'
    输出: 'writes shared library'

    输出格式为 "verb object_type"，与技术侧三元组的 verb+object 部分对齐。
    """
    event_str = event_str.strip()
    if not event_str:
        return ""

    parts = event_str.split(None, 1)
    event_type = parts[0]
    obj = parts[1] if len(parts) > 1 else ""

    action = EVENT_MAP.get(event_type, "")
    if not action:
        return ""

    # 如果事件已经包含了完整的 "verb object_type"（如 "sends network connection"），直接返回
    if " " in action:
        return action

    if not obj:
        return action

    # 将文件路径映射为系统级类型，同时保留原始路径/文件名
    file_type = get_file_type(obj)
    if file_type:
        return f"{action} {obj} {file_type}"

    # 未命中映射表的文件 → 保留原始路径
    return f"{action} {obj}"


# ============================================================
# 技术侧翻译函数
# ============================================================

def map_intent_subject(subject: str) -> str:
    """技术侧：意图级主语 → 系统级主体。"""
    s = subject.strip().lower()
    return INTENT_SUBJECT_MAP.get(s, "process")


def map_intent_verb(verb: str) -> Optional[List[str]]:
    """技术侧：意图动词 → 操作动词列表。不在表中返回 None（丢弃该三元组）。"""
    v = verb.strip().lower()
    # 先精确匹配
    if v in INTENT_VERB_MAP:
        return INTENT_VERB_MAP[v]
    # 词干匹配（如 injecting → inject）
    for intent_v, ops in INTENT_VERB_MAP.items():
        if v.startswith(intent_v) or intent_v.startswith(v):
            return ops
    return None


def map_intent_object(obj: str) -> str:
    """技术侧：意图级宾语 → 系统级对象类型。"""
    o = obj.strip().lower()
    # 去掉意图修饰词
    for modifier in ["malicious", "stolen", "legitimate", "arbitrary",
                     "suspicious", "unauthorized", "compromised",
                     "sensitive", "valid", "additional"]:
        o = o.replace(modifier, "").strip()

    # 精确匹配
    if o in INTENT_OBJECT_MAP:
        return INTENT_OBJECT_MAP[o]
    # 子串匹配
    for key, val in INTENT_OBJECT_MAP.items():
        if key in o:
            return val
    return o
