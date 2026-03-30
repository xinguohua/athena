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
    "EVENT_FORK": "creates child process",
    "EVENT_EXIT": "exits",
    "EVENT_CONNECT": "connects to",
    "EVENT_SENDTO": "sends network data",
    "EVENT_RECVFROM": "receives network data",
    "EVENT_SENDMSG": "sends message",
    "EVENT_RECVMSG": "receives message",
    "EVENT_MODIFY_PROCESS": "modifies process",
    "EVENT_CREATE_OBJECT": "creates",
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

LOW_INFO_EVENTS = {"closes", "exits"}

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
    # SSH service
    "sshd": "SSH service",
    "ssh": "SSH client",
    "telnetd": "telnet service",
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
    # network utility
    "curl": "network utility",
    "wget": "network utility",
    "nc": "network utility",
    "ncat": "network utility",
    "netcat": "network utility",
    # system service manager
    "systemd": "system service manager",
    "init": "system service manager",
    "launchd": "system service manager",
    # powershell
    "powershell": "powershell",
    "pwsh": "powershell",
    "powershell.exe": "powershell",
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
    ".crt": "certificate file",
    ".cer": "certificate file",
    ".pgp": "authentication key file",
    ".gpg": "authentication key file",
    # executable
    ".exe": "executable",
    ".elf": "executable",
    # script
    ".sh": "shell script",
    ".py": "python script",
    ".pl": "perl script",
    ".rb": "ruby script",
    ".js": "javascript",
    ".vbs": "vbscript",
    ".ps1": "powershell script",
    ".bat": "batch script",
    ".cmd": "batch script",
    # data
    ".db": "database file",
    ".sqlite": "database file",
    ".json": "json file",
    ".xml": "xml file",
    ".csv": "data file",
    # archive
    ".zip": "archive file",
    ".tar": "archive file",
    ".gz": "compressed file",
    ".bz2": "compressed file",
    ".7z": "archive file",
    ".rar": "archive file",
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
    (".ssh/", "SSH configuration"),
    # scheduled task
    ("crontab", "scheduled task configuration"),
    ("/etc/cron", "scheduled task configuration"),
    ("/etc/init.d/", "scheduled task configuration"),
    ("/etc/systemd/", "scheduled task configuration"),
    # proc filesystem
    ("/proc/", "process information"),
    # log
    ("/var/log/", "log file"),
    # general paths
    ("/tmp/", "temporary directory"),
    ("/etc/", "system configuration directory"),
    ("/dev/", "device"),
    ("/bin/", "binary directory"),
    ("/sbin/", "system binary directory"),
    ("/usr/bin/", "user binary directory"),
    ("/home/", "user home directory"),
    ("/root/", "root home directory"),
]

# ============================================================
# 日志侧：端口号 → 协议名
# ============================================================

PORT_MAP = {
    "80": "HTTP",
    "443": "HTTPS",
    "22": "SSH",
    "53": "DNS",
    "25": "SMTP",
    "587": "SMTP",
    "110": "POP3",
    "143": "IMAP",
    "21": "FTP",
    "23": "telnet",
    "3306": "MySQL",
    "5432": "PostgreSQL",
    "6379": "Redis",
    "3389": "RDP",
    "445": "SMB",
    "139": "SMB",
    "8080": "HTTP proxy",
    "8443": "HTTPS",
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
    """文件路径 → 系统级类型描述。"""
    fp = filepath.strip()

    # 先匹配具体路径
    for prefix, desc in FILE_PATH_MAP:
        if prefix in fp:
            return desc

    # 再匹配扩展名
    for ext, desc in FILE_EXT_MAP.items():
        if fp.endswith(ext):
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
    """日志侧：将单个系统调用事件翻译为自然语言。

    输入: 'EVENT_WRITE /tmp/memhelp.so'
    输出: 'writes shared library in temporary directory'
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

    # 对象描述
    descriptions = []

    # 文件类型
    file_type = get_file_type(obj)
    if file_type:
        descriptions.append(file_type)
    else:
        # 从文件名中提取有意义的词
        basename = obj.rsplit("/", 1)[-1] if "/" in obj else obj
        name_stem = basename.rsplit(".", 1)[0] if "." in basename else basename
        words = re.findall(r'[a-zA-Z]+', name_stem)
        meaningful = [w.lower() for w in words if len(w) > 2]
        if meaningful:
            descriptions.append(" ".join(meaningful))

    if descriptions:
        return f"{action} {' '.join(descriptions)}"
    return action


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
