"""
基于 Chroma + sentence-transformers 的技术语义映射器。

用途：
- 从 CSV (Body 列) 构建/打开向量库
- 输入自然语言查询（由快照派生的描述），返回最相似的条目
- 提供将快照批量映射为“技术码”的能力（通过配置选择 code 列）

依赖：pandas, langchain-community, chromadb, sentence-transformers
注意：内部使用延迟导入，若依赖缺失，会抛出异常，调用方可捕获并降级。
"""
from __future__ import annotations
from typing import List, Tuple, Optional, Any, Callable
import json
import os
import re
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import chromadb
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class TechniqueSemanticMapper:
    def __init__(
        self,
        *,
        csv_path: str = "data/attack_techniques.csv",
        persist_dir: str = os.path.join(os.path.dirname(__file__), "chroma_db"),
        model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
        page_content_column: str = "Body",
        code_column: str = "Subject",
        top_k: int = 5,
        query_mode: str = "nodes_json",
        summary_max_nodes: int = 200,
        node_scope: str = "malicious",  # "all" | "malicious"
    ) -> None:

        # ======== 最常用配置 ========
        self.csv_path = csv_path
        self.persist_dir = persist_dir
        self.model_name = model_name
        self.page_content_column = page_content_column
        self.code_column = code_column
        self.top_k = int(max(1, top_k))
        self.summary_max_nodes = max(1, int(summary_max_nodes))
        self.node_scope = (node_scope or "all").lower()

        from process.llm_clients.chatanywhere_client import make_chatanywhere_summarizer
        self.llm_summarizer = make_chatanywhere_summarizer(
            api_key="sk-doxpoyNwE1kfZtZeYZEwqwd0MyHxYsr5pP8OG3NYcepsbQdM",
            endpoint="https://api.chatanywhere.org/v1"
        )
        self.summarize = "simple"

        # ======== 构建或打开向量库 ========
        self._vectordb, self._emb = self._open_or_build()


    def predict_top(self, query: str) -> Optional[Tuple[str, float, Any]]:
        """
        返回 (mitre_id, score, best_doc)
        - mitre_id 自动从 metadata['filepath'] 中解析，例如:
            https://attack.mitre.org/techniques/T1090/001/ → T1090/001
        - score 越小越相似
        - best_doc 为向量库文档对象
        """
        if not query or not query.strip():
            return None

        # 调用 Chroma 检索
        try:
            results = self._vectordb.similarity_search_with_score(query, k=self.top_k)
        except Exception as ex:
            print(f"[predict_top] similarity_search_with_score failed: {ex}")
            return None

        if not results:
            return None

        # 找出分值最小(最相似)的文档
        best_doc, best_score = None, None
        for doc, score in results:
            try:
                score = float(score)
            except Exception:
                continue
            if best_score is None or score < best_score:
                best_doc, best_score = doc, score

        if best_doc is None:
            return None

        # 从 metadata 中取 filepath
        filepath = str(best_doc.metadata.get("filepath", "")).strip()

        # 提取 MITRE ID: 允许 T#### / T####/### / T####.###
        # 例如: T1090/001, T1055, T1547.001
        import re
        m = re.search(r"\bT\d{4}(?:[/.]\d{3})?\b", filepath, flags=re.IGNORECASE)
        mitre_id = m.group(0).replace(".", "/") if m else "UNKNOWN"

        return mitre_id, float(best_score), best_doc

    def predict_codes(self, queries: List[str]) -> List[str]:
        """批量查询，返回 code 列（没有则为 UNKNOWN）。"""
        outs: List[str] = []
        for q in queries:
            item = self.predict_top(q)
            outs.append(item[0] if item else "UNKNOWN")
        return outs

    # ------------------------------
    # 快照 -> 查询 文本的简单规则
    # ------------------------------
    def snapshot_to_query(self, snapshot) -> str:
        """构造查询串：
        - query_mode = "nodes_json": 返回 {"nodes":[{"type","properties","frequency"}, ...]}
        - query_mode = "summary_text": 返回聚合文本；可选 simple/llm 摘要
        """
        nodes: List[dict] = []
        for v in snapshot.vs:
            attrs = v.attributes()
            # 仅当 node_scope=="malicious" 时收集 label==1 的节点
            if self.node_scope == "malicious":
                try:
                    lab = int(attrs.get("label", 0))
                except Exception:
                    lab = 0
                if lab != 1:
                    continue
            t = attrs.get("type") or attrs.get("type_name") or ""
            props = attrs.get("properties") or ""
            freq = attrs.get("frequency", "")
            nodes.append({"type": str(t), "properties": str(props), "frequency": freq})

        text = self._nodes_to_text(nodes)
        if self.summarize == "llm" and self.llm_summarizer:
            try:
                return str(self.llm_summarizer(text))
            except Exception:
                return text
        elif self.summarize == "simple":
            return text
        else:
            return text


    # ------------------------------
    # 节点类型翻译
    # ------------------------------
    _TYPE_MAP = {
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
    _EVENT_MAP = {
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
    _EXT_MAP = {
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
    _PATH_MAP = {
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

    def _translate_event(self, event_str: str) -> str:
        """将单个事件字符串翻译为自然语言。
        输入: ' EVENT_WRITE memhelp.so' 或 ' EVENT_SENDTO'
        输出: 'writes shared library memhelp.so' 或 'sends network data'
        """
        event_str = event_str.strip()
        if not event_str:
            return ""

        # 分离事件类型和操作对象
        parts = event_str.split(None, 1)
        event_type = parts[0]
        obj = parts[1] if len(parts) > 1 else ""

        # 翻译事件类型
        action = self._EVENT_MAP.get(event_type, event_type.replace("EVENT_", "").lower())

        if not obj:
            return action

        # 翻译操作对象
        obj_desc = self._describe_object(obj)
        return f"{action} {obj_desc}"

    def _describe_object(self, obj: str) -> str:
        """为文件路径/对象名生成自然语言描述。"""
        obj = obj.strip()
        descriptions = []

        # 提取路径前缀描述
        for path_prefix, desc in self._PATH_MAP.items():
            if path_prefix in obj:
                descriptions.append(f"in {desc}")
                break

        # 提取文件扩展名描述
        for ext, desc in self._EXT_MAP.items():
            if obj.endswith(ext) or f"{ext}" in obj:
                descriptions.append(desc)
                break

        # 提取文件名中有意义的词（如 inject, backdoor, shell 等）
        basename = obj.rsplit("/", 1)[-1] if "/" in obj else obj
        # 去掉扩展名取文件名主体
        name_parts = basename.rsplit(".", 1)
        name_stem = name_parts[0] if name_parts else basename
        # 用驼峰/下划线分割提取有意义的词
        import re as _re
        words = _re.findall(r'[a-zA-Z]+', name_stem)
        meaningful_words = [w.lower() for w in words if len(w) > 2]
        if meaningful_words:
            descriptions.append("(" + " ".join(meaningful_words) + ")")

        if descriptions:
            return obj + " " + " ".join(descriptions)
        return obj

    # ------------------------------
    # 内部：将节点聚合为自然语言文本
    # ------------------------------
    def _nodes_to_text(self, nodes: List[dict]) -> str:
        try:
            def _freq(x: Any) -> int:
                try:
                    return int(x)
                except Exception:
                    return 0

            nodes_sorted = sorted(nodes, key=lambda d: _freq(d.get("frequency")), reverse=True)
            nodes_cut = nodes_sorted[: self.summary_max_nodes]

            lines = []
            for n in nodes_cut:
                node_type = n.get("type", "").strip()
                props = n.get("properties", "").strip()

                # 翻译节点类型
                type_desc = self._TYPE_MAP.get(node_type, node_type.lower())

                # 解析 properties 中的事件列表
                # 格式: "{' EVENT_WRITE memhelp.so', ' EVENT_CLOSE', ...}"
                events_raw = props.strip("{} '\"")
                event_items = [e.strip().strip("'\"") for e in events_raw.split(",")]
                event_items = [e for e in event_items if e]

                # 翻译每个事件
                translated = []
                for e in event_items:
                    t = self._translate_event(e)
                    if t and t not in ("closes",):  # 过滤掉低信息量的 close 事件
                        translated.append(t)

                # 去重保持顺序
                seen = set()
                unique = []
                for t in translated:
                    if t not in seen:
                        seen.add(t)
                        unique.append(t)

                if unique:
                    lines.append(f"{type_desc}: {', '.join(unique)}")

            return ". ".join(lines) if lines else ""
        except Exception:
            return "\n".join(str(n.get("properties", "")) for n in nodes[: self.summary_max_nodes])

    # ------------------------------
    # 内部：打开或构建 Chroma 向量库
    # ------------------------------
    def _open_or_build(self):

        emb = HuggingFaceEmbeddings(model_name=self.model_name)

        vectordb = Chroma(persist_directory=self.persist_dir, embedding_function=emb)
        try:
            cnt = vectordb._collection.count()
        except Exception:
            cnt = 0
        if cnt and cnt > 0:
            return vectordb, emb

        if not (self.csv_path and os.path.exists(self.csv_path)):
            raise RuntimeError("Chroma 向量库为空且未找到 CSV 构建源")

        df = pd.read_csv(self.csv_path)
        if self.page_content_column not in df.columns:
            raise ValueError(f"CSV 缺少列: {self.page_content_column}")
        loader = DataFrameLoader(df, page_content_column=self.page_content_column)
        documents = loader.load()
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=emb,
            persist_directory=self.persist_dir,
        )
        vectordb.persist()
        return vectordb, emb
