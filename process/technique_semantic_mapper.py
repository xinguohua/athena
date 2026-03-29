"""
基于 Chroma + sentence-transformers 的 ATT&CK 技术语义映射器。

职责：
1. 查询侧翻译：将快照中的系统调用日志翻译为系统事件自然语言描述
2. 向量检索：用翻译后的查询文本在 ATT&CK 技术向量库中检索最相似的技术
3. 技术码提取：从检索结果中提取 MITRE ATT&CK 技术 ID

翻译规则定义在 process/translation_rules.py 中。
"""
from __future__ import annotations
from typing import List, Tuple, Optional, Any
import os
import re
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DataFrameLoader
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from process.translation_rules import (
    TYPE_MAP, EVENT_MAP, LOW_INFO_EVENTS,
    translate_event,
)


# ============================================================
# 查询侧翻译：快照 → 系统事件自然语言
# ============================================================

def snapshot_to_query(snapshot, *, node_scope: str = "malicious", max_nodes: int = 200) -> str:
    """将快照图翻译为系统事件自然语言查询文本。"""
    # 1. 收集节点
    nodes = []
    for v in snapshot.vs:
        attrs = v.attributes()
        if node_scope == "malicious":
            try:
                if int(attrs.get("label", 0)) != 1:
                    continue
            except Exception:
                continue
        node_type = str(attrs.get("type") or attrs.get("type_name") or "")
        props = str(attrs.get("properties") or "")
        freq = attrs.get("frequency", 0)
        try:
            freq = int(freq)
        except Exception:
            freq = 0
        nodes.append({"type": node_type, "properties": props, "frequency": freq})

    # 2. 按频率排序，截取
    nodes.sort(key=lambda d: d["frequency"], reverse=True)
    nodes = nodes[:max_nodes]

    # 3. 翻译每个节点
    lines = []
    for n in nodes:
        type_desc = TYPE_MAP.get(n["type"].strip(), n["type"].strip().lower())

        # 解析 properties: "{' EVENT_WRITE memhelp.so', ' EVENT_CLOSE', ...}"
        events_raw = n["properties"].strip("{} '\"")
        event_items = [e.strip().strip("'\"") for e in events_raw.split(",") if e.strip()]

        # 翻译并去重
        seen = set()
        translated = []
        for e in event_items:
            t = translate_event(e)
            if t and t not in LOW_INFO_EVENTS and t not in seen:
                seen.add(t)
                translated.append(t)

        if translated:
            lines.append(f"{type_desc}: {', '.join(translated)}")

    return ". ".join(lines) if lines else ""


# ============================================================
# 向量库检索
# ============================================================

class TechniqueSemanticMapper:
    """ATT&CK 技术语义匹配器：管理向量库，执行检索。"""

    def __init__(
        self,
        *,
        csv_path: str = os.path.join(os.path.dirname(__file__), "data/attack_techniques.csv"),
        persist_dir: str = os.path.join(os.path.dirname(__file__), "chroma_db"),
        model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
        page_content_column: str = "Body",
        top_k: int = 5,
    ) -> None:
        self.csv_path = csv_path
        self.persist_dir = persist_dir
        self.model_name = model_name
        self.page_content_column = page_content_column
        self.top_k = int(max(1, top_k))
        self._vectordb, self._emb = self._open_or_build()

    def snapshot_to_query(self, snap) -> str:
        """兼容旧接口：将快照翻译为查询文本。"""
        return snapshot_to_query(snap)

    def predict_top(self, query: str) -> Optional[Tuple[str, float, Any]]:
        """返回最佳匹配的 (mitre_id, score, doc)，score 越小越相似。"""
        if not query or not query.strip():
            return None
        try:
            results = self._vectordb.similarity_search_with_score(query, k=self.top_k)
        except Exception as ex:
            print(f"[predict_top] 检索失败: {ex}")
            return None
        if not results:
            return None

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

        mitre_id = _extract_mitre_id(best_doc.metadata.get("filepath", ""))
        return mitre_id, float(best_score), best_doc

    def predict_codes(self, queries: List[str]) -> List[str]:
        """批量查询，返回技术 ID 列表。"""
        return [
            (self.predict_top(q) or (None,))[0] or "UNKNOWN"
            for q in queries
        ]

    def _open_or_build(self):
        """打开已有向量库，或从 CSV 构建新库。"""
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


def _extract_mitre_id(filepath: str) -> str:
    """从 ATT&CK URL 中提取技术 ID，如 T1055/009。"""
    filepath = str(filepath).strip()
    m = re.search(r"\bT\d{4}(?:[/.]\d{3})?\b", filepath, flags=re.IGNORECASE)
    return m.group(0).replace(".", "/") if m else "UNKNOWN"
