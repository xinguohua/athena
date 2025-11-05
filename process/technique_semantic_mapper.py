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
        csv_path: str = "data/mitreembed_master_Chroma.csv",
        persist_dir: str = "./chroma_db",
        model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
        page_content_column: str = "Body",
        code_column: str = "Subject",
        top_k: int = 5,
        query_mode: str = "nodes_json",
        summary_max_nodes: int = 200,
    ) -> None:

        # ======== 最常用配置 ========
        self.csv_path = csv_path
        self.persist_dir = persist_dir
        self.model_name = model_name
        self.page_content_column = page_content_column
        self.code_column = code_column
        self.top_k = int(max(1, top_k))
        self.summary_max_nodes = max(1, int(summary_max_nodes))

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
    # 内部：将节点聚合为文本
    # ------------------------------
    def _nodes_to_text(self, nodes: List[dict]) -> str:
        # 限制节点数量，优先频率高者
        try:
            def _freq(x: Any) -> int:
                try:
                    return int(x)
                except Exception:
                    return 0

            nodes_sorted = sorted(nodes, key=lambda d: _freq(d.get("frequency")), reverse=True)
            nodes_cut = nodes_sorted[: self.summary_max_nodes]

            # 类型计数
            type_counts = {}
            for n in nodes_cut:
                t = n.get("type", "").strip()
                type_counts[t] = type_counts.get(t, 0) + 1

            header = "Types:" + ", ".join(f"{k}={v}" for k, v in type_counts.items() if k)
            lines = [header]
            for n in nodes_cut:
                t = n.get("type", "")
                f = n.get("frequency", "")
                p = n.get("properties", "")
                lines.append(f"[{t}|freq={f}] {p}")
            return "\n".join(lines)
        except Exception:
            # 最小回退：拼接 properties
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
