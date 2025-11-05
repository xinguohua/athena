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
        # 外部不再关心这两个参数
        query_mode: str = "nodes_json",
        summarize: Optional[str] = None,
        summary_max_nodes: int = 200,
        llm_summarizer: Optional[Callable[[str], str]] = None,
    ) -> None:

        self.csv_path = csv_path
        self.persist_dir = persist_dir
        self.model_name = model_name
        self.page_content_column = page_content_column
        self.code_column = code_column
        self.top_k = int(max(1, top_k))

        #  内部自动判断 LLM 是否可用
        api_key = os.environ.get("CHATANYWHERE_API_KEY", "").strip()
        if api_key:
            from process.llm_clients.chatanywhere_client import make_chatanywhere_summarizer
            endpoint = os.environ.get("CHATANYWHERE_ENDPOINT", "https://api.openai.com/v1/chat/completions").strip()
            self.llm_summarizer = make_chatanywhere_summarizer(api_key=api_key, endpoint=endpoint)
            self.summarize = "llm"
            print("[Map] 自动启用 LLM 摘要（已检测到 API Key）")
        else:
            self.llm_summarizer = None
            self.summarize = None
            print("[Map] 未检测到 API Key，使用纯本地语义检索模式。")

        self.summary_max_nodes = max(1, int(summary_max_nodes))
        self.query_mode = query_mode

        #  构建向量库
        self._vectordb, self._emb = self._open_or_build()

    # ------------------------------
    # 对外 API
    # ------------------------------
    def predict_top(self, query: str) -> Optional[Tuple[str, float, Any]]:
        """返回 (code, score, document)；score 越小越相似；若失败或无结果返回 None。"""
        if not query or not query.strip():
            return None
        try:
            results = self._vectordb.similarity_search_with_score(query, k=self.top_k)
        except Exception:
            return None
        if not results:
            return None
        # 取最小得分者
        best_doc, best_score = None, None
        for doc, score in results:
            if best_score is None or float(score) < float(best_score):
                best_doc, best_score = doc, float(score)
        if best_doc is None:
            return None
        code = str(best_doc.metadata.get(self.code_column, "")).strip() or "UNKNOWN"
        return code, float(best_score), best_doc

    def predict_codes(self, queries: List[str]) -> List[str]:
        """批量查询，返回 code 列（没有则为 UNKNOWN）。"""
        outs: List[str] = []
        for q in queries:
            item = self.predict_top(q)
            outs.append(item[0] if item else "UNKNOWN")
        return outs

    def map_snapshots(self, snapshots: List) -> List[str]:
        """将快照批量映射为 code（默认使用 Subject 列）。"""
        queries = [self.snapshot_to_query(s) for s in snapshots]
        return self.predict_codes(queries)

    # ------------------------------
    # 快照 -> 查询 文本的简单规则
    # ------------------------------
    def snapshot_to_query(self, snapshot) -> str:
        """构造查询串：
        - query_mode = "nodes_json": 返回 {"nodes":[{"type","properties","frequency"}, ...]}
        - query_mode = "summary_text": 返回聚合文本；可选 simple/llm 摘要
        """
        try:
            nodes: List[dict] = []
            for v in snapshot.vs:
                attrs = v.attributes()
                t = attrs.get("type") or attrs.get("type_name") or ""
                props = attrs.get("properties") or ""
                freq = attrs.get("frequency", "")
                nodes.append({"type": str(t), "properties": str(props), "frequency": freq})

            if self.query_mode == "summary_text":
                text = self._nodes_to_text(nodes)
                if self.summarize == "llm" and self.llm_summarizer:
                    try:
                        return str(self.llm_summarizer(text))
                    except Exception:
                        # 回退到 simple 文本
                        return text
                elif self.summarize == "simple":
                    return text
                else:
                    return text

            # 默认 JSON
            return json.dumps({"nodes": nodes}, ensure_ascii=False)
        except Exception:
            if self.query_mode == "summary_text":
                return ""
            return json.dumps({"nodes": []}, ensure_ascii=False)

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
