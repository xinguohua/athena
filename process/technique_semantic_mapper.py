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
from typing import List, Tuple, Optional, Any


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
    ) -> None:
        self.csv_path = csv_path
        self.persist_dir = persist_dir
        self.model_name = model_name
        self.page_content_column = page_content_column
        self.code_column = code_column
        self.top_k = int(max(1, top_k))

        # 延迟导入与构建
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
    @staticmethod
    def snapshot_to_query(snapshot) -> str:
        """从快照构造一个简短的行为描述。尽量通用，避免依赖特定字段。

        策略：
        - 汇总节点 type_name 计数；
        - 收集部分节点 name（去重、限制数量）；
        - 简短模板化输出。
        """
        try:
            type_count = {}
            names = []
            for v in snapshot.vs:
                attrs = v.attributes()
                t = str(attrs.get("type_name", attrs.get("type", "UNKNOWN")))
                type_count[t] = type_count.get(t, 0) + 1
                nm = str(attrs.get("name", ""))
                if nm:
                    names.append(nm)
            # 去重并截断
            uniq_names = []
            seen = set()
            for nm in names:
                if nm not in seen:
                    uniq_names.append(nm)
                    seen.add(nm)
                if len(uniq_names) >= 10:
                    break
            parts = [
                "Snapshot behavior summary:",
                "Types: " + ", ".join(f"{k}:{v}" for k, v in sorted(type_count.items(), key=lambda x: (-x[1], x[0]))),
            ]
            if uniq_names:
                parts.append("Nodes: " + ", ".join(uniq_names))
            return "; ".join(parts)
        except Exception:
            return "Snapshot behavior summary: (unavailable)"

    # ------------------------------
    # 内部：打开或构建 Chroma 向量库
    # ------------------------------
    def _open_or_build(self):
        import os
        import pandas as pd  # type: ignore
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
        from langchain_community.document_loaders import DataFrameLoader  # type: ignore
        from langchain_community.vectorstores import Chroma  # type: ignore

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
