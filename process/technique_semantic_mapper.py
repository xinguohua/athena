"""
基于 sentence-transformers 的 ATT&CK 技术语义映射器。

职责：
1. 日志侧提升：将快照中的系统调用日志翻译为系统事件自然语言描述
2. 技术侧库：从预计算的操作级三元组构建技术描述库
3. 向量检索：用 Sentence-BERT 编码后计算余弦相似度，检索最相似的技术

技术侧描述来源：process/data/technique_triples_transformed.json
日志侧翻译规则：process/translation_rules.py
"""
from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import json
import os
import re
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from process.translation_rules import (
    TYPE_MAP, EVENT_MAP, LOW_INFO_EVENTS,
    translate_event, get_process_role,
)


# ============================================================
# 日志侧提升：快照 → 系统事件自然语言
# ============================================================

def snapshot_to_query(snapshot, *, node_scope: str = "malicious", max_nodes: int = 200) -> str:
    """将快照图翻译为系统事件自然语言查询文本。

    输出格式与技术侧描述一致：
    "subject verb object. subject verb object. ..."

    例如：command shell writes shared library. command shell sends network connection.
    """
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

    nodes.sort(key=lambda d: d["frequency"], reverse=True)
    nodes = nodes[:max_nodes]

    triples = []
    seen = set()
    for n in nodes:
        raw_type = n["type"].strip()

        # 主语：进程节点 → 进程角色，其他节点跳过（只有进程发起动作）
        if raw_type != "SUBJECT_PROCESS":
            continue

        proc_name = _extract_process_name(n["properties"])
        subject = get_process_role(proc_name) if proc_name else "process"

        # 解析事件列表
        events_raw = n["properties"].strip("{} '\"")
        event_items = [e.strip().strip("'\"") for e in events_raw.split(",") if e.strip()]

        for e in event_items:
            t = translate_event(e)
            if not t or t in LOW_INFO_EVENTS:
                continue
            # 构造 "subject verb object" 格式的三元组
            triple = f"{subject} {t}"
            if triple not in seen:
                seen.add(triple)
                triples.append(triple)

    return ". ".join(triples) if triples else ""


# ============================================================
# 技术侧描述库
# ============================================================

def _load_technique_descriptions(
    json_path: str,
) -> Dict[str, str]:
    """从转换后的三元组 JSON 构建技术描述库。

    每个技术的三元组 [{subject, verb, object}, ...] 拼接为一段描述：
    "subject verb object. subject verb object. ..."
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    descriptions = {}
    for tech_id, triples in data.items():
        if not triples:
            continue
        parts = []
        for t in triples:
            parts.append(f"{t['subject']} {t['verb']} {t['object']}")
        descriptions[tech_id] = ". ".join(parts)

    return descriptions


# ============================================================
# 语义匹配器
# ============================================================

class TechniqueSemanticMapper:
    """ATT&CK 技术语义匹配器。

    使用 Sentence-BERT 对日志侧查询和技术侧描述进行编码，
    通过余弦相似度检索最匹配的技术。
    """

    def __init__(
        self,
        *,
        triples_path: str = os.path.join(
            os.path.dirname(__file__), "data/technique_triples_transformed.json"
        ),
        model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
        top_k: int = 5,
        threshold: float = 0.0,
        # 兼容旧接口参数（忽略）
        **kwargs,
    ) -> None:
        self.triples_path = triples_path
        self.model_name = model_name
        self.top_k = int(max(1, top_k))
        self.threshold = threshold

        # 加载技术描述库
        self._tech_descs = _load_technique_descriptions(triples_path)
        self._tech_ids = list(self._tech_descs.keys())
        self._tech_texts = [self._tech_descs[tid] for tid in self._tech_ids]

        # 加载 Sentence-BERT 模型
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "需要 sentence-transformers: pip install sentence-transformers"
            )

        # 预编码技术侧描述
        print(f"[SemMapper] 编码 {len(self._tech_ids)} 个技术描述...")
        self._tech_embeddings = self._model.encode(
            self._tech_texts, show_progress_bar=False, normalize_embeddings=True
        )
        print(f"[SemMapper] 技术描述库就绪。")

    def snapshot_to_query(self, snap) -> str:
        """将快照翻译为查询文本。"""
        return snapshot_to_query(snap)

    def predict_top(self, query: str) -> Optional[Tuple[str, float]]:
        """返回最佳匹配的 (mitre_id, cosine_similarity)。

        similarity 越大越相似（范围 [-1, 1]）。
        """
        if not query or not query.strip():
            return None

        # 编码查询
        q_emb = self._model.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        )

        # 计算余弦相似度（已归一化，点积即余弦）
        similarities = np.dot(self._tech_embeddings, q_emb.T).flatten()

        # 取 top_k
        top_indices = np.argsort(similarities)[::-1][:self.top_k]

        best_idx = top_indices[0]
        best_score = float(similarities[best_idx])

        if best_score < self.threshold:
            return None

        mitre_id = self._tech_ids[best_idx]
        return mitre_id, best_score

    def predict_top_k(self, query: str) -> List[Tuple[str, float]]:
        """返回 top_k 个最匹配的 (mitre_id, cosine_similarity)。"""
        if not query or not query.strip():
            return []

        q_emb = self._model.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        )
        similarities = np.dot(self._tech_embeddings, q_emb.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:self.top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= self.threshold:
                results.append((self._tech_ids[idx], score))

        return results

    def predict_codes(self, queries: List[str]) -> List[str]:
        """批量查询，返回技术 ID 列表。"""
        return [
            (self.predict_top(q) or (None,))[0] or "UNKNOWN"
            for q in queries
        ]

    def predict_codes_batch(self, queries: List[str]) -> List[str]:
        """批量查询（向量化），返回技术 ID 列表。"""
        if not queries:
            return []

        # 批量编码
        q_embs = self._model.encode(
            queries, show_progress_bar=False, normalize_embeddings=True
        )

        # 批量余弦相似度
        similarities = np.dot(q_embs, self._tech_embeddings.T)  # (n_queries, n_techs)

        results = []
        for i in range(len(queries)):
            if not queries[i] or not queries[i].strip():
                results.append("UNKNOWN")
                continue
            best_idx = int(np.argmax(similarities[i]))
            best_score = float(similarities[i, best_idx])
            if best_score < self.threshold:
                results.append("UNKNOWN")
            else:
                results.append(self._tech_ids[best_idx])

        return results


# ============================================================
# 工具函数
# ============================================================

def _extract_process_name(properties: str) -> str:
    """从节点 properties 中提取进程名。"""
    props = properties.strip()
    m = re.search(r"'name'\s*:\s*'([^']+)'", props)
    if m:
        return m.group(1)
    m = re.search(r'"name"\s*:\s*"([^"]+)"', props)
    if m:
        return m.group(1)
    if props and not props.startswith("{") and len(props) < 50:
        return props
    return ""
