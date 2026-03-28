"""
攻击技术匹配诊断脚本 — 只跑1个恶意快照，重点对比查询文本 vs 匹配到的技术描述。

远程运行：
    conda activate prographer && cd /home/nsas2020/fuzz/prographer && python -m process.debug_matching

产出：debug_matching_output.json
"""
import json
import os
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ========== 日志同时写终端和文件 ==========
LOG_FILE = "debug_matching.log"

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, msg):
        for s in self.streams:
            s.write(msg)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

_log_fh = open(LOG_FILE, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log_fh)

# ========== 配置 ==========
GLOBAL_ID = "xgh"
SNAPSHOT_FILE = os.path.join(os.path.dirname(__file__), f"snapshot_data_{GLOBAL_ID}.pkl")
OUTPUT_FILE = "debug_matching_output.json"
# 只诊断第几个恶意快照（从0开始），选一个 true_label=1 的效果最好
DIAG_INDEX = 0

MAPPER_CONFIG = {
    "csv_path": os.path.join(os.path.dirname(__file__), "data/mitreembed_master_Chroma.csv"),
    "persist_dir": os.path.join(os.path.dirname(__file__), "chroma_db"),
    "model_name": "sentence-transformers/all-MiniLM-L12-v2",
    "page_content_column": "Body",
    "code_column": "Subject",
    "top_k": 5,
}


def main():
    # 1) 加载快照
    if not os.path.exists(SNAPSHOT_FILE):
        print(f"未找到 {SNAPSHOT_FILE}，请先运行 train_all.py")
        return

    with open(SNAPSHOT_FILE, "rb") as f:
        snapshot_data = pickle.load(f)

    all_snapshots = snapshot_data["all_snapshots"]
    mal_start = snapshot_data["malicious_idx_start"]
    mal_end = snapshot_data["malicious_idx_end"]
    mal_snapshots = all_snapshots[mal_start: mal_end + 1]
    print(f"恶意快照总数: {len(mal_snapshots)}")

    # 找恶意节点数最多的快照
    target_idx = 0
    max_mal = 0
    for i, snap in enumerate(mal_snapshots):
        try:
            cnt = sum(1 for v in snap.vs if int(v.attributes().get("label", 0)) == 1)
            if cnt > max_mal:
                max_mal = cnt
                target_idx = i
        except Exception:
            pass
    print(f"选择恶意节点最多的快照: [{target_idx}] 恶意节点数={max_mal}")

    snap = mal_snapshots[target_idx]
    print(f"\n===== 诊断快照 [{target_idx}] (全局索引 {mal_start + target_idx}) =====")

    # 2) 初始化语义映射器
    from process.technique_semantic_mapper import TechniqueSemanticMapper
    mapper = TechniqueSemanticMapper(**MAPPER_CONFIG)

    db_count = 0
    try:
        db_count = mapper._vectordb._collection.count()
    except Exception:
        pass
    print(f"向量库文档数: {db_count}")

    # 3) 快照节点详情
    node_count = snap.vcount()
    mal_nodes = []
    all_nodes = []
    for v in snap.vs:
        attrs = v.attributes()
        label = 0
        try:
            label = int(attrs.get("label", 0))
        except Exception:
            pass
        node_info = {
            "type": str(attrs.get("type") or attrs.get("type_name") or ""),
            "properties": str(attrs.get("properties") or ""),
            "frequency": attrs.get("frequency", ""),
            "label": label,
        }
        all_nodes.append(node_info)
        if label == 1:
            mal_nodes.append(node_info)

    print(f"节点总数: {node_count}, 恶意节点数: {len(mal_nodes)}")

    # 4) 生成查询文本（完整，不截断）
    query = mapper.snapshot_to_query(snap)
    print(f"\n----- 查询文本 (长度={len(query)}) -----")
    print(query)

    # 5) 向量检索 top_k，输出完整技术描述
    print(f"\n----- Top {mapper.top_k} 匹配结果 -----")
    candidates = []
    try:
        raw_results = mapper._vectordb.similarity_search_with_score(query, k=mapper.top_k)
        for rank, (doc, score) in enumerate(raw_results):
            filepath = str(doc.metadata.get("filepath", ""))
            m = re.search(r"\bT\d{4}(?:[/.]\d{3})?\b", filepath, flags=re.IGNORECASE)
            tech_id = m.group(0).replace(".", "/") if m else "UNKNOWN"
            full_content = doc.page_content

            candidate = {
                "rank": rank + 1,
                "tech_id": tech_id,
                "score": round(float(score), 6),
                "filepath": filepath,
                "metadata": {k: str(v) for k, v in doc.metadata.items()},
                "full_content": full_content,
            }
            candidates.append(candidate)

            print(f"\n  [{rank+1}] {tech_id}  score={float(score):.6f}")
            print(f"      filepath: {filepath}")
            print(f"      内容 (前500字):")
            print(f"      {full_content[:500]}")
    except Exception as ex:
        print(f"  检索失败: {ex}")

    # 6) 语义差距分析
    print(f"\n----- 语义差距分析 -----")
    if candidates:
        best = candidates[0]
        print(f"最佳匹配: {best['tech_id']}  距离={best['score']:.6f}")
        if len(candidates) >= 2:
            gap = candidates[1]["score"] - candidates[0]["score"]
            print(f"第1名 vs 第2名 距离差: {gap:.6f} ({'区分度高' if gap > 0.05 else '区分度低，匹配不确定'})")
        # 查询文本里的关键词 vs 最佳匹配内容里的关键词
        query_words = set(query.lower().split())
        best_words = set(best["full_content"].lower().split())
        overlap = query_words & best_words
        only_query = query_words - best_words
        print(f"词汇重叠数: {len(overlap)}")
        print(f"查询独有词(前30): {list(only_query)[:30]}")

    # 7) 输出 JSON
    output = {
        "config": {
            "snapshot_idx": target_idx,
            "global_idx": mal_start + target_idx,
            "vectordb_doc_count": db_count,
        },
        "query_text": query,
        "query_length": len(query),
        "snapshot_nodes": {
            "total": node_count,
            "malicious_count": len(mal_nodes),
            "malicious_nodes": mal_nodes[:50],
            "all_nodes_sample": all_nodes[:20],
        },
        "candidates": candidates,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n诊断结果已写入: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
