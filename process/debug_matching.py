"""
攻击技术匹配诊断脚本。

远程运行：
    conda activate prographer && cd /home/nsas2020/fuzz/prographer && python -m process.debug_matching

产出：debug_matching_output.json — 推到 GitHub 后本地分析
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
SNAPSHOT_FILE = f"snapshot_data_{GLOBAL_ID}.pkl"
OUTPUT_FILE = "debug_matching_output.json"

MAPPER_CONFIG = {
    "csv_path": "data/mitreembed_master_Chroma.csv",
    "persist_dir": "./chroma_db",
    "model_name": "sentence-transformers/all-MiniLM-L12-v2",
    "page_content_column": "Body",
    "code_column": "Subject",
    "top_k": 5,
}
SEQ_LIBRARY_PATH = "technique_sequences.txt"


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
    print(f"恶意快照数量: {len(mal_snapshots)}")

    # 2) 初始化语义映射器
    from process.technique_semantic_mapper import TechniqueSemanticMapper
    mapper = TechniqueSemanticMapper(**MAPPER_CONFIG)

    # 3) 向量库基本信息
    db_count = 0
    try:
        db_count = mapper._vectordb._collection.count()
    except Exception:
        pass
    print(f"向量库文档数: {db_count}")

    # 4) 逐快照诊断
    results = []
    for i, snap in enumerate(mal_snapshots):
        # 真实标签
        true_label = 0
        try:
            for v in snap.vs:
                if int(v.attributes().get("label", 0)) == 1:
                    true_label = 1
                    break
        except Exception:
            pass

        # 生成查询文本
        query = mapper.snapshot_to_query(snap)

        # 节点统计
        node_count = snap.vcount()
        mal_node_count = 0
        try:
            mal_node_count = sum(1 for v in snap.vs if int(v.attributes().get("label", 0)) == 1)
        except Exception:
            pass

        # 向量检索 top_k 全部候选
        candidates = []
        try:
            raw_results = mapper._vectordb.similarity_search_with_score(query, k=mapper.top_k)
            for doc, score in raw_results:
                filepath = str(doc.metadata.get("filepath", ""))
                m = re.search(r"\bT\d{4}(?:[/.]\d{3})?\b", filepath, flags=re.IGNORECASE)
                tech_id = m.group(0).replace(".", "/") if m else "UNKNOWN"
                candidates.append({
                    "tech_id": tech_id,
                    "score": round(float(score), 6),
                    "filepath": filepath,
                    "content_preview": doc.page_content[:300],
                })
        except Exception as ex:
            candidates.append({"error": str(ex)})

        best_tech = candidates[0]["tech_id"] if candidates and "tech_id" in candidates[0] else "UNKNOWN"
        best_score = candidates[0].get("score") if candidates else None

        # 候选之间的分数差距（区分度）
        score_gap = None
        if len(candidates) >= 2 and "score" in candidates[0] and "score" in candidates[1]:
            score_gap = round(candidates[1]["score"] - candidates[0]["score"], 6)

        results.append({
            "snapshot_idx": i,
            "global_idx": mal_start + i,
            "true_label": true_label,
            "node_count": node_count,
            "malicious_node_count": mal_node_count,
            "query_text": query[:2000],
            "query_length": len(query),
            "best_match": best_tech,
            "best_score": best_score,
            "score_gap_1st_2nd": score_gap,
            "candidates": candidates,
        })

        status = f"score={best_score:.4f} gap={score_gap:.4f}" if best_score and score_gap else "FAILED"
        print(f"  [{i:3d}] label={true_label} nodes={node_count:4d} mal={mal_node_count:3d} -> {best_tech:12s} {status}")

    # 5) 加载序列库
    lib = []
    if os.path.exists(SEQ_LIBRARY_PATH):
        with open(SEQ_LIBRARY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [x for x in line.replace("\t", " ").replace(",", " ").split(" ") if x]
                if parts:
                    lib.append(parts)

    # 6) 汇总
    all_scores = [r["best_score"] for r in results if r["best_score"] is not None]
    all_gaps = [r["score_gap_1st_2nd"] for r in results if r["score_gap_1st_2nd"] is not None]
    tech_seq_all = [r["best_match"] for r in results]

    # 统计每个技术码出现次数
    tech_freq = {}
    for t in tech_seq_all:
        tech_freq[t] = tech_freq.get(t, 0) + 1

    # 真阳性快照的技术码分布
    tech_freq_positive = {}
    for r in results:
        if r["true_label"] == 1:
            t = r["best_match"]
            tech_freq_positive[t] = tech_freq_positive.get(t, 0) + 1

    output = {
        "config": {
            "global_id": GLOBAL_ID,
            "mal_start": mal_start,
            "mal_end": mal_end,
            "mal_snapshot_count": len(mal_snapshots),
            "vectordb_doc_count": db_count,
            "mapper_config": MAPPER_CONFIG,
        },
        "summary": {
            "score_min": min(all_scores) if all_scores else None,
            "score_max": max(all_scores) if all_scores else None,
            "score_mean": round(sum(all_scores) / len(all_scores), 6) if all_scores else None,
            "gap_min": min(all_gaps) if all_gaps else None,
            "gap_max": max(all_gaps) if all_gaps else None,
            "gap_mean": round(sum(all_gaps) / len(all_gaps), 6) if all_gaps else None,
            "tech_frequency": tech_freq,
            "tech_frequency_positive_only": tech_freq_positive,
            "unique_techs_matched": len(set(tech_seq_all) - {"UNKNOWN"}),
            "unknown_count": tech_seq_all.count("UNKNOWN"),
        },
        "tech_sequence_all": tech_seq_all,
        "tech_sequence_library": lib,
        "snapshots": results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n========== 汇总 ==========")
    print(f"诊断结果已写入: {OUTPUT_FILE}")
    print(f"分数范围: [{output['summary']['score_min']}, {output['summary']['score_max']}]  均值: {output['summary']['score_mean']}")
    print(f"1st-2nd 差距范围: [{output['summary']['gap_min']}, {output['summary']['gap_max']}]  均值: {output['summary']['gap_mean']}")
    print(f"技术码分布: {tech_freq}")
    print(f"真阳性技术码分布: {tech_freq_positive}")
    print(f"UNKNOWN 数量: {output['summary']['unknown_count']}")
    print(f"序列库: {lib}")


if __name__ == "__main__":
    main()
