"""
本地测试语义匹配 — 不需要远程数据集，只需要三元组JSON和sentence-transformers。

用法：
    python -m process.local_test_matching

可修改 QUERY 和 TRIPLES_SOURCE 快速迭代测试。
"""
import json
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ============================================================
# 配置
# ============================================================

# 技术侧描述来源：raw（原始三元组）或 transformed（转换后三元组）
TRIPLES_SOURCE = "raw"  # "raw" 或 "transformed"

# 模拟日志侧查询（从远程 debug_matching 的输出复制过来）
# 这是 CADETS 数据集中的进程注入快照
QUERY = "process writes shared library. process sends network connection. process reads file. process writes file. process receives network connection"

# 期望匹配到的技术（用于验证）
EXPECTED = ["T1055", "T1055.001", "T1055.011", "T1055.012"]

MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
TOP_K = 20


# ============================================================
# 主逻辑
# ============================================================

def load_technique_descriptions(source: str):
    """加载技术描述库。"""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if source == "raw":
        path = os.path.join(data_dir, "technique_triples_raw.json")
    else:
        path = os.path.join(data_dir, "technique_triples_transformed.json")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tech_ids = []
    tech_texts = []
    for tid, triples in data.items():
        if not triples:
            continue
        desc = ". ".join(f"{t['subject']} {t['verb']} {t['object']}" for t in triples)
        tech_ids.append(tid)
        tech_texts.append(desc)

    return tech_ids, tech_texts


def main():
    print(f"技术侧来源: {TRIPLES_SOURCE}")
    print(f"查询: {QUERY}")
    print(f"期望匹配: {EXPECTED}\n")

    # 加载技术描述
    tech_ids, tech_texts = load_technique_descriptions(TRIPLES_SOURCE)
    print(f"技术数: {len(tech_ids)}")

    # 加载模型
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)

    # 编码
    print("编码技术描述...")
    tech_embs = model.encode(tech_texts, normalize_embeddings=True, show_progress_bar=False)
    q_emb = model.encode([QUERY], normalize_embeddings=True, show_progress_bar=False)

    # 余弦相似度
    sims = np.dot(tech_embs, q_emb.T).flatten()
    top_indices = np.argsort(sims)[::-1][:TOP_K]

    # 结果
    print(f"\n{'='*60}")
    print(f"Top {TOP_K} 匹配结果")
    print(f"{'='*60}\n")

    for rank, idx in enumerate(top_indices):
        tid = tech_ids[idx]
        score = sims[idx]
        desc = tech_texts[idx][:200]
        marker = " <<<" if tid in EXPECTED or tid.split('.')[0] in EXPECTED else ""
        print(f"  [{rank+1:2d}] {tid:12s}  similarity={score:.4f}{marker}")
        print(f"       {desc}")
        print()

    # 期望技术的排名
    print(f"{'='*60}")
    print(f"期望技术排名")
    print(f"{'='*60}\n")
    for tid in EXPECTED:
        if tid in tech_ids:
            idx = tech_ids.index(tid)
            rank = int(np.sum(sims > sims[idx]) + 1)
            print(f"  {tid:12s}  similarity={sims[idx]:.4f}  排名={rank}")
        else:
            print(f"  {tid:12s}  不在技术库中")

    # 对比两种来源
    if TRIPLES_SOURCE == "raw":
        print(f"\n提示: 当前用原始三元组。试试改 TRIPLES_SOURCE = 'transformed' 对比效果。")
    else:
        print(f"\n提示: 当前用转换后三元组。试试改 TRIPLES_SOURCE = 'raw' 对比效果。")


if __name__ == "__main__":
    main()
