"""
从 ATT&CK 技术描述中数据驱动地推导系统级类型标签。

方法：
1. 从 691 项技术的结构化三元组中收集所有宾语短语
2. 用 Sentence-BERT 将宾语编码为语义向量
3. Ward 层次聚类，将语义相近的宾语分组
4. 每个簇即为一种系统实体类别，用簇内最短成员命名

运行：
    python -m process.derive_type_labels
"""
import json
import os
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ============================================================
# 配置
# ============================================================

TRIPLES_PATH = os.path.join(os.path.dirname(__file__), "data/technique_triples_raw.json")
MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
N_CLUSTERS = 17


def main():
    print("=" * 70)
    print("数据驱动推导系统级类型标签")
    print("=" * 70)
    print(f"输入: {TRIPLES_PATH}")
    print(f"模型: {MODEL_NAME}")
    print(f"簇数: {N_CLUSTERS}")
    print()

    # ----------------------------------------------------------
    # 第一步：收集宾语
    # ----------------------------------------------------------
    with open(TRIPLES_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    obj_tids = defaultdict(set)
    for tid, triples in raw.items():
        for t in triples:
            obj_tids[t["object"]].add(tid)

    unique_objects = list(obj_tids.keys())
    print(f"[Step 1] 收集宾语: {sum(len(v) for v in raw.values())} 个三元组 → {len(unique_objects)} 个不同宾语")

    # ----------------------------------------------------------
    # 第二步：Sentence-BERT 嵌入
    # ----------------------------------------------------------
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    print(f"[Step 2] 编码 {len(unique_objects)} 个宾语...")
    embeddings = model.encode(unique_objects, normalize_embeddings=True, show_progress_bar=True)

    # ----------------------------------------------------------
    # 第三步：Ward 层次聚类
    # ----------------------------------------------------------
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist

    print(f"[Step 3] 计算余弦距离 + Ward 层次聚类...")
    distances = pdist(embeddings, metric='cosine')
    Z = linkage(distances, method='ward')
    labels = fcluster(Z, t=N_CLUSTERS, criterion='maxclust')
    print(f"[Step 3] 得到 {len(set(labels))} 个簇")

    # ----------------------------------------------------------
    # 第四步：输出结果
    # ----------------------------------------------------------
    clusters = defaultdict(list)
    for obj, label in zip(unique_objects, labels):
        clusters[int(label)].append(obj)

    print(f"\n{'='*70}")
    print(f"聚类结果：{len(clusters)} 个簇")
    print(f"{'='*70}")

    for label in sorted(clusters.keys(), key=lambda l: -len(clusters[l])):
        members = clusters[label]
        short = sorted(members, key=len)[:5]
        techs = set()
        for m in members:
            techs.update(obj_tids[m])

        print(f"\n簇{label:2d} | {len(members):>3d}个宾语 | {len(techs):>3d}个技术")
        print(f"  代表成员:")
        for s in short:
            print(f"    {s[:90]}")

    print(f"\n{'='*70}")
    print(f"以上 {N_CLUSTERS} 个簇由 Sentence-BERT 语义聚类自动产出，")
    print(f"不依赖任何预定义类别。可通过运行本脚本独立复现。")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
