import platform
import numpy as np
import torch
import yaml

from datahandlers import get_handler
from embedders import get_embedder_by_name
from process.classfy import get_classfy

# ---------------- 配置参数 ----------------
CONFIG_PATH = "config.yaml"
DATASET_NAME = "atlas"          # 可切换数据集
# DATASET_NAME = "cadets"          # 可切换数据集
EMBEDDER_NAME = "roland"    # 嵌入器
# EMBEDDER_NAME = "prographer"
CLASSIFY_NAME = "prographer"     # 训练器
# EMBEDDER_NAME = "unicorn"    # 嵌入器
# CLASSIFY_NAME = "unicorn"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_deviation(arr: np.ndarray, benign_start: int, benign_end: int, metric: str = "cosine") -> np.ndarray:
    """计算每个快照相对良性中心的偏离（cosine 或 L2）。返回 shape (T,)。"""
    if arr is None or arr.size == 0:
        return np.zeros(0, dtype=np.float32)
    X = np.asarray(arr, dtype=np.float32)
    T = X.shape[0]
    lo, hi = min(benign_start, benign_end), max(benign_start, benign_end)
    lo = max(0, min(lo, T - 1))
    hi = max(0, min(hi, T - 1))
    benign = X[lo:hi + 1] if lo <= hi else X
    center = benign.mean(axis=0)
    metric_l = (metric or "cosine").lower()
    if metric_l == "l2":
        dev = np.linalg.norm(X - center.reshape(1, -1), axis=1)
    else:
        eps = 1e-12
        center = center / (np.linalg.norm(center) + eps)
        norms = np.linalg.norm(X, axis=1, keepdims=True) + eps
        unit_x = X / norms
        cos = (unit_x @ center.reshape(-1, 1)).reshape(-1)
        dev = 1.0 - cos
    return dev.astype(np.float32)


def plot_tsne_embeddings(
    arr: np.ndarray,
    benign_start: int,
    benign_end: int,
    annotate: bool = True,
    mode: str = "topk",           # "topk" 或 "all"
    top_k: int = 10,
    group: str = "per-group",      # "per-group" 或 "global"
    metric: str = "cosine",        # "cosine" 或 "l2"
    save_path: str = "snapshot_embeddings_tsne.png",
):
    """仅使用 t-SNE 可视化快照嵌入二维分布，标记良性区间。

    Args:
        arr: (T, D) 的快照嵌入矩阵
        benign_start: 良性起始索引（包含）
        benign_end: 良性结束索引（包含）
        annotate: 是否在点旁标注索引
        save_path: 图片保存路径
    """
    if arr is None or getattr(arr, "size", 0) == 0:
        print("[Viz] 空的快照嵌入，跳过 t-SNE 可视化。")
        return
    try:
        from sklearn.manifold import TSNE  # type: ignore
    except Exception as ex:
        print(f"[Viz] 未安装 scikit-learn，无法运行 t-SNE：{ex}")
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as ex:
        print(f"[Viz] 未安装 matplotlib，无法绘图：{ex}")
        return

    X = np.asarray(arr, dtype=np.float32)
    T = X.shape[0]
    lo, hi = min(benign_start, benign_end), max(benign_start, benign_end)
    lo = max(0, min(lo, T - 1))
    hi = max(0, min(hi, T - 1))

    perplexity = int(min(30, max(5, T // 3)))
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, init="pca", random_state=42)
    coords = tsne.fit_transform(X)

    xs, ys = coords[:, 0], coords[:, 1]
    mask_benign = np.zeros(T, dtype=bool)
    if lo <= hi:
        mask_benign[lo:hi + 1] = True

    plt.figure(figsize=(8, 6))
    plt.scatter(xs[mask_benign], ys[mask_benign], c="#2ca02c", label=f"Benign [{lo}-{hi}]", s=40, alpha=0.85, edgecolors="white")
    if (~mask_benign).any():
        plt.scatter(xs[~mask_benign], ys[~mask_benign], c="#d62728", label="Others", s=40, alpha=0.85, edgecolors="white")
    if annotate:
        indices_to_annotate: list[int] = []
        labels_for_indices: dict[int, str] = {}
        mode_l = (mode or "topk").lower()
        group_l = (group or "per-group").lower()
        metric_l = (metric or "cosine").lower()

        if mode_l == "all":
            b_all = list(np.where(mask_benign)[0])
            m_all = list(np.where(~mask_benign)[0])
            for r, idx in enumerate(b_all):
                indices_to_annotate.append(int(idx))
                labels_for_indices[int(idx)] = f"B{r}"
            for r, idx in enumerate(m_all):
                indices_to_annotate.append(int(idx))
                labels_for_indices[int(idx)] = f"M{r}"
        else:
            dev = _compute_deviation(X, lo, hi, metric=metric_l)
            if group_l == "per-group":
                b_all = np.where(mask_benign)[0]
                m_all = np.where(~mask_benign)[0]
                k_b = int(min(top_k, len(b_all)))
                k_m = int(min(top_k, len(m_all)))
                if k_b > 0:
                    order_b = b_all[np.argsort(-dev[b_all])[:k_b]]
                    for r, idx in enumerate(order_b):
                        indices_to_annotate.append(int(idx))
                        labels_for_indices[int(idx)] = f"B{r}"
                if k_m > 0:
                    order_m = m_all[np.argsort(-dev[m_all])[:k_m]]
                    for r, idx in enumerate(order_m):
                        indices_to_annotate.append(int(idx))
                        labels_for_indices[int(idx)] = f"M{r}"
            else:
                k = int(min(max(1, top_k), T))
                order = np.argsort(-dev)[:k]
                b_count = 0
                m_count = 0
                for idx in order:
                    idx = int(idx)
                    indices_to_annotate.append(idx)
                    if mask_benign[idx]:
                        labels_for_indices[idx] = f"B{b_count}"
                        b_count += 1
                    else:
                        labels_for_indices[idx] = f"M{m_count}"
                        m_count += 1

        for idx in indices_to_annotate:
            lbl = labels_for_indices.get(idx, str(idx))
            color = "#2ca02c" if mask_benign[idx] else "#d62728"
            plt.annotate(lbl, (xs[idx], ys[idx]), fontsize=14, fontweight="bold", color=color, alpha=0.95)
    plt.title("Snapshot Embeddings (t-SNE 2D)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(loc="best")
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=150)
        print(f"[Viz] t-SNE 图已保存: {save_path}")
    except Exception as ex:
        print(f"[Viz] 保存 t-SNE 图片失败：{ex}")
    finally:
        plt.close()


def load_config(path: str) -> dict:
    """加载 YAML 配置，并根据系统环境选择配置分支"""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    system = platform.system().lower()
    return config["local"] if "windows" in system else config["remote"]


def prepare_data(path_map: dict):
    """加载数据并生成快照"""
    handler = get_handler(DATASET_NAME, True, path_map)
    handler.load()
    handler.build_graph()
    return handler


def build_embeddings(handler):
    """构建并训练嵌入器，仅使用良性快照训练"""
    embedder_cls = get_embedder_by_name(EMBEDDER_NAME)
    if EMBEDDER_NAME.lower() == "roland":
        benign_range = range(handler.benign_idx_start, handler.benign_idx_end + 1)
        print(f"[Train] 仅使用良性快照训练编码器: {handler.benign_idx_start}~{handler.benign_idx_end}")
        embedder = embedder_cls(handler.snapshots, train_indices=benign_range)
    else:
        embedder = embedder_cls(handler.snapshots)
    embedder.train()
    snapshot_embeddings = embedder.get_snapshot_embeddings()
    print("\n--- Encoder 过程完成 ---")
    print(f"[嵌入] 快照嵌入序列: {snapshot_embeddings.shape}")
    print(snapshot_embeddings)
    return snapshot_embeddings


def main():
    env_config = load_config(CONFIG_PATH)
    path_map = env_config["path_map"]

    # 数据准备
    handler = prepare_data(path_map)

    # 嵌入训练
    snapshot_embeddings = build_embeddings(handler)

    # 仅 t-SNE 可视化（Top-K 标注）
    try:
        plot_tsne_embeddings(
            snapshot_embeddings,
            handler.benign_idx_start,
            handler.benign_idx_end,
            annotate=True,
            mode="topk",       # 只标注 Top-K
            top_k=10,          # 每组取 K 个
            group="per-group", # 每组独立 Top-K
            metric="cosine",  # 用 cosine 偏离
            save_path="snapshot_embeddings_tsne.png",
        )
    except Exception as ex:
        print(f"[Viz] t-SNE 可视化失败：{ex}")

    # 模型训练
    benign_embeddings = snapshot_embeddings[handler.benign_idx_start:handler.benign_idx_end + 1]
    classify = get_classfy(CLASSIFY_NAME)
    classify.train(benign_embeddings)

if __name__ == "__main__":
    main()