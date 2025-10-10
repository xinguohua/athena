import platform
import numpy as np
import torch
import yaml

from datahandlers import get_handler
from embedders import get_embedder_by_name
from process.classfy import get_classfy

# ---------------- 配置参数 ----------------
CONFIG_PATH = "config.yaml"
# DATASET_NAME = "atlas"          # 可切换数据集
DATASET_NAME = "cadets"          # 可切换数据集
EMBEDDER_NAME = "roland"    # 嵌入器
# EMBEDDER_NAME = "prographer"
CLASSIFY_NAME = "prographer"     # 训练器
# EMBEDDER_NAME = "unicorn"    # 嵌入器
# CLASSIFY_NAME = "unicorn"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_tsne_embeddings(arr: np.ndarray, benign_start: int, benign_end: int, annotate: bool = False, save_path: str = "snapshot_embeddings_tsne.png"):
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
        for i in range(T):
            plt.annotate(str(i), (xs[i], ys[i]), fontsize=8, alpha=0.7)
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

    # 仅 t-SNE 可视化
    try:
        plot_tsne_embeddings(
            snapshot_embeddings,
            handler.benign_idx_start,
            handler.benign_idx_end,
            annotate=False,
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