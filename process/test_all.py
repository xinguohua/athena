import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Optional, List, TYPE_CHECKING
import pickle
import os
import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from process.classfy import get_classfy

# --- 项目模块 ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from process.embedders import get_embedder_by_name

# 仅用于类型提示，避免在运行时引入重型依赖
if TYPE_CHECKING:  # pragma: no cover
    from process.technique_semantic_mapper import TechniqueSemanticMapper


# ========================================================================
# 全局配置
# ========================================================================
# EMBEDDER_NAME = "prographer"
EMBEDDER_NAME = "gcc_dev"
# EMBEDDER_NAME = "gcc"
CLASSIFY_NAME = "topk"
# EMBEDDER_NAME = "unicorn"
# CLASSIFY_NAME = "unicorn"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 二次筛选配置（仅基于“攻击技术序列库”）
SEQ_FILTER = {
    "enable": True,           # 开关：是否启用二次筛选
    "library_path": "technique_sequences.txt",  # 技术序列库文件（每行一条序列，逗号/空白分隔）
    "lcs_min_ratio": 0.6,     # LCS 匹配阈值：LCS_len / len(库内序列) >= 该比例才视为命中
}


# ========================================================================
# 工具函数
# ========================================================================

def cosine(a, b, eps=1e-12):
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return np.dot(a, b) / (na * nb)

def deviation_from_center(vec, center):
    return 1.0 - cosine(vec, center)

def inject_snapshots_deviation(
    embeddings: np.ndarray,
    target_idxs: list[int],
    mode: str = "away",
    alpha: float = 3.0,
    renormalize: bool = True,
    rng_seed: int = 0
):
    """
    同时对多个 snapshot 嵌入施加偏离（方法1: 图级拉开）

    embeddings: (T, D) 原始快照嵌入矩阵
    target_idxs: 要偏离的快照索引列表（如 [71, 73, 80]）
    mode:
      - "away": 沿远离中心的方向拉开
      - "random": 使用随机方向
    alpha: 偏移强度
    renormalize: 是否重新 L2 归一化
    rng_seed: 随机数种子，保证可复现
    """
    np.random.seed(rng_seed)
    T, D = embeddings.shape
    emb = embeddings.copy()
    center = emb.mean(axis=0)

    info_list = []

    for idx in target_idxs:
        if not (0 <= idx < T):
            print(f"⚠️ 跳过非法索引 {idx}")
            continue

        before_dev = deviation_from_center(emb[idx], center)

        if mode == "away":
            direction = emb[idx] - center
            if np.linalg.norm(direction) < 1e-12:
                direction = np.random.randn(D)
        elif mode == "random":
            direction = np.random.randn(D)
        else:
            raise ValueError("mode must be 'away' or 'random'")

        direction = direction / (np.linalg.norm(direction) + 1e-12)
        emb[idx] = emb[idx] + alpha * direction

        if renormalize:
            emb[idx] = emb[idx] / (np.linalg.norm(emb[idx]) + 1e-12)

        after_dev = deviation_from_center(emb[idx], center)

        info_list.append({
            "target_idx": idx,
            "before_dev": float(before_dev),
            "after_dev": float(after_dev),
            "alpha": alpha,
            "mode": mode
        })

    return emb, info_list

def _compute_deviation(arr: np.ndarray, benign_start: int, benign_end: int, metric: str = "cosine") -> np.ndarray:
    """计算每个快照相对良性中心的偏离（cosine 或 L2）。返回 shape (T,)。

    注意：仅用于可视化标注 Top-K 偏离用。
    """
    if arr is None or getattr(arr, "size", 0) == 0:
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


def _compute_step_deviation(arr: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """基于相邻快照的偏离：dev[0]=0，dev[t]=dist(X[t], X[t-1]).

    metric:
      - "l2": 欧氏距离 ||x_t - x_{t-1}||
      - 其他（默认 "cosine"）：1 - cos(x_t, x_{t-1})，内部做 L2 归一化
    """
    if arr is None or getattr(arr, "size", 0) == 0:
        return np.zeros(0, dtype=np.float32)
    X = np.asarray(arr, dtype=np.float32)
    T = X.shape[0]
    if T <= 0:
        return np.zeros(0, dtype=np.float32)
    dev = np.zeros(T, dtype=np.float32)
    if T == 1:
        return dev
    metric_l = (metric or "cosine").lower()
    if metric_l == "l2":
        diffs = X[1:] - X[:-1]
        dev[1:] = np.linalg.norm(diffs, axis=1).astype(np.float32)
    else:
        eps = 1e-12
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)
        cos = np.sum(Xn[1:] * Xn[:-1], axis=1)
        dev[1:] = (1.0 - cos).astype(np.float32)
    return dev


def plot_tsne_embeddings(
    arr: np.ndarray,
    benign_start: int,
    benign_end: int,
    annotate: bool = True,
    mode: str = "topk",            # "topk" 或 "all"
    top_k: int = 10,
    group: str = "per-group",       # "per-group" 或 "global"
    metric: str = "cosine",         # "cosine" 或 "l2"
    save_path: str = "snapshot_embeddings_tsne.png",
    which: str = "all",             # "all" | "benign" | "malicious"
    malicious_start: Optional[int] = None,
    malicious_end: Optional[int] = None,
    selected_mal_ids_in_slice: Optional[List[int]] = None,  # 仅当 which="malicious" 时生效，片段内索引（0 开始）
):
    """仅使用 t-SNE 可视化快照嵌入二维分布，标记良性区间。"""
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
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=1000,
        init="pca",
        random_state=42,
    )
    coords = tsne.fit_transform(X)

    xs, ys = coords[:, 0], coords[:, 1]
    mask_benign = np.zeros(T, dtype=bool)
    if lo <= hi:
        mask_benign[lo:hi + 1] = True

    # 恶意掩码：优先使用显式传入的恶意区间；否则使用良性交集的补集
    if malicious_start is not None and malicious_end is not None:
        m_lo, m_hi = min(malicious_start, malicious_end), max(malicious_start, malicious_end)
        m_lo = max(0, min(m_lo, T - 1))
        m_hi = max(0, min(m_hi, T - 1))
        mask_mal = np.zeros(T, dtype=bool)
        if m_lo <= m_hi:
            mask_mal[m_lo:m_hi + 1] = True
    else:
        mask_mal = ~mask_benign

    which_l = (which or "all").lower()

    # 如果只画恶性且给定了片段内恶意索引，则仅绘制这些点
    mask_mal_plot = mask_mal
    if which_l == "malicious" and selected_mal_ids_in_slice is not None:
        if malicious_start is None:
            print("[Viz] 提供了恶意片段内索引，但缺少 malicious_start/end。将忽略该选择，绘制全部恶意点。")
        else:
            mal_sel_mask = np.zeros(T, dtype=bool)
            for i in selected_mal_ids_in_slice:
                try:
                    gi = int(malicious_start + int(i))
                except Exception:
                    continue
                if 0 <= gi < T:
                    mal_sel_mask[gi] = True
            mask_mal_plot = mask_mal & mal_sel_mask

    # 若提供了 selected_mal_ids_in_slice，则构建全局索引 -> 片段内索引 的映射，用于标注显示 M<slice_id>
    slice_id_map: Dict[int, int] = {}
    if selected_mal_ids_in_slice is not None and malicious_start is not None and which_l in ("malicious", "all"):
        for sid in selected_mal_ids_in_slice:
            try:
                gi = int(malicious_start + int(sid))
            except Exception:
                continue
            if 0 <= gi < T:
                slice_id_map[gi] = int(sid)

    plt.figure(figsize=(8, 6))
    drew_any = False
    if which_l in ("all", "benign") and mask_benign.any():
        plt.scatter(
            xs[mask_benign], ys[mask_benign], c="#2ca02c",
            label=f"Benign [{lo}-{hi}]", s=40, alpha=0.85, edgecolors="white"
        )
        drew_any = True
    if which_l in ("all", "malicious") and (mask_mal_plot if which_l == "malicious" else mask_mal).any():
        plt.scatter(
            xs[mask_mal_plot] if which_l == "malicious" else xs[mask_mal],
            ys[mask_mal_plot] if which_l == "malicious" else ys[mask_mal],
            c="#d62728",
            label="Malicious", s=40, alpha=0.85, edgecolors="white"
        )
        drew_any = True

    if not drew_any:
        print("[Viz] 所选类别无点可画，跳过绘图。")
        return

    if annotate:
        indices_to_annotate: List[int] = []
        labels_for_indices: Dict[int, str] = {}
        mode_l = (mode or "topk").lower()
        group_l = (group or "per-group").lower()
        metric_l = (metric or "cosine").lower()

        if which_l == "benign":
            target_mask = mask_benign
            if mode_l == "all":
                b_all = list(np.where(target_mask)[0])
                for r, idx in enumerate(b_all):
                    idx = int(idx)
                    indices_to_annotate.append(idx)
                    labels_for_indices[idx] = f"B{r}"
            else:
                dev = _compute_deviation(X, lo, hi, metric=metric_l)
                b_all = np.where(target_mask)[0]
                k_b = int(min(top_k, len(b_all)))
                if k_b > 0:
                    order_b = b_all[np.argsort(-dev[b_all])[:k_b]]
                    for r, idx in enumerate(order_b):
                        idx = int(idx)
                        indices_to_annotate.append(idx)
                        labels_for_indices[idx] = f"B{r}"
        elif which_l == "malicious":
            target_mask = mask_mal_plot
            if mode_l == "all":
                m_all = list(np.where(target_mask)[0])
                for r, idx in enumerate(m_all):
                    idx = int(idx)
                    indices_to_annotate.append(idx)
                    # 优先使用提供的片段内索引映射；否则若提供恶意起点，则用片段内索引；再否则退化为顺序编号
                    if idx in slice_id_map:
                        labels_for_indices[idx] = f"M{slice_id_map[idx]}"
                    elif malicious_start is not None:
                        labels_for_indices[idx] = f"M{idx - int(malicious_start)}"
                    else:
                        labels_for_indices[idx] = f"M{r}"
            else:
                dev = _compute_deviation(X, lo, hi, metric=metric_l)
                m_all = np.where(target_mask)[0]
                k_m = int(min(top_k, len(m_all)))
                if k_m > 0:
                    order_m = m_all[np.argsort(-dev[m_all])[:k_m]]
                    for r, idx in enumerate(order_m):
                        idx = int(idx)
                        indices_to_annotate.append(idx)
                        if idx in slice_id_map:
                            labels_for_indices[idx] = f"M{slice_id_map[idx]}"
                        elif malicious_start is not None:
                            labels_for_indices[idx] = f"M{idx - int(malicious_start)}"
                        else:
                            labels_for_indices[idx] = f"M{r}"
        else:  # which_l == "all"
            if mode_l == "all":
                b_all = list(np.where(mask_benign)[0])
                m_all = list(np.where(mask_mal)[0])
                for r, idx in enumerate(b_all):
                    indices_to_annotate.append(int(idx))
                    labels_for_indices[int(idx)] = f"B{r}"
                for r, idx in enumerate(m_all):
                    indices_to_annotate.append(int(idx))
                    if int(idx) in slice_id_map:
                        labels_for_indices[int(idx)] = f"M{slice_id_map[int(idx)]}"
                    elif malicious_start is not None:
                        labels_for_indices[int(idx)] = f"M{int(idx) - int(malicious_start)}"
                    else:
                        labels_for_indices[int(idx)] = f"M{r}"
            else:
                dev = _compute_deviation(X, lo, hi, metric=metric_l)
                if group_l == "per-group":
                    b_all = np.where(mask_benign)[0]
                    m_all = np.where(mask_mal)[0]
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
                            if int(idx) in slice_id_map:
                                labels_for_indices[int(idx)] = f"M{slice_id_map[int(idx)]}"
                            elif malicious_start is not None:
                                labels_for_indices[int(idx)] = f"M{int(idx) - int(malicious_start)}"
                            else:
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
                        elif mask_mal[idx]:
                            if idx in slice_id_map:
                                labels_for_indices[idx] = f"M{slice_id_map[idx]}"
                            elif malicious_start is not None:
                                labels_for_indices[idx] = f"M{idx - int(malicious_start)}"
                            else:
                                labels_for_indices[idx] = f"M{m_count}"
                            m_count += 1

        # 直接在点上标注（不做重叠避让）
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


def plot_deviation_changes(
    arr: np.ndarray,
    benign_start: int,
    benign_end: int,
    metric: str = "cosine",           # "cosine" 或 "l2"
    smooth_k: int = 1,                 # 移动平均窗口，>1 时启用平滑
    save_path: str = "snapshot_deviation_curve.png",
    malicious_start: Optional[int] = None,
    malicious_end: Optional[int] = None,
    annotate_top_k: int = 10,          # 标注偏离最大的前 K 个点（当 annotate_mode="topk" 时生效）
    annotate_mode: str = "topk",      # "none" | "topk" | "all"
    which: str = "all",               # "all" | "benign" | "malicious"
    # 仅绘制某个子区间（例如只看恶意段）：提供 focus_start/focus_end 即可
    focus_start: Optional[int] = None,
    focus_end: Optional[int] = None,
    # 是否在子区间上重算 μ/σ（默认 False：仍以良性段为基准）
    recompute_stats: bool = False,
    # 偏离的定义："step" 基于相邻快照，"center" 相对良性中心
    deviation_mode: str = "step",
):
    """可视化每个快照的偏离变化曲线。

        - 偏离定义：
            * deviation_mode="center" 时：相对“当前绘制范围（focus）”内的中心向量
            * deviation_mode="step" 时：相邻快照变化量（默认）
            * 距离度量：cosine 或 L2
        - 视觉元素：
            * 绿色区域标出良性区间 [benign_start, benign_end]
            * 若提供恶意范围，则淡红色区域标出 [malicious_start, malicious_end]
            * 用均值±σ 阈值线辅助判断异常（默认基于良性段，或根据 recompute_stats 改为 focus）
            * 标注 Top-K 偏离点（或全部/不标注，受 annotate_mode 控制）
        """
    if arr is None or getattr(arr, "size", 0) == 0:
        print("[Viz] 空的快照嵌入，跳过偏离曲线可视化。")
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

    # 选择绘制的子区间（focus）。若显式提供 focus_* 则优先；否则根据 which 推断。
    if focus_start is not None and focus_end is not None:
        f_lo, f_hi = min(int(focus_start), int(focus_end)), max(int(focus_start), int(focus_end))
        f_lo = max(0, min(f_lo, T - 1))
        f_hi = max(0, min(f_hi, T - 1))
        focus_mask = np.zeros(T, dtype=bool)
        if f_lo <= f_hi:
            focus_mask[f_lo:f_hi + 1] = True
    else:
        which_l = (which or "all").lower()
        if which_l == "benign":
            f_lo, f_hi = lo, hi
        elif which_l == "malicious" and (malicious_start is not None and malicious_end is not None):
            f_lo, f_hi = min(int(malicious_start), int(malicious_end)), max(int(malicious_start), int(malicious_end))
            f_lo = max(0, min(f_lo, T - 1))
            f_hi = max(0, min(f_hi, T - 1))
        else:
            f_lo, f_hi = 0, T - 1
        focus_mask = np.zeros(T, dtype=bool)
        if f_lo <= f_hi:
            focus_mask[f_lo:f_hi + 1] = True

    # 计算偏离：center 基于 focus 范围的中心；step 基于相邻快照
    if (deviation_mode or "step").lower() == "center":
        # 若 focus 无效（f_lo>f_hi），退化为使用全局范围
        _c_lo, _c_hi = (f_lo, f_hi) if f_lo <= f_hi else (0, T - 1)
        dev = _compute_deviation(X, _c_lo, _c_hi, metric=(metric or "cosine").lower())
    else:
        dev = _compute_step_deviation(X, metric=(metric or "cosine").lower())

    # 平滑（可选）
    if smooth_k and smooth_k > 1:
        k = int(smooth_k)
        k = max(1, min(k, T))
        if k > 1:
            kernel = np.ones(k, dtype=np.float32) / float(k)
            dev_sm = np.convolve(dev, kernel, mode="same").astype(np.float32)
        else:
            dev_sm = dev
    else:
        dev_sm = dev

    # 阈值参考：默认用良性区；若要求在子区间上重算，则改用子区间
    if recompute_stats:
        ref_seg = dev[focus_mask]
    else:
        ref_seg = dev[lo: hi + 1] if lo <= hi else dev
    mu = float(np.mean(ref_seg)) if ref_seg.size > 0 else float(np.mean(dev))
    sd = float(np.std(ref_seg)) if ref_seg.size > 0 else float(np.std(dev))
    thr1 = mu + sd
    thr2 = mu + 2 * sd

    xs = np.arange(T)
    xs_plot = xs[focus_mask]
    dev_plot = dev_sm[focus_mask]
    plt.figure(figsize=(10, 4.5))
    plt.plot(xs_plot, dev_plot, color="#1f77b4", lw=2.0, label=f"Deviation ({metric})")

    # 良性区高亮（与 focus 取交集提高可读性）
    if lo <= hi:
        b_lo = max(lo, f_lo)
        b_hi = min(hi, f_hi)
        if b_lo <= b_hi:
            plt.axvspan(b_lo, b_hi, color="#2ca02c", alpha=0.10, label=f"Benign [{lo}-{hi}]")

    # 恶意区高亮（可选）
    if malicious_start is not None and malicious_end is not None:
        m_lo, m_hi = min(malicious_start, malicious_end), max(malicious_start, malicious_end)
        m_lo = max(0, min(m_lo, T - 1))
        m_hi = max(0, min(m_hi, T - 1))
        if m_lo <= m_hi:
            mm_lo = max(m_lo, f_lo)
            mm_hi = min(m_hi, f_hi)
            if mm_lo <= mm_hi:
                plt.axvspan(mm_lo, mm_hi, color="#d62728", alpha=0.08, label=f"Malicious [{m_lo}-{m_hi}]")

    # 阈值线
    plt.axhline(thr1, color="#ff7f0e", lw=1.4, ls="--", label="μ + 1σ")
    plt.axhline(thr2, color="#d62728", lw=1.4, ls=":", label="μ + 2σ")

    # 高亮超过 2σ 的点
    over_mask = (dev > thr2) & focus_mask
    if over_mask.any():
        plt.scatter(xs[over_mask], dev_sm[over_mask], c="#d62728", s=28, zorder=3, label="> μ+2σ")

    # 标注（开关：none/topk/all）。优先展示片段内索引：恶意 M<idx - malicious_start> / 良性 B<idx - benign_start>
    mode_l = (annotate_mode or "topk").lower()
    if mode_l != "none":
        cand = np.where(focus_mask)[0]
        if mode_l == "all":
            to_annotate = cand
        else:
            k = int(min(max(0, annotate_top_k), int(focus_mask.sum())))
            to_annotate = cand[np.argsort(-dev[cand])[:k]] if k > 0 else np.array([], dtype=int)
        for idx in to_annotate:
            yi = float(dev_sm[int(idx)])
            # 计算片段内显示标签
            lbl = None
            if malicious_start is not None and malicious_end is not None:
                _mlo, _mhi = min(int(malicious_start), int(malicious_end)), max(int(malicious_start), int(malicious_end))
                if _mlo <= int(idx) <= _mhi:
                    lbl = f"M{int(idx) - int(malicious_start)}"
            if lbl is None and (lo <= int(idx) <= hi):
                lbl = f"B{int(idx) - int(lo)}"
            if lbl is None:
                lbl = f"{int(idx)}"
            plt.annotate(
                lbl,
                (int(idx), yi),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                color="#333333",
            )

    title_suffix = ""
    if focus_start is not None and focus_end is not None:
        title_suffix = f" [focus {f_lo}-{f_hi}]"
    else:
        wl = (which or "all").lower()
        if wl in ("benign", "malicious"):
            title_suffix = f" [{wl}]"
    plt.title(f"Snapshot Deviation Curve{title_suffix}")
    plt.xlabel("Snapshot Index")
    plt.ylabel(f"Deviation ({metric})")
    plt.legend(loc="best")
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=150)
        print(f"[Viz] 偏离曲线图已保存: {save_path}")
        # 简要统计
        over1 = int((dev > thr1).sum())
        over2 = int((dev > thr2).sum())
        print(f"[Viz] 超过 μ+1σ 的快照: {over1} / {T}; 超过 μ+2σ 的快照: {over2} / {T}")
    except Exception as ex:
        print(f"[Viz] 保存偏离曲线失败：{ex}")
    finally:
        plt.close()


def plot_deviation(
    arr: np.ndarray,
    benign_start: int,
    benign_end: int,
    *,
    mode: str = "all",                 # all | benign | malicious | range
    malicious_start: Optional[int] = None,
    malicious_end: Optional[int] = None,
    range_start: Optional[int] = None,
    range_end: Optional[int] = None,
    metric: str = "cosine",
    smooth_k: int = 5,
    save_path: str = "snapshot_deviation_curve.png",
    annotate_top_k: int = 10,
    annotate_mode: str = "topk",
    recompute_stats: bool = False,
    deviation_mode: str = "step",
):
    """简化版偏离曲线接口：
    - mode=all：全段
    - mode=benign：只看良性区间
    - mode=malicious：只看恶意区间（需提供 malicious_start/end）
    - mode=range：只看任意区间（需提供 range_start/end）
    其他参数直接透传给底层绘制函数。
    """
    mode_l = (mode or "all").lower()
    if mode_l == "benign":
        return plot_deviation_changes(
            arr,
            benign_start,
            benign_end,
            metric=metric,
            smooth_k=smooth_k,
            save_path=save_path,
            malicious_start=malicious_start,
            malicious_end=malicious_end,
            annotate_top_k=annotate_top_k,
            annotate_mode=annotate_mode,
            which="benign",
            recompute_stats=recompute_stats,
            deviation_mode=deviation_mode,
        )
    elif mode_l == "malicious":
        return plot_deviation_changes(
            arr,
            benign_start,
            benign_end,
            metric=metric,
            smooth_k=smooth_k,
            save_path=save_path,
            malicious_start=malicious_start,
            malicious_end=malicious_end,
            annotate_top_k=annotate_top_k,
            annotate_mode=annotate_mode,
            which="malicious",
            recompute_stats=recompute_stats,
            deviation_mode=deviation_mode,
        )
    elif mode_l == "range":
        return plot_deviation_changes(
            arr,
            benign_start,
            benign_end,
            metric=metric,
            smooth_k=smooth_k,
            save_path=save_path,
            malicious_start=malicious_start,
            malicious_end=malicious_end,
            annotate_top_k=annotate_top_k,
            annotate_mode=annotate_mode,
            focus_start=range_start,
            focus_end=range_end,
            recompute_stats=recompute_stats,
            deviation_mode=deviation_mode,
        )
    else:  # all
        return plot_deviation_changes(
            arr,
            benign_start,
            benign_end,
            metric=metric,
            smooth_k=smooth_k,
            save_path=save_path,
            malicious_start=malicious_start,
            malicious_end=malicious_end,
            annotate_top_k=annotate_top_k,
            annotate_mode=annotate_mode,
            which="all",
            recompute_stats=recompute_stats,
            deviation_mode=deviation_mode,
        )
def save_snapshot_nodes(all_snapshots, output_file: Path = Path("test_snapshot.txt")) -> Optional[Path]:
    """保存快照节点信息到文件"""
    print(f"[INFO] 保存快照节点信息到: {output_file}")
    try:
        with output_file.open("w", encoding="utf-8") as f:
            f.write("=== 测试快照节点详情报告 ===\n")
            f.write(f"总快照数: {len(all_snapshots)}\n")
            f.write("=" * 60 + "\n\n")

            for i, snapshot in enumerate(all_snapshots):
                f.write(f"快照 {i}:\n")
                f.write(f"  节点总数: {len(snapshot.vs)}\n")
                f.write(f"  边总数: {len(snapshot.es)}\n")
                f.write("  节点详情:\n")

                node_type_count = defaultdict(int)
                malicious_count = sum(v.attributes().get("label", 0) == 1 for v in snapshot.vs)

                for v in snapshot.vs:
                    attrs = v.attributes()
                    node_name = attrs.get("name", "UNKNOWN")
                    node_type = attrs.get("type", "UNKNOWN")
                    label = attrs.get("label", 0)

                    # 更新类型统计
                    node_type_count[node_type] += 1

                    # 状态
                    status = "🔴恶意" if label == 1 else "🟢正常"

                    # 除了 name/label，其余属性全打印出来
                    extra_attrs = {k: v for k, v in attrs.items() if k not in ("name", "label", "type")}
                    extra_str = " | ".join(f"{k}:{v}" for k, v in extra_attrs.items())

                    f.write(f"    {node_name} | 类型:{node_type} | 状态:{status}")
                    if extra_str:
                        f.write(" | " + extra_str)
                    f.write("\n")

                f.write(f"  恶意节点数: {malicious_count}/{len(snapshot.vs)}\n")
                f.write("  节点类型分布:\n")
                for t, c in sorted(node_type_count.items()):
                    f.write(f"      {t}: {c}个\n")
                f.write("\n" + "-" * 50 + "\n\n")

        print(f"[INFO] 快照信息已写入 {output_file}")
        return output_file
    except Exception as e:
        print(f"[ERROR] 保存快照失败: {e}")
        return None


def get_true_labels(snapshots) -> np.ndarray:
    """提取快照真实标签"""
    return np.array([int(any(v["label"] == 1 for v in s.vs)) for s in snapshots])

def print_debug_info(all_snapshots, eval_true, eval_pred, eval_start_idx):
    """
    详细打印TP、FP、FN、TN快照的调试信息，显示导致分类的具体节点。
    """
    print("\n" + "="*70)
    print(" 🔍 详细调试信息 (TP / FP / FN / TN 完整分析)")
    print("="*70)

    # 分类收集各种情况的快照索引
    tp_indices = []  # 真阳性：真实恶意 + 预测恶意
    fp_indices = []  # 假阳性：真实良性 + 预测恶意
    fn_indices = []  # 假阴性：真实恶意 + 预测良性
    tn_indices = []  # 真阴性：真实良性 + 预测良性

    for i in range(len(eval_true)):
        snapshot_idx = i + eval_start_idx
        true_label = eval_true[i]
        pred_label = eval_pred[i]

        if true_label == 1 and pred_label == 1:
            tp_indices.append(snapshot_idx)
        elif true_label == 0 and pred_label == 1:
            fp_indices.append(snapshot_idx)
        elif true_label == 1 and pred_label == 0:
            fn_indices.append(snapshot_idx)
        else:  # true_label == 0 and pred_label == 0
            tn_indices.append(snapshot_idx)

    # 打印统计概览
    print("\n📊 快照分类统计:")
    print(f"  ✅ 真阳性 (TP): {len(tp_indices)} 个快照")
    print(f"  ❌ 假阳性 (FP): {len(fp_indices)} 个快照")
    print(f"  ⚠️  假阴性 (FN): {len(fn_indices)} 个快照")
    print(f"  ✓  真阴性 (TN): {len(tn_indices)} 个快照")

    # === 详细分析 TP 快照 ===
    if tp_indices:
        print("\n" + "="*50)
        print("✅ 真阳性 (TP) 快照详细分析 - 正确检测到的恶意快照")
        print("="*50)
        for snapshot_idx in tp_indices:
            snapshot = all_snapshots[snapshot_idx]
            print(f"\n🎯 快照 {snapshot_idx}:")

            # 分析节点类型
            malicious_nodes = []
            benign_nodes = []
            node_types_count = {}

            for v in snapshot.vs:
                node_name = v['name']
                node_type = v.attributes().get('type_name', 'UNKNOWN')
                frequency = v.attributes().get('frequency', 'UNKNOWN')
                node_types_count[node_type] = node_types_count.get(node_type, 0) + 1

                if v.attributes().get('label') == 1:
                    malicious_nodes.append(f"{node_name}({node_type})【{frequency}】")
                else:
                    benign_nodes.append(f"{node_name}({node_type})")

            print(f"  📈 总节点数: {len(snapshot.vs)}, 边数: {len(snapshot.es)}")
            print(f"  🔴 恶意节点 ({len(malicious_nodes)}个):")
            if malicious_nodes:
                malicious_str = ', '.join(malicious_nodes[:10])  # 最多显示10个
                if len(malicious_nodes) > 10:
                    malicious_str += f" ... (+{len(malicious_nodes)-10}个更多)"
                print(f"      {malicious_str}")

            print(f"  📊 节点类型分布: {dict(sorted(node_types_count.items()))}")

    # === 详细分析 FP 快照 ===
    if fp_indices:
        print("\n" + "="*50)
        print("❌ 假阳性 (FP) 快照详细分析 - 误报的良性快照")
        print("="*50)
        for snapshot_idx in fp_indices:
            snapshot = all_snapshots[snapshot_idx]
            print(f"\n🚨 快照 {snapshot_idx} (误报):")

            # 分析节点类型分布，寻找误报原因
            node_types_count = {}
            suspicious_patterns = []
            all_nodes = []

            for v in snapshot.vs:
                node_name = v['name']
                node_type = v.attributes().get('type_name', 'UNKNOWN')
                node_types_count[node_type] = node_types_count.get(node_type, 0) + 1
                frequency = v.attributes().get('frequency', 'UNKNOWN')

                all_nodes.append(f"{node_name}({node_type})【{frequency}】")

                # 检查可能导致误报的模式
                if 'SUBJECT_PROCESS' in node_type and any(word in node_name.lower()
                                                          for word in ['system', 'admin', 'service', 'daemon']):
                    suspicious_patterns.append(f"系统进程: {node_name}")
                elif 'NETFLOW' in node_type:
                    suspicious_patterns.append(f"网络流: {node_name}")

            print(f"  📈 总节点数: {len(snapshot.vs)}, 边数: {len(snapshot.es)}")
            print(f"  📊 节点类型分布: {dict(sorted(node_types_count.items()))}")

            if suspicious_patterns:
                print("  ⚡ 可能的误报原因:")
                for pattern in suspicious_patterns[:5]:  # 最多显示5个
                    print(f"      • {pattern}")

            # 显示部分节点名称用于分析
            print("  📝 部分节点 (前10个):")
            sample_nodes = ', '.join(all_nodes[:10])
            if len(all_nodes) > 10:
                sample_nodes += f" ... (+{len(all_nodes)-10}个更多)"
            wrapped_nodes = textwrap.fill(sample_nodes, width=70, initial_indent='      ', subsequent_indent='      ')
            print(wrapped_nodes)

    # === 详细分析 FN 快照 ===
    if fn_indices:
        print("\n" + "="*50)
        print("⚠️ 假阴性 (FN) 快照详细分析 - 漏检的恶意快照")
        print("="*50)
        for snapshot_idx in fn_indices:
            snapshot = all_snapshots[snapshot_idx]
            print(f"\n⚠️  快照 {snapshot_idx} (漏检):")

            # 分析为什么这些恶意节点没被检测到
            malicious_nodes = []
            benign_nodes = []
            node_types_count = {}

            for v in snapshot.vs:
                node_name = v['name']
                node_type = v.attributes().get('type_name', 'UNKNOWN')
                node_types_count[node_type] = node_types_count.get(node_type, 0) + 1
                frequency = v.attributes().get('frequency', 'UNKNOWN')

                if v.attributes().get('label') == 1:
                    malicious_nodes.append(f"{node_name}({node_type})【{frequency}】")
                else:
                    benign_nodes.append(f"{node_name}({node_type})")

            print(f"  📈 总节点数: {len(snapshot.vs)}, 边数: {len(snapshot.es)}")
            print(f"  🔴 被漏检的恶意节点 ({len(malicious_nodes)}个):")
            if malicious_nodes:
                malicious_str = ', '.join(malicious_nodes)
                wrapped_malicious = textwrap.fill(malicious_str, width=70, initial_indent='      ', subsequent_indent='      ')
                print(wrapped_malicious)

            print(f"  📊 节点类型分布: {dict(sorted(node_types_count.items()))}")
            print(f"  💡 可能的漏检原因: 恶意节点比例较低 ({len(malicious_nodes)}/{len(snapshot.vs)} = {len(malicious_nodes)/len(snapshot.vs)*100:.1f}%)")

    # === 简要显示 TN 快照统计 ===
    if tn_indices:
        print("\n" + "="*50)
        print("✓ 真阴性 (TN) 快照统计 - 正确识别的良性快照")
        print("="*50)
        print(f"  ✅ 共有 {len(tn_indices)} 个快照被正确识别为良性")

        # 统计TN快照的节点类型分布
        if len(tn_indices) > 0:
            sample_tn = all_snapshots[tn_indices[0]]  # 取一个样本
            tn_node_types = {}
            for v in sample_tn.vs:
                node_type = v.attributes().get('type_name', 'UNKNOWN')
                tn_node_types[node_type] = tn_node_types.get(node_type, 0) + 1
                frequency = v.attributes().get('frequency', 'UNKNOWN')

            print(f"  📊 典型良性快照的节点类型分布 (快照{tn_indices[0]}): {dict(sorted(tn_node_types.items()))}【{frequency}】")

    print("\n" + "="*70)
    print("🎯 调试分析总结:")
    print(f"  • 总共分析了 {len(eval_true)} 个快照")
    print(f"  • 检测准确率: {(len(tp_indices) + len(tn_indices))/len(eval_true)*100:.1f}%")
    if len(tp_indices) + len(fn_indices) > 0:
        print(f"  • 恶意快照召回率: {len(tp_indices)/(len(tp_indices) + len(fn_indices))*100:.1f}%")
    if len(tp_indices) + len(fp_indices) > 0:
        print(f"  • 恶意检测精确率: {len(tp_indices)/(len(tp_indices) + len(fp_indices))*100:.1f}%")
    print("="*70)

def predict_snapshots(
    snapshot_embeddings: np.ndarray,
) -> Tuple[np.ndarray, Dict]:
    """预测快照异常标签"""

    classify = get_classfy(CLASSIFY_NAME)
    classify.load()
    pred_labels, diff_vectors  = classify.predict(snapshot_embeddings)

    return pred_labels, diff_vectors


def _map_snapshot_to_technique(snapshot) -> str:
    """将一个快照映射为“攻击技术码”。

    优先级：
    1) 图/节点已有显式字段（attack_technique/technique/mitre_technique 等）则直接使用（多数值取众数）。
    2) 否则根据节点类型粗粒度合成签名码，如 "PROC+FILE+NET"（按字母序拼接去重）。
    """
    # 1) 图级属性（若 igraph.Graph 支持）
    try:
        for key in ("attack_technique", "technique", "mitre_technique"):
            if key in snapshot.attributes():  # type: ignore[attr-defined]
                val = snapshot[key]
                if isinstance(val, str) and val.strip():
                    return val.strip().upper()
    except Exception:
        pass

    # 2) 节点级显式技术字段（取众数）
    cand = []
    try:
        for v in snapshot.vs:
            attrs = v.attributes()
            for key in ("attack_technique", "technique", "mitre_technique"):
                t = attrs.get(key)
                if isinstance(t, str) and t.strip():
                    cand.append(t.strip().upper())
        if cand:
            # 众数 / 最多出现者
            vals, cnts = np.unique(np.array(cand, dtype=object), return_counts=True)
            return str(vals[int(np.argmax(cnts))])
    except Exception:
        pass

    # 3) 粗粒度基于 type_name 的签名
    coarse = set()
    try:
        for v in snapshot.vs:
            t = str(v.attributes().get("type_name", "UNKNOWN")).upper()
            if "NET" in t:
                coarse.add("NET")
            elif "PROCESS" in t or "PROC" in t or "SUBJECT" in t:
                coarse.add("PROC")
            elif "FILE" in t:
                coarse.add("FILE")
            elif "REG" in t or "REGISTRY" in t:
                coarse.add("REG")
            else:
                # 保留一个 OTHER 以避免过度细分
                coarse.add("OTHER")
        if coarse:
            return "+".join(sorted(coarse))
    except Exception:
        pass

    return "UNKNOWN"


def _load_technique_sequence_library(path: Optional[str]) -> List[List[str]]:
    """加载技术序列库。每行一条序列，逗号/空白分隔，支持 # 注释。"""
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        print(f"[SeqFilter-Tech] 未找到技术序列库文件：{p}，将回退到旧规则。")
        return []
    lib: List[List[str]] = []
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # 兼容逗号或空白分隔
                parts = [x for x in line.replace("\t", " ").replace(",", " ").split(" ") if x]
                if parts:
                    lib.append(parts)
    except Exception as ex:
        print(f"[SeqFilter-Tech] 读取技术序列库失败：{ex}")
        return []
    print(f"[SeqFilter-Tech] 已加载技术序列库 {p}：{len(lib)} 条。")
    return lib


def _best_lcs_keep_mask(seq: List[str], lib: List[List[str]]) -> Tuple[List[bool], int, int]:
    """在库中为 seq 寻找最佳 LCS 匹配。

    返回：
    - keep_mask: 与 seq 等长的布尔列表，仅 LCS 中的元素为 True（用于保留对应报警快照）
    - best_lcs_len: 最优 LCS 长度
    - best_lib_len: 该最优匹配对应的库序列长度（用于计算比例）
    """
    best_keep: List[bool] = [False] * len(seq)
    best_len = 0
    best_lib_len = 0
    seq_u = [s.strip().upper() for s in seq]

    for L in lib:
        L_u = [t.strip().upper() for t in L if t.strip()]
        if not L_u:
            continue
        keep_mask, lcs_len = _lcs_indices_keep_mask(seq_u, L_u)
        if lcs_len > best_len:
            best_len = lcs_len
            best_keep = keep_mask
            best_lib_len = len(L_u)

    return best_keep, best_len, best_lib_len


def _lcs_indices_keep_mask(a: List[str], b: List[str]) -> Tuple[List[bool], int]:
    """计算 a 与 b 的 LCS，并返回：
    - keep_mask: 与 a 等长的布尔列表，表示 a 中哪些位置参与了 LCS 匹配
    - lcs_len: LCS 的长度

    复杂度 O(len(a) * len(b))。
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return [False] * n, 0
    # DP 表与回溯
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = dp[i - 1][j] if dp[i - 1][j] >= dp[i][j - 1] else dp[i][j - 1]

    # 回溯找出 a 中被选中的索引
    keep = [False] * n
    i, j = n, m
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            keep[i - 1] = True
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return keep, dp[n][m]


# =========================
# 解耦：映射 / 运行段提取 / LCS过滤
# =========================

def map_pred_positive_to_techniques(
    pred_labels: np.ndarray,
    snapshots: List,
    *,
    semantic_mapper: Optional['TechniqueSemanticMapper'] = None,
):
    """仅将“预测为恶意(=1)”的快照映射为技术码。

    返回：
    - idx_pos: np.ndarray[int]，预测为 1 的全局索引（相对于传入 snapshots 的起点）
    - tech_seq: List[str]，与 idx_pos 等长的技术码序列
    """
    y = np.asarray(pred_labels, dtype=int)
    idx_pos = np.where(y == 1)[0]
    tech_seq: List[str] = []
    if semantic_mapper is not None:
        # 使用外部注入的语义映射器（独立类）
        try:
            queries = []
            for k in idx_pos:
                try:
                    queries.append(semantic_mapper.snapshot_to_query(snapshots[int(k)]))
                except Exception:
                    queries.append("")
            tech_seq = semantic_mapper.predict_codes(queries)
        except Exception as ex:
            print(f"[Map] 语义映射器失败，回退到属性规则：{ex}")
            tech_seq = []

    if not tech_seq:
        for k in idx_pos:
            try:
                tech_seq.append(_map_snapshot_to_technique(snapshots[int(k)]))
            except Exception:
                tech_seq.append("UNKNOWN")
    return idx_pos, tech_seq





def filter_positive_by_tech_lcs(
    pred_labels: np.ndarray,
    idx_pos: np.ndarray,
    tech_seq: List[str],
    lib: List[List[str]],
    *,
    lcs_min_ratio: float = 0.6,
) -> np.ndarray:
    """对“已映射出的预测阳性技术序列”执行一次性 LCS 过滤，并返回新的标签数组。"""
    if pred_labels is None or len(pred_labels) == 0:
        return pred_labels
    y = np.asarray(pred_labels, dtype=int).copy()
    if not lib or len(tech_seq) == 0 or getattr(idx_pos, "size", 0) == 0:
        return y

    keep_mask, best_lcs_len, best_lib_len = _best_lcs_keep_mask(tech_seq, lib)
    ratio = (best_lcs_len / best_lib_len) if best_lib_len > 0 else 0.0

    if best_lcs_len > 0 and ratio >= float(lcs_min_ratio):
        dropped = 0
        for t, keep in enumerate(keep_mask):
            if not keep:
                y[int(idx_pos[t])] = 0
                dropped += 1
        print(
            f"[SeqFilter-Tech] 全局 LCS 命中：库长={best_lib_len}, LCS={best_lcs_len}, 比例={ratio:.3f}；移除非主干 {dropped} 个阳性。"
        )
    else:
        removed = int(getattr(idx_pos, "size", 0))
        y[idx_pos] = 0
        print(
            f"[SeqFilter-Tech] 全局 LCS 未达阈值（库长={best_lib_len}, LCS={best_lcs_len}, 比例={ratio:.3f}），清零 {removed} 个阳性。"
        )

    return y


def run_evaluation(path_map: dict) -> None:
    snapshot_file = "snapshot_data.pkl"
    if not os.path.exists(snapshot_file):
        print(f"❌ 错误：快照数据文件不存在: {snapshot_file}")
        print("请先运行 train_darpa.py 来生成快照数据")
        return

    with open(snapshot_file, 'rb') as f:
        snapshot_data = pickle.load(f)

    # 提取快照数据
    all_snapshots = snapshot_data['all_snapshots']
    benign_idx_start = snapshot_data['benign_idx_start']
    benign_idx_end = snapshot_data['benign_idx_end']
    malicious_idx_start = snapshot_data['malicious_idx_start']
    malicious_idx_end = snapshot_data['malicious_idx_end']

    print("✅ 快照数据加载成功:")
    print(f"  - 总快照数: {len(all_snapshots)}")
    print(f"  - 良性快照范围: {benign_idx_start} 到 {benign_idx_end}")
    print(f"  - 恶意快照范围: {malicious_idx_start} 到 {malicious_idx_end}")
    mal_snapshots = all_snapshots[malicious_idx_start: malicious_idx_end + 1]
    if not mal_snapshots:
        print("[ERROR] 未能构建快照")
        return
    save_snapshot_nodes(mal_snapshots)
    true_labels = get_true_labels(mal_snapshots)
    # 打印恶意快照片段内索引（从0开始）
    mal_idx_in_slice = np.where(true_labels == 1)[0]
    print(f"恶意快照片段内索引: {mal_idx_in_slice.tolist()}")
    print("\n[DEBUG] 快照信息")
    print(f"  - 总快照数: {len(mal_snapshots)}")
    print(f"  - 真实标签数: {len(true_labels)}")
    print(f"  - 真实标签: {true_labels.tolist()}")

    embedder_cls = get_embedder_by_name(EMBEDDER_NAME)
    embedder = embedder_cls.load(snapshot_sequence=all_snapshots)
    snapshot_embeddings = embedder.get_snapshot_embeddings()
    # 统计恶意节点偏离（仅当快照中存在恶意节点时会输出/写日志）
    try:
        # 这里只看“恶意节点在所有节点偏离中的排名”，相对于“快照节点简单平均的中心向量”
        embedder.compute_malicious_deviation_per_snapshot(center_weighting='none')
    except Exception as ex:
        print(f"[Test] 恶意节点偏离统计失败：{ex}")
    # snapshot_embeddings, info = inject_snapshots_deviation(snapshot_embeddings, target_idxs=[29, 43, 70, 71, 72, 73], mode="away", alpha=3.0)

    # 在测试阶段进行 t-SNE 可视化（Top-K 标注，默认每组各取 5 个，cosine 偏离）
    try:
        plot_tsne_embeddings(
            snapshot_embeddings,
            benign_idx_start,
            benign_idx_end,
            annotate=True,
            mode="all",
            top_k=5,
            group="per-group",
            metric="cosine",
            save_path="snapshot_embeddings_tsne.png",
            which="malicious",
            malicious_start=malicious_idx_start,
            malicious_end=malicious_idx_end,
            # selected_mal_ids_in_slice=[40, 41, 42, 43,44,45,46]
        )
    except Exception as ex:
        print(f"[Viz] t-SNE 可视化失败：{ex}")

    # 偏离变化曲线可视化
    try:
        # 简化接口：默认画“恶意区间”的变化；改为 mode="benign"/"all"/"range" 可自由切换
        plot_deviation(
            snapshot_embeddings,
            benign_idx_start,
            benign_idx_end,
            mode="malicious",
            malicious_start=malicious_idx_start,
            malicious_end=malicious_idx_end,
            metric="cosine",
            smooth_k=5,
            save_path="snapshot_deviation_curve.png",
            annotate_top_k=10,
            annotate_mode="all",  # 开关："none" | "topk" | "all"
            recompute_stats=False,
            deviation_mode="center",# “step | center ”
        )
    except Exception as ex:
        print(f"[Viz] 偏离曲线可视化失败：{ex}")

    pred_labels, diff_vectors = predict_snapshots(
        snapshot_embeddings[malicious_idx_start: malicious_idx_end + 1]
    )
    print(f"检测到 {len(diff_vectors)} 个异常快照")
    print(f"预测标签长度: {len(pred_labels)}")

    if SEQ_FILTER.get("enable", False):
        try:
            from process.technique_semantic_mapper import TechniqueSemanticMapper  # type: ignore
            sem_mapper = TechniqueSemanticMapper(
                csv_path="data/mitreembed_master_Chroma.csv",
                persist_dir="./chroma_db",
                model_name="sentence-transformers/all-MiniLM-L12-v2",
                page_content_column="Body",
                code_column="Subject",
                top_k=5,
            )
        except Exception as ex:
            sem_mapper = None
            print(f"[Map] 语义映射器初始化失败：{ex}")
        lib: List[List[str]] = _load_technique_sequence_library(SEQ_FILTER.get("library_path"))
        if len(lib) > 0:
            idx_pos, tech_seq = map_pred_positive_to_techniques(
                pred_labels,
                mal_snapshots,
                semantic_mapper=sem_mapper,
            )
            y_ref = filter_positive_by_tech_lcs(
                pred_labels,
                idx_pos,
                tech_seq,
                lib,
                lcs_min_ratio=float(SEQ_FILTER.get("lcs_min_ratio", 0.6)),
            )
            removed = int(np.sum(pred_labels) - np.sum(y_ref))
            print(f"[SeqFilter] 技术序列库筛选：移除 {removed} 个不匹配告警。")
            pred_labels_refined = y_ref.astype(int)
        else:
            print("[SeqFilter] 未找到技术序列库，跳过二次筛选。")
            pred_labels_refined = pred_labels
    else:
        pred_labels_refined = pred_labels

    # 原始与二次筛选后的指标
    def _metrics(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        return acc, prec, rec, f1, tp, fp, tn, fn

    acc0, prec0, rec0, f10, tp0, fp0, tn0, fn0 = _metrics(true_labels, pred_labels)
    acc, prec, rec, f1, tp, fp, tn, fn = _metrics(true_labels, pred_labels_refined)

    # 未筛选的指标
    print("\n=== 评估结果（未筛选）===")
    print("\n" + "=" * 50)
    print(" 快照级别评估结果 (所有快照)")
    print("=" * 50)
    print(f" 真阳性 (TP): {tp0}")
    print(f" 假阳性 (FP): {fp0}")
    print(f" 真阴性 (TN): {tn0}")
    print(f" 假阴性 (FN): {fn0}")
    print("\n 性能评分:")
    print(f" 准确率: {acc0:.4f}")
    print(f" 精确率: {prec0:.4f}")
    print(f" 召回率: {rec0:.4f}")
    print(f" F1分数: {f10:.4f}")
    print("=" * 50)

    # 二次筛选后的指标
    print("\n=== 评估结果（序列二次筛选后）===")
    print("\n" + "=" * 50)
    print(" 快照级别评估结果 (所有快照)")
    print("=" * 50)
    print(f" 真阳性 (TP): {tp}")
    print(f" 假阳性 (FP): {fp}")
    print(f" 真阴性 (TN): {tn}")
    print(f" 假阴性 (FN): {fn}")
    print("\n 性能评分:")
    print(f" 准确率: {acc:.4f}")
    print(f" 精确率: {prec:.4f}")
    print(f" 召回率: {rec:.4f}")
    print(f" F1分数: {f1:.4f}")
    print("=" * 50)
    print_debug_info(mal_snapshots, true_labels, pred_labels_refined, 0)  # 从索引0开始



# ========================================================================
# 主入口
# ========================================================================
if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env = config["local"] if "windows" in sys.platform else config["remote"]

    run_evaluation(env["path_map"])