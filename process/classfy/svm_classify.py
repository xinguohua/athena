import os
import pickle
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np
from .base import BaseClassify


# 轻量标准化器（避免第三方依赖）
class _NumpyScaler:
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        self.scale_ = std
        return (X - self.mean_) / self.scale_

    def is_fitted(self) -> bool:
        return (self.mean_ is not None) and (self.scale_ is not None)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if not self.is_fitted():
            # 未拟合则直接返回原数据（预测时允许无训练快速路径）
            return X
        return (X - self.mean_) / self.scale_


# ========== Trainer 配置（Top-K 偏离度，无阈值） ==========
@dataclass
class TopKDeviationConfig:
    # 直接取偏离度前 k 个作为异常（无阈值）
    k: int = 5
    # 数据预处理
    use_scaler: bool = True  # 是否对特征做标准化（z-score）
    # 说明：训练中心已移除，预测时一律使用当前批次均值作为中心
    # 保存路径（沿用原结构，便于集成）
    scaler_save_path: str = "topk_scaler.pkl"
    meta_save_path: str = "topk_meta.pkl"  # 存储中心向量/配置等


# ========== Trainer 实现 ==========
class TopKDeviationClassify(BaseClassify):
    """使用 Top-K 偏离度替换原 One-Class SVM：
    - 训练：仅估计（可选标准化后的）中心向量 center
    - 推断：对每个样本计算与 center 的 L2 距离，取偏离度 Top-K 判为异常
    - 无阈值超参，只有 k
    """

    def __init__(self, cfg: Optional[TopKDeviationConfig] = None, **kwargs):
        super().__init__()
        self.cfg = cfg or TopKDeviationConfig()
        # 允许动态覆盖配置
        for k, v in kwargs.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)

        # 轻量标准化器（避免额外依赖）
        self.scaler = _NumpyScaler() if self.cfg.use_scaler else None
        self.model = self._build_model()

    def _build_model(self):
        """Top-K 偏离度不需要具体模型，这里返回 None 作为占位。"""
        return None

    # 覆盖父类训练：Top-K 偏离度无需训练，直接返回自身
    def train(self, embeddings, **kwargs):
        print("[TopK] 跳过训练：该分类器无需训练，预测时直接以批次均值为中心计算偏离度。")
        return self

    def _train_loop(self, snapshot_embeddings: Any, labels: Optional[np.ndarray] = None, **kwargs) -> Dict[str, list]:
        """训练循环：估计中心向量（标准化后）并保存元数据"""
        cfg = self.cfg
        print(f"[TopKDeviation Trainer] config={cfg}")

        X = np.asarray(snapshot_embeddings, dtype=np.float32)
        print(f"[TopK] 输入数据形状: {X.shape}")

        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
            print("[TopK] 数据已标准化 (z-score)")

        # 仅用正常样本估计中心（若提供标签）
        if labels is not None:
            normal_mask = (labels == 0)
            X_normal = X[normal_mask]
            print(f"[TopK] 使用正常样本估计中心: {X_normal.shape}")
        else:
            X_normal = X
            print(f"[TopK] 使用全部样本估计中心: {X_normal.shape}")

        # 保存 scaler 与元数据（不再保存中心）
        if self.scaler is not None:
            with open(cfg.scaler_save_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"[Save] scaler -> {cfg.scaler_save_path}")

        try:
            meta = {
                "k": int(cfg.k),
                "config": self.cfg.__dict__,
            }
            with open(cfg.meta_save_path, 'wb') as f:
                pickle.dump(meta, f)
            print(f"[Save] meta -> {cfg.meta_save_path} (k={cfg.k})")
        except Exception as e:
            print(f"[Save] meta 失败：{e}")

        return {"train_info": [{"n_samples": int(X_normal.shape[0])}]}

    def predict(
        self,
        embeddings: np.ndarray,
        k: Optional[int] = None,
        plot: bool = True,
        plot_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """基于 Top-K 偏离度的预测（无阈值，支持“无需训练”的直接测试）

        Args:
            embeddings: 快照嵌入向量
            k: 取偏离度前 k 个（默认使用训练配置的 k）
            plot: 是否绘制偏离度可视化（柱状图，按偏离度降序）
            plot_path: 若未提供且 plot=True，将默认保存至 "viz/deviation.png"；
                       若提供则将图保存到该路径（优先级高于弹窗显示）
            title: 图标题，可选

        Returns:
            pred_labels: 预测标签 (0=正常, 1=异常)
            diff_vectors: 异常详情字典（包含偏离度）
        """
        X = np.asarray(embeddings, dtype=np.float32)
        if self.scaler is not None:
            # 若标准化器未拟合，则跳过标准化，允许直接测试
            X = self.scaler.transform(X)

        # 选择中心参考：始终使用当前输入批次的均值（训练中心已移除）
        center_ref = np.mean(X, axis=0).astype(np.float32)

        # 计算每个样本的偏离度（与参考中心的 L2 距离）
        diffs = X - center_ref
        dev = np.linalg.norm(diffs, axis=1)

        k_eff = int(self.cfg.k if k is None else k)
        k_eff = int(np.clip(k_eff, 0, len(dev)))

        pred_labels = np.zeros(len(dev), dtype=int)
        diff_vectors: Dict[int, Dict] = {}
        if k_eff > 0:
            top_idx = np.argsort(-dev)[:k_eff]
            pred_labels[top_idx] = 1
            for i in top_idx:
                diff_vectors[i] = {
                    "position": int(i),
                    "deviation": float(dev[i]),
                    "embedding": embeddings[i],
                }

        print("\n--- 快照检测结果 (Top-K 偏离度) ---")
        print("快照索引 | 偏离度 | 状态")
        print("-" * 40)
        for i, d in enumerate(dev):
            status = "🔴 异常" if pred_labels[i] == 1 else "🟢 正常"
            print(f"快照 {i:2d}   | {d:.6f} | {status}")
        print("-" * 40 + "\n")
        # 可视化（可选）
        if plot or plot_path:
            # 若开启绘图但未提供路径，则生成默认保存路径
            if plot and not plot_path:
                plot_path = os.path.join("viz", "deviation.png")
            try:
                self._plot_deviation(dev, k_eff, plot_path=plot_path, title=title)
            except Exception as e:
                print(f"[Plot] 可视化失败：{e}")
        return pred_labels, diff_vectors

    def _plot_deviation(self, dev: np.ndarray, k: int, plot_path: Optional[str] = None, title: Optional[str] = None):
        """绘制偏离度柱状图：按偏离度降序，Top-K 用红色，其余用灰色。

        若提供 plot_path 则保存到文件，否则弹窗显示。
        若未安装 matplotlib，将给出友好提示并打印 Top-10 偏离度。
        """
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            top_idx = np.argsort(-dev)[:min(k, len(dev))]
            print("[Plot] 未安装 matplotlib，打印 Top-偏离度替代：")
            for rank, i in enumerate(top_idx, 1):
                print(f"  #{rank:2d} idx={i:3d} deviation={dev[i]:.6f}")
            return

        n = len(dev)
        order = np.argsort(-dev)
        dev_sorted = dev[order]
        colors = ["crimson" if i < min(k, n) else "#cccccc" for i in range(n)]

        width = max(6.0, min(16.0, 0.2 * n + 2))
        fig, ax = plt.subplots(figsize=(width, 4.5))
        ax.bar(range(n), dev_sorted, color=colors, alpha=0.9, edgecolor="#444444", linewidth=0.4)
        ax.set_xlabel("samples (sorted by deviation)")
        ax.set_ylabel("L2 deviation")
        ax.set_title(title or f"Deviation ranking (Top-{min(k, n)})")
        ax.grid(axis="y", linestyle=":", alpha=0.4)

        # 辅助线：Top-K 阈值
        if n > 0 and k is not None and k > 0:
            thr = dev_sorted[min(k - 1, n - 1)]
            ax.axhline(thr, color="orange", linestyle="--", linewidth=1.0, alpha=0.8, label=f"Top-{min(k, n)} threshold")
            ax.legend(loc="upper right")

        fig.tight_layout()
        if plot_path:
            os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"[Plot] 偏离度图已保存到: {plot_path}")
            plt.close(fig)
        else:
            plt.show()

    def load(self, scaler_path: Optional[str] = None, meta_path: Optional[str] = None):
        """加载标准化器与中心元数据"""
        scaler_path = scaler_path or self.cfg.scaler_save_path
        meta_path = meta_path or self.cfg.meta_save_path

        if self.cfg.use_scaler and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"[Load] scaler <- {scaler_path}")

        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                # 同步配置中的 k
                if "k" in meta:
                    self.cfg.k = int(meta["k"])
                print(f"[Load] meta <- {meta_path} (k={self.cfg.k})")
            except Exception as e:
                print(f"[Load] meta 失败：{e}")
        return self

