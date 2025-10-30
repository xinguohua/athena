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
    # 中心选择策略：
    # - 'batch': 始终使用当前输入批次的均值作为中心（推荐用于“全是恶意样本”的场景）
    # - 'trained': 始终使用训练得到的中心（若无则报错）
    # - 'auto': 优先使用训练中心，若无则退回 batch 均值
    center_mode: str = "batch"
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
        # 中心向量（标准化空间下）
        self.center = None  # type: Optional[np.ndarray]

    def _build_model(self):
        """Top-K 偏离度不需要具体模型，这里返回 None 作为占位。"""
        return None

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

        # 中心向量
        self.center = np.mean(X_normal, axis=0).astype(np.float32)
        print(f"[TopK] 中心范数: {np.linalg.norm(self.center):.6f}")

        # 保存 scaler 与元数据
        if self.scaler is not None:
            with open(cfg.scaler_save_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"[Save] scaler -> {cfg.scaler_save_path}")

        try:
            meta = {
                "center": self.center,
                "k": int(cfg.k),
                "config": self.cfg.__dict__,
            }
            with open(cfg.meta_save_path, 'wb') as f:
                pickle.dump(meta, f)
            print(f"[Save] meta -> {cfg.meta_save_path} (k={cfg.k})")
        except Exception as e:
            print(f"[Save] meta 失败：{e}")

        return {"train_info": [{"n_samples": int(X_normal.shape[0])}]}

    def predict(self, embeddings: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """基于 Top-K 偏离度的预测（无阈值，支持“无需训练”的直接测试）

        Args:
            embeddings: 快照嵌入向量
            k: 取偏离度前 k 个（默认使用训练配置的 k）

        Returns:
            pred_labels: 预测标签 (0=正常, 1=异常)
            diff_vectors: 异常详情字典（包含偏离度）
        """
        X = np.asarray(embeddings, dtype=np.float32)
        if self.scaler is not None:
            # 若标准化器未拟合，则跳过标准化，允许直接测试
            X = self.scaler.transform(X)

        # 选择中心参考：按配置中心模式
        mode = (self.cfg.center_mode or "auto").lower()
        if mode == "trained":
            assert self.center is not None, "中心模式为 'trained' 但未训练中心。可改为 center_mode='batch' 或先调用 train()。"
            center_ref = self.center
        elif mode == "batch":
            center_ref = np.mean(X, axis=0).astype(np.float32)
        else:  # 'auto'
            center_ref = self.center if self.center is not None else np.mean(X, axis=0).astype(np.float32)

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
        return pred_labels, diff_vectors

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
                self.center = meta.get("center", None)
                # 同步配置中的 k
                if "k" in meta:
                    self.cfg.k = int(meta["k"])
                print(f"[Load] meta <- {meta_path} (k={self.cfg.k}, center_dim={None if self.center is None else len(self.center)})")
            except Exception as e:
                print(f"[Load] meta 失败：{e}")
        return self

