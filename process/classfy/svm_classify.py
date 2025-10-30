import os
import pickle
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from .base import BaseClassify


# ========== Trainer 配置 ==========
@dataclass
class SVMConfig:
    # One-Class SVM参数
    nu: float = 0.1  # 训练误差上界和支持向量下界（异常比例）
    kernel: str = 'rbf'  # 核函数: 'linear', 'poly', 'rbf', 'sigmoid'
    gamma: str = 'scale'  # 核系数: 'scale', 'auto', 或浮点数
    degree: int = 3  # poly核的度数
    
    # 数据预处理
    use_scaler: bool = True
    
    # 模型保存
    model_save_path: str = "svm_detector.pkl"
    scaler_save_path: str = "svm_scaler.pkl"
    meta_save_path: str = "svm_meta.pkl"  # 存储阈值等元数据
    # 推断阈值：
    # - 若为 None：允许训练阶段自动标定（按 nu 分位数）
    # - 若为具体数值：固定使用该阈值，训练阶段跳过自动标定
    decision_threshold: Optional[float] = -0.1


# ========== Trainer 实现 ==========
class SVMClassify(BaseClassify):
    def __init__(self, cfg: Optional[SVMConfig] = None, **kwargs):
        super().__init__()
        self.cfg = cfg or SVMConfig()
        # 允许动态覆盖配置
        for k, v in kwargs.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
        
        self.scaler = StandardScaler() if self.cfg.use_scaler else None
        self.model = self._build_model()
        # 运行期阈值（优先级：predict(threshold) > self.threshold > 0.0）
        self.threshold: Optional[float] = self.cfg.decision_threshold

    def _build_model(self):
        """返回具体模型"""
        cfg = self.cfg
        
        # One-Class SVM (异常检测)
        model = OneClassSVM(
            kernel=cfg.kernel,
            gamma=cfg.gamma,
            nu=cfg.nu,
            degree=cfg.degree,
            verbose=False
        )
        
        return model

    def _train_loop(self, snapshot_embeddings: Any, labels: Optional[np.ndarray] = None, **kwargs) -> Dict[str, list]:
        """训练循环，返回训练历史"""
        cfg = self.cfg
        print(f"[One-Class SVM Trainer] config={cfg}")

        X = np.asarray(snapshot_embeddings, dtype=np.float32)
        print(f"[SVM] 输入数据形状: {X.shape}")
        
        # 数据标准化
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
            print("[SVM] 数据已标准化")
        
        # One-Class SVM模式（仅用正常数据训练）
        if labels is not None:
            # 如果提供了标签，只用正常样本训练
            normal_mask = (labels == 0)
            X_normal = X[normal_mask]
            print(f"[One-Class SVM] 使用正常样本训练: {X_normal.shape}")
        else:
            # 假设所有数据都是正常的
            X_normal = X
            print(f"[One-Class SVM] 使用全部样本训练: {X_normal.shape}")
        
        print("[SVM] 开始训练One-Class SVM...")
        self.model.fit(X_normal)
        
        # 在训练数据上评估
        decision_train = self.model.decision_function(X_normal)
        train_pred = (decision_train < 0.0).astype(int)  # 仅用于统计（默认0阈值）
        n_outliers = int(np.sum(train_pred == 1))
        n_inliers = int(np.sum(train_pred == 0))
        
        print("\n[One-Class SVM] 训练集预测:")
        print(f"  内点(正常): {n_inliers}")
        print(f"  离群点(异常): {n_outliers}")
        print(f"  离群点比例: {n_outliers/len(train_pred):.4f}")
        print(f"  决策分数: mean={np.mean(decision_train):.6f} std={np.std(decision_train):.6f} min={np.min(decision_train):.6f} max={np.max(decision_train):.6f}")

        # === 阈值策略 ===
        # 若用户已在配置中给出固定阈值，则跳过自动标定；否则按 nu 分位数自动标定
        if self.cfg.decision_threshold is None:
            try:
                q = float(np.clip(self.cfg.nu, 0.001, 0.5))  # 安全范围
                thr = float(np.quantile(decision_train, q))
                self.threshold = thr
                self.cfg.decision_threshold = thr
                print(f"[Calibrate] 基于训练集分位数自动标定阈值: nu={self.cfg.nu} -> threshold={thr:+.6f}")
            except Exception as e:
                print(f"[Calibrate] 阈值标定失败，使用默认0.0：{e}")
                self.threshold = self.threshold if self.threshold is not None else 0.0
        else:
            self.threshold = float(self.cfg.decision_threshold)
            print(f"[Calibrate] 跳过自动标定，使用固定阈值: {self.threshold:+.6f}")
        
        history = {"train_info": [{
            "n_inliers": int(n_inliers),
            "n_outliers": int(n_outliers),
            "n_support_vectors": len(self.model.support_vectors_)
        }]}
        
        # ===== 保存模型 =====
        save_dir = os.path.dirname(cfg.model_save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        with open(cfg.model_save_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"[Save] model -> {cfg.model_save_path}")
        
        if self.scaler is not None:
            with open(cfg.scaler_save_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"[Save] scaler -> {cfg.scaler_save_path}")

        # 保存元数据（阈值、配置快照）
        try:
            meta = {
                "decision_threshold": self.threshold,
                "config": self.cfg.__dict__,
            }
            with open(cfg.meta_save_path, 'wb') as f:
                pickle.dump(meta, f)
            print(f"[Save] meta -> {cfg.meta_save_path} (threshold={self.threshold})")
        except Exception as e:
            print(f"[Save] meta 失败：{e}")
        
        return history

    def predict(self, embeddings: np.ndarray, threshold: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
        """
        用训练好的模型预测快照是否异常
        
        Args:
            embeddings: 快照嵌入向量
            threshold: 决策函数阈值(默认0.0)，负值表示更严格的异常判定
        
        Returns:
            pred_labels: 预测标签 (0=正常, 1=异常)
            diff_vectors: 异常详情字典
        """
        assert self.model is not None, "model 未训练或未加载"
        
        X = np.asarray(embeddings, dtype=np.float32)
        
        # 数据标准化
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        pred_labels = np.zeros(len(X), dtype=int)
        diff_vectors, scores = {}, {}
        
        # One-Class SVM
        if threshold is None:
            threshold = self.threshold if self.threshold is not None else 0.0
        
        # 决策函数: 正值=内点(正常), 负值=离群点(异常)
        decision = self.model.decision_function(X)
        if float(np.std(decision)) < 1e-8:
            print("[Warn] 所有决策分数几乎相同，可能存在以下情况：\n"
                  "  - 嵌入向量近乎常数/坍缩（请检查编码器/embedding生成）\n"
                  "  - 训练/测试使用的标准化器不匹配（请确认在同一流程训练/加载）\n"
                  "  - SVM 参数不当（尝试调整 gamma/nu 或关闭标准化）")
        pred_labels = (decision < threshold).astype(int)
        
        for i in range(len(X)):
            score = decision[i]
            scores[i] = score
            if pred_labels[i] == 1:
                diff_vectors[i] = {
                    "position": i,
                    "decision_value": float(score),
                    "embedding": embeddings[i]
                }
        
        print("\n--- 快照检测结果 ---")
        print("快照索引 | 得分(Score) | 状态")
        print("-" * 40)
        for i, score in scores.items():
            status = "🔴 异常" if pred_labels[i] == 1 else "🟢 正常"
            print(f"快照 {i:2d}   | {score:+.6f}  | {status}")
        print("-" * 40 + "\n")
        return pred_labels, diff_vectors

    def load(self, path=None, scaler_path=None):
        """加载已保存的模型"""
        path = path or self.cfg.model_save_path
        scaler_path = scaler_path or self.cfg.scaler_save_path
        
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"[Load] model <- {path}")
        
        if self.cfg.use_scaler and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"[Load] scaler <- {scaler_path}")

        # 加载阈值元数据（可选）
        meta_path = self.cfg.meta_save_path
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                self.threshold = meta.get("decision_threshold", self.threshold)
                print(f"[Load] meta <- {meta_path} (threshold={self.threshold})")
            except Exception as e:
                print(f"[Load] meta 失败：{e}")
        
        return self.model

