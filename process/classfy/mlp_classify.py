"""
MLP 分类器 — 论文 Section IV-B.4

两层 MLP + cross-entropy loss，在冻结的快照嵌入上训练二分类器。
训练时需要良性和恶意快照的嵌入及标签。

用法:
    classify = MLPClassify(gid="bench")
    classify.train(benign_embeddings, malicious_embeddings, mal_labels)
    pred_labels, details = classify.predict(test_embeddings)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from .base import BaseClassify


# ========== 模型 ==========
class TwoLayerMLP(nn.Module):
    """两层 MLP: input -> hidden -> ReLU -> dropout -> output(2)"""
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 2),  # 二分类: [benign, malicious]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ========== 配置 ==========
@dataclass
class MLPConfig:
    hidden_dim: int = 128
    dropout: float = 0.3
    lr: float = 1e-3
    num_epochs: int = 50
    batch_size: int = 64
    # 持久化路径
    model_save_path: str = "mlp_classifier.pth"
    meta_save_path: str = "mlp_meta.pkl"


# ========== 分类器 ==========
class MLPClassify(BaseClassify):
    """
    论文描述的两层 MLP 分类器（有监督，cross-entropy loss）。

    与 TopK 的关键区别：
    - TopK 无监督：不需要标签，仅计算偏离度取 Top-K
    - MLP 有监督：需要良性+恶意嵌入和标签，训练二分类器
    """

    def __init__(self, cfg: Optional[MLPConfig] = None, gid: Optional[str] = None, **kwargs):
        super().__init__(gid=gid)
        self.cfg = cfg or MLPConfig()
        for k, v in kwargs.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)

        if gid:
            self.cfg.model_save_path = self.with_gid_suffix(self.cfg.model_save_path)
            self.cfg.meta_save_path = self.with_gid_suffix(self.cfg.meta_save_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim: Optional[int] = None
        self.model: Optional[TwoLayerMLP] = None

    def _build_model(self):
        if self.input_dim is None:
            raise ValueError("input_dim 未设置，请先调用 train()")
        return TwoLayerMLP(
            input_dim=self.input_dim,
            hidden_dim=self.cfg.hidden_dim,
            dropout=self.cfg.dropout,
        ).to(self.device)

    def train(self, benign_embeddings: np.ndarray,
              malicious_embeddings: np.ndarray = None,
              malicious_labels: np.ndarray = None,
              **kwargs) -> "MLPClassify":
        """
        训练 MLP 分类器。

        Args:
            benign_embeddings: 良性快照嵌入 (N_b, D)，标签全部为 0
            malicious_embeddings: 恶意快照嵌入 (N_m, D)
            malicious_labels: 恶意快照真实标签 (N_m,)，1=恶意, 0=良性
                若不提供则全部视为恶意（标签=1）
        """
        X_b = np.asarray(benign_embeddings, dtype=np.float32)
        self.input_dim = X_b.shape[1]

        # 构建训练集：良性(label=0) + 恶意区间(label=0或1)
        labels_b = np.zeros(X_b.shape[0], dtype=np.int64)

        if malicious_embeddings is not None:
            X_m = np.asarray(malicious_embeddings, dtype=np.float32)
            if malicious_labels is not None:
                labels_m = np.asarray(malicious_labels, dtype=np.int64)
            else:
                labels_m = np.ones(X_m.shape[0], dtype=np.int64)

            X_all = np.concatenate([X_b, X_m], axis=0)
            y_all = np.concatenate([labels_b, labels_m])
        else:
            # 仅有良性数据：无法有效训练，打印警告
            print("[MLP] 警告：无恶意样本，分类器可能无法有效训练")
            X_all = X_b
            y_all = labels_b

        self.model = self._build_model()
        self._train_loop(X_all, labels=y_all)
        self._save()
        return self

    def _train_loop(self, embeddings, labels=None, **kwargs):
        X = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(self.device)
        y = torch.from_numpy(np.asarray(labels, dtype=np.int64)).to(self.device)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)

        # 类别权重：处理正负样本不平衡
        n_pos = int((y == 1).sum().item())
        n_neg = int((y == 0).sum().item())
        if n_pos > 0 and n_neg > 0:
            weight = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32, device=self.device)
        else:
            weight = None
        criterion = nn.CrossEntropyLoss(weight=weight)

        self.model.train()
        for epoch in range(self.cfg.num_epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            for xb, yb in loader:
                logits = self.model(xb)
                loss = criterion(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                acc = correct / total if total > 0 else 0
                avg_loss = total_loss / total if total > 0 else 0
                print(f"[MLP] epoch {epoch+1}/{self.cfg.num_epochs} "
                      f"loss={avg_loss:.4f} acc={acc:.4f}")

        return {}

    def predict(self, embeddings: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        预测快照标签。

        Returns:
            (pred_labels, details):
            - pred_labels: (N,) 0=良性 1=恶意
            - details: {idx: {"prob": float, "logits": array}}
        """
        if self.model is None:
            self.load()

        X = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

        pred_labels = preds.cpu().numpy()
        probs_np = probs.cpu().numpy()

        details = {}
        for i in range(len(pred_labels)):
            if pred_labels[i] == 1:
                details[i] = {
                    "position": int(i),
                    "prob_malicious": float(probs_np[i, 1]),
                    "logits": logits[i].cpu().numpy(),
                }

        n_mal = int(pred_labels.sum())
        print(f"[MLP] 预测: {len(pred_labels)} 个快照, {n_mal} 个恶意")

        return pred_labels, details

    def _save(self):
        """保存模型和元数据"""
        try:
            torch.save(self.model.state_dict(), self.cfg.model_save_path)
            print(f"[MLP] 模型已保存: {self.cfg.model_save_path}")
        except Exception as e:
            print(f"[MLP] 保存模型失败: {e}")

        try:
            meta = {
                "input_dim": self.input_dim,
                "config": self.cfg.__dict__,
            }
            with open(self.cfg.meta_save_path, 'wb') as f:
                pickle.dump(meta, f)
        except Exception as e:
            print(f"[MLP] 保存元数据失败: {e}")

    def load(self):
        """加载已保存的模型"""
        try:
            with open(self.cfg.meta_save_path, 'rb') as f:
                meta = pickle.load(f)
            self.input_dim = meta["input_dim"]
            self.model = self._build_model()
            self.model.load_state_dict(
                torch.load(self.cfg.model_save_path, map_location=self.device)
            )
            self.model.eval()
            print(f"[MLP] 模型已加载: {self.cfg.model_save_path}")
        except Exception as e:
            print(f"[MLP] 加载失败: {e}")
