import os
from dataclasses import dataclass
from typing import Sequence, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict
from tqdm import tqdm
from typing import Optional
from .base import BaseClassify


# ========== 模型 ==========
class AnomalyDetector(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, num_layers: int,
                 dropout: float, kernel_sizes: Sequence[int], num_filters: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_dim * 2, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)          # (B, L, 2H)
        conv_in = lstm_out.permute(0, 2, 1) # (B, 2H, L)

        pooled = []
        for conv in self.convs:
            conv_out = F.relu(conv(conv_in))
            pooled_out = F.max_pool1d(conv_out, kernel_size=conv_out.shape[2]).squeeze(2)
            pooled.append(pooled_out)

        out = torch.cat(pooled, dim=1)
        out = self.dropout(out)
        return self.fc(out)


# ========== Trainer 配置 ==========
@dataclass
class PrographerConfig:
    sequence_length_L: int = 12
    embedding_dim: int = 256
    hidden_dim: int = 128
    num_layers: int = 5
    dropout_rate: float = 0.2
    kernel_sizes: Tuple[int, ...] = (3, 4, 5)
    num_filters: int = 100
    seq_lr: float = 3e-4
    seq_epochs: int = 30
    batch_size: int = 32
    patience: int = 5
    grad_clip_norm: float = 1.0
    model_save_path: str = "prographer_detector.pth"


# ========== Trainer 实现 ==========
class PrographerClassify(BaseClassify):
    def __init__(self, cfg: Optional[PrographerConfig] = None, **kwargs):
        super().__init__()
        self.cfg = cfg or PrographerConfig()
        # 允许动态覆盖配置
        for k, v in kwargs.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self) -> nn.Module:
        """返回具体模型"""
        return AnomalyDetector(
            embedding_dim=self.cfg.embedding_dim,
            hidden_dim=self.cfg.hidden_dim,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout_rate,
            kernel_sizes=self.cfg.kernel_sizes,
            num_filters=self.cfg.num_filters,
        ).to(self.device)

    @staticmethod
    def _make_windows(x: torch.Tensor, L: int, pad_value: float = 0.0):
        """构造滑动窗口序列"""
        T, D = x.size(0), x.size(1)
        if T < L:
            pad_len = L - T
            padded = torch.cat([x, torch.full((pad_len, D), pad_value, device=x.device)], dim=0)
            return padded.unsqueeze(0), x[-1].unsqueeze(0)

        seqs, tars = [], []
        for i in range(T - L + 1):
            seqs.append(x[i: i + L])
            tars.append(x[i + L - 1])
        return torch.stack(seqs), torch.stack(tars)

    def _train_loop(self, snapshot_embeddings: Any, **kwargs) -> Dict[str, list]:
        """训练循环，返回训练历史"""
        cfg = self.cfg
        print(f"[Trainer] device={self.device}, config={cfg}")

        x = torch.as_tensor(snapshot_embeddings, dtype=torch.float32)
        if x.size(0) <= cfg.sequence_length_L:
            raise ValueError(f"快照数量 {x.size(0)} 不足以构成 {cfg.sequence_length_L} 序列")

        split = int(0.8 * x.size(0))
        train_x, train_y = self._make_windows(x[:split], cfg.sequence_length_L)
        val_x, val_y     = self._make_windows(x[split:], cfg.sequence_length_L)

        train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=cfg.batch_size, shuffle=True)
        val_loader   = DataLoader(TensorDataset(val_x, val_y), batch_size=cfg.batch_size)

        optimizer = optim.Adam(self.model.parameters(), lr=cfg.seq_lr)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=3)
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        best_val, bad_epochs = float("inf"), 0
        best_state = OrderedDict((k, v.cpu()) for k, v in self.model.state_dict().items())
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(cfg.seq_epochs):
            # --- Train ---
            self.model.train()
            total = 0.0
            for xb, yb in tqdm(train_loader, desc=f"Train {epoch+1}/{cfg.seq_epochs}", leave=False):
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    pred = self.model(xb)
                    loss = criterion(pred, yb)
                    # print(f"DEBUG SHAPES (train) xb={tuple(xb.shape)} pred={tuple(pred.shape)} yb={tuple(yb.shape)}")

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                total += loss.item() * xb.size(0)
            train_loss = total / len(train_x)

            # --- Val ---
            self.model.eval()
            vtotal = 0.0
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = self.model(xb)
                    vtotal += criterion(pred, yb).item() * xb.size(0)
                    # print(f"DEBUG SHAPES (val) xb={tuple(xb.shape)} pred={tuple(pred.shape)} yb={tuple(yb.shape)}")

            val_loss = vtotal / len(val_x)
            scheduler.step(val_loss)

            print(f"Epoch {epoch+1}: Train {train_loss:.6f} | Val {val_loss:.6f}")
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            # --- Save Best ---
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    print("[EarlyStop] Triggered.")
                    break

        # ===== 保存最优模型 =====
        self.model.load_state_dict(best_state)
        save_dir = os.path.dirname(cfg.model_save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        torch.save(self.model.state_dict(), cfg.model_save_path)
        print(f"[Save] model -> {cfg.model_save_path}")
        return history

    # float = 0.016/0.0022/0.004
    def predict(self, embeddings: np.ndarray, threshold: float = 0.0003) -> Tuple[np.ndarray, Dict]:
        """
        用训练好的模型预测快照是否异常
        """
        assert self.model is not None, "model 未训练或未加载"
        self.model.eval()

        x = torch.as_tensor(embeddings, dtype=torch.float32, device=self.device)
        seq_len = self.cfg.sequence_length_L
        pred_labels = np.zeros(len(x), dtype=int)
        diff_vectors, scores = {}, {}

        with torch.no_grad():
            for i in range(len(x)):
                # 构造输入序列
                if i < seq_len:
                    pad = torch.zeros(seq_len - i, x.size(1), device=self.device)
                    seq = torch.cat([pad, x[:i]], dim=0) if i > 0 else x[0].repeat(seq_len, 1)
                else:
                    seq = x[i - seq_len:i]

                seq = seq.unsqueeze(0)
                target = x[i]
                pred = self.model(seq).squeeze(0)
                error = F.mse_loss(pred, target).item()
                scores[i] = error
                if error > threshold:
                    pred_labels[i] = 1
                    diff_vectors[i] = {
                        "position": i,
                        "error": error,
                        "diff_vector": (pred - target).cpu().numpy(),
                        "real_embedding": target.cpu().numpy(),
                        "pred_embedding": pred.cpu().numpy(),
                    }
        print("\n--- 快照检测结果 ---")
        print("快照索引 | 得分(Error) | 状态")
        print("-" * 40)
        for i, score in scores.items():
            status = "🔴 异常" if score > threshold else "🟢 正常"
            print(f"快照 {i:2d}   | {score:.6f}   | {status}")
        print("-" * 40 + "\n")
        return pred_labels, diff_vectors

    def load(self, path=None):
        path = path or self.cfg.model_save_path
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(path))
        return self.model