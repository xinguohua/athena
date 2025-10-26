"""
GCCEmbedderDev: 开发版 Graph Contrastive Coding-style 预训练编码器

说明：这是对 gcc_embedder.py 的开发拷贝，便于独立调参与修改，不影响原版。
主要差异：
- 类名改为 GCCEmbedderDev
- 默认模型路径改为 gcc_encoder_dev.pth
"""
from __future__ import annotations

from typing import Optional, Iterable, Tuple, List, Dict, Union
from collections import Counter
import random
import re
import os
import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from process.embedders.base import GraphEmbedderBase


# ---------------- GIN 组件 ----------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        dim = in_dim
        for _ in range(max(0, num_layers - 1)):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GINConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, eps: float = 0.0, mlp_hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(float(eps)))
        self.mlp = MLP(in_dim, mlp_hidden, out_dim, num_layers=2, dropout=dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x)
        if src.numel() > 0:
            agg.index_add_(0, dst, x[src])
        out = (1.0 + self.eps) * x + agg
        out = self.mlp(out)
        return out


class GINEncoder(nn.Module):
    """GIN 编码器，支持按层吐出中间表示。

    - 默认行为：返回最后一层输出
    - return_all=True：返回 [h1, h2, ..., hL]
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        # 逐层通道维度（用于外部按层时序单元的维度对齐）
        self.layer_dims = [hidden_dim] * (self.num_layers - 1) + [out_dim]
        dims = [in_dim] + [hidden_dim] * (self.num_layers - 1) + [out_dim]
        for i in range(self.num_layers):
            self.layers.append(GINConv(dims[i], dims[i + 1], eps=0.0, mlp_hidden=hidden_dim, dropout=dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, return_all: bool = False):
        h = x
        outs = []
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
            outs.append(h)
        return outs if return_all else outs[-1]


# ---------------- 分层时序递推（GRUCell） ----------------
class TemporalPerLayer(nn.Module):
    """为每一层增加一个 GRUCell 进行时间递推：H_l,t = GRU(Z_l,t, H_l,t-1)

    约定：每层的通道维度不变（d_l -> d_l），便于直接替换或融合。
    """
    def __init__(self, layer_dims: List[int]):
        super().__init__()
        self.layer_dims = [int(d) for d in layer_dims]
        self.grus = nn.ModuleList([nn.GRUCell(d, d) for d in self.layer_dims])
        # 每层的节点隐藏状态表（CPU 常驻）：List[Dict[node_id -> Tensor[d_l]]]
        self.tables: List[Dict[str, torch.Tensor]] = [dict() for _ in self.layer_dims]

    def forward(self, Z_list: List[torch.Tensor], H_prev_list: Optional[List[Optional[torch.Tensor]]] = None) -> List[torch.Tensor]:
        H_list: List[torch.Tensor] = []
        for li, gru in enumerate(self.grus):
            Zl = Z_list[li]
            Hl_prev = None if (H_prev_list is None or li >= len(H_prev_list)) else H_prev_list[li]
            if Hl_prev is None or Hl_prev.shape != Zl.shape:
                Hl_prev = torch.zeros_like(Zl)
            Hl = gru(Zl, Hl_prev)
            H_list.append(Hl)
        return H_list

    def reset(self):
        self.tables = [dict() for _ in self.layer_dims]

    def fetch(self, node_ids: List[str], device) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for li, d in enumerate(self.layer_dims):
            table = self.tables[li]
            if not table:
                out.append(torch.zeros((len(node_ids), int(d)), dtype=torch.float32, device=device))
                continue
            prev_ids = list(table.keys())
            try:
                H_prev = torch.stack([table[nid] for nid in prev_ids], dim=0).to(dtype=torch.float32)
            except Exception:
                out.append(torch.zeros((len(node_ids), int(d)), dtype=torch.float32, device=device))
                continue
            H_aligned = self.align_prev_state(H_prev=H_prev.to(device), curr_ids=node_ids, prev_ids=prev_ids, dim=int(d), device=device)
            out.append(H_aligned)
        return out

    def align_prev_state(self,
                         H_prev: Optional[torch.Tensor],
                         curr_ids: List[str],
                         prev_ids: Optional[List[str]],
                         dim: int,
                         device) -> torch.Tensor:
        """将上一时刻的隐状态按当前节点顺序对齐；缺失节点填 0。"""
        if H_prev is None or not prev_ids:
            return torch.zeros((len(curr_ids), dim), device=device, dtype=torch.float32)

        id2pos_prev = {nid: i for i, nid in enumerate(prev_ids)}
        N_curr = len(curr_ids)
        out = torch.zeros((N_curr, dim), device=device, dtype=H_prev.dtype)
        idx_prev = []
        idx_curr = []

        for i, nid in enumerate(curr_ids):
            j = id2pos_prev.get(nid, -1)
            if j >= 0:
                idx_curr.append(i)
                idx_prev.append(j)

        if idx_curr:
            ic = torch.tensor(idx_curr, device=device, dtype=torch.long)
            ip = torch.tensor(idx_prev, device=device, dtype=torch.long)
            out[ic] = H_prev[ip]

        return out

    def commit(self, node_ids: List[str], H_list: List[torch.Tensor]):
        for li in range(min(len(self.tables), len(H_list))):
            table = self.tables[li]
            Hl = H_list[li]
            for i, nid in enumerate(node_ids):
                table[nid] = Hl[i].detach().to('cpu').contiguous()


# ---------------- GCC Embedder Dev ----------------
class GCCEmbedderDev(GraphEmbedderBase):
    _default_path = 'gcc_encoder_dev.pth'

    def __init__(
        self,
        snapshots,
        features=None,
        mapp=None,
        # 输入/编码器尺寸
        prop_feat_dim: int = 128,
        enc_hidden_dim: int = 128,
        enc_out_dim: int = 256,
        gin_layers: int = 3,
        dropout: float = 0.1,
        # 训练
        num_epochs: int = 30,
        steps_per_epoch: int = 200,
        batch_size: int = 64,
        # batch_size: int = 10,
        lr: float = 1e-3,
        # 对比学习
        temperature: float = 0.2,
        # 子图采样
        r_hop: int = 2,
        ego_max_nodes: int = 64,
        # 增强
        drop_edge_p: float = 0.2,
        feat_mask_p: float = 0.2,
        # 训练集选择
        train_indices: Optional[Union[Iterable[int], Tuple[int, int], int]] = None,
        model_path: Optional[str] = None,
        # 异常活跃驱动损失参数（仅频率异常）
        anomaly_alpha: float = 1.0,      # 加权强度，>0 表示异常越大权重越大
        w2v_window: int = 5,
        w2v_min_count: int = 1,
        w2v_sg: int = 1,
        w2v_epochs: int = 20,
        w2v_pretrained_path: Optional[str] = None,
    ):
        super().__init__(snapshots, features, mapp)
        self.snapshots = snapshots
        self.prop_feat_dim = int(prop_feat_dim)
        self.enc_hidden_dim = int(enc_hidden_dim)
        self.enc_out_dim = int(enc_out_dim)
        self.gin_layers = int(gin_layers)
        self.dropout = float(dropout)
        self.num_epochs = int(num_epochs)
        self.steps_per_epoch = int(steps_per_epoch)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.temperature = float(temperature)
        self.r_hop = int(r_hop)
        self.ego_max_nodes = int(ego_max_nodes)
        self.drop_edge_p = float(drop_edge_p)
        self.feat_mask_p = float(feat_mask_p)
        self.model_path = model_path or self._default_path
        # 异常活跃参数（仅频率异常）
        self.anomaly_alpha = float(anomaly_alpha)

        # Word2Vec 配置（唯一特征来源）
        self.w2v_window = int(w2v_window)
        self.w2v_min_count = int(w2v_min_count)
        self.w2v_sg = int(w2v_sg)
        self.w2v_epochs = int(w2v_epochs)
        self.w2v_pretrained_path = w2v_pretrained_path
        self._w2v_model = None  # 延迟加载/训练

        # 语义相似度参数
        # - sem_fp_bits: 指纹长度（哈希位数），用于快速近似 Tanimoto 计算
        # - sem_push_weight: 对不相似对在分母端增强推开，0 表示不额外增强
        self.sem_fp_bits = 1024
        self.sem_push_weight = 0.0

        # 是否使用“恶意语料”来生成额外负样本；以及腐化强度与每个节点替换的 token 数
        self.use_malicious_negatives = True
        self.mal_neg_ratio: float = 0.3  # 每个子图中替换为恶意向量的节点比例
        self.mal_neg_token_len: int = 16  # 生成恶意向量时采样的恶意 token 数

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 特征缓存（properties -> 向量）
        self._prop_cache: Dict[str, np.ndarray] = {}
        # 恶意 token 容器：累计出现在恶意节点及其邻域的 tokens
        self.malicious_token_counter = Counter()

        # 训练快照索引
        self.train_snapshot_indices = self._resolve_train_indices(train_indices)

        # 编码器 + 投影头（对比用）
        in_dim = self.prop_feat_dim if self.prop_feat_dim > 0 else 1
        self.encoder = GINEncoder(in_dim, self.enc_hidden_dim, self.enc_out_dim, num_layers=self.gin_layers, dropout=self.dropout).to(self.device)
        self.proj_head = nn.Sequential(
            nn.Linear(self.enc_out_dim, self.enc_out_dim),
            nn.ReLU(),
            nn.Linear(self.enc_out_dim, self.enc_out_dim),
        ).to(self.device)
        # 分层时序单元（训练期使用，内置状态管理）
        self.temporal = TemporalPerLayer(self.encoder.layer_dims).to(self.device)
        # 优化器包含 encoder、projection head、temporal
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.proj_head.parameters()) + list(self.temporal.parameters()),
            lr=self.lr,
            weight_decay=1e-4,
        )

        # 训练后缓存：每快照一个 {node_id: vec}
        self.snapshot_node_embeddings: List[Dict[str, np.ndarray]] = []

        # 初始化时序状态
        self.temporal.reset()

    def train(self):
        """动态图对比学习主训练循环（优化版）"""
        if not self.train_snapshot_indices:
            raise RuntimeError("没有可用于训练的快照。请检查 train_indices 设置。")

        # 初始化词向量和负样本池
        self._ensure_w2v_model()
        if self.use_malicious_negatives:
            self._precollect_malicious_tokens()

        print(
            f"[GCC-Dev] Pretrain on {len(self.train_snapshot_indices)} snapshots | batch={self.batch_size} | tau={self.temperature}")

        for epoch in range(self.num_epochs):
            self.temporal.reset()  # 每个 epoch 重置时序状态
            epoch_loss = 0.0
            steps_done = 0

            # 按时间顺序遍历 snapshot
            for sidx in sorted(self.train_snapshot_indices):
                g = self.snapshots[sidx]
                if g is None or g.vcount() == 0:
                    continue

                # 对当前 snapshot 训练一次 batch
                batch_loss = self._train_one_snapshot(g)
                epoch_loss += batch_loss
                steps_done += 1

                print(f"[GCC-Dev] Epoch {epoch + 1}/{self.num_epochs} | Snapshot {sidx} | Loss={batch_loss:.6f}")

            avg = epoch_loss / max(1, steps_done)
            print(f"[GCC-Dev] Epoch {epoch + 1}/{self.num_epochs} DONE | AvgLoss={avg:.6f}")

        # 训练结束后生成节点嵌入（静态路径，若需时序请调用 use_temporal=True）并保存模型
        self.generate_node_embeddings(use_temporal=False)
        self.save_model()

    def _train_one_snapshot(self, g) -> float:
        """在单个 snapshot 上执行一轮训练"""
        Z_chunks: List[torch.Tensor] = []
        sample_weights: List[float] = []
        commit_payloads: List[Tuple[List[str], List[torch.Tensor]]] = []
        # 统一对比损失需要的缓存
        row_owner: List[int] = []              # 每一行向量属于哪个样本（-1 表示纯负样本，不参与正对）
        sample_rows: List[List[int]] = []      # 每个样本对应的行索引列表（通常每样本2行：两视角）
        sample_fps: List[np.ndarray] = []      # 每个样本一个领域指纹（用于语义相似度）

        self.optimizer.zero_grad()

        # 遍历所有节点作为中心（不再随机采样）
        for center in range(g.vcount()):
            sub = self._ego_subgraph(g, center, r=self.r_hop, max_nodes=self.ego_max_nodes)
            if sub.vcount() == 0:
                continue

            # 节点特征与边
            x_np = self._build_node_features(sub)
            edge_index = self._igraph_edges_to_edge_index(sub)
            x = torch.from_numpy(x_np).to(self.device)
            curr_ids = [sub.vs[i]['name'] for i in range(sub.vcount())]

            # 获取上一个时间片的隐藏状态
            H_prev_aligned = self.temporal.fetch(curr_ids, device=self.device)
            e1 = edge_index
            x1 = x
            Z_list_1 = self.encoder(x1, e1, return_all=True)
            H_list_1 = self.temporal(Z_list_1, H_prev_aligned)
            z_1 = self.proj_head(H_list_1[-1].mean(dim=0, keepdim=True))
            Z_chunks.append(F.normalize(z_1, dim=-1))

            # 记录行归属（单视角属于该“样本”）
            sample_id = len(sample_rows)
            sample_rows.append([])
            # 将刚刚追加的该行标记为该样本
            row_i = len(Z_chunks) - 1
            row_owner.append(sample_id)
            sample_rows[sample_id].append(row_i)

            # 生成样本的领域指纹
            try:
                fp = self._subgraph_fingerprint(sub, m_bits=int(self.sem_fp_bits)) if int(self.sem_fp_bits) > 0 else np.zeros(0, dtype=np.float32)
            except Exception:
                fp = np.zeros(int(max(1, self.sem_fp_bits)), dtype=np.float32)
            sample_fps.append(fp)

            # 可选：恶意负样本
            extra_views = 0
            if self.use_malicious_negatives and len(self.malicious_token_counter) > 0:
                x_neg_np = self._corrupt_features_with_malicious(
                    sub, x_np, ratio=self.mal_neg_ratio, token_len=self.mal_neg_token_len
                )
                x_neg = torch.from_numpy(x_neg_np).to(self.device)

                for _ in range(2):
                    e_neg = self._augment_edges(edge_index, drop_p=self.drop_edge_p)
                    x_neg_aug = self._augment_features(x_neg, mask_p=self.feat_mask_p)
                    Z_list_neg = self.encoder(x_neg_aug, e_neg, return_all=True)
                    H_list_neg = self.temporal(Z_list_neg, H_prev_aligned)
                    z_neg = self.proj_head(H_list_neg[-1].mean(dim=0, keepdim=True))
                    Z_chunks.append(F.normalize(z_neg, dim=-1))
                    # 这些额外视角作为“负样本”，不参与正对
                    row_owner.append(-1)
                    extra_views += 1

            # 样本权重
            try:
                freq_val = float(g.vs[center].get("frequency", 1.0))
            except Exception:
                freq_val = 1.0
            freq_val = max(0.0, freq_val) if np.isfinite(freq_val) else 0.0
            w = 1.0 + max(0.0, self.anomaly_alpha) * freq_val
            for _ in range(1 + extra_views):
                sample_weights.append(w)

            # 保存状态以更新记忆
            commit_payloads.append((curr_ids, [h.detach() for h in H_list_1]))

        # 如果没采到有效子图，跳过
        if len(Z_chunks) < 2:
            return 0.0

        # 统一加权对比损失（SupCon 风格）：实例正对 + 语义正对，分母保留全体（含负样本）
        Z = torch.cat(Z_chunks, dim=0)
        N = Z.size(0)
        device = Z.device
        Z = F.normalize(Z, dim=-1)
        sim = torch.mm(Z, Z.t()) / float(self.temperature)
        eye = torch.eye(N, device=device, dtype=torch.bool)
        sim = sim.masked_fill(eye, -1e9)
        exp_sim = torch.exp(sim)

    # 权重矩阵 W：语义正对=S（分摊到目标样本的各视角）
        W = torch.zeros((N, N), dtype=torch.float32, device=device)

        # 语义正对：按 Tanimoto 相似度加权
        B = len(sample_rows)
        if B >= 2 and len(sample_fps) == B:
            FP_np = np.stack(sample_fps, axis=0).astype(np.float32)
            FP = torch.from_numpy(FP_np).to(device=device, dtype=torch.float32)
            inter = FP @ FP.t()
            a1 = FP.sum(dim=1, keepdim=True)
            denom = (a1 + a1.t() - inter).clamp_min(1e-6)
            S = inter / denom
            S.fill_diagonal_(0.0)
            # 为每个锚 i（属于样本 s），把 λ·S[s,t] 平均分给 t 的各视角 j
            # 负样本行（owner=-1）不参与正对
            # 先反查每行的样本 id
            for s in range(B):
                Rs = sample_rows[s]
                if not Rs:
                    continue
                for t in range(B):
                    if t == s:
                        continue
                    st = float(S[s, t].item())
                    if st <= 0.0:
                        continue
                    Rt = sample_rows[t]
                    if not Rt:
                        continue
                    add_each = st / float(len(Rt))
                    for i in Rs:
                        # i 一定是有效行（属于样本 s）
                        for j in Rt:
                            W[i, j] += add_each

        # 分母加权：对“结构不相似”的对按 (1 + beta*(1-S)) 放大，强化推开
        denom_w = torch.ones((N, N), dtype=torch.float32, device=device)
        if getattr(self, 'sem_push_weight', 0.0) > 0.0 and B >= 2:
            beta = float(self.sem_push_weight)
            # 仅对有样本归属的行/列生效（恶意负样本列自然保留权重1）
            for s in range(B):
                Rs = sample_rows[s]
                if not Rs:
                    continue
                for t in range(B):
                    if t == s:
                        continue
                    Rt = sample_rows[t]
                    if not Rt:
                        continue
                    # 若上面未计算 S，则在此计算一次（通常已在语义正对分支求出；为稳妥再算一遍）
                    if 'S' not in locals():
                        FP_np_tmp = np.stack(sample_fps, axis=0).astype(np.float32)
                        FP_tmp = torch.from_numpy(FP_np_tmp).to(device=device, dtype=torch.float32)
                        inter_tmp = FP_tmp @ FP_tmp.t()
                        a1_tmp = FP_tmp.sum(dim=1, keepdim=True)
                        denom_tmp = (a1_tmp + a1_tmp.t() - inter_tmp).clamp_min(1e-6)
                        S = inter_tmp / denom_tmp
                        S.fill_diagonal_(0.0)
                    st = float(S[s, t].item()) if 'S' in locals() else 0.0
                    scale = 1.0 + beta * max(0.0, (1.0 - st))
                    if scale == 1.0:
                        continue
                    for i in Rs:
                        for j in Rt:
                            denom_w[i, j] = denom_w[i, j] * scale

        # 统一损失：-log( 分子/分母 )
        numerator = (W * exp_sim).sum(dim=1)
        denominator = (denom_w * exp_sim).sum(dim=1).clamp_min(1e-12)
        valid = numerator > 0.0
        if valid.any():
            loss_vec = -torch.log((numerator.clamp_min(1e-12)) / denominator)
            loss_vec = loss_vec[valid]
            w_tensor = torch.tensor(sample_weights, dtype=torch.float32, device=device)
            w_tensor = w_tensor[valid]
            denom_w = w_tensor.sum().clamp_min(1e-6)
            loss = (w_tensor * loss_vec).sum() / denom_w
        else:
            # 没有任何正对（极端情况），退化为 0
            loss = torch.tensor(0.0, dtype=torch.float32, device=device)

        # 反向传播与梯度裁剪
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.proj_head.parameters()) + list(self.temporal.parameters()),
            max_norm=5.0,
        )
        self.optimizer.step()

        # 更新时序记忆
        for node_ids, H_list_t in commit_payloads:
            self.temporal.commit(node_ids, H_list_t)

        return float(loss.detach().cpu().item())


    # ---------- 恶意 tokens 支持（用于负样本） ----------
    def _precollect_malicious_tokens(self):
        if getattr(self, 'malicious_token_counter', None) is None:
            return
        if len(self.malicious_token_counter) > 0:
            return
        # 遍历全部快照，收集 label==1 的节点 tokens 与其 1-hop 邻域 tokens
        for g in self.snapshots:
            if g is None or g.vcount() == 0:
                continue
            for i in range(g.vcount()):
                try:
                    lab = int(g.vs[i].attributes().get('label', 0))
                except Exception:
                    lab = 0
                if lab != 1:
                    continue
                self_tokens = self._get_node_tokens(g, i)
                nei_tokens = self._gather_neighbor_tokens(g, i)
                self.malicious_token_counter.update(self_tokens)
                # self.malicious_token_counter.update(nei_tokens)

    def _sample_malicious_tokens(self, k: int) -> List[str]:
        if len(self.malicious_token_counter) == 0 or k <= 0:
            return []
        items = list(self.malicious_token_counter.items())
        toks = [t for t, _ in items]
        wts = np.array([max(1, int(c)) for _, c in items], dtype=np.float64)
        wts = wts / (wts.sum() + 1e-12)
        # 按权重独立采样 k 次（允许重复）
        idx = np.random.choice(len(toks), size=int(k), replace=True, p=wts)
        return [toks[j] for j in idx]

    def _corrupt_features_with_malicious(self, g, X_base: np.ndarray, ratio: float, token_len: int) -> np.ndarray:
        n = g.vcount()
        out = X_base.copy()
        if ratio <= 0 or len(self.malicious_token_counter) == 0:
            return out
        for i in range(n):
            if random.random() < ratio:
                tokens = self._sample_malicious_tokens(max(1, int(token_len)))
                vec = self._w2v_vector_from_tokens(tokens)
                out[i] = vec.astype(np.float32)
        return out

    # ---------- 语义指纹（Tanimoto）支持 ----------
    def _collect_subgraph_tokens(self, sub, max_tokens: int = 512) -> List[str]:
        """收集子图的领域 tokens（基于节点 properties），限制总量。
        当前实现：合并所有节点的 properties 分词。
        """
        toks: List[str] = []
        for i in range(sub.vcount()):
            toks.extend(self._get_node_tokens(sub, i))
            if len(toks) >= int(max_tokens):
                break
        if len(toks) > int(max_tokens):
            toks = toks[: int(max_tokens)]
        return toks

    def _fingerprint_from_tokens(self, tokens: List[str], m_bits: int = 1024) -> np.ndarray:
        """将 tokens 映射为长度为 m_bits 的 0/1 指纹向量（MD5 哈希取模）。"""
        m = max(1, int(m_bits))
        fp = np.zeros(m, dtype=np.float32)
        if not tokens:
            return fp
        for t in tokens:
            # 使用 MD5 哈希映射到 [0, m)
            h = hashlib.md5(t.encode('utf-8')).hexdigest()
            idx = int(h, 16) % m
            fp[idx] = 1.0
        return fp

    def _subgraph_fingerprint(self, sub, m_bits: int = 1024) -> np.ndarray:
        toks = self._collect_subgraph_tokens(sub)
        return self._fingerprint_from_tokens(toks, m_bits=m_bits)

    def embed_nodes(self):
        return self.snapshot_node_embeddings[-1] if self.snapshot_node_embeddings else {}

    def embed_edges(self):
        return {}

    def prepare_text_encoder(self):
        """可选的显式预处理：提前训练/加载 Word2Vec 模型。
        某些离线流程（例如只做快照嵌入而不调用 train）可先调用本方法。
        """
        self._ensure_w2v_model()

    def get_snapshot_embeddings(self, snapshot_sequence=None):
        if not self.snapshot_node_embeddings:
            raise RuntimeError("还没有节点嵌入，请先调用 train()")
        if snapshot_sequence is None:
            snapshot_sequence = list(range(len(self.snapshots)))
        result: List[np.ndarray] = []
        for i in snapshot_sequence:
            g = self.snapshots[i]
            if g is None:
                result.append(np.zeros(self.enc_out_dim, dtype=np.float32))
                continue
            emb_dict = self.snapshot_node_embeddings[i] if i < len(self.snapshot_node_embeddings) else {}
            if not emb_dict:
                result.append(np.zeros(self.enc_out_dim, dtype=np.float32))
                continue
            try:
                freqs = g.vs['frequency'] if 'frequency' in g.vs.attributes() else None
            except Exception:
                freqs = None
            degrees = g.degree()
            weighted = np.zeros(self.enc_out_dim, dtype=np.float32)
            total_w = 0.0
            for local_idx in range(g.vcount()):
                nid = g.vs[local_idx]['name']
                vec = emb_dict.get(nid)
                if vec is None:
                    continue
                if freqs is not None:
                    try:
                        w = float(freqs[local_idx])
                    except Exception:
                        w = 0.0
                    if not np.isfinite(w) or w < 0:
                        w = 0.0
                else:
                    w = float(degrees[local_idx])
                if w <= 0:
                    continue
                weighted += (w * vec)
                total_w += w
            if total_w <= 0:
                allv = np.array(list(emb_dict.values()), dtype=np.float32)
                gvec = allv.mean(axis=0) if allv.size > 0 else np.zeros(self.enc_out_dim, dtype=np.float32)
            else:
                gvec = weighted / (total_w + 1e-12)
            gvec = gvec / (np.linalg.norm(gvec) + 1e-12)
            result.append(gvec.astype(np.float32))
        arr = np.vstack(result).astype(np.float32) if result else np.zeros((0, self.enc_out_dim), dtype=np.float32)
        print(f"[GCC-Dev] Snapshot embeddings: {arr.shape}")
        return arr

    def save_model(self, path: Optional[str] = None):
        path = path or self.model_path
        state = {
            'params': {
                'prop_feat_dim': self.prop_feat_dim,
                'enc_hidden_dim': self.enc_hidden_dim,
                'enc_out_dim': self.enc_out_dim,
                'gin_layers': self.gin_layers,
                'dropout': self.dropout,
                'num_epochs': self.num_epochs,
                'steps_per_epoch': self.steps_per_epoch,
                'batch_size': self.batch_size,
                'lr': self.lr,
                'temperature': self.temperature,
                'r_hop': self.r_hop,
                'ego_max_nodes': self.ego_max_nodes,
                'drop_edge_p': self.drop_edge_p,
                'feat_mask_p': self.feat_mask_p,
                'train_indices': self.train_snapshot_indices,
                'model_path': self.model_path,
                'anomaly_alpha': self.anomaly_alpha,
                # Semantic settings
                'sem_fp_bits': self.sem_fp_bits,
                'sem_push_weight': self.sem_push_weight,
                # W2V 配置
                'w2v_window': self.w2v_window,
                'w2v_min_count': self.w2v_min_count,
                'w2v_sg': self.w2v_sg,
                'w2v_epochs': self.w2v_epochs,
                'w2v_pretrained_path': self.w2v_pretrained_path,
            },
            'encoder': self.encoder.state_dict(),
            'proj_head': self.proj_head.state_dict(),
            'temporal': self.temporal.state_dict(),
            'snapshot_node_embeddings': self.snapshot_node_embeddings,
        }
        torch.save(state, path)
        print(f"[GCC-Dev] Model saved to {path}")

    @classmethod
    def load(cls, snapshot_sequence, path: Optional[str] = None):
        path = path or cls._default_path
        print(f"[GCC-Dev] Loading model from {path}…")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state = torch.load(path, map_location=device)
        raw_params = dict(state.get('params', {}))
        allowed = {
            'prop_feat_dim','enc_hidden_dim','enc_out_dim','gin_layers','dropout',
            'num_epochs','steps_per_epoch','batch_size','lr','temperature',
            'r_hop','ego_max_nodes','drop_edge_p','feat_mask_p','train_indices','model_path',
            'anomaly_alpha',
            # W2V 配置
            'w2v_window','w2v_min_count','w2v_sg','w2v_epochs','w2v_pretrained_path'
        }
        params = {k: v for k, v in raw_params.items() if k in allowed}
        inst = cls(snapshot_sequence, **params)
        # 恢复语义拉近配置（不作为构造参数传入，以保持构造签名简洁）
        try:
            inst.sem_fp_bits = int(raw_params.get('sem_fp_bits', inst.sem_fp_bits))
        except Exception:
            pass
        try:
            inst.sem_push_weight = float(raw_params.get('sem_push_weight', inst.sem_push_weight))
        except Exception:
            pass
        inst.encoder.load_state_dict(state['encoder'])
        inst.proj_head.load_state_dict(state['proj_head'])
        if 'temporal' in state:
            try:
                inst.temporal.load_state_dict(state['temporal'])
            except Exception as e:
                print(f"[GCC-Dev] Warning: 加载 temporal 失败：{e}")
        inst.snapshot_node_embeddings = state.get('snapshot_node_embeddings', [])
        print("[GCC-Dev] Model loaded successfully")
        return inst

    # ---------- 内部工具 ----------
    def _resolve_train_indices(self, indices: Optional[Union[Iterable[int], Tuple[int, int], int]]) -> List[int]:
        total = len(self.snapshots)
        if total == 0:
            return []
        if indices is None:
            raw = list(range(total))
        elif isinstance(indices, int):
            raw = [indices]
        elif isinstance(indices, tuple) and len(indices) == 2:
            a, b = int(indices[0]), int(indices[1])
            if a > b:
                a, b = b, a
            raw = list(range(a, b + 1))
        else:
            raw = list(indices)  # type: ignore[arg-type]
        valid = sorted({int(i) for i in raw if 0 <= int(i) < total})
        if not valid:
            raise ValueError("train_indices 不包含有效索引")
        return valid

    def _ego_subgraph(self, g, center: int, r: int, max_nodes: int):
        try:
            nodes = set(g.neighborhood(vertices=center, order=r))
        except Exception:
            nodes = {center}
        if len(nodes) > max_nodes:
            # 保留中心 + 采样其他
            nodes = {center} | set(random.sample(list(nodes - {center}), k=max_nodes - 1))
        nodes_sorted = sorted(nodes)
        return g.subgraph(nodes_sorted)

    # ---------- Word2Vec 支持 ----------
    def _tokenize_properties(self, text: str) -> List[str]:
        if not text:
            return []
        return [t for t in re.split(r'[^A-Za-z0-9]+', str(text).lower()) if t]

    def _get_node_tokens(self, g, i: int) -> List[str]:
        try:
            prop = g.vs[i]['properties']
        except Exception:
            prop = g.vs[i].attributes().get('properties', '')
        return self._tokenize_properties(str(prop))

    def _gather_neighbor_tokens(self, g, i: int) -> List[str]:
        """固定策略的邻域 token 收集：1-hop，包含自身，限制 256 个 token。"""
        try:
            nodes = set(g.neighborhood(vertices=i, order=1))
        except Exception:
            nodes = {i}
        out: List[str] = []
        for nid in nodes:
            out.extend(self._get_node_tokens(g, nid))
            if len(out) >= 256:
                break
        if len(out) > 256:
            out = out[:256]
        return out

    def _augment_tokens(self, tokens: List[str]) -> List[str]:
        """对 token 做轻量样本增强：
        - 10% 概率丢弃单词
        - 追加顺序 bigram（相邻两词拼接）
        - 最多保留 256 个 token
        """
        if not tokens:
            return []
        # 随机丢弃
        kept = [t for t in tokens if random.random() > 0.1]
        if not kept:
            kept = list(tokens)
        # 追加 bigram
        bigrams = [kept[i] + '_' + kept[i + 1] for i in range(len(kept) - 1)] if len(kept) > 1 else []
        out = kept + bigrams
        if len(out) > 256:
            out = out[:256]
        return out

    def _collect_w2v_corpus(self) -> List[List[str]]:
        seen_props: Dict[str, List[str]] = {}
        ids = self.train_snapshot_indices or list(range(len(self.snapshots)))
        for sidx in ids:
            g = self.snapshots[sidx]
            if g is None or g.vcount() == 0:
                continue
            for i in range(g.vcount()):
                try:
                    prop = g.vs[i]['properties']
                except Exception:
                    prop = g.vs[i].attributes().get('properties', '')
                key = str(prop)
                if key not in seen_props:
                    seen_props[key] = self._tokenize_properties(key)
        corpus = [tokens for tokens in seen_props.values() if tokens]
        return corpus

    def _ensure_w2v_model(self):
        if self._w2v_model is not None:
            return
        try:
            import importlib
            _w2v_mod = importlib.import_module('gensim.models')
            Word2Vec = getattr(_w2v_mod, 'Word2Vec')
        except Exception:
            raise RuntimeError("[GCC-Dev] 需要 gensim 才能使用 Word2Vec 特征，请先安装 gensim。")
        # 预训练优先
        if isinstance(self.w2v_pretrained_path, str) and os.path.exists(self.w2v_pretrained_path):
            try:
                self._w2v_model = Word2Vec.load(self.w2v_pretrained_path)
                vec_dim = int(getattr(self._w2v_model.wv, 'vector_size', self.prop_feat_dim))
                if int(vec_dim) == int(self.prop_feat_dim):
                    print("[GCC-Dev] 已加载预训练 Word2Vec 模型。")
                    return
                else:
                    print(f"[GCC-Dev] 预训练向量维度({vec_dim}) != prop_feat_dim({self.prop_feat_dim})，将改为自训练以匹配维度。")
                    self._w2v_model = None
            except Exception as e:
                print(f"[GCC-Dev] 加载预训练 Word2Vec 失败：{e}，将尝试自训练。")
        # 自训练
        corpus = self._collect_w2v_corpus()
        if not corpus:
            raise RuntimeError("[GCC-Dev] W2V 语料为空，无法构建 Word2Vec 特征。")
        print(f"[GCC-Dev] 正在训练word2vec | 语料={len(corpus)} | dim={int(self.prop_feat_dim)} | window={int(self.w2v_window)} | min_count={int(self.w2v_min_count)} | sg={int(self.w2v_sg)} | epochs={int(self.w2v_epochs)}")
        self._w2v_model = Word2Vec(
            sentences=corpus,
            vector_size=int(self.prop_feat_dim),
            window=int(self.w2v_window),
            min_count=int(self.w2v_min_count),
            sg=int(self.w2v_sg),
            workers=4,
            epochs=int(self.w2v_epochs),
        )
        print(f"[GCC-Dev] 训练 Word2Vec 完成：语料={len(corpus)} 条，dim={int(self.prop_feat_dim)}")

    def _w2v_vector_from_tokens(self, tokens: List[str]) -> np.ndarray:
        if not tokens or self._w2v_model is None:
            return np.zeros(int(self.prop_feat_dim), dtype=np.float32)
        vecs = []
        wv = self._w2v_model.wv
        for t in tokens:
            if t in wv:
                vecs.append(wv[t])
        if not vecs:
            return np.zeros(int(self.prop_feat_dim), dtype=np.float32)
        v = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
        n = np.linalg.norm(v) + 1e-12
        return (v / n).astype(np.float32)

    def _build_node_features(self, g) -> np.ndarray:
        n = g.vcount()
        if self.prop_feat_dim <= 0:
            # 用常数特征占位
            return np.ones((n, 1), dtype=np.float32)
        # 确保 W2V 模型就绪（延迟）
        if self._w2v_model is None:
            self._ensure_w2v_model()
        X = np.zeros((n, int(self.prop_feat_dim)), dtype=np.float32)
        for i in range(n):
            # 原始路径：仅节点自身 properties → tokens → 向量（带缓存）
            try:
                prop = g.vs[i]['properties']
            except Exception:
                prop = g.vs[i].attributes().get('properties', '')
            key = str(prop)
            if key in self._prop_cache:
                X[i] = self._prop_cache[key]
                continue
            tokens = self._tokenize_properties(key)
            vec = self._w2v_vector_from_tokens(tokens)
            self._prop_cache[key] = vec
            X[i] = vec
        return X

    def get_malicious_top_tokens(self, k: int = 50):
        """返回收集到的恶意 token 的 Top-K 高频列表 [(token, count), ...]。"""
        return self.malicious_token_counter.most_common(int(k))

    def _igraph_edges_to_edge_index(self, g) -> torch.Tensor:
        edges = g.get_edgelist()
        if len(edges) == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)
        src = []
        dst = []
        for u, v in edges:
            src.append(u)
            dst.append(v)
            # 无向化
            src.append(v)
            dst.append(u)
        return torch.tensor([src, dst], dtype=torch.long, device=self.device)

    def _augment_edges(self, edge_index: torch.Tensor, drop_p: float) -> torch.Tensor:
        if edge_index.numel() == 0 or drop_p <= 0:
            return edge_index
        E = edge_index.size(1)
        keep = torch.rand(E, device=edge_index.device) > drop_p
        if keep.sum() < 1:
            # 至少保留一条边（若原本有边）
            keep[random.randrange(0, E)] = True
        return edge_index[:, keep]

    def _augment_features(self, x: torch.Tensor, mask_p: float) -> torch.Tensor:
        if x.numel() == 0 or mask_p <= 0:
            return x
        mask = (torch.rand_like(x) < mask_p).float()
        return x * (1.0 - mask)

    def _nt_xent_loss(self, Z: torch.Tensor, temperature: float, sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Z: [2N, D], 每相邻两行构成一对正样本 (x1, x2)
        Z = F.normalize(Z, dim=-1)
        sim = torch.mm(Z, Z.t()) / temperature  # [2N,2N]
        N2 = Z.size(0)
        labels = torch.arange(N2, device=Z.device)
        pos = labels ^ 1  # 0<->1, 2<->3, ... （相邻为正对）
        # 去除对角自身
        mask = torch.eye(N2, device=Z.device).bool()
        sim = sim.masked_fill(mask, -1e9)
        loss_vec = F.cross_entropy(sim, pos, reduction='none')  # [2N]
        if sample_weights is not None:
            # 归一化权重，防止梯度过大
            w = sample_weights.clamp_min(0.0)
            denom = w.sum().clamp_min(1e-6)
            return (w * loss_vec).sum() / denom
        else:
            return loss_vec.mean()

    def generate_node_embeddings(self, use_temporal: bool = False):
        """生成节点嵌入（单一实现，use_temporal 一个开关）。
        - use_temporal=False: 仅 encoder（静态）
        - use_temporal=True: 重置时序记忆后，按时间顺序 fetch→encoder(return_all=True)→temporal→commit（干净视角）
        结果写入 self.snapshot_node_embeddings
        """
        self.encoder.eval()
        if use_temporal:
            # 统一语义：推理前一律重置，确保可复现与独立性
            self.temporal.reset()
        self.snapshot_node_embeddings.clear()
        with torch.no_grad():
            for g in self.snapshots:
                if g is None or g.vcount() == 0:
                    self.snapshot_node_embeddings.append({})
                    continue
                x_np = self._build_node_features(g)
                eidx = self._igraph_edges_to_edge_index(g)
                x = torch.from_numpy(x_np).to(self.device)
                curr_ids = [g.vs[i]['name'] for i in range(g.vcount())]
                if use_temporal:
                    H_prev = self.temporal.fetch(curr_ids, device=self.device)
                    Z_list = self.encoder(x, eidx, return_all=True)
                    H_list = self.temporal(Z_list, H_prev)
                    # commit 干净视角的状态
                    self.temporal.commit(curr_ids, [h.detach() for h in H_list])
                    h_last = H_list[-1]
                else:
                    h_last = self.encoder(x, eidx)
                emb_dict: Dict[str, np.ndarray] = {}
                for i in range(g.vcount()):
                    nid = g.vs[i]['name']
                    emb_dict[nid] = h_last[i].detach().cpu().numpy().astype(np.float32)
                self.snapshot_node_embeddings.append(emb_dict)
        mode = 'temporal' if use_temporal else 'static'
        print(f"[GCC-Dev] Generated {mode} node embeddings: {len(self.snapshot_node_embeddings)} snapshots")

