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
import os
import re
import hashlib
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
try:
    from tqdm import tqdm as _tqdm
except Exception:
    def _tqdm(x, **kwargs):
        return x
from .base import GraphEmbedderBase


# ----------------------- GIN 基元 -----------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GINConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.mlp = MLP(in_dim, out_dim, out_dim, dropout)
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            agg = torch.zeros_like(x)
        else:
            src, dst = edge_index[0], edge_index[1]
            agg = torch.zeros_like(x)
            agg.index_add_(0, dst, x[src])
        out = self.mlp((1 + self.eps) * x + agg)
        return out


class GINEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        num_layers = int(max(1, num_layers))
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([GINConv(dims[i], dims[i + 1], dropout) for i in range(num_layers)])
        self.layer_dims = dims[1:]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, return_all: bool = False):
        Zs = []
        h = x
        for conv in self.layers:
            h = conv(h, edge_index)
            h = F.relu(h)
            Zs.append(h)
        return Zs if return_all else h


class TemporalPerLayer(nn.Module):
    def __init__(self, layer_dims: List[int]):
        super().__init__()
        self.layer_dims = [int(d) for d in layer_dims]
        self.cells = nn.ModuleList([nn.GRUCell(d, d) for d in self.layer_dims])
        self.tables: List[Dict[str, torch.Tensor]] = [dict() for _ in self.layer_dims]

    def reset(self):
        for t in self.tables:
            t.clear()

    def state_dict(self):
        return {'cells': self.cells.state_dict()}

    def load_state_dict(self, state):
        try:
            self.cells.load_state_dict(state.get('cells', {}))
        except Exception:
            pass

    def fetch(self, node_ids: List[str], device: torch.device) -> List[torch.Tensor]:
        H_prev: List[torch.Tensor] = []
        n = len(node_ids)
        for li, dim in enumerate(self.layer_dims):
            table = self.tables[li]
            H = torch.zeros((n, dim), dtype=torch.float32, device=device)
            for i, nid in enumerate(node_ids):
                if nid in table:
                    H[i] = table[nid].to(device)
            H_prev.append(H)
        return H_prev

    def forward(self, Z_list: List[torch.Tensor], H_prev: List[torch.Tensor]) -> List[torch.Tensor]:
        H_list: List[torch.Tensor] = []
        for li, cell in enumerate(self.cells):
            z = Z_list[li]
            h0 = H_prev[li]
            h1 = cell(z, h0)
            H_list.append(h1)
        return H_list

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
        num_epochs: int = 1,
        steps_per_epoch: int = 200,
        batch_size: int = 1000,
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
        # 相似度/权重相关可选参数
        sim_measure: str = 'wl',            # 'tanimoto' | 'cosine' | 'wl'
        wl_height: int = 2,
        sem_push_weight: float = 0.0,
        sem_fp_bits: int = 1024,
        # 恶意负样本与推开强度
        use_malicious_negatives: bool = True,
        mal_neg_ratio: float = 0.3,
        mal_neg_token_len: int = 16,
        mal_neg_push_gamma: float = 3.0,
        # Top-K 相似（可选）
        topk_pos: Optional[int] = 3,   # 每个 anchor 选择的 Top-K 相似正样本（基于 S）
        topk_pos_min_sim: float = 0.0, # 仅当相似度 > 此阈值时才将样本纳入 Top-K 正样本
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
        self.sem_fp_bits = int(sem_fp_bits)
        self.sem_push_weight = float(sem_push_weight)
        # 相似度度量方式：'tanimoto' | 'cosine' | 'wl'
        self.sim_measure = str(sim_measure)
        # WL 子树核参数（用于 sim_measure='wl'）
        self.wl_height = int(wl_height)

        # 是否使用“恶意语料”来生成额外负样本；以及腐化强度与每个节点替换的 token 数
        self.use_malicious_negatives = bool(use_malicious_negatives)
        self.mal_neg_ratio = float(mal_neg_ratio)  # 每个子图中替换为恶意向量的节点比例
        self.mal_neg_token_len = int(mal_neg_token_len)  # 生成恶意向量时采样的恶意 token 数
        self.mal_neg_push_gamma = float(mal_neg_push_gamma)

        # Top-K 采样配置
        self.topk_pos = int(topk_pos) if topk_pos is not None else None
        self.topk_pos_min_sim = float(topk_pos_min_sim)

        # 调试参数（只保留一个开关，其它使用内置默认值；默认关闭，可运行时直接改属性开启）
        self.debug_sim_dump = True
        self.debug_dump_dir = './gcc_debug'
        self.debug_rows_per_batch = 100
        self.debug_max_batches = 1
        self._debug_dumped_batches = 0

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 特征缓存（properties -> 向量）
        self._prop_cache = {}
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
        self.snapshot_node_embeddings = []

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
                batch_loss = self._train_one_snapshot(g, sidx=sidx)
                epoch_loss += batch_loss
                steps_done += 1

                print(f"[GCC-Dev] Epoch {epoch + 1}/{self.num_epochs} | Snapshot {sidx} | Loss={batch_loss:.6f}")

            avg = epoch_loss / max(1, steps_done)
            print(f"[GCC-Dev] Epoch {epoch + 1}/{self.num_epochs} DONE | AvgLoss={avg:.6f}")

        # 训练结束后生成节点嵌入（静态路径，若需时序请调用 use_temporal=True）并保存模型
        self.generate_node_embeddings(use_temporal=False)
        self.save_model()

    def _train_one_snapshot(self, g, sidx: Optional[int] = None) -> float:
        """单个 snapshot 训练：按 batch 聚合多个中心子图，一次性 GNN 前向。"""
        device = self.device
        centers = list(range(g.vcount()))
        if not centers:
            return 0.0
        total_loss = 0.0
        total_steps = 0

        bsz = max(1, int(self.batch_size))
        total_batches = math.ceil(len(centers) / bsz)
        iterator = range(0, len(centers), bsz)
        iterator = _tqdm(iterator, total=total_batches, leave=False, desc=f"Snapshot {sidx} Batches")

        batch_idx = 0
        for start in iterator:
            end = min(len(centers), start + bsz)
            batch_centers = centers[start:end]

            subs: List = []
            x_list: List[np.ndarray] = []
            e_list: List[torch.Tensor] = []
            ids_list: List[List[str]] = []
            node_counts: List[int] = []
            freq_weights: List[float] = []
            fps: List[np.ndarray] = []
            semvecs: List[np.ndarray] = []
            wl_counters: List[Counter] = []

            for center in batch_centers:
                sub = self._ego_subgraph(g, center, r=self.r_hop, max_nodes=self.ego_max_nodes)
                if sub.vcount() == 0:
                    continue
                subs.append(sub)
                xi = self._build_node_features(sub)
                ei = self._igraph_edges_to_edge_index(sub)
                ids = [sub.vs[i]['name'] for i in range(sub.vcount())]
                x_list.append(xi)
                e_list.append(ei)
                ids_list.append(ids)
                node_counts.append(sub.vcount())
                freq = float(g.vs[center]['frequency'])
                freq_weights.append(1.0 + max(0.0, self.anomaly_alpha) * freq)
                # 准备相似度特征（按配置）
                if getattr(self, 'sim_measure', 'wl') == 'tanimoto':
                    try:
                        fp = self._subgraph_fingerprint(sub, m_bits=int(self.sem_fp_bits)) if int(self.sem_fp_bits) > 0 else np.zeros(0, dtype=np.float32)
                    except Exception:
                        fp = np.zeros(int(max(1, self.sem_fp_bits)), dtype=np.float32)
                    fps.append(fp)
                elif getattr(self, 'sim_measure', 'wl') == 'cosine':
                    try:
                        sv = self._subgraph_semantic_vector(sub)
                    except Exception:
                        sv = np.zeros(int(self.prop_feat_dim), dtype=np.float32)
                    semvecs.append(sv)
                else:  # 'wl'
                    try:
                        cnt = self._wl_subtree_counter(sub, h=int(getattr(self, 'wl_height', 2)))
                    except Exception:
                        cnt = Counter()
                    wl_counters.append(cnt)

            if not subs:
                continue

            # ===== 经典 NT-Xent：为每个子图构造两种增强视角，正对为同一子图的两视角；可选再追加基于相似度的 Top-K 正对 =====
            # 索引与展平信息
            offsets = np.cumsum([0] + node_counts[:-1]).tolist()
            flat_ids = [nid for ids in ids_list for nid in ids]
            graph_ids = torch.tensor([gi for gi, n in enumerate(node_counts) for _ in range(n)], dtype=torch.long, device=device)

            # 取上一次时刻的隐藏状态（按当前顺序对齐）
            H_prev = self.temporal.fetch(flat_ids, device=device)

            # 两个视角的边/特征增强（逐子图，再合并）
            x_tensors = [torch.from_numpy(xi).to(device) for xi in x_list]
            e_tensors = e_list
            x1_list = [self._augment_features(xi, mask_p=self.feat_mask_p) for xi in x_tensors]
            x2_list = [self._augment_features(xi, mask_p=self.feat_mask_p) for xi in x_tensors]
            e1_cols = []
            e2_cols = []
            for ei, off in zip(e_tensors, offsets):
                if ei.numel() > 0:
                    e1_cols.append(self._augment_edges(ei, drop_p=self.drop_edge_p) + off)
                    e2_cols.append(self._augment_edges(ei, drop_p=self.drop_edge_p) + off)
            E1 = torch.cat(e1_cols, dim=1) if e1_cols else torch.zeros((2, 0), dtype=torch.long, device=device)
            E2 = torch.cat(e2_cols, dim=1) if e2_cols else torch.zeros((2, 0), dtype=torch.long, device=device)
            X1 = torch.cat(x1_list, dim=0) if x1_list else torch.empty(0, int(self.prop_feat_dim), device=device)
            X2 = torch.cat(x2_list, dim=0) if x2_list else torch.empty(0, int(self.prop_feat_dim), device=device)

            # 视角1前向
            Z1_list = self.encoder(X1, E1, return_all=True)
            H1_list = self.temporal(Z1_list, H_prev)
            H1_last = H1_list[-1]
            Bc = len(node_counts)
            sums1 = torch.zeros((Bc, H1_last.size(1)), dtype=H1_last.dtype, device=device)
            cnts1 = torch.zeros((Bc,), dtype=torch.float32, device=device)
            sums1.index_add_(0, graph_ids, H1_last)
            cnts1.index_add_(0, graph_ids, torch.ones_like(graph_ids, dtype=torch.float32))
            means1 = sums1 / (cnts1.clamp_min(1e-6).unsqueeze(1))
            Z_view1 = F.normalize(self.proj_head(means1), dim=-1)

            # 视角2前向（使用相同 H_prev 以保持同一时刻的对比）
            Z2_list = self.encoder(X2, E2, return_all=True)
            H2_list = self.temporal(Z2_list, H_prev)
            H2_last = H2_list[-1]
            sums2 = torch.zeros((Bc, H2_last.size(1)), dtype=H2_last.dtype, device=device)
            cnts2 = torch.zeros((Bc,), dtype=torch.float32, device=device)
            sums2.index_add_(0, graph_ids, H2_last)
            cnts2.index_add_(0, graph_ids, torch.ones_like(graph_ids, dtype=torch.float32))
            means2 = sums2 / (cnts2.clamp_min(1e-6).unsqueeze(1))
            Z_view2 = F.normalize(self.proj_head(means2), dim=-1)

            # ===== 基于恶意语料：为每个子图再构造两组“恶意视角”，直接作为完整样本对拼进 batch =====
            Z_neg_blocks: List[torch.Tensor] = []
            if getattr(self, 'use_malicious_negatives', False) and len(self.malicious_token_counter) > 0:
                # 按子图构造恶意特征（节点级），再按 batch 展平
                X_neg_list = []
                for sub, xi in zip(subs, x_list):
                    xneg_np = self._corrupt_features_with_malicious(
                        sub, xi, ratio=float(getattr(self, 'mal_neg_ratio', 0.3)), token_len=int(getattr(self, 'mal_neg_token_len', 16))
                    )
                    X_neg_list.append(torch.from_numpy(xneg_np).to(device))
                X_neg = torch.cat(X_neg_list, dim=0) if X_neg_list else torch.empty(0, int(self.prop_feat_dim), device=device)

                # 生成两组恶意视角（与原实现一致，保证相邻配对）
                for _ in range(2):
                    en_cols = []
                    for ei, off in zip(e_tensors, offsets):
                        if ei.numel() > 0:
                            en_cols.append(self._augment_edges(ei, drop_p=self.drop_edge_p) + off)
                    EN = torch.cat(en_cols, dim=1) if en_cols else torch.zeros((2, 0), dtype=torch.long, device=device)
                    XN = self._augment_features(X_neg, mask_p=self.feat_mask_p)

                    ZN_list = self.encoder(XN, EN, return_all=True)
                    HN_list = self.temporal(ZN_list, H_prev)
                    NL = HN_list[-1]
                    sums_n = torch.zeros((Bc, NL.size(1)), dtype=NL.dtype, device=device)
                    cnts_n = torch.zeros((Bc,), dtype=torch.float32, device=device)
                    sums_n.index_add_(0, graph_ids, NL)
                    cnts_n.index_add_(0, graph_ids, torch.ones_like(graph_ids, dtype=torch.float32))
                    means_n = sums_n / (cnts_n.clamp_min(1e-6).unsqueeze(1))
                    Z_neg_blocks.append(F.normalize(self.proj_head(means_n), dim=-1))

            # 以相邻为正对的顺序把所有视角（正常+恶意）拼接进 Z，并一次性计算 sim/损失
            rows: List[torch.Tensor] = []
            views_per_graph = 2 + (2 if len(Z_neg_blocks) == 2 else 0)
            for gi in range(Bc):
                rows.append(Z_view1[gi].unsqueeze(0))
                rows.append(Z_view2[gi].unsqueeze(0))
                if len(Z_neg_blocks) == 2:
                    rows.append(Z_neg_blocks[0][gi].unsqueeze(0))
                    rows.append(Z_neg_blocks[1][gi].unsqueeze(0))
            Z_batch = torch.cat(rows, dim=0)  # [views_per_graph*B, D]

            # 与 Z 顺序对齐的样本权重（恶意视角不追加额外权重，与 gcc_embedder.py 保持一致）
            w_list: List[float] = []
            for w in freq_weights:
                w_list.append(float(w))
                w_list.append(float(w))
            w_tensor = torch.tensor(w_list, dtype=torch.float32, device=device)

            # 多正样本 NT-Xent：相邻正对 + WL Top-K 相似子图映射到当前/配对视角
            N = Z_batch.size(0)
            sim_mat = torch.mm(Z_batch, Z_batch.t()) / float(self.temperature)
            eye_mask = torch.eye(N, dtype=torch.bool, device=device)
            sim_mat = sim_mat.masked_fill(eye_mask, -1e9)
            exp_sim = torch.exp(sim_mat)

            # 计算 WL 子图相似度矩阵 S（子图级 B×B）
            Bcur = Bc
            S = None
            if Bcur >= 2:
                if wl_counters:
                    vocab = sorted(set().union(*[set(c.keys()) for c in wl_counters[:Bcur]]))
                else:
                    vocab = []
                if len(vocab) > 0:
                    vid = {k: i for i, k in enumerate(vocab)}
                    Fw = np.zeros((Bcur, len(vocab)), dtype=np.float32)
                    for rr, cnt in enumerate(wl_counters[:Bcur]):
                        for k, v in cnt.items():
                            idx = vid.get(k)
                            if idx is not None:
                                Fw[rr, idx] = float(v)
                    Fw_t = torch.tensor(Fw, dtype=torch.float32, device=device)
                    K = Fw_t @ Fw_t.t()
                    d = torch.diag(K).clamp_min(1e-12).sqrt()
                    S = K / (d.unsqueeze(1) * d.unsqueeze(0) + 1e-12)
                    S.fill_diagonal_(0.0)

            # 构造多正样本掩码
            pos_mask = torch.zeros((N, N), dtype=torch.float32, device=device)
            views_per_graph = 2 + (2 if len(Z_neg_blocks) == 2 else 0)
            def pair_slot(vs: int) -> int:
                return vs ^ 1  # 0<->1, 2<->3
            # Top-K 配置
            if S is not None and isinstance(getattr(self, 'topk_pos', None), int) and int(getattr(self, 'topk_pos')) > 0:
                kpos = min(int(getattr(self, 'topk_pos')), max(0, Bcur - 1))
            else:
                kpos = 0
            tau_sim = float(getattr(self, 'topk_pos_min_sim', 0.0))

            for r in range(N):
                gidx = r // views_per_graph
                vslot = r % views_per_graph
                pos_idx: List[int] = []
                # 主正对（相邻）
                if views_per_graph >= 2:
                    pos_idx.append(gidx * views_per_graph + pair_slot(vslot))
                # WL Top-K 扩增正样本：当前 slot 与其配对 slot
                if S is not None and kpos > 0:
                    rowS = S[gidx, :Bcur].clone()
                    rowS[gidx] = -1e9
                    vals, idxs = torch.topk(rowS, k=kpos, largest=True)
                    if vals.numel() > 0:
                        m = vals > tau_sim
                        neigh = idxs[m].tolist() if m.any() else []
                        for t in neigh:
                            pos_idx.append(t * views_per_graph + vslot)
                            pos_idx.append(t * views_per_graph + pair_slot(vslot))
                if pos_idx:
                    pos_mask[r, torch.tensor(pos_idx, dtype=torch.long, device=device)] = 1.0

            neg_mask = 1.0 - pos_mask
            neg_mask = neg_mask.masked_fill(eye_mask, 0.0)
            numerator = (pos_mask * exp_sim).sum(dim=1)
            denominator = (neg_mask * exp_sim).sum(dim=1).clamp_min(1e-12)
            valid = numerator > 0
            if not valid.any():
                # 若无有效正样本，跳过并提交一次记忆
                for gi, (ids, n) in enumerate(zip(ids_list, node_counts)):
                    beg, endi = offsets[gi], offsets[gi] + n
                    slice_H = [Hl[beg:endi].detach() for Hl in H1_list]
                    self.temporal.commit(ids, slice_H)
                continue
            loss_vec = -torch.log(numerator[valid] / denominator[valid])
            w_t = w_tensor[valid]
            loss = (w_t * loss_vec).sum() / w_t.sum().clamp_min(1e-6)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.proj_head.parameters()) + list(self.temporal.parameters()),
                max_norm=5.0,
            )
            self.optimizer.step()

            # 更新时序记忆（采用视角1的隐状态作为当期提交）
            for gi, (ids, n) in enumerate(zip(ids_list, node_counts)):
                beg, endi = offsets[gi], offsets[gi] + n
                slice_H = [Hl[beg:endi].detach() for Hl in H1_list]
                self.temporal.commit(ids, slice_H)

            total_loss += float(loss.detach().cpu().item())
            total_steps += 1
            batch_idx += 1

        return total_loss / max(1, total_steps)


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
                _nei_tokens = self._gather_neighbor_tokens(g, i)
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

    def _subgraph_semantic_vector(self, sub) -> np.ndarray:
        """将子图 tokens 聚合为稠密语义向量（Word2Vec 平均，已归一化）。"""
        toks = self._collect_subgraph_tokens(sub)
        return self._w2v_vector_from_tokens(toks)

    def _wl_subtree_counter(self, sub, h: int = 2) -> Counter:
        n = sub.vcount()
        if n == 0:
            return Counter()

        # 初始标签：节点 properties
        labels = [str(sub.vs[i]['properties']) for i in range(n)]
        ctr = Counter(f"0:{lab}" for lab in labels)
        # 构建邻接信息（默认有向 + 有 actions）
        neighbors_info: List[List[Tuple[str, str, int]]] = [[] for _ in range(n)]
        for e in sub.es:
            u = e.source  # 获取起点 ID
            v= e.target  # 获取终点 ID
            etype = e['actions']
            neighbors_info[u].append(('out', etype, v))
            neighbors_info[v].append(('in', etype, u))
        # WL 迭代
        for k in range(1, h + 1):
            new_labels = []
            for i in range(n):
                neigh = neighbors_info[i]
                if neigh:
                    ms = [f"{d}:{et}:{labels[j]}" for (d, et, j) in neigh]
                    ms.sort()
                    agg = '#'.join(ms)
                    new_lab = labels[i] + '|' + agg
                else:
                    new_lab = labels[i]
                new_labels.append(new_lab)

            labels = new_labels
            ctr.update(f"{k}:{lab}" for lab in labels)
        return ctr

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

