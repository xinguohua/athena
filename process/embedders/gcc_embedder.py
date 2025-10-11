"""
GCCEmbedder: Graph Contrastive Coding-style pretraining for GNNs

设计目标：
- 使用子图实例对比学习（InfoNCE/NT-Xent）自监督预训练节点编码器
- 子图实例：从快照图中对某节点抽取 r-hop ego 子图
- 两视角数据增强：随机删边 + 特征掩码
- 编码器：GIN（Graph Isomorphism Network）多层 + 投影头（仅用于对比损失）
- 产出：训练后对整段快照计算图级嵌入（度加权 + L2）

实现说明（避免版权问题）：
- 参考论文思想与公开描述实现一个简化版本，并非拷贝开源库代码。
"""
from __future__ import annotations

from typing import Optional, Iterable, Tuple, List, Dict, Union
import random
import re
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
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            self.layers.append(GINConv(dims[i], dims[i + 1], eps=0.0, mlp_hidden=hidden_dim, dropout=dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


# ---------------- GCC Embedder ----------------
class GCCEmbedder(GraphEmbedderBase):
    _default_path = 'gcc_encoder.pth'

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

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 特征缓存（properties -> hashed vec）
        self._prop_cache: Dict[str, np.ndarray] = {}

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
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.proj_head.parameters()), lr=self.lr, weight_decay=1e-4)

        # 训练后缓存：每快照一个 {node_id: vec}
        self.snapshot_node_embeddings: List[Dict[str, np.ndarray]] = []

    # ---------- 公共 API ----------
    def train(self):
        if not self.train_snapshot_indices:
            raise RuntimeError("没有可用于训练的快照。请检查 train_indices 设置。")
        print(f"[GCC] Pretrain on {len(self.train_snapshot_indices)}/{len(self.snapshots)} snapshots | batch={self.batch_size} | tau={self.temperature}")

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for _ in range(self.steps_per_epoch):
                batch_graphs: List[Tuple[torch.Tensor, torch.Tensor]] = []  # (x, edge_index)
                # 采样 batch 个子图实例
                for _b in range(self.batch_size):
                    sidx = random.choice(self.train_snapshot_indices)
                    g = self.snapshots[sidx]
                    if g is None or g.vcount() == 0:
                        continue
                    center = random.randrange(0, g.vcount())
                    sub = self._ego_subgraph(g, center, r=self.r_hop, max_nodes=self.ego_max_nodes)
                    if sub.vcount() == 0:
                        continue
                    x_np = self._build_node_features(sub)
                    edge_index = self._igraph_edges_to_edge_index(sub)
                    x = torch.from_numpy(x_np).to(self.device)
                    # 两视角增强
                    e1 = self._augment_edges(edge_index, drop_p=self.drop_edge_p)
                    x1 = self._augment_features(x, mask_p=self.feat_mask_p)
                    e2 = self._augment_edges(edge_index, drop_p=self.drop_edge_p)
                    x2 = self._augment_features(x, mask_p=self.feat_mask_p)
                    batch_graphs.append((x1, e1))
                    batch_graphs.append((x2, e2))

                if len(batch_graphs) < 2:
                    continue

                Z = []
                self.optimizer.zero_grad()
                for x_i, e_i in batch_graphs:
                    h_i = self.encoder(x_i, e_i)
                    g_i = h_i.mean(dim=0, keepdim=True)  # 简单平均池化
                    z_i = self.proj_head(g_i)
                    Z.append(F.normalize(z_i, dim=-1))
                Z = torch.cat(Z, dim=0)  # [2N, D]

                loss = self._nt_xent_loss(Z, temperature=self.temperature)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.proj_head.parameters()), max_norm=5.0)
                self.optimizer.step()
                epoch_loss += float(loss.detach().cpu().item())

            avg = epoch_loss / max(1, self.steps_per_epoch)
            print(f"[GCC] Epoch {epoch+1}/{self.num_epochs} | Loss={avg:.6f}")

        self._generate_snapshot_node_embeddings()
        self.save_model()

    def embed_nodes(self):
        return self.snapshot_node_embeddings[-1] if self.snapshot_node_embeddings else {}

    def embed_edges(self):
        return {}

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
            # 使用节点出现频率作为权重，若不可用则回退到度
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
                    # 频率优先，非法值回退为 0
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
        print(f"[GCC] Snapshot embeddings: {arr.shape}")
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
            },
            'encoder': self.encoder.state_dict(),
            'proj_head': self.proj_head.state_dict(),
            'snapshot_node_embeddings': self.snapshot_node_embeddings,
        }
        torch.save(state, path)
        print(f"[GCC] Model saved to {path}")

    @classmethod
    def load(cls, snapshot_sequence, path: Optional[str] = None):
        path = path or cls._default_path
        print(f"[GCC] Loading model from {path}…")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state = torch.load(path, map_location=device)
        raw_params = dict(state.get('params', {}))
        allowed = {
            'prop_feat_dim','enc_hidden_dim','enc_out_dim','gin_layers','dropout',
            'num_epochs','steps_per_epoch','batch_size','lr','temperature',
            'r_hop','ego_max_nodes','drop_edge_p','feat_mask_p','train_indices','model_path'
        }
        params = {k: v for k, v in raw_params.items() if k in allowed}
        inst = cls(snapshot_sequence, **params)
        inst.encoder.load_state_dict(state['encoder'])
        inst.proj_head.load_state_dict(state['proj_head'])
        inst.snapshot_node_embeddings = state.get('snapshot_node_embeddings', [])
        print("[GCC] Model loaded successfully")
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

    def _text_hash_vector(self, text: str, dim: int) -> np.ndarray:
        if dim <= 0:
            return np.zeros(0, dtype=np.float32)
        v = np.zeros(dim, dtype=np.float32)
        if not text:
            return v
        tokens = [t for t in re.split(r'[^A-Za-z0-9]+', str(text).lower()) if t]
        for tok in tokens:
            h = int(hashlib.md5(tok.encode('utf-8')).hexdigest(), 16) % dim
            v[h] += 1.0
        n = np.linalg.norm(v) + 1e-12
        return (v / n).astype(np.float32)

    def _build_node_features(self, g) -> np.ndarray:
        n = g.vcount()
        if self.prop_feat_dim <= 0:
            # 用常数特征占位
            return np.ones((n, 1), dtype=np.float32)
        X = np.zeros((n, self.prop_feat_dim), dtype=np.float32)
        for i in range(n):
            try:
                prop = g.vs[i]['properties']
            except Exception:
                prop = g.vs[i].attributes().get('properties', '')
            key = str(prop)
            if key in self._prop_cache:
                X[i] = self._prop_cache[key]
            else:
                vec = self._text_hash_vector(key, self.prop_feat_dim)
                self._prop_cache[key] = vec
                X[i] = vec
        return X

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

    def _nt_xent_loss(self, Z: torch.Tensor, temperature: float) -> torch.Tensor:
        # Z: [2N, D], 每相邻两行构成一对正样本 (x1, x2)
        Z = F.normalize(Z, dim=-1)
        sim = torch.mm(Z, Z.t()) / temperature  # [2N,2N]
        N2 = Z.size(0)
        labels = torch.arange(N2, device=Z.device)
        pos = labels ^ 1  # 0<->1, 2<->3, ... （相邻为正对）
        # 去除对角自身
        mask = torch.eye(N2, device=Z.device).bool()
        sim = sim.masked_fill(mask, -1e9)
        loss_i = F.cross_entropy(sim, pos)
        return loss_i

    def _generate_snapshot_node_embeddings(self):
        self.encoder.eval()
        self.snapshot_node_embeddings.clear()
        with torch.no_grad():
            for g in self.snapshots:
                if g is None or g.vcount() == 0:
                    self.snapshot_node_embeddings.append({})
                    continue
                x_np = self._build_node_features(g)
                eidx = self._igraph_edges_to_edge_index(g)
                x = torch.from_numpy(x_np).to(self.device)
                h = self.encoder(x, eidx)
                emb_dict: Dict[str, np.ndarray] = {}
                for i in range(g.vcount()):
                    nid = g.vs[i]['name']
                    emb_dict[nid] = h[i].detach().cpu().numpy().astype(np.float32)
                self.snapshot_node_embeddings.append(emb_dict)
