"""
ROLAND Separation Embedder (clean minimal implementation)

- GraphSAGE(mean) backbone for node embeddings
- Degree-weighted mean readout for graph embeddings + L2 norm
- Neighbor-predict + variance regularization for stability
- Center-margin separation loss on graph-level embeddings

Use via: get_embedder_by_name("roland-sep")
"""
from __future__ import annotations

from typing import Dict, Tuple, Optional, Iterable, Union, List
import re
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from process.embedders.base import GraphEmbedderBase


# ---- GraphSAGE (mean) ----
class GraphSAGEConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, l2_norm: bool = False):
        super().__init__()
        self.lin = nn.Linear(in_channels * 2, out_channels, bias=bias)
        self.l2_norm = l2_norm

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, node_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.numel() == 0:
            return x
        src, dst = edge_index[0], edge_index[1]
        N = x.size(0)
        if node_weight is not None:
            w = node_weight[src].unsqueeze(1) if src.numel() > 0 else None
            nei_sum = torch.zeros_like(x)
            if src.numel() > 0:
                nei_sum.index_add_(0, dst, x[src] * w)
            wdeg = torch.zeros(N, device=x.device, dtype=x.dtype)
            if src.numel() > 0:
                wdeg.index_add_(0, dst, node_weight[src])
            nei_mean = nei_sum / torch.clamp(wdeg, min=1e-8).unsqueeze(-1)
        else:
            deg = torch.bincount(dst, minlength=N).float()
            nei_sum = torch.zeros_like(x)
            if src.numel() > 0:
                nei_sum.index_add_(0, dst, x[src])
            nei_mean = nei_sum / torch.clamp(deg, min=1.0).unsqueeze(-1)
        h_cat = torch.cat([x, nei_mean], dim=-1)
        out = F.relu(self.lin(h_cat))
        if self.l2_norm:
            out = F.normalize(out, p=2, dim=-1)
        return out


class _Backbone(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_mlp_layers: int = 2,
        dropout: float = 0.1,
        prop_dim: int = 0,
    ):
        super().__init__()
        self.node_embedding = nn.Embedding(num_embeddings=num_nodes, embedding_dim=embedding_dim)
        self.prop_dim = int(prop_dim)
        self.use_props = self.prop_dim > 0
        if self.use_props:
            self.prop_proj = nn.Linear(self.prop_dim, embedding_dim)
        self.sage1 = GraphSAGEConv(embedding_dim, hidden_dim, l2_norm=False)
        self.sage2 = GraphSAGEConv(hidden_dim, hidden_dim, l2_norm=False)
        self.dropout = nn.Dropout(dropout)
        mlp_layers: List[nn.Module] = []
        for _ in range(num_mlp_layers):
            mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*mlp_layers)
        # Predict neighbor-mean -> node output (self-supervised head)
        self.f_phi = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        node_indices: torch.Tensor,
        edge_index: torch.Tensor,
        prop_feats: Optional[torch.Tensor] = None,
        node_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x0 = self.node_embedding(node_indices)
        if self.use_props and prop_feats is not None and prop_feats.numel() > 0:
            x0 = x0 + self.prop_proj(prop_feats)
        h1 = self.sage1(x0, edge_index, node_weight)
        h2 = self.sage2(self.dropout(h1), edge_index, node_weight)
        if h2.shape == h1.shape:
            h2 = h2 + h1
        h = self.mlp(h2)
        return h


class ROLANDSeparationEmbedder(GraphEmbedderBase):
    _default_path = 'roland_sep_encoder.pth'

    def __init__(
        self,
        snapshots,
        features=None,
        mapp=None,
        embedding_dim: int = 256,
        hidden_conv_1: int = 128,
        hidden_conv_2: int = 256,
        num_epochs: int = 30,
        lr: float = 5e-4,
        # 属性
        prop_feat_dim: int = 128,
        prop_vec_mode: str = 'hash',
        prop_w2v_path: Optional[str] = None,
        prop_w2v_window: int = 5,
        prop_w2v_min_count: int = 1,
        prop_w2v_epochs: int = 10,
        # 损失权重
        neigh_pred_weight: float = 1.0,
        variance_weight: float = 0.1,
        separation_weight: float = 1.0,
        # Center-Margin 参数
        sep_margin: float = 0.5,
        snapshots_per_batch: int = 8,
        train_indices: Optional[Union[Iterable[int], Tuple[int, int], int]] = None,
        model_path: Optional[str] = None,
    ):
        super().__init__(snapshots, features, mapp)
        self.snapshots = snapshots
        self.embedding_dim = int(embedding_dim)
        self.hidden_conv_1 = int(hidden_conv_1)
        self.hidden_conv_2 = int(hidden_conv_2)
        self.num_epochs = int(num_epochs)
        self.lr = float(lr)
        self.model_path = model_path or self._default_path

        # 损失配置
        self.neigh_pred_weight = float(neigh_pred_weight)
        self.variance_weight = float(variance_weight)
        self.separation_weight = float(separation_weight)
        self.sep_margin = float(sep_margin)
        self.snapshots_per_batch = max(1, int(snapshots_per_batch))

        # 属性配置
        self.prop_feat_dim = int(prop_feat_dim)
        self.prop_vec_mode = str(prop_vec_mode or 'hash').lower()
        if self.prop_vec_mode not in ('word2vec', 'hash'):
            self.prop_vec_mode = 'hash'
        self.prop_w2v_path = prop_w2v_path
        self.prop_w2v_window = int(prop_w2v_window)
        self.prop_w2v_min_count = int(prop_w2v_min_count)
        self.prop_w2v_epochs = int(prop_w2v_epochs)
        self._prop_cache: Dict[str, np.ndarray] = {}
        self._w2v_kv = None  # lazy init

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 全局节点词表
        all_nodes_set = set()
        for g in snapshots:
            if g is None:
                continue
            for v in range(g.vcount()):
                all_nodes_set.add(g.vs[v]['name'])
        self.all_nodes = sorted(all_nodes_set)
        self.num_nodes = len(self.all_nodes)
        self.node_id_map = {nid: i for i, nid in enumerate(self.all_nodes)}

        # 训练快照索引
        self.train_snapshot_indices = self._resolve_train_indices(train_indices)
        self._train_snapshot_index_set = set(self.train_snapshot_indices)

        # 模型 & 优化器
        self.model = _Backbone(
            num_nodes=self.num_nodes,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_conv_1,
            output_dim=self.hidden_conv_2,
            prop_dim=self.prop_feat_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        # 缓存节点嵌入（训练后）
        self.snapshot_embeddings_list: List[Dict[str, np.ndarray]] = []

    # ---- 公共 API ----
    def train(self):
        if not self.train_snapshot_indices:
            raise RuntimeError("没有可用于训练的快照。请检查 train_indices 设置。")
        print(f"[ROLAND-SEP] Train on {len(self.train_snapshot_indices)}/{len(self.snapshots)} snapshots, nodes={self.num_nodes}")
        print(f"[ROLAND-SEP] Loss: neigh={self.neigh_pred_weight}, var={self.variance_weight}, sep={self.separation_weight} | margin={self.sep_margin} | batch={self.snapshots_per_batch}")

        indices = self.train_snapshot_indices
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, len(indices), self.snapshots_per_batch):
                batch_indices = indices[start : start + self.snapshots_per_batch]
                batch_graph_embs: List[torch.Tensor] = []
                batch_labels: List[int] = []
                batch_neigh = 0.0
                batch_var = 0.0

                self.optimizer.zero_grad()

                for sidx in batch_indices:
                    g = self.snapshots[sidx]
                    if g is None:
                        continue
                    node_ids, edge_index, prop_feats, node_weight = self._igraph_to_torch(g)
                    z = self.model(node_ids, edge_index, prop_feats, node_weight)

                    # 邻居预测 + 方差正则
                    src, dst = edge_index[0], edge_index[1]
                    N_local = z.size(0)
                    if node_weight is not None:
                        w = node_weight[src].unsqueeze(1) if src.numel() > 0 else None
                        neigh_sum = torch.zeros_like(z)
                        if src.numel() > 0:
                            neigh_sum.index_add_(0, dst, z[src] * w)
                        wdeg = torch.zeros(N_local, device=z.device, dtype=z.dtype)
                        if src.numel() > 0:
                            wdeg.index_add_(0, dst, node_weight[src])
                        neigh_mean = neigh_sum / torch.clamp(wdeg, min=1e-8).unsqueeze(-1)
                        deg = wdeg
                    else:
                        deg = torch.bincount(dst, minlength=N_local).float()
                        neigh_sum = torch.zeros_like(z)
                        if src.numel() > 0:
                            neigh_sum.index_add_(0, dst, z[src])
                        neigh_mean = neigh_sum / torch.clamp(deg, min=1.0).unsqueeze(-1)

                    mask = deg > 0
                    if mask.any():
                        pred = self.model.f_phi(neigh_mean[mask])
                        pred_h = pred[:, : self.hidden_conv_2]
                        target_h = z[mask].detach()
                        neigh_loss = F.mse_loss(pred_h, target_h)
                        eps = 1e-4
                        std = z[mask].std(dim=0) + eps
                        var_loss = torch.mean(F.relu(1.0 - std))
                    else:
                        neigh_loss = z.sum() * 0.0
                        var_loss = z.sum() * 0.0

                    (self.neigh_pred_weight * neigh_loss + self.variance_weight * var_loss).backward(retain_graph=True)
                    batch_neigh += float(neigh_loss.detach().cpu().item())
                    batch_var += float(var_loss.detach().cpu().item())

                    # 图级向量
                    graph_vec = self._readout_graph_embedding(g, z)
                    batch_graph_embs.append(graph_vec)
                    batch_labels.append(self._snapshot_label(g))

                # 分离损失（batch 维度）
                sep_loss = self._separation_center_margin_loss(batch_graph_embs, batch_labels)
                (self.separation_weight * sep_loss).backward()

                torch.nn.utils.clip_grad_norm_(list(self.model.parameters()), max_norm=5.0)
                self.optimizer.step()

                total = batch_neigh + batch_var + float(self.separation_weight * sep_loss.detach().cpu().item())
                epoch_loss += total
                num_batches += 1

            avg = epoch_loss / max(1, num_batches)
            print(f"[ROLAND-SEP] Epoch {epoch+1}/{self.num_epochs} | AvgLoss={avg:.6f}")

        self._generate_final_embeddings()
        self.save_model()

    def embed_nodes(self):
        if not self.snapshot_embeddings_list:
            raise RuntimeError("还没有节点嵌入，请先调用 train()")
        return self.snapshot_embeddings_list[-1] if self.snapshot_embeddings_list else {}

    def embed_edges(self):
        return {}

    def get_snapshot_embeddings(self, snapshot_sequence=None):
        if not self.snapshot_embeddings_list:
            raise RuntimeError("还没有快照嵌入，请先调用 train()")
        if snapshot_sequence is None:
            snapshot_sequence = list(range(len(self.snapshot_embeddings_list)))

        result: List[np.ndarray] = []
        for i in snapshot_sequence:
            emb_dict = self.snapshot_embeddings_list[i] if i < len(self.snapshot_embeddings_list) else None
            g = self.snapshots[i] if i < len(self.snapshots) else None
            if not emb_dict or g is None:
                result.append(np.zeros(self.hidden_conv_2, dtype=np.float32))
                continue
            node_gids = [g.vs[vid]['name'] for vid in range(g.vcount())]
            degrees = g.degree()
            weighted_sum = np.zeros(self.hidden_conv_2, dtype=np.float32)
            total_w = 0.0
            for local_idx, nid in enumerate(node_gids):
                vec = emb_dict.get(nid)
                if vec is None:
                    continue
                w = float(degrees[local_idx])
                if w <= 0:
                    continue
                weighted_sum += (w * vec.astype(np.float32))
                total_w += w
            if total_w <= 0:
                all_embs = np.array(list(emb_dict.values()), dtype=np.float32)
                snapshot_vec = all_embs.mean(axis=0) if all_embs.size > 0 else np.zeros(self.hidden_conv_2, dtype=np.float32)
            else:
                snapshot_vec = weighted_sum / (total_w + 1e-12)
            n = np.linalg.norm(snapshot_vec) + 1e-12
            snapshot_vec = (snapshot_vec / n).astype(np.float32)
            result.append(snapshot_vec)
        arr = np.vstack(result).astype(np.float32) if result else np.zeros((0, self.hidden_conv_2), dtype=np.float32)
        print(f"[ROLAND-SEP] Snapshot embeddings: {arr.shape}")
        return arr

    def save_model(self, path: Optional[str] = None):
        path = path or self.model_path
        state = {
            'params': {
                'embedding_dim': self.embedding_dim,
                'hidden_conv_1': self.hidden_conv_1,
                'hidden_conv_2': self.hidden_conv_2,
                'num_epochs': self.num_epochs,
                'lr': self.lr,
                'train_indices': self.train_snapshot_indices,
                'prop_feat_dim': self.prop_feat_dim,
                'prop_vec_mode': self.prop_vec_mode,
                'prop_w2v_path': self.prop_w2v_path,
                'prop_w2v_window': self.prop_w2v_window,
                'prop_w2v_min_count': self.prop_w2v_min_count,
                'prop_w2v_epochs': self.prop_w2v_epochs,
                'neigh_pred_weight': self.neigh_pred_weight,
                'variance_weight': self.variance_weight,
                'separation_weight': self.separation_weight,
                'sep_margin': self.sep_margin,
                'snapshots_per_batch': self.snapshots_per_batch,
                'model_path': self.model_path,
            },
            'model_state': self.model.state_dict(),
            'snapshot_embeddings': self.snapshot_embeddings_list,
            'all_nodes': self.all_nodes,
            'prop_cache': self._prop_cache,
        }
        torch.save(state, path)
        print(f"[ROLAND-SEP] Model saved to {path}")

    @classmethod
    def load(cls, snapshot_sequence, path: Optional[str] = None):
        path = path or cls._default_path
        print(f"[ROLAND-SEP] Loading model from {path}…")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state = torch.load(path, map_location=device)
        raw_params = dict(state.get('params', {}))
        allowed = {
            'embedding_dim', 'hidden_conv_1', 'hidden_conv_2',
            'num_epochs', 'lr', 'train_indices',
            'prop_feat_dim', 'prop_vec_mode', 'prop_w2v_path',
            'prop_w2v_window', 'prop_w2v_min_count', 'prop_w2v_epochs',
            'neigh_pred_weight', 'variance_weight', 'separation_weight',
            'sep_margin', 'snapshots_per_batch', 'model_path'
        }
        params = {k: v for k, v in raw_params.items() if k in allowed}
        instance = cls(snapshot_sequence, **params)
        instance.model.load_state_dict(state['model_state'])
        instance.snapshot_embeddings_list = state['snapshot_embeddings']
        instance.all_nodes = state['all_nodes']
        if 'prop_cache' in state and isinstance(state['prop_cache'], dict):
            instance._prop_cache = state['prop_cache']
        instance.node_id_map = {nid: i for i, nid in enumerate(instance.all_nodes)}
        instance.num_nodes = len(instance.all_nodes)
        print("[ROLAND-SEP] Model loaded successfully")
        return instance

    # ---- 内部工具 ----
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

    def _igraph_to_torch(self, g) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        node_gids = [g.vs[vid]['name'] for vid in range(g.vcount())]
        node_ids = torch.tensor([self.node_id_map[nid] for nid in node_gids], dtype=torch.long, device=self.device)
        edges = g.get_edgelist()
        if len(edges) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
        else:
            src = [u for (u, v) in edges]
            dst = [v for (u, v) in edges]
            edge_index = torch.tensor([src, dst], dtype=torch.long, device=self.device)
        # 属性
        prop_feats_t: Optional[torch.Tensor] = None
        if self.prop_feat_dim > 0:
            mat = np.zeros((len(node_gids), self.prop_feat_dim), dtype=np.float32)
            for i, _ in enumerate(node_gids):
                mat[i] = self._get_node_prop_vec(g, i)
            prop_feats_t = torch.from_numpy(mat).to(self.device)
        # 节点权重
        try:
            freqs = g.vs['frequency'] if 'frequency' in g.vs.attributes() else [1.0] * g.vcount()
        except Exception:
            freqs = [1.0] * g.vcount()
        freq_arr = np.array([float(x) if x is not None else 1.0 for x in freqs], dtype=np.float32)
        node_weight_t = torch.from_numpy(freq_arr).to(self.device)
        return node_ids, edge_index, prop_feats_t, node_weight_t

    def _get_node_prop_vec(self, g, local_idx: int) -> np.ndarray:
        try:
            prop_str = g.vs[local_idx]['properties']
        except Exception:
            try:
                prop_str = g.vs[local_idx].attributes().get('properties', '')
            except Exception:
                prop_str = ''
        key = str(prop_str)
        if key in self._prop_cache:
            return self._prop_cache[key]
        if self.prop_feat_dim <= 0:
            vec = np.zeros((0,), dtype=np.float32)
        elif self.prop_vec_mode == 'word2vec' and self._w2v_kv is not None:
            # 简化：若需要，可改为 word2vec 平均
            vec = np.zeros(self.prop_feat_dim, dtype=np.float32)
        else:
            vec = self._text_hash_vector(key, self.prop_feat_dim)
        self._prop_cache[key] = vec
        return vec

    @staticmethod
    def _text_hash_vector(text: str, dim: int) -> np.ndarray:
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

    def _readout_graph_embedding(self, g, node_emb: torch.Tensor) -> torch.Tensor:
        node_gids = [g.vs[vid]['name'] for vid in range(g.vcount())]
        degrees = g.degree()
        weighted_sum = torch.zeros(self.hidden_conv_2, device=node_emb.device, dtype=node_emb.dtype)
        total_w = 0.0
        for local_idx, _ in enumerate(node_gids):
            if local_idx >= node_emb.size(0):
                continue
            vec = node_emb[local_idx]
            w = float(degrees[local_idx])
            if w <= 0:
                continue
            weighted_sum += (w * vec)
            total_w += w
        if total_w <= 0:
            gvec = node_emb.mean(dim=0) if node_emb.size(0) > 0 else torch.zeros(self.hidden_conv_2, device=node_emb.device, dtype=node_emb.dtype)
        else:
            gvec = weighted_sum / (total_w + 1e-12)
        gvec = F.normalize(gvec, p=2, dim=0)
        return gvec

    def _snapshot_label(self, g) -> int:
        try:
            for v in g.vs:
                if int(v.attributes().get('label', 0)) == 1:
                    return 1
        except Exception:
            pass
        return 0

    def _separation_center_margin_loss(self, graph_embs: List[torch.Tensor], labels: List[int]) -> torch.Tensor:
        if not graph_embs:
            return torch.tensor(0.0, device=self.device)
        G = torch.stack(graph_embs, dim=0)
        y = torch.tensor(labels, dtype=torch.long, device=self.device)
        mask0 = (y == 0)
        mask1 = (y == 1)
        loss_within = torch.tensor(0.0, device=self.device)
        loss_between = torch.tensor(0.0, device=self.device)
        mu0 = None
        mu1 = None
        if mask0.any():
            mu0 = G[mask0].mean(dim=0)
            loss_within = loss_within + ((G[mask0] - mu0).pow(2).sum(dim=1).mean())
        if mask1.any():
            mu1 = G[mask1].mean(dim=0)
            loss_within = loss_within + ((G[mask1] - mu1).pow(2).sum(dim=1).mean())
        if (mu0 is not None) and (mu1 is not None):
            dist = torch.norm(mu0 - mu1, p=2)
            loss_between = F.relu(self.sep_margin - dist)
        return loss_within + loss_between

    def _generate_final_embeddings(self):
        self.model.eval()
        self.snapshot_embeddings_list.clear()
        with torch.no_grad():
            for g in self.snapshots:
                if g is None:
                    self.snapshot_embeddings_list.append({})
                    continue
                node_ids, edge_index, prop_feats, node_weight = self._igraph_to_torch(g)
                node_emb = self.model(node_ids, edge_index, prop_feats, node_weight)
                node_gids = [g.vs[vid]['name'] for vid in range(g.vcount())]
                emb_dict: Dict[str, np.ndarray] = {}
                for local_idx, nid in enumerate(node_gids):
                    if local_idx < node_emb.size(0):
                        emb_dict[nid] = node_emb[local_idx].detach().cpu().numpy()
                self.snapshot_embeddings_list.append(emb_dict)
