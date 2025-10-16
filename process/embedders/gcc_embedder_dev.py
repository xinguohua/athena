"""
GCCEmbedderDev: 开发版 Graph Contrastive Coding-style 预训练编码器

说明：这是对 gcc_embedder.py 的开发拷贝，便于独立调参与修改，不影响原版。
主要差异：
- 类名改为 GCCEmbedderDev
- 默认模型路径改为 gcc_encoder_dev.pth
"""
from __future__ import annotations

from typing import Optional, Iterable, Tuple, List, Dict, Union
import random
import re
import os

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

        # 对 (节点tokens + 1-hop邻域tokens) 做轻量增强（丢词+bigram）
        self.use_token_augmentation = True

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 特征缓存（properties -> 向量）
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
        # 先确保 Word2Vec 模型已准备好（预训练路径加载或用训练快照语料自训练）
        self._ensure_w2v_model()
        print(f"[GCC-Dev] Pretrain on {len(self.train_snapshot_indices)}/{len(self.snapshots)} snapshots | batch={self.batch_size} | tau={self.temperature}")

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for _ in range(self.steps_per_epoch):
                batch_graphs: List[Tuple[torch.Tensor, torch.Tensor]] = []  # (x, edge_index)
                sample_weights: List[float] = []  # 与 batch_graphs 一一对应（注意每个子图视角各占一行）
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
                    # 基于“节点异常=节点频率”的子图权重：取中心节点的频率作为异常度
                    try:
                        if 'frequency' in g.vs.attributes():
                            freq_val = float(g.vs[center]['frequency'])
                        else:
                            freq_val = 1.0
                    except Exception:
                        freq_val = 1.0
                    if not np.isfinite(freq_val) or freq_val < 0:
                        freq_val = 0.0
                    w = 1.0 + max(0.0, self.anomaly_alpha) * float(freq_val)
                    sample_weights.append(w)
                    sample_weights.append(w)

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

                w_tensor = None
                if len(sample_weights) == Z.size(0):
                    w_tensor = torch.tensor(sample_weights, dtype=torch.float32, device=Z.device)
                loss = self._nt_xent_loss(Z, temperature=self.temperature, sample_weights=w_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.proj_head.parameters()), max_norm=5.0)
                self.optimizer.step()
                epoch_loss += float(loss.detach().cpu().item())

            avg = epoch_loss / max(1, self.steps_per_epoch)
            print(f"[GCC-Dev] Epoch {epoch+1}/{self.num_epochs} | Loss={avg:.6f}")

        self._generate_snapshot_node_embeddings()
        self.save_model()

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
                # W2V 配置
                'w2v_window': self.w2v_window,
                'w2v_min_count': self.w2v_min_count,
                'w2v_sg': self.w2v_sg,
                'w2v_epochs': self.w2v_epochs,
                'w2v_pretrained_path': self.w2v_pretrained_path,
            },
            'encoder': self.encoder.state_dict(),
            'proj_head': self.proj_head.state_dict(),
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
        inst.encoder.load_state_dict(state['encoder'])
        inst.proj_head.load_state_dict(state['proj_head'])
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
            from gensim.models import Word2Vec
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
        from gensim.models import Word2Vec
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
            if self.use_token_augmentation:
                # 增强路径：节点 + 1-hop 邻域 tokens，做轻量增强后编码
                self_tokens = self._get_node_tokens(g, i)
                nei_tokens = self._gather_neighbor_tokens(g, i)
                all_tokens = self_tokens + nei_tokens
                tokens = self._augment_tokens(all_tokens)
                vec = self._w2v_vector_from_tokens(tokens)
                vec = vec / (np.linalg.norm(vec) + 1e-12)
                X[i] = vec.astype(np.float32)
            else:
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

