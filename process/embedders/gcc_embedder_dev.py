"""
GCCEmbedderDev: 开发版 Graph Contrastive Coding-style 预训练编码器

说明：这是对 gcc_embedder.py 的开发拷贝，便于独立调参与修改，不影响原版。
主要差异：
- 类名改为 GCCEmbedderDev
- 默认模型路径改为 gcc_encoder_dev.pth
"""
from __future__ import annotations
from collections import deque
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

    def state_dict(self, *args, **kwargs):
        base_state = super().state_dict(*args, **kwargs)
        return base_state

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=strict)

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
        # 是否使用时序记忆（TemporalPerLayer）
        use_temporal: bool = False,
        # 输入/编码器尺寸
        prop_feat_dim: int = 128,
        enc_hidden_dim: int = 128,
        enc_out_dim: int = 256,
        gin_layers: int = 3,
        dropout: float = 0.1,
        # 训练
        num_epochs: int = 3,
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
        anomaly_alpha: float = 1,      # 加权强度，>0 表示异常越大权重越大
        # 是否使用样本权重（基于频率/异常强度），关闭时统一使用均匀权重
        use_sample_weights: bool = True,
        w2v_window: int = 5,
        w2v_min_count: int = 1,
        w2v_sg: int = 1,
        w2v_epochs: int = 20,
        w2v_pretrained_path: Optional[str] = None,
        # 相似度/权重相关可选参数
        sim_measure: str = 'wl',            # 'tanimoto' | 'cosine' | 'wl'
        wl_height: int = 2,
        sem_fp_bits: int = 1024,
        use_malicious_negatives: bool = False,
        mal_neg_ratio: float = 0.3,
        mal_neg_token_len: int = 16,
        mal_stopwords=None,
            # [
            # 'event', 'read', 'write'
            # , 'execute'
            # ],
            # 恶意token停用词列表，传入[]表示不过滤
        mal_print_tokens: bool = True,  # 是否打印恶意token统计信息
    # Top-K 相似（可选，先关闭）
    topk_pos: Optional[int] = 0,   # 先关闭 Top-K 扩增，回到经典 NT-Xent
        topk_pos_min_sim: float = 0.5, # 仅当相似度 > 此阈值时才将样本纳入 Top-K 正样本
    ):
        super().__init__(snapshots, features, mapp)
        if mal_stopwords is None:
            mal_stopwords = [
                'event', 'read', 'write', 'execute'
            ]
        self.snapshots = snapshots
        # 时序使用开关
        self.use_temporal = bool(use_temporal)
        self.prop_feat_dim = int(prop_feat_dim)
        self.enc_hidden_dim = int(enc_hidden_dim)
        self.enc_out_dim = int(enc_out_dim)
        self.gin_layers = int(gin_layers)
        self.dropout = float(dropout)
        self.num_epochs = int(num_epochs)
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
        # 样本权重开关
        self.use_sample_weights = bool(use_sample_weights)

        # Word2Vec 配置（唯一特征来源）
        self.w2v_window = int(w2v_window)
        self.w2v_min_count = int(w2v_min_count)
        self.w2v_sg = int(w2v_sg)
        self.w2v_epochs = int(w2v_epochs)
        self.w2v_pretrained_path = w2v_pretrained_path
        self._w2v_model = None  # 延迟加载/训练

        # 语义相似度参数
        # - sem_fp_bits: 指纹长度（哈希位数），用于快速近似 Tanimoto 计算
        self.sem_fp_bits = int(sem_fp_bits)
        # 相似度度量方式：'tanimoto' | 'cosine' | 'wl'
        self.sim_measure = str(sim_measure)
        # WL 子树核参数（用于 sim_measure='wl'）
        self.wl_height = int(wl_height)

        # 是否使用“恶意语料”来生成额外负样本；以及腐化强度与每个节点替换的 token 数
        self.use_malicious_negatives = bool(use_malicious_negatives)
        self.mal_neg_ratio = float(mal_neg_ratio)  # 每个子图中替换为恶意向量的节点比例
        self.mal_neg_token_len = int(mal_neg_token_len)  # 生成恶意向量时采样的恶意 token 数
        
        # 恶意token停用词：直接使用传入的列表转为set（[]表示不过滤）
        # 归一化停用词：容忍嵌套(list/tuple/set)并转为扁平字符串集合，避免出现 list 内嵌 list 导致 set() 报错
        def _flatten_to_str_set(obj):
            out = []
            if obj is None:
                return set()
            if isinstance(obj, (list, tuple, set)):
                for it in obj:
                    if isinstance(it, (list, tuple, set)):
                        out.extend(str(x) for x in it)
                    else:
                        out.append(str(it))
                return set(out)
            return {str(obj)}

        self.mal_stopwords = _flatten_to_str_set(mal_stopwords)
        self.mal_print_tokens = bool(mal_print_tokens)  # 是否打印恶意token统计

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
        # 优化器包含 encoder、projection head，且在启用时包含 temporal
        opt_params = list(self.encoder.parameters()) + list(self.proj_head.parameters())
        if self.use_temporal:
            opt_params += list(self.temporal.parameters())
        self.optimizer = torch.optim.Adam(opt_params, lr=self.lr, weight_decay=1e-4)

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
            if self.use_temporal:
                self.temporal.reset()  # 每个 epoch 重置时序状态（仅开启时）
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

        # 训练结束后生成节点嵌入：遵循全局开关 self.use_temporal
        self.generate_node_embeddings(use_temporal=self.use_temporal)
        self.save_model()

    def _train_one_snapshot(self, g, sidx: Optional[int] = None) -> float:
        """单个 snapshot 训练（优化 + 稳定版 NT-Xent）"""
        device = self.device
        num_nodes = g.vcount()
        if num_nodes == 0:
            return 0.0

        centers = list(range(num_nodes))
        total_loss, total_steps = 0.0, 0
        bsz = max(1, int(self.batch_size))
        total_batches = math.ceil(len(centers) / bsz)
        print(f"  [Snapshot {sidx}] nodes={num_nodes}, batches={total_batches}")

        # 初始化缓存（保存之前 batch 的 subs, x_list, e_list, freq_weights）
        if not hasattr(self, "_ego_cache"):
            self._ego_cache = deque(maxlen=50)

        for start in _tqdm(range(0, num_nodes, bsz), total=total_batches, leave=False, desc=f"Snapshot {sidx} Batches"):
            end = min(num_nodes, start + bsz)
            batch_centers = centers[start:end]
            if not batch_centers:
                continue

            subs, node_counts, freq_weights = [], [], []
            x_list, e_list, ids_list = [], [], []

            # 构造 batch ego graph
            for c in batch_centers:
                sub = self._ego_subgraph(g, c, r=self.r_hop, max_nodes=self.ego_max_nodes)
                if sub.vcount() == 0:
                    continue
                subs.append(sub)
                node_counts.append(sub.vcount())
                ids_list.append([sub.vs[i]['name'] for i in range(sub.vcount())])
                x_list.append(torch.from_numpy(self._build_node_features(sub)).to(device))
                e_list.append(self._igraph_edges_to_edge_index(sub))
                freq = float(g.vs[c]['frequency']) if 'frequency' in g.vs.attributes() else 1.0
                freq_weights.append(1.0 + max(0.0, self.anomaly_alpha) * freq)

            if not subs:
                continue

            Bc = len(subs)
            offsets = np.cumsum([0] + node_counts[:-1]).tolist()
            graph_ids = torch.tensor(
                [gi for gi, n in enumerate(node_counts) for _ in range(n)], device=device
            )

            # 两视角增强
            def build_aug_views(x_list, e_list, offsets):
                e_cols1, e_cols2, x1, x2 = [], [], [], []
                for xi, ei, off in zip(x_list, e_list, offsets):
                    e_cols1.append(self._augment_edges(ei, self.drop_edge_p) + off)
                    e_cols2.append(self._augment_edges(ei, self.drop_edge_p) + off)
                    x1.append(self._augment_features(xi, self.feat_mask_p))
                    x2.append(self._augment_features(xi, self.feat_mask_p))
                return (
                    torch.cat(x1, dim=0), torch.cat(x2, dim=0),
                    torch.cat(e_cols1, dim=1), torch.cat(e_cols2, dim=1)
                )

            X1, X2, E1, E2 = build_aug_views(x_list, e_list, offsets)

            # 编码函数
            def encode_view(X, E):
                Z_layers = self.encoder(X, E, return_all=True)
                H_last = Z_layers[-1]
                sums = torch.zeros((Bc, H_last.size(1)), device=device)
                cnts = torch.zeros(Bc, device=device)
                sums.index_add_(0, graph_ids, H_last)
                cnts.index_add_(0, graph_ids, torch.ones_like(graph_ids, dtype=torch.float32))
                means = sums / (cnts.clamp_min(1e-6).unsqueeze(1))
                return F.normalize(self.proj_head(means), dim=-1)

            Z_view1 = encode_view(X1, E1)
            Z_view2 = encode_view(X2, E2)

            # 跨 batch 恶意负样本
            Z_neg_blocks = []
            freq_weights_neg = torch.zeros(Bc, device=device)

            if getattr(self, 'use_malicious_negatives', False) \
                    and len(self.malicious_token_counter) > 0 \
                    and len(self._ego_cache) > 0:

                all_subs, all_x, all_e, all_w = [], [], [], []
                for subs_prev, x_prev, e_prev, w_prev in self._ego_cache:
                    all_subs.extend(subs_prev)
                    all_x.extend(x_prev)
                    all_e.extend(e_prev)
                    all_w.extend(w_prev)

                total_prev = len(all_subs)
                if total_prev >= Bc:
                    idxs = np.random.choice(total_prev, size=Bc, replace=False)

                    X_neg_list, E_neg_list, node_counts_neg, freq_neg = [], [], [], []
                    for i in idxs:
                        sub, xi, ei, w = all_subs[i], all_x[i], all_e[i], all_w[i]
                        xneg_np = self._corrupt_features_with_malicious(
                            sub, xi.cpu().numpy(),
                            ratio=float(getattr(self, 'mal_neg_ratio', 0.3)),
                            token_len=int(getattr(self, 'mal_neg_token_len', 16))
                        )
                        X_neg_list.append(torch.from_numpy(xneg_np).to(device))
                        E_neg_list.append(ei)
                        node_counts_neg.append(sub.vcount())
                        freq_neg.append(w)

                    offsets_neg = np.cumsum([0] + node_counts_neg[:-1]).tolist()
                    graph_ids_neg = torch.tensor(
                        [gi for gi, n in enumerate(node_counts_neg) for _ in range(n)],
                        device=device
                    )
                    X_neg = torch.cat(X_neg_list, dim=0)

                    for _ in range(2):
                        e_cols = [self._augment_edges(ei, self.drop_edge_p) + off for ei, off in
                                  zip(E_neg_list, offsets_neg)]
                        EN = torch.cat(e_cols, dim=1)
                        XN = self._augment_features(X_neg, self.feat_mask_p)
                        ZN_layers = self.encoder(XN, EN, return_all=True)
                        NL = ZN_layers[-1]
                        sums = torch.zeros((Bc, NL.size(1)), device=device)
                        cnts = torch.zeros(Bc, device=device)
                        sums.index_add_(0, graph_ids_neg, NL)
                        cnts.index_add_(0, graph_ids_neg, torch.ones_like(graph_ids_neg, dtype=torch.float32))
                        means = sums / (cnts.clamp_min(1e-6).unsqueeze(1))
                        Z_neg_blocks.append(F.normalize(self.proj_head(means), dim=-1))

                    freq_weights_neg = torch.tensor(freq_neg, dtype=torch.float32, device=device)

            # 拼接视角
            Z_batch = torch.cat(
                [
                    torch.cat(
                        [Z_view1[gi:gi + 1], Z_view2[gi:gi + 1],
                         *(Z_neg_blocks[k][gi:gi + 1] for k in range(len(Z_neg_blocks)))],
                        dim=0
                    )
                    for gi in range(Bc)
                ],
                dim=0
            )

            # 权重：受 use_sample_weights 开关控制
            if self.use_sample_weights:
                sample_weights = []
                if len(Z_neg_blocks) == 2:
                    for w_pos, w_neg in zip(freq_weights, freq_weights_neg):
                        sample_weights.extend([w_pos, w_pos, w_neg, w_neg])
                else:
                    for w_pos in freq_weights:
                        sample_weights.extend([w_pos, w_pos])

                w_tensor = torch.tensor(sample_weights, dtype=torch.float32, device=device)
                assert Z_batch.shape[0] == w_tensor.shape[0], \
                    f"Weight mismatch: Z_batch={Z_batch.shape[0]}, w_tensor={w_tensor.shape[0]}"
            else:
                w_tensor = None

            # 损失计算
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._nt_xent_loss(Z_batch, temperature=self.temperature, sample_weights=w_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.proj_head.parameters()), max_norm=5.0
            )
            self.optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            total_steps += 1

            # 当前 batch 训练完再缓存
            self._ego_cache.append((subs, x_list, e_list, freq_weights))

        return total_loss / max(1, total_steps)


    # ---------- 恶意 tokens 支持（用于负样本） ----------
    def _precollect_malicious_tokens(self):
        """收集恶意节点的 tokens，用于生成对比学习的负样本"""
        if getattr(self, 'malicious_token_counter', None) is None:
            return
        if len(self.malicious_token_counter) > 0:
            return
        
        print("[恶意Token收集] 开始收集恶意节点的tokens...")
        
        # 统计信息
        total_nodes = 0
        malicious_nodes = 0
        total_snapshots = 0
        
        # 遍历全部快照，收集 label==1 的节点 tokens
        for g in self.snapshots:
            if g is None or g.vcount() == 0:
                continue
            total_snapshots += 1
            for i in range(g.vcount()):
                total_nodes += 1
                try:
                    lab = int(g.vs[i].attributes().get('label', 0))
                except Exception:
                    lab = 0
                if lab != 1:
                    continue
                malicious_nodes += 1
                self_tokens = self._get_node_tokens(g, i)
                self.malicious_token_counter.update(self_tokens)
        
        # 应用停用词过滤
        if self.mal_stopwords:
            total_before = sum(self.malicious_token_counter.values())
            filtered_counter = Counter()
            for token, count in self.malicious_token_counter.items():
                if token not in self.mal_stopwords:
                    filtered_counter[token] = count
            total_after = sum(filtered_counter.values())
            self.malicious_token_counter = filtered_counter
            if self.mal_print_tokens:
                print(f"[恶意Token] 停用词过滤: {total_before} tokens -> {total_after} tokens (去除 {total_before - total_after} 个)")
        
        # 打印统计信息
        if self.mal_print_tokens:
            total_mal_tokens = sum(self.malicious_token_counter.values())
            unique_mal_tokens = len(self.malicious_token_counter)
            top_mal_tokens = self.malicious_token_counter.most_common(20)
            
            print(f"\n{'='*60}")
            print("[恶意Token统计]")
            print(f"  处理快照数: {total_snapshots}")
            print(f"  总节点数: {total_nodes}")
            print(f"  恶意节点数: {malicious_nodes} ({malicious_nodes/max(1,total_nodes)*100:.2f}%)")
            print(f"  收集到的恶意token总数: {total_mal_tokens}")
            print(f"  去重后恶意词汇量: {unique_mal_tokens}")
            if malicious_nodes > 0:
                print(f"  平均每个恶意节点的token数: {total_mal_tokens/malicious_nodes:.2f}")
            print("  Top-20 高频恶意词:")
            for word, count in top_mal_tokens:
                print(f"    {word:20s} : {count:6d}")
            if self.mal_stopwords:
                print(f"  停用词数量: {len(self.mal_stopwords)}")
                print(f"  停用词列表: {sorted(list(self.mal_stopwords))[:10]}{'...' if len(self.mal_stopwords) > 10 else ''}")
            print(f"{'='*60}\n")

    def _sample_malicious_tokens(self, k: int) -> List[str]:
        """按恶意 token 频次做加权随机采样（允许重复）。
        - 权重来源：self.malicious_token_counter[token] 统计频次
        - 回退策略：若权重异常（和为0等），退回到均匀随机
        """
        if len(self.malicious_token_counter) == 0 or k <= 0:
            return []
        k = int(k)
        toks = list(self.malicious_token_counter.keys())
        weights = [max(0, int(self.malicious_token_counter[t])) for t in toks]
        try:
            # Python 内置按权重抽样（有放回）
            return random.choices(toks, weights=weights, k=k)
        except Exception:
            # 兼容回退：使用 numpy 实现；若权重无效则退回均匀随机
            w = np.asarray(weights, dtype=np.float64)
            s = float(w.sum())
            if s <= 0:
                return [random.choice(toks) for _ in range(k)]
            p = (w / s).astype(np.float64)
            idx = np.random.choice(len(toks), size=k, replace=True, p=p)
            return [toks[i] for i in idx]


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
                'use_temporal': self.use_temporal,
                'prop_feat_dim': self.prop_feat_dim,
                'enc_hidden_dim': self.enc_hidden_dim,
                'enc_out_dim': self.enc_out_dim,
                'gin_layers': self.gin_layers,
                'dropout': self.dropout,
                'num_epochs': self.num_epochs,
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
                'use_sample_weights': self.use_sample_weights,
                # Semantic settings
                'sem_fp_bits': self.sem_fp_bits,
                # W2V 配置
                'w2v_window': self.w2v_window,
                'w2v_min_count': self.w2v_min_count,
                'w2v_sg': self.w2v_sg,
                'w2v_epochs': self.w2v_epochs,
                'w2v_pretrained_path': self.w2v_pretrained_path,
                # 恶意Token配置
                'mal_stopwords': list(self.mal_stopwords) if self.mal_stopwords else [],
                'mal_print_tokens': self.mal_print_tokens,
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
            'use_temporal',
            'prop_feat_dim','enc_hidden_dim','enc_out_dim','gin_layers','dropout',
            'num_epochs','batch_size','lr','temperature',
            'r_hop','ego_max_nodes','drop_edge_p','feat_mask_p','train_indices','model_path',
            'anomaly_alpha','use_sample_weights',
            # W2V 配置
            'w2v_window','w2v_min_count','w2v_sg','w2v_epochs','w2v_pretrained_path',
            # 恶意Token配置
            'mal_stopwords','mal_print_tokens'
        }
        params = {k: v for k, v in raw_params.items() if k in allowed}
        inst = cls(snapshot_sequence, **params)
        # 恢复语义拉近配置（不作为构造参数传入，以保持构造签名简洁）
        try:
            inst.sem_fp_bits = int(raw_params.get('sem_fp_bits', inst.sem_fp_bits))
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

