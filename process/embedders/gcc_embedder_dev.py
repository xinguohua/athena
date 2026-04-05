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


# ----------------------- 边类别分组 -----------------------
# 4 个语义分组：进程操作、文件操作、网络操作、内存操作
EDGE_CATEGORY = {
    'EVENT_EXECUTE': 0, 'EVENT_FORK': 0, 'EVENT_CLONE': 0, 'EVENT_EXIT': 0,
    'EVENT_READ': 1, 'EVENT_WRITE': 1, 'EVENT_OPEN': 1, 'EVENT_CLOSE': 1,
    'EVENT_UNLINK': 1, 'EVENT_RENAME': 1,
    'EVENT_CONNECT': 2, 'EVENT_SENDTO': 2, 'EVENT_RECVFROM': 2, 'EVENT_ACCEPT': 2,
    'EVENT_MMAP': 3, 'EVENT_MPROTECT': 3,
}
NUM_EDGE_CATEGORIES = 4


def classify_edge(action_str: str) -> int:
    """将边的 action 字符串分类为 0-3 的类别索引。多操作取第一个有效操作。"""
    for act in action_str.split(','):
        act = act.strip()
        if act in EDGE_CATEGORY:
            return EDGE_CATEGORY[act]
    return 1  # 默认归为文件操作（最常见）


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


class TypedGINConv(nn.Module):
    """分组聚合 GIN 卷积层。

    按边类别（进程/文件/网络/内存）分开聚合邻居消息，拼接后通过 MLP。
    EXEC 邻居不会被 READ 邻居稀释——它们在独立通道。
    """
    def __init__(self, in_dim: int, out_dim: int, num_categories: int = NUM_EDGE_CATEGORIES,
                 dropout: float = 0.0):
        super().__init__()
        # 每个类别的聚合结果 + 自身特征 → MLP
        self.mlp = MLP(in_dim * (num_categories + 1), out_dim, out_dim, dropout)
        self.num_categories = num_categories

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_cat: torch.Tensor = None, **kwargs) -> torch.Tensor:
        n = x.size(0)
        d = x.size(1)

        if edge_index.numel() == 0 or edge_cat is None:
            # 无边：所有类别聚合为零
            agg = torch.zeros(n, d * self.num_categories, device=x.device)
        else:
            src, dst = edge_index[0], edge_index[1]
            # 按类别分组聚合
            agg_parts = []
            for cat_id in range(self.num_categories):
                mask = (edge_cat == cat_id)
                cat_agg = torch.zeros(n, d, device=x.device)
                if mask.any():
                    cat_agg.index_add_(0, dst[mask], x[src[mask]])
                agg_parts.append(cat_agg)
            agg = torch.cat(agg_parts, dim=1)  # [n, d * num_categories]

        # 拼接自身特征 + 各类别聚合
        combined = torch.cat([x, agg], dim=1)  # [n, d * (num_categories + 1)]
        return self.mlp(combined)


class GINEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 3, dropout: float = 0.1, **kwargs):
        super().__init__()
        num_layers = int(max(1, num_layers))
        # 第一层: in_dim*(C+1) → hidden_dim, 后续层: hidden_dim*(C+1) → hidden/out
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([
            TypedGINConv(dims[i], dims[i + 1], dropout=dropout) for i in range(num_layers)
        ])
        self.layer_dims = dims[1:]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_feat: torch.Tensor = None, return_all: bool = False, **kwargs):
        Zs = []
        h = x
        for conv in self.layers:
            h = conv(h, edge_index, edge_cat=edge_feat)
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
            enc_hidden_dim: int = 64,
            enc_out_dim: int = 256,
            gin_layers: int = 3,
            dropout: float = 0.1,
            # 训练
            num_epochs: int = 3,
            batch_size: int = 64,
            lr: float = 1e-3,
            # 对比学习
            temperature: float = 0.07,
            # 子图采样
            r_hop: int = 4,
            ego_max_nodes: int = 64,
            # 增强
            drop_edge_p: float = 0.2,
            feat_mask_p: float = 0.2,
            # 训练集选择
            train_indices: Optional[Union[Iterable[int], Tuple[int, int], int]] = None,
            model_path: Optional[str] = None,
            # 异常活跃驱动损失参数（仅频率异常）
            anomaly_alpha: float = 1,  # 加权强度，>0 表示异常越大权重越大
            # 是否使用样本权重（基于频率/异常强度），关闭时统一使用均匀权重
            use_sample_weights: bool = True,
            w2v_window: int = 5,
            w2v_min_count: int = 1,
            w2v_sg: int = 1,
            w2v_epochs: int = 20,
            w2v_pretrained_path: Optional[str] = None,
            use_malicious_snapshots: bool = True,
            use_malicious_negatives: bool = False,
            # 第三个开关：两种策略按比例混合
            combine: bool = False,
            combine_ratio: float = 0.8,
            mal_neg_ratio: float = 0.3,
            mal_neg_node_token_len: int = 1,
            mal_stopwords=None,
            # [
            # 'event', 'read', 'write'
            # , 'execute'
            # ],
            # 恶意token停用词列表，传入[]表示不过滤
            mal_print_tokens: bool = True,  # 是否打印恶意token统计信息
            # Top-K 相似（可选，先关闭）
            topk_pos: Optional[int] = 0,  # 先关闭 Top-K 扩增，回到经典 NT-Xent
            topk_pos_min_sim: float = 0.5,  # 仅当相似度 > 此阈值时才将样本纳入 Top-K 正样本
            # 新增：是否使用“度感知 点-边协同增强”（默认关闭，保持原策略不变）
            use_degree_coop_augment: bool = True,
                # 负样本权重缩放（超参数）：用于提高恶意样本在损失中的占比
                neg_weight_scale: float = 100.0,
                # 快照聚合“权重混合”系数：attr_weight_alpha ∈ [0,1]
                # 使用两个权重向量做加权相加：
                #   - w_base: 节点基础权重（frequency 优先，其次 degree）
                #   - w_attr: 属性稀少权重（来自 g 内属性相对频率的反比）
                # 最终节点权重：w_eff = (1 - alpha) * norm(w_base) + alpha * norm(w_attr)
                attr_weight_alpha: float = 0.3,
    ):
        super().__init__(snapshots, features, mapp)
        if mal_stopwords is None:
            mal_stopwords = []
            #     [
            #     'event', 'read', 'write', 'execute'
            # ]
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

        self.use_malicious_snapshots = bool(use_malicious_snapshots)
        # 是否使用“恶意语料”来生成额外负样本；以及腐化强度与每个节点替换的 token 数
        self.use_malicious_negatives = bool(use_malicious_negatives)
        # 组合策略（第三开关 + 比例）
        self.combine = bool(combine)
        self.combine_ratio = float(combine_ratio)
        self.mal_neg_ratio = float(mal_neg_ratio)  # 每个子图中替换为恶意向量的节点比例
        self.mal_neg_node_token_len = int(mal_neg_node_token_len)  # 生成恶意向量时采样的恶意节点数
        self.mal_use_type_group = False

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
        # 增强策略开关
        self.use_degree_coop_augment = bool(use_degree_coop_augment)
        # 负样本权重缩放超参数
        self.neg_weight_scale = float(neg_weight_scale)
        # 属性频率降权参数（单一 alpha）
        self.attr_weight_alpha = float(attr_weight_alpha)
        # 是否使用正子图融合恶意子图构造负样本（调用 _build_neg_block_from_snapshots_with_pos）
        self.use_pos_fusion_neg = True  # 运行时可直接设 True 开启
        self.pos_fusion_ratio = 0.5  # 正子图内部节点采样比例
        self.pos_fusion_cross_ratio = 0.2  # 跨连边比例
        self.pos_fusion_cross_max = 8  # 跨连边最大数

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
        self.malicious_node_tokens = []

        # 训练快照索引
        self.train_snapshot_indices = self._resolve_train_indices(train_indices)

        # 编码器 + 投影头（对比用）
        in_dim = self.prop_feat_dim if self.prop_feat_dim > 0 else 1
        self.encoder = GINEncoder(in_dim, self.enc_hidden_dim, self.enc_out_dim, num_layers=self.gin_layers,
                                  dropout=self.dropout).to(self.device)
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
        if self.combine:
            # 组合策略：两类都准备
            self._precollect_malicious_tokens()
            self._precollect_malicious_snapshots()
        else:
            if self.use_malicious_negatives:
                self._precollect_malicious_tokens()
            if self.use_malicious_snapshots:
                self._precollect_malicious_snapshots()

        print(
            f"[GCC-Dev] Pretrain on {len(self.train_snapshot_indices)} snapshots | batch={self.batch_size} | tau={self.temperature}")

        for epoch in range(self.num_epochs):
            if self.use_temporal:
                self.temporal.reset()  # 每个 epoch 重置时序状态（仅开启时）
            epoch_loss = 0.0
            steps_done = 0

            # 按时间顺序遍历 snapshot，小快照打包一次训练
            SMALL_THRESHOLD = 64  # 节点数 <= 此值视为小快照
            sorted_indices = sorted(self.train_snapshot_indices)
            small_batch = []  # 收集连续小快照

            for sidx in sorted_indices:
                g = self.snapshots[sidx]
                if g is None or g.vcount() == 0:
                    continue

                if g.vcount() <= SMALL_THRESHOLD:
                    small_batch.append((sidx, g))
                    # 攒够一批或到最后一个才训练
                    if len(small_batch) < 16 and sidx != sorted_indices[-1]:
                        continue
                    # 打包训练小快照
                    batch_loss = self._train_small_snapshots_packed(small_batch)
                    n_packed = len(small_batch)
                    epoch_loss += batch_loss * n_packed
                    steps_done += n_packed
                    if n_packed > 1:
                        print(f"[GCC-Dev] Epoch {epoch + 1}/{self.num_epochs} | Packed {n_packed} small snapshots (idx {small_batch[0][0]}~{small_batch[-1][0]}) | Loss={batch_loss:.6f}")
                    else:
                        print(f"[GCC-Dev] Epoch {epoch + 1}/{self.num_epochs} | Snapshot {small_batch[0][0]} | Loss={batch_loss:.6f}")
                    small_batch = []
                else:
                    # 先清空小快照缓冲
                    if small_batch:
                        bl = self._train_small_snapshots_packed(small_batch)
                        n_packed = len(small_batch)
                        epoch_loss += bl * n_packed
                        steps_done += n_packed
                        print(f"[GCC-Dev] Epoch {epoch + 1}/{self.num_epochs} | Packed {n_packed} small snapshots (idx {small_batch[0][0]}~{small_batch[-1][0]}) | Loss={bl:.6f}")
                        small_batch = []

                    # 大图单独训练
                    batch_loss = self._train_one_snapshot(g, sidx=sidx)
                    epoch_loss += batch_loss
                    steps_done += 1
                    torch.cuda.empty_cache()
                    print(f"[GCC-Dev] Epoch {epoch + 1}/{self.num_epochs} | Snapshot {sidx} | Loss={batch_loss:.6f}")

            # 清空末尾残余
            if small_batch:
                bl = self._train_small_snapshots_packed(small_batch)
                n_packed = len(small_batch)
                epoch_loss += bl * n_packed
                steps_done += n_packed
                print(f"[GCC-Dev] Epoch {epoch + 1}/{self.num_epochs} | Packed {n_packed} small snapshots | Loss={bl:.6f}")

            avg = epoch_loss / max(1, steps_done)
            print(f"[GCC-Dev] Epoch {epoch + 1}/{self.num_epochs} DONE | AvgLoss={avg:.6f}")

        self.save_malicious_snapshot_stats()
        # 训练结束后生成节点嵌入：遵循全局开关 self.use_temporal
        self.generate_node_embeddings(use_temporal=self.use_temporal)
        self.save_model()

    def _train_small_snapshots_packed(self, snapshot_batch: list) -> float:
        """多个小快照打包成一个 batch 训练，减少 per-snapshot 开销。

        Args:
            snapshot_batch: [(sidx, graph), ...]

        Returns:
            平均 loss
        """
        device = self.device
        all_x, all_e, all_node_counts = [], [], []
        total_nodes_offset = 0

        all_ef = []
        for sidx, g in snapshot_batch:
            if g is None or g.vcount() == 0:
                continue
            self._preheat_snapshot_properties(g)
            x_np = self._build_node_features(g)
            x_t = torch.from_numpy(x_np).to(device)
            e_t, ef_t = self._igraph_edges_to_edge_index(g)
            if e_t.numel() > 0:
                e_t = e_t + total_nodes_offset
            all_x.append(x_t)
            all_e.append(e_t)
            all_ef.append(ef_t)
            all_node_counts.append(g.vcount())
            total_nodes_offset += g.vcount()

        if not all_x:
            return 0.0

        Bc = len(all_x)
        X_pos = torch.cat(all_x, dim=0)
        E_pos = torch.cat(all_e, dim=1) if any(e.numel() > 0 for e in all_e) else torch.zeros((2, 0), dtype=torch.long, device=device)
        EF_pos = torch.cat(all_ef, dim=0) if all_ef else None
        graph_ids = torch.tensor(
            [gi for gi, n in enumerate(all_node_counts) for _ in range(n)], device=device
        )

        # Forward
        Z_layers = self.encoder(X_pos, E_pos, edge_feat=EF_pos, return_all=True)
        H_last = Z_layers[-1]
        sums = torch.zeros((Bc, H_last.size(1)), device=device)
        cnts = torch.zeros(Bc, device=device)
        sums.index_add_(0, graph_ids, H_last)
        cnts.index_add_(0, graph_ids, torch.ones_like(graph_ids, dtype=torch.float32))
        means = sums / (cnts.clamp_min(1e-6).unsqueeze(1))
        Z_pos = F.normalize(self.proj_head(means), dim=-1)

        # 负样本 N(b) = 攻击池采样 + 专属变异图 ego 子图
        Z_neg_parts = []

        has_ego = bool(self.use_malicious_snapshots and hasattr(self, '_mal_ego_pool') and len(
            self._mal_ego_pool) > 0)
        if has_ego:
            Z_attack = self._build_neg_augmented(Bc, device, mode='standard')
            if Z_attack is not None:
                Z_neg_parts.append(Z_attack)

        # 打包的小快照：收集所有专属变异图的 ego 子图
        mutation_map = getattr(self, 'mutation_map', None)
        if mutation_map:
            for sidx, _ in snapshot_batch:
                if sidx in mutation_map:
                    g_neg = mutation_map[sidx]
                    try:
                        Z_mut = self._encode_ego_subgraphs_from_graph(g_neg, device)
                        if Z_mut is not None and Z_mut.size(0) > 0:
                            Z_neg_parts.append(Z_mut)
                    except Exception:
                        pass

        Z_neg = torch.cat(Z_neg_parts, dim=0) if Z_neg_parts else None

        # Loss + backward
        self.optimizer.zero_grad(set_to_none=True)
        loss = self._weighted_contrastive_loss(Z_pos, Z_neg, temperature=self.temperature)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.proj_head.parameters()), max_norm=5.0
        )
        self.optimizer.step()
        return float(loss.detach().cpu().item())

    def _encode_ego_subgraphs_from_graph(self, g, device) -> Optional[torch.Tensor]:
        """从图中提取 label=1 节点的 ego 子图，编码为嵌入向量。

        返回 [N_attack, D]，每个攻击节点一个 ego 子图嵌入。
        粒度跟 _mal_ego_pool 一致。
        """
        # 找 label=1 节点
        attack_centers = []
        labels = g.vs['label'] if 'label' in g.vs.attributes() else None
        if labels is None:
            return None
        for i, lab in enumerate(labels):
            if int(lab) == 1:
                attack_centers.append(i)
        if not attack_centers:
            return None

        self._preheat_snapshot_properties(g)
        ego_embeddings = []

        for c in attack_centers:
            sub = self._ego_subgraph(g, c, r=self.r_hop, max_nodes=self.ego_max_nodes)
            if sub.vcount() == 0:
                continue
            x_np = self._build_node_features(sub)
            x_t = torch.from_numpy(x_np).to(device)
            e_t, ef_t = self._igraph_edges_to_edge_index(sub)

            Z_layers = self.encoder(x_t, e_t, edge_feat=ef_t, return_all=True)
            H = Z_layers[-1]
            graph_emb = H.mean(dim=0, keepdim=True)
            ego_embeddings.append(graph_emb)

        if not ego_embeddings:
            return None

        Z = torch.cat(ego_embeddings, dim=0)  # [N_attack, hidden]
        return F.normalize(self.proj_head(Z), dim=-1)  # [N_attack, D]

    def _train_one_snapshot(self, g, sidx: Optional[int] = None) -> float:
        """单个 snapshot 训练（优化 + 稳定版 NT-Xent）"""
        device = self.device
        num_nodes = g.vcount()
        if num_nodes == 0:
            return 0.0

        # 快照级属性预热：将本快照所有节点 properties 向量化并写入缓存，避免训练内首次命中开销
        self._preheat_snapshot_properties(g)

        # ------- Step 1: 全局带权随机采样 (不降权) -------
        # 大图采样上限：避免节点过多导致显存爆炸
        MAX_CENTERS_PER_SNAPSHOT = 512  # 512 足够对比学习，减少 ego 子图提取开销
        sample_size = min(num_nodes, MAX_CENTERS_PER_SNAPSHOT)

        centers = list(range(num_nodes))
        if 'frequency' in g.vs.attributes():
            freqs = np.array([float(f) for f in g.vs['frequency']])
            freqs = freqs + 1e-6  # 防止为0
            probs = freqs / freqs.sum()
        else:
            probs = np.ones(num_nodes) / num_nodes

        # 按频次随机采样，大图时限制采样数量
        sampled_centers = np.random.choice(centers, size=sample_size, replace=(sample_size > num_nodes), p=probs)
        centers = sampled_centers.tolist()

        # ------- Step 2: 初始化 -------
        total_loss, total_steps = 0.0, 0
        bsz = max(1, int(self.batch_size))
        total_batches = math.ceil(len(centers) / bsz)
        print(f"  [Snapshot {sidx}] nodes={num_nodes}, sampled={sample_size}, batches={total_batches}")

        # 初始化缓存（保存之前 batch 的 subs, x_list, e_list, freq_weights）
        if not hasattr(self, "_ego_cache"):
            self._ego_cache = deque(maxlen=8)

        # ------- Step 3: 按 batch 训练 -------

        n_centers = len(centers)
        for start in _tqdm(range(0, n_centers, bsz), total=total_batches, leave=False, desc=f"Snapshot {sidx} Batches"):
            end = min(n_centers, start + bsz)
            batch_centers = centers[start:end]
            if not batch_centers:
                continue

            subs, node_counts, freq_weights = [], [], []
            x_list, e_list, ef_list, ids_list = [], [], [], []

            # 构造 batch ego graph
            for c in batch_centers:
                sub = self._ego_subgraph(g, c, r=self.r_hop, max_nodes=self.ego_max_nodes)
                if sub.vcount() == 0:
                    continue
                xi_t = torch.from_numpy(self._build_node_features(sub)).to(device)
                ei_t, efi_t = self._igraph_edges_to_edge_index(sub)
                subs.append(sub)
                node_counts.append(sub.vcount())
                ids_list.append([sub.vs[i]['name'] for i in range(sub.vcount())])
                x_list.append(xi_t)
                e_list.append(ei_t)
                ef_list.append(efi_t)
                freq = float(g.vs[c]['frequency']) if 'frequency' in g.vs.attributes() else 1.0
                freq_weights.append(1.0 + max(0.0, self.anomaly_alpha) * freq)

            if not subs:
                continue

            Bc = len(subs)
            offsets = np.cumsum([0] + node_counts[:-1]).tolist()
            graph_ids = torch.tensor(
                [gi for gi, n in enumerate(node_counts) for _ in range(n)], device=device
            )

            # ======== 正样本：良性 ego subgraph 直接编码 ========
            X_pos = torch.cat(x_list, dim=0)
            E_pos_cols = [ei + off for ei, off in zip(e_list, offsets)]
            E_pos = torch.cat(E_pos_cols, dim=1)
            EF_pos = torch.cat(ef_list, dim=0) if ef_list else None

            def encode_batch(X, E, n_graphs, gids, ef=None):
                Z_layers = self.encoder(X, E, edge_feat=ef, return_all=True)
                H_last = Z_layers[-1]
                sums = torch.zeros((n_graphs, H_last.size(1)), device=device)
                cnts = torch.zeros(n_graphs, device=device)
                sums.index_add_(0, gids, H_last)
                cnts.index_add_(0, gids, torch.ones_like(gids, dtype=torch.float32))
                means = sums / (cnts.clamp_min(1e-6).unsqueeze(1))
                return F.normalize(self.proj_head(means), dim=-1)

            Z_pos = encode_batch(X_pos, E_pos, Bc, graph_ids, ef=EF_pos)  # [Bc, D]

            # ======== 负样本 N(b) = 攻击池采样(共享) + G̃_b ego子图(专属) ========
            Z_neg_parts = []

            # (1) 共享攻击池：随机采样 Bc 个 ego 子图
            has_ego = bool(self.use_malicious_snapshots and hasattr(self, '_mal_ego_pool') and len(
                self._mal_ego_pool) > 0)
            if has_ego:
                if getattr(self, 'mimicry_mode', False):
                    Z_attack = self._build_neg_augmented(
                        Bc, device, x_list, e_list, node_counts, mode='mimicry')
                else:
                    Z_attack = self._build_neg_augmented(Bc, device, mode='standard')
                if Z_attack is not None:
                    Z_neg_parts.append(Z_attack)

            # (2) 专属变异图：提取 label=1 节点的 ego 子图（粒度跟攻击池一致）
            mutation_map = getattr(self, 'mutation_map', None)
            if mutation_map and sidx is not None and sidx in mutation_map:
                g_neg = mutation_map[sidx]
                try:
                    Z_mut = self._encode_ego_subgraphs_from_graph(g_neg, device)
                    if Z_mut is not None and Z_mut.size(0) > 0:
                        Z_neg_parts.append(Z_mut)
                except Exception:
                    pass

            Z_neg = torch.cat(Z_neg_parts, dim=0) if Z_neg_parts else None

            # ======== 损失计算：有监督加权对比损失（论文公式 5, 6） ========
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._weighted_contrastive_loss(
                Z_pos, Z_neg,
                temperature=self.temperature,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.proj_head.parameters()), max_norm=5.0
            )
            self.optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            total_steps += 1

            # 缓存到 CPU（仅在需要 token 负样本时才存，避免无谓的显存/内存占用）
            if self.use_malicious_negatives:
                cpu_x = [xi.detach().cpu() for xi in x_list]
                cpu_e = [ei.cpu() for ei in e_list]
                self._ego_cache.append((subs, cpu_x, cpu_e, freq_weights))

            # 定期释放 GPU 碎片
            if total_steps % 50 == 0:
                torch.cuda.empty_cache()

        return total_loss / max(1, total_steps)

    def _preheat_snapshot_properties(self, g) -> None:
        """将快照中所有节点的 properties 预先向量化写入 `_prop_cache`。
        不改变外部语义，仅减少训练期间的首次计算开销。"""
        n = g.vcount()
        if n == 0:
            return
        if self.prop_feat_dim <= 0:
            return
        if self._w2v_model is None:
            self._ensure_w2v_model()

        # 批量读取 properties
        if 'properties' in g.vs.attributes():
            try:
                props: List[str] = [str(p) for p in g.vs['properties']]
            except Exception:
                props = [str(g.vs[i].attributes().get('properties', '')) for i in range(n)]
        else:
            props = [str(g.vs[i].attributes().get('properties', '')) for i in range(n)]

        # 仅处理未缓存的唯一键
        uncached = {k for k in props if k not in self._prop_cache}
        if not uncached:
            return
        for key in uncached:
            tokens = self._tokenize_properties(key)
            vec = self._w2v_vector_from_tokens(tokens)
            self._prop_cache[key] = vec

    # ---------- 恶意 tokens 支持（用于负样本） ----------
    def _precollect_malicious_tokens(self, save_path: str = "malicious_tokens_log.txt"):
        """收集恶意节点 tokens（节点级组织 + 可保存到文件），同时记录每个节点的来源快照索引。"""
        if getattr(self, "malicious_node_tokens", None) is None:
            self.malicious_node_tokens = []
        # 新增：一一对应的来源快照索引列表
        if getattr(self, "malicious_node_origin", None) is None:
            self.malicious_node_origin = []

        total_nodes, malicious_nodes, total_snapshots = 0, 0, 0
        stop = getattr(self, "mal_stopwords", set())

        with open(save_path, "w", encoding="utf-8") as f:
            f.write("[恶意Token收集日志]\n")
            f.write("=" * 60 + "\n")

            for snap_idx, g in enumerate(self.snapshots):
                if g is None or g.vcount() == 0:
                    continue
                total_snapshots += 1

                snapshot_node_map = {}

                for i in range(g.vcount()):
                    total_nodes += 1
                    try:
                        lab = int(g.vs[i].attributes().get("label", 0))
                    except Exception:
                        lab = 0
                    if lab != 1:
                        continue

                    malicious_nodes += 1
                    toks = self._get_node_tokens(g, i)
                    if stop:
                        toks = [t for t in toks if t not in stop]
                    if not toks:
                        continue

                    # 存储 tokens，并记录该节点来自的 snapshot index
                    self.malicious_node_tokens.append(toks)
                    self.malicious_node_origin.append(snap_idx)

                    snapshot_node_map[i] = toks

                # 打印 + 写文件
                if snapshot_node_map:
                    header = f"\n[Snapshot {snap_idx:02d}] 恶意节点token映射:"
                    print(header)
                    f.write(header + "\n")
                    for nid, toks in snapshot_node_map.items():
                        line = f"  {nid}: {toks}"
                        print(line)
                        f.write(line + "\n")

            # 汇总统计
            from collections import Counter
            counter = Counter(t for toks in self.malicious_node_tokens for t in toks)
            total_mal_tokens = sum(counter.values())

            summary = "\n" + "=" * 60 + "\n"
            summary += "[恶意Token统计-节点级]\n"
            summary += f"  快照数: {total_snapshots}\n"
            summary += f"  总节点数: {total_nodes}\n"
            summary += f"  恶意节点数: {malicious_nodes}\n"
            summary += f"  恶意节点token集合数: {len(self.malicious_node_tokens)}\n"
            summary += f"  收集到的token总数: {total_mal_tokens}\n"
            summary += f"  Top-10: {counter.most_common(10)}\n"
            if stop:
                summary += f"  停用词数量: {len(stop)}\n"
            summary += "=" * 60 + "\n"

            print(summary)
            f.write(summary)

        print(f"[✅ 日志已保存到]: {save_path}")

    def _sample_malicious_tokens(self, num_nodes: int) -> List[str]:
        """
        从按节点组织的恶意语料中抽取 token 列表（每个被抽中的节点的全部 token 都会被收集）。
        - num_nodes: 要抽取多少个恶意节点（尽量无重复）。若 num_nodes > 可用节点数，会补齐（允许重复）。
        返回：flatten 后的 token 列表（长度 = sum(len(node_tokens) for chosen nodes)；若语料不足，返回已有的）。
        --- 注意：不再按 token 数限制，每个被选节点的所有 token 都被采纳（整节点替换）。
        """
        num_nodes = int(max(0, num_nodes))
        if num_nodes == 0:
            return []

        # 要求存在按节点存储的恶意语料
        if not hasattr(self, "malicious_node_tokens") or not self.malicious_node_tokens:
            return []

        #  初始化全局统计计数器（一次性创建）
        if not hasattr(self, "malicious_snapshot_stats"):
            self.malicious_snapshot_stats = {}  # {snapshot_id: count}

        node_lists = self.malicious_node_tokens  # List[List[str]]
        node_origins = getattr(self, "malicious_node_origin", None)  # List[int]
        total_nodes = len(node_lists)

        # 先选节点索引：若足够则无重复抽样，否则先取全部再补齐（允许重复补齐）
        if total_nodes >= num_nodes:
            chosen_idx = random.sample(range(total_nodes), k=num_nodes)
        else:
            chosen_idx = list(range(total_nodes))
            need = num_nodes - total_nodes
            if total_nodes > 0 and need > 0:
                chosen_idx.extend(random.choices(range(total_nodes), k=need))

        out_tokens: List[str] = []
        for idx in chosen_idx:
            toks = node_lists[idx]
            if not toks:
                continue
            # 把该节点的所有 token 全部加入（不做截断）
            out_tokens.extend(toks)
            if node_origins is not None:
                sid = node_origins[idx]
                self.malicious_snapshot_stats[sid] = self.malicious_snapshot_stats.get(sid, 0) + 1

        # 兜底：若所有选节点都没有 token（极端），从所有节点打平随机抽若干补齐（保证非空时尽量返回东西）
        if not out_tokens:
            flat = [t for toks in node_lists for t in toks]
            if not flat:
                return []
            out_tokens.append(random.choice(flat))

        return out_tokens

    # ---------- 恶意快照池与负样本块 ----------
    def _precollect_malicious_snapshots(self):
        """
        收集恶意节点的 ego 图采样池：
        - 每次运行都会新建（覆盖） malicious_tokens_log.txt；
        - 以 (snapshot_idx, local_node_idx) 保存；
        - 如实记录节点属性，不截断；
        - 输出和日志均完整。
        """
        self._mal_ego_pool: List[Tuple[int, int]] = []
        train_ids = list(range(len(self.snapshots)))
        per_snapshot_mal: Dict[int, int] = {}
        log_path = "malicious_tokens_log.txt"

        # --- 每次运行都新建日志文件（覆盖旧的） ---
        try:
            f_log = open(log_path, "w", encoding="utf-8")  # ⚠️ 用 "w" 模式重写
            f_log.write("[恶意EGO子图收集 - 完整记录模式]\n" + "=" * 80 + "\n")
        except Exception as e:
            print(f"[警告] 无法创建日志文件: {e}")
            f_log = None

        # 缓存恶意ego子图的特征与边，避免后续融合重复构造
        if not hasattr(self, '_mal_ego_cache'):
            self._mal_ego_cache: List[Tuple[torch.Tensor, torch.Tensor, int]] = []  # (X, edge_index, node_count)

        for sidx in train_ids:
            g = self.snapshots[sidx]
            if g is None or g.vcount() == 0:
                continue

            try:
                labels = g.vs['label'] if 'label' in g.vs.attributes() else None
            except Exception:
                labels = None

            if labels is not None:
                iterator = enumerate(labels)
            else:
                iterator = ((i, g.vs[i].attributes().get('label', 0)) for i in range(g.vcount()))

            for i, lab in iterator:
                try:
                    if int(lab) == 1:
                        self._mal_ego_pool.append((sidx, i))
                        per_snapshot_mal[sidx] = per_snapshot_mal.get(sidx, 0) + 1

                        sub = self._ego_subgraph(g, center=i, r=self.r_hop, max_nodes=self.ego_max_nodes)
                        nv, ne = sub.vcount(), sub.ecount()
                        line = f"[Snapshot {sidx:02d}] center={i} -> ego(nodes={nv}, edges={ne})"
                        print(line)
                        if f_log:
                            f_log.write(line + "\n")

                        # 预先缓存特征与边索引，供融合负样本使用，减少重复计算
                        try:
                            x_m_np = self._build_node_features(sub)
                            e_m, ef_m = self._igraph_edges_to_edge_index(sub)
                            self._mal_ego_cache.append((torch.from_numpy(x_m_np), e_m, sub.vcount(), ef_m))
                        except Exception as ce:
                            if f_log:
                                f_log.write(f"[缓存失败] Snapshot {sidx:02d} center={i}: {ce}\n")

                        # ---- 完整记录节点属性 ----
                        for vi in range(sub.vcount()):
                            v = sub.vs[vi]
                            attrs = v.attributes()
                            name = str(attrs.get('name', ''))
                            lab = str(attrs.get('label', ''))
                            freq = str(attrs.get('frequency', ''))
                            prop = str(attrs.get('properties', ''))
                            node_line = f"    node[{vi}]: name={name} label={lab} freq={freq} props={prop}"
                            print(node_line)
                            if f_log:
                                f_log.write(node_line + "\n")

                except Exception as ex:
                    warn = f"[EGO保存失败] Snapshot {sidx:02d}, center={i}: {ex}"
                    print(warn)
                    if f_log:
                        f_log.write(warn + "\n")

        # --- 汇总统计 ---
        print(f"[恶意EGO] 已收集: {len(self._mal_ego_pool)} 个恶意中心，日志已写入 {log_path}")
        if f_log:
            try:
                f_log.write("\n[汇总各快照恶意中心数]\n")
                for sid in sorted(per_snapshot_mal.keys()):
                    m = per_snapshot_mal.get(sid, 0)
                    f_log.write(f"  Snapshot {sid:02d}: 恶意中心={m}\n")
                f_log.write("=" * 80 + "\n")
            finally:
                f_log.close()

    def _build_neg_augmented(self, Bc: int, device: torch.device,
                             benign_x_list=None, benign_e_list=None,
                             benign_node_counts=None, mode='standard'):
        """
        有监督对比学习：构造 Bc 个增强后的负样本嵌入。
        增强只作用在负样本（攻击 ego subgraph）侧。

        mode:
          'standard' — 用当前策略的增强（drop_edge_p / feat_mask_p / degree_aware）
          'mimicry'  — 向恶意子图注入良性边和特征

        返回: Z_neg [Bc, D] 或 None
        """
        pool = self._mal_ego_pool if hasattr(self, '_mal_ego_pool') else None
        if not pool:
            return None

        chosen = [random.choice(pool) for _ in range(Bc)]

        _ef_zero = torch.zeros(0, dtype=torch.long, device=device)
        neg_x_list, neg_e_list, neg_ef_list, neg_node_counts = [], [], [], []
        for (sidx, center) in chosen:
            g = self.snapshots[sidx]
            if g is None or g.vcount() == 0:
                neg_x_list.append(torch.zeros((1, self.prop_feat_dim), dtype=torch.float32, device=device))
                neg_e_list.append(torch.zeros((2, 0), dtype=torch.long, device=device))
                neg_ef_list.append(_ef_zero)
                neg_node_counts.append(1)
                continue
            try:
                sub = self._ego_subgraph(g, center=center, r=self.r_hop, max_nodes=self.ego_max_nodes)
            except Exception:
                sub = None
            if sub is None or sub.vcount() == 0:
                neg_x_list.append(torch.zeros((1, self.prop_feat_dim), dtype=torch.float32, device=device))
                neg_e_list.append(torch.zeros((2, 0), dtype=torch.long, device=device))
                neg_ef_list.append(_ef_zero)
                neg_node_counts.append(1)
                continue
            x_np = self._build_node_features(sub)
            eidx, ef = self._igraph_edges_to_edge_index(sub)
            neg_x_list.append(torch.from_numpy(x_np).to(device))
            neg_e_list.append(eidx.to(device))
            neg_ef_list.append(ef.to(device))
            neg_node_counts.append(sub.vcount())

        if sum(neg_node_counts) == 0:
            return None

        Bn = len(neg_node_counts)
        offsets_neg = np.cumsum([0] + neg_node_counts[:-1]).tolist()
        graph_ids_neg = torch.tensor(
            [gi for gi, n in enumerate(neg_node_counts) for _ in range(n)], device=device
        )

        # 拼接原始特征和边
        X_neg_raw = torch.cat(neg_x_list, dim=0)

        # ---- 根据策略增强负样本 ----
        if mode == 'mimicry' and benign_x_list is not None and len(benign_x_list) > 0:
            # Mimicry [32]：向攻击节点连接良性边，模糊恶意信号
            # 将良性 ego 的边注入到恶意 ego 中，使攻击图的连接模式更像良性
            X_neg_aug = X_neg_raw
            aug_e_cols = []
            total_neg_nodes = X_neg_raw.size(0)
            benign_all_x = torch.cat(benign_x_list, dim=0)
            n_benign_nodes = benign_all_x.size(0)

            for ei, off, nc in zip(neg_e_list, offsets_neg, neg_node_counts):
                aug_e_cols.append(ei + off)
                if nc > 0 and n_benign_nodes > 0:
                    # 为每个恶意子图添加若干条良性边（连接恶意节点到其他恶意节点，用良性边模式）
                    n_inject = max(1, nc // 3)  # 注入约 1/3 节点数的边
                    src = torch.randint(off, off + nc, (n_inject,), device=device)
                    dst = torch.randint(off, off + nc, (n_inject,), device=device)
                    injected = torch.stack([src, dst], dim=0)
                    aug_e_cols.append(injected)

            E_neg = torch.cat(aug_e_cols, dim=1)
        else:
            E_neg_cols = [ei + off for ei, off in zip(neg_e_list, offsets_neg)]
            E_neg = torch.cat(E_neg_cols, dim=1)
            X_neg_aug = X_neg_raw

        EF_neg = torch.cat(neg_ef_list, dim=0) if neg_ef_list else None

        # 编码
        Z_layers = self.encoder(X_neg_aug, E_neg, edge_feat=EF_neg, return_all=True)
        H_last = Z_layers[-1]
        sums = torch.zeros((Bn, H_last.size(1)), device=device)
        cnts = torch.zeros(Bn, device=device)
        sums.index_add_(0, graph_ids_neg, H_last)
        cnts.index_add_(0, graph_ids_neg, torch.ones_like(graph_ids_neg, dtype=torch.float32))
        means = sums / (cnts.clamp_min(1e-6).unsqueeze(1))
        return F.normalize(self.proj_head(means), dim=-1)  # [Bn, D]

    def _build_neg_block_from_snapshots(self, Bc: int, device: torch.device):
        """从恶意节点 ego 池中采样 Bc 个中心节点，构造其 r-hop ego 子图并编码成两个视角的负样本块（每块 Bc×D）。
        返回: ([Z_neg_block_view1[Bc,D], Z_neg_block_view2[Bc,D]], freq_weights_neg[Bc])；若池为空返回 ([], zeros)。
        """
        pool = self._mal_ego_pool if hasattr(self, '_mal_ego_pool') else None
        if not pool:
            return [], torch.zeros(Bc, device=device)

        # 若池子不足，允许有放回采样
        if len(pool) >= Bc:
            chosen = random.sample(pool, k=Bc)
        else:
            chosen = [random.choice(pool) for _ in range(Bc)]

        x_list, e_list, node_counts = [], [], []
        for (sidx, center) in chosen:
            g = self.snapshots[sidx]
            if g is None or g.vcount() == 0:
                x_list.append(torch.zeros((0, self.prop_feat_dim), dtype=torch.float32, device=device))
                e_list.append(torch.zeros((2, 0), dtype=torch.long, device=device))
                node_counts.append(0)
                continue
            try:
                sub = self._ego_subgraph(g, center=center, r=self.r_hop, max_nodes=self.ego_max_nodes)
            except Exception:
                sub = None
            if sub is None or sub.vcount() == 0:
                x_list.append(torch.zeros((0, self.prop_feat_dim), dtype=torch.float32, device=device))
                e_list.append(torch.zeros((2, 0), dtype=torch.long, device=device))
                node_counts.append(0)
                continue
            x_np = self._build_node_features(sub)
            eidx, _ef = self._igraph_edges_to_edge_index(sub)
            x_list.append(torch.from_numpy(x_np).to(device))
            e_list.append(eidx.to(device))
            node_counts.append(sub.vcount())

        if sum(node_counts) == 0:
            return [], torch.zeros(Bc, device=device)

        offsets_neg = np.cumsum([0] + node_counts[:-1]).tolist()
        graph_ids_neg = torch.tensor(
            [gi for gi, n in enumerate(node_counts) for _ in range(n)],
            device=device
        )
        X_neg = torch.cat([xi for xi in x_list if xi.numel() > 0], dim=0) if any(
            n > 0 for n in node_counts) else torch.zeros((0, self.prop_feat_dim), device=device)

        # 两次增强与编码，得到两视角负样本块
        Z_blocks: List[torch.Tensor] = []
        for _ in range(2):
            if any(n > 0 for n in node_counts):
                if self.use_degree_coop_augment:
                    e_cols = [self._augment_edges_degree_aware(ei, self.drop_edge_p) + off for ei, off in
                              zip(e_list, offsets_neg)]
                else:
                    e_cols = [self._augment_edges(ei, self.drop_edge_p) + off for ei, off in zip(e_list, offsets_neg)]
            else:
                e_cols = []
            EN = torch.cat(e_cols, dim=1) if e_cols else torch.zeros((2, 0), dtype=torch.long, device=device)
            if self.use_degree_coop_augment:
                XN = self._augment_features_degree_aware(X_neg, self.feat_mask_p, EN)
            else:
                XN = self._augment_features(X_neg, self.feat_mask_p)
            ZN_layers = self.encoder(XN, EN, edge_feat=None, return_all=True)
            NL = ZN_layers[-1]
            sums = torch.zeros((Bc, NL.size(1)), device=device)
            cnts = torch.zeros(Bc, device=device)
            sums.index_add_(0, graph_ids_neg, NL)
            cnts.index_add_(0, graph_ids_neg, torch.ones_like(graph_ids_neg, dtype=torch.float32))
            means = sums / (cnts.clamp_min(1e-6).unsqueeze(1))
            Z_block = F.normalize(self.proj_head(means), dim=-1)
            Z_blocks.append(Z_block)

        # 简单起见，负样本权重统一为 1
        w_neg = torch.ones(Bc, dtype=torch.float32, device=device)
        return Z_blocks, w_neg

    def _build_neg_block_mimicry(self, Bc: int, device: torch.device,
                                 benign_x_list=None, benign_e_list=None,
                                 benign_node_counts=None):
        """Mimicry [ProvNinja] 风格的负样本构造：
        从恶意 ego 池采样子图，向其中注入良性边和良性节点特征，
        使恶意子图伪装成良性，作为难负样本。

        核心操作：
        1. 采样恶意子图
        2. 从当前 batch 的良性子图中随机选取节点
        3. 在恶意子图与良性节点之间添加跨连边
        4. 替换部分恶意节点的特征为良性特征（属性模糊）
        """
        pool = self._mal_ego_pool if hasattr(self, '_mal_ego_pool') else None
        if not pool:
            return [], torch.zeros(Bc, device=device)

        # 采样恶意子图
        if len(pool) >= Bc:
            chosen = random.sample(pool, k=Bc)
        else:
            chosen = [random.choice(pool) for _ in range(Bc)]

        x_list, e_list, node_counts = [], [], []
        for (sidx, center) in chosen:
            g = self.snapshots[sidx]
            if g is None or g.vcount() == 0:
                x_list.append(torch.zeros((0, self.prop_feat_dim), dtype=torch.float32, device=device))
                e_list.append(torch.zeros((2, 0), dtype=torch.long, device=device))
                node_counts.append(0)
                continue
            try:
                sub = self._ego_subgraph(g, center=center, r=self.r_hop, max_nodes=self.ego_max_nodes)
            except Exception:
                sub = None
            if sub is None or sub.vcount() == 0:
                x_list.append(torch.zeros((0, self.prop_feat_dim), dtype=torch.float32, device=device))
                e_list.append(torch.zeros((2, 0), dtype=torch.long, device=device))
                node_counts.append(0)
                continue
            x_np = self._build_node_features(sub)
            eidx, _ef = self._igraph_edges_to_edge_index(sub)
            x_list.append(torch.from_numpy(x_np).to(device))
            e_list.append(eidx.to(device))
            node_counts.append(sub.vcount())

        if sum(node_counts) == 0:
            return [], torch.zeros(Bc, device=device)

        # Mimicry 增强：向恶意子图注入良性信号
        has_benign = (benign_x_list is not None and len(benign_x_list) > 0
                      and benign_node_counts is not None and sum(benign_node_counts) > 0)

        if has_benign:
            # 拼接所有良性节点特征，作为特征替换的源
            benign_feats_all = torch.cat([bx for bx in benign_x_list if bx.numel() > 0], dim=0)

            for gi in range(len(x_list)):
                xi = x_list[gi]
                ei = e_list[gi]
                nc = node_counts[gi]
                if nc == 0 or xi.numel() == 0:
                    continue

                # (1) 特征替换：将 30% 的恶意节点特征替换为随机良性节点特征
                replace_ratio = 0.3
                n_replace = max(1, int(nc * replace_ratio))
                replace_idx = torch.randperm(nc, device=device)[:n_replace]
                benign_sample_idx = torch.randint(0, benign_feats_all.size(0), (n_replace,), device=device)
                xi_new = xi.clone()
                xi_new[replace_idx] = benign_feats_all[benign_sample_idx]
                x_list[gi] = xi_new

                # (2) 边注入：在恶意节点与"虚拟良性节点"之间添加边
                #     这里简化为在已有节点之间添加随机边（模拟良性交互模式）
                n_inject = max(1, int(ei.size(1) * 0.2))  # 注入 20% 的边
                src_new = torch.randint(0, nc, (n_inject,), device=device)
                dst_new = torch.randint(0, nc, (n_inject,), device=device)
                inject_edges = torch.stack([
                    torch.cat([src_new, dst_new]),
                    torch.cat([dst_new, src_new])
                ])  # 无向化
                e_list[gi] = torch.cat([ei, inject_edges], dim=1)

        # 编码：两个视角
        offsets_neg = np.cumsum([0] + node_counts[:-1]).tolist()
        graph_ids_neg = torch.tensor(
            [gi for gi, n in enumerate(node_counts) for _ in range(n)],
            device=device
        )
        X_neg = torch.cat([xi for xi in x_list if xi.numel() > 0], dim=0) if any(
            n > 0 for n in node_counts) else torch.zeros((0, self.prop_feat_dim), device=device)

        Z_blocks: List[torch.Tensor] = []
        for _ in range(2):
            if any(n > 0 for n in node_counts):
                e_cols = [self._augment_edges(ei, self.drop_edge_p) + off
                          for ei, off in zip(e_list, offsets_neg)]
            else:
                e_cols = []
            EN = torch.cat(e_cols, dim=1) if e_cols else torch.zeros((2, 0), dtype=torch.long, device=device)
            XN = self._augment_features(X_neg, self.feat_mask_p)
            ZN_layers = self.encoder(XN, EN, edge_feat=None, return_all=True)
            NL = ZN_layers[-1]
            sums = torch.zeros((Bc, NL.size(1)), device=device)
            cnts = torch.zeros(Bc, device=device)
            sums.index_add_(0, graph_ids_neg, NL)
            cnts.index_add_(0, graph_ids_neg, torch.ones_like(graph_ids_neg, dtype=torch.float32))
            means = sums / (cnts.clamp_min(1e-6).unsqueeze(1))
            Z_block = F.normalize(self.proj_head(means), dim=-1)
            Z_blocks.append(Z_block)

        w_neg = torch.ones(Bc, dtype=torch.float32, device=device)
        return Z_blocks, w_neg

    def _build_neg_block_from_snapshots_with_pos(
            self,
            Bc: int,
            device: torch.device,
            pos_x_list: List[torch.Tensor],
            pos_e_list: List[torch.Tensor],
            pos_node_counts: List[int],
            pos_ratio: float = 0.5,
            cross_edge_ratio: float = 0.2,
            cross_edge_max: int = 8,
    ):
        """
        基于当前 batch 的正子图 + 恶意子图缓存，构造若干“融合负子图”，再做两视角编码。
        返回: ([Z_neg_view1[N_neg,D], Z_neg_view2[N_neg,D]], w_neg[N_neg])
        注：不再强制构造 Bc 个，按可融合数量返回。
        """
        # ---- 0. 恶意缓存检查 ----
        mal_cache = getattr(self, "_mal_ego_cache", None)
        if not mal_cache:  # None 或空
            return [], torch.zeros(0, device=device)

        # 比例参数裁剪到 [0,1]
        pos_ratio = float(min(max(pos_ratio, 0.0), 1.0))
        cross_edge_ratio = float(min(max(cross_edge_ratio, 0.0), 1.0))
        cross_edge_max = int(max(0, cross_edge_max))

        x_list: List[torch.Tensor] = []
        e_list: List[torch.Tensor] = []
        node_counts: List[int] = []

        total_pos = len(pos_x_list)
        # 不按 Bc 约束：融合数量等于恶意缓存数
        num_build = len(mal_cache)
        if total_pos == 0:
            # 没有正子图：直接返回空（不做单纯恶意子图列表，保持融合语义）
            return [], torch.zeros(0, device=device)

        # ---- 1. 按恶意子图遍历：每个恶意子图随机挑一个正子图融合 ----
        for m_idx, cache_entry in enumerate(mal_cache):
            x_m_cached, e_m_cached, mal_cnt = cache_entry[0], cache_entry[1], cache_entry[2]
            ef_m_cached = cache_entry[3] if len(cache_entry) > 3 else None
            # 随机挑选一个正子图索引（不做循环复用顺序，增加随机性）
            pi = random.randrange(total_pos)
            xi = pos_x_list[pi]
            ei = pos_e_list[pi]
            nc = int(pos_node_counts[pi]) if pi < len(pos_node_counts) else (
                int(xi.size(0)) if isinstance(xi, torch.Tensor) else 0)

            use_pos = (
                    xi is not None and ei is not None and
                    isinstance(xi, torch.Tensor) and isinstance(ei, torch.Tensor) and
                    nc > 0 and xi.numel() > 0 and ei.numel() > 0
            )

            # ---- 1.1 正子图不可用：直接使用一个恶意子图占位 ----
            if not use_pos:
                # 正子图不可用：重新随机挑一个恶意子图填充（允许与当前不同）
                ridx = random.randrange(len(mal_cache))
                x_rand, e_rand, rand_cnt = mal_cache[ridx][0], mal_cache[ridx][1], mal_cache[ridx][2]
                x_list.append(x_rand.to(device))
                e_list.append(e_rand.to(device))
                node_counts.append(int(rand_cnt))
                continue

            # ---- 1.2 正子图可用：采样部分正节点 + 一个恶意子图，拼接融合 ----
            xi = xi.to(device)
            ei = ei.to(device)
            nc = int(nc)

            # ① 正节点子集
            k_pos = max(1, int(round(pos_ratio * nc)))
            k_pos = min(k_pos, nc)

            src_pos, dst_pos = ei[0], ei[1]
            mask_pos = (src_pos < k_pos) & (dst_pos < k_pos)
            ei_pos_sub = ei[:, mask_pos]
            xi_pos_sub = xi[:k_pos, :]

            # ② 使用当前遍历的恶意子图（不再重新随机抽取）
            x_m = x_m_cached.to(device)
            e_m = e_m_cached.to(device)
            mal_cnt = int(mal_cnt)

            # ③ 节点/边拼接
            x_fused = torch.cat([xi_pos_sub, x_m], dim=0)  # [k_pos + mal_cnt, F]
            e_m_shift = e_m + k_pos  # 恶意子图节点索引整体平移
            e_fused = torch.cat([ei_pos_sub, e_m_shift], dim=1)
            n_fused = k_pos + mal_cnt

            # ④ 随机跨连边（正节点 ↔ 恶意节点）
            if k_pos > 0 and mal_cnt > 0 and cross_edge_ratio > 0.0 and cross_edge_max > 0:
                base = min(k_pos, mal_cnt)
                target = int(round(cross_edge_ratio * base))
                num_cross = max(1, min(base, cross_edge_max, target))

                pos_idx = torch.randint(0, k_pos, (num_cross,), device=device)
                mal_idx = torch.randint(0, mal_cnt, (num_cross,), device=device) + k_pos

                cross_edges = torch.stack([pos_idx, mal_idx], dim=0)
                cross_edges_rev = torch.stack([mal_idx, pos_idx], dim=0)
                e_cross = torch.cat([cross_edges, cross_edges_rev], dim=1)

                e_fused = torch.cat([e_fused, e_cross], dim=1)

            x_list.append(x_fused)
            e_list.append(e_fused)
            node_counts.append(n_fused)

        # 全部为空的情况
        if sum(node_counts) == 0:
            return [], torch.zeros(0, device=device)

        # ---- 2. 打平成一个大图的节点/边，准备做两视角增强 + 编码 ----
        offsets_neg = np.cumsum([0] + node_counts[:-1]).tolist()
        graph_ids_neg = torch.tensor(
            [gi for gi, n in enumerate(node_counts) for _ in range(n)],
            device=device
        )

        if any(n > 0 for n in node_counts):
            X_neg = torch.cat([xi for xi in x_list if xi.numel() > 0], dim=0)
        else:
            X_neg = torch.zeros((0, self.prop_feat_dim), dtype=torch.float32, device=device)

        has_nodes = X_neg.numel() > 0

        # ---- 3. 做两视角增强 + 编码，得到两个 [N_neg, D] 的负样本块 ----
        Z_blocks: List[torch.Tensor] = []
        for _ in range(2):
            if has_nodes:
                if self.use_degree_coop_augment:
                    e_cols = [
                        self._augment_edges_degree_aware(ei, self.drop_edge_p) + off
                        for ei, off in zip(e_list, offsets_neg)
                    ]
                else:
                    e_cols = [
                        self._augment_edges(ei, self.drop_edge_p) + off
                        for ei, off in zip(e_list, offsets_neg)
                    ]
                EN = torch.cat(e_cols, dim=1) if e_cols else torch.zeros((2, 0), dtype=torch.long, device=device)

                if self.use_degree_coop_augment:
                    XN = self._augment_features_degree_aware(X_neg, self.feat_mask_p, EN)
                else:
                    XN = self._augment_features(X_neg, self.feat_mask_p)

                ZN_layers = self.encoder(XN, EN, edge_feat=None, return_all=True)
                NL = ZN_layers[-1]

                N_neg = len(node_counts)
                sums = torch.zeros((N_neg, NL.size(1)), device=device)
                cnts = torch.zeros(N_neg, device=device)
                sums.index_add_(0, graph_ids_neg, NL)
                cnts.index_add_(0, graph_ids_neg, torch.ones_like(graph_ids_neg, dtype=torch.float32))
                means = sums / (cnts.clamp_min(1e-6).unsqueeze(1))

                Z_blocks.append(F.normalize(self.proj_head(means), dim=-1))
            else:
                # 极端兜底：没有节点时给 0 向量
                Z_blocks.append(torch.zeros((0, self.enc_out_dim), dtype=torch.float32, device=device))

            # 目前对难负样本一律给权重 1（长度为 N_neg）
            N_neg = len(node_counts)
            w_neg = torch.ones(N_neg, dtype=torch.float32, device=device)
        return Z_blocks, w_neg

    def _build_neg_block_from_tokens(self, Bc: int, device: torch.device):
        """基于语料腐化的负样本块构建，返回两个视角的 Bc×D block 与权重。
        依赖 self._ego_cache（历史子图）与 self.malicious_node_tokens。
        若条件不足则返回空块与零权重。
        """
        if not (self.use_malicious_negatives
                and hasattr(self, 'malicious_node_tokens') and len(self.malicious_node_tokens) > 0
                and hasattr(self, '_ego_cache') and len(self._ego_cache) > 0):
            return [], torch.zeros(Bc, device=device)

        all_subs, all_x, all_e, all_w = [], [], [], []
        for subs_prev, x_prev, e_prev, w_prev in self._ego_cache:
            all_subs.extend(subs_prev)
            all_x.extend(x_prev)
            all_e.extend(e_prev)
            all_w.extend(w_prev)

        total_prev = len(all_subs)
        if total_prev == 0:
            return [], torch.zeros(Bc, device=device)

        # 若不足 Bc，允许有放回采样
        replace = total_prev < Bc
        idxs = np.random.choice(total_prev, size=Bc, replace=replace)

        X_neg_list, E_neg_list, node_counts_neg, freq_neg = [], [], [], []
        for i in idxs:
            sub, xi, ei, w = all_subs[i], all_x[i], all_e[i], all_w[i]
            xneg_np = self._corrupt_features_with_malicious(
                sub, xi.cpu().numpy(),
                ratio=float(self.mal_neg_ratio),
                node_token_len=int(self.mal_neg_node_token_len)
            )
            X_neg_list.append(torch.from_numpy(xneg_np).to(device))
            E_neg_list.append(ei.to(device) if ei.device != device else ei)
            node_counts_neg.append(sub.vcount())
            freq_neg.append(w)

        offsets_neg = np.cumsum([0] + node_counts_neg[:-1]).tolist()
        graph_ids_neg = torch.tensor(
            [gi for gi, n in enumerate(node_counts_neg) for _ in range(n)],
            device=device
        )
        X_neg = torch.cat(X_neg_list, dim=0)

        Z_neg_blocks: List[torch.Tensor] = []
        for _ in range(2):
            if self.use_degree_coop_augment:
                e_cols = [self._augment_edges_degree_aware(ei, self.drop_edge_p) + off for ei, off in
                          zip(E_neg_list, offsets_neg)]
            else:
                e_cols = [self._augment_edges(ei, self.drop_edge_p) + off for ei, off in zip(E_neg_list, offsets_neg)]
            EN = torch.cat(e_cols, dim=1)
            if self.use_degree_coop_augment:
                XN = self._augment_features_degree_aware(X_neg, self.feat_mask_p, EN)
            else:
                XN = self._augment_features(X_neg, self.feat_mask_p)
            ZN_layers = self.encoder(XN, EN, edge_feat=None, return_all=True)
            NL = ZN_layers[-1]
            sums = torch.zeros((Bc, NL.size(1)), device=device)
            cnts = torch.zeros(Bc, device=device)
            sums.index_add_(0, graph_ids_neg, NL)
            cnts.index_add_(0, graph_ids_neg, torch.ones_like(graph_ids_neg, dtype=torch.float32))
            means = sums / (cnts.clamp_min(1e-6).unsqueeze(1))
            Z_neg_blocks.append(F.normalize(self.proj_head(means), dim=-1))

        w_neg = torch.tensor(freq_neg, dtype=torch.float32, device=device)
        return Z_neg_blocks, w_neg

    def _corrupt_features_with_malicious(self, g, X_base: np.ndarray, ratio: float, node_token_len: int) -> np.ndarray:
        n = g.vcount()
        out = X_base.copy()
        if ratio <= 0 or not hasattr(self, "malicious_node_tokens") or len(self.malicious_node_tokens) == 0:
            return out
        for i in range(n):
            # 按比例随机替换部分节点
            if random.random() < ratio:
                # 从节点语料中抽取若干恶意token
                tokens = self._sample_malicious_tokens(max(1, int(node_token_len)))
                if not tokens:
                    continue
                # 转换为 embedding 向量（通过已有的 W2V 模型）
                vec = self._w2v_vector_from_tokens(tokens)
                out[i] = vec.astype(np.float32)

        return out

    def save_malicious_snapshot_stats(self, save_path: str = "malicious_tokens_log.txt"):
        """将全局恶意节点采样来源统计附加写入日志文件"""
        stats = getattr(self, "malicious_snapshot_stats", None)
        if not stats:
            print("[⚠️ 没有可保存的恶意节点采样统计]")
            return

        total = sum(stats.values())

        with open(save_path, "a", encoding="utf-8") as f:
            f.write("\n[📊 全局恶意节点采样统计]\n")
            for sid, cnt in sorted(stats.items()):
                pct = cnt / total * 100
                f.write(f"  Snapshot {sid:02d}: {cnt} 次 ({pct:.2f}%)\n")
            f.write(f"  总计: {total} 次采样\n")
            f.write("=" * 60 + "\n")

        print(f"[✅ 已将采样统计附加保存到]: {save_path}")

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

    # 已移除 WL 子树核与引导相似度损失相关方法

    def embed_nodes(self):
        return self.snapshot_node_embeddings[-1] if self.snapshot_node_embeddings else {}

    def embed_edges(self):
        return {}

    def prepare_text_encoder(self):
        """可选的显式预处理：提前训练/加载 Word2Vec 模型。
        某些离线流程（例如只做快照嵌入而不调用 train）可先调用本方法。
        """
        self._ensure_w2v_model()

    # 最正宗的
    def get_snapshot_embeddings(self, snapshot_sequence=None):
        if not self.snapshot_node_embeddings:
            raise RuntimeError("还没有节点嵌入，请先调用 train()")
        if snapshot_sequence is None:
            snapshot_sequence = list(range(len(self.snapshots)))

        result = []
        α = float(np.clip(self.attr_weight_alpha, 0.0, 1.0))

        for i in snapshot_sequence:
            g = self.snapshots[i]
            if g is None or g.vcount() == 0:
                result.append(np.zeros(self.enc_out_dim, dtype=np.float32))
                continue

            emb = self.snapshot_node_embeddings[i]
            if not emb:
                result.append(np.zeros(self.enc_out_dim, dtype=np.float32))
                continue

            N = g.vcount()

            # ===== 读取 node → vector =====
            vecs = np.zeros((N, self.enc_out_dim), dtype=np.float32)
            valid = np.zeros(N, dtype=bool)
            for j in range(N):
                nid = g.vs[j]['name']
                v = emb.get(nid)
                if v is not None:
                    vecs[j] = v
                    valid[j] = True

            if not valid.any():
                result.append(np.zeros(self.enc_out_dim, dtype=np.float32))
                continue

            # ===== base 权重（频率优先，否则度）=====
            if 'frequency' in g.vs.attributes():
                base_w = np.array(g.vs['frequency'], dtype=np.float32)
                base_w = np.maximum(base_w, 0)
            else:
                base_w = np.maximum(np.array(g.degree(), dtype=np.float32), 0)

            # 避免除 0
            b_norm = base_w / (base_w.mean() + 1e-12)

            # ===== 属性罕见性权重 1 - p(attr) =====
            # igraph 的 Vertex 无 .get 方法，需用 attributes()/序列访问
            if 'properties' in g.vs.attributes():
                props = [str(p) for p in g.vs['properties']]
            else:
                props = [''] * N

            # 聚合属性词的加权频率
            prop_w = {}
            for p, w in zip(props, base_w):
                if w > 0:
                    prop_w[p] = prop_w.get(p, 0.0) + w

            if prop_w:
                maxv = max(prop_w.values())
                prop_norm = {k: v / maxv for k, v in prop_w.items()}
            else:
                prop_norm = {}

            a = np.array([1.0 - prop_norm.get(props[j], 0.0) for j in range(N)], dtype=np.float32)
            a_norm = a / (a.mean() + 1e-12)

            # ===== 最终权重 w_eff =====
            w_eff = (1 - α) * b_norm + α * a_norm
            w_eff = np.maximum(w_eff, 0)

            if w_eff.sum() == 0:
                snapshot_vec = vecs[valid].mean(axis=0)
            else:
                snapshot_vec = (vecs * w_eff[:, None]).sum(axis=0) / (w_eff.sum() + 1e-12)

            result.append(snapshot_vec.astype(np.float32))

        arr = np.vstack(result) if result else np.zeros((0, self.enc_out_dim), dtype=np.float32)
        print(f"[GCC-Dev] Snapshot embeddings: {arr.shape}")
        return arr

    def compute_malicious_deviation_per_snapshot(
            self,
            snapshot_sequence: Optional[List[int]] = None,
            metric: str = 'cosine',
            center_weighting: str = 'none',
            save_path: str = "malicious_tokens_log.txt",
    ) -> List[Dict[str, object]]:
        """
        计算并保存：每个快照中恶意节点在全体节点“偏离降序排名”的百分比（rank_pct），
        并统计平均偏离、良性占比、最大/最小偏离节点等。
        """

        if not self.snapshot_node_embeddings:
            raise RuntimeError("还没有节点嵌入，请先调用 train()")

        if snapshot_sequence is None:
            snapshot_sequence = list(range(len(self.snapshots)))

        def _weights_for(g):
            if center_weighting == 'none':
                return None
            if center_weighting == 'degree':
                deg = np.asarray(g.degree(), dtype=np.float32)
                return np.maximum(deg, 0.0)
            # auto: 频率优先，退化度
            try:
                freqs = g.vs['frequency'] if 'frequency' in g.vs.attributes() else None
            except Exception:
                freqs = None
            if freqs is not None:
                w = np.zeros(g.vcount(), dtype=np.float32)
                for idx in range(g.vcount()):
                    try:
                        v = float(freqs[idx])
                    except Exception:
                        v = 0.0
                    if np.isfinite(v) and v > 0:
                        w[idx] = v
                if w.sum() > 0:
                    return w
            deg = np.asarray(g.degree(), dtype=np.float32)
            return np.maximum(deg, 0.0)

        rows: List[Dict[str, object]] = []

        with open(save_path, "a", encoding="utf-8") as f:
            f.write("\n[恶意节点偏离排名百分比统计]\n")
            f.write("=" * 70 + "\n")

            for i in snapshot_sequence:
                g = self.snapshots[i]
                if g is None or g.vcount() == 0:
                    continue

                emb_dict = self.snapshot_node_embeddings[i] if i < len(self.snapshot_node_embeddings) else {}
                if not emb_dict:
                    continue

                names, vecs, labels = [], [], []
                for local_idx in range(g.vcount()):
                    nid = g.vs[local_idx]['name']
                    vec = emb_dict.get(nid)
                    if vec is None:
                        continue
                    names.append(nid)
                    vecs.append(np.asarray(vec, dtype=np.float32))
                    try:
                        lab = int(g.vs[local_idx].attributes().get('label', 0))
                    except Exception:
                        lab = 0
                    labels.append(lab)

                if not vecs:
                    continue

                V = np.vstack(vecs).astype(np.float32)
                W = _weights_for(g)
                if W is not None and len(names) == g.vcount() and W.sum() > 0:
                    center = (V * W[:, None]).sum(axis=0) / (W.sum() + 1e-12)
                else:
                    center = V.mean(axis=0)

                # 偏离值
                if metric == 'l2':
                    devs = np.linalg.norm(V - center, axis=1)
                else:
                    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
                    cn = center / (np.linalg.norm(center) + 1e-12)
                    devs = 1.0 - np.matmul(Vn, cn)

                devs1d = np.asarray(devs).reshape(-1)
                N = int(devs1d.shape[0])

                # 降序排名（偏离大→排前）
                order = np.argsort(-devs1d)  # indices
                rank_map = {int(idx): int(r + 1) for r, idx in enumerate(order)}  # 1..N

                # 最大/最小偏离节点
                max_idx = int(np.argmax(devs1d))
                min_idx = int(np.argmin(devs1d))
                max_name, max_val = names[max_idx], float(devs1d[max_idx])
                min_name, min_val = names[min_idx], float(devs1d[min_idx])

                # 节点集合
                mal_idx = [k for k, lab in enumerate(labels) if lab == 1]
                ben_idx = [k for k, lab in enumerate(labels) if lab == 0]
                if not mal_idx:
                    continue

                num_mal = len(mal_idx)
                num_benign = len(ben_idx)
                benign_ratio = num_benign / N if N > 0 else 0.0

                # 快照整体均偏离
                mean_dev_all = float(np.mean(devs1d))

                # 恶意节点排名百分比
                mal_rank_entries = []
                rank_pcts = []
                for idx in mal_idx:
                    rk = rank_map[idx]  # 1..N
                    rk_pct = rk / N * 100.0
                    rank_pcts.append(rk_pct)
                    mal_rank_entries.append((names[idx], rk, rk_pct, float(devs1d[idx])))

                # 平均排名百分比（恶意）
                mean_mal_rank_pct = float(np.mean(rank_pcts)) if rank_pcts else 0.0

                # 控制台打印
                print(f"\n[Snapshot {i:02d}]")
                print(f"  平均偏离: {mean_dev_all:.6f}")
                print(f"  良性节点数: {num_benign} ({benign_ratio:.2%})")
                print(f"  最大偏离节点: {max_name} ({max_val:.6f})")
                print(f"  最小偏离节点: {min_name} ({min_val:.6f})")
                print(f"  恶意节点 平均排名百分比: {mean_mal_rank_pct:.2f}%")
                print("  恶意节点偏离排名：")
                # 按 rank 升序展示（更直观）
                mal_rank_entries.sort(key=lambda x: x[1])
                for name, rk, rk_pct, dev in mal_rank_entries:
                    print(f"    - {name}: rank={rk}, rank_pct={rk_pct:.2f}%, dev={dev:.6f}")

                # 写入日志
                f.write(f"Snapshot {i:02d}: 平均偏离={mean_dev_all:.6f}\n")
                f.write(f"  良性节点数={num_benign} ({benign_ratio:.2%})\n")
                f.write(f"  最大偏离节点: {max_name} ({max_val:.6f})\n")
                f.write(f"  最小偏离节点: {min_name} ({min_val:.6f})\n")
                f.write(f"  恶意节点 平均排名百分比: {mean_mal_rank_pct:.2f}%\n")
                for name, rk, rk_pct, dev in mal_rank_entries:
                    f.write(f"    - {name}: rank={rk}, rank_pct={rk_pct:.2f}%, dev={dev:.6f}\n")
                f.write("-" * 70 + "\n")

                rows.append({
                    'snapshot': i,
                    'num_nodes': N,
                    'num_mal': num_mal,
                    'num_benign': num_benign,
                    'benign_ratio': benign_ratio,
                    'mean_dev_all': mean_dev_all,
                    'max_dev_node': max_name,
                    'max_dev_val': max_val,
                    'min_dev_node': min_name,
                    'min_dev_val': min_val,
                    'mean_mal_rank_pct': mean_mal_rank_pct,
                    # 每个恶意节点：(name, rank, rank_pct, deviation)
                    'mal_rank_table': mal_rank_entries,
                })

            f.write("=" * 70 + "\n")

        print(f"[GCC-Dev] 恶意节点偏离排名统计已保存到: {save_path}")
        return rows

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
                # W2V 配置
                'w2v_window': self.w2v_window,
                'w2v_min_count': self.w2v_min_count,
                'w2v_sg': self.w2v_sg,
                'w2v_epochs': self.w2v_epochs,
                'w2v_pretrained_path': self.w2v_pretrained_path,
                # 恶意Token配置
                'mal_stopwords': list(self.mal_stopwords) if self.mal_stopwords else [],
                'mal_print_tokens': self.mal_print_tokens,
                # 增强策略
                'use_degree_coop_augment': self.use_degree_coop_augment,
                # 属性频率降权参数
                'attr_weight_alpha': self.attr_weight_alpha,
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
            'prop_feat_dim', 'enc_hidden_dim', 'enc_out_dim', 'gin_layers', 'dropout',
            'num_epochs', 'batch_size', 'lr', 'temperature',
            'r_hop', 'ego_max_nodes', 'drop_edge_p', 'feat_mask_p', 'train_indices', 'model_path',
            'anomaly_alpha', 'use_sample_weights',
            # W2V 配置
            'w2v_window', 'w2v_min_count', 'w2v_sg', 'w2v_epochs', 'w2v_pretrained_path',
            # 恶意Token配置
            'mal_stopwords', 'mal_print_tokens',
            # 增强策略
            'use_degree_coop_augment',
        }
        params = {k: v for k, v in raw_params.items() if k in allowed}
        inst = cls(snapshot_sequence, **params)
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
        """改进版 token 提取（保留路径、事件名、UUID、数字等结构，不去重）"""
        if not text:
            return []

        s = str(text).strip()
        # 使用正则提取连续的 [A-Za-z0-9_-.:/\] 段，保留路径、UUID、数字
        tokens = re.findall(r"[A-Za-z0-9_\-./:\\]+", s)

        # 直接返回，不去重
        return tokens

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
                    print(
                        f"[GCC-Dev] 预训练向量维度({vec_dim}) != prop_feat_dim({self.prop_feat_dim})，将改为自训练以匹配维度。")
                    self._w2v_model = None
            except Exception as e:
                print(f"[GCC-Dev] 加载预训练 Word2Vec 失败：{e}，将尝试自训练。")
        # 自训练
        corpus = self._collect_w2v_corpus()
        if not corpus:
            raise RuntimeError("[GCC-Dev] W2V 语料为空，无法构建 Word2Vec 特征。")
        print(
            f"[GCC-Dev] 正在训练word2vec | 语料={len(corpus)} | dim={int(self.prop_feat_dim)} | window={int(self.w2v_window)} | min_count={int(self.w2v_min_count)} | sg={int(self.w2v_sg)} | epochs={int(self.w2v_epochs)}")
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
            return np.ones((n, 1), dtype=np.float32)
        if self._w2v_model is None:
            self._ensure_w2v_model()
        X = np.zeros((n, int(self.prop_feat_dim)), dtype=np.float32)
        for i in range(n):
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

    def _igraph_edges_to_edge_index(self, g):
        """返回 (edge_index [2, E*2], edge_cat [E*2] 整数类别索引)"""
        edges = g.get_edgelist()
        if len(edges) == 0:
            return (torch.zeros((2, 0), dtype=torch.long, device=self.device),
                    torch.zeros(0, dtype=torch.long, device=self.device))
        src, dst, cats = [], [], []
        has_actions = 'actions' in g.es.attributes() if g.ecount() > 0 else False
        for i, (u, v) in enumerate(edges):
            action_str = str(g.es[i].attributes().get('actions', '')) if has_actions else ''
            cat = classify_edge(action_str)
            src.append(u); dst.append(v); cats.append(cat)
            src.append(v); dst.append(u); cats.append(cat)
        return (torch.tensor([src, dst], dtype=torch.long, device=self.device),
                torch.tensor(cats, dtype=torch.long, device=self.device))

    def _augment_edges(self, edge_index: torch.Tensor, drop_p: float) -> torch.Tensor:
        """原始（均匀随机）删边增强：保持不变"""
        if edge_index.numel() == 0 or drop_p <= 0:
            return edge_index
        E = edge_index.size(1)
        keep = torch.rand(E, device=edge_index.device) > drop_p
        if keep.sum() < 1:
            keep[random.randrange(0, E)] = True
        return edge_index[:, keep]

    def _augment_features(self, x: torch.Tensor, mask_p: float) -> torch.Tensor:
        """原始（均匀随机）特征掩盖：保持不变"""
        if x.numel() == 0 or mask_p <= 0:
            return x
        mask = (torch.rand_like(x) < mask_p).float()
        return x * (1.0 - mask)

    # ---- 度感知的“点-边协同增强”实现 ----
    def _augment_edges_degree_aware(self, edge_index: torch.Tensor, drop_p: float) -> torch.Tensor:
        if edge_index.numel() == 0 or drop_p <= 0:
            return edge_index
        device = edge_index.device
        src, dst = edge_index[0], edge_index[1]
        if src.numel() == 0:
            return edge_index
        num_nodes = int(torch.max(torch.stack([src, dst])).item() + 1)
        deg = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        deg.index_add_(0, src, torch.ones_like(src, dtype=torch.float32))
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        dmin = torch.min(deg)
        dmax = torch.max(deg)
        if float(dmax.item() - dmin.item()) < 1e-12:
            deg_norm = torch.zeros_like(deg)
        else:
            deg_norm = (deg - dmin) / (dmax - dmin + 1e-12)
        s_e = 0.5 * (deg_norm[src] + deg_norm[dst])
        p_e = torch.clamp(drop_p * s_e, 0.0, 1.0)
        keep = (torch.rand_like(p_e) > p_e)
        if keep.sum() < 1:
            keep[random.randrange(0, keep.numel())] = True
        return edge_index[:, keep]

    def _augment_features_degree_aware(self, x: torch.Tensor, mask_p: float, edge_index: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0 or mask_p <= 0 or edge_index.numel() == 0:
            return x
        device = x.device
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        num_nodes = max(N, int(torch.max(torch.stack([src, dst])).item() + 1))
        deg = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        deg.index_add_(0, src, torch.ones_like(src, dtype=torch.float32))
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        deg = deg[:N]
        dmin = torch.min(deg)
        dmax = torch.max(deg)
        if float(dmax.item() - dmin.item()) < 1e-12:
            deg_norm = torch.zeros_like(deg)
        else:
            deg_norm = (deg - dmin) / (dmax - dmin + 1e-12)
        p_node = torch.clamp(mask_p * (1.0 - deg_norm), 0.0, 1.0)
        rand = torch.rand_like(x)
        mask = (rand < p_node.view(-1, 1)).float()
        return x * (1.0 - mask)

    def _weighted_contrastive_loss(
        self,
        Z_pos: torch.Tensor,
        Z_neg: Optional[torch.Tensor],
        temperature: float,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """
        论文公式 (5)(6): 有监督加权对比损失。

        Z_pos: [2*Bp, D] 良性样本嵌入（两视角交错: v1_0, v2_0, v1_1, v2_1, ...）
        Z_neg: [2*Bn, D] 恶意样本嵌入（两视角交错），可为 None
        temperature: 温度参数 τ
        beta: 聚焦强度 β，控制难负样本权重的集中程度

        正样本集 P(b): 同一锚点的另一视角 + 其他良性样本的所有视角
        负样本集 N(b): 所有恶意样本的所有视角
        权重 w_n = softmax(β * sim(z_b, z_n) / τ) 聚焦难负样本
        """
        Z_pos = F.normalize(Z_pos, dim=-1)
        Np = Z_pos.size(0)  # 2*Bp

        if Z_neg is not None and Z_neg.size(0) > 0:
            Z_neg = F.normalize(Z_neg, dim=-1)
            Nn = Z_neg.size(0)  # 2*Bn
        else:
            Nn = 0

        # 正样本间相似度 [Np, Np]
        sim_pp = torch.mm(Z_pos, Z_pos.t()) / temperature
        # 去除自身
        mask_self = torch.eye(Np, device=Z_pos.device).bool()
        sim_pp = sim_pp.masked_fill(mask_self, -1e9)

        if Nn > 0:
            # 正-负相似度 [Np, Nn]
            sim_pn = torch.mm(Z_pos, Z_neg.t()) / temperature

            # 公式 (5): 基于相似度的负样本权重 w_n = softmax(β * s_bn)
            w_neg = F.softmax(beta * sim_pn, dim=-1)  # [Np, Nn]

            # 加权负样本 log-sum-exp: |N(b)| * Σ w_n * exp(s_bn)
            weighted_neg = (Nn * w_neg * torch.exp(sim_pn)).sum(dim=-1)  # [Np]
        else:
            weighted_neg = torch.zeros(Np, device=Z_pos.device)

        # 公式 (6): 对每个锚点 b，遍历其正样本集
        # P(b) = 所有其他良性样本（排除自身）
        # L = -1/|P(b)| * Σ_{j∈P(b)} log [ exp(s_bj) / (Σ_{j'∈P(b)} exp(s_bj') + |N(b)|·Σ w_n·exp(s_bn)) ]
        exp_pp = torch.exp(sim_pp)  # [Np, Np], 对角已被 mask 为 ~0
        denom = exp_pp.sum(dim=-1) + weighted_neg  # [Np]

        # 每个锚点对所有正样本取平均
        log_prob = sim_pp - torch.log(denom.unsqueeze(1).clamp_min(1e-9))  # [Np, Np]
        # 只对非自身的正样本求和
        pos_mask = (~mask_self).float()
        n_pos = pos_mask.sum(dim=-1).clamp_min(1.0)  # [Np]
        loss = -(pos_mask * log_prob).sum(dim=-1) / n_pos  # [Np]

        return loss.mean()

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
                eidx, ef = self._igraph_edges_to_edge_index(g)
                x = torch.from_numpy(x_np).to(self.device)
                curr_ids = [g.vs[i]['name'] for i in range(g.vcount())]
                if use_temporal:
                    H_prev = self.temporal.fetch(curr_ids, device=self.device)
                    Z_list = self.encoder(x, eidx, edge_feat=ef, return_all=True)
                    H_list = self.temporal(Z_list, H_prev)
                    self.temporal.commit(curr_ids, [h.detach() for h in H_list])
                    h_last = H_list[-1]
                else:
                    h_last = self.encoder(x, eidx, edge_feat=ef)
                emb_dict: Dict[str, np.ndarray] = {}
                for i in range(g.vcount()):
                    nid = g.vs[i]['name']
                    emb_dict[nid] = h_last[i].detach().cpu().numpy().astype(np.float32)
                self.snapshot_node_embeddings.append(emb_dict)
        mode = 'temporal' if use_temporal else 'static'
        print(f"[GCC-Dev] Generated {mode} node embeddings: {len(self.snapshot_node_embeddings)} snapshots")
