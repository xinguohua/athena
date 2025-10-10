"""
ROLAND Embedder (GNNCL 风格)
改为“邻居预测自己”训练范式：
- 节点初始向量：nn.Embedding(num_nodes, embedding_dim)
- 轻量 GraphSAGE 卷积（自实现，避免外部依赖）
- MLP 产出节点表示（output_dim = hidden_conv_2）
- f_phi 预测头：用邻居均值去回归中心节点的 (h, tag)

目标：用结构邻域信息约束表示学习，去除原 DGI/对比学习逻辑。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Iterable, Union, List

from process.embedders.base import GraphEmbedderBase


class SimpleSAGEConv(nn.Module):
    """轻量 GraphSAGE 卷积：h' = ReLU(W_self·h + W_nei·mean_neigh(h))"""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.lin_self = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_nei = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        src, dst = edge_index[0], edge_index[1]
        N = x.size(0)
        deg = torch.bincount(dst, minlength=N).float()
        nei_sum = torch.zeros_like(x)
        if src.numel() > 0:
            nei_sum.index_add_(0, dst, x[src])
        nei_mean = nei_sum / torch.clamp(deg, min=1.0).unsqueeze(-1)
        out = self.lin_self(x) + self.lin_nei(nei_mean)
        return F.relu(out)


class GNNCL(nn.Module):
    """节点嵌入 + SAGE + MLP + f_phi 预测头"""

    def __init__(self, num_nodes: int, embedding_dim: int, hidden_dim: int, output_dim: int, num_mlp_layers: int = 2):
        super().__init__()
        self.node_embedding = nn.Embedding(num_embeddings=num_nodes, embedding_dim=embedding_dim)
        self.sage_conv = SimpleSAGEConv(embedding_dim, hidden_dim)
        mlp_layers: List[nn.Module] = []
        for _ in range(num_mlp_layers):
            mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*mlp_layers)
        # 预测头：从邻居均值预测 (h, tag[3])
        self.f_phi = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim + 3),
        )

    def forward(self, node_indices: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x0 = self.node_embedding(node_indices)  # (N, embedding_dim)
        x1 = self.sage_conv(x0, edge_index)    # (N, hidden_dim)
        h = self.mlp(x1)                       # (N, output_dim)
        return h


class ROLANDGraphEmbedder(GraphEmbedderBase):
    """ROLAND 图嵌入器（包含训练逻辑）"""
    
    _default_path = 'roland_encoder.pth'

    def __init__(
        self,
        snapshots,
        features=None,
        mapp=None,
        embedding_dim=256,
        hidden_conv_1=128,
        hidden_conv_2=256,
        num_epochs=50,
        lr=0.0005,
        tau: float = 0.2,
        edge_drop_rate: float = 0.2,
        feat_drop_rate: float = 0.2,
        neg_edge_drop_rate: float = 0.3,
        neg_edge_add_ratio: float = 0.1,
        neigh_pred_weight: float = 1.0,
        variance_weight: float = 0.1,
        train_indices: Optional[Union[Iterable[int], Tuple[int, int], int]] = None,
        model_path=None
    ):
        """
        Args:
            snapshots: list of igraph.Graph
            embedding_dim: 输入特征维度 (默认256)
            hidden_conv_1: 第1层GCN输出维度 (默认128)
            hidden_conv_2: 第2层GCN输出维度 (默认256，最终嵌入维度，匹配分类器)
            num_epochs: 训练轮数
            lr: 学习率
            tau: InfoNCE 温度参数
            edge_drop_rate: 视图增强的随机删边比例
            feat_drop_rate: 视图增强的随机特征掩码比例
            train_indices: 可选，仅训练指定索引的快照（支持 range、列表或(start, end)元组）
        """
        super().__init__(snapshots, features, mapp)
        self.snapshots = self.G
        self.embedding_dim = embedding_dim
        self.hidden_conv_1 = hidden_conv_1
        self.hidden_conv_2 = hidden_conv_2
        self.num_epochs = num_epochs
        self.lr = lr
        self.tau = float(tau)
        self.edge_drop_rate = float(edge_drop_rate)
        self.feat_drop_rate = float(feat_drop_rate)
        self.neg_edge_drop_rate = float(neg_edge_drop_rate)
        self.neg_edge_add_ratio = float(neg_edge_add_ratio)
        self.neigh_pred_weight = float(neigh_pred_weight)
        self.variance_weight = float(variance_weight)
        self.model_path = model_path or self._default_path

        # 训练快照筛选（默认全部）
        self.train_snapshot_indices = self._resolve_train_indices(train_indices)
        self._train_snapshot_index_set = set(self.train_snapshot_indices)

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 获取节点数量
        all_nodes_set = set()
        for g in snapshots:
            if g is not None:
                for v in range(g.vcount()):
                    all_nodes_set.add(g.vs[v]['name'])
        self.all_nodes = sorted(all_nodes_set)
        self.num_nodes = len(self.all_nodes)
        self.node_id_map = {nid: i for i, nid in enumerate(self.all_nodes)}
        
        # 初始化模型
        # 模型（GNNCL 风格）与优化器
        self.model = GNNCL(
            num_nodes=self.num_nodes,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_conv_1,
            output_dim=self.hidden_conv_2,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        # 存储最终嵌入
        self.snapshot_embeddings_list = []

    def _resolve_train_indices(
        self,
        indices: Optional[Union[Iterable[int], Tuple[int, int], int]]
    ) -> list[int]:
        total = len(self.snapshots)
        if total == 0:
            return []

        if indices is None:
            raw = list(range(total))
        elif isinstance(indices, int):
            raw = [indices]
        elif isinstance(indices, tuple) and len(indices) == 2:
            start, end = indices
            start_int, end_int = int(start), int(end)
            if start_int > end_int:
                start_int, end_int = end_int, start_int
            raw = list(range(start_int, end_int + 1))
        else:
            try:
                raw = list(indices)  # type: ignore[arg-type]
            except TypeError as exc:
                raise TypeError("train_indices 必须是可迭代的索引或(start, end)元组") from exc

        valid = sorted({int(idx) for idx in raw if 0 <= int(idx) < total})
        if not valid:
            raise ValueError("train_indices 不包含有效的快照索引")
        return valid

    def train(self):
        """无监督训练：每个 snapshot 反复训练多轮"""
        if not self.train_snapshot_indices:
            raise RuntimeError("没有可用于训练的快照。请检查 train_indices 设置。")

        print(
            f"[ROLAND] Training on {len(self.train_snapshot_indices)}/{len(self.snapshots)} snapshots, {self.num_nodes} nodes"
        )
        print(
            f"[ROLAND] Objective: NeighborPredictOnly (GNNCL) | Epochs/snapshot: {self.num_epochs}, LR: {self.lr}, WD: 1e-4, "
            f"neigh_w: {self.neigh_pred_weight}, var_w: {self.variance_weight}"
        )
        
        # 外层按 epoch 训练，内层遍历所有参与训练的快照
        overall_loss = 0.0
        epoch_count = 0
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            trained_cnt = 0
            for sidx, g in enumerate(self.snapshots):
                if sidx not in self._train_snapshot_index_set:
                    continue
                if g is None:
                    continue

                # 转换为本地张量：节点全局ID (N_local,) 与本地边索引 (2, E)
                node_ids, edge_index = self._igraph_to_torch(g)

                # 前向：得到本快照节点的表示 h_i (N_local, d)
                z = self.model(node_ids, edge_index)

                # 基于本地边计算邻居平均向量
                src, dst = edge_index[0], edge_index[1]
                N_local = z.size(0)
                deg = torch.bincount(dst, minlength=N_local).float()
                neigh_sum = torch.zeros_like(z)
                if src.numel() > 0:
                    neigh_sum.index_add_(0, dst, z[src])
                neigh_mean = neigh_sum / torch.clamp(deg, min=1.0).unsqueeze(-1)

                # 仅对有邻居的节点计算损失
                mask = deg > 0
                if mask.any():
                    pred = self.model.f_phi(neigh_mean[mask])
                    pred_h = pred[:, : self.hidden_conv_2]
                    # stop-gradient on target to避免双边同时塌缩
                    target_h = z[mask].detach()
                    neigh_loss = F.mse_loss(pred_h, target_h)
                    # variance regularization (VICReg 风格，仅抑制塌缩)
                    eps = 1e-4
                    std = z[mask].std(dim=0) + eps
                    var_loss = torch.mean(F.relu(1.0 - std))
                else:
                    neigh_loss = torch.tensor(0.0, device=self.device)
                    var_loss = torch.tensor(0.0, device=self.device)

                total_loss = self.neigh_pred_weight * neigh_loss + self.variance_weight * var_loss

                # 反向与更新
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.model.parameters()), max_norm=5.0)
                self.optimizer.step()

                # 日志（每个快照一行）
                with torch.no_grad():
                    tl = total_loss.item()
                    nl = neigh_loss.item() if torch.is_tensor(neigh_loss) else float(neigh_loss)
                    vl = var_loss.item() if torch.is_tensor(var_loss) else float(var_loss)
                    print(f"    [Epoch {epoch} | Snap {sidx}] total={tl:.4f} neigh={nl:.4f} var={vl:.4f} N={int(N_local)}")

                epoch_loss += total_loss.item()
                trained_cnt += 1

            avg_epoch_loss = epoch_loss / trained_cnt if trained_cnt > 0 else 0.0
            overall_loss += avg_epoch_loss
            epoch_count += 1
            print(f"[ROLAND] Epoch {epoch+1}/{self.num_epochs} Avg Loss: {avg_epoch_loss:.6f} on {trained_cnt} snapshots")

        avg_total_loss = overall_loss / epoch_count if epoch_count > 0 else 0.0
        print(f"[ROLAND] Training completed! Overall Avg Loss: {avg_total_loss:.6f}")
        
    # 生成最终嵌入（推理模式）：逐快照按当前模型重算并缓存
        self._generate_final_embeddings()
        
        # 自动保存模型
        self.save_model()

    def _present_node_indices(self, g) -> torch.Tensor:
        """返回当前快照在全局节点表中的索引（tensor）。"""
        node_gids = [g.vs[vid]['name'] for vid in range(g.vcount())]
        idxs = [self.node_id_map[nid] for nid in node_gids if nid in self.node_id_map]
        if not idxs:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        return torch.tensor(idxs, dtype=torch.long, device=self.device)

    def _generate_final_embeddings(self):
        """训练后生成最终嵌入（用于后续任务）"""
        self.model.eval()
        with torch.no_grad():
            for sidx, g in enumerate(self.snapshots):
                if g is None:
                    self.snapshot_embeddings_list.append({})
                    continue
                
                node_ids, edge_index = self._igraph_to_torch(g)
                final_emb = self.model(node_ids, edge_index)
                # 提取嵌入字典
                embeddings_dict = self._extract_final_embeddings(g, final_emb)
                self.snapshot_embeddings_list.append(embeddings_dict)
    
    def _igraph_to_torch(self, g) -> Tuple[torch.Tensor, torch.Tensor]:
        """将 igraph 转为 (节点全局ID, 本地边索引) 以适配 GNNCL 前向。

        Returns:
            node_ids: (N_local,) 当前快照节点对应的全局ID（顺序与 g.vs 对齐）
            edge_index: (2, E) 使用本地索引(0..N_local-1) 的 COO 边
        """
        node_gids = [g.vs[vid]['name'] for vid in range(g.vcount())]
        node_ids = torch.tensor([self.node_id_map[nid] for nid in node_gids], dtype=torch.long, device=self.device)
        edges = g.get_edgelist()  # 本地索引
        if len(edges) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
        else:
            src = [u for (u, v) in edges]
            dst = [v for (u, v) in edges]
            edge_index = torch.tensor([src, dst], dtype=torch.long, device=self.device)
        return node_ids, edge_index
    
    def _extract_final_embeddings(self, g, final_emb: torch.Tensor) -> Dict[str, np.ndarray]:
        """提取节点嵌入字典（使用本地索引对齐 final_emb 行）。"""
        node_gids = [g.vs[vid]['name'] for vid in range(g.vcount())]
        embeddings_dict: Dict[str, np.ndarray] = {}
        for local_idx, nid in enumerate(node_gids):
            if local_idx < final_emb.size(0):
                embeddings_dict[nid] = final_emb[local_idx].detach().cpu().numpy()
        return embeddings_dict

    def get_snapshot_embeddings(self, snapshot_sequence=None):
        """返回快照级别的嵌入矩阵（使用已缓存的节点嵌入做聚合）。

        训练结束后，_generate_final_embeddings 已为每个快照缓存了节点嵌入字典。
        这里直接对缓存的节点嵌入做“度加权聚合 + L2 归一化”，避免重复前向计算，
        同时保留结构判别性，效率更高。
        """
        if not self.snapshot_embeddings_list:
            raise RuntimeError("还没有快照嵌入，请先调用 train()")

        if snapshot_sequence is None:
            snapshot_sequence = list(range(len(self.snapshot_embeddings_list)))

        result = []
        eps = 1e-12
        for i in snapshot_sequence:
            emb_dict = self.snapshot_embeddings_list[i] if i < len(self.snapshot_embeddings_list) else None
            g = self.snapshots[i] if i < len(self.snapshots) else None

            if not emb_dict or g is None:
                result.append(np.zeros(self.hidden_conv_2, dtype=np.float32))
                continue

            node_gids = [g.vs[vid]['name'] for vid in range(g.vcount())]
            degrees = g.degree()  # 与 node_gids 对齐

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
                # 退化：无边，回退为简单均值
                all_embs = np.array(list(emb_dict.values()), dtype=np.float32)
                snapshot_vec = all_embs.mean(axis=0) if all_embs.size > 0 else np.zeros(self.hidden_conv_2, dtype=np.float32)
            else:
                snapshot_vec = weighted_sum / (total_w + eps)

            # L2 归一化
            norm = np.linalg.norm(snapshot_vec) + eps
            snapshot_vec = (snapshot_vec / norm).astype(np.float32)
            result.append(snapshot_vec)

        arr = np.vstack(result).astype(np.float32) if result else np.zeros((0, self.hidden_conv_2), dtype=np.float32)
        print(f"[ROLAND] Snapshot embeddings: {arr.shape}")
        return arr

    def embed_nodes(self):
        """返回最后一个快照的节点嵌入字典"""
        if not self.snapshot_embeddings_list:
            raise RuntimeError("还没有节点嵌入，请先调用 train()")
        
        return self.snapshot_embeddings_list[-1] if self.snapshot_embeddings_list else {}

    def embed_edges(self):
        """边嵌入（暂未实现）"""
        return {}

    def save_model(self, path=None):
        """保存模型状态"""
        path = path or self.model_path
        state = {
            'params': {
                'embedding_dim': self.embedding_dim,
                'hidden_conv_1': self.hidden_conv_1,
                'hidden_conv_2': self.hidden_conv_2,
                'num_epochs': self.num_epochs,
                'lr': self.lr,
                'train_indices': self.train_snapshot_indices,
                'objective': 'neighbor_predict',
                'tau': self.tau,
                'edge_drop_rate': self.edge_drop_rate,
                'feat_drop_rate': self.feat_drop_rate,
                'neg_edge_drop_rate': self.neg_edge_drop_rate,
                'neg_edge_add_ratio': self.neg_edge_add_ratio,
                'neigh_pred_weight': self.neigh_pred_weight,
                'variance_weight': self.variance_weight,
            },
            'model_state': self.model.state_dict(),
            'snapshot_embeddings': self.snapshot_embeddings_list,
            'all_nodes': self.all_nodes,  # 保存节点列表（load时重建node_id_map）
        }
        torch.save(state, path)
        print(f"[ROLAND] Model saved to {path}")

    @classmethod
    def load(cls, snapshot_sequence, path=None):
        """加载预训练模型"""
        path = path or cls._default_path
        
        print(f"[ROLAND] Loading model from {path}...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state = torch.load(path, map_location=device)
        
        # 创建实例
        raw_params = dict(state.get('params', {}))
        # 过滤掉 __init__ 不接受的键，避免 TypeError（例如 'objective'）
        allowed_keys = {
            'embedding_dim', 'hidden_conv_1', 'hidden_conv_2',
            'num_epochs', 'lr', 'tau', 'edge_drop_rate', 'feat_drop_rate',
            'neg_edge_drop_rate', 'neg_edge_add_ratio',
            'neigh_pred_weight', 'variance_weight',
            'train_indices', 'model_path'
        }
        params = {k: v for k, v in raw_params.items() if k in allowed_keys}
        instance = cls(snapshot_sequence, **params)
        
        # 恢复模型权重和嵌入
        instance.model.load_state_dict(state['model_state'])
        instance.snapshot_embeddings_list = state['snapshot_embeddings']
        instance.all_nodes = state['all_nodes']
        
        # 重建 node_id_map
        instance.node_id_map = {nid: i for i, nid in enumerate(instance.all_nodes)}
        instance.num_nodes = len(instance.all_nodes)

        print("[ROLAND] Model loaded successfully")
        return instance
