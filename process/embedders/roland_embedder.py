"""
ROLAND Contrastive Embedder
静态图两层GCN + 对比学习（BPR）损失，用于生成快照嵌入。

流程：
1. 预处理 MLP：input_dim → 256 → 128
2. 两层图卷积（邻居求和 + 线性变换）
3. 对比损失：正边 vs 负边的 BPR 排序损失
4. 每个快照独立训练多个 epoch
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Iterable, Union

from process.embedders.base import GraphEmbedderBase


class ContrastiveGCNEncoder(nn.Module):
    """两层简单图卷积编码器（无时间依赖）"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.pre1 = nn.Linear(input_dim, 256)
        self.pre2 = nn.Linear(256, 128)
        self.conv1 = nn.Linear(128, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """返回最终节点表示"""
        h = F.leaky_relu(self.pre1(x))
        h = F.leaky_relu(self.pre2(h))
        h = self._gcn_forward(h, edge_index, self.conv1)
        h = self.dropout(h)
        h = self._gcn_forward(h, edge_index, self.conv2)
        return F.normalize(h, p=2, dim=-1)

    def _gcn_forward(self, h: torch.Tensor, edge_index: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        aggr = torch.zeros_like(h)
        aggr.index_add_(0, dst, h[src])
        out = linear(aggr)
        return F.leaky_relu(out)


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
        num_epochs=10,
        lr=0.001,
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
            train_indices: 可选，仅训练指定索引的快照（支持 range、列表或(start, end)元组）
        """
        super().__init__(snapshots, features, mapp)
        self.snapshots = self.G
        self.embedding_dim = embedding_dim
        self.hidden_conv_1 = hidden_conv_1
        self.hidden_conv_2 = hidden_conv_2
        self.num_epochs = num_epochs
        self.lr = lr
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
        self.model = ContrastiveGCNEncoder(
            input_dim=embedding_dim,
            hidden_dim=hidden_conv_1,
            output_dim=hidden_conv_2
        ).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
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
        print(f"[ROLAND] Epochs per snapshot: {self.num_epochs}, Learning Rate: {self.lr}")
        
        total_loss = 0.0
        num_trained = 0
        
        for sidx, g in enumerate(self.snapshots):
            if sidx not in self._train_snapshot_index_set:
                continue
            if g is None:
                continue
            
            # 转换 igraph 到 PyTorch
            x, edge_index = self._igraph_to_torch(g)
            
            # 每个 snapshot 训练多个 epoch
            snapshot_loss = 0.0
            for epoch in range(self.num_epochs):
                # Forward
                node_emb = self.model(x, edge_index)
                
                # BPR 对比学习损失（边预测）
                # 正边：实际存在的边
                src, dst = edge_index[0], edge_index[1]
                pos_scores = (node_emb[src] * node_emb[dst]).sum(dim=-1)  # (E,)
                
                # 负采样：为每条正边采样一条负边
                num_edges = edge_index.size(1)
                neg_dst = torch.randint(0, self.num_nodes, (num_edges,), device=self.device)
                neg_scores = (node_emb[src] * node_emb[neg_dst]).sum(dim=-1)  # (E,)
                
                # BPR Loss: -log(sigmoid(pos - neg))
                # 目标：让正边得分高于负边得分
                loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                snapshot_loss += loss.item()
            
            # 更新历史状态（训练完当前 snapshot 后）
            avg_snapshot_loss = snapshot_loss / self.num_epochs
            total_loss += avg_snapshot_loss
            num_trained += 1
            
            if (sidx + 1) % 10 == 0:  # 每 10 个 snapshot 打印一次
                print(f"  Snapshot {sidx+1}/{len(self.snapshots)}, Avg Loss: {avg_snapshot_loss:.6f}")
        
        avg_total_loss = total_loss / num_trained if num_trained > 0 else 0
        print(f"[ROLAND] Training completed! Overall Avg Loss: {avg_total_loss:.6f}")
        
        # 生成最终嵌入（推理模式）
        self._generate_final_embeddings()
        
        # 自动保存模型
        self.save_model()
    
    def _generate_final_embeddings(self):
        """训练后生成最终嵌入（用于后续任务）"""
        self.model.eval()
        with torch.no_grad():
            for sidx, g in enumerate(self.snapshots):
                if g is None:
                    self.snapshot_embeddings_list.append({})
                    continue
                
                x, edge_index = self._igraph_to_torch(g)
                final_emb = self.model(x, edge_index)
                # 提取嵌入字典
                embeddings_dict = self._extract_final_embeddings(g, final_emb)
                self.snapshot_embeddings_list.append(embeddings_dict)
    
    def _igraph_to_torch(self, g) -> Tuple[torch.Tensor, torch.Tensor]:
        """将 igraph 转换为 PyTorch 张量
        
        Returns:
            x: 节点特征 (num_nodes, embedding_dim)
            edge_index: COO边索引 (2, num_edges)
        """
        # 节点特征：使用 properties 哈希
        node_gids = [g.vs[vid]['name'] for vid in range(g.vcount())]
        node_features_list = []
        
        for nid in self.all_nodes:
            feat = np.zeros(self.embedding_dim, dtype=np.float32)
            
            if nid in node_gids:
                local_idx = node_gids.index(nid)
                try:
                    properties_str = g.vs[local_idx]['properties']
                except (KeyError, AttributeError):
                    properties_str = ''
                
                if properties_str:
                    import hashlib
                    prop_hash = int(hashlib.md5(properties_str.encode()).hexdigest()[:16], 16)
                    for i in range(min(64, self.embedding_dim)):
                        feat[i] = float((prop_hash >> i) & 1)
            
            node_features_list.append(feat)
        
        x = torch.from_numpy(np.array(node_features_list)).to(self.device)
        
        # 边索引
        edges = g.get_edgelist()
        src = [self.node_id_map[node_gids[u]] for u, v in edges]
        dst = [self.node_id_map[node_gids[v]] for u, v in edges]
        edge_index = torch.LongTensor([src, dst]).to(self.device)
        
        return x, edge_index
    
    def _extract_final_embeddings(self, g, final_emb: torch.Tensor) -> Dict[str, np.ndarray]:
        """提取节点嵌入字典"""
        node_gids = [g.vs[vid]['name'] for vid in range(g.vcount())]
        embeddings_dict = {}
        
        for nid in node_gids:
            idx = self.node_id_map[nid]
            embeddings_dict[nid] = final_emb[idx].cpu().numpy()
        
        return embeddings_dict

    def get_snapshot_embeddings(self, snapshot_sequence=None):
        """返回快照级别的嵌入矩阵（评估时逐快照前向计算）。

        对每个快照：
        1) 使用当前训练好的编码器对该快照前向得到节点嵌入；
        2) 使用度加权对节点嵌入做图级聚合；
        3) 对聚合后的图向量做 L2 归一化；
        因此不同快照的图向量由其自身结构与节点表征决定，不会“都一样”。
        """
        if snapshot_sequence is None:
            snapshot_sequence = list(range(len(self.snapshots)))

        self.model.eval()
        result = []
        eps = 1e-12
        with torch.no_grad():
            for i in snapshot_sequence:
                g = self.snapshots[i] if i < len(self.snapshots) else None
                if g is None:
                    result.append(np.zeros(self.hidden_conv_2, dtype=np.float32))
                    continue

                # 前向：节点嵌入
                x, edge_index = self._igraph_to_torch(g)
                node_emb = self.model(x, edge_index).cpu().numpy()  # (N_all, D)

                # 仅聚合当前快照包含的节点（用度作为权重）
                node_gids = [g.vs[vid]['name'] for vid in range(g.vcount())]
                degrees = g.degree()

                weighted_sum = np.zeros(self.hidden_conv_2, dtype=np.float32)
                total_w = 0.0
                for local_idx, nid in enumerate(node_gids):
                    idx = self.node_id_map.get(nid, None)
                    if idx is None:
                        continue
                    w = float(degrees[local_idx])
                    if w <= 0:
                        continue
                    weighted_sum += (w * node_emb[idx].astype(np.float32))
                    total_w += w

                if total_w <= 0:
                    # 退化：无边，回退为简单均值
                    idxs = [self.node_id_map[nid] for nid in node_gids if nid in self.node_id_map]
                    if idxs:
                        snapshot_vec = node_emb[idxs].mean(axis=0).astype(np.float32)
                    else:
                        snapshot_vec = np.zeros(self.hidden_conv_2, dtype=np.float32)
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
        params = dict(state.get('params', {}))
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
