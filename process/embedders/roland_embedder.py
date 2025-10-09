"""
ROLAND (Graph Learning Framework for Dynamic Graphs) Embedder
基于 snap-stanford/roland 的无监督时序图嵌入器

架构：
1. Pre-processing: 2层MLP (input_dim → 256 → 128)
2. GCN Layers: 简单图卷积 (128→64→32)
3. Temporal Update: 每层独立GRU/MLP/moving_average更新
4. 训练: 无监督结构重建 MSE(sigmoid(h@h.T), A_true)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from process.embedders.base import GraphEmbedderBase


class ROLANDUnsupervised(nn.Module):
    """ROLAND 无监督时序 GNN 模型（基于结构重建）"""
    
    def __init__(
        self, 
        input_dim: int,
        num_nodes: int,
        hidden_conv_1: int = 64,
        hidden_conv_2: int = 32,
        temporal_type: str = "gru"  # gru/mlp/moving_average
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.hidden_conv_1 = hidden_conv_1
        self.hidden_conv_2 = hidden_conv_2
        self.temporal_type = temporal_type
        
        # Pre-processing MLP (256 → 128)
        self.pre1 = nn.Linear(input_dim, 256)
        self.pre2 = nn.Linear(256, 128)
        
        # GCN Layers (简单邻接矩阵乘法)
        self.conv1 = nn.Linear(128, hidden_conv_1)
        self.conv2 = nn.Linear(hidden_conv_1, hidden_conv_2)
        
        # Temporal Update (每层独立)
        if temporal_type == "gru":
            self.gru1 = nn.GRUCell(hidden_conv_1, hidden_conv_1)
            self.gru2 = nn.GRUCell(hidden_conv_2, hidden_conv_2)
        elif temporal_type == "mlp":
            self.mlp1 = nn.Sequential(
                nn.Linear(hidden_conv_1 * 2, hidden_conv_1),
                nn.LeakyReLU()
            )
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_conv_2 * 2, hidden_conv_2),
                nn.LeakyReLU()
            )
        # moving_average: alpha * h_new + (1-alpha) * h_old
        
        # Previous embeddings (两层独立状态)
        self.register_buffer('prev_emb_1', torch.zeros(num_nodes, hidden_conv_1))
        self.register_buffer('prev_emb_2', torch.zeros(num_nodes, hidden_conv_2))
    
    def reset_embeddings(self):
        """重置历史状态（每个epoch开始时调用，防止未来信息泄漏）"""
        self.prev_emb_1.zero_()
        self.prev_emb_2.zero_()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Args:
            x: 节点特征 (N, input_dim)
            edge_index: COO格式边索引 (2, E)
        
        Returns:
            final_emb: 最终节点嵌入 (N, hidden_conv_2)
            new_embeddings: 两层的新状态 [layer1_emb, layer2_emb]
        """
        # Pre-processing MLP
        h = F.leaky_relu(self.pre1(x))
        h = F.leaky_relu(self.pre2(h))  # (N, 128)
        
        # GCN Layer 1: 简单邻接矩阵乘法
        h = self._gcn_forward(h, edge_index, self.conv1)  # (N, 64)
        
        # Temporal Update Layer 1
        h = self._temporal_update(h, self.prev_emb_1, layer=1)
        new_emb_1 = h.clone()
        
        # GCN Layer 2
        h = self._gcn_forward(h, edge_index, self.conv2)  # (N, 32)
        
        # Temporal Update Layer 2
        h = self._temporal_update(h, self.prev_emb_2, layer=2)
        new_emb_2 = h.clone()
        
        return h, [new_emb_1, new_emb_2]
    
    def _gcn_forward(self, h: torch.Tensor, edge_index: torch.Tensor, linear: nn.Linear):
        """简单 GCN: A @ H @ W (无自环/归一化版本)"""
        # 聚合邻居特征: sum_{j in N(i)} h_j
        src, dst = edge_index[0], edge_index[1]
        aggr = torch.zeros_like(h)
        aggr.index_add_(0, dst, h[src])
        
        # 线性变换
        out = linear(aggr)
        out = F.leaky_relu(out)
        
        return out
    
    def _temporal_update(self, h_new: torch.Tensor, h_old: torch.Tensor, layer: int):
        """时序状态更新（GRU/MLP/moving_average）"""
        if self.temporal_type == "gru":
            gru = self.gru1 if layer == 1 else self.gru2
            return gru(h_new, h_old)
        
        elif self.temporal_type == "mlp":
            mlp = self.mlp1 if layer == 1 else self.mlp2
            combined = torch.cat([h_new, h_old], dim=-1)
            return mlp(combined)
        
        else:  # moving_average
            alpha = 0.7  # 新状态权重
            return alpha * h_new + (1 - alpha) * h_old


class ROLANDGraphEmbedder(GraphEmbedderBase):
    """ROLAND 图嵌入器（包含训练逻辑）"""
    
    _default_path = 'roland_encoder.pth'

    def __init__(
        self, 
        snapshots, 
        features=None, 
        mapp=None,
        embedding_dim=256,
        hidden_conv_1=64,
        hidden_conv_2=32,
        temporal_type="gru",
        num_epochs=10,
        lr=0.001,
        model_path=None
    ):
        """
        Args:
            snapshots: list of igraph.Graph
            embedding_dim: 输入特征维度 (默认256)
            hidden_conv_1: 第1层GCN输出维度 (默认64)
            hidden_conv_2: 第2层GCN输出维度 (默认32，最终嵌入维度)
            temporal_type: 时序更新类型 (gru/mlp/moving_average)
            num_epochs: 训练轮数
            lr: 学习率
        """
        super().__init__(snapshots, features, mapp)
        self.snapshots = self.G
        self.embedding_dim = embedding_dim
        self.hidden_conv_1 = hidden_conv_1
        self.hidden_conv_2 = hidden_conv_2
        self.temporal_type = temporal_type
        self.num_epochs = num_epochs
        self.lr = lr
        self.model_path = model_path or self._default_path

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
        
        print(f"[ROLAND Init] Total nodes: {self.num_nodes:,}")
        print(f"[ROLAND Init] Total snapshots: {len(snapshots)}")
        
        # 统计边数
        total_edges = sum(g.ecount() if g else 0 for g in snapshots)
        avg_edges = total_edges / len([g for g in snapshots if g]) if snapshots else 0
        print(f"[ROLAND Init] Total edges: {total_edges:,}, Avg per snapshot: {avg_edges:.0f}")
        
        # 初始化模型
        self.model = ROLANDUnsupervised(
            input_dim=embedding_dim,
            num_nodes=self.num_nodes,
            hidden_conv_1=hidden_conv_1,
            hidden_conv_2=hidden_conv_2,
            temporal_type=temporal_type
        ).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # 存储最终嵌入
        self.snapshot_embeddings_list = []

    def train(self):
        """无监督训练：边重建损失（避免 O(N²) 显存）"""
        print(f"[ROLAND] Training on {len(self.snapshots)} snapshots, {self.num_nodes} nodes")
        print(f"[ROLAND] Epochs: {self.num_epochs}, Learning Rate: {self.lr}")
        
        for epoch in range(self.num_epochs):
            # 每个epoch重置嵌入（防止未来信息泄漏）
            self.model.reset_embeddings()
            
            epoch_loss = 0.0
            for sidx, g in enumerate(self.snapshots):
                if g is None:
                    continue
                
                # 转换 igraph 到 PyTorch
                x, edge_index = self._igraph_to_torch(g)
                
                # Forward
                node_emb, new_embeddings = self.model(x, edge_index)
                
                # 边重建损失（只计算边上的点积，避免完整邻接矩阵）
                # 正边：实际存在的边 (label=1)
                src, dst = edge_index[0], edge_index[1]
                pos_scores = torch.sigmoid((node_emb[src] * node_emb[dst]).sum(dim=-1))
                pos_loss = -torch.log(pos_scores + 1e-8).mean()
                
                # 负采样：随机采样不存在的边 (label=0)
                num_neg = min(edge_index.size(1), 1000)  # 限制负样本数量
                neg_src = torch.randint(0, self.num_nodes, (num_neg,), device=self.device)
                neg_dst = torch.randint(0, self.num_nodes, (num_neg,), device=self.device)
                neg_scores = torch.sigmoid((node_emb[neg_src] * node_emb[neg_dst]).sum(dim=-1))
                neg_loss = -torch.log(1 - neg_scores + 1e-8).mean()
                
                loss = pos_loss + neg_loss
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 更新历史状态（detach）
                self.model.prev_emb_1 = new_embeddings[0].detach()
                self.model.prev_emb_2 = new_embeddings[1].detach()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(self.snapshots)
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{self.num_epochs}, Avg Loss: {avg_loss:.6f}")
        
        print("[ROLAND] Training completed!")
        
        # 生成最终嵌入（推理模式）
        self._generate_final_embeddings()
    
    def _generate_final_embeddings(self):
        """训练后生成最终嵌入（用于后续任务）"""
        self.model.eval()
        self.model.reset_embeddings()
        
        with torch.no_grad():
            for sidx, g in enumerate(self.snapshots):
                if g is None:
                    self.snapshot_embeddings_list.append({})
                    continue
                
                x, edge_index = self._igraph_to_torch(g)
                final_emb, new_embeddings = self.model(x, edge_index)
                
                # 更新状态
                self.model.prev_emb_1 = new_embeddings[0]
                self.model.prev_emb_2 = new_embeddings[1]
                
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
        """返回快照嵌入矩阵（快照级别聚合）"""
        if not self.snapshot_embeddings_list:
            raise RuntimeError("还没有快照嵌入，请先调用 train()")

        if snapshot_sequence is None:
            snapshot_sequence = list(range(len(self.snapshot_embeddings_list)))

        # 直接返回字典形式的嵌入
        result = []
        for i in snapshot_sequence:
            embeddings = self.snapshot_embeddings_list[i]
            if embeddings:
                # 聚合所有节点嵌入：平均值
                all_embs = np.array(list(embeddings.values()))
                snapshot_emb = np.mean(all_embs, axis=0)
            else:
                snapshot_emb = np.zeros(self.hidden_conv_2)
            result.append(snapshot_emb)
        
        arr = np.array(result, dtype=np.float32)
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
                'temporal_type': self.temporal_type,
                'num_epochs': self.num_epochs,
                'lr': self.lr,
            },
            'model_state': self.model.state_dict(),
            'snapshot_embeddings': self.snapshot_embeddings_list,
            'all_nodes': self.all_nodes,
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
        instance = cls(snapshot_sequence, **state['params'])
        instance.model.load_state_dict(state['model_state'])
        instance.snapshot_embeddings_list = state['snapshot_embeddings']
        instance.all_nodes = state['all_nodes']

        print("[ROLAND] Model loaded successfully")
        return instance
