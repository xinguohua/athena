"""
ROLAND (Graph Learning Framework for Dynamic Graphs) Embedder
基于 snap-stanford/roland 仓库和论文实现的动态图嵌入器

核心思想：
1. EdgeBank: 缓存历史边信息用于快速检索
2. 时序 GNN: residual edge convolution 聚合邻居特征
3. 节点状态更新: moving_average / GRU 更新节点 embedding
4. 快照级别嵌入: 聚合快照内所有节点状态得到图级表示
"""
import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from process.embedders.base import GraphEmbedderBase


class EdgeBank:
    """缓存历史边信息，用于快速邻居查询"""
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.edge_cache = defaultdict(set)  # node -> set of (neighbor, edge_type, timestamp)

    def add_edges(self, edges, types, timestamps):
        """批量添加边到缓存"""
        for (u, v), etype, ts in zip(edges, types, timestamps):
            self.edge_cache[u].add((v, etype, ts))
            self.edge_cache[v].add((u, f"rev_{etype}", ts))

    def get_neighbors(self, node_id):
        """获取节点的所有历史邻居"""
        return list(self.edge_cache.get(node_id, []))

    def get_degree(self, node_id):
        """获取节点度数"""
        return len(self.edge_cache.get(node_id, []))


class ResidualEdgeConv(nn.Module):
    """残差边卷积层（ROLAND 核心 GNN 层）"""
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 16):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 消息传递网络
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

        # 残差连接（如果维度不匹配需要投影）
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, node_features, edge_index, edge_features=None):
        """
        node_features: (N, in_dim)
        edge_index: (2, E)
        edge_features: (E, edge_dim) 可选
        """
        device = node_features.device
        src, dst = edge_index[0], edge_index[1]

        # 构造消息：[src_feat, dst_feat, edge_feat]
        if edge_features is None:
            edge_features = torch.zeros(edge_index.shape[1], 16, device=device)

        msg_input = torch.cat([
            node_features[src],
            node_features[dst],
            edge_features
        ], dim=-1)

        # 计算消息
        messages = self.msg_mlp(msg_input)

        # 聚合消息（scatter add）
        aggr = torch.zeros(node_features.size(0), self.out_dim, device=device)
        aggr.index_add_(0, dst, messages)

        # 残差连接
        out = aggr + self.residual(node_features)
        return F.relu(out)


class ROLANDGraphEmbedder(GraphEmbedderBase):
    """ROLAND 风格的动态图嵌入器"""
    
    # 类级别的默认模型保存路径
    _default_path = 'roland_encoder.pth'

    def __init__(self, snapshots, features=None, mapp=None,
                 embedding_dim=256, num_layers=2,
                 update_method='moving_average', alpha=0.5,
                 model_path=None):
        """
        Args:
            snapshots: list of igraph.Graph
            embedding_dim: 节点嵌入维度 (默认256,与Prographer分类器匹配)
            num_layers: GNN 层数
            update_method: 'moving_average' 或 'gru'
            alpha: moving_average 的更新率
            model_path: 模型保存路径，默认使用 _default_path
        """
        super().__init__(snapshots, features, mapp)
        self.snapshots = self.G
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.update_method = update_method
        self.alpha = alpha
        self.model_path = model_path or self._default_path

        # 初始化组件
        self.edge_bank = None
        self.node_states = {}  # node_id -> embedding vector
        self.gnn_layers = []
        self.snapshot_embeddings_list = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 边类型编码器（简单映射）
        self.edge_type_vocab = {}

        # 构建 GNN 模型
        self._build_model()

    def _build_model(self):
        """构建多层 GNN"""
        dims = [self.embedding_dim] * (self.num_layers + 1)
        for i in range(self.num_layers):
            layer = ResidualEdgeConv(dims[i], dims[i+1], edge_dim=16)
            self.gnn_layers.append(layer.to(self.device))

        # GRU 更新器（如果使用 GRU 模式）
        if self.update_method == 'gru':
            self.gru_cell = nn.GRUCell(self.embedding_dim, self.embedding_dim).to(self.device)

    def _init_node_state(self, node_id):
        """初始化节点状态（随机小值）"""
        if node_id not in self.node_states:
            self.node_states[node_id] = np.random.randn(self.embedding_dim) * 0.01

    def _update_node_state(self, node_id, new_embedding):
        """更新节点状态（moving average 或 GRU）"""
        if self.update_method == 'moving_average':
            old = self.node_states.get(node_id, np.zeros(self.embedding_dim))
            self.node_states[node_id] = (1 - self.alpha) * old + self.alpha * new_embedding
        elif self.update_method == 'gru':
            old = torch.FloatTensor(self.node_states.get(node_id, np.zeros(self.embedding_dim))).to(self.device)
            new = torch.FloatTensor(new_embedding).to(self.device)
            updated = self.gru_cell(new.unsqueeze(0), old.unsqueeze(0)).squeeze(0)
            self.node_states[node_id] = updated.detach().cpu().numpy()
        else:
            self.node_states[node_id] = new_embedding

    def train(self):
        """训练：逐快照处理，更新 EdgeBank 和节点状态"""
        if not hasattr(self, "snapshots") or not self.snapshots:
            raise RuntimeError("self.snapshots 为空")

        # 推断节点总数
        all_nodes = set()
        for g in self.snapshots:
            if g is not None:
                for v in range(g.vcount()):
                    all_nodes.add(g.vs[v]['name'])
        num_nodes = len(all_nodes)

        # 初始化 EdgeBank
        self.edge_bank = EdgeBank(num_nodes)
        node_id_map = {nid: i for i, nid in enumerate(sorted(all_nodes))}

        print(f"[ROLAND] Training on {len(self.snapshots)} snapshots, {num_nodes} nodes")

        for sidx, g in enumerate(self.snapshots):
            if g is None:
                continue

            t0 = time.time()

            # 提取边和特征
            edges = g.get_edgelist()
            
            # 处理空快照（没有边的情况）
            if len(edges) == 0:
                print(f"[snapshot {sidx}] 跳过空快照（无边）")
                # 即使没有新边，也要保存当前状态作为快照嵌入
                if self.node_states:
                    current_states = np.array([self.node_states.get(nid, np.zeros(self.embedding_dim)) 
                                              for nid in sorted(all_nodes)], dtype=np.float32)
                    snapshot_emb = current_states.mean(axis=0)
                else:
                    snapshot_emb = np.zeros(self.embedding_dim, dtype=np.float32)
                self.snapshot_embeddings_list.append(snapshot_emb)
                continue
                
            types = g.es["actions"]
            # 假设有 timestamp 属性，若无则用 sidx
            try:
                timestamps = g.es["timestamp"]
            except (KeyError, AttributeError):
                timestamps = [sidx] * len(edges)

            # 映射到全局节点 ID
            node_gids = [g.vs[vid]['name'] for vid in range(g.vcount())]
            for nid in node_gids:
                self._init_node_state(nid)

            # 构建边类型词汇表（首次出现时添加）
            for etype in types:
                if etype not in self.edge_type_vocab:
                    self.edge_type_vocab[etype] = len(self.edge_type_vocab)

            # 更新 EdgeBank
            global_edges = [(node_gids[u], node_gids[v]) for u, v in edges]
            self.edge_bank.add_edges(global_edges, types, timestamps)

            # 构造当前快照的 PyTorch 数据 (edge_index: shape (2, E))
            src_nodes = [node_id_map[node_gids[u]] for u, v in edges]
            dst_nodes = [node_id_map[node_gids[v]] for u, v in edges]
            edge_index = torch.LongTensor([src_nodes, dst_nodes]).to(self.device)

            # 节点特征：使用当前状态 (先转 numpy array 再转 tensor,避免警告)
            node_features_np = np.array([
                self.node_states.get(nid, np.zeros(self.embedding_dim))
                for nid in sorted(all_nodes)
            ], dtype=np.float32)
            node_features = torch.from_numpy(node_features_np).to(self.device)

            # 边特征：编码边类型（one-hot 或 embedding）
            edge_type_indices = [self.edge_type_vocab.get(types[i], 0) for i in range(len(edges))]
            edge_features = torch.zeros(len(edges), 16, device=self.device)
            for i, type_idx in enumerate(edge_type_indices):
                if type_idx < 16:  # 简单映射到前16维
                    edge_features[i, type_idx] = 1.0

            # 多层 GNN 传播
            x = node_features
            for layer in self.gnn_layers:
                x = layer(x, edge_index, edge_features)

            # 更新活跃节点的状态 (CUDA tensor 需要先 .cpu() 再 .numpy())
            active_node_ids = set(node_gids)
            for nid in active_node_ids:
                idx = node_id_map[nid]
                self._update_node_state(nid, x[idx].detach().cpu().numpy())

            # 快照级嵌入：平均池化所有节点状态（或只用活跃节点）
            snapshot_emb = x.mean(dim=0).detach().cpu().numpy()
            self.snapshot_embeddings_list.append(snapshot_emb)

            t_elapsed = time.time() - t0
            print(f"[snapshot {sidx}] processed {len(edges)} edges, {len(active_node_ids)} nodes, {t_elapsed:.3f}s")

        # 训练完成后自动保存模型
        self.save_model(self.model_path)

    def get_snapshot_embeddings(self, snapshot_sequence=None):
        """返回快照嵌入矩阵"""
        if not self.snapshot_embeddings_list:
            raise RuntimeError("还没有快照嵌入，请先调用 train()")

        if snapshot_sequence is None:
            snapshot_sequence = list(range(len(self.snapshot_embeddings_list)))

        embeddings = [self.snapshot_embeddings_list[i] for i in snapshot_sequence]
        arr = np.array(embeddings, dtype=np.float32)
        print(f"[ROLAND] Snapshot embeddings: {arr.shape}")
        return arr

    def embed_nodes(self):
        """返回节点嵌入字典"""
        return {nid: emb.copy() for nid, emb in self.node_states.items()}

    def embed_edges(self):
        """边嵌入（暂未实现，可扩展）"""
        return {}

    def save_model(self, path=None):
        """保存模型状态"""
        path = path or self._default_path
        state = {
            'params': {
                'embedding_dim': self.embedding_dim,
                'num_layers': self.num_layers,
                'update_method': self.update_method,
                'alpha': self.alpha,
            },
            'gnn_layers_state': [layer.state_dict() for layer in self.gnn_layers],
            'gru_cell_state': self.gru_cell.state_dict() if self.update_method == 'gru' else None,
            'node_states': self.node_states,
            'edge_type_vocab': self.edge_type_vocab,
            'snapshot_embeddings': self.snapshot_embeddings_list,
            'num_snapshots': len(self.snapshot_embeddings_list),
        }
        torch.save(state, path)
        print(f"[ROLAND] Encoder model saved to {path}")

    @classmethod
    def load(cls, snapshot_sequence, path=None):
        """加载预训练模型"""
        path = path or cls._default_path
        
        print(f"[ROLAND] Loading encoder model from {path}...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state = torch.load(path, map_location=device)

        # 创建实例
        instance = cls(snapshot_sequence, **state['params'])

        # 恢复 GNN 层状态
        for i, layer_state in enumerate(state['gnn_layers_state']):
            instance.gnn_layers[i].load_state_dict(layer_state)
            instance.gnn_layers[i].to(device)
            instance.gnn_layers[i].eval()

        # 恢复 GRU 状态
        if state['gru_cell_state'] is not None:
            instance.gru_cell.load_state_dict(state['gru_cell_state'])
            instance.gru_cell.to(device)
            instance.gru_cell.eval()

        # 恢复其他状态
        instance.node_states = state['node_states']
        instance.edge_type_vocab = state['edge_type_vocab']
        instance.snapshot_embeddings_list = state['snapshot_embeddings']

        print(f"[ROLAND] Encoder model loaded successfully (Original snapshots: {state['num_snapshots']}, Current snapshots: {len(snapshot_sequence)})")
        return instance
