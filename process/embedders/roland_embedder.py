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
        
        # 添加 LayerNorm 提高数值稳定性
        self.layer_norm = nn.LayerNorm(out_dim)

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
        
        # LayerNorm + ReLU
        out = self.layer_norm(out)
        out = F.relu(out)
        
        # 温和的梯度裁剪,保留更多信息
        out = torch.clamp(out, -50.0, 50.0)
        
        return out


class ROLANDGraphEmbedder(GraphEmbedderBase):
    """ROLAND 风格的动态图嵌入器"""
    
    # 类级别的默认模型保存路径
    _default_path = 'roland_encoder.pth'

    def __init__(self, snapshots, features=None, mapp=None,
                 embedding_dim=256, num_layers=2,
                 update_method='moving_average', alpha=0.7,
                 model_path=None):
        """
        Args:
            snapshots: list of igraph.Graph
            embedding_dim: 节点嵌入维度 (默认256,与Prographer分类器匹配)
            num_layers: GNN 层数
            update_method: 'moving_average' 或 'gru'
            alpha: moving_average 的更新率 (默认0.7,更重视新信息)
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
        self.node_type_vocab = {}  # 添加节点类型词汇表

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
        """初始化节点状态（Xavier初始化）"""
        if node_id not in self.node_states:
            # 使用 Xavier/Glorot 初始化,更适合深度网络
            limit = np.sqrt(6.0 / self.embedding_dim)
            self.node_states[node_id] = np.random.uniform(-limit, limit, self.embedding_dim).astype(np.float32)

    def _get_keep_ratio(self, node_id, existing_degree, new_degree, mode='linear'):
        """
        计算节点的 keep_ratio (ROLAND 核心创新)
        state[t] = state[t-1] * keep_ratio + new_emb * (1 - keep_ratio)
        
        Args:
            node_id: 节点ID
            existing_degree: 历史累积度数
            new_degree: 当前快照中的度数
            mode: 'linear', 'constant', 'sqrt'
        """
        if mode == 'constant':
            # 固定权重 (等价于原始 moving_average)
            if existing_degree == 0 and new_degree > 0:
                return 0.0  # 新节点完全使用新特征
            return 1 - self.alpha
        
        elif mode == 'linear':
            # 根据度数比例动态调整 (ROLAND 官方推荐)
            if existing_degree == 0 and new_degree > 0:
                return 0.0  # 新节点
            total = existing_degree + new_degree
            if total == 0:
                return 1.0  # 孤立节点保持不变
            return existing_degree / total  # 历史越多,保留越多
        
        elif mode == 'sqrt':
            # 平方根衰减 (平滑版本)
            if existing_degree == 0:
                return 0.0
            return np.sqrt(existing_degree) / (np.sqrt(existing_degree) + np.sqrt(new_degree + 1))
        
        else:
            return 1 - self.alpha
    
    def _update_node_state(self, node_id, new_embedding, existing_degree=0, new_degree=0):
        """更新节点状态（moving average 或 GRU，支持动态 keep_ratio）"""
        old = self.node_states.get(node_id, np.zeros(self.embedding_dim))
        
        if self.update_method == 'moving_average':
            # ROLAND 官方方式: 动态 keep_ratio
            keep_ratio = self._get_keep_ratio(node_id, existing_degree, new_degree, mode='linear')
            self.node_states[node_id] = keep_ratio * old + (1 - keep_ratio) * new_embedding
        
        elif self.update_method == 'gru':
            old_tensor = torch.FloatTensor(old).to(self.device)
            new_tensor = torch.FloatTensor(new_embedding).to(self.device)
            updated = self.gru_cell(new_tensor.unsqueeze(0), old_tensor.unsqueeze(0)).squeeze(0)
            self.node_states[node_id] = updated.detach().cpu().numpy()
        
        elif self.update_method == 'masked_gru':
            # 只更新活跃节点 (new_degree > 0)
            if new_degree > 0:
                old_tensor = torch.FloatTensor(old).to(self.device)
                new_tensor = torch.FloatTensor(new_embedding).to(self.device)
                updated = self.gru_cell(new_tensor.unsqueeze(0), old_tensor.unsqueeze(0)).squeeze(0)
                self.node_states[node_id] = updated.detach().cpu().numpy()
            # 否则保持原状态不变
        
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

        # 初始化 EdgeBank 和度数跟踪
        self.edge_bank = EdgeBank(num_nodes)
        node_id_map = {nid: i for i, nid in enumerate(sorted(all_nodes))}
        node_cumulative_degree = {nid: 0 for nid in all_nodes}  # 跟踪累积度数

        print(f"[ROLAND] Training on {len(self.snapshots)} snapshots, {num_nodes} nodes")

        for sidx, g in enumerate(self.snapshots):
            if g is None:
                continue

            t0 = time.time()

            # 提取边和特征
            edges = g.get_edgelist()
            types = g.es["actions"]
            timestamps = g.es["timestamp"]

            # 初始化节点状态（首次出现)
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

            # 节点特征：使用当前快照的原始特征 (properties, degree等)
            node_features_list = []
            
            for nid in sorted(all_nodes):
                # 构造当前快照的节点特征向量
                node_feat = np.zeros(self.embedding_dim, dtype=np.float32)
                
                # 如果节点在当前快照中活跃,提取其特征
                if nid in node_gids:
                    local_idx = node_gids.index(nid)
                    
                    # 特征1: 当前度数 (归一化)
                    degree = len([e for e in edges if node_gids[e[0]] == nid or node_gids[e[1]] == nid])
                    node_feat[0] = min(degree / 50.0, 1.0)  # 归一化到 [0,1]
                    
                    # 特征2-65: 节点属性的 hash 特征
                    try:
                        properties_str = g.vs[local_idx]['properties']
                    except (KeyError, AttributeError):
                        properties_str = ''
                    
                    if properties_str and len(properties_str) > 2:
                        prop_hash = hash(properties_str)
                        # 使用64位 hash 作为二进制特征
                        for i in range(min(64, self.embedding_dim - 1)):
                            bit_val = (prop_hash >> i) & 1
                            node_feat[i + 1] = float(bit_val)
                
                # 非活跃节点保持零特征 (表示不在当前快照中)
                node_features_list.append(node_feat)
            
            node_features_np = np.array(node_features_list, dtype=np.float32)
            node_features = torch.from_numpy(node_features_np).to(self.device)

            # 边特征：编码边类型 + 归一化的时间戳
            edge_type_indices = [self.edge_type_vocab.get(types[i], 0) for i in range(len(edges))]
            edge_features = torch.zeros(len(edges), 16, device=self.device)
            
            # One-hot 编码边类型
            for i, type_idx in enumerate(edge_type_indices):
                if type_idx < 15:  # 前15维用于类型
                    edge_features[i, type_idx] = 1.0
            
            # 第16维用于归一化的时间信息
            if len(timestamps) > 0:
                ts_array = np.array(timestamps, dtype=np.float32)
                ts_min, ts_max = ts_array.min(), ts_array.max()
                if ts_max > ts_min:
                    ts_normalized = (ts_array - ts_min) / (ts_max - ts_min)
                else:
                    ts_normalized = np.ones_like(ts_array) * 0.5
                edge_features[:, 15] = torch.from_numpy(ts_normalized).to(self.device)

            # 多层 GNN 传播
            x = node_features
            for layer in self.gnn_layers:
                x = layer(x, edge_index, edge_features)

            # 更新活跃节点的状态
            # 计算当前快照中每个节点的度数
            node_current_degree = {}
            for u, v in edges:
                u_gid, v_gid = node_gids[u], node_gids[v]
                node_current_degree[u_gid] = node_current_degree.get(u_gid, 0) + 1
                node_current_degree[v_gid] = node_current_degree.get(v_gid, 0) + 1
            
            active_node_ids = set(node_gids)
            for nid in active_node_ids:
                idx = node_id_map[nid]
                new_degree = node_current_degree.get(nid, 0)
                existing_degree = node_cumulative_degree[nid]
                
                # 使用动态 keep_ratio 更新节点状态
                self._update_node_state(
                    nid, 
                    x[idx].detach().cpu().numpy(),
                    existing_degree=existing_degree,
                    new_degree=new_degree
                )
                
                # 累积度数
                node_cumulative_degree[nid] += new_degree

            # 快照级嵌入：使用多种聚合方式增强表达能力
            active_indices = [node_id_map[nid] for nid in active_node_ids]
            if active_indices:
                active_x = x[active_indices]  # 只取活跃节点
                # 组合 mean, max, std 三种统计量
                snapshot_mean = active_x.mean(dim=0).detach().cpu().numpy()
                snapshot_max = active_x.max(dim=0)[0].detach().cpu().numpy()
                snapshot_std = active_x.std(dim=0).detach().cpu().numpy()
                
                # 拼接并归一化 (使用前 embedding_dim 维度保持一致)
                snapshot_emb = snapshot_mean * 0.5 + snapshot_max * 0.3 + snapshot_std * 0.2
            else:
                snapshot_emb = np.zeros(self.embedding_dim, dtype=np.float32)
            
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
            'node_type_vocab': self.node_type_vocab,  # 保存节点类型词汇表
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
        instance.node_type_vocab = state.get('node_type_vocab', {})  # 兼容旧模型
        instance.snapshot_embeddings_list = state['snapshot_embeddings']

        print(f"[ROLAND] Encoder model loaded successfully (Original snapshots: {state['num_snapshots']}, Current snapshots: {len(snapshot_sequence)})")
        return instance
