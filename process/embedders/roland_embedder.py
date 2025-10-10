"""
ROLAND Embedder (GNNCL 风格)
改为“邻居预测自己”训练范式：
- 节点初始向量：nn.Embedding(num_nodes, embedding_dim)
- 轻量 GraphSAGE 卷积（自实现，避免外部依赖）
- MLP 产出节点表示（output_dim = hidden_conv_2）
- f_phi 预测头：用邻居均值去回归中心节点的 (h, tag)

目标：用结构邻域信息约束表示学习，移除 DGI/负样本等复杂逻辑，保持接口简洁。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Iterable, Union, List
import re
import hashlib

from process.embedders.base import GraphEmbedderBase


class GraphSAGEConv(nn.Module):
    """标准 GraphSAGE (mean) 聚合：
    h' = ReLU( W · concat( h, mean_{u∈N(v)} h_u ) )
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, l2_norm: bool = False):
        super().__init__()
        self.lin = nn.Linear(in_channels * 2, out_channels, bias=bias)
        self.l2_norm = l2_norm

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
        h_cat = torch.cat([x, nei_mean], dim=-1)
        out = F.relu(self.lin(h_cat))
        if self.l2_norm:
            out = F.normalize(out, p=2, dim=-1)
        return out


class GNNCL(nn.Module):
    """节点嵌入 + SAGE + MLP + f_phi 预测头"""

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
            # 将属性向量投影到 embedding_dim，并与 ID embedding 相加
            self.prop_proj = nn.Linear(self.prop_dim, embedding_dim)
        # 两层 GraphSAGE：embed→hidden，hidden→hidden
        self.sage1 = GraphSAGEConv(embedding_dim, hidden_dim, l2_norm=False)
        self.sage2 = GraphSAGEConv(hidden_dim, hidden_dim, l2_norm=False)
        self.dropout = nn.Dropout(dropout)
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

    def forward(self, node_indices: torch.Tensor, edge_index: torch.Tensor, prop_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        x0 = self.node_embedding(node_indices)  # (N, embedding_dim)
        if self.use_props and prop_feats is not None and prop_feats.numel() > 0:
            x0 = x0 + self.prop_proj(prop_feats)
        h1 = self.sage1(x0, edge_index)         # (N, hidden)
        h2_in = self.dropout(h1)
        h2 = self.sage2(h2_in, edge_index)      # (N, hidden)
        # 残差：形状一致时相加
        if h2.shape == h1.shape:
            h2 = h2 + h1
        h = self.mlp(h2)                        # (N, output)
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
        prop_feat_dim: int = 128,
        # 属性向量模式：'word2vec' 或 'hash'（默认 word2vec）
        prop_vec_mode: str = 'word2vec',
        # 预训练或缓存的 KeyedVectors 路径（.kv 或 .bin），为空则基于属性语料训练
        prop_w2v_path: Optional[str] = None,
        # 训练 W2V 的超参（仅在未提供路径时生效）
        prop_w2v_window: int = 5,
        prop_w2v_min_count: int = 1,
        prop_w2v_epochs: int = 10,
        neigh_pred_weight: float = 1.0,
        variance_weight: float = 0.1,
        train_indices: Optional[Union[Iterable[int], Tuple[int, int], int]] = None,
        model_path=None
    ):
        """
        Args:
            snapshots: list of igraph.Graph
            embedding_dim: 节点初始嵌入维度（nn.Embedding）
            hidden_conv_1: SAGE 隐层维度
            hidden_conv_2: 最终节点表示维度（也是下游读出维度）
            num_epochs: 训练轮数
            lr: 学习率
            neigh_pred_weight: 邻居预测损失权重
            variance_weight: 方差正则权重（抑制表示塌缩）
            train_indices: 仅训练给定范围/集合内的快照（range、列表或 (start, end)）
            model_path: 保存/加载路径
        """
        super().__init__(snapshots, features, mapp)
        self.snapshots = self.G
        self.embedding_dim = embedding_dim
        self.hidden_conv_1 = hidden_conv_1
        self.hidden_conv_2 = hidden_conv_2
        self.num_epochs = num_epochs
        self.lr = lr
        self.neigh_pred_weight = float(neigh_pred_weight)
        self.variance_weight = float(variance_weight)
        self.model_path = model_path or self._default_path
        self.prop_feat_dim = int(prop_feat_dim)
        # 属性向量模式与 W2V 选项
        self.prop_vec_mode = str(prop_vec_mode or 'word2vec').lower()
        if self.prop_vec_mode not in ('word2vec', 'hash'):
            self.prop_vec_mode = 'word2vec'
        self.prop_w2v_path = prop_w2v_path
        self.prop_w2v_window = int(prop_w2v_window)
        self.prop_w2v_min_count = int(prop_w2v_min_count)
        self.prop_w2v_epochs = int(prop_w2v_epochs)
        # 文本属性缓存：属性文本 -> 向量（避免重复计算；快照间属性不同也能区分）
        self._prop_cache: Dict[str, np.ndarray] = {}
        # 运行期 W2V 词向量（KeyedVectors），仅在训练期需要；load 后推理可直接使用 prop_cache
        self._w2v_kv = None  # 延迟初始化，避免导入 gensim 报错

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
            prop_dim=self.prop_feat_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        # 存储最终嵌入
        self.snapshot_embeddings_list = []
        
        # 若使用 Word2Vec，则准备语料并训练/加载词向量（仅训练阶段用；推理期依赖 prop_cache）
        if self.prop_feat_dim > 0 and self.prop_vec_mode == 'word2vec':
            try:
                self._prepare_word2vec()
            except Exception as ex:
                print(f"[ROLAND] Word2Vec 初始化失败，将回退为零向量属性：{ex}")
                self._w2v_kv = None

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
            f"[ROLAND] Objective: NeighborPredictOnly | Epochs: {self.num_epochs}, LR: {self.lr}, WD: 1e-4, "
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
                node_ids, edge_index, prop_feats = self._igraph_to_torch(g)

                # 前向：得到本快照节点的表示 h_i (N_local, d)
                z = self.model(node_ids, edge_index, prop_feats)

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

    # 已不需要 present 索引辅助（仅用本地节点顺序与本地边）

    def _generate_final_embeddings(self):
        """训练后生成最终嵌入（用于后续任务）"""
        self.model.eval()
        with torch.no_grad():
            for sidx, g in enumerate(self.snapshots):
                if g is None:
                    self.snapshot_embeddings_list.append({})
                    continue
                
                node_ids, edge_index, prop_feats = self._igraph_to_torch(g)
                final_emb = self.model(node_ids, edge_index, prop_feats)
                # 提取嵌入字典
                embeddings_dict = self._extract_final_embeddings(g, final_emb)
                self.snapshot_embeddings_list.append(embeddings_dict)
    
    def _igraph_to_torch(self, g) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """将 igraph 转为 (节点全局ID, 本地边索引, 属性特征) 以适配 GNNCL 前向。

        Returns:
            node_ids: (N_local,) 当前快照节点对应的全局ID（顺序与 g.vs 对齐）
            edge_index: (2, E) 使用本地索引(0..N_local-1) 的 COO 边
            prop_feats: (N_local, prop_feat_dim) 若启用属性特征，否则为 None
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
        # 属性特征（可选）
        prop_feats_t: Optional[torch.Tensor] = None
        if self.prop_feat_dim > 0:
            mat = np.zeros((len(node_gids), self.prop_feat_dim), dtype=np.float32)
            for i, nid in enumerate(node_gids):
                mat[i] = self._get_node_prop_vec(g, i)
            prop_feats_t = torch.from_numpy(mat).to(self.device)
        return node_ids, edge_index, prop_feats_t

    def _get_node_prop_vec(self, g, local_idx: int) -> np.ndarray:
        """取节点属性字符串，按模式生成向量（缓存按属性文本）。"""
        try:
            prop_str = g.vs[local_idx].get('properties', '') or ''
        except Exception:
            prop_str = ''
        key = str(prop_str)
        if key in self._prop_cache:
            return self._prop_cache[key]
        if self.prop_feat_dim <= 0:
            vec = np.zeros((0,), dtype=np.float32)
        elif self.prop_vec_mode == 'word2vec':
            vec = self._text_w2v_vector(key, self.prop_feat_dim)
        else:
            vec = self._text_hash_vector(key, self.prop_feat_dim)
        self._prop_cache[key] = vec
        return vec

    @staticmethod
    def _text_hash_vector(text: str, dim: int) -> np.ndarray:
        """将文本转成固定维度的哈希计数向量（L2 归一化）。"""
        if dim <= 0:
            return np.zeros(0, dtype=np.float32)
        v = np.zeros(dim, dtype=np.float32)
        if not text:
            return v
        # 简单分词：按非字母数字切分
        tokens = [t for t in re.split(r'[^A-Za-z0-9]+', str(text).lower()) if t]
        for tok in tokens:
            h = int(hashlib.md5(tok.encode('utf-8')).hexdigest(), 16) % dim
            v[h] += 1.0
        # L2 归一化，防止长度差异干扰
        n = np.linalg.norm(v) + 1e-12
        return (v / n).astype(np.float32)

    # ===== Word2Vec 支持 =====
    def _prepare_word2vec(self):
        """构建属性语料并训练/加载 Word2Vec（KeyedVectors 保存在内存）。"""
        # 动态导入，避免环境未装 gensim 时顶层报错
        try:
            from gensim.models import Word2Vec, KeyedVectors  # type: ignore
        except Exception as ex:  # pragma: no cover - 环境问题
            raise RuntimeError("需要安装 gensim 才能使用 word2vec 属性特征: pip install gensim") from ex

        corpus: List[List[str]] = []
        # 仅使用训练快照（通常是良性集）构建语料，避免泄漏
        for sidx in self.train_snapshot_indices:
            if sidx < 0 or sidx >= len(self.snapshots):
                continue
            g = self.snapshots[sidx]
            if g is None:
                continue
            for i in range(g.vcount()):
                try:
                    prop_str = g.vs[i].get('properties', '') or ''
                except Exception:
                    prop_str = ''
                tokens = [t for t in re.split(r'[^A-Za-z0-9]+', str(prop_str).lower()) if t]
                if tokens:
                    corpus.append(tokens)

        if self.prop_w2v_path:
            try:
                kv = KeyedVectors.load(self.prop_w2v_path)
            except Exception:
                # 兼容常见的二进制格式（如 GoogleNews 或自保存 .kv）
                try:
                    kv = KeyedVectors.load_word2vec_format(self.prop_w2v_path, binary=True)
                except Exception as ex:
                    raise RuntimeError(f"无法从 {self.prop_w2v_path} 加载 KeyedVectors: {ex}")
            if kv.vector_size != self.prop_feat_dim:
                print(f"[ROLAND] 注意：加载的词向量维度 {kv.vector_size} 与 prop_feat_dim={self.prop_feat_dim} 不一致，将截断/填零对齐。")
            self._w2v_kv = kv
        else:
            if not corpus:
                print("[ROLAND] 属性语料为空，Word2Vec 将退化为零向量。")
                self._w2v_kv = None
                return
            # 训练一个轻量 Skip-gram 模型
            sg = 1  # skip-gram 通常对小语料更稳健
            model = Word2Vec(
                sentences=corpus,
                vector_size=self.prop_feat_dim,
                window=self.prop_w2v_window,
                min_count=self.prop_w2v_min_count,
                sg=sg,
                epochs=self.prop_w2v_epochs,
                workers=1,
            )
            self._w2v_kv = model.wv

    def _text_w2v_vector(self, text: str, dim: int) -> np.ndarray:
        """用 Word2Vec 对文本分词后取词向量平均；若无可用词则返回零向量。"""
        if dim <= 0:
            return np.zeros(0, dtype=np.float32)
        if not text:
            return np.zeros(dim, dtype=np.float32)
        if self._w2v_kv is None:
            # 没有可用的词向量（比如未成功初始化），返回零向量
            return np.zeros(dim, dtype=np.float32)
        tokens = [t for t in re.split(r'[^A-Za-z0-9]+', str(text).lower()) if t]
        if not tokens:
            return np.zeros(dim, dtype=np.float32)
        vecs = []
        for tkn in tokens:
            if tkn in self._w2v_kv.key_to_index:
                v = self._w2v_kv[tkn]
                if v.shape[0] != dim:
                    # 对齐维度（截断或补零）
                    if v.shape[0] > dim:
                        v = v[:dim]
                    else:
                        pad = np.zeros(dim, dtype=np.float32)
                        pad[: v.shape[0]] = v
                        v = pad
                vecs.append(v.astype(np.float32))
        if not vecs:
            return np.zeros(dim, dtype=np.float32)
        m = np.mean(np.stack(vecs, axis=0), axis=0)
        # L2 归一化
        n = np.linalg.norm(m) + 1e-12
        return (m / n).astype(np.float32)
    
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
                'neigh_pred_weight': self.neigh_pred_weight,
                'variance_weight': self.variance_weight,
                'prop_feat_dim': self.prop_feat_dim,
                'prop_vec_mode': self.prop_vec_mode,
                'prop_w2v_path': self.prop_w2v_path,
                'prop_w2v_window': self.prop_w2v_window,
                'prop_w2v_min_count': self.prop_w2v_min_count,
                'prop_w2v_epochs': self.prop_w2v_epochs,
            },
            'model_state': self.model.state_dict(),
            'snapshot_embeddings': self.snapshot_embeddings_list,
            'all_nodes': self.all_nodes,  # 保存节点列表（load时重建node_id_map）
            # 属性缓存（以属性字符串为键），避免推理时依赖 gensim
            'prop_cache': self._prop_cache,
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
            'num_epochs', 'lr',
            'neigh_pred_weight', 'variance_weight',
            'prop_feat_dim',
            'prop_vec_mode', 'prop_w2v_path', 'prop_w2v_window', 'prop_w2v_min_count', 'prop_w2v_epochs',
            'train_indices', 'model_path'
        }
        params = {k: v for k, v in raw_params.items() if k in allowed_keys}
        instance = cls(snapshot_sequence, **params)
        
        # 恢复模型权重和嵌入
        instance.model.load_state_dict(state['model_state'])
        instance.snapshot_embeddings_list = state['snapshot_embeddings']
        instance.all_nodes = state['all_nodes']
        # 恢复属性缓存（若存在）
        if 'prop_cache' in state and isinstance(state['prop_cache'], dict):
            instance._prop_cache = state['prop_cache']
        
        # 重建 node_id_map
        instance.node_id_map = {nid: i for i, nid in enumerate(instance.all_nodes)}
        instance.num_nodes = len(instance.all_nodes)

        print("[ROLAND] Model loaded successfully")
        return instance
