import time
import math
import random
import mmh3
from collections import defaultdict
import numpy as np

from process.embedders.base import GraphEmbedderBase


# --- WL histogram + decay (流式 / 全局直方图) ---
class WLHistogram:
    def __init__(self, R=3, decay_lambda=0.0005):
        self.R = R
        self.decay_lambda = decay_lambda
        self.hist = defaultdict(float)     # global histogram: label -> weight
        self.labels = {}                   # node -> current label (string)
        self.adj = defaultdict(dict)       # node -> edge_type -> set(neighbors)
        self.last_decay_ts = time.time()
        # 哈希缓存，显著减少 mmh3 调用次数
        self._label_hash_cache = {}        # str(label) -> int64
        self._pair_hash_cache  = {}        # (etype, label) -> int64

    # ---------- Hash helpers (with cache) ----------
    def _hash_label(self, lab: str) -> int:
        """缓存后的标签哈希"""
        h = self._label_hash_cache.get(lab)
        if h is None:
            h = mmh3.hash64(lab)[0]
            self._label_hash_cache[lab] = h
        return h

    def _hash_pair(self, et: str, lbl: str) -> int:
        """缓存后的 (edge_type, label) 组合哈希"""
        key = (et, lbl)
        h = self._pair_hash_cache.get(key)
        if h is None:
            # 这里用 | 作为分隔符，避免 f-string 大量分配；cache 已经避免重复了
            h = mmh3.hash64(et + '|' + lbl)[0]
            self._pair_hash_cache[key] = h
        return h

    def ingest_edges(self, edges, types, node_gids, node_labels=None):
        """
        批量 ingest 多条边：
        - 建邻接
        - 一次性设置初始标签
        - 只对“新边端点集合”做局部 WL（随后按 R 轮向邻居传播）
        """
        if node_labels is not None:
            for vid_local, label in node_labels.items():
                gid = node_gids[vid_local]
                self.labels.setdefault(gid, label)

        affected = set()
        for i, (u_local, v_local) in enumerate(edges):
            et = types[i]
            u = node_gids[u_local]
            v = node_gids[v_local]

            et_map_u = self.adj[u]
            if et not in et_map_u:
                et_map_u[et] = set()
            et_map_u[et].add(v)

            rev_type = f"rev_{et}"
            et_map_v = self.adj[v]
            if rev_type not in et_map_v:
                et_map_v[rev_type] = set()
            et_map_v[rev_type].add(u)

            affected.add(u)
            affected.add(v)

        if affected:
            t0_upd = time.time()
            self.update_wl_local(affected)
            t_upd = time.time() - t0_upd
            print(f"[ingest_edges] update_wl_local on {len(affected)} nodes: {t_upd:.4f}s")

    # ---------- Decay ----------
    def _decay(self):
        now = time.time()
        dt = now - self.last_decay_ts
        if dt <= 0:
            return
        factor = math.exp(-self.decay_lambda * dt)
        keys = list(self.hist.keys())
        for k in keys:
            self.hist[k] *= factor
            if self.hist[k] < 1e-12:
                del self.hist[k]
        self.last_decay_ts = now

    # ---------- Local WL update ----------
    def update_wl_local(self, affected_nodes):
        """
        只更新受影响节点，并按 R 轮向外层邻居传播（同步式）：
        - 每一轮使用“上一轮的标签快照”计算
        - 使用交换律哈希（XOR 聚合）避免排序 + 字符串拼接
        - 使用缓存减少哈希调用
        """
        if not affected_nodes:
            return

        self._decay()

        # 同步式：每轮基于上一轮的标签快照
        labels_round = self.labels.copy()
        frontier = set(n for n in affected_nodes if n in self.adj)  # 没有邻接的节点可跳过

        for _ in range(self.R):
            if not frontier:
                break

            nxt = {}
            # 仅对 frontier 节点更新
            for n in frontier:
                lab = labels_round.get(n, self.labels.get(n, "")) or ""
                et_map = self.adj.get(n, {})
                neigh_hash = 0
                for et, nbrs in et_map.items():
                    for x in nbrs:
                        lbl_n = labels_round.get(x, self.labels.get(x, "")) or ""
                        neigh_hash ^= self._hash_pair(et, lbl_n)

                self_hash = self._hash_label(lab)
                sig_val = (self_hash ^ neigh_hash) & ((1 << 64) - 1)
                new_lab = hex(sig_val)
                nxt[n] = new_lab
                self.hist[new_lab] += 1.0

            # 应用这一轮结果到快照 & 全局 labels
            labels_round.update(nxt)
            self.labels.update(nxt)

            # 下一轮的 frontier：本轮更新节点的所有邻居（同步传播）
            new_frontier = set()
            for n in nxt.keys():
                for et, nbrs in self.adj.get(n, {}).items():
                    new_frontier.update(nbrs)
            frontier = new_frontier


# --- 极简 HistoSketch 占位实现（把直方图压成固定长度 sketch） ---
class HistoSketch:
    """高效实现的 HistoSketch：将加权直方图映射为固定长度 sketch 向量。"""

    def __init__(self, sketch_size: int = 64, seed: int = 42):
        if sketch_size <= 0:
            raise ValueError("sketch_size 必须为正整数")

        self.K = sketch_size
        random.seed(seed)
        # Consistent Weighted Sampling 参数
        self.a = [random.random() + 1e-9 for _ in range(self.K)]

    def _cws_hash(self, key: str, weight: float, k: int) -> tuple[int, float]:
        """为单个键值对计算第 k 个候选哈希与得分。"""
        h = mmh3.hash64(key, seed=k, signed=False)
        h = h[0] if isinstance(h, tuple) else h
        score = -math.log(max(weight, 1e-9)) / self.a[k]
        return h, score

    def sketch(self, histogram: dict) -> list[int]:
        """将直方图 {key: weight} 压缩为固定长度 sketch 向量。"""
        sig = [(0, float('inf')) for _ in range(self.K)]
        for k_str, w in ((str(k), v) for k, v in histogram.items() if v > 0):
            for i in range(self.K):
                h, s = self._cws_hash(k_str, w, i)
                if s < sig[i][1]:
                    sig[i] = (h, s)
        return [int(h) for h, _ in sig]


# --- UNICORN 风格嵌入器 ---
class UnicornGraphEmbedder(GraphEmbedderBase):
    def __init__(self, snapshots, features=None, mapp=None,
                 R=3, decay_lambda=0.0005, sketch_size=256,
                 snapshot_edges=2000, wl_batch=500):
        super().__init__(snapshots, features, mapp)
        self.snapshots = self.G
        self.wl = WLHistogram(R=R, decay_lambda=decay_lambda)
        self.hs = HistoSketch(sketch_size=sketch_size)
        self.snapshot_edges = snapshot_edges
        self.wl_batch = wl_batch
        self.sketch_snapshots = []         # list[(ts, sketch)]
        self.snapshot_embeddings = None
        self._node_embeddings = {}
        self._edge_embeddings = {}

    def train(self):
        """
        基于 self.snapshots（每个元素是 igraph.Graph）生成快照嵌入：
        - 批量 ingest 边与初始标签
        - 仅对新增边端点集合做局部 WL（R 轮向外传播）
        - 取全局直方图做 HistoSketch
        """
        if not hasattr(self, "snapshots") or not self.snapshots:
            raise RuntimeError("self.snapshots 为空，请先设置快照数据")

        for sidx, g in enumerate(self.snapshots):
            if g is None:
                continue

            edges = g.get_edgelist()          # [(u,v), ...]
            types = g.es["type"]              # len == |E|
            props = g.vs["properties"]        # len == |V|

            vcount = g.vcount()
            node_gids = {vid: g.vs[vid]['name'] for vid in range(vcount)}
            node_labels = {vid: props[vid] for vid in range(vcount)}

            # 一次性建邻接 + 局部 WL
            t0 = time.time()
            self.wl.ingest_edges(edges, types, node_gids, node_labels=node_labels)
            t_ingest = time.time() - t0
            print(f"[snapshot {sidx}] ingest_edges: {t_ingest:.4f}s")

            # 从全局直方图得到定长 sketch
            t0s = time.time()
            sketch = self.hs.sketch(self.wl.hist)
            t_sketch = time.time() - t0s
            print(f"[snapshot {sidx}] sketch: {t_sketch:.4f}s (bins={len(self.wl.hist)})")
            self.sketch_snapshots.append((time.time(), sketch))

    def get_snapshot_embeddings(self, snapshot_sequence=None):
        """
        返回按索引序列选取的快照 sketch 堆叠成的矩阵 (n_snapshots, sketch_size)
        """
        if not self.sketch_snapshots:
            raise RuntimeError("还没有任何快照，请先调用 train() 生成。")

        if snapshot_sequence is None:
            snapshot_sequence = list(range(len(self.sketch_snapshots)))

        t0 = time.time()
        embeddings = []
        for idx in snapshot_sequence:
            _, sketch = self.sketch_snapshots[idx]
            embeddings.append(sketch)
        # 原始 sketch 由 64-bit 整数构成，直接用这些巨大整数训练会导致数值不稳定（Inf）
        arr = np.array(embeddings)

        # 把 uint64 hash 映射到 [0, 1) 的浮点数，然后按列做标准化 (mean=0, std=1)
        try:
            # 确保是无符号 64 位范围
            arr_u = arr.astype(np.uint64)
            floats = arr_u.astype(np.float64) / float(1 << 64)
        except Exception:
            # 退回安全的转换路径
            floats = arr.astype(np.float64)

        # 列标准化，避免某些列的常数或非常小的方差导致除零
        col_mean = floats.mean(axis=0)
        col_std = floats.std(axis=0)
        col_std[col_std == 0.0] = 1.0
        normed = (floats - col_mean) / col_std

        t_total = time.time() - t0
        print(f"[get_snapshot_embeddings] build array: {t_total:.4f}s, raw_shape={arr.shape}, normed_shape={normed.shape}")
        # 打印一些统计信息帮助调试
        print(f"[get_snapshot_embeddings] col mean (first3)={col_mean[:3]}, col std (first3)={col_std[:3]}")
        return normed.astype(np.float32)

    def embed_edges(self):
        pass

    def embed_nodes(self):
        pass