import time
import math
import random
import mmh3
from collections import defaultdict
import numpy as np

from process.embedders.base import GraphEmbedderBase


# ===========================================================
# === WL Histogram with Time Decay + Recent Bin Pruning =====
# ===========================================================
class WLHistogram:
    def __init__(self, R=3, decay_lambda=0.0005, max_bins=20000):
        """
        R: WL传播层数
        decay_lambda: 时间衰减系数
        max_bins: 最大bin数（仅保留最新的max_bins条）
        """
        self.R = R
        self.decay_lambda = decay_lambda
        self.max_bins = max_bins

        self.hist = defaultdict(float)     # label -> weight
        self.last_update = {}              # label -> 最近更新时间
        self.labels = {}                   # node -> current label
        self.adj = defaultdict(dict)       # node -> edge_type -> set(neighbors)
        self.last_decay_ts = time.time()

        # 缓存加速
        self._label_hash_cache = {}
        self._pair_hash_cache = {}

    # ---------- Hash helpers ----------
    def _hash_label(self, lab: str) -> int:
        h = self._label_hash_cache.get(lab)
        if h is None:
            h = mmh3.hash64(lab)[0]
            self._label_hash_cache[lab] = h
        return h

    def _hash_pair(self, et: str, lbl: str) -> int:
        key = (et, lbl)
        h = self._pair_hash_cache.get(key)
        if h is None:
            h = mmh3.hash64(et + '|' + lbl)[0]
            self._pair_hash_cache[key] = h
        return h

    # ---------- 单bin懒衰减 ----------
    def _bump_bin(self, key: str, ts: float, delta: float = 1.0):
        last_ts = self.last_update.get(key, None)
        if last_ts is None:
            w_prev = 0.0
        else:
            dt = max(0.0, ts - last_ts)
            w_prev = self.hist.get(key, 0.0) * math.exp(-self.decay_lambda * dt)
        self.hist[key] = w_prev + float(delta)
        self.last_update[key] = ts

        # ---------- 被动衰减 ----------
    def _decay_passive(self, current_ts: float, affected_keys: set):
        """
        对非活跃的 bin 执行被动衰减：
        - affected_keys: 本轮已更新过的 bin，不再重复衰减
        """
        for k, last_ts in list(self.last_update.items()):
            if k in affected_keys:
                continue
            dt = current_ts - last_ts
            if dt > 0:
                decay_factor = math.exp(-self.decay_lambda * dt)
                w = self.hist.get(k, 0.0) * decay_factor
                if w < 1e-12:
                    del self.hist[k]
                    self.last_update.pop(k, None)
                else:
                    self.hist[k] = w
                    self.last_update[k] = current_ts

    # ---------- 全局清理 ----------
    def _clean_bins(self):
        """删除过小的或过旧的 bin，限制数量"""
        # 删除非常小的
        to_del = [k for k, w in self.hist.items() if w < 1e-12]
        for k in to_del:
            del self.hist[k]
            self.last_update.pop(k, None)

        # 限制最大 bin 数
        print(f"_clean_bins self.hist{len(self.hist)}, self.max_bins{self.max_bins}")
        if len(self.hist) > self.max_bins:
            sorted_bins = sorted(self.last_update.items(), key=lambda x: x[1], reverse=True)
            keep = set(k for k, _ in sorted_bins[:self.max_bins])
            for k in list(self.hist.keys()):
                if k not in keep:
                    del self.hist[k]
                    self.last_update.pop(k, None)
            print(f"[WLHistogram] pruned to {self.max_bins} bins")

    def ingest_edges(self, edges, types, node_gids, node_labels, timestamps):
        if node_labels:
            for vid_local, label in node_labels.items():
                gid = node_gids[vid_local]
                self.labels.setdefault(gid, label)

        affected_time = {}
        for i, (u_local, v_local) in enumerate(edges):
            et = types[i]
            u = node_gids[u_local]
            v = node_gids[v_local]
            ts = timestamps[i] if timestamps is not None else 0
            self.adj[u].setdefault(et, set()).add(v)
            self.adj[v].setdefault(f"rev_{et}", set()).add(u)
            if u not in affected_time or ts > affected_time[u]:
                affected_time[u] = ts
            if v not in affected_time or ts > affected_time[v]:
                affected_time[v] = ts


        t0 = time.time()
        updated_keys = self.update_wl_local(affected_time)
        print(f"[ingest_edges] update_wl_local({len(affected_time)} nodes): {time.time() - t0:.4f}s")

        current_ts = max(affected_time.values())

        # 第二阶段：衰减未更新的 bin
        self._decay_passive(current_ts, updated_keys)

        # 第三阶段：清理过期 / 超限的 bin
        self._clean_bins()

    def update_wl_local(self, affected_time: dict):
        """
        affected_time: {node_gid: ts}
        - 每个节点用自己的时间戳更新；
        - 邻居传播时取上层时间的最大值。
        """
        if not affected_time:
            return

        labels_round = self.labels.copy()
        frontier = set(affected_time.keys())
        frontier_ts = dict(affected_time)
        updated_keys = set()  # 本轮更新过的 bin

        for _ in range(self.R):
            if not frontier:
                break

            nxt_labels = {}
            next_frontier_ts = {}

            for n in frontier:
                ts_n = frontier_ts[n]
                lab = labels_round.get(n, self.labels.get(n, "")) or ""

                neigh_hash = 0
                for et, nbrs in self.adj.get(n, {}).items():
                    for x in nbrs:
                        lbl_x = labels_round.get(x, self.labels.get(x, "")) or ""
                        neigh_hash ^= self._hash_pair(et, lbl_x)

                sig_val = (self._hash_label(lab) ^ neigh_hash) & ((1 << 64) - 1)
                new_lab = hex(sig_val)
                nxt_labels[n] = new_lab

                self._bump_bin(new_lab, ts_n, delta=1.0)
                updated_keys.add(new_lab)

                # 邻居传播
                for et, nbrs in self.adj.get(n, {}).items():
                    for x in nbrs:
                        if x not in next_frontier_ts or ts_n > next_frontier_ts[x]:
                            next_frontier_ts[x] = ts_n

            labels_round.update(nxt_labels)
            self.labels.update(nxt_labels)
            frontier = set(next_frontier_ts.keys())
            frontier_ts = next_frontier_ts

        # 返回更新过的 bin key，用于后续被动衰减
        return updated_keys




# ===========================================================
# === HistoSketch ===========================================
# ===========================================================
class HistoSketch:
    def __init__(self, sketch_size=64, seed=42):
        if sketch_size <= 0:
            raise ValueError("sketch_size 必须为正整数")
        self.K = sketch_size
        random.seed(seed)
        self.a = [random.random() + 1e-9 for _ in range(self.K)]

    def _cws_hash(self, key: str, weight: float, k: int):
        h = mmh3.hash64(key, seed=k, signed=False)
        h = h[0] if isinstance(h, tuple) else h
        score = -math.log(max(weight, 1e-9)) / self.a[k]
        return h, score

    def sketch(self, histogram: dict):
        sig = [(0, float('inf')) for _ in range(self.K)]
        for k_str, w in ((str(k), v) for k, v in histogram.items() if v > 0):
            for i in range(self.K):
                h, s = self._cws_hash(k_str, w, i)
                if s < sig[i][1]:
                    sig[i] = (h, s)
        return [int(h) for h, _ in sig]


# ===========================================================
# === UnicornGraphEmbedder =================================
# ===========================================================
class UnicornGraphEmbedder(GraphEmbedderBase):
    def __init__(self, snapshots, features=None, mapp=None,
                 R=3, decay_lambda=0.0005, sketch_size=256,
                 max_bins=20000):
        super().__init__(snapshots, features, mapp)
        self.snapshots = self.G
        self.wl = WLHistogram(R=R, decay_lambda=decay_lambda, max_bins=max_bins)
        self.hs = HistoSketch(sketch_size=sketch_size)
        self.sketch_snapshots = []

    def train(self):
        for sidx, g in enumerate(self.snapshots):
            if g is None:
                continue
            edges = g.get_edgelist()
            types = g.es["type"]
            props = g.vs["properties"]
            timestamps = g.es["timestamp"]
            node_gids = {vid: g.vs[vid]['name'] for vid in range(g.vcount())}
            node_labels = {vid: props[vid] for vid in range(g.vcount())}

            t0 = time.time()
            self.wl.ingest_edges(edges, types, node_gids, node_labels, timestamps)
            print(f"[snapshot {sidx}] ingest_edges: {time.time()-t0:.4f}s")

            t0s = time.time()
            sketch = self.hs.sketch(self.wl.hist)
            print(f"[snapshot {sidx}] sketch: {time.time()-t0s:.4f}s (bins={len(self.wl.hist)})")
            self.sketch_snapshots.append((max(timestamps), sketch))

    def get_snapshot_embeddings(self, snapshot_sequence=None):
        arr = np.array([s for _, s in self.sketch_snapshots], dtype=np.uint64)
        floats = arr.astype(np.float64) / float(1 << 64)
        normed = (floats - floats.mean(0)) / (floats.std(0) + 1e-9)
        return normed.astype(np.float32)

    def embed_nodes(self):
        """暂不实现节点嵌入"""
        return {}

    def embed_edges(self):
        """暂不实现边嵌入"""
        return {}