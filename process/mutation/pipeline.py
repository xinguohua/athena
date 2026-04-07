"""
MutationPipeline: 为每个良性快照生成专属难负样本 G̃_b

训练时负样本集 N(b) = 攻击图(共享) + G̃_b(专属)，加权对比损失自动放大难负样本。
验证不通过换攻击图重试，直到成功或穷尽候选。

EgoMutationPipeline: ego 级变异（推荐）。
在 ego 子图（~32 节点）级别做变异，避免快照级变异（~500 节点）导致
攻击信号被良性上下文稀释。
"""
from __future__ import annotations
from typing import List, Tuple, Optional, Callable, Set, Dict
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random

try:
    import igraph as ig
except ImportError:
    ig = None

from .structural import aligned_region_search, subgraph_replacement
from .semantic import apply_semantic_mutation_llm, _collect_benign_corpus
from .verification import verify_mutation, build_historical_profiles
from .wl_kernel import top_k_similar_attacks


def _extract_ego(g, center: int, r_hop: int = 2, max_nodes: int = 32):
    """BFS 顺序提取 ego 子图（与 gcc_embedder_dev._ego_subgraph 一致）"""
    visited = [center]
    visited_set = {center}
    queue = deque([center])
    while queue and len(visited) < max_nodes:
        v = queue.popleft()
        for nb in g.neighbors(v, mode="all"):
            if nb not in visited_set and len(visited) < max_nodes:
                visited.append(nb)
                visited_set.add(nb)
                queue.append(nb)
    return g.subgraph(sorted(visited))


class MutationPipeline:

    def __init__(
        self,
        snapshots: list,
        benign_range: Tuple[int, int],
        attack_range: Tuple[int, int],
        delta_h: float = 0.3,
        delta_h_upper: float = 0.95,
        top_k: int = 5,
        top_m: int = 3,
        max_region_size: int = 32,
    ):
        self.snapshots = snapshots
        self.benign_range = benign_range
        self.attack_range = attack_range
        self.delta_h = delta_h
        self.delta_h_upper = delta_h_upper
        self.top_k = top_k
        self.top_m = top_m
        self.max_region_size = max_region_size

        b_start, b_end = benign_range
        a_start, a_end = attack_range
        self.benign_graphs = [
            (snapshots[i], i) for i in range(b_start, b_end + 1)
            if i < len(snapshots) and snapshots[i] is not None
        ]
        self.attack_graphs = [
            (snapshots[i], i) for i in range(a_start, a_end + 1)
            if i < len(snapshots) and snapshots[i] is not None
        ]

        self.benign_commands: Set[str] = set()
        self.benign_args: Set[str] = set()
        self.benign_files: Set[str] = set()
        self.entity_ops: Dict[str, Set[str]] = {}
        self.type_attrs: Dict[str, Set[str]] = {}
        self._build_profiles()

    def _build_profiles(self):
        t0 = time.time()
        self.benign_commands, self.benign_args, self.benign_files = \
            _collect_benign_corpus(self.benign_graphs)
        self.entity_ops, self.type_attrs = \
            build_historical_profiles(self.benign_graphs)

        # 预计算所有图的 WL histogram，避免重复计算
        from .wl_kernel import wl_subtree_labels, _kernel_from_histograms
        self._wl_cache = {}  # {snapshot_idx: histograms}

        for g, idx in self.benign_graphs + self.attack_graphs:
            if g is not None and g.vcount() > 0:
                self._wl_cache[idx] = wl_subtree_labels(g, h=3)

        # 预计算每个良性图的 Top-K 攻击图（一次性算完所有配对）
        self._topk_cache = {}  # {benign_idx: [(attack_graph, attack_idx, sim), ...]}
        for g_b, b_idx in self.benign_graphs:
            if b_idx not in self._wl_cache:
                continue
            hist_b = self._wl_cache[b_idx]
            scored = []
            for g_a, a_idx in self.attack_graphs:
                if a_idx not in self._wl_cache:
                    continue
                sim = _kernel_from_histograms(hist_b, self._wl_cache[a_idx])
                scored.append((g_a, a_idx, sim))
            scored.sort(key=lambda x: -x[2])
            self._topk_cache[b_idx] = scored[:self.top_k]

        print(f"[MutPipeline] 剖面+WL预计算完成: "
              f"{len(self.benign_commands)} 命令, "
              f"{len(self._wl_cache)} 图histogram, "
              f"耗时 {time.time()-t0:.1f}s")

    def generate(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        max_mutations: int = 100,
        skip_verification: bool = False,
    ) -> Dict[int, object]:
        """
        为每个良性快照生成专属难负样本。
        有 LLM 时做语义变异（伪装攻击 properties），无 LLM 时保留原始攻击 properties。

        Returns:
            {benign_idx: mutated_graph}
        """
        print(f"\n[MutPipeline] 为每个良性快照生成专属难负样本: "
              f"良性={len(self.benign_graphs)}, 攻击={len(self.attack_graphs)}, "
              f"LLM={'有' if llm_fn else '无(仅结构变异)'}")

        t0 = time.time()
        result: Dict[int, object] = {}
        n_success = 0
        n_fallback = 0

        for bi, (g_b, b_idx) in enumerate(self.benign_graphs):
            if g_b is None or g_b.vcount() == 0:
                continue

            g_mut = self._generate_one(
                g_b, b_idx, llm_fn=llm_fn,
                skip_verification=skip_verification,
            )

            if g_mut is not None:
                result[b_idx] = g_mut
                n_success += 1
            else:
                # 兜底：用最相似攻击图
                topk = self._topk_cache.get(b_idx, [])
                if topk:
                    result[b_idx] = topk[0][0]
                    n_fallback += 1

            if (bi + 1) % 20 == 0 or bi == len(self.benign_graphs) - 1:
                print(f"[MutPipeline] 进度 {bi+1}/{len(self.benign_graphs)}: "
                      f"变异={n_success}, 兜底={n_fallback}")

        total_time = time.time() - t0
        print(f"[MutPipeline] 完成: {n_success} 变异 + {n_fallback} 兜底, "
              f"覆盖 {len(result)}/{len(self.benign_graphs)}, "
              f"总耗时 {total_time:.1f}s")
        return result

    def _generate_one(
        self, g_b, b_idx, llm_fn=None,
        skip_verification=False,
    ) -> Optional:
        """为单个良性快照生成变异图。不通过就换攻击图重试。"""
        similar = self._topk_cache.get(b_idx, [])

        for g_a, a_idx, sim in similar:
            candidates = aligned_region_search(
                g_b, g_a, max_region_size=self.max_region_size
            )
            for S_b, S_a, pi, score in candidates[:self.top_m]:
                if len(pi) < 2:
                    continue

                g_mut = subgraph_replacement(g_b, g_a, S_b, S_a, pi)
                if g_mut is None:
                    continue

                replaced = set(S_b)

                # 语义变异：只在有 LLM 时做，无 LLM 保留原始攻击 properties
                if llm_fn is not None:
                    g_mut = apply_semantic_mutation_llm(
                        g_mut, list(replaced),
                        self.benign_commands, self.benign_args,
                        llm_fn=llm_fn, r_hop=2,
                    )
                    if g_mut is None:
                        continue

                if not skip_verification:
                    passed, _ = verify_mutation(
                        g_mut, g_b, replaced,
                        self.entity_ops, self.type_attrs,
                        delta_h=self.delta_h, delta_h_upper=self.delta_h_upper,
                    )
                    if not passed:
                        continue

                return g_mut

        return None


class EgoMutationPipeline:
    """Ego 级变异 pipeline。

    在 ego 子图（~32节点）粒度做 WL 匹配 + 子图替换，
    而非在整个快照（~500节点）上操作。

    优势：变异后攻击节点占 ego 的 30-50%（而非 4%），
    对比学习的负样本信号清晰，不会和良性 ego 混淆。
    """

    def __init__(
        self,
        snapshots: list,
        benign_range: Tuple[int, int],
        attack_range: Tuple[int, int],
        r_hop: int = 2,
        ego_max_nodes: int = 32,
        top_k: int = 5,
        max_region_size: int = 16,
    ):
        self.snapshots = snapshots
        self.r_hop = r_hop
        self.ego_max_nodes = ego_max_nodes
        self.top_k = top_k
        self.max_region_size = max_region_size

        b_start, b_end = benign_range
        a_start, a_end = attack_range

        t0 = time.time()

        # 提取攻击 ego：以攻击节点为中心
        self.attack_egos = []  # [(ego_graph, snapshot_idx)]
        for i in range(a_start, a_end + 1):
            g = snapshots[i] if i < len(snapshots) else None
            if g is None or g.vcount() == 0:
                continue
            for v in range(g.vcount()):
                if g.vs[v].attributes().get('label', 0) == 1:
                    ego = _extract_ego(g, v, r_hop, ego_max_nodes)
                    if ego.vcount() >= 3:
                        self.attack_egos.append((ego, i))

        # 良性快照索引
        self.benign_indices = [
            i for i in range(b_start, b_end + 1)
            if i < len(snapshots) and snapshots[i] is not None
               and snapshots[i].vcount() > 0
        ]

        # 攻击 ego WL histogram 缓存
        from .wl_kernel import wl_subtree_labels, _kernel_from_histograms
        self._wl_fn = wl_subtree_labels
        self._kernel_fn = _kernel_from_histograms
        self._attack_wl = []
        for ego, sidx in self.attack_egos:
            self._attack_wl.append(self._wl_fn(ego, h=3))

        # 良性语料（用于语义变异）
        benign_graphs = [
            (snapshots[i], i) for i in self.benign_indices
        ]
        self.benign_commands, self.benign_args, self.benign_files = \
            _collect_benign_corpus(benign_graphs)

        print(f"[EgoMutPipeline] 初始化: "
              f"{len(self.attack_egos)} 攻击ego, "
              f"{len(self.benign_indices)} 良性快照, "
              f"耗时 {time.time()-t0:.1f}s")

    def _find_similar_attack_egos(self, ego_b, top_k=5):
        """为一个良性 ego 找 Top-K 最相似的攻击 ego"""
        hist_b = self._wl_fn(ego_b, h=3)
        scored = []
        for idx, hist_a in enumerate(self._attack_wl):
            sim = self._kernel_fn(hist_b, hist_a)
            scored.append((idx, sim))
        scored.sort(key=lambda x: -x[1])
        return [(self.attack_egos[idx][0], sim) for idx, sim in scored[:top_k]]

    def _structural_mutate(self, ego_b, ego_a):
        """仅做结构变异，返回 (g_mut, replaced_nodes) 或 None"""
        candidates = aligned_region_search(
            ego_b, ego_a, max_region_size=self.max_region_size
        )
        for S_b, S_a, pi, score in candidates[:3]:
            if len(pi) < 2:
                continue
            g_mut = subgraph_replacement(ego_b, ego_a, S_b, S_a, pi)
            if g_mut is None:
                continue
            n_atk = sum(1 for v in range(g_mut.vcount())
                        if g_mut.vs[v].attributes().get('label', 0) == 1)
            if n_atk < 2:
                continue
            return g_mut, list(set(S_b))
        return None

    def _semantic_mutate(self, g_mut, replaced, llm_fn, model_name="unknown"):
        """对结构变异后的图做 LLM 语义变异"""
        return apply_semantic_mutation_llm(
            g_mut, replaced,
            self.benign_commands, self.benign_args,
            llm_fn=llm_fn, r_hop=1, model_name=model_name,
        )

    def _mutate_one_ego(self, ego_b, ego_a, llm_fn=None):
        """对一对 (良性ego, 攻击ego) 做子图替换变异"""
        result = self._structural_mutate(ego_b, ego_a)
        if result is None:
            return None
        g_mut, replaced = result
        if llm_fn is not None:
            g_mut = self._semantic_mutate(g_mut, replaced, llm_fn, model_name=model_name)
        return g_mut

    def generate(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        egos_per_snapshot: int = 5,
        llm_workers: int = 8,
        model_name: str = "unknown",
    ) -> Dict[int, list]:
        """为每个良性快照生成 ego 级难负样本。

        两阶段流水线：
        1. 串行：WL 匹配 + 结构变异（很快，~0.01s/ego）
        2. 并发：LLM 语义变异（瓶颈，用线程池并发加速）

        Args:
            egos_per_snapshot: 每个良性快照生成的变异 ego 数量。
            llm_workers: LLM 并发线程数（默认 8）。

        Returns:
            {benign_snapshot_idx: [mutated_ego_graph, ...]}
        """
        print(f"\n[EgoMutPipeline] 生成 ego 级难负样本: "
              f"每快照{egos_per_snapshot}个, "
              f"{len(self.benign_indices)}个良性快照, "
              f"LLM={'有(并发'+str(llm_workers)+'线程)' if llm_fn else '无'}")

        if not self.attack_egos:
            print("[EgoMutPipeline] 无攻击 ego，跳过")
            return {}

        t0 = time.time()

        # ---- 阶段 1: 串行结构变异（收集待 LLM 处理的图）----
        # pending: [(b_idx, g_mut, replaced_nodes), ...]
        pending = []
        # no_llm_result: 不需要 LLM 的直接结果
        no_llm_result: Dict[int, list] = {}
        n_struct_total, n_struct_ok = 0, 0

        t_struct_start = time.time()
        for bi, b_idx in enumerate(self.benign_indices):
            g_b = self.snapshots[b_idx]
            if g_b is None or g_b.vcount() == 0:
                continue

            all_nodes = list(range(g_b.vcount()))
            n_sample = min(len(all_nodes), egos_per_snapshot * 3)
            sample_nodes = random.sample(all_nodes, n_sample) if n_sample < len(all_nodes) else all_nodes

            snap_egos = []
            for c in sample_nodes:
                if len(snap_egos) >= egos_per_snapshot:
                    break
                ego_b = _extract_ego(g_b, c, self.r_hop, self.ego_max_nodes)
                if ego_b.vcount() < 5:
                    continue

                similar = self._find_similar_attack_egos(ego_b, self.top_k)
                n_struct_total += 1
                for ego_a, sim in similar:
                    res = self._structural_mutate(ego_b, ego_a)
                    if res is not None:
                        g_mut, replaced = res
                        if llm_fn is not None:
                            pending.append((b_idx, g_mut, replaced))
                        else:
                            snap_egos.append(g_mut)
                        n_struct_ok += 1
                        break

            if not llm_fn and snap_egos:
                no_llm_result[b_idx] = snap_egos

            if (bi + 1) % 20 == 0 or bi == len(self.benign_indices) - 1:
                print(f"[阶段1] 结构变异 {bi+1}/{len(self.benign_indices)}: "
                      f"成功 {n_struct_ok}/{n_struct_total}", flush=True)

        dt_struct = time.time() - t_struct_start
        print(f"[阶段1] 结构变异完成: {n_struct_ok}个, 耗��� {dt_struct:.1f}s", flush=True)

        if llm_fn is None:
            print(f"[EgoMutPipeline] 完成(无LLM): {sum(len(v) for v in no_llm_result.values())} 变异ego, "
                  f"覆盖 {len(no_llm_result)}/{len(self.benign_indices)} 快照, "
                  f"耗时 {time.time()-t0:.1f}s")
            return no_llm_result

        # ---- 阶段 2: 并发 LLM 语义变异 ----
        print(f"[阶段2] LLM 语义变异: {len(pending)}个ego, {llm_workers}线程并发", flush=True)
        t_llm_start = time.time()

        def _llm_task(item):
            b_idx, g_mut, replaced = item
            try:
                g_out = self._semantic_mutate(g_mut, replaced, llm_fn, model_name=model_name)
                return b_idx, g_out
            except Exception as ex:
                return b_idx, None

        result: Dict[int, list] = {}
        n_llm_ok = 0

        with ThreadPoolExecutor(max_workers=llm_workers) as pool:
            futures = {pool.submit(_llm_task, item): item for item in pending}
            for i, future in enumerate(as_completed(futures)):
                b_idx, g_out = future.result()
                if g_out is not None:
                    result.setdefault(b_idx, []).append(g_out)
                    n_llm_ok += 1
                if (i + 1) % 20 == 0 or i == len(pending) - 1:
                    elapsed = time.time() - t_llm_start
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (len(pending) - i - 1) / rate if rate > 0 else 0
                    print(f"[阶段2] LLM {i+1}/{len(pending)}: "
                          f"成功{n_llm_ok}, {elapsed:.0f}s已用, "
                          f"~{eta:.0f}s剩余 ({rate:.1f}个/s)", flush=True)

        # 截断每个快照的 ego 数量
        for b_idx in list(result.keys()):
            if len(result[b_idx]) > egos_per_snapshot:
                result[b_idx] = result[b_idx][:egos_per_snapshot]

        dt_llm = time.time() - t_llm_start
        total_egos = sum(len(v) for v in result.values())
        print(f"[阶段2] LLM 完成: {n_llm_ok}/{len(pending)}成功, 耗时 {dt_llm:.1f}s")
        print(f"[EgoMutPipeline] 完成: {total_egos} 变异ego, "
              f"覆盖 {len(result)}/{len(self.benign_indices)} 快照, "
              f"总耗时 {time.time()-t0:.1f}s")
        return result
