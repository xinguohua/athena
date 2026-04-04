"""
MutationPipeline: 为每个良性快照生成专属难负样本 G̃_b

训练时负样本集 N(b) = 攻击图(共享) + G̃_b(专属)，加权对比损失自动放大难负样本。
验证不通过换攻击图重试，直到成功或穷尽候选。
"""
from __future__ import annotations
from typing import List, Tuple, Optional, Callable, Set, Dict
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
