"""
MutationPipeline: 完整的 LLM-guided graph mutation 流水线

整合结构变异、语义变异、验证三个阶段，生成用于对比学习的难负样本。

用法:
    pipeline = MutationPipeline(snapshots, benign_range, attack_range)
    mutated_graphs = pipeline.generate(llm_fn=my_llm_fn, max_mutations=100)
"""
from __future__ import annotations
from typing import List, Tuple, Optional, Callable, Set, Dict
import time

try:
    import igraph as ig
except ImportError:
    ig = None

from .structural import generate_structural_mutations, aligned_region_search, subgraph_replacement
from .semantic import apply_semantic_mutation_llm, _collect_benign_corpus
from .verification import build_historical_profiles, verify_mutation


class MutationPipeline:
    """
    论文 Section IV-C 的完整变异流水线。

    Attributes:
        snapshots: 全部快照列表（igraph.Graph）
        benign_range: (start, end) 良性快照索引范围
        attack_range: (start, end) 恶意快照索引范围
        delta_h: hardness 检查阈值
        top_k: 每个良性图检索的攻击图数
        top_m: 每个攻击图的候选区域数
    """

    def __init__(
        self,
        snapshots: list,
        benign_range: Tuple[int, int],
        attack_range: Tuple[int, int],
        delta_h: float = 0.5,
        top_k: int = 5,
        top_m: int = 3,
        max_region_size: int = 32,
    ):
        self.snapshots = snapshots
        self.benign_range = benign_range
        self.attack_range = attack_range
        self.delta_h = delta_h
        self.top_k = top_k
        self.top_m = top_m
        self.max_region_size = max_region_size

        # 预构建良性/攻击图列表
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

        # 预构建良性语料和历史行为剖面
        self.benign_commands: Set[str] = set()
        self.benign_args: Set[str] = set()
        self.benign_files: Set[str] = set()
        self.entity_ops: Dict[str, Set[str]] = {}
        self.type_attrs: Dict[str, Set[str]] = {}

        self._build_profiles()

    def _build_profiles(self):
        """预构建良性行为剖面"""
        t0 = time.time()

        # 良性语料（用于语义变异策略分配）
        self.benign_commands, self.benign_args, self.benign_files = \
            _collect_benign_corpus(self.benign_graphs)

        # 历史行为剖面（用于验证）
        self.entity_ops, self.type_attrs = \
            build_historical_profiles(self.benign_graphs)

        print(f"[MutPipeline] 良性剖面构建完成: "
              f"{len(self.benign_commands)} 命令, "
              f"{len(self.benign_args)} 参数, "
              f"{len(self.benign_files)} 文件, "
              f"耗时 {time.time()-t0:.1f}s")

    def generate(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        max_mutations: int = 100,
        skip_verification: bool = False,
    ) -> List[Tuple]:
        """
        执行完整变异流水线。

        Args:
            llm_fn: LLM 调用函数 (prompt) -> response_text
                    若为 None，语义变异将使用规则 fallback
            max_mutations: 最大变异图数量
            skip_verification: 是否跳过验证（调试用）

        Returns:
            [(mutated_graph, benign_idx, attack_idx), ...]
            通过验证的变异图列表
        """
        print(f"\n[MutPipeline] 开始变异生成: "
              f"良性图={len(self.benign_graphs)}, "
              f"攻击图={len(self.attack_graphs)}, "
              f"LLM={'有' if llm_fn else '无(规则)'}")

        t0 = time.time()

        # ---- 阶段 1: 结构变异 ----
        print("[MutPipeline] 阶段1: 结构变异...")
        t1 = time.time()
        struct_mutations = generate_structural_mutations(
            self.benign_graphs,
            self.attack_graphs,
            top_k=self.top_k,
            top_m=self.top_m,
            max_region_size=self.max_region_size,
            max_mutations=max_mutations * 3,  # 过量生成，后面验证会筛掉一部分
        )
        print(f"[MutPipeline] 结构变异完成: {len(struct_mutations)} 个, "
              f"耗时 {time.time()-t1:.1f}s")

        # ---- 阶段 2: 语义变异 ----
        print("[MutPipeline] 阶段2: 语义变异...")
        t2 = time.time()
        sem_mutations = []
        for g_mut, b_idx, a_idx in struct_mutations:
            if len(sem_mutations) >= max_mutations * 2:
                break

            # 找出被替换（攻击）的节点
            attack_nodes = []
            for v_idx in range(g_mut.vcount()):
                if g_mut.vs[v_idx].get("label", 0) == 1:
                    attack_nodes.append(v_idx)

            # 语义变异
            g_sem = apply_semantic_mutation_llm(
                g_mut,
                attack_nodes,
                self.benign_commands,
                self.benign_args,
                llm_fn=llm_fn,
                r_hop=2,
            )

            if g_sem is not None:
                sem_mutations.append((g_sem, b_idx, a_idx, set(attack_nodes)))

        print(f"[MutPipeline] 语义变异完成: {len(sem_mutations)} 个, "
              f"耗时 {time.time()-t2:.1f}s")

        # ---- 阶段 3: 验证 ----
        if skip_verification:
            verified = [(g, bi, ai) for g, bi, ai, _ in sem_mutations[:max_mutations]]
            print(f"[MutPipeline] 跳过验证, 保留 {len(verified)} 个")
        else:
            print("[MutPipeline] 阶段3: 统一验证...")
            t3 = time.time()
            verified = []
            n_checked = 0
            n_passed = 0

            for g_sem, b_idx, a_idx, replaced in sem_mutations:
                if len(verified) >= max_mutations:
                    break

                n_checked += 1
                g_anchor = self.snapshots[b_idx]

                passed, failed = verify_mutation(
                    g_sem, g_anchor, replaced,
                    self.entity_ops, self.type_attrs,
                    delta_h=self.delta_h,
                )

                if passed:
                    n_passed += 1
                    verified.append((g_sem, b_idx, a_idx))

            print(f"[MutPipeline] 验证完成: {n_checked} 检查, "
                  f"{n_passed} 通过 ({n_passed/max(n_checked,1)*100:.0f}%), "
                  f"耗时 {time.time()-t3:.1f}s")

        total_time = time.time() - t0
        print(f"[MutPipeline] 总计生成 {len(verified)} 个变异图, "
              f"总耗时 {total_time:.1f}s")

        return verified
