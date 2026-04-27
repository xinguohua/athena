"""
benchmark_augmentation.py — Table IV: 不同增强策略在 DARPA E3 上的检测性能对比

用法（远程服务器）：
    conda activate prographer
    cd /home/nsas2020/fuzz/prographer

    # 跑全部策略
    python -m process.benchmark_augmentation

    # 只跑单个策略
    python -m process.benchmark_augmentation --strategy no_aug
    python -m process.benchmark_augmentation --strategy graphcl
    python -m process.benchmark_augmentation --strategy gca
    python -m process.benchmark_augmentation --strategy mimicry
    python -m process.benchmark_augmentation --strategy llm_guided

输出：每个策略的 Acc / Prec / F1 / Rec，汇总为 Table IV 格式
"""
import argparse
import os
import sys
import time
import platform
import pickle
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from process.datahandlers import get_handler
from process.embedders import get_embedder_by_name
from process.classfy import get_classfy

# ========================================================================
# 全局配置
# ========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GLOBAL_ID = "bench"  # 基准测试专用 ID，避免覆盖其他人的模型
CLASSIFY_NAME = "mlp"  # 论文方法：两层 MLP + cross-entropy（有监督）

# 数据集：DARPA E3（可通过 --dataset 命令行参数覆盖）
# 可根据需要取消注释以跑多个数据集
DATASET_SCENES = [
    ("cadets", "cadets314"),   # E3-Cadets
    # ("trace", None),         # E3-Trace
    # ("theia", None),         # E3-Theia
    # ("clearscope", None),    # E3-ClearScope
]

# ========================================================================
# 增强策略预设
# ========================================================================
AUGMENT_PRESETS = {
    # 无增强：不做边删除、不做特征掩盖，但使用有监督对比（恶意快照作为负样本）
    "no_aug": {
        "drop_edge_p": 0.0,
        "feat_mask_p": 0.0,
        "use_degree_coop_augment": False,
        "use_malicious_snapshots": True,
        "use_malicious_negatives": False,
        "combine": False,
    },
    # GraphCL [29]：标准随机增强（均匀边删除 + 特征掩盖）
    "graphcl": {
        "drop_edge_p": 0.2,
        "feat_mask_p": 0.2,
        "use_degree_coop_augment": False,  # 关键：使用均匀随机，非度感知
        "use_malicious_snapshots": True,
        "use_malicious_negatives": False,
        "combine": False,
    },
    # GCA [44]：自适应度感知增强
    "gca": {
        "drop_edge_p": 0.2,
        "feat_mask_p": 0.2,
        "use_degree_coop_augment": True,  # 关键：度感知增强
        "use_malicious_snapshots": True,
        "use_malicious_negatives": False,
        "combine": False,
    },
    # Mimicry [32]：将良性边连接到攻击节点，模糊恶意信号
    "mimicry": {
        "drop_edge_p": 0.2,
        "feat_mask_p": 0.2,
        "use_degree_coop_augment": False,
        "use_malicious_snapshots": True,  # 使用恶意快照作为负样本源
        "use_malicious_negatives": False,
        "combine": False,
        "mimicry_mode": True,  # 特殊标志：启用 mimicry 增强
    },
    # LLM-guided mutation（论文方法）：WL kernel 检索 + 结构变异 + 语义变异 + 验证
    "llm_guided": {
        "drop_edge_p": 0.2,
        "feat_mask_p": 0.2,
        "use_degree_coop_augment": True,
        "use_malicious_snapshots": True,
        "use_malicious_negatives": False,
        "combine": False,
        "use_mutation_pipeline": True,
        "llm_model": None,
        "delta_h": 0.1,
        "delta_h_upper": 0.98,
        "n_ensemble": 1,
    },
    # ---- 不同 LLM 后端对比（Table IX: LLM model comparison） ----
    # provider: "chatanywhere" 或 "siliconflow"，决定用哪组 API key/endpoint
    "llm_gpt4o": {
        "drop_edge_p": 0.2, "feat_mask_p": 0.2,
        "use_degree_coop_augment": True, "use_malicious_snapshots": True,
        "use_malicious_negatives": False, "combine": False,
        "use_mutation_pipeline": True,
        "llm_model": "gpt-4o", "llm_provider": "chatanywhere",
    },
    "llm_qwen25_7b": {
        "drop_edge_p": 0.2, "feat_mask_p": 0.2,
        "use_degree_coop_augment": True, "use_malicious_snapshots": True,
        "use_malicious_negatives": False, "combine": False,
        "use_mutation_pipeline": True,
        "llm_model": "Qwen/Qwen2.5-7B-Instruct", "llm_provider": "siliconflow",
    },
    "llm_qwen25_14b": {
        "drop_edge_p": 0.2, "feat_mask_p": 0.2,
        "use_degree_coop_augment": True, "use_malicious_snapshots": True,
        "use_malicious_negatives": False, "combine": False,
        "use_mutation_pipeline": True,
        "llm_model": "Qwen/Qwen2.5-14B-Instruct", "llm_provider": "siliconflow",
    },
    "llm_deepseek_v3": {
        "drop_edge_p": 0.2, "feat_mask_p": 0.2,
        "use_degree_coop_augment": True, "use_malicious_snapshots": True,
        "use_malicious_negatives": False, "combine": False,
        "use_mutation_pipeline": True,
        "llm_model": "deepseek-ai/DeepSeek-V3", "llm_provider": "siliconflow",
        "skip_semantic": True,  # 跳过语义变异，只用结构变异+LLM验证
        "max_mutations": 15, "delta_h": 0.2,
    },
    "llm_glm4_9b": {
        "drop_edge_p": 0.2, "feat_mask_p": 0.2,
        "use_degree_coop_augment": True, "use_malicious_snapshots": True,
        "use_malicious_negatives": False, "combine": False,
        "use_mutation_pipeline": True,
        "llm_model": "THUDM/GLM-4-9B-0414", "llm_provider": "siliconflow",
    },
    # ---- MoE 三策略融合（论文方法改进版）----
    "llm_qwen25_14b_moe": {
        "drop_edge_p": 0.2, "feat_mask_p": 0.2,
        "use_degree_coop_augment": True, "use_malicious_snapshots": True,
        "use_malicious_negatives": False, "combine": False,
        "use_mutation_pipeline": True,
        "llm_model": "Qwen/Qwen2.5-14B-Instruct", "llm_provider": "siliconflow",
        "use_strategy_moe": True,
        "use_multi_strategy": True,
    },
}


class LLMStats:
    """LLM 调用统计（Table VI 数据收集，按快照统计）"""
    def __init__(self):
        self.total_calls = 0
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency = 0.0
        self.n_snapshots = 0  # 处理的快照数

    def record(self, usage: dict, latency: float):
        self.total_calls += 1
        self.total_tokens += usage.get("total_tokens", 0)
        self.total_prompt_tokens += usage.get("prompt_tokens", 0)
        self.total_completion_tokens += usage.get("completion_tokens", 0)
        self.total_latency += latency

    def set_snapshots(self, n: int):
        self.n_snapshots = n

    def summary(self) -> dict:
        n_call = max(self.total_calls, 1)
        n_snap = max(self.n_snapshots, 1)
        return {
            "calls": self.total_calls,
            "n_snapshots": self.n_snapshots,
            "total_tokens": self.total_tokens,
            "tokens_per_snapshot": round(self.total_tokens / n_snap),
            "total_latency_s": round(self.total_latency, 1),
            "latency_per_snapshot": round(self.total_latency / n_snap, 2),
        }

    def print_summary(self, model_name: str = ""):
        s = self.summary()
        print(f"\n[Table VI] LLM 开销统计 ({model_name})")
        print(f"  快照数: {s['n_snapshots']}, LLM 调用次数: {s['calls']}")
        print(f"  总 token: {s['total_tokens']}, 平均 token/快照: {s['tokens_per_snapshot']}")
        print(f"  总延迟: {s['total_latency_s']}s, 平均延迟/快照: {s['latency_per_snapshot']}s")

    def save(self, model_name: str, provider: str = ""):
        """保存 Table VI 数据到 JSON"""
        safe_name = model_name.replace("/", "_")
        path = f"llm_stats_{safe_name}.json"
        s = self.summary()
        s["model"] = model_name
        s["provider"] = provider
        s["prompt_tokens"] = self.total_prompt_tokens
        s["completion_tokens"] = self.total_completion_tokens
        import json
        with open(path, "w") as f:
            json.dump(s, f, indent=2)
        print(f"[Table VI] 已保存: {path}")


# 全局 LLM 统计实例
_llm_stats = LLMStats()


def _make_llm_fn(model_name: str, provider: str = "chatanywhere"):
    """构造 LLM 调用函数（根据 provider 选择 API key/endpoint），自动收集 token/latency"""
    global _llm_stats
    _llm_stats = LLMStats()  # 每次重新计数

    if not model_name:
        return None

    try:
        from process.local_settings import (
            CHATANYWHERE_API_KEY, CHATANYWHERE_ENDPOINT,
            SILICONFLOW_API_KEY, SILICONFLOW_ENDPOINT,
        )
    except ImportError:
        print("[WARN] 未找到 local_settings.py，LLM 语义变异将使用规则 fallback")
        return None

    if provider == "siliconflow":
        api_key, endpoint = SILICONFLOW_API_KEY, SILICONFLOW_ENDPOINT
    else:
        api_key, endpoint = CHATANYWHERE_API_KEY, CHATANYWHERE_ENDPOINT

    from process.llm_clients.chatanywhere_client import chatanywhere_summarize

    print(f"[LLM] 使用 {provider} / {model_name}")

    def llm_fn(prompt: str) -> str:
        t0 = time.time()
        result, usage = chatanywhere_summarize(
            prompt,
            api_key=api_key,
            endpoint=endpoint,
            model=model_name,
            temperature=0.2,
            timeout=60.0,
            return_usage=True,
        )
        _llm_stats.record(usage, time.time() - t0)
        return result
    return llm_fn


def _run_mutation_pipeline(embedder, handler, preset: dict):
    """
    运行 EgoMutationPipeline，在 ego 子图级别生成难负样本。
    结果存入 embedder.mutation_map = {benign_idx: [mutated_ego, ...]}。

    ego 级变异（~32节点）vs 旧的快照级变异（~500节点）：
    攻击节点占比从 ~4% 提升到 ~30-50%，负样本信号清晰。
    """
    from process.mutation.pipeline import EgoMutationPipeline

    print("[LLM-guided] 运行 EgoMutationPipeline（ego 级变异）...")

    pipeline = EgoMutationPipeline(
        snapshots=handler.snapshots,
        benign_range=(handler.benign_idx_start, handler.benign_idx_end),
        attack_range=(handler.malicious_idx_start, handler.malicious_idx_end),
        r_hop=2,
        ego_max_nodes=32,
        top_k=5,
        max_region_size=16,
        use_multi_strategy=preset.get("use_multi_strategy", False),
    )

    llm_model = preset.get("llm_model", None)
    llm_provider = preset.get("llm_provider", "chatanywhere")
    llm_fn = _make_llm_fn(llm_model, provider=llm_provider)

    # 每快照 5 个变异 ego，提供多样性；MLP 采样会过采样到和良性平衡
    mutation_map = pipeline.generate(llm_fn=llm_fn, egos_per_snapshot=5, model_name=llm_model or "no_llm")

    embedder.mutation_map = mutation_map
    n_egos = sum(len(v) for v in mutation_map.values())
    print(f"[LLM-guided] {n_egos} 个 ego 级难负样本，覆盖 {len(mutation_map)} 快照")

    # 保存变异 ego 到文件
    strategy_name = f"{llm_model or 'no_llm'}".replace("/", "_")
    save_path = f"mutation_egos_{strategy_name}.pkl"
    import pickle
    with open(save_path, 'wb') as f:
        pickle.dump(mutation_map, f)
    print(f"[LLM-guided] 变异 ego 已保存: {save_path}")

    # Table VI: LLM 开销统计（打印 + 存文件）
    if llm_model and _llm_stats.total_calls > 0:
        _llm_stats.set_snapshots(len(mutation_map))
        _llm_stats.print_summary(f"{llm_model} ({llm_provider})")
        _llm_stats.save(llm_model, llm_provider)

    # Table V: 变异质量自动评估
    _evaluate_mutation_quality(mutation_map, handler, llm_model or "no_llm")


def _evaluate_mutation_quality(mutation_map: dict, handler, model_name: str):
    """Table V 自动评估：变异 ego 质量（操作合法性、格式、攻击语义保留）"""
    ATK_KEYWORDS = ['gtcache', 'pass_mgr', 'profile (deleted)', '/var/log/wdev', '/tmp/memtrace']
    MALICIOUS_PATTERNS = ['&>/dev/null', '/native-messaging-hosts/', '(deleted)']

    # 收集原始攻击节点 properties 作为对照
    orig_atk_props = set()
    for i in range(handler.malicious_idx_start, handler.malicious_idx_end + 1):
        g = handler.snapshots[i] if i < len(handler.snapshots) else None
        if g is None or g.vcount() == 0:
            continue
        for v in range(g.vcount()):
            if g.vs[v].attributes().get('label', 0) == 1:
                vtype = str(g.vs[v].attributes().get('type', ''))
                if 'PROCESS' in vtype or 'SUBJECT' in vtype:
                    prop = str(g.vs[v].attributes().get('properties', '')).strip().strip("{}'\"")
                    orig_atk_props.add(prop)

    # 分析变异后的进程攻击节点
    total_proc = 0
    unchanged = 0
    format_ok = 0
    metadata_leak = 0
    atk_semantic_kept = 0  # 保留了至少一个恶意行为模式
    atk_keyword_kept = 0   # 保留了攻击关键词

    for _, egos in mutation_map.items():
        for g in egos:
            for v in range(g.vcount()):
                attrs = g.vs[v].attributes()
                if attrs.get('label', 0) != 1:
                    continue
                vtype = str(attrs.get('type', ''))
                if 'PROCESS' not in vtype and 'SUBJECT' not in vtype:
                    continue
                total_proc += 1
                prop = str(attrs.get('properties', '')).strip().strip("{}'\"")

                # 是否未变异
                if prop in orig_atk_props:
                    unchanged += 1

                # 格式检查
                parts = prop.split(",")
                has_tgid = any(p.strip().isdigit() for p in parts[1:] if p.strip())
                if len(parts) >= 3 and has_tgid:
                    format_ok += 1

                # 元数据泄漏
                if any(kw in prop for kw in ['associated_nodes=', 'strategy=', 'C={']):
                    metadata_leak += 1

                # 攻击语义保留
                if any(p in prop for p in MALICIOUS_PATTERNS):
                    atk_semantic_kept += 1
                if any(kw in prop for kw in ATK_KEYWORDS):
                    atk_keyword_kept += 1

    changed = total_proc - unchanged
    print(f"\n[Table V] 变异质量自动评估 ({model_name})")
    print(f"  进程攻击节点: {total_proc}")
    print(f"  未变异(保留原始): {unchanged} ({unchanged*100//max(total_proc,1)}%)")
    print(f"  LLM 变异过: {changed} ({changed*100//max(total_proc,1)}%)")
    print(f"  格式正确率: {format_ok}/{total_proc} ({format_ok*100//max(total_proc,1)}%)")
    print(f"  元数据泄漏: {metadata_leak}/{total_proc} ({metadata_leak*100//max(total_proc,1)}%)")
    print(f"  攻击关键词保留: {atk_keyword_kept}/{total_proc} ({atk_keyword_kept*100//max(total_proc,1)}%)")
    print(f"  恶意行为模式保留: {atk_semantic_kept}/{total_proc} ({atk_semantic_kept*100//max(total_proc,1)}%)")


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    system = platform.system().lower()
    return config["local"] if "windows" in system else config["remote"]


def prepare_data(path_map: dict, dataset_name: str, scene_name: str = None):
    """加载数据集并构建快照（可多策略共享）。优先从缓存加载，跳过耗时的日志解析。"""
    t0 = time.time()
    # 缓存文件名带上数据集和场景名，避免不同数据集共用同一个缓存
    cache_tag = f"{dataset_name}_{scene_name}" if scene_name else dataset_name
    snapshot_file = f"snapshot_data_{GLOBAL_ID}_{cache_tag}.pkl"

    if os.path.exists(snapshot_file):
        print(f"[数据准备] 发现缓存 {snapshot_file}，直接加载...")
        with open(snapshot_file, 'rb') as f:
            snapshot_data = pickle.load(f)
        # 构造一个轻量 handler 对象，只填充下游需要的字段
        handler = get_handler(dataset_name, True, path_map, scene_name=scene_name)
        handler.snapshots = snapshot_data['all_snapshots']
        handler.benign_idx_start = snapshot_data['benign_idx_start']
        handler.benign_idx_end = snapshot_data['benign_idx_end']
        handler.malicious_idx_start = snapshot_data['malicious_idx_start']
        handler.malicious_idx_end = snapshot_data['malicious_idx_end']
        print(f"[数据准备] 从缓存加载 {len(handler.snapshots)} 个快照，耗时: {time.time()-t0:.1f}s")
        return handler

    # 无缓存：完整加载，gid 带数据集名以生成对应的缓存文件
    gid = f"{GLOBAL_ID}_{cache_tag}"
    handler = get_handler(dataset_name, True, path_map, scene_name=scene_name)
    handler.load()
    handler.build_graph(gid)
    print(f"[数据准备] {dataset_name}/{scene_name or 'all'} 耗时: {time.time()-t0:.1f}s")
    return handler


def _set_seed(seed: int = 42):
    """固定所有随机种子"""
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_single_encoder(strategy_name, handler, preset, seed_val):
    """训练单个编码器并返回快照嵌入"""
    _set_seed(seed_val)
    tag = f"{GLOBAL_ID}_{strategy_name}_{seed_val}"

    embedder_cls = get_embedder_by_name("gcc_dev")
    default_model = getattr(embedder_cls, "_default_path", "embedder_model.pth")
    _p = Path(default_model)
    model_path = f"{_p.stem}_{tag}{_p.suffix}"

    train_indices = list(range(handler.benign_idx_start, handler.benign_idx_end + 1))
    embedder_kwargs = {
        "model_path": model_path,
        "drop_edge_p": preset["drop_edge_p"],
        "feat_mask_p": preset["feat_mask_p"],
        "use_degree_coop_augment": preset["use_degree_coop_augment"],
        "use_malicious_snapshots": preset["use_malicious_snapshots"],
        "use_malicious_negatives": preset["use_malicious_negatives"],
        "combine": preset["combine"],
        "train_indices": train_indices,
        "num_epochs": preset.get("num_epochs", 3),
        "attr_weight_alpha": preset.get("attr_weight_alpha", 0.3),
        "use_strategy_moe": preset.get("use_strategy_moe", False),
    }

    embedder = embedder_cls(handler.snapshots, **embedder_kwargs)

    if preset.get("mimicry_mode", False):
        embedder.use_pos_fusion_neg = False
        embedder.mimicry_mode = True

    if preset.get("use_mutation_pipeline", False):
        _run_mutation_pipeline(embedder, handler, preset)

    embedder.train()
    return embedder


def train_with_strategy(strategy_name: str, handler, path_map: dict, seed: int = 42) -> dict:
    """用指定增强策略训练编码器 + 分类器，返回评估指标"""
    preset = AUGMENT_PRESETS[strategy_name]
    tag = f"{GLOBAL_ID}_{strategy_name}"

    print(f"\n{'='*60}")
    print(f" 策略: {strategy_name}")
    print(f"{'='*60}")
    print(f" 参数: {json.dumps(preset, indent=2, ensure_ascii=False)}")

    # ---- 2. 编码器训练（多编码器集成以稳定结果） ----
    t0 = time.time()
    n_ensemble = preset.get("n_ensemble", 1)

    if n_ensemble > 1:
        # 多编码器集成：用不同种子训练多个编码器，平均嵌入
        ensemble_seeds = [seed + i * 1000 for i in range(n_ensemble)]
        all_embs = []
        for es in ensemble_seeds:
            print(f"[{strategy_name}] 集成编码器 seed={es}...")
            emb = _train_single_encoder(strategy_name, handler, preset, es)
            all_embs.append(emb[:len(handler.snapshots)])  # 只取原始快照嵌入
        # 截断到原始长度并平均
        min_len = min(e.shape[0] for e in all_embs)
        snapshot_embeddings = np.mean([e[:min_len] for e in all_embs], axis=0)
        print(f"[{strategy_name}] 集成 {n_ensemble} 个编码器完成")
    else:
        embedder = _train_single_encoder(strategy_name, handler, preset, seed)

    snapshot_embeddings = embedder.get_snapshot_embeddings()
    print(f"[{strategy_name}] 编码器训练耗时: {time.time()-t0:.1f}s")
    print(f"[{strategy_name}] 嵌入维度: {snapshot_embeddings.shape}")

    # ---- 3. Stage 2: MLP（仿 SupCon，实时增强+冻结 encoder） ----
    t0 = time.time()
    mal_start = handler.malicious_idx_start
    mal_end = handler.malicious_idx_end
    ego_cache = embedder.train_ego_cache  # 对比学习采样的 ego = 训练集

    n_attack = sum(1 for _, _, _, lab, _ in ego_cache if lab == 1)
    n_benign = sum(1 for _, _, _, lab, _ in ego_cache if lab == 0)
    feat_dim = embedder.enc_out_dim * 2  # center-aware pooling: [center ‖ mean]
    print(f"[{strategy_name}] Stage 2: {len(ego_cache)} ego子图 (良性={n_benign}, 攻击={n_attack})")

    # MLP 分类器
    classifier = torch.nn.Sequential(
        torch.nn.Linear(feat_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(128, 2),
    ).to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()  # 1:1 平衡后不需要 class_weight
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    embedder.encoder.eval()
    classifier.train()

    BATCH_SIZE = 128
    NUM_EPOCHS = 10
    n_samples = len(ego_cache)

    # 1:1 平衡：下采样良性 + 过采样攻击
    import random as _rng
    benign_indices = [i for i, (_, _, _, lab, _) in enumerate(ego_cache) if lab == 0]
    attack_indices = [i for i, (_, _, _, lab, _) in enumerate(ego_cache) if lab == 1]
    # 良性最多取攻击的 10 倍，避免过多
    max_benign = min(len(benign_indices), max(len(attack_indices) * 10, 500))
    sampled_benign = _rng.sample(benign_indices, max_benign)
    # 攻击过采样匹配良性数量
    balanced_attack = (attack_indices * (max_benign // max(len(attack_indices), 1) + 1))[:max_benign]
    balanced_indices = sampled_benign + balanced_attack
    n_balanced = len(balanced_indices)
    print(f"[{strategy_name}] 平衡采样: {len(sampled_benign)} 良性 + {len(balanced_attack)} 攻击(增强x{len(balanced_attack)//max(len(attack_indices),1)})")

    # 去重：过采样导致同一 ego 在 balanced_indices 中出现多次，
    # 每 epoch 只需对 unique ego 各做一次 augment+encoder forward，再索引复用
    unique_indices = sorted(set(balanced_indices))
    idx_to_pos = {idx: pos for pos, idx in enumerate(unique_indices)}
    balanced_pos = torch.tensor([idx_to_pos[i] for i in balanced_indices], dtype=torch.long)
    unique_labels = torch.tensor(
        [ego_cache[i][3] for i in unique_indices], dtype=torch.long, device=DEVICE)
    print(f"[{strategy_name}] unique ego: {len(unique_indices)} (balanced: {n_balanced})")

    for ep in range(NUM_EPOCHS):
        # 每 epoch 对 unique ego 做一次 augment + encoder forward
        ep_feats = []
        with torch.no_grad():
            for i in unique_indices:
                x_np, ei, ef, lab, _ = ego_cache[i]
                x = torch.from_numpy(x_np).to(DEVICE)
                ei_d, ef_d = ei.to(DEVICE), ef.to(DEVICE)
                x_a, ei_a, ef_a = embedder.augment_ego(
                    x, ei_d, ef_d, drop_edge_p=0.2, feat_mask_p=0.2)
                h = embedder.encoder(x_a, ei_a, edge_feat=ef_a)
                ep_feats.append(torch.cat([h[0], h.mean(dim=0)], dim=0))
        all_feats = torch.stack(ep_feats)  # (n_unique, feat_dim)

        # MLP 训练：索引预计算的嵌入
        perm = torch.randperm(n_balanced)
        ep_loss, n_batches = 0.0, 0

        for start in range(0, n_balanced, BATCH_SIZE):
            batch_perm = perm[start:start + BATCH_SIZE]
            batch_pos = balanced_pos[batch_perm]
            feats = all_feats[batch_pos]
            labels = unique_labels[batch_pos]

            output = classifier(feats)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n_batches += 1

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  [MLP] Epoch {ep+1}/{NUM_EPOCHS} Loss={ep_loss/max(n_batches,1):.4f}")

    print(f"[{strategy_name}] Stage 2 训练耗时: {time.time()-t0:.1f}s")

    # ---- 4. ego 级评估 + 详细 FP/FN 分析 ----
    t0 = time.time()
    classifier.eval()
    embedder.encoder.eval()
    test_cache = embedder.test_ego_cache

    # 逐 ego 预测，记录 ego 自身特征
    ego_preds = []  # [(lab, pred, n_nodes, n_edges, edge_cats, vtype, prop, sidx, center_vid)]
    with torch.no_grad():
        for entry in test_cache:
            x_np, ei, ef, lab = entry[0], entry[1], entry[2], entry[3]
            sidx = entry[4] if len(entry) > 4 else -1
            vtype = entry[5] if len(entry) > 5 else ''
            prop = entry[6] if len(entry) > 6 else ''
            center_vid = entry[7] if len(entry) > 7 else -1
            x = torch.from_numpy(x_np).to(DEVICE)
            ei_d, ef_d = ei.to(DEVICE), ef.to(DEVICE)
            h = embedder.encoder(x, ei_d, edge_feat=ef_d)
            # center-aware pooling: [center_node ‖ mean_all]（与训练一致）
            feat = torch.cat([h[0], h.mean(dim=0)], dim=0)
            pred = classifier(feat.unsqueeze(0)).argmax(dim=1).item()
            # ego 特征：节点数、边数、边类型分布
            n_nodes = x_np.shape[0]
            n_edges = ei.shape[1] // 2 if ei.numel() > 0 else 0
            # 边类型统计（从 ef 的整数类别）
            if ef.numel() > 0:
                cats = ef.numpy().tolist()
                edge_cats = {0: 0, 1: 0, 2: 0, 3: 0}  # proc/file/net/mem
                for c in cats:
                    if c in edge_cats:
                        edge_cats[c] += 1
            else:
                edge_cats = {0: 0, 1: 0, 2: 0, 3: 0}
            ego_preds.append((lab, pred, n_nodes, n_edges, edge_cats, vtype, prop, sidx, center_vid))

    true_labels = np.array([r[0] for r in ego_preds], dtype=int)
    pred_labels = np.array([r[1] for r in ego_preds], dtype=int)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(true_labels, pred_labels) * 100
    prec = precision_score(true_labels, pred_labels, zero_division=0) * 100
    rec = recall_score(true_labels, pred_labels, zero_division=0) * 100
    f1 = f1_score(true_labels, pred_labels, zero_division=0) * 100
    tp = int(np.sum((true_labels == 1) & (pred_labels == 1)))
    fp = int(np.sum((true_labels == 0) & (pred_labels == 1)))
    tn = int(np.sum((true_labels == 0) & (pred_labels == 0)))
    fn = int(np.sum((true_labels == 1) & (pred_labels == 0)))

    print(f"\n--- [{strategy_name}] ego 级评估 ---")
    print(f"  Acc={acc:.2f}% Prec={prec:.2f}% Rec={rec:.2f}% F1={f1:.2f}%")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")

    # FN 详情
    if fn > 0:
        print(f"\n  FN ({fn}个攻击 ego 漏检):")
        for lab, pred, nn, ne, ec, vt, pr, si, cv in ego_preds:
            if lab == 1 and pred == 0:
                print(f"    snap[{si}] {vt} | {pr[:50]} | nodes={nn} proc={ec[0]} file={ec[1]} net={ec[2]} mem={ec[3]}")

    # FP 详情（取前 20 个样例 + 统计）
    if fp > 0:
        from collections import Counter
        fp_types = Counter()
        fp_props = Counter()
        print(f"\n  FP ({fp}个良性 ego 误判):")
        for lab, pred, nn, ne, ec, vt, pr, si, cv in ego_preds:
            if lab == 0 and pred == 1:
                fp_types[vt] += 1
                fp_props[pr[:30]] += 1
                print(f"    snap[{si}] {vt} | {pr[:50]} | nodes={nn} proc={ec[0]} file={ec[1]} net={ec[2]} mem={ec[3]}")

        print(f"\n  FP 按节点类型:")
        for vt, c in fp_types.most_common():
            total = sum(1 for l, p, *_ in ego_preds if l == 0 and ego_preds[ego_preds.index((l, p, *_))][-2] == vt)
            print(f"    {vt}: {c}")

        print(f"\n  FP 最常见 properties (top10):")
        for pr, c in fp_props.most_common(10):
            print(f"    {c:4d}x {pr}")

    if tp > 0:
        print(f"\n  TP ({tp}个攻击 ego 检出，全部):")
        for lab, pred, nn, ne, ec, vt, pr, si, cv in ego_preds:
            if lab == 1 and pred == 1:
                print(f"    snap[{si}] {vt} | {pr[:50]} | nodes={nn} proc={ec[0]} file={ec[1]} net={ec[2]} mem={ec[3]}")

    # ---- 5. 快照级评估：任意 ego 预测为攻击 → 快照为恶意 ----
    from collections import defaultdict
    snap_true = defaultdict(int)   # sidx → 1 if any ego label=1
    snap_pred = defaultdict(int)   # sidx → 1 if any ego pred=1
    for lab, pred, nn, ne, ec, vt, pr, si, cv in ego_preds:
        if lab == 1:
            snap_true[si] = 1
        if pred == 1:
            snap_pred[si] = 1

    all_sids = sorted(set(e[7] for e in ego_preds))  # sidx 在第 7 位
    s_tp = sum(1 for s in all_sids if snap_true.get(s, 0) == 1 and snap_pred.get(s, 0) == 1)
    s_fp = sum(1 for s in all_sids if snap_true.get(s, 0) == 0 and snap_pred.get(s, 0) == 1)
    s_tn = sum(1 for s in all_sids if snap_true.get(s, 0) == 0 and snap_pred.get(s, 0) == 0)
    s_fn = sum(1 for s in all_sids if snap_true.get(s, 0) == 1 and snap_pred.get(s, 0) == 0)
    s_total = s_tp + s_fp + s_tn + s_fn
    s_acc = 100 * (s_tp + s_tn) / max(s_total, 1)
    s_prec = 100 * s_tp / max(s_tp + s_fp, 1)
    s_rec = 100 * s_tp / max(s_tp + s_fn, 1)
    s_f1 = 100 * 2 * s_tp / max(2 * s_tp + s_fp + s_fn, 1)

    print(f"\n--- [{strategy_name}] 快照级评估 (any-ego-positive) ---")
    print(f"  Acc={s_acc:.2f}% Prec={s_prec:.2f}% Rec={s_rec:.2f}% F1={s_f1:.2f}%")
    print(f"  TP={s_tp} FP={s_fp} TN={s_tn} FN={s_fn} (共{s_total}个快照)")

    # FP 快照详情
    if s_fp > 0:
        print(f"\n  FP 快照 ({s_fp}个良性快照误判):")
        for s in all_sids:
            if snap_true.get(s, 0) == 0 and snap_pred.get(s, 0) == 1:
                n_ego = sum(1 for e in ego_preds if e[7] == s)
                n_fp_ego = sum(1 for e in ego_preds if e[7] == s and e[0] == 0 and e[1] == 1)
                print(f"    snap[{s}]: {n_fp_ego}/{n_ego} ego 误判")

    # FN 快照详情
    if s_fn > 0:
        print(f"\n  FN 快照 ({s_fn}个恶意快照漏检):")
        for s in all_sids:
            if snap_true.get(s, 0) == 1 and snap_pred.get(s, 0) == 0:
                n_att = sum(1 for e in ego_preds if e[7] == s and e[0] == 1)
                print(f"    snap[{s}]: {n_att}个攻击ego全部漏检")

    # ---- 6. 保存 FP/FN/TP 详情到 JSON，含完整 ego 节点信息 ----
    from collections import deque as _deque
    snapshots = embedder.snapshots

    def _get_ego_graph_info(sid, center_vid):
        """从原图提取 ego 子图的完整图信息（节点+边）"""
        g = snapshots[sid] if 0 <= sid < len(snapshots) else None
        if g is None or g.vcount() == 0 or center_vid < 0:
            return [], []
        # BFS 重建 ego 节点列表
        visited = [center_vid]
        visited_set = {center_vid}
        queue = _deque([center_vid])
        max_nodes = embedder.ego_max_nodes
        while queue and len(visited) < max_nodes:
            v = queue.popleft()
            for nb in g.neighbors(v, mode="all"):
                if nb not in visited_set and len(visited) < max_nodes:
                    visited.append(nb)
                    visited_set.add(nb)
                    queue.append(nb)
        nodes_info = []
        for v in visited:
            nodes_info.append({
                "vid": v,
                "type": str(g.vs[v].attributes().get('type', '')),
                "label": int(g.vs[v].attributes().get('label', 0)),
                "prop": str(g.vs[v].attributes().get('properties', ''))[:100],
            })
        # 提取子图内的边
        edges_info = []
        vid_set = set(visited)
        for ei in range(g.ecount()):
            s, t = g.es[ei].source, g.es[ei].target
            if s in vid_set and t in vid_set:
                edges_info.append({
                    "src": s, "dst": t,
                    "action": str(g.es[ei].attributes().get('actions', ''))[:60],
                })
        return nodes_info, edges_info

    analysis_data = {
        "strategy": strategy_name,
        "metrics": {"ego_acc": acc, "ego_prec": prec, "ego_rec": rec, "ego_f1": f1,
                     "snap_acc": s_acc, "snap_prec": s_prec, "snap_rec": s_rec, "snap_f1": s_f1},
        "FP": [], "FN": [], "TP": [],
    }
    for lab, pred, nn, ne, ec, vt, pr, si, cv in ego_preds:
        if not ((lab == 0 and pred == 1) or (lab == 1 and pred == 0) or (lab == 1 and pred == 1)):
            continue
        ego_nodes, ego_edges = _get_ego_graph_info(si, cv)
        n_attack_in_ego = sum(1 for n in ego_nodes if n["label"] == 1)
        entry = {
            "snap": si, "center_vid": cv, "center_type": vt, "center_prop": pr,
            "n_nodes": nn, "n_edges": ne,
            "proc": ec[0], "file": ec[1], "net": ec[2], "mem": ec[3],
            "n_attack_in_ego": n_attack_in_ego,
            "ego_graph": {"nodes": ego_nodes, "edges": ego_edges},
        }
        if lab == 0 and pred == 1:
            analysis_data["FP"].append(entry)
        elif lab == 1 and pred == 0:
            analysis_data["FN"].append(entry)
        elif lab == 1 and pred == 1:
            analysis_data["TP"].append(entry)

    analysis_path = f"analysis_{strategy_name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    print(f"\n  分析数据已保存: {analysis_path} (FP={len(analysis_data['FP'])} FN={len(analysis_data['FN'])} TP={len(analysis_data['TP'])})")

    metrics = {"strategy": strategy_name,
               "ego_acc": acc, "ego_prec": prec, "ego_f1": f1, "ego_rec": rec,
               "ego_tp": tp, "ego_fp": fp, "ego_tn": tn, "ego_fn": fn,
               "snap_acc": s_acc, "snap_prec": s_prec, "snap_f1": s_f1, "snap_rec": s_rec,
               "snap_tp": s_tp, "snap_fp": s_fp, "snap_tn": s_tn, "snap_fn": s_fn}
    return metrics


def evaluate_strategy(strategy_name: str, classify,
                      test_embeddings: np.ndarray,
                      true_labels: np.ndarray) -> dict:
    """评估单个策略，返回指标字典"""
    if len(true_labels) == 0:
        print(f"[{strategy_name}] 无测试数据，跳过评估")
        return {}

    # 使用分类器预测（返回 (pred_labels, details)）
    result = classify.predict(test_embeddings)
    if isinstance(result, tuple):
        pred_labels = result[0]
    else:
        pred_labels = result
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()
    pred_labels = np.asarray(pred_labels, dtype=int)

    # 计算指标
    acc = accuracy_score(true_labels, pred_labels) * 100
    prec = precision_score(true_labels, pred_labels, zero_division=0) * 100
    rec = recall_score(true_labels, pred_labels, zero_division=0) * 100
    f1 = f1_score(true_labels, pred_labels, zero_division=0) * 100
    fpr = _false_positive_rate(true_labels, pred_labels) * 100

    tp = int(np.sum((true_labels == 1) & (pred_labels == 1)))
    fp = int(np.sum((true_labels == 0) & (pred_labels == 1)))
    tn = int(np.sum((true_labels == 0) & (pred_labels == 0)))
    fn = int(np.sum((true_labels == 1) & (pred_labels == 0)))

    metrics = {
        "strategy": strategy_name,
        "acc": acc, "prec": prec, "f1": f1, "rec": rec, "fpr": fpr,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }

    print(f"\n--- [{strategy_name}] 评估结果 ---")
    print(f"  Acc:  {acc:.2f}%")
    print(f"  Prec: {prec:.2f}%")
    print(f"  F1:   {f1:.2f}%")
    print(f"  Rec:  {rec:.2f}%")
    print(f"  FPR:  {fpr:.2f}%")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")

    return metrics


def _false_positive_rate(y_true, y_pred):
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def print_summary_table(all_results: list):
    """汇总打印 Table IV 格式（ego 级 + 快照级）"""
    print("\n" + "=" * 90)
    print(" TABLE IV: Detection Performance of Different Augmentation Strategies")
    print("=" * 90)
    print(f"{'Strategy':<16} | {'--- Ego 级 ---':^36} | {'--- 快照级 ---':^36}")
    print(f"{'':16} | {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} | {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 90)
    for r in all_results:
        ea = r.get('ego_acc', r.get('acc', 0))
        ep = r.get('ego_prec', r.get('prec', 0))
        er = r.get('ego_rec', r.get('rec', 0))
        ef = r.get('ego_f1', r.get('f1', 0))
        sa = r.get('snap_acc', 0)
        sp = r.get('snap_prec', 0)
        sr = r.get('snap_rec', 0)
        sf = r.get('snap_f1', 0)
        print(f"{r['strategy']:<16} | {ea:6.2f}% {ep:6.2f}% {er:6.2f}% {ef:6.2f}% | {sa:6.2f}% {sp:6.2f}% {sr:6.2f}% {sf:6.2f}%")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Table IV 增强策略基准测试")
    parser.add_argument(
        "--strategy", type=str, default=None,
        choices=list(AUGMENT_PRESETS.keys()),
        help="只跑指定策略；不传则跑全部"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="数据集名（如 cadets, trace），不传则使用默认列表"
    )
    parser.add_argument(
        "--scene", type=str, default=None,
        help="场景名（如 cadets314），不传则使用默认配置"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子"
    )
    args = parser.parse_args()

    env_config = load_config()
    path_map = env_config["path_map"]

    # 确定要跑的策略
    if args.strategy:
        strategies = [args.strategy]
    else:
        strategies = list(AUGMENT_PRESETS.keys())

    # 确定数据集
    if args.dataset:
        scenes = [(args.dataset, args.scene)]
    else:
        scenes = DATASET_SCENES

    all_results = []

    for dataset_name, scene_name in scenes:
        # 每个数据集只加载一次
        handler = prepare_data(path_map, dataset_name, scene_name)
        ds_label = f"{dataset_name}/{scene_name or 'all'}"

        for strat in strategies:
            try:
                metrics = train_with_strategy(strat, handler, path_map, seed=args.seed)
                if metrics:
                    metrics["dataset"] = ds_label
                    all_results.append(metrics)
            except Exception as e:
                print(f"\n[ERROR] 策略 {strat} 在 {ds_label} 上失败: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "strategy": strat,
                    "dataset": ds_label,
                    "acc": 0, "prec": 0, "f1": 0, "rec": 0, "fpr": 0,
                    "error": str(e),
                })

    # 汇总输出
    print_summary_table(all_results)

    # 保存 JSON 结果
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = f"table4_results_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")

    # 保存汇总 txt（便于直接查看）
    txt_path = f"table4_results_{ts}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Table IV: Detection Performance of Different Augmentation Strategies\n")
        f.write(f"Dataset: {scenes}\n")
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 120 + "\n")
        f.write(f"{'Strategy':<16} | {'Ego_Acc':>8} {'Ego_Prec':>9} {'Ego_Rec':>8} {'Ego_F1':>8}"
                f" | {'Snap_Acc':>9} {'Snap_Prec':>10} {'Snap_Rec':>9} {'Snap_F1':>8}"
                f" | {'TP':>4} {'FP':>4} {'TN':>5} {'FN':>4}\n")
        f.write("-" * 120 + "\n")
        for r in all_results:
            ea = r.get('ego_acc', r.get('acc', 0))
            ep = r.get('ego_prec', r.get('prec', 0))
            er = r.get('ego_rec', r.get('rec', 0))
            ef = r.get('ego_f1', r.get('f1', 0))
            sa = r.get('snap_acc', 0)
            sp = r.get('snap_prec', 0)
            sr = r.get('snap_rec', 0)
            sf = r.get('snap_f1', 0)
            tp = r.get('ego_tp', r.get('tp', 0))
            fp = r.get('ego_fp', r.get('fp', 0))
            tn = r.get('ego_tn', r.get('tn', 0))
            fn = r.get('ego_fn', r.get('fn', 0))
            err = r.get('error', '')
            if err:
                f.write(f"{r['strategy']:<16} | ERROR: {err}\n")
            else:
                f.write(f"{r['strategy']:<16} | {ea:7.2f}% {ep:8.2f}% {er:7.2f}% {ef:7.2f}%"
                        f" | {sa:8.2f}% {sp:9.2f}% {sr:8.2f}% {sf:7.2f}%"
                        f" | {tp:4d} {fp:4d} {tn:5d} {fn:4d}\n")
        f.write("=" * 120 + "\n")
    print(f"汇总 TXT 已保存: {txt_path}")


if __name__ == "__main__":
    # 同时输出到终端与日志文件
    log_name = f"bench_aug_{time.strftime('%Y%m%d-%H%M%S')}.log"

    class _Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, s):
            for f in self.files:
                f.write(s)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    logf = open(log_name, 'a', encoding='utf-8')
    sys.stdout = _Tee(sys.stdout, logf)
    sys.stderr = _Tee(sys.stderr, logf)
    print(f"[Log] writing to {log_name}")

    main()
