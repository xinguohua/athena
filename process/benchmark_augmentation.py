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
}


def _make_llm_fn(model_name: str, provider: str = "chatanywhere"):
    """构造 LLM 调用函数（根据 provider 选择 API key/endpoint）"""
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
        return chatanywhere_summarize(
            prompt,
            api_key=api_key,
            endpoint=endpoint,
            model=model_name,
            temperature=0.2,
            timeout=60.0,
        )
    return llm_fn


def _run_mutation_pipeline(embedder, handler, preset: dict):
    """
    运行 MutationPipeline，为每个良性快照生成专属难负样本。
    结果存入 embedder.mutation_map = {benign_idx: mutated_graph}。
    训练时每个良性快照的负样本集 = 攻击图(共享池) + G̃_b(专属变异图)。
    """
    from process.mutation import MutationPipeline

    print("[LLM-guided] 运行 MutationPipeline...")

    delta_h = preset.get("delta_h", 0.3)
    delta_h_upper = preset.get("delta_h_upper", 0.95)

    pipeline = MutationPipeline(
        snapshots=handler.snapshots,
        benign_range=(handler.benign_idx_start, handler.benign_idx_end),
        attack_range=(handler.malicious_idx_start, handler.malicious_idx_end),
        delta_h=delta_h,
        delta_h_upper=delta_h_upper,
        top_k=5,
        top_m=3,
    )

    llm_model = preset.get("llm_model", None)
    llm_provider = preset.get("llm_provider", "chatanywhere")
    llm_fn = _make_llm_fn(llm_model, provider=llm_provider)

    mutation_map = pipeline.generate(
        llm_fn=llm_fn,
        skip_verification=False,
    )

    # 变异图不进共享 pool，只作为专属难负样本
    # mutation_map 直接存图对象，训练时按 sidx 查找
    embedder.mutation_map = mutation_map
    print(f"[LLM-guided] {len(mutation_map)} 个良性快照有专属难负样本（不进共享pool）")


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
    }

    embedder = embedder_cls(handler.snapshots, **embedder_kwargs)

    if preset.get("mimicry_mode", False):
        embedder.use_pos_fusion_neg = False
        embedder.mimicry_mode = True

    if preset.get("use_mutation_pipeline", False):
        _run_mutation_pipeline(embedder, handler, preset)

    embedder.train()
    return embedder.get_snapshot_embeddings()


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
        snapshot_embeddings = _train_single_encoder(strategy_name, handler, preset, seed)

    print(f"[{strategy_name}] 编码器训练耗时: {time.time()-t0:.1f}s")
    print(f"[{strategy_name}] 嵌入维度: {snapshot_embeddings.shape}")

    # ---- 3. 分类器训练（有监督：MLP + cross-entropy） ----
    t0 = time.time()
    benign_embeddings = snapshot_embeddings[handler.benign_idx_start:handler.benign_idx_end + 1]

    # 恶意区间的嵌入和真实标签（用于有监督训练）
    mal_start = handler.malicious_idx_start
    mal_end = handler.malicious_idx_end
    mal_snapshots = handler.snapshots[mal_start: mal_end + 1]
    mal_embeddings = snapshot_embeddings[mal_start: mal_end + 1]
    mal_labels = np.array([
        int(any(v["label"] == 1 for v in g.vs))
        for g in mal_snapshots
    ], dtype=int)

    # 划分训练/测试：分层随机采样，确保正负样本在训练和测试集中都有分布
    from sklearn.model_selection import StratifiedShuffleSplit
    n_mal = len(mal_labels)
    n_pos = int(mal_labels.sum())
    print(f"[{strategy_name}] 恶意范围: {n_mal} 快照, 含攻击={n_pos}")

    if n_pos >= 2:
        # 分层采样：30% 训练，70% 测试
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.7, random_state=42)
        train_idx, test_idx = next(sss.split(mal_embeddings, mal_labels))
    else:
        # 正样本太少，退回顺序划分
        n_train_mal = max(1, int(n_mal * 0.3))
        train_idx = list(range(n_train_mal))
        test_idx = list(range(n_train_mal, n_mal))

    train_mal_emb = mal_embeddings[train_idx]
    train_mal_labels = mal_labels[train_idx]
    test_mal_emb = mal_embeddings[test_idx]
    test_mal_labels = mal_labels[test_idx]
    print(f"[{strategy_name}] 训练集: {len(train_idx)} (攻击={int(train_mal_labels.sum())}), "
          f"测试集: {len(test_idx)} (攻击={int(test_mal_labels.sum())})")

    classify = get_classfy(CLASSIFY_NAME, gid=tag)
    classify.train(benign_embeddings, train_mal_emb, train_mal_labels)
    print(f"[{strategy_name}] 分类器训练耗时: {time.time()-t0:.1f}s")

    # ---- 4. 评估 ----
    t0 = time.time()
    metrics = evaluate_strategy(
        strategy_name, classify, test_mal_emb, test_mal_labels
    )
    print(f"[{strategy_name}] 评估耗时: {time.time()-t0:.1f}s")

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
    """汇总打印 Table IV 格式"""
    print("\n" + "=" * 70)
    print(" TABLE IV: Detection Performance of Different Augmentation Strategies")
    print("=" * 70)
    print(f"{'Strategy':<16} {'Acc (%)':<10} {'Prec (%)':<10} {'F1 (%)':<10} {'Rec (%)':<10}")
    print("-" * 56)
    for r in all_results:
        print(f"{r['strategy']:<16} {r['acc']:<10.2f} {r['prec']:<10.2f} {r['f1']:<10.2f} {r['rec']:<10.2f}")
    print("=" * 70)


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
    out_path = f"table4_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")


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
