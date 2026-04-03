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
    # 无增强：不做边删除、不做特征掩盖、不使用恶意样本
    "no_aug": {
        "drop_edge_p": 0.0,
        "feat_mask_p": 0.0,
        "use_degree_coop_augment": False,
        "use_malicious_snapshots": False,
        "use_malicious_negatives": False,
        "combine": False,
    },
    # GraphCL [29]：标准随机增强（均匀边删除 + 特征掩盖）
    "graphcl": {
        "drop_edge_p": 0.2,
        "feat_mask_p": 0.2,
        "use_degree_coop_augment": False,  # 关键：使用均匀随机，非度感知
        "use_malicious_snapshots": False,
        "use_malicious_negatives": False,
        "combine": False,
    },
    # GCA [44]：自适应度感知增强
    "gca": {
        "drop_edge_p": 0.2,
        "feat_mask_p": 0.2,
        "use_degree_coop_augment": True,  # 关键：度感知增强
        "use_malicious_snapshots": False,
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
        "use_mutation_pipeline": True,  # 启用完整变异流水线
        "llm_model": None,  # None=规则 fallback, "gpt-4o"/"llama3-7b"/...
    },
}


def _make_llm_fn(model_name: str):
    """构造 LLM 调用函数（根据模型名选择后端）"""
    if not model_name:
        return None

    try:
        from process.local_settings import CHATANYWHERE_API_KEY, CHATANYWHERE_ENDPOINT
    except ImportError:
        print("[WARN] 未找到 local_settings.py，LLM 语义变异将使用规则 fallback")
        return None

    from process.llm_clients.chatanywhere_client import chatanywhere_summarize

    def llm_fn(prompt: str) -> str:
        return chatanywhere_summarize(
            prompt,
            api_key=CHATANYWHERE_API_KEY,
            endpoint=CHATANYWHERE_ENDPOINT,
            model=model_name,
            temperature=0.2,
            timeout=60.0,
        )
    return llm_fn


def _run_mutation_pipeline(embedder, handler, preset: dict):
    """
    运行 MutationPipeline 生成变异图，并将变异图注入到 embedder 的恶意 ego 池中。
    """
    from process.mutation import MutationPipeline

    print("[LLM-guided] 运行 MutationPipeline 生成变异图...")

    pipeline = MutationPipeline(
        snapshots=handler.snapshots,
        benign_range=(handler.benign_idx_start, handler.benign_idx_end),
        attack_range=(handler.malicious_idx_start, handler.malicious_idx_end),
        delta_h=0.5,
        top_k=5,
        top_m=3,
    )

    llm_model = preset.get("llm_model", None)
    llm_fn = _make_llm_fn(llm_model)

    mutated = pipeline.generate(
        llm_fn=llm_fn,
        max_mutations=50,
        skip_verification=(llm_fn is None),  # 无 LLM 时跳过验证
    )

    if mutated:
        # 将变异图注入 embedder 的快照列表末尾，作为额外的恶意样本
        # 这些图带有 label=1 的攻击节点，会被 embedder 用作负样本
        n_before = len(embedder.snapshots)
        for g_mut, b_idx, a_idx in mutated:
            embedder.snapshots.append(g_mut)
        print(f"[LLM-guided] 注入 {len(mutated)} 个变异图到快照列表 "
              f"(索引 {n_before}~{n_before + len(mutated) - 1})")
    else:
        print("[LLM-guided] 未生成变异图，将使用原始恶意快照")


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    system = platform.system().lower()
    return config["local"] if "windows" in system else config["remote"]


def prepare_data(path_map: dict, dataset_name: str, scene_name: str = None):
    """加载数据集并构建快照（可多策略共享）"""
    t0 = time.time()
    handler = get_handler(dataset_name, True, path_map, scene_name=scene_name)
    handler.load()
    handler.build_graph(GLOBAL_ID)
    print(f"[数据准备] {dataset_name}/{scene_name or 'all'} 耗时: {time.time()-t0:.1f}s")
    return handler


def train_with_strategy(strategy_name: str, handler, path_map: dict) -> dict:
    """用指定增强策略训练编码器 + 分类器，返回评估指标"""
    preset = AUGMENT_PRESETS[strategy_name]
    tag = f"{GLOBAL_ID}_{strategy_name}"  # 每个策略独立的模型文件后缀

    print(f"\n{'='*60}")
    print(f" 策略: {strategy_name}")
    print(f"{'='*60}")
    print(f" 参数: {json.dumps(preset, indent=2, ensure_ascii=False)}")

    # ---- 2. 编码器训练 ----
    t0 = time.time()
    embedder_cls = get_embedder_by_name("gcc_dev")
    default_model = getattr(embedder_cls, "_default_path", "embedder_model.pth")
    _p = Path(default_model)
    model_path = f"{_p.stem}_{tag}{_p.suffix}"

    # 构建编码器参数：基础参数 + 策略预设
    embedder_kwargs = {
        "model_path": model_path,
        "drop_edge_p": preset["drop_edge_p"],
        "feat_mask_p": preset["feat_mask_p"],
        "use_degree_coop_augment": preset["use_degree_coop_augment"],
        "use_malicious_snapshots": preset["use_malicious_snapshots"],
        "use_malicious_negatives": preset["use_malicious_negatives"],
        "combine": preset["combine"],
    }

    embedder = embedder_cls(handler.snapshots, **embedder_kwargs)

    # Mimicry 模式：使用恶意快照但用 mimicry 策略构造负样本（注入良性边+特征替换）
    if preset.get("mimicry_mode", False):
        embedder.use_pos_fusion_neg = False
        embedder.mimicry_mode = True

    # LLM-guided 模式：先通过 MutationPipeline 生成变异图，注入训练
    if preset.get("use_mutation_pipeline", False):
        _run_mutation_pipeline(embedder, handler, preset)

    embedder.train()
    snapshot_embeddings = embedder.get_snapshot_embeddings()
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

    # 划分训练/测试：用 30% 恶意数据训练，70% 用于评估（论文 7:3 split）
    n_mal = len(mal_labels)
    n_train_mal = max(1, int(n_mal * 0.3))
    # 按序划分（保持时序）
    train_mal_emb = mal_embeddings[:n_train_mal]
    train_mal_labels = mal_labels[:n_train_mal]

    classify = get_classfy(CLASSIFY_NAME, gid=tag)
    classify.train(benign_embeddings, train_mal_emb, train_mal_labels)
    print(f"[{strategy_name}] 分类器训练耗时: {time.time()-t0:.1f}s")

    # ---- 4. 评估（用训练集以外的 70% 恶意数据） ----
    t0 = time.time()
    test_mal_emb = mal_embeddings[n_train_mal:]
    test_mal_labels = mal_labels[n_train_mal:]
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
                metrics = train_with_strategy(strat, handler, path_map)
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
