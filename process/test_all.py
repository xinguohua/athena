import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Optional
import pickle
import os
import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from process.classfy import get_classfy

# --- 项目模块 ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from process.datahandlers import get_handler
from process.embedders import get_embedder_by_name


# ========================================================================
# 全局配置
# ========================================================================
# EMBEDDER_NAME = "prographer"
# CLASSIFY_NAME = "prographer"
# EMBEDDER_NAME = "unicorn"
# CLASSIFY_NAME = "unicorn"

SEQUENCE_LENGTH_L = 12
DETECTION_THRESHOLD = 0.016
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ========================================================================
# 工具函数
# ========================================================================
def save_snapshot_nodes(all_snapshots, output_file: Path = Path("test_snapshot.txt")) -> Optional[Path]:
    """保存快照节点信息到文件"""
    print(f"[INFO] 保存快照节点信息到: {output_file}")
    try:
        with output_file.open("w", encoding="utf-8") as f:
            f.write("=== 测试快照节点详情报告 ===\n")
            f.write(f"总快照数: {len(all_snapshots)}\n")
            f.write("=" * 60 + "\n\n")

            for i, snapshot in enumerate(all_snapshots):
                f.write(f"快照 {i}:\n")
                f.write(f"  节点总数: {len(snapshot.vs)}\n")
                f.write(f"  边总数: {len(snapshot.es)}\n")
                f.write("  节点详情:\n")

                node_type_count = defaultdict(int)
                malicious_count = sum(v.attributes().get("label", 0) == 1 for v in snapshot.vs)

                for v in snapshot.vs:
                    attrs = v.attributes()
                    node_name = attrs.get("name", "UNKNOWN")
                    node_type = attrs.get("type", "UNKNOWN")
                    label = attrs.get("label", 0)

                    # 更新类型统计
                    node_type_count[node_type] += 1

                    # 状态
                    status = "🔴恶意" if label == 1 else "🟢正常"

                    # 除了 name/label，其余属性全打印出来
                    extra_attrs = {k: v for k, v in attrs.items() if k not in ("name", "label", "type")}
                    extra_str = " | ".join(f"{k}:{v}" for k, v in extra_attrs.items())

                    f.write(f"    {node_name} | 类型:{node_type} | 状态:{status}")
                    if extra_str:
                        f.write(" | " + extra_str)
                    f.write("\n")

                f.write(f"  恶意节点数: {malicious_count}/{len(snapshot.vs)}\n")
                f.write("  节点类型分布:\n")
                for t, c in sorted(node_type_count.items()):
                    f.write(f"      {t}: {c}个\n")
                f.write("\n" + "-" * 50 + "\n\n")

        print(f"[INFO] 快照信息已写入 {output_file}")
        return output_file
    except Exception as e:
        print(f"[ERROR] 保存快照失败: {e}")
        return None


def get_true_labels(snapshots) -> np.ndarray:
    """提取快照真实标签"""
    return np.array([int(any(v["label"] == 1 for v in s.vs)) for s in snapshots])

def print_debug_info(all_snapshots, eval_true, eval_pred, eval_start_idx):
    """
    详细打印TP、FP、FN、TN快照的调试信息，显示导致分类的具体节点。
    """
    print("\n" + "="*70)
    print(" 🔍 详细调试信息 (TP / FP / FN / TN 完整分析)")
    print("="*70)

    # 分类收集各种情况的快照索引
    tp_indices = []  # 真阳性：真实恶意 + 预测恶意
    fp_indices = []  # 假阳性：真实良性 + 预测恶意
    fn_indices = []  # 假阴性：真实恶意 + 预测良性
    tn_indices = []  # 真阴性：真实良性 + 预测良性

    for i in range(len(eval_true)):
        snapshot_idx = i + eval_start_idx
        true_label = eval_true[i]
        pred_label = eval_pred[i]

        if true_label == 1 and pred_label == 1:
            tp_indices.append(snapshot_idx)
        elif true_label == 0 and pred_label == 1:
            fp_indices.append(snapshot_idx)
        elif true_label == 1 and pred_label == 0:
            fn_indices.append(snapshot_idx)
        else:  # true_label == 0 and pred_label == 0
            tn_indices.append(snapshot_idx)

    # 打印统计概览
    print(f"\n📊 快照分类统计:")
    print(f"  ✅ 真阳性 (TP): {len(tp_indices)} 个快照")
    print(f"  ❌ 假阳性 (FP): {len(fp_indices)} 个快照")
    print(f"  ⚠️  假阴性 (FN): {len(fn_indices)} 个快照")
    print(f"  ✓  真阴性 (TN): {len(tn_indices)} 个快照")

    # === 详细分析 TP 快照 ===
    if tp_indices:
        print("\n" + "="*50)
        print("✅ 真阳性 (TP) 快照详细分析 - 正确检测到的恶意快照")
        print("="*50)
        for snapshot_idx in tp_indices:
            snapshot = all_snapshots[snapshot_idx]
            print(f"\n🎯 快照 {snapshot_idx}:")

            # 分析节点类型
            malicious_nodes = []
            benign_nodes = []
            node_types_count = {}

            for v in snapshot.vs:
                node_name = v['name']
                node_type = v.attributes().get('type_name', 'UNKNOWN')
                frequency = v.attributes().get('frequency', 'UNKNOWN')
                node_types_count[node_type] = node_types_count.get(node_type, 0) + 1

                if v.attributes().get('label') == 1:
                    malicious_nodes.append(f"{node_name}({node_type})【{frequency}】")
                else:
                    benign_nodes.append(f"{node_name}({node_type})")

            print(f"  📈 总节点数: {len(snapshot.vs)}, 边数: {len(snapshot.es)}")
            print(f"  🔴 恶意节点 ({len(malicious_nodes)}个):")
            if malicious_nodes:
                malicious_str = ', '.join(malicious_nodes[:10])  # 最多显示10个
                if len(malicious_nodes) > 10:
                    malicious_str += f" ... (+{len(malicious_nodes)-10}个更多)"
                print(f"      {malicious_str}")

            print(f"  📊 节点类型分布: {dict(sorted(node_types_count.items()))}")

    # === 详细分析 FP 快照 ===
    if fp_indices:
        print("\n" + "="*50)
        print("❌ 假阳性 (FP) 快照详细分析 - 误报的良性快照")
        print("="*50)
        for snapshot_idx in fp_indices:
            snapshot = all_snapshots[snapshot_idx]
            print(f"\n🚨 快照 {snapshot_idx} (误报):")

            # 分析节点类型分布，寻找误报原因
            node_types_count = {}
            suspicious_patterns = []
            all_nodes = []

            for v in snapshot.vs:
                node_name = v['name']
                node_type = v.attributes().get('type_name', 'UNKNOWN')
                node_types_count[node_type] = node_types_count.get(node_type, 0) + 1
                frequency = v.attributes().get('frequency', 'UNKNOWN')

                all_nodes.append(f"{node_name}({node_type})【{frequency}】")

                # 检查可能导致误报的模式
                if 'SUBJECT_PROCESS' in node_type and any(word in node_name.lower()
                                                          for word in ['system', 'admin', 'service', 'daemon']):
                    suspicious_patterns.append(f"系统进程: {node_name}")
                elif 'NETFLOW' in node_type:
                    suspicious_patterns.append(f"网络流: {node_name}")

            print(f"  📈 总节点数: {len(snapshot.vs)}, 边数: {len(snapshot.es)}")
            print(f"  📊 节点类型分布: {dict(sorted(node_types_count.items()))}")

            if suspicious_patterns:
                print(f"  ⚡ 可能的误报原因:")
                for pattern in suspicious_patterns[:5]:  # 最多显示5个
                    print(f"      • {pattern}")

            # 显示部分节点名称用于分析
            print(f"  📝 部分节点 (前10个):")
            sample_nodes = ', '.join(all_nodes[:10])
            if len(all_nodes) > 10:
                sample_nodes += f" ... (+{len(all_nodes)-10}个更多)"
            wrapped_nodes = textwrap.fill(sample_nodes, width=70, initial_indent='      ', subsequent_indent='      ')
            print(wrapped_nodes)

    # === 详细分析 FN 快照 ===
    if fn_indices:
        print("\n" + "="*50)
        print("⚠️ 假阴性 (FN) 快照详细分析 - 漏检的恶意快照")
        print("="*50)
        for snapshot_idx in fn_indices:
            snapshot = all_snapshots[snapshot_idx]
            print(f"\n⚠️  快照 {snapshot_idx} (漏检):")

            # 分析为什么这些恶意节点没被检测到
            malicious_nodes = []
            benign_nodes = []
            node_types_count = {}

            for v in snapshot.vs:
                node_name = v['name']
                node_type = v.attributes().get('type_name', 'UNKNOWN')
                node_types_count[node_type] = node_types_count.get(node_type, 0) + 1
                frequency = v.attributes().get('frequency', 'UNKNOWN')

                if v.attributes().get('label') == 1:
                    malicious_nodes.append(f"{node_name}({node_type})【{frequency}】")
                else:
                    benign_nodes.append(f"{node_name}({node_type})")

            print(f"  📈 总节点数: {len(snapshot.vs)}, 边数: {len(snapshot.es)}")
            print(f"  🔴 被漏检的恶意节点 ({len(malicious_nodes)}个):")
            if malicious_nodes:
                malicious_str = ', '.join(malicious_nodes)
                wrapped_malicious = textwrap.fill(malicious_str, width=70, initial_indent='      ', subsequent_indent='      ')
                print(wrapped_malicious)

            print(f"  📊 节点类型分布: {dict(sorted(node_types_count.items()))}")
            print(f"  💡 可能的漏检原因: 恶意节点比例较低 ({len(malicious_nodes)}/{len(snapshot.vs)} = {len(malicious_nodes)/len(snapshot.vs)*100:.1f}%)")

    # === 简要显示 TN 快照统计 ===
    if tn_indices:
        print("\n" + "="*50)
        print("✓ 真阴性 (TN) 快照统计 - 正确识别的良性快照")
        print("="*50)
        print(f"  ✅ 共有 {len(tn_indices)} 个快照被正确识别为良性")

        # 统计TN快照的节点类型分布
        if len(tn_indices) > 0:
            sample_tn = all_snapshots[tn_indices[0]]  # 取一个样本
            tn_node_types = {}
            for v in sample_tn.vs:
                node_type = v.attributes().get('type_name', 'UNKNOWN')
                tn_node_types[node_type] = tn_node_types.get(node_type, 0) + 1
                frequency = v.attributes().get('frequency', 'UNKNOWN')

            print(f"  📊 典型良性快照的节点类型分布 (快照{tn_indices[0]}): {dict(sorted(tn_node_types.items()))}【{frequency}】")

    print("\n" + "="*70)
    print("🎯 调试分析总结:")
    print(f"  • 总共分析了 {len(eval_true)} 个快照")
    print(f"  • 检测准确率: {(len(tp_indices) + len(tn_indices))/len(eval_true)*100:.1f}%")
    if len(tp_indices) + len(fn_indices) > 0:
        print(f"  • 恶意快照召回率: {len(tp_indices)/(len(tp_indices) + len(fn_indices))*100:.1f}%")
    if len(tp_indices) + len(fp_indices) > 0:
        print(f"  • 恶意检测精确率: {len(tp_indices)/(len(tp_indices) + len(fp_indices))*100:.1f}%")
    print("="*70)

def predict_snapshots(
    snapshot_embeddings: np.ndarray,
) -> Tuple[np.ndarray, Dict]:
    """预测快照异常标签"""

    classify = get_classfy(CLASSIFY_NAME)
    classify.load()
    pred_labels, diff_vectors  = classify.predict(snapshot_embeddings)

    return pred_labels, diff_vectors


def run_evaluation(path_map: dict) -> None:
    snapshot_file = "snapshot_data.pkl"
    if not os.path.exists(snapshot_file):
        print(f"❌ 错误：快照数据文件不存在: {snapshot_file}")
        print("请先运行 train_darpa.py 来生成快照数据")
        return

    with open(snapshot_file, 'rb') as f:
        snapshot_data = pickle.load(f)

    # 提取快照数据
    all_snapshots = snapshot_data['all_snapshots']
    benign_idx_start = snapshot_data['benign_idx_start']
    benign_idx_end = snapshot_data['benign_idx_end']
    malicious_idx_start = snapshot_data['malicious_idx_start']
    malicious_idx_end = snapshot_data['malicious_idx_end']

    print(f"✅ 快照数据加载成功:")
    print(f"  - 总快照数: {len(all_snapshots)}")
    print(f"  - 良性快照范围: {benign_idx_start} 到 {benign_idx_end}")
    print(f"  - 恶意快照范围: {malicious_idx_start} 到 {malicious_idx_end}")
    mal_snapshots = all_snapshots[malicious_idx_start: malicious_idx_end + 1]
    if not mal_snapshots:
        print("[ERROR] 未能构建快照")
        return
    save_snapshot_nodes(mal_snapshots)
    true_labels = get_true_labels(mal_snapshots)

    print("\n[DEBUG] 快照信息")
    print(f"  - 总快照数: {len(mal_snapshots)}")
    print(f"  - 真实标签数: {len(true_labels)}")
    print(f"  - 真实标签: {true_labels.tolist()}")

    embedder_cls = get_embedder_by_name(EMBEDDER_NAME)
    embedder = embedder_cls.load(snapshot_sequence=all_snapshots)
    snapshot_embeddings = embedder.get_snapshot_embeddings()

    pred_labels, diff_vectors = predict_snapshots(
        snapshot_embeddings[malicious_idx_start: malicious_idx_end + 1]
    )
    print(f"检测到 {len(diff_vectors)} 个异常快照")
    print(f"预测标签长度: {len(pred_labels)}")

    acc = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels, zero_division=0)
    rec = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    tp = np.sum((true_labels == 1) & (pred_labels == 1))
    fp = np.sum((true_labels == 0) & (pred_labels == 1))
    tn = np.sum((true_labels == 0) & (pred_labels == 0))
    fn = np.sum((true_labels == 1) & (pred_labels == 0))

    print("\n=== 评估结果 ===")
    print("\n" + "=" * 50)
    print(" 快照级别评估结果 (所有快照)")
    print("=" * 50)
    print(f" 真阳性 (TP): {tp}")
    print(f" 假阳性 (FP): {fp}")
    print(f" 真阴性 (TN): {tn}")
    print(f" 假阴性 (FN): {fn}")
    print("\n 性能评分:")
    print(f" 准确率: {acc:.4f}")
    print(f" 精确率: {prec:.4f}")
    print(f" 召回率: {rec:.4f}")
    print(f" F1分数: {f1:.4f}")
    print("=" * 50)
    print_debug_info(mal_snapshots, true_labels, pred_labels, 0)  # 从索引0开始



# ========================================================================
# 主入口
# ========================================================================
if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env = config["local"] if "windows" in sys.platform else config["remote"]

    run_evaluation(env["path_map"])