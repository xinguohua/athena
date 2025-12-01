import platform
import os
import torch
import yaml

from datahandlers import get_handler
from embedders import get_embedder_by_name
from process.classfy import get_classfy

# ---------------- 配置参数 ----------------
CONFIG_PATH = "config.yaml"
# DATASET_NAME = "atlas"          # 可切换数据集
DATASET_NAME = "cadets"          # 可切换数据集 (基础数据集名)
# 若仅训练/加载特定场景，请在此设置，例如: SCENE_NAME = "cadets314"；为 None 则加载全部
SCENE_NAME = "cadets314"
EMBEDDER_NAME = "gcc_dev"    # 嵌入器
# EMBEDDER_NAME = "gcc"    # 嵌入器
# EMBEDDER_NAME = "prographer"
# CLASSIFY_NAME = "prographer"     # 训练器
CLASSIFY_NAME = "topk"     # 训练器

# CLASSIFY_NAME = "svm"     # 训练器
# EMBEDDER_NAME = "unicorn"    # 嵌入器
# CLASSIFY_NAME = "unicorn"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 全局身份（用于区分不同人的训练产物），用于拼接输出文件名
GLOBAL_ID = "xgh"  # 按需修改，例如 "B"、"alice"、"ci-001"




def load_config(path: str) -> dict:
    """加载 YAML 配置，并根据系统环境选择配置分支"""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    system = platform.system().lower()
    return config["local"] if "windows" in system else config["remote"]


def prepare_data(path_map: dict):
    """加载数据并生成快照。内部使用全局 SCENE_NAME 控制场景过滤。"""
    handler = get_handler(DATASET_NAME, True, path_map, scene_name=SCENE_NAME)
    handler.load()
    handler.build_graph(GLOBAL_ID)
    return handler


def build_embeddings(handler):
    """构建并训练嵌入器，仅使用良性快照训练"""
    embedder_cls = get_embedder_by_name(EMBEDDER_NAME)
    # 统一：为嵌入器模型文件添加身份后缀，避免多人混用
    # 若类提供 _default_path，则基于该默认名拼接后缀；否则使用通用名
    from pathlib import Path
    default_model_name = getattr(embedder_cls, "_default_path", "embedder_model.pth")
    _p = Path(default_model_name)
    embedder_model_path = f"{_p.stem}_{GLOBAL_ID}{_p.suffix}"

    if EMBEDDER_NAME.lower() == "roland":
        benign_range = range(handler.benign_idx_start, handler.benign_idx_end + 1)
        print(f"[Train] 仅使用良性快照训练编码器: {handler.benign_idx_start}~{handler.benign_idx_end}")
        embedder = embedder_cls(handler.snapshots, train_indices=benign_range, model_path=embedder_model_path)
    else:
        # 对于支持 model_path 的编码器类（如 GCC/GCC-Dev/Prographer），传入带身份后缀的保存路径
        try:
            embedder = embedder_cls(handler.snapshots, model_path=embedder_model_path)
        except TypeError:
            # 某些编码器可能不接受 model_path 参数，则退回不传该参数
            embedder = embedder_cls(handler.snapshots)
    embedder.train()
    snapshot_embeddings = embedder.get_snapshot_embeddings()
    # 统计恶意节点偏离（仅当快照中存在恶意节点时会输出/写日志）
    try:
        # 训练阶段同样仅统计“恶意节点在全体节点中的排名”，相对于“快照节点简单平均中心”
        embedder.compute_malicious_deviation_per_snapshot(center_weighting='none')
    except Exception as ex:
        print(f"[Train] 恶意节点偏离统计失败：{ex}")
    print("\n--- Encoder 过程完成 ---")
    print(f"[嵌入] 快照嵌入序列: {snapshot_embeddings.shape}")
    print(snapshot_embeddings)
    return snapshot_embeddings


def main():
    env_config = load_config(CONFIG_PATH)
    path_map = env_config["path_map"]

    # 数据准备
    handler = prepare_data(path_map)

    # 嵌入训练
    snapshot_embeddings = build_embeddings(handler)

    # 模型训练
    benign_embeddings = snapshot_embeddings[handler.benign_idx_start:handler.benign_idx_end + 1]
    # 统一：将 gid 传入分类器，内部自行拼接/处理持久化路径
    classify = get_classfy(CLASSIFY_NAME, gid=GLOBAL_ID)
    classify.train(benign_embeddings)

if __name__ == "__main__":
    main()