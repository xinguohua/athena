#!/bin/bash
# RQ2 全部实验：Table IV (4数据集×5策略) + LLM模型对比
set -e

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate prographer
cd /home/nsas2020/fuzz/prographer

LOG="rq2_all_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "[RQ2] 开始时间: $(date)"

# ============================================================
# Table IV: 5种增强策略 × 4个数据集
# ============================================================
# cadets 已完成，跳过
# trace: 前4个策略已有结果(日志)，补跑 llm_guided
echo "========== Table IV: trace llm_guided =========="
python -m process.benchmark_augmentation --strategy llm_guided --dataset trace --scene trace315

# theia: 全部5策略
echo "========== Table IV: theia (全部策略) =========="
python -m process.benchmark_augmentation --dataset theia --scene theia311

# clearscope: 全部5策略
echo "========== Table IV: clearscope (全部策略) =========="
python -m process.benchmark_augmentation --dataset clearscope --scene clearscope3.6

# ============================================================
# LLM 模型对比 (Table IX): 不同 LLM 后端在 cadets 上的效果
# ============================================================
# llm_gpt4o 正在跑/已跑完，跳过

echo "========== LLM对比: Qwen2.5-7B =========="
python -m process.benchmark_augmentation --strategy llm_qwen25_7b --dataset cadets --scene cadets314

echo "========== LLM对比: Qwen2.5-14B =========="
python -m process.benchmark_augmentation --strategy llm_qwen25_14b --dataset cadets --scene cadets314

echo "========== LLM对比: DeepSeek-V3 =========="
python -m process.benchmark_augmentation --strategy llm_deepseek_v3 --dataset cadets --scene cadets314

echo "========== LLM对比: GLM-4-9B =========="
python -m process.benchmark_augmentation --strategy llm_glm4_9b --dataset cadets --scene cadets314

echo "[RQ2] 全部完成: $(date)"
echo "[RQ2] 日志: $LOG"
