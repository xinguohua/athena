#!/bin/bash
# Table IV：5种基础策略 × 剩余2个数据集（trace已跑完）
set -e
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate prographer
cd /home/nsas2020/fuzz/prographer

LOG="table4_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "[Table IV] 开始: $(date)"

STRATEGIES="no_aug graphcl gca mimicry llm_guided"

for ds_scene in "theia theia311" "clearscope clearscope3.6"; do
    set -- $ds_scene
    ds=$1; scene=$2
    echo "========== $ds/$scene =========="
    for strat in $STRATEGIES; do
        echo "--- $strat ---"
        python -m process.benchmark_augmentation --strategy $strat --dataset $ds --scene $scene
    done
done

echo "[Table IV] 完成: $(date)"
