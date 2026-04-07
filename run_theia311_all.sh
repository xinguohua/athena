#!/bin/bash
set -e
source activate prographer 2>/dev/null || conda activate prographer

STRATEGIES="no_aug graphcl gca mimicry llm_guided"
DATASET="theia"
SCENE="theia311"

echo "===== 开始跑 $DATASET/$SCENE 全部策略: $STRATEGIES ====="
echo "时间: $(date)"

for s in $STRATEGIES; do
    echo ""
    echo "########## 策略: $s ##########"
    python -m process.benchmark_augmentation --strategy "$s" --dataset "$DATASET" --scene "$SCENE" 2>&1
    echo "########## $s 完成 ##########"
    echo ""
done

echo "===== 全部完成 ====="
echo "时间: $(date)"
