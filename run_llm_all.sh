#!/bin/bash
PY=/home/nsas2020/anaconda3/envs/prographer/bin/python
cd /home/nsas2020/fuzz/prographer

echo "=== Starting LLM-guided (DeepSeek-V3) on all datasets ==="
date

for ds_scene in "cadets cadets314" "trace trace315" "theia theia311" "clearscope clearscope3.6"; do
    set -- $ds_scene
    ds=$1; scene=$2
    echo ">>> Running $ds/$scene ..."
    $PY -u -m process.benchmark_augmentation --strategy llm_deepseek_v3 --dataset $ds --scene $scene 2>&1 | tee -a llm_deepseek_all.log
    echo ">>> Done $ds/$scene"
    date
done

echo "=== All done ==="
