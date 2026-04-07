#!/bin/bash
PY=/home/nsas2020/anaconda3/envs/prographer/bin/python
cd /home/nsas2020/fuzz/prographer

SEEDS="42 123 456"
STRATEGIES="gca llm_deepseek_v3"
DATASETS="cadets:cadets314 trace:trace315 theia:theia311 clearscope:clearscope3.6"

for seed in $SEEDS; do
    echo "========== Seed $seed =========="
    for ds_scene in $DATASETS; do
        ds=$(echo $ds_scene | cut -d: -f1)
        scene=$(echo $ds_scene | cut -d: -f2)
        for strat in $STRATEGIES; do
            echo ">>> $strat on $ds/$scene (seed=$seed)"
            $PY -u -m process.benchmark_augmentation --strategy $strat --dataset $ds --scene $scene --seed $seed 2>&1
        done
    done
done

echo "========== All done =========="
date
