#!/bin/bash
PY=/home/nsas2020/anaconda3/envs/prographer/bin/python
cd /home/nsas2020/fuzz/prographer
ROUNDS=3

for round in $(seq 1 $ROUNDS); do
    echo "========== Round $round / $ROUNDS =========="
    date
    for ds_scene in "cadets cadets314" "trace trace315" "theia theia311" "clearscope clearscope3.6"; do
        set -- $ds_scene
        $PY -u -m process.benchmark_augmentation --strategy llm_deepseek_v3 --dataset $1 --scene $2 2>&1
    done
done

echo "========== All $ROUNDS rounds done =========="
date
