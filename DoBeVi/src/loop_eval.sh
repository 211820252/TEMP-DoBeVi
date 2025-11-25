#!/bin/bash

# chmod +x loop_eval.sh

# 循环次数
loop_times=2

for ((i=1; i<=loop_times; i++))
do
    clear
    echo "==== 循环 $i 开始 ===="

    clear
    echo "[Step 1] 当前 beam_size = 8, expansion = 600, timeout = 1800"
    cp 8.env ./.env 
    echo "[Step 1] 执行 python -m eval"
    python -m eval
    rm ./.env

    echo "[Step 2] 当前 b_max=16, b_min=4, lambda=15"
    cp ./16_layer_dropout.env ./.env
    echo "[Step 2] 执行 python -m eval"
    python -m eval 
    rm ./.env    

    clear
    echo "[Step 3] 当前 beam_size = 16, expansion = 600, timeout = 1800"
    cp 16.env ./.env 
    echo "[Step 3] 执行 python -m eval"
    python -m eval
    rm ./.env

    clear
    echo "[Step 4] 当前 beam_size = 32, expansion = 600, timeout = 1800"
    cp 32.env ./.env 
    echo "[Step 4] 执行 python -m eval"
    python -m eval
    rm ./.env
    
    echo "==== 循环 $i 完成 ===="
    echo
done
