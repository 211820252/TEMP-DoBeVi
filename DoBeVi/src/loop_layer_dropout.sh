#!/bin/bash
# chmod +x loop_layer_dropout.sh
# 自动运行多组 b_max / b_min / lambda 参数组合

# set -e  # 出错时立即退出

loop_times=1

for ((i=1; i<=loop_times; i++)); do
    clear
    echo "==== 循环 $i 开始 ===="

    # echo "[Step 1] 当前 b_max=16, b_min=4, lambda=5"
    # cp ./16.env ./.env
    # cp ./search/layer_dropout_algo_4_5.py ./search/layer_dropout_algo.py
    # python -m eval
    # rm ./.env

    # echo "[Step 2] 当前 b_max=16, b_min=4, lambda=30"
    # cp ./16.env ./.env
    # cp ./search/layer_dropout_algo_4_30.py ./search/layer_dropout_algo.py
    # python -m eval
    # rm ./.env

    # echo "[Step 3] 当前 b_max=32, b_min=4, lambda=15"
    # cp ./32.env ./.env
    # cp ./search/layer_dropout_algo_4_15.py ./search/layer_dropout_algo.py
    # python -m eval
    # rm ./.env

    # echo "[Step 4] 当前 b_max=16, b_min=2, lambda=15"
    # cp ./16.env ./.env
    # cp ./search/layer_dropout_algo_2_15.py ./search/layer_dropout_algo.py
    # python -m eval 
    # rm ./.env

    # echo "[Step 5] 当前 b_max=32, b_min=16, lambda=15"
    # cp ./32.env ./.env
    # cp ./search/layer_dropout_algo_16_15.py ./search/layer_dropout_algo.py
    # python -m eval 
    # rm ./.env

    # echo "[Step 6] 当前 b_max=8, b_min=2, lambda=15"
    # cp ./8.env ./.env
    # cp ./search/layer_dropout_algo_2_15.py ./search/layer_dropout_algo.py
    # python -m eval 
    # rm ./.env

    echo "[Step 7] 当前 b_max=8, b_min=2, lambda=15"
    cp ./8.env ./.env
    cp ./search/layer_dropout_algo_2_15.py ./search/layer_dropout_algo.py
    python -m eval
    rm ./.env

    echo "[Step 8] 当前 b_max=8, b_min=4, lambda=15"
    cp ./8.env ./.env
    cp ./search/layer_dropout_algo_2_15.py ./search/layer_dropout_algo.py
    python -m eval
    rm ./.env

    echo "==== 循环 $i 完成 ===="
    echo
done
