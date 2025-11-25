#!/bin/bash
# chmod +x loop_qwen.sh

# set -e  # 出错时立即退出

loop_times=2

for ((i=1; i<=loop_times; i++)); do
    clear
    echo "==== 循环 $i 开始 ===="

    echo "[Step 1] 当前 zero_shot minif2f"
    cp ./8.env ./.env
    cp ./search/qwen_algo_zero_shot.py ./search/qwen_algo.py
    python -m eval
    rm ./.env

    echo "[Step 2] 当前 few_shot minif2f"
    cp ./8.env ./.env
    cp ./search/qwen_algo_few_shot.py ./search/qwen_algo.py
    python -m eval
    rm ./.env

    echo "[Step 3] 当前 zero_shot_tactics minif2f"
    cp ./8.env ./.env
    cp ./search/qwen_algo_zero_shot_tactics.py ./search/qwen_algo.py
    python -m eval
    rm ./.env

    echo "[Step 4] 当前 few_shot_tactics minif2f"
    cp ./8.env ./.env
    cp ./search/qwen_algo_few_shot_tactics.py ./search/qwen_algo.py
    python -m eval 
    rm ./.env

    echo "[Step 5] 当前 zero_shot proofnet"
    cp ./8_proofnet.env ./.env
    cp ./search/qwen_algo_zero_shot.py ./search/qwen_algo.py
    python -m eval
    rm ./.env

    echo "[Step 6] 当前 few_shot proofnet"
    cp ./8_proofnet.env ./.env
    cp ./search/qwen_algo_few_shot.py ./search/qwen_algo.py
    python -m eval
    rm ./.env   

    echo "[Step 7] 当前 zero_shot_tactics proofnet"
    cp ./8_proofnet.env ./.env
    cp ./search/qwen_algo_zero_shot_tactics.py ./search/qwen_algo.py
    python -m eval
    rm ./.env

    echo "[Step 8] 当前 few_shot_tactics proofnet"
    cp ./8_proofnet.env ./.env
    cp ./search/qwen_algo_few_shot_tactics.py ./search/qwen_algo.py
    python -m eval
    rm ./.env

    echo "==== 循环 $i 完成 ===="
    echo
done
