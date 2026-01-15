#!/bin/bash

# ================= 全局配置 (安全模式) =================
# 显存安全水位：降回 768 (参考以前成功的图表)
BASE_HIDDEN=768
BASE_EXPERTS=16
GPUS_PER_NODE=4

# 结果文件
CSV_FILE="trend_results_safe.csv"

# 环境变量
export CC=gcc
export OMP_NUM_THREADS=1

# 初始化 CSV
if [ ! -f "$CSV_FILE" ]; then
    echo "Experiment,Batch,SeqLen,HiddenDim,Experts,TopK,Baseline_Time(s),Custom_Time(s),Speedup" > $CSV_FILE
fi

# 通用运行函数
run_benchmark() {
    local EXP_NAME=$1
    local B=$2
    local S=$3
    local H=$4
    local E=$5
    local K=$6
    
    echo "----------------------------------------------------------"
    echo "▶ [$EXP_NAME] Batch=$B | Seq=$S | Hidden=$H | TopK=$K"
    echo "----------------------------------------------------------"

    # 1. Baseline
    export RUN_CUSTOM=0
    OUT_BASE=$(torchrun --nproc_per_node=$GPUS_PER_NODE benchmark.py \
      --batch $B --seq_len $S --hidden_dim $H \
      --num_experts $E --topk $K --chunk_size 32 2>&1)
    TIME_BASE=$(echo "$OUT_BASE" | grep -oP "mode=torch time=\K[0-9.]+" | head -n 1)

    # 2. Custom
    export RUN_CUSTOM=1
    OUT_CUST=$(torchrun --nproc_per_node=$GPUS_PER_NODE benchmark.py \
      --batch $B --seq_len $S --hidden_dim $H \
      --num_experts $E --topk $K --chunk_size 32 2>&1)
    TIME_CUST=$(echo "$OUT_CUST" | grep -oP "mode=custom time=\K[0-9.]+" | head -n 1)

    # 3. 记录
    if [ ! -z "$TIME_BASE" ] && [ ! -z "$TIME_CUST" ]; then
        SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $TIME_BASE / $TIME_CUST}")
        echo ">> Base: ${TIME_BASE}s | Custom: ${TIME_CUST}s | 🔥 Speedup: x$SPEEDUP"
        echo "$EXP_NAME,$B,$S,$H,$E,$K,$TIME_BASE,$TIME_CUST,$SPEEDUP" >> $CSV_FILE
    else
        echo "⚠️  Error parsing time."
        echo "$EXP_NAME,$B,$S,$H,$E,$K,ERROR,ERROR,0" >> $CSV_FILE
    fi
}

echo "=========================================================="
echo ">>> Starting Trend Sweep (Safe Mode) >>> Results: $CSV_FILE"
echo "=========================================================="

# === 实验 1: Batch Size 影响 (Seq=2048, Hidden=768) ===
echo ""
echo ">>> Running Exp 1: Batch Size Scaling..."
for B in 1 2 4 8; do
    run_benchmark "Batch_Scaling" $B 2048 $BASE_HIDDEN $BASE_EXPERTS 2
done

# === 实验 2: Sequence Length 影响 (Batch=4, Hidden=768) ===
# 移除了 8192 防止 OOM
echo ""
echo ">>> Running Exp 2: Seq Len Scaling..."
for S in 1024 2048 4096; do
    run_benchmark "Seq_Scaling" 4 $S $BASE_HIDDEN $BASE_EXPERTS 2
done

# === 实验 3: Top-K 影响 (Batch=4, Seq=2048) ===
echo ""
echo ">>> Running Exp 3: Top-K Scaling..."
for K in 2 4; do
    run_benchmark "TopK_Scaling" 4 2048 $BASE_HIDDEN $BASE_EXPERTS $K
done

echo ""
echo "=========================================================="
echo ">>> All Sweeps Finished! Data saved to $CSV_FILE"
echo "=========================================================="