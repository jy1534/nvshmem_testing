#!/bin/bash

# 定义结果文件 (CSV 格式)
LOG_FILE="gh200_full_sweep.csv"

# 写入表头
echo "Batch,SeqLen,Experts,TopK,HiddenDim,Baseline_Time(ms),Custom_Time(ms)" > $LOG_FILE

echo "Starting Sweep on GH200 (4 GPUs)..."

# === 循环嵌套 (54组) ===
for BATCH in 1 2 4; do
    for SEQ in 512 1024 2048; do
        for EXPERTS in 16 32; do
            for TOPK in 2 4 8; do
                
                # 固定参数
                HIDDEN=768
                CHUNK=32 
                
                echo "Running: Batch=$BATCH Seq=$SEQ E=$EXPERTS K=$TOPK..."
                
                # 运行命令
                OUTPUT=$(torchrun --nproc_per_node=4 benchmark.py \
                    --batch $BATCH \
                    --seq_len $SEQ \
                    --num_experts $EXPERTS \
                    --topk $TOPK \
                    --hidden_dim $HIDDEN \
                    --chunk_size $CHUNK 2>&1)

                # 提取数据的简单逻辑
                # 将输出存临时文件以便 grep
                echo "$OUTPUT" > temp_run.log
                
                # 根据你之前的日志格式 [rank 0] mode=torch time=0.249s 来提取数字
                # 这里只提取 mode=torch 和 mode=custom 后面的时间数字
                BASE_TIME=$(grep "mode=torch" temp_run.log | head -n 1 | grep -oP "time=\K[0-9.]+")
                CUSTOM_TIME=$(grep "mode=custom" temp_run.log | head -n 1 | grep -oP "time=\K[0-9.]+")
                
                # 如果为空则填 N/A
                if [ -z "$BASE_TIME" ]; then BASE_TIME="N/A"; fi
                if [ -z "$CUSTOM_TIME" ]; then CUSTOM_TIME="N/A"; fi
                
                # 写入 CSV
                echo "$BATCH,$SEQ,$EXPERTS,$TOPK,$HIDDEN,$BASE_TIME,$CUSTOM_TIME" >> $LOG_FILE
                
                echo "   -> Result: Base=$BASE_TIME, Custom=$CUSTOM_TIME"
                
            done
        done
    done
done

echo "========================================"
echo "All done! Data saved to $LOG_FILE"
