#!/bin/bash
set -e

echo "=== 1. 清理残留 ==="
pkill -f run_sft.py 2>/dev/null || true
sleep 2

echo "=== 2. 设置环境变量 ==="
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "=== 3. 测试显存 ==="
python3 -c "
import torch, gc
gc.collect()
torch.cuda.empty_cache()
print(f'可用显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')
print(f'已用: {torch.cuda.memory_allocated()/1024**3:.1f}GB')
"

echo "=== 4. 启动训练 ==="
cd ~/sft_project
mkdir -p output
nohup \
SFT_MODEL_NAME=${SFT_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct} \
SFT_MAX_LENGTH=${SFT_MAX_LENGTH:-1024} \
SFT_BATCH_SIZE=${SFT_BATCH_SIZE:-1} \
SFT_GRAD_ACCUM=${SFT_GRAD_ACCUM:-8} \
SFT_GRAD_CHECKPOINT=${SFT_GRAD_CHECKPOINT:-1} \
python3 scripts/train_h800.py > output/train_h800.log 2>&1 &
PID=$!
echo "训练 PID: $PID"

echo "=== 5. 等 30 秒看日志 ==="
sleep 30
tail -20 output/train_h800.log
