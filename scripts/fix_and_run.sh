#!/bin/bash
set -e
# SFT 一键启动脚本（Qwen3-8B + LoRA, 5090 32GB）

echo "=== 1. 清理残留 ==="
pkill -f train_sft.py 2>/dev/null || true
sleep 2

echo "=== 2. 设置环境 ==="
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-/root/autodl-fs/model_cache}

echo "=== 3. 显存状态 ==="
python3 -c "
import torch, gc
gc.collect()
torch.cuda.empty_cache()
print(f'可用显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')
print(f'已用: {torch.cuda.memory_allocated()/1024**3:.1f}GB')
"

echo "=== 4. 启动训练 ==="
cd /root/autodl-tmp/sft_project
mkdir -p output

SFT_MODEL_NAME=${SFT_MODEL_NAME:-Qwen/Qwen3-8B} \
SFT_MAX_LENGTH=${SFT_MAX_LENGTH:-2048} \
SFT_BATCH_SIZE=${SFT_BATCH_SIZE:-1} \
SFT_GRAD_ACCUM=${SFT_GRAD_ACCUM:-8} \
SFT_GRAD_CHECKPOINT=${SFT_GRAD_CHECKPOINT:-1} \
nohup python3 scripts/train_sft.py > output/train_sft.log 2>&1 &
PID=$!
echo "训练 PID: $PID"

echo "=== 5. 等 30 秒看日志 ==="
sleep 30
tail -20 output/train_sft.log
