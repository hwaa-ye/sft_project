#!/bin/bash
set -e
# GRPO 一键启动脚本：从 SFT LoRA 出发，RL 微调

echo "=== 1. 环境 ==="
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "=== 2. 显存状态 ==="
python3 -c "
import torch, gc
gc.collect()
torch.cuda.empty_cache()
print(f'可用: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')
print(f'已用: {torch.cuda.memory_allocated()/1024**3:.1f}GB')
"

echo "=== 3. 启动 GRPO 训练 ==="
cd /root/autodl-tmp/sft_project
mkdir -p output/grpo_qwen3

GRPO_PROMPTS=4 \
GRPO_RESPONSES=4 \
GRPO_KL_BETA=0.04 \
GRPO_CLIP=0.2 \
GRPO_LR=5e-5 \
GRPO_MAX_STEPS=150 \
PYTHONUNBUFFERED=1 \
nohup /root/miniconda3/bin/python3 scripts/train_grpo.py > output/train_grpo.log 2>&1 &

PID=$!
echo "PID: $PID"
echo "=== 等 15 秒看日志 ==="
sleep 15
tail -30 output/train_grpo.log
