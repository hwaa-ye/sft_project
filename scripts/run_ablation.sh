#!/bin/bash
set -e
# Reward Ablation 实验：3 组 × 50 步

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd /root/autodl-tmp/sft_project
mkdir -p output/grpo_ablation

echo "=========================================="
echo "Reward Ablation 实验"
echo "A1: 去掉 completeness (0.9/0.0/0.1)"
echo "A2: 去掉 efficiency     (0.8/0.2/0.0)"
echo "A3: 高 completeness     (0.5/0.4/0.1)"
echo "每组 50 步, ~10h/组, 总计 ~30h"
echo "=========================================="

# ─── A1 ───
echo "=== [A1] 去掉 completeness (0.9/0.0/0.1) ==="
env GRPO_RW_ACC=0.9 GRPO_RW_COMP=0.0 GRPO_RW_EFF=0.1 \
    GRPO_OUTPUT_DIR=output/grpo_ablation/no_completeness \
    GRPO_MAX_STEPS=50 GRPO_PROMPTS=4 GRPO_RESPONSES=4 \
    GRPO_KL_BETA=0.04 GRPO_CLIP=0.2 GRPO_LR=5e-5 \
    PYTHONUNBUFFERED=1 \
    nohup /root/miniconda3/bin/python3 scripts/train_grpo.py > output/ablation_no_comp.log 2>&1 &
PID1=$!
echo "  PID: $PID1"
sleep 30
tail -5 output/ablation_no_comp.log
echo "  等待 A1 完成..."
wait $PID1
echo "  A1 完成"

# ─── A2 ───
echo ""
echo "=== [A2] 去掉 efficiency (0.8/0.2/0.0) ==="
env GRPO_RW_ACC=0.8 GRPO_RW_COMP=0.2 GRPO_RW_EFF=0.0 \
    GRPO_OUTPUT_DIR=output/grpo_ablation/no_efficiency \
    GRPO_MAX_STEPS=50 GRPO_PROMPTS=4 GRPO_RESPONSES=4 \
    GRPO_KL_BETA=0.04 GRPO_CLIP=0.2 GRPO_LR=5e-5 \
    PYTHONUNBUFFERED=1 \
    nohup /root/miniconda3/bin/python3 scripts/train_grpo.py > output/ablation_no_eff.log 2>&1 &
PID2=$!
echo "  PID: $PID2"
sleep 30
tail -5 output/ablation_no_eff.log
echo "  等待 A2 完成..."
wait $PID2
echo "  A2 完成"

# ─── A3 ───
echo ""
echo "=== [A3] 高 completeness (0.5/0.4/0.1) ==="
env GRPO_RW_ACC=0.5 GRPO_RW_COMP=0.4 GRPO_RW_EFF=0.1 \
    GRPO_OUTPUT_DIR=output/grpo_ablation/high_completeness \
    GRPO_MAX_STEPS=50 GRPO_PROMPTS=4 GRPO_RESPONSES=4 \
    GRPO_KL_BETA=0.04 GRPO_CLIP=0.2 GRPO_LR=5e-5 \
    PYTHONUNBUFFERED=1 \
    nohup /root/miniconda3/bin/python3 scripts/train_grpo.py > output/ablation_high_comp.log 2>&1 &
PID3=$!
echo "  PID: $PID3"
sleep 30
tail -5 output/ablation_high_comp.log
echo "  等待 A3 完成..."
wait $PID3
echo "  A3 完成"

echo ""
echo "=========================================="
echo "Ablation 全部完成!"
echo "=========================================="
