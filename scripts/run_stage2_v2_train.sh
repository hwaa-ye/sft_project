#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/sft_project

OUT_DIR=output/stage2_pilot_v2_experiment
mkdir -p "${OUT_DIR}"
printf '%s START stage2_v2_experiment\n' "$(date -Is)" > "${OUT_DIR}/status.txt"

env \
  PYTHONUNBUFFERED=1 \
  TRANSFORMERS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 \
  SFT_MODEL_NAME=/root/autodl-tmp/model_cache/Qwen/Qwen3-8B \
  SFT_INIT_LORA=output/sft_clean_repair_v1/final \
  SFT_DATA_DIR=data/stage2_pilot_v2/tokenized_experiment \
  SFT_OUTPUT_DIR="${OUT_DIR}" \
  SFT_BATCH_SIZE=1 \
  SFT_GRAD_ACCUM=8 \
  SFT_LR=1e-4 \
  SFT_EPOCHS=2 \
  SFT_MAX_LENGTH=2048 \
  SFT_SEED=20260713 \
  /root/miniconda3/bin/python scripts/train_sft.py 2>&1 | tee "${OUT_DIR}/train.log"

printf '%s ALL_DONE stage2_v2_experiment\n' "$(date -Is)" >> "${OUT_DIR}/status.txt"
