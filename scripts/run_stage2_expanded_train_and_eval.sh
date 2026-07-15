#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/sft_project

PYTHON=/root/miniconda3/bin/python
BASE_MODEL=/root/autodl-tmp/model_cache/Qwen/Qwen3-8B
STAGE1_LORA=output/sft_clean_repair_v1/final
TRAIN_OUT=output/stage2_expanded_experiment
EVAL_OUT=output/stage2_expanded_eval
STATUS_FILE=${TRAIN_OUT}/status.txt

mkdir -p "${TRAIN_OUT}" "${EVAL_OUT}"
printf '%s START train\n' "$(date -Is)" > "${STATUS_FILE}"

env \
  PYTHONUNBUFFERED=1 \
  TRANSFORMERS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 \
  SFT_MODEL_NAME="${BASE_MODEL}" \
  SFT_INIT_LORA="${STAGE1_LORA}" \
  SFT_DATA_DIR=data/stage2_expanded/tokenized_experiment \
  SFT_OUTPUT_DIR="${TRAIN_OUT}" \
  SFT_BATCH_SIZE=1 \
  SFT_GRAD_ACCUM=8 \
  SFT_LR=5e-5 \
  SFT_EPOCHS=1 \
  SFT_MAX_LENGTH=2048 \
  SFT_SEED=20260715 \
  "${PYTHON}" scripts/train_sft.py 2>&1 | tee "${TRAIN_OUT}/train.log"

printf '%s DONE train\n' "$(date -Is)" | tee -a "${STATUS_FILE}"

run_eval() {
  local name=$1
  local test_path=$2
  local pred_path=$3
  local seed=$4

  printf '%s START %s\n' "$(date -Is)" "${name}" | tee -a "${STATUS_FILE}"
  env \
    PYTHONUNBUFFERED=1 \
    SFT_BASE_MODEL="${BASE_MODEL}" \
    SFT_INIT_LORA="${STAGE1_LORA}" \
    SFT_LORA_PATH="${TRAIN_OUT}/final" \
    SFT_TEST_PATH="${test_path}" \
    SFT_PRED_PATH="${pred_path}" \
    SFT_MAX_NEW_TOKENS=2048 \
    SFT_TEMPERATURE=0.7 \
    SFT_BATCH_SIZE=4 \
    SFT_SEED="${seed}" \
    SFT_RESUME=1 \
    "${PYTHON}" scripts/inference.py 2>&1 | tee "${EVAL_OUT}/${name}.log"
  printf '%s DONE %s lines=%s\n' \
    "$(date -Is)" "${name}" "$(wc -l < "${pred_path}")" | tee -a "${STATUS_FILE}"
}

run_eval \
  hard \
  data/hard_stage2_candidates/hard_validation.jsonl \
  "${EVAL_OUT}/hard.jsonl" \
  20260713

run_eval \
  mixed \
  data/eval_mixed_484.jsonl \
  "${EVAL_OUT}/mixed.jsonl" \
  42

printf '%s ALL_DONE\n' "$(date -Is)" | tee -a "${STATUS_FILE}"
