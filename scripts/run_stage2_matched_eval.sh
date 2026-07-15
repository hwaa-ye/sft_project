#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/sft_project

PYTHON=/root/miniconda3/bin/python
BASE_MODEL=/root/autodl-tmp/model_cache/Qwen/Qwen3-8B
TEST_PATH=data/hard_stage2_candidates/hard_validation.jsonl
OUT_DIR=output/stage2_pilot_eval
STATUS_FILE=${OUT_DIR}/status.txt

mkdir -p "${OUT_DIR}"

run_eval() {
  local name=$1
  local init_lora=$2
  local lora_path=$3
  local pred_path=${OUT_DIR}/hard_${name}.jsonl

  printf '%s START %s\n' "$(date -Is)" "${name}" | tee -a "${STATUS_FILE}"
  env \
    PYTHONUNBUFFERED=1 \
    SFT_BASE_MODEL="${BASE_MODEL}" \
    SFT_INIT_LORA="${init_lora}" \
    SFT_LORA_PATH="${lora_path}" \
    SFT_TEST_PATH="${TEST_PATH}" \
    SFT_PRED_PATH="${pred_path}" \
    SFT_MAX_NEW_TOKENS=2048 \
    SFT_TEMPERATURE=0.7 \
    SFT_BATCH_SIZE=4 \
    SFT_SEED=20260713 \
    SFT_RESUME=1 \
    "${PYTHON}" scripts/inference.py 2>&1 | tee "${OUT_DIR}/hard_${name}.log"
  printf '%s DONE %s lines=%s\n' "$(date -Is)" "${name}" "$(wc -l < "${pred_path}")" | tee -a "${STATUS_FILE}"
}

run_eval \
  stage1 \
  "" \
  output/sft_clean_repair_v1/final

run_eval \
  control \
  output/sft_clean_repair_v1/final \
  output/stage2_pilot_control/final

run_eval \
  experiment \
  output/sft_clean_repair_v1/final \
  output/stage2_pilot_experiment/final

printf '%s ALL_DONE\n' "$(date -Is)" | tee -a "${STATUS_FILE}"
