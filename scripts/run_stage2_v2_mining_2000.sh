#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/sft_project

OUT_DIR=output/stage2_v2_mining_2000
PRED_PATH=data/hard_stage2_mining_predictions_v2_2000.jsonl
QUEUE_PATH=data/hard_stage2_teacher_queue_v2_failures.jsonl
REPORT_PATH=data/hard_stage2_failure_mining_v2_report.json
STATUS_FILE=${OUT_DIR}/status.txt

mkdir -p "${OUT_DIR}"
printf '%s START inference\n' "$(date -Is)" > "${STATUS_FILE}"

env \
  PYTHONUNBUFFERED=1 \
  SFT_BASE_MODEL=/root/autodl-tmp/model_cache/Qwen/Qwen3-8B \
  SFT_INIT_LORA=output/sft_clean_repair_v1/final \
  SFT_LORA_PATH=output/stage2_pilot_v2_experiment/final \
  SFT_TEST_PATH=data/hard_stage2_mining_pool_v2_2000.jsonl \
  SFT_PRED_PATH="${PRED_PATH}" \
  SFT_MAX_NEW_TOKENS=2048 \
  SFT_TEMPERATURE=0.7 \
  SFT_BATCH_SIZE=4 \
  SFT_SEED=20260714 \
  SFT_RESUME=1 \
  /root/miniconda3/bin/python scripts/inference.py 2>&1 | tee "${OUT_DIR}/inference.log"

printf '%s DONE inference lines=%s\n' "$(date -Is)" "$(wc -l < "${PRED_PATH}")" | tee -a "${STATUS_FILE}"
printf '%s START select_failures\n' "$(date -Is)" | tee -a "${STATUS_FILE}"

/root/miniconda3/bin/python scripts/select_hard_stage2_failures.py \
  --input "${PRED_PATH}" \
  --output "${QUEUE_PATH}" \
  --report "${REPORT_PATH}" \
  --limit 1200 \
  --truncation-share 0.40 \
  2>&1 | tee "${OUT_DIR}/select_failures.log"

printf '%s ALL_DONE queue_lines=%s\n' "$(date -Is)" "$(wc -l < "${QUEUE_PATH}")" | tee -a "${STATUS_FILE}"
