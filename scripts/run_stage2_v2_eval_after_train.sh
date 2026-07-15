#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/sft_project

TRAIN_STATUS=output/stage2_pilot_v2_experiment/status.txt
OUT_DIR=output/stage2_pilot_eval
STATUS_FILE=${OUT_DIR}/v2_status.txt
PYTHON=/root/miniconda3/bin/python
BASE_MODEL=/root/autodl-tmp/model_cache/Qwen/Qwen3-8B
INIT_LORA=output/sft_clean_repair_v1/final
LORA_PATH=output/stage2_pilot_v2_experiment/final

mkdir -p "${OUT_DIR}"
printf '%s WAITING_FOR_TRAIN\n' "$(date -Is)" > "${STATUS_FILE}"

while ! grep -q 'ALL_DONE stage2_v2_experiment' "${TRAIN_STATUS}" 2>/dev/null; do
  if ! pgrep -f '/root/miniconda3/bin/python scripts/train_sft.py' >/dev/null; then
    printf '%s TRAIN_FAILED_OR_STOPPED\n' "$(date -Is)" >> "${STATUS_FILE}"
    exit 1
  fi
  sleep 30
done

run_eval() {
  local name=$1
  local test_path=$2
  local pred_path=$3
  local seed=$4

  printf '%s START %s\n' "$(date -Is)" "${name}" | tee -a "${STATUS_FILE}"
  env \
    PYTHONUNBUFFERED=1 \
    SFT_BASE_MODEL="${BASE_MODEL}" \
    SFT_INIT_LORA="${INIT_LORA}" \
    SFT_LORA_PATH="${LORA_PATH}" \
    SFT_TEST_PATH="${test_path}" \
    SFT_PRED_PATH="${pred_path}" \
    SFT_MAX_NEW_TOKENS=2048 \
    SFT_TEMPERATURE=0.7 \
    SFT_BATCH_SIZE=4 \
    SFT_SEED="${seed}" \
    SFT_RESUME=1 \
    "${PYTHON}" scripts/inference.py 2>&1 | tee "${OUT_DIR}/${name}.log"
  printf '%s DONE %s lines=%s\n' "$(date -Is)" "${name}" "$(wc -l < "${pred_path}")" | tee -a "${STATUS_FILE}"
}

run_eval \
  hard_v2 \
  data/hard_stage2_candidates/hard_validation.jsonl \
  "${OUT_DIR}/hard_v2.jsonl" \
  20260713

run_eval \
  mixed_v2 \
  data/eval_mixed_484.jsonl \
  "${OUT_DIR}/mixed_v2.jsonl" \
  42

printf '%s ALL_DONE\n' "$(date -Is)" | tee -a "${STATUS_FILE}"
