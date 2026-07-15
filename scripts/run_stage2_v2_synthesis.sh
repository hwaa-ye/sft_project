#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/sft_project

OUT_DIR=data/hard_stage2_deepseek_v2_1156_min128
LOG_DIR=output/stage2_v2_synthesis
STATUS_FILE=${LOG_DIR}/status.txt

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is not set" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
printf '%s START synthesis\n' "$(date -Is)" > "${STATUS_FILE}"

env \
  PYTHONUNBUFFERED=1 \
  DEEPSEEK_BASE_URL=https://api.deepseek.com \
  /root/miniconda3/bin/python scripts/synthesize_hard_stage2.py \
    --input data/hard_stage2_teacher_queue_v2_failures.jsonl \
    --output-dir "${OUT_DIR}" \
    --limit 1156 \
    --seed 20260715 \
    --teacher-model deepseek-v4-pro \
    --verifier-model deepseek-v4-pro \
    --tokenizer /root/autodl-tmp/model_cache/Qwen/Qwen3-8B \
    --max-length 2048 \
    --min-reasoning-tokens 128 \
    --max-attempts 3 \
    --max-source-chars 12000 \
    --teacher-max-tokens 1800 \
    --verifier-max-tokens 800 \
    2>&1 | tee "${LOG_DIR}/synthesis.log"

accepted=0
rejected=0
transient=0
[[ -f "${OUT_DIR}/accepted.jsonl" ]] && accepted=$(wc -l < "${OUT_DIR}/accepted.jsonl")
[[ -f "${OUT_DIR}/rejected.jsonl" ]] && rejected=$(wc -l < "${OUT_DIR}/rejected.jsonl")
[[ -f "${OUT_DIR}/transient_errors.jsonl" ]] && transient=$(wc -l < "${OUT_DIR}/transient_errors.jsonl")
printf '%s ALL_DONE accepted=%s rejected=%s transient=%s\n' \
  "$(date -Is)" "${accepted}" "${rejected}" "${transient}" | tee -a "${STATUS_FILE}"
