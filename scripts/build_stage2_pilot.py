"""Finalize hard Stage-2 samples and build matched continual-SFT pilot arms."""

import argparse
import bisect
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from prepare_hard_stage2_candidates import (
    has_unsupported_control_char,
    near_eval_duplicate,
    ngrams,
    normalize_question,
)
from repair_sft_answers import extract_balanced_boxed
from sft_data_utils import format_math_sft, instruction_prefix
from synthesize_hard_stage2 import candidate_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthesis-dir",
        default="data/hard_stage2_deepseek_pro_clipped_calibration_10",
    )
    parser.add_argument(
        "--queue", default="data/hard_stage2_teacher_queue_all_failures_276.jsonl"
    )
    parser.add_argument(
        "--clean", default="data/clean_sft_repair_final_2048/train.jsonl"
    )
    parser.add_argument(
        "--eval",
        action="append",
        default=[
            "data/test_predictions_v2.jsonl",
            "data/hard_stage2_candidates/hard_validation.jsonl",
        ],
    )
    parser.add_argument("--output-dir", default="data/stage2_pilot")
    parser.add_argument(
        "--tokenizer",
        default=str(Path.home() / ".cache/modelscope/hub/models/Qwen/Qwen3-8B"),
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.85)
    parser.add_argument("--hard-token-fraction", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_key(row, seed, namespace=""):
    payload = (
        f"{namespace}\n{seed}\n{row.get('source', '')}\n"
        f"{normalize_question(row.get('instruction', ''))}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def token_counts(tokenizer, row):
    full_ids = tokenizer(
        format_math_sft(row), add_special_tokens=False, truncation=False
    )["input_ids"]
    prefix_ids = tokenizer(
        instruction_prefix(row), add_special_tokens=False, truncation=False
    )["input_ids"]
    return len(full_ids), len(full_ids) - len(prefix_ids)


def build_eval_signatures(paths):
    signatures = []
    for path in paths:
        for row in read_jsonl(path):
            normalized = normalize_question(row.get("instruction", ""))
            signatures.append((normalized, ngrams(normalized)))
    return signatures


def nearest_control_rows(clean_rows, desired_lengths, seed):
    """Match each experiment row to a clean row of similar supervised length."""
    pool = sorted(
        clean_rows,
        key=lambda row: (
            row["stage2_supervised_tokens"], stable_key(row, seed, "control_pool")
        ),
    )
    lengths = [row["stage2_supervised_tokens"] for row in pool]
    selected = []
    for target in sorted(desired_lengths, reverse=True):
        index = bisect.bisect_left(lengths, target)
        choices = []
        if index < len(pool):
            choices.append(index)
        if index > 0:
            choices.append(index - 1)
        best = min(
            choices,
            key=lambda i: (
                abs(lengths[i] - target), stable_key(pool[i], seed, "control_tie")
            ),
        )
        row = pool.pop(best)
        lengths.pop(best)
        selected.append({**row, "stage2_role": "control_replay"})
    return selected, pool


def optimize_control_total(selected, remaining, target_tokens, seed):
    """Swap clean rows until the control supervision total closely matches target."""
    current = sum(row["stage2_supervised_tokens"] for row in selected)
    for _ in range(100):
        delta = target_tokens - current
        if abs(delta) <= 10 or not remaining:
            break
        remaining.sort(
            key=lambda row: (
                row["stage2_supervised_tokens"], stable_key(row, seed, "control_remaining")
            )
        )
        remaining_lengths = [row["stage2_supervised_tokens"] for row in remaining]
        best = None
        for selected_index, old in enumerate(selected):
            desired_new_length = old["stage2_supervised_tokens"] + delta
            insertion = bisect.bisect_left(remaining_lengths, desired_new_length)
            for remaining_index in {insertion - 1, insertion, insertion + 1}:
                if not 0 <= remaining_index < len(remaining):
                    continue
                new = remaining[remaining_index]
                candidate_total = (
                    current - old["stage2_supervised_tokens"]
                    + new["stage2_supervised_tokens"]
                )
                score = (
                    abs(target_tokens - candidate_total),
                    stable_key(new, seed, "control_swap"),
                )
                if best is None or score < best[0]:
                    best = (score, selected_index, remaining_index, candidate_total)
        if best is None or best[0][0] >= abs(target_tokens - current):
            break
        _, selected_index, remaining_index, current = best
        old = selected[selected_index]
        new = remaining.pop(remaining_index)
        remaining.append({key: value for key, value in old.items() if key != "stage2_role"})
        selected[selected_index] = {**new, "stage2_role": "control_replay"}
    return selected


def main():
    args = parse_args()
    if not 0 < args.hard_token_fraction < 1:
        raise ValueError("--hard-token-fraction must be between 0 and 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "hard": output_dir / "hard_final.jsonl",
        "hard_rejected": output_dir / "hard_rejected.jsonl",
        "unresolved": output_dir / "unresolved.jsonl",
        "experiment": output_dir / "experiment_hard_replay.jsonl",
        "control": output_dir / "control_replay_only.jsonl",
        "manifest": output_dir / "manifest.json",
    }
    if any(path.exists() for path in output_paths.values()) and not args.overwrite:
        raise FileExistsError(f"outputs already exist in {output_dir}; pass --overwrite")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, local_files_only=True
    )
    queue_rows = read_jsonl(args.queue)
    queue_by_id = {candidate_id(row): row for row in queue_rows}
    synthesis_dir = Path(args.synthesis_dir)
    accepted_rows = read_jsonl(synthesis_dir / "accepted.jsonl")
    rejected_rows = read_jsonl(synthesis_dir / "rejected.jsonl")
    transient_rows = read_jsonl(synthesis_dir / "transient_errors.jsonl")
    accepted_ids = {row["candidate_id"] for row in accepted_rows}
    permanent_rejected_ids = {row["candidate_id"] for row in rejected_rows}
    transient_ids = {row["candidate_id"] for row in transient_rows}

    eval_signatures = build_eval_signatures(args.eval)
    seen_questions = {}
    final_hard = []
    final_rejected = []
    reject_counts = Counter()
    for row in accepted_rows:
        cid = row["candidate_id"]
        source_row = queue_by_id.get(cid)
        reasons = []
        if source_row is None:
            reasons.append("missing_queue_metadata")
        fields = (row.get("instruction", ""), row.get("reasoning", ""), row.get("answer", ""))
        if any(has_unsupported_control_char(value) for value in fields):
            reasons.append("unsupported_control_character")
        normalized = normalize_question(row.get("instruction", ""))
        if normalized in seen_questions:
            reasons.append("internal_duplicate")
        if any(normalized == eval_normalized for eval_normalized, _ in eval_signatures):
            reasons.append("exact_eval_duplicate")
        elif near_eval_duplicate(
            row.get("instruction", ""), eval_signatures, args.near_duplicate_threshold
        ):
            reasons.append("near_eval_duplicate")
        boxes = [value.strip() for value in extract_balanced_boxed(row.get("reasoning", "")) if value.strip()]
        if boxes and boxes[-1] != str(row.get("answer", "")).strip():
            reasons.append("reasoning_box_answer_mismatch")
        verifier = row.get("verifier", {})
        if not (
            verifier.get("valid") is True
            and verifier.get("supports_answer") is True
            and verifier.get("missing_key_step") is False
        ):
            reasons.append("verifier_not_strictly_valid")
        full_tokens, supervised_tokens = token_counts(tokenizer, row)
        if full_tokens > args.max_length:
            reasons.append("formatted_too_long")
        if supervised_tokens <= 0:
            reasons.append("empty_supervision")
        if reasons:
            for reason in reasons:
                reject_counts[reason] += 1
            final_rejected.append({**row, "stage2_finalize_reasons": reasons})
            continue
        seen_questions[normalized] = cid
        final_hard.append({
            **row,
            "mining_failure_type": source_row.get("mining_failure_type"),
            "mining_completion_reason": source_row.get("mining_completion_reason"),
            "mining_predicted_answer": source_row.get("mining_predicted_answer"),
            "stage2_full_tokens": full_tokens,
            "stage2_supervised_tokens": supervised_tokens,
            "stage2_role": "hard",
        })

    unresolved_ids = sorted(transient_ids - accepted_ids - permanent_rejected_ids)
    unresolved = []
    transient_by_id = {}
    for row in transient_rows:
        transient_by_id.setdefault(row["candidate_id"], []).append(row.get("reject_reason"))
    for cid in unresolved_ids:
        source_row = queue_by_id.get(cid, {})
        unresolved.append({
            **source_row,
            "candidate_id": cid,
            "stage2_status": "unresolved_api_error",
            "transient_attempt_records": len(transient_by_id.get(cid, [])),
            "transient_reasons": transient_by_id.get(cid, []),
        })

    clean_rows = read_jsonl(args.clean)
    clean_with_tokens = []
    for row in clean_rows:
        full_tokens, supervised_tokens = token_counts(tokenizer, row)
        if full_tokens > args.max_length or supervised_tokens <= 0:
            continue
        clean_with_tokens.append({
            **row,
            "stage2_full_tokens": full_tokens,
            "stage2_supervised_tokens": supervised_tokens,
        })

    hard_supervised = sum(row["stage2_supervised_tokens"] for row in final_hard)
    replay_target = round(
        hard_supervised * (1 - args.hard_token_fraction) / args.hard_token_fraction
    )
    replay_candidates = sorted(
        clean_with_tokens, key=lambda row: stable_key(row, args.seed, "experiment_replay")
    )
    replay_rows = []
    replay_tokens = 0
    for row in replay_candidates:
        replay_rows.append({**row, "stage2_role": "clean_replay"})
        replay_tokens += row["stage2_supervised_tokens"]
        if replay_tokens >= replay_target:
            break

    experiment = final_hard + replay_rows
    experiment.sort(key=lambda row: stable_key(row, args.seed, "experiment_order"))
    desired_lengths = [row["stage2_supervised_tokens"] for row in experiment]
    control, remaining_clean = nearest_control_rows(
        clean_with_tokens, desired_lengths, args.seed
    )
    experiment_supervised = sum(
        row["stage2_supervised_tokens"] for row in experiment
    )
    control = optimize_control_total(
        control, remaining_clean, experiment_supervised, args.seed
    )
    control.sort(key=lambda row: stable_key(row, args.seed, "control_order"))

    write_jsonl(output_paths["hard"], final_hard)
    write_jsonl(output_paths["hard_rejected"], final_rejected)
    write_jsonl(output_paths["unresolved"], unresolved)
    write_jsonl(output_paths["experiment"], experiment)
    write_jsonl(output_paths["control"], control)

    def summarize(rows):
        return {
            "samples": len(rows),
            "full_tokens": sum(row["stage2_full_tokens"] for row in rows),
            "supervised_tokens": sum(row["stage2_supervised_tokens"] for row in rows),
            "sources": dict(Counter(row.get("source", "unknown") for row in rows)),
            "roles": dict(Counter(row.get("stage2_role", "unknown") for row in rows)),
        }

    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "policy": "strictly_verified_real_model_failures_plus_token_matched_clean_replay",
        "inputs": {
            "queue": args.queue,
            "synthesis_dir": args.synthesis_dir,
            "clean": args.clean,
            "eval": args.eval,
        },
        "hard_finalize": {
            "synthesis_accepted_unique": len(accepted_ids),
            "synthesis_permanent_rejected_unique": len(permanent_rejected_ids),
            "synthesis_unresolved_unique": len(unresolved_ids),
            "accepted": len(final_hard),
            "rejected_after_finalize": len(final_rejected),
            "reject_reasons": dict(reject_counts),
            "failure_types": dict(Counter(row.get("mining_failure_type") for row in final_hard)),
        },
        "mixture": {
            "requested_hard_supervised_token_fraction": args.hard_token_fraction,
            "actual_hard_supervised_token_fraction": (
                hard_supervised
                / max(1, sum(row["stage2_supervised_tokens"] for row in experiment))
            ),
            "experiment": summarize(experiment),
            "control": summarize(control),
            "sample_count_match": len(experiment) == len(control),
            "supervised_token_difference": (
                summarize(experiment)["supervised_tokens"]
                - summarize(control)["supervised_tokens"]
            ),
        },
        "outputs": {name: str(path) for name, path in output_paths.items()},
    }
    with open(output_paths["manifest"], "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
