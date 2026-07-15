"""Build the expanded Stage-2 SFT mixture from pilot and newly synthesized hard data."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from build_stage2_pilot import (
    build_eval_signatures,
    nearest_control_rows,
    optimize_control_total,
    read_jsonl,
    stable_key,
    token_counts,
    write_jsonl,
)
from prepare_hard_stage2_candidates import (
    has_unsupported_control_char,
    near_eval_duplicate,
    normalize_question,
)
from repair_sft_answers import extract_balanced_boxed
from synthesize_hard_stage2 import candidate_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-hard", default="data/stage2_pilot/hard_final.jsonl")
    parser.add_argument("--new-queue", default="data/hard_stage2_teacher_queue_v2_failures.jsonl")
    parser.add_argument(
        "--new-synthesis-dir", default="data/hard_stage2_deepseek_v2_1156_min128"
    )
    parser.add_argument("--clean", default="data/clean_sft_repair_final_2048/train.jsonl")
    parser.add_argument(
        "--eval", action="append", default=[
            "data/eval_mixed_484.jsonl",
            "data/hard_stage2_candidates/hard_validation.jsonl",
        ]
    )
    parser.add_argument("--output-dir", default="data/stage2_expanded")
    parser.add_argument(
        "--tokenizer", default=str(Path.home() / ".cache/modelscope/hub/models/Qwen/Qwen3-8B")
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--hard-token-fraction", type=float, default=0.20)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_source_balanced_replay(clean_rows, target_tokens, seed):
    """Sample replay with source token shares matching the full clean corpus."""
    groups = defaultdict(list)
    for row in clean_rows:
        groups[str(row.get("source", "unknown"))].append(row)
    source_tokens = {
        source: sum(row["stage2_supervised_tokens"] for row in rows)
        for source, rows in groups.items()
    }
    all_tokens = sum(source_tokens.values())
    selected = []
    selected_keys = set()
    for source, rows in sorted(groups.items()):
        quota = round(target_tokens * source_tokens[source] / max(1, all_tokens))
        rows = sorted(rows, key=lambda row: stable_key(row, seed, f"expanded_replay_{source}"))
        current = 0
        for row in rows:
            if current >= quota:
                break
            selected.append({**row, "stage2_role": "clean_replay"})
            selected_keys.add(stable_key(row, seed, "clean_identity"))
            current += row["stage2_supervised_tokens"]

    current = sum(row["stage2_supervised_tokens"] for row in selected)
    if current < target_tokens:
        remainder = sorted(
            clean_rows, key=lambda row: stable_key(row, seed, "expanded_replay_fill")
        )
        for row in remainder:
            identity = stable_key(row, seed, "clean_identity")
            if identity in selected_keys:
                continue
            selected.append({**row, "stage2_role": "clean_replay"})
            selected_keys.add(identity)
            current += row["stage2_supervised_tokens"]
            if current >= target_tokens:
                break
    return selected


def summarize(rows):
    return {
        "samples": len(rows),
        "full_tokens": sum(row["stage2_full_tokens"] for row in rows),
        "supervised_tokens": sum(row["stage2_supervised_tokens"] for row in rows),
        "sources": dict(Counter(row.get("source", "unknown") for row in rows)),
        "roles": dict(Counter(row.get("stage2_role", "unknown") for row in rows)),
        "provenance": dict(Counter(row.get("stage2_provenance", "unknown") for row in rows)),
    }


def main():
    args = parse_args()
    if not 0 < args.hard_token_fraction < 1:
        raise ValueError("--hard-token-fraction must be between 0 and 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "hard": output_dir / "hard_final.jsonl",
        "rejected": output_dir / "hard_rejected.jsonl",
        "unresolved": output_dir / "unresolved.jsonl",
        "experiment": output_dir / "experiment_hard_replay.jsonl",
        "control": output_dir / "control_replay_only.jsonl",
        "manifest": output_dir / "manifest.json",
    }
    if any(path.exists() for path in paths.values()) and not args.overwrite:
        raise FileExistsError(f"outputs already exist in {output_dir}; pass --overwrite")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, local_files_only=True
    )
    eval_signatures = build_eval_signatures(args.eval)

    pilot_hard = read_jsonl(args.pilot_hard)
    queue = read_jsonl(args.new_queue)
    queue_by_id = {candidate_id(row): row for row in queue}
    synthesis_dir = Path(args.new_synthesis_dir)
    new_accepted = read_jsonl(synthesis_dir / "accepted.jsonl")
    new_rejected = read_jsonl(synthesis_dir / "rejected.jsonl")
    transient = read_jsonl(synthesis_dir / "transient_errors.jsonl")
    terminal_ids = {
        row["candidate_id"] for row in new_accepted + new_rejected
    }
    transient_ids = {row["candidate_id"] for row in transient}
    unresolved_ids = transient_ids - terminal_ids

    final_hard = []
    rejected_after_finalize = []
    seen_questions = set()
    reject_counts = Counter()

    for row in pilot_hard:
        full_tokens, supervised_tokens = token_counts(tokenizer, row)
        normalized = normalize_question(row.get("instruction", ""))
        if normalized in seen_questions:
            raise ValueError("pilot hard contains duplicate normalized questions")
        seen_questions.add(normalized)
        final_hard.append({
            **row,
            "stage2_full_tokens": full_tokens,
            "stage2_supervised_tokens": supervised_tokens,
            "stage2_role": "hard",
            "stage2_provenance": "pilot_hard",
        })

    for row in new_accepted:
        source_row = queue_by_id.get(row["candidate_id"])
        reasons = []
        if source_row is None:
            reasons.append("missing_queue_metadata")
        fields = (row.get("instruction", ""), row.get("reasoning", ""), row.get("answer", ""))
        if any(has_unsupported_control_char(value) for value in fields):
            reasons.append("unsupported_control_character")
        normalized = normalize_question(row.get("instruction", ""))
        if normalized in seen_questions:
            reasons.append("internal_or_pilot_duplicate")
        if any(normalized == eval_normalized for eval_normalized, _ in eval_signatures):
            reasons.append("exact_eval_duplicate")
        elif near_eval_duplicate(
            row.get("instruction", ""), eval_signatures, args.near_duplicate_threshold
        ):
            reasons.append("near_eval_duplicate")
        boxes = [
            value.strip() for value in extract_balanced_boxed(row.get("reasoning", ""))
            if value.strip()
        ]
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
            reject_counts.update(reasons)
            rejected_after_finalize.append({**row, "stage2_finalize_reasons": reasons})
            continue
        seen_questions.add(normalized)
        final_hard.append({
            **row,
            "mining_failure_type": source_row.get("mining_failure_type"),
            "mining_completion_reason": source_row.get("mining_completion_reason"),
            "mining_predicted_answer": source_row.get("mining_predicted_answer"),
            "original_sft_token_length": source_row.get(
                "sft_token_length", row.get("original_sft_token_length")
            ),
            "stage2_full_tokens": full_tokens,
            "stage2_supervised_tokens": supervised_tokens,
            "stage2_role": "hard",
            "stage2_provenance": "expanded_teacher",
        })

    unresolved = []
    for cid in sorted(unresolved_ids):
        unresolved.append({
            **queue_by_id.get(cid, {}),
            "candidate_id": cid,
            "stage2_status": "unresolved_api_or_parse_error",
        })

    clean_rows = []
    for row in read_jsonl(args.clean):
        full_tokens, supervised_tokens = token_counts(tokenizer, row)
        if full_tokens <= args.max_length and supervised_tokens > 0:
            clean_rows.append({
                **row,
                "stage2_full_tokens": full_tokens,
                "stage2_supervised_tokens": supervised_tokens,
                "stage2_provenance": "clean_replay_pool",
            })

    hard_tokens = sum(row["stage2_supervised_tokens"] for row in final_hard)
    replay_target = round(
        hard_tokens * (1 - args.hard_token_fraction) / args.hard_token_fraction
    )
    replay = select_source_balanced_replay(clean_rows, replay_target, args.seed)
    experiment = final_hard + replay
    experiment.sort(key=lambda row: stable_key(row, args.seed, "expanded_experiment_order"))

    print(json.dumps({
        "build_diagnostics": {
            "hard_samples": len(final_hard),
            "hard_supervised_tokens": hard_tokens,
            "replay_target_tokens": replay_target,
            "replay_samples": len(replay),
            "replay_supervised_tokens": sum(
                row["stage2_supervised_tokens"] for row in replay
            ),
            "experiment_samples": len(experiment),
            "clean_pool_samples": len(clean_rows),
        }
    }, ensure_ascii=False), flush=True)

    desired_lengths = [row["stage2_supervised_tokens"] for row in experiment]
    control, remaining = nearest_control_rows(clean_rows, desired_lengths, args.seed)
    experiment_tokens = sum(row["stage2_supervised_tokens"] for row in experiment)
    control = optimize_control_total(control, remaining, experiment_tokens, args.seed)
    control = [
        {**row, "stage2_provenance": "expanded_control"} for row in control
    ]
    control.sort(key=lambda row: stable_key(row, args.seed, "expanded_control_order"))

    write_jsonl(paths["hard"], final_hard)
    write_jsonl(paths["rejected"], rejected_after_finalize)
    write_jsonl(paths["unresolved"], unresolved)
    write_jsonl(paths["experiment"], experiment)
    write_jsonl(paths["control"], control)

    hard_summary = summarize(final_hard)
    experiment_summary = summarize(experiment)
    control_summary = summarize(control)
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "policy": "pilot_plus_new_verified_failures_with_source_balanced_clean_replay",
        "inputs": {
            "pilot_hard": args.pilot_hard,
            "new_queue": args.new_queue,
            "new_synthesis_dir": args.new_synthesis_dir,
            "clean": args.clean,
            "eval": args.eval,
        },
        "finalize": {
            "pilot_input": len(pilot_hard),
            "new_accepted_input": len(new_accepted),
            "new_permanent_rejected": len(new_rejected),
            "new_unresolved_unique": len(unresolved),
            "new_rejected_after_finalize": len(rejected_after_finalize),
            "final_hard": hard_summary,
            "failure_types": dict(Counter(
                row.get("mining_failure_type", "unknown") for row in final_hard
            )),
            "reject_reasons": dict(reject_counts),
        },
        "mixture": {
            "requested_hard_supervised_token_fraction": args.hard_token_fraction,
            "actual_hard_supervised_token_fraction": (
                hard_tokens / max(1, experiment_summary["supervised_tokens"])
            ),
            "experiment": experiment_summary,
            "control": control_summary,
            "sample_count_match": len(experiment) == len(control),
            "supervised_token_difference": (
                experiment_summary["supervised_tokens"]
                - control_summary["supervised_tokens"]
            ),
        },
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    with paths["manifest"].open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
