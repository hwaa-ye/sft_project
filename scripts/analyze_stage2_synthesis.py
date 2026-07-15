"""Audit Stage-2 synthesis outputs against their teacher queue."""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from synthesize_hard_stage2 import candidate_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True)
    parser.add_argument("--synthesis-dir", required=True)
    parser.add_argument("--cache-hit-cny-per-million", type=float, default=0.025)
    parser.add_argument("--cache-miss-cny-per-million", type=float, default=3.0)
    parser.add_argument("--output-cny-per-million", type=float, default=6.0)
    return parser.parse_args()


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def length_bucket(value):
    value = int(value or 0)
    if value <= 4096:
        return "2049-4096"
    if value <= 8192:
        return "4097-8192"
    return ">8192"


def reject_category(reason):
    reason = str(reason or "unknown")
    if reason.startswith("teacher_rejected"):
        return "teacher_rejected"
    if reason.startswith("verifier_rejected"):
        return "verifier_rejected"
    if reason.startswith("api_or_parse_error"):
        return "api_or_parse_error"
    return reason.split(":", 1)[0]


def summarize_status(queue, status_by_id, field):
    table = defaultdict(Counter)
    for row in queue:
        value = row.get(field, "unknown")
        if field == "sft_token_length":
            value = length_bucket(value)
        table[str(value)][status_by_id[candidate_id(row)]] += 1
    return {key: dict(value) for key, value in sorted(table.items())}


def main():
    args = parse_args()
    queue = read_jsonl(args.queue)
    synthesis_dir = Path(args.synthesis_dir)
    accepted = read_jsonl(synthesis_dir / "accepted.jsonl")
    rejected = read_jsonl(synthesis_dir / "rejected.jsonl")
    transient = read_jsonl(synthesis_dir / "transient_errors.jsonl")

    accepted_ids = {row["candidate_id"] for row in accepted}
    rejected_ids = {row["candidate_id"] for row in rejected}
    transient_ids = {row["candidate_id"] for row in transient}
    queue_ids = {candidate_id(row) for row in queue}
    unresolved_ids = transient_ids - accepted_ids - rejected_ids

    status_by_id = {}
    for cid in queue_ids:
        if cid in accepted_ids:
            status_by_id[cid] = "accepted"
        elif cid in rejected_ids:
            status_by_id[cid] = "rejected"
        elif cid in unresolved_ids:
            status_by_id[cid] = "transient"
        else:
            status_by_id[cid] = "missing"

    usage = Counter()
    calls_by_stage = Counter()
    for row in accepted + rejected + transient:
        for item in row.get("usage", []):
            calls_by_stage[str(item.get("stage", "unknown"))] += 1
            usage["cache_hit"] += int(item.get("prompt_cache_hit_tokens", 0) or 0)
            usage["cache_miss"] += int(
                item.get("prompt_cache_miss_tokens", item.get("prompt_tokens", 0)) or 0
            )
            usage["output"] += int(item.get("completion_tokens", 0) or 0)

    cost = (
        usage["cache_hit"] / 1_000_000 * args.cache_hit_cny_per_million
        + usage["cache_miss"] / 1_000_000 * args.cache_miss_cny_per_million
        + usage["output"] / 1_000_000 * args.output_cny_per_million
    )

    reasoning_lengths = sorted(int(row["compressed_reasoning_tokens"]) for row in accepted)
    sft_lengths = sorted(int(row["compressed_sft_tokens"]) for row in accepted)

    report = {
        "queue_samples": len(queue),
        "unique_queue_ids": len(queue_ids),
        "status": dict(Counter(status_by_id.values())),
        "raw_records": {
            "accepted": len(accepted),
            "rejected": len(rejected),
            "transient": len(transient),
        },
        "id_overlaps": {
            "accepted_rejected": len(accepted_ids & rejected_ids),
            "accepted_transient": len(accepted_ids & transient_ids),
            "rejected_transient": len(rejected_ids & transient_ids),
        },
        "by_failure_type": summarize_status(queue, status_by_id, "mining_failure_type"),
        "by_source": summarize_status(queue, status_by_id, "source"),
        "by_original_length": summarize_status(queue, status_by_id, "sft_token_length"),
        "reject_categories": dict(Counter(
            reject_category(row.get("reject_reason")) for row in rejected
        )),
        "accepted_lengths": {
            "reasoning_min": min(reasoning_lengths) if reasoning_lengths else 0,
            "reasoning_p50": statistics.median(reasoning_lengths) if reasoning_lengths else 0,
            "reasoning_p90": reasoning_lengths[int(0.9 * (len(reasoning_lengths) - 1))] if reasoning_lengths else 0,
            "reasoning_max": max(reasoning_lengths) if reasoning_lengths else 0,
            "formatted_min": min(sft_lengths) if sft_lengths else 0,
            "formatted_p50": statistics.median(sft_lengths) if sft_lengths else 0,
            "formatted_p90": sft_lengths[int(0.9 * (len(sft_lengths) - 1))] if sft_lengths else 0,
            "formatted_max": max(sft_lengths) if sft_lengths else 0,
        },
        "usage": {
            "calls_by_stage": dict(calls_by_stage),
            "prompt_cache_hit_tokens": usage["cache_hit"],
            "prompt_cache_miss_tokens": usage["cache_miss"],
            "completion_tokens": usage["output"],
            "estimated_cost_cny": round(cost, 4),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
