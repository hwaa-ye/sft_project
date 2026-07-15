"""Turn blind-test predictions into a small, eval-disjoint teacher queue.

This selects only samples the current clean-SFT model failed at a 2048-token
cap.  The input must originate from ``hard_stage2_mining_pool`` rather than an
evaluation set.  It deliberately does not send anything to an API.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from diagnose_truncation import check_correct, classify, extract_answer


SOURCE_ORDER = [
    "amc_aime",
    "Haijian/Advanced-Math",
    "EduChat-Math",
    "meta-math/GSM8K_zh",
    "gavinluo/applied_math",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/hard_stage2_mining_predictions_400.jsonl")
    parser.add_argument("--output", default="data/hard_stage2_teacher_queue_100.jsonl")
    parser.add_argument("--report", default="data/hard_stage2_failure_mining_report.json")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--truncation-share", type=float, default=0.60)
    return parser.parse_args()


def take_round_robin(records, limit):
    buckets = defaultdict(list)
    for record in records:
        buckets[record.get("source", "unknown")].append(record)
    selected = []
    order = [source for source in SOURCE_ORDER if buckets[source]]
    order += sorted(source for source in buckets if source not in SOURCE_ORDER)
    while len(selected) < limit and any(buckets.values()):
        progressed = False
        for source in order:
            if len(selected) >= limit:
                break
            if buckets[source]:
                selected.append(buckets[source].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def main():
    args = parse_args()
    rows = [json.loads(line) for line in open(args.input, encoding="utf-8") if line.strip()]
    failures = {"truncated": [], "wrong_complete": []}
    all_counts = Counter()
    for row in rows:
        complete, completion_reason = classify(row.get("prediction", ""))
        prediction_answer = extract_answer(row.get("prediction", ""))
        correct = bool(prediction_answer and check_correct(prediction_answer, row.get("answer", "")))
        if not complete:
            failure_type = "truncated"
        elif not correct:
            failure_type = "wrong_complete"
        else:
            failure_type = "correct_complete"
        all_counts[failure_type] += 1
        if failure_type in failures:
            failures[failure_type].append({
                **row,
                "mining_failure_type": failure_type,
                "mining_completion_reason": completion_reason,
                "mining_predicted_answer": prediction_answer,
            })

    trunc_limit = min(round(args.limit * args.truncation_share), len(failures["truncated"]))
    wrong_limit = min(args.limit - trunc_limit, len(failures["wrong_complete"]))
    # Backfill either category when the other is scarcer than its target.
    trunc_limit = min(args.limit - wrong_limit, len(failures["truncated"]))
    selected = take_round_robin(failures["truncated"], trunc_limit)
    selected += take_round_robin(failures["wrong_complete"], wrong_limit)
    if len(selected) < args.limit:
        selected_keys = {
            (row.get("instruction", ""), row.get("answer", "")) for row in selected
        }
        selected += take_round_robin(
            [
                row
                for row in failures["truncated"] + failures["wrong_complete"]
                if (row.get("instruction", ""), row.get("answer", "")) not in selected_keys
            ],
            args.limit - len(selected),
        )
    selected = selected[:args.limit]

    output = Path(args.output)
    with open(output, "w", encoding="utf-8") as handle:
        for row in selected:
            row.pop("prediction", None)  # Teacher receives reference reasoning, never model rambling.
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "input": args.input,
        "input_samples": len(rows),
        "input_must_be_eval_disjoint": True,
        "classification_counts": all_counts,
        "selected": len(selected),
        "selected_by_failure_type": Counter(row["mining_failure_type"] for row in selected),
        "selected_by_source": Counter(row.get("source", "unknown") for row in selected),
    }
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
