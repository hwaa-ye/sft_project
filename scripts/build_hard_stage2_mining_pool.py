"""Build an eval-disjoint pool for mining current-model hard failures.

The resulting records are *not* training data.  They are sent to the current
SFT checkpoint for inference first; only its wrong or length-capped examples
may subsequently be considered for teacher synthesis.  This prevents the
held-out evaluation and long-probe examples from leaking back into Stage 2.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_QUOTAS = {
    "amc_aime": 140,
    "Haijian/Advanced-Math": 140,
    "EduChat-Math": 90,
    "meta-math/GSM8K_zh": 15,
    "gavinluo/applied_math": 15,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/hard_stage2_candidates/train_candidates.jsonl")
    parser.add_argument("--output", default="data/hard_stage2_mining_pool_400.jsonl")
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="JSONL whose instruction/answer pairs must not be selected; repeatable",
    )
    parser.add_argument(
        "--quota",
        action="append",
        default=[],
        metavar="SOURCE=COUNT",
        help="Override source quotas; repeatable. If supplied, replaces defaults.",
    )
    return parser.parse_args()


def parse_quotas(values):
    if not values:
        return dict(DEFAULT_QUOTAS)
    quotas = {}
    for value in values:
        source, separator, count = value.rpartition("=")
        if not separator or not source or not count.isdigit():
            raise ValueError(f"invalid --quota {value!r}; expected SOURCE=COUNT")
        quotas[source] = int(count)
    if not quotas or any(count < 0 for count in quotas.values()):
        raise ValueError("quotas must contain non-negative counts")
    return quotas


def main():
    args = parse_args()
    quotas = parse_quotas(args.quota)
    records = [json.loads(line) for line in open(args.input, encoding="utf-8") if line.strip()]
    excluded_keys = set()
    for path in args.exclude:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                excluded_keys.add((record.get("instruction", ""), str(record.get("answer", ""))))

    buckets = defaultdict(list)
    for record in records:
        key = (record.get("instruction", ""), str(record.get("answer", "")))
        if key in excluded_keys:
            continue
        buckets[str(record.get("source", "unknown"))].append(record)

    rng = random.Random(args.seed)
    selected = []
    selected_keys = set()
    for source, quota in quotas.items():
        bucket = buckets.get(source, [])
        rng.shuffle(bucket)
        for record in bucket[:quota]:
            key = (record.get("instruction", ""), record.get("answer", ""))
            if key not in selected_keys:
                selected.append(record)
                selected_keys.add(key)

    # Preserve the fixed size if a source was unavailable, without silently
    # admitting duplicate prompt/answer pairs.
    remainder = [
        record
        for source, bucket in buckets.items()
        for record in bucket[quotas.get(source, 0):]
    ]
    rng.shuffle(remainder)
    for record in remainder:
        if len(selected) >= sum(quotas.values()):
            break
        key = (record.get("instruction", ""), record.get("answer", ""))
        if key not in selected_keys:
            selected.append(record)
            selected_keys.add(key)

    if len(selected) != sum(quotas.values()):
        raise ValueError(
            f"expected {sum(quotas.values())} unique candidates, got {len(selected)}"
        )

    output = Path(args.output)
    with open(output, "w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(output),
        "samples": len(selected),
        "source_counts": Counter(record.get("source", "unknown") for record in selected),
        "excluded_pairs": len(excluded_keys),
        "quotas": quotas,
        "input_already_eval_disjoint": True,
    }, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
