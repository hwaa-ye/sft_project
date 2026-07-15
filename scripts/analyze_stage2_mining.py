"""Summarize Stage-2 mining predictions by failure type, source, and length."""

import argparse
import json
from collections import Counter, defaultdict

from diagnose_truncation import check_correct, classify, extract_answer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    return parser.parse_args()


def length_bucket(value):
    value = int(value or 0)
    if value <= 4096:
        return "2049-4096"
    if value <= 8192:
        return "4097-8192"
    return ">8192"


def classify_row(row):
    complete, _ = classify(row.get("prediction", ""))
    answer = extract_answer(row.get("prediction", ""))
    correct = bool(answer and check_correct(answer, row.get("answer", "")))
    if not complete:
        return "truncated"
    return "correct_complete" if correct else "wrong_complete"


def main():
    args = parse_args()
    rows = [json.loads(line) for line in open(args.input, encoding="utf-8") if line.strip()]
    overall = Counter()
    by_source = defaultdict(Counter)
    by_length = defaultdict(Counter)
    by_source_length = defaultdict(Counter)

    for row in rows:
        result = classify_row(row)
        source = str(row.get("source", "unknown"))
        bucket = length_bucket(row.get("sft_token_length"))
        overall[result] += 1
        by_source[source][result] += 1
        by_length[bucket][result] += 1
        by_source_length[f"{source}|{bucket}"][result] += 1

    def serializable(mapping):
        return {key: dict(value) for key, value in sorted(mapping.items())}

    print(json.dumps({
        "input": args.input,
        "samples": len(rows),
        "overall": dict(overall),
        "by_source": serializable(by_source),
        "by_length": serializable(by_length),
        "by_source_length": serializable(by_source_length),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
