"""Apply the final conservative consistency gate to repaired SFT data."""

import argparse
import json
from collections import Counter
from pathlib import Path

from repair_sft_answers import extract_balanced_boxed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/clean_sft_repair_v2_2048/train.jsonl")
    parser.add_argument("--output-dir", default="data/clean_sft_repair_final_2048")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    manifest_path = output_dir / "manifest.json"
    if any(path.exists() for path in (train_path, rejected_path, manifest_path)) and not args.overwrite:
        raise FileExistsError(f"outputs exist in {output_dir}; pass --overwrite")

    counts = Counter()
    with open(args.input, encoding="utf-8") as source, \
            open(train_path, "w", encoding="utf-8") as accepted, \
            open(rejected_path, "w", encoding="utf-8") as rejected:
        for line_number, line in enumerate(source, 1):
            record = json.loads(line)
            counts["input"] += 1
            reasoning_boxes = [
                value for value in extract_balanced_boxed(record.get("reasoning", "")) if value
            ]
            answer = record.get("answer", "").strip()
            if reasoning_boxes and reasoning_boxes[-1].strip() != answer:
                counts["reasoning_answer_exact_mismatch"] += 1
                rejected.write(json.dumps({
                    **record,
                    "finalize_status": "rejected",
                    "finalize_reject_reason": "reasoning_answer_exact_mismatch",
                    "reasoning_last_box": reasoning_boxes[-1],
                    "input_line": line_number,
                }, ensure_ascii=False) + "\n")
                continue
            record["finalize_status"] = "accepted"
            accepted.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["accepted"] += 1

    manifest = {
        "schema_version": 1,
        "policy": "reject_any_exact_mismatch_between_answer_and_last_nonempty_reasoning_box",
        "input": args.input,
        "counts": dict(counts),
        "outputs": {"train": str(train_path), "rejected": str(rejected_path)},
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
