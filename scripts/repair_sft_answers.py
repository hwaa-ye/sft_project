"""Repair clean SFT answers from pre-merge intermediate datasets.

Only answers found in a balanced ``\boxed{...}`` block in the intermediate
answer field are accepted. We deliberately do not guess from prose, numbers,
or reasoning in this conservative v1 repair.
"""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from sft_data_utils import format_math_sft


def extract_balanced_boxed(text):
    """Return all balanced \boxed contents, including nested LaTeX braces."""
    results = []
    marker = r"\boxed{"
    position = 0
    while True:
        start = text.find(marker, position)
        if start < 0:
            return results
        content_start = start + len(marker)
        depth = 1
        cursor = content_start
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            results.append(text[content_start:cursor - 1].strip())
            position = cursor
        else:
            position = content_start


def record_key(record):
    return (
        str(record.get("source", "")),
        record.get("instruction", ""),
        record.get("reasoning", ""),
    )


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-input", default="data/clean_sft_repair_2048/train.jsonl")
    parser.add_argument("--mathr-input", default="data/train_math.jsonl")
    parser.add_argument("--r1-input", default="data/train_math_r1.jsonl")
    parser.add_argument("--output-dir", default="data/clean_sft_repair_v2_2048")
    parser.add_argument("--tokenizer", default=os.environ.get(
        "QWEN_TOKENIZER", os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen3-8B")
    ))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_intermediates(paths):
    index = defaultdict(list)
    counts = Counter()
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                index[record_key(record)].append(record)
                counts["intermediate_records"] += 1
    return index, counts


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    manifest_path = output_dir / "manifest.json"
    if any(path.exists() for path in (train_path, rejected_path, manifest_path)) and not args.overwrite:
        raise FileExistsError(f"outputs exist in {output_dir}; pass --overwrite")

    intermediate_paths = [args.mathr_input, args.r1_input]
    index, counts = load_intermediates(intermediate_paths)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    by_source = defaultdict(Counter)

    with open(args.clean_input, encoding="utf-8") as source, \
            open(train_path, "w", encoding="utf-8") as accepted, \
            open(rejected_path, "w", encoding="utf-8") as rejected:
        for line_number, line in enumerate(source, 1):
            record = json.loads(line)
            source_name = str(record.get("source", "unknown"))
            counts["clean_input"] += 1
            by_source[source_name]["input"] += 1
            matches = index.get(record_key(record), [])
            if not matches:
                reason = "intermediate_join_missing"
                repaired = None
                raw_answer = None
            else:
                boxed_candidate = next(
                    (candidate for candidate in matches
                     if extract_balanced_boxed(candidate.get("answer", ""))),
                    None,
                )
                intermediate = boxed_candidate if boxed_candidate is not None else matches[0]
                matches.remove(intermediate)
                raw_answer = intermediate.get("answer", "")
                boxes = extract_balanced_boxed(raw_answer)
                if boxes and boxes[-1]:
                    repaired = boxes[-1]
                    reason = None
                else:
                    repaired = None
                    reason = "no_balanced_box_in_intermediate_answer"

            if reason:
                counts[reason] += 1
                by_source[source_name][reason] += 1
                rejected.write(json.dumps({
                    **record,
                    "answer_repair_status": "rejected",
                    "answer_reject_reason": reason,
                    "intermediate_answer": raw_answer,
                    "clean_input_line": line_number,
                }, ensure_ascii=False) + "\n")
                continue

            repaired_record = {
                **record,
                "answer": repaired,
                "answer_repair_status": "accepted",
                "answer_repair_method": "intermediate_answer_last_balanced_box",
            }
            token_length = len(tokenizer(
                format_math_sft(repaired_record), add_special_tokens=False
            )["input_ids"])
            repaired_record["sft_token_length"] = token_length
            if token_length > args.max_length:
                counts["repaired_too_long"] += 1
                by_source[source_name]["repaired_too_long"] += 1
                rejected.write(json.dumps({
                    **repaired_record,
                    "answer_repair_status": "rejected",
                    "answer_reject_reason": "repaired_too_long",
                    "max_length": args.max_length,
                    "intermediate_answer": raw_answer,
                    "clean_input_line": line_number,
                }, ensure_ascii=False) + "\n")
                continue

            counts["accepted"] += 1
            by_source[source_name]["accepted"] += 1
            accepted.write(json.dumps(repaired_record, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": 1,
        "policy": "conservative_answer_repair_from_intermediate_answer_balanced_box_only",
        "inputs": {
            args.clean_input: sha256(args.clean_input),
            args.mathr_input: sha256(args.mathr_input),
            args.r1_input: sha256(args.r1_input),
        },
        "tokenizer": args.tokenizer,
        "max_length": args.max_length,
        "counts": dict(counts),
        "sources": {name: dict(value) for name, value in sorted(by_source.items())},
        "outputs": {"train": str(train_path), "rejected": str(rejected_path)},
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
