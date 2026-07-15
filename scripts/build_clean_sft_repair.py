"""Build an auditable, zero-truncation SFT-repair dataset.

The first repair experiment keeps only examples whose complete formatted
prompt+response fits in max_length. Long examples are preserved verbatim in a
separate reject file for a later compression/capability stage.
"""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from sft_data_utils import format_math_sft


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/train_math_all.jsonl")
    parser.add_argument("--output-dir", default="data/clean_sft_repair_2048")
    parser.add_argument("--tokenizer", default=os.environ.get(
        "QWEN_TOKENIZER", os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen3-8B")
    ))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_path = output_dir / "train.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    manifest_path = output_dir / "manifest.json"

    existing = [p for p in (clean_path, rejected_path, manifest_path) if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"outputs already exist; pass --overwrite: {existing}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    counts = Counter()
    kept_by_source = Counter()
    rejected_by_source_reason = defaultdict(Counter)
    length_buckets = Counter()
    lengths_by_source = defaultdict(list)

    with open(input_path, encoding="utf-8") as src, \
            open(clean_path, "w", encoding="utf-8") as clean, \
            open(rejected_path, "w", encoding="utf-8") as rejected:
        for line_number, line in enumerate(src, 1):
            counts["input_lines"] += 1
            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                counts["invalid_json"] += 1
                rejected.write(json.dumps({
                    "reject_reason": "invalid_json", "line_number": line_number,
                    "error": str(exc), "raw_line": line.rstrip("\n"),
                }, ensure_ascii=False) + "\n")
                continue

            missing = [key for key in ("instruction", "reasoning", "answer")
                       if not isinstance(example.get(key), str) or not example[key].strip()]
            source = str(example.get("source", "unknown"))
            if missing:
                counts["missing_required_text"] += 1
                rejected_by_source_reason[source]["missing_required_text"] += 1
                rejected.write(json.dumps({
                    **example, "reject_reason": "missing_required_text",
                    "missing_fields": missing, "line_number": line_number,
                }, ensure_ascii=False) + "\n")
                continue

            token_length = len(tokenizer(
                format_math_sft(example), add_special_tokens=False
            )["input_ids"])
            lengths_by_source[source].append(token_length)
            if token_length <= 1024:
                length_buckets["le_1024"] += 1
            elif token_length <= args.max_length:
                length_buckets["1025_to_max"] += 1
            elif token_length <= 4096:
                length_buckets["max_to_4096"] += 1
            else:
                length_buckets["gt_4096"] += 1

            record = {**example, "sft_token_length": token_length}
            if token_length > args.max_length:
                counts["too_long"] += 1
                rejected_by_source_reason[source]["too_long"] += 1
                rejected.write(json.dumps({
                    **record, "reject_reason": "too_long",
                    "max_length": args.max_length, "line_number": line_number,
                }, ensure_ascii=False) + "\n")
                continue

            counts["kept"] += 1
            kept_by_source[source] += 1
            clean.write(json.dumps(record, ensure_ascii=False) + "\n")

    def source_stats(source, values):
        ordered = sorted(values)
        rejected = rejected_by_source_reason[source]
        return {
            "seen_valid": len(ordered),
            "kept": kept_by_source[source],
            "rejected_too_long": rejected["too_long"],
            "rejected_missing_required_text": rejected["missing_required_text"],
            "token_min": ordered[0],
            "token_p50": ordered[len(ordered) // 2],
            "token_p95": ordered[min(len(ordered) - 1, int(len(ordered) * .95))],
            "token_max": ordered[-1],
        }

    manifest = {
        "schema_version": 1,
        "policy": "keep_complete_examples_only_no_truncation",
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "tokenizer": args.tokenizer,
        "max_length": args.max_length,
        "format": "Qwen3 ChatML with complete think and answer blocks",
        "counts": dict(counts),
        "length_buckets": dict(length_buckets),
        "sources": {s: source_stats(s, v) for s, v in sorted(lengths_by_source.items())},
        "outputs": {"train": str(clean_path), "rejected": str(rejected_path)},
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
