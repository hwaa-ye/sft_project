"""Prepare deduplicated long-example candidates for Stage-2 reasoning SFT."""

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from repair_sft_answers import extract_balanced_boxed, record_key


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rejected", default="data/clean_sft_repair_2048/rejected.jsonl")
    parser.add_argument("--mathr", default="data/train_math.jsonl")
    parser.add_argument("--r1", default="data/train_math_r1.jsonl")
    parser.add_argument("--eval", default="data/test_predictions_v2.jsonl")
    parser.add_argument("--output-dir", default="data/hard_stage2_candidates")
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.85)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_question(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text)


def repair_known_latex_control_chars(text):
    """Repair control characters produced by interpreting \b/\f in LaTeX."""
    return text.replace("\x08oxed", r"\boxed").replace("\x0crac", r"\frac")


def has_unsupported_control_char(text):
    return any(ord(ch) < 32 and ch not in "\n\r\t" for ch in text)


def ngrams(text, n=5):
    if len(text) <= n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def near_eval_duplicate(question, eval_signatures, threshold):
    normalized = normalize_question(question)
    signature = ngrams(normalized)
    for eval_normalized, eval_signature in eval_signatures:
        length_ratio = len(normalized) / max(1, len(eval_normalized))
        if not 0.75 <= length_ratio <= 1.33:
            continue
        union = len(signature | eval_signature)
        if union and len(signature & eval_signature) / union >= threshold:
            return True
    return False


def stable_score(record):
    payload = f"{record.get('source', '')}\n{normalize_question(record['instruction'])}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def length_bucket(token_length):
    if token_length <= 4096:
        return "2049-4096"
    if token_length <= 8192:
        return "4097-8192"
    return ">8192"


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_candidates.jsonl"
    validation_path = output_dir / "hard_validation.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    manifest_path = output_dir / "manifest.json"
    outputs = (train_path, validation_path, rejected_path, manifest_path)
    if any(path.exists() for path in outputs) and not args.overwrite:
        raise FileExistsError(f"outputs exist in {output_dir}; pass --overwrite")

    intermediate = defaultdict(list)
    for path in (args.mathr, args.r1):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                intermediate[record_key(record)].append(record)

    eval_signatures = []
    with open(args.eval, encoding="utf-8") as handle:
        for line in handle:
            question = json.loads(line).get("instruction", "")
            normalized = normalize_question(question)
            eval_signatures.append((normalized, ngrams(normalized)))

    counts = Counter()
    candidates_by_question = defaultdict(list)
    rejected_records = []
    with open(args.rejected, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("reject_reason") != "too_long":
                continue
            counts["too_long_input"] += 1
            answers = []
            for source_record in intermediate.get(record_key(record), []):
                answers.extend(
                    value for value in extract_balanced_boxed(source_record.get("answer", "")) if value
                )
            if not answers:
                counts["no_reliable_answer"] += 1
                rejected_records.append({**record, "stage2_reject_reason": "no_reliable_answer"})
                continue
            record["answer"] = answers[-1].strip()
            record["answer_repair_method"] = "intermediate_answer_last_nonempty_balanced_box"
            for field in ("instruction", "reasoning", "answer"):
                record[field] = repair_known_latex_control_chars(record.get(field, ""))
            if any(has_unsupported_control_char(record[field])
                   for field in ("instruction", "reasoning", "answer")):
                counts["unsupported_control_character"] += 1
                rejected_records.append({
                    **record, "stage2_reject_reason": "unsupported_control_character"
                })
                continue
            normalized = normalize_question(record["instruction"])
            if any(normalized == item[0] for item in eval_signatures):
                counts["exact_eval_duplicate"] += 1
                rejected_records.append({**record, "stage2_reject_reason": "exact_eval_duplicate"})
                continue
            if near_eval_duplicate(record["instruction"], eval_signatures, args.near_duplicate_threshold):
                counts["near_eval_duplicate"] += 1
                rejected_records.append({**record, "stage2_reject_reason": "near_eval_duplicate"})
                continue
            candidates_by_question[normalized].append(record)

    deduplicated = []
    for records in candidates_by_question.values():
        answers = {record["answer"] for record in records}
        if len(answers) != 1:
            counts["duplicate_conflicting_answers"] += len(records)
            for record in records:
                rejected_records.append({
                    **record, "stage2_reject_reason": "duplicate_conflicting_answers"
                })
            continue
        records.sort(key=lambda item: (item.get("sft_token_length", 10**9), stable_score(item)))
        deduplicated.append(records[0])
        for duplicate in records[1:]:
            counts["internal_duplicate"] += 1
            rejected_records.append({**duplicate, "stage2_reject_reason": "internal_duplicate"})

    strata = defaultdict(list)
    for record in deduplicated:
        key = (str(record.get("source", "unknown")), length_bucket(record["sft_token_length"]))
        strata[key].append(record)
    for records in strata.values():
        records.sort(key=stable_score)

    total = len(deduplicated)
    target = min(args.validation_size, total)
    allocations = {}
    assigned = 0
    for key, records in sorted(strata.items()):
        amount = min(len(records), int(target * len(records) / max(1, total)))
        allocations[key] = amount
        assigned += amount
    remainder = target - assigned
    for key in sorted(strata, key=lambda item: (-len(strata[item]), item)):
        if remainder <= 0:
            break
        if allocations[key] < len(strata[key]):
            allocations[key] += 1
            remainder -= 1

    validation = []
    train = []
    for key, records in strata.items():
        amount = allocations[key]
        validation.extend(records[:amount])
        train.extend(records[amount:])
    validation.sort(key=stable_score)
    train.sort(key=stable_score)

    for path, records in ((train_path, train), (validation_path, validation),
                          (rejected_path, rejected_records)):
        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts["deduplicated_eligible"] = len(deduplicated)
    counts["train_candidates"] = len(train)
    counts["hard_validation"] = len(validation)
    manifest = {
        "schema_version": 1,
        "policy": "reliable_answer_long_samples_eval_deduplicated",
        "near_duplicate_threshold": args.near_duplicate_threshold,
        "counts": dict(counts),
        "validation_strata": {
            f"{source}|{bucket}": allocations[(source, bucket)]
            for source, bucket in sorted(allocations)
        },
        "outputs": {
            "train_candidates": str(train_path),
            "hard_validation": str(validation_path),
            "rejected": str(rejected_path),
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
