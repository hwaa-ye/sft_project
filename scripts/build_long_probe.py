"""Build a reliable-gold probe from long examples capped at 2048 generation tokens."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from repair_sft_answers import extract_balanced_boxed, record_key
from diagnose_truncation import classify


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-eval", default="data/test_predictions_v2.jsonl")
    parser.add_argument("--new-predictions", default="data/predictions_clean_sft_repair_final.jsonl")
    parser.add_argument("--mathr", default="data/train_math.jsonl")
    parser.add_argument("--r1", default="data/train_math_r1.jsonl")
    parser.add_argument("--tokenizer", default=str(Path.home() / ".cache/modelscope/hub/models/Qwen/Qwen3-8B"))
    parser.add_argument("--min-gold-reasoning-chars", type=int, default=4000)
    parser.add_argument("--generation-cap", type=int, default=2048)
    parser.add_argument("--output", default="data/long_cap_probe_45.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    intermediate = defaultdict(list)
    for path in (args.mathr, args.r1):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                intermediate[record_key(record)].append(record)

    old_rows = [json.loads(line) for line in open(args.old_eval, encoding="utf-8")]
    new_rows = [json.loads(line) for line in open(args.new_predictions, encoding="utf-8")]
    if len(old_rows) != len(new_rows):
        raise ValueError("evaluation and prediction row counts differ")

    output = []
    for eval_index, (old, new) in enumerate(zip(old_rows, new_rows)):
        if old.get("instruction") != new.get("instruction"):
            raise ValueError(f"row {eval_index}: instruction mismatch")
        if len(old.get("reasoning", "")) < args.min_gold_reasoning_chars:
            continue
        if classify(new.get("prediction", ""))[0]:
            continue
        prediction = new.get("prediction", "")
        start = prediction.find("<think>")
        generated = prediction[start:] if start >= 0 else prediction
        generated_tokens = len(tokenizer(generated, add_special_tokens=False)["input_ids"])
        if generated_tokens < args.generation_cap:
            continue

        answers = []
        for source_record in intermediate.get(record_key(old), []):
            answers.extend(
                value for value in extract_balanced_boxed(source_record.get("answer", "")) if value
            )
        if not answers:
            continue
        output.append({
            "instruction": old["instruction"],
            "reasoning": old.get("reasoning", ""),
            "answer": answers[-1].strip(),
            "source": old.get("source", "unknown"),
            "eval_index": eval_index,
            "gold_reasoning_chars": len(old.get("reasoning", "")),
            "prediction_2048": prediction,
            "prediction_2048_tokens": generated_tokens,
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in output:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output_path), "samples": len(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
