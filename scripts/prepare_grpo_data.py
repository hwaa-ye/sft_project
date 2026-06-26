"""
GRPO 数据准备:
  1. MATH 数据集 → 提取 prompt + reference_answer
  2. Firefly Cot → 筛选数学相关 + 提取答案
  3. 合并去重 → 按难度分层 split → data/grpo_train.jsonl / data/grpo_val.jsonl
"""

import json
import re
import os
import random

MATH_PATH = "data/train_math.jsonl"
FIREFLY_PATH = "data/train.jsonl"
OUTPUT_TRAIN = "data/grpo_train.jsonl"
OUTPUT_VAL = "data/grpo_val.jsonl"

random.seed(42)

# ─── MATH 提取 ───

def extract_boxed(text: str) -> str | None:
    idx = text.find("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start:i - 1].strip()
    return None


def load_math() -> list[dict]:
    prompts = []
    with open(MATH_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ref = extract_boxed(d.get("answer", ""))
            if ref is None:
                continue
            prompts.append({
                "prompt": d["instruction"],
                "reference_answer": ref,
                "source": "math",
            })
    return prompts


# ─── Firefly Cot 提取 ───

MATH_INDICATORS = [
    r"\d+\s*[\+\-\*\/\×\÷]\s*\d+",
    r"[＝=]\s*\d+",
    r"\\\\frac", r"\\\\sqrt",
    r"\bx\s*=", r"\by\s*=",
    r"\d+°",
    r"(sum|product|total|average|probability|percent)",
    r"(多少|计算|求解|等于|方程|面积|体积|角度|概率)",
    r"(angle|triangle|circle|square|rectangle)",
]


def load_firefly() -> list[dict]:
    prompts = []
    with open(FIREFLY_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["kind"] != "Cot":
                continue

            combined = d["input"] + " " + d["target"]
            if not any(re.search(pat, combined, re.IGNORECASE) for pat in MATH_INDICATORS):
                continue

            m = re.search(r"答案是[：:]\s*(.+)", d["target"], re.DOTALL)
            if not m:
                continue

            raw = m.group(1).strip().split("\n")[0].strip().rstrip(".。")
            if len(raw) > 50:
                continue

            prompts.append({
                "prompt": d["input"],
                "reference_answer": raw.strip(),
                "source": "firefly",
            })
    return prompts


# ─── 合并去重 ───

def merge_and_dedup(math_prompts, firefly_prompts):
    seen = set()
    merged = []

    # MATH 优先保留
    for p in math_prompts:
        key = p["prompt"].strip().lower()
        if key not in seen:
            seen.add(key)
            merged.append(p)

    math_count = len(merged)

    for p in firefly_prompts:
        key = p["prompt"].strip().lower()
        if key not in seen:
            seen.add(key)
            merged.append(p)

    firefly_added = len(merged) - math_count
    return merged, math_count, firefly_added


def main():
    math_prompts = load_math()
    print(f"MATH: {len(math_prompts)} 条")

    firefly_prompts = load_firefly()
    print(f"Firefly 数学 Cot: {len(firefly_prompts)} 条")

    all_prompts, math_count, firefly_added = merge_and_dedup(math_prompts, firefly_prompts)
    random.shuffle(all_prompts)

    split = int(len(all_prompts) * 0.95)
    train = all_prompts[:split]
    val = all_prompts[split:]

    for path, data in [(OUTPUT_TRAIN, train), (OUTPUT_VAL, val)]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for p in data:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    train_lens = [len(p["prompt"]) for p in train]
    train_lens.sort()

    print(f"\n合并去重后: {len(all_prompts)} 条 (MATH {math_count} + Firefly {firefly_added})")
    print(f"train: {len(train)} 条 → {OUTPUT_TRAIN}")
    print(f"val:   {len(val)} 条 → {OUTPUT_VAL}")
    print(f"prompt 长度: min={train_lens[0]}, max={train_lens[-1]}, median={train_lens[len(train_lens)//2]}")


if __name__ == "__main__":
    main()
