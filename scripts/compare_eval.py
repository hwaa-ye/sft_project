"""GRPO vs SFT 对比评估：一键跑4组"""
import json, re, sys

def normalize_answer(ans):
    ans = str(ans).strip().lower()
    ans = re.sub(r"\\text\{(.*?)\}", r"\1", ans)
    ans = re.sub(r"\\boxed\{(.*?)\}", r"\1", ans)
    ans = re.sub(r"\\mathrm\{(.*?)\}", r"\1", ans)
    ans = ans.replace("×", "错误").replace("✗", "错误").replace("✓", "正确").replace("√", "正确")
    ans = re.sub(r"\s+", "", ans)
    return ans.strip("\"'$")

def extract_number(s):
    s = re.sub(r"\s+", "", str(s))
    m = re.match(r"-?[\d,]+\.?\d*", s)
    if m:
        try:
            return float(m.group().replace(",", ""))
        except ValueError:
            pass
    return None

def check_correct(pred, gold):
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p == g:
        return True
    pn, gn = extract_number(p), extract_number(g)
    if pn is not None and gn is not None and abs(pn - gn) < 1e-6:
        return True
    return False

def extract_answer(text):
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else None

def eval_file(path, name):
    results = []
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            pred = item.get("prediction", "")
            gold = str(item.get("answer", "")).strip()
            pred_ans = extract_answer(pred)
            is_correct = check_correct(pred_ans or "", gold)
            has_think = "</think>" in pred
            has_answer = "</answer>" in pred
            complete = has_think and has_answer
            think_len = 0
            if has_think:
                tm = re.search(r"<think>(.*?)</think>", pred, re.DOTALL)
                think_len = len(tm.group(1)) if tm else 0
            truncated = not has_think and not has_answer

            error_type = "correct"
            if not is_correct:
                if truncated:
                    error_type = "truncated"
                elif not has_answer:
                    error_type = "no_answer"
                else:
                    error_type = "wrong"

            results.append({
                "is_correct": is_correct, "has_think": has_think,
                "has_answer": has_answer, "complete": complete,
                "think_len": think_len, "error_type": error_type
            })
            samples.append({
                "pred_ans": pred_ans, "gold": gold,
                "has_think": has_think, "has_answer": has_answer,
                "think_len": think_len, "is_correct": is_correct
            })

    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    complete_ct = sum(1 for r in results if r["complete"])
    truncated_ct = sum(1 for r in results if r["error_type"] == "truncated")
    wrong_ct = sum(1 for r in results if r["error_type"] == "wrong")
    no_ans_ct = sum(1 for r in results if r["error_type"] == "no_answer")
    think_lens = [r["think_len"] for r in results if r["think_len"] > 0]
    avg_think = sum(think_lens) / len(think_lens) if think_lens else 0
    # 有think的样本中，think长度的中位数（更有参考价值）
    think_lens_sorted = sorted(think_lens)
    median_think = think_lens_sorted[len(think_lens_sorted)//2] if think_lens_sorted else 0

    return {
        "name": name, "total": total,
        "accuracy": correct / total,
        "completeness": complete_ct / total,
        "truncated_rate": truncated_ct / total,
        "wrong_rate": wrong_ct / total,
        "no_answer_rate": no_ans_ct / total,
        "avg_think_len": avg_think,
        "median_think_len": median_think,
        "samples": samples,
    }

paths = {
    "SFT": "data/test_predictions_v2.jsonl",
    "GRPO_100": "output/grpo_qwen3/predictions_grpo_100.jsonl",
    "GRPO_200": "output/grpo_qwen3/predictions_grpo_200.jsonl",
    "GRPO_300": "output/grpo_qwen3/predictions_grpo_300.jsonl",
}

results = []
for name, path in paths.items():
    r = eval_file(path, name)
    results.append(r)

# 打印对比表
print(f"{'Model':<12} {'Acc':>8} {'Comp':>8} {'Trunc':>8} {'Wrong':>8} {'NoAns':>8} {'Think(avg)':>10} {'Think(med)':>10}")
print("-" * 75)
for r in results:
    print(f"{r['name']:<12} {r['accuracy']:>7.2%} {r['completeness']:>7.2%} {r['truncated_rate']:>7.2%} {r['wrong_rate']:>7.2%} {r['no_answer_rate']:>7.2%} {r['avg_think_len']:>10.0f} {r['median_think_len']:>10.0f}")

print()
print("vs SFT 变化 (Δ):")
print(f"{'Model':<12} {'ΔAcc':>8} {'ΔComp':>8} {'ΔTrunc':>8}")
sft = results[0]
for r in results[1:]:
    da = r["accuracy"] - sft["accuracy"]
    dc = r["completeness"] - sft["completeness"]
    dt = r["truncated_rate"] - sft["truncated_rate"]
    print(f"{r['name']:<12} {da:>+7.2%} {dc:>+7.2%} {dt:>+7.2%}")

# GRPO 修复了多少 SFT badcase
print()
sft_samples = {i: s for i, s in enumerate(sft["samples"])}
for grpo_r in results[1:]:
    fixed = 0
    broken = 0
    for i, s in enumerate(grpo_r["samples"]):
        if i >= len(sft_samples):
            break
        sft_wrong = not sft_samples[i]["is_correct"]
        grpo_correct = s["is_correct"]
        if sft_wrong and grpo_correct:
            fixed += 1
        elif not sft_wrong and not grpo_correct:
            broken += 1
    print(f"{grpo_r['name']}: SFT错误→GRPO正确={fixed}, SFT正确→GRPO错误={broken}")
