"""
截断诊断（统一口径）: 对多个 prediction 文件用同一套截断定义重新评估。

统一截断定义（一条预测只要命中任一条即视为 truncated/不完整）:
  1. 缺 </think>
  2. 缺 <answer>
  3. 缺 </answer>
  4. <answer>...</answer> 存在但抽取结果为空

complete = 四项全过。truncated = not complete。

用法:
  python scripts/diagnose_truncation.py
输出:
  - 终端打印
  - markdown 报告写入 analysis/<日期>_truncation_unified_sft_vs_grpo.md
"""
import json
import re
import statistics as st
import datetime
from collections import defaultdict

OUT_DIR = "analysis"
DATE = datetime.date.today().isoformat()
OUT_MD = f"{OUT_DIR}/{DATE}_truncation_unified_sft_vs_grpo.md"


# ─── 复用与 compare_eval.py 完全一致的答案判定逻辑 ───
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


# ─── 统一截断判定 ───
def classify(pred: str):
    """返回 (is_complete, reason)。"""
    if "</think>" not in pred:
        return False, "no_</think>"
    if "<answer>" not in pred:
        return False, "no_<answer>"
    if "</answer>" not in pred:
        return False, "no_</answer>"
    if not extract_answer(pred):
        return False, "empty_answer"
    return True, "complete"


def pct(sorted_list, p):
    if not sorted_list:
        return 0
    return sorted_list[min(len(sorted_list) - 1, int(len(sorted_list) * p))]


# md 累加器
_md = []
def emit(line=""):
    print(line)
    _md.append(line)


def diagnose(path, name):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            pred = d.get("prediction", "") or ""
            gold = str(d.get("answer", "")).strip()
            complete, reason = classify(pred)
            ans = extract_answer(pred)
            rows.append({
                "src": d.get("source", "?"), "complete": complete, "reason": reason,
                "correct": bool(ans and check_correct(ans, gold)),
                "out_len": len(pred), "gold_r_len": len(d.get("reasoning", "") or ""),
            })

    n = len(rows)
    trunc = sum(1 for r in rows if not r["complete"])
    correct = sum(1 for r in rows if r["correct"])
    out_lens = sorted(r["out_len"] for r in rows)

    emit(f"## {name}  (n={n})")
    emit()
    emit(f"- accuracy: **{100*correct/n:.2f}%**")
    emit(f"- truncation_rate: **{100*trunc/n:.2f}%** ({trunc}/{n})")
    emit(f"- complete_rate: {100*(n-trunc)/n:.2f}%")
    emit(f"- avg_output_len: {st.mean(out_lens):.0f} 字符")
    emit(f"- output_len p50/p90/p95/p99: {pct(out_lens,.5)} / {pct(out_lens,.9)} / {pct(out_lens,.95)} / {pct(out_lens,.99)}")

    reasons = defaultdict(int)
    for r in rows:
        if not r["complete"]:
            reasons[r["reason"]] += 1
    emit(f"- 截断原因分布: {dict(sorted(reasons.items(), key=lambda x: -x[1]))}")
    emit()

    emit("### by_source")
    emit()
    emit("| source | n | trunc | acc |")
    emit("|---|---|---|---|")
    by_src = defaultdict(lambda: {"n": 0, "trunc": 0, "correct": 0})
    for r in rows:
        s = by_src[r["src"]]
        s["n"] += 1
        s["trunc"] += 0 if r["complete"] else 1
        s["correct"] += 1 if r["correct"] else 0
    for s, v in sorted(by_src.items(), key=lambda x: -x[1]["trunc"] / max(1, x[1]["n"])):
        emit(f"| {s} | {v['n']} | {100*v['trunc']/v['n']:.1f}% | {100*v['correct']/v['n']:.1f}% |")
    emit()

    emit("### by_gold_reasoning_len (gold 参考解字符长度)")
    emit()
    emit("| 长度桶 | n | trunc |")
    emit("|---|---|---|")
    buckets = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 10**9)]
    labels = ["<1k", "1k-2k", "2k-4k", "4k-8k", ">8k"]
    for (lo, hi), lab in zip(buckets, labels):
        grp = [r for r in rows if lo <= r["gold_r_len"] < hi]
        if not grp:
            continue
        tr = sum(1 for r in grp if not r["complete"])
        emit(f"| {lab} | {len(grp)} | {100*tr/len(grp):.1f}% |")
    emit()
    return {"n": n, "acc": 100*correct/n, "trunc": 100*trunc/n, "name": name}


if __name__ == "__main__":
    emit(f"# 截断诊断（统一口径） — {DATE}")
    emit()
    emit("**统一截断定义**: 缺 `</think>` / 缺 `<answer>` / 缺 `</answer>` / answer 抽取为空，命中任一即判为截断。")
    emit()
    emit("> 注: 旧 compare_eval.py 只在「`</think>` 和 `</answer>` 同时缺失」时才算截断（过于宽松），"
         "本脚本统一为「任一缺失即截断」，因此截断率高于旧记录，这是尺子变准，不是模型变差。")
    emit()

    files = [
        ("data/test_predictions_v2.jsonl", "SFT (test_predictions_v2)"),
        ("output/grpo_qwen3/predictions_grpo_100.jsonl", "GRPO_100"),
        ("output/grpo_qwen3/predictions_grpo_200.jsonl", "GRPO_200"),
        ("output/grpo_qwen3/predictions_grpo_300.jsonl", "GRPO_300"),
    ]
    summary = []
    for path, name in files:
        try:
            summary.append(diagnose(path, name))
        except FileNotFoundError:
            emit(f"_[skip] 文件不存在: {path}_\n")

    emit("## 汇总对比（统一口径）")
    emit()
    emit("| 模型 | n | accuracy | truncation_rate |")
    emit("|---|---|---|---|")
    for s in summary:
        emit(f"| {s['name']} | {s['n']} | {s['acc']:.2f}% | {s['trunc']:.2f}% |")
    emit()

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(_md) + "\n")
    print(f"\n[saved] {OUT_MD}")
