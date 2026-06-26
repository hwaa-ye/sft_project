"""
数学推理评估脚本
评估维度：①答案正确 ②推理完整性 ③推理一致性
错误分类：计算错误 / 逻辑断裂 / 跳步 / 幻觉 / 格式错误 / 语义误解
"""

import json
import re
import os
from typing import Dict, List, Tuple


# ─── 1. 答案提取 ───
def extract_answer(text: str) -> str | None:
    """从模型输出中提取 <answer> 标签内的最终答案"""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # fallback: 取最后一行
    lines = text.strip().split("\n")
    return lines[-1].strip() if lines else None


# ─── 2. 答案比对 ───
def normalize_answer(ans: str) -> str:
    """数学答案标准化：去空格、统一符号"""
    ans = ans.strip().lower()
    ans = re.sub(r"\s+", "", ans)
    ans = ans.replace(" ", "").replace(",", "")
    return ans


def check_correct(pred: str, gold: str) -> bool:
    """比较预测答案与标准答案"""
    return normalize_answer(pred) == normalize_answer(gold)


# ─── 3. 推理完整性（简单版：检查 think 段落是否有足够步骤） ───
def check_reasoning_steps(text: str) -> Dict:
    """检查推理链是否有跳步"""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if not think_match:
        return {"has_think": False, "sentence_count": 0, "likely_skip": True}

    think_text = think_match.group(1)
    sentences = re.split(r"[。！\.!\n]+", think_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return {
        "has_think": True,
        "sentence_count": len(sentences),
        "likely_skip": len(sentences) < 3,  # 推理步骤少于3句可能跳步
    }


# ─── 4. 错误分类 ───
def classify_error(text: str, gold_answer: str, is_correct: bool, reasoning: Dict) -> str:
    """将错误归类到五种类型之一"""
    if is_correct:
        return "正确"

    if not reasoning["has_think"]:
        return "格式错误"

    if reasoning["likely_skip"]:
        return "跳步"

    pred = extract_answer(text)
    if pred is None:
        return "格式错误"

    # 简单启发式（后续可接 LLM-as-judge 细化）
    # 检查是否提取到了完全不相关的答案
    gold_norm = normalize_answer(gold_answer)
    pred_norm = normalize_answer(pred)

    # 如果有数字但不对 → 可能是计算错误
    has_number = bool(re.search(r"\d", pred_norm))
    if has_number:
        return "计算错误"

    return "语义误解"


# ─── 5. 单条评估 ───
def eval_one(pred_text: str, gold_answer: str) -> Dict:
    """对单条预测进行完整评估"""
    pred_answer = extract_answer(pred_text)
    is_correct = check_correct(pred_answer or "", gold_answer)
    reasoning = check_reasoning_steps(pred_text)
    error_type = classify_error(pred_text, gold_answer, is_correct, reasoning)

    return {
        "pred_answer": pred_answer,
        "gold_answer": gold_answer,
        "is_correct": is_correct,
        "has_think": reasoning["has_think"],
        "reasoning_steps": reasoning["sentence_count"],
        "likely_skip": reasoning["likely_skip"],
        "error_type": error_type,
    }


# ─── 6. 批量评估 + 报告 ───
def eval_dataset(data_path: str) -> Dict:
    """跑全量评估，输出统计报告"""
    results = []
    with open(data_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            pred = item.get("prediction", "")
            gold = item.get("answer", "")
            result = eval_one(pred, gold)
            result["index"] = i
            results.append(result)

    # 统计
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    errors = [r for r in results if not r["is_correct"]]

    error_counts = {}
    for r in errors:
        t = r["error_type"]
        error_counts[t] = error_counts.get(t, 0) + 1

    report = {
        "total": total,
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "error_distribution": error_counts,
        "avg_reasoning_steps": (
            sum(r["reasoning_steps"] for r in results) / total
            if total > 0
            else 0
        ),
        "skip_rate": sum(1 for r in results if r["likely_skip"]) / total
        if total > 0
        else 0,
    }

    return report


# ─── 7. 打印报告 ───
def print_report(report: Dict):
    print("=" * 50)
    print("评估报告")
    print("=" * 50)
    print(f"总样本数: {report['total']}")
    print(f"准确率:   {report['accuracy']:.2%}")
    print(f"正确:     {report['correct']}/{report['total']}")
    print(f"平均推理步数: {report['avg_reasoning_steps']:.1f}")
    print(f"跳步率:       {report['skip_rate']:.2%}")
    print("\n错误分布:")
    for error_type, count in sorted(
        report["error_distribution"].items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        pct = count / report["total"] * 100
        print(f"  {error_type}: {count} ({pct:.1f}%)")
    print("=" * 50)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        report = eval_dataset(sys.argv[1])
    else:
        # 演示用假数据
        demo = [
            {
                "prediction": "<think>\nx+2=5\nx=5-2\nx=3\n</think>\n<answer>3</answer>",
                "answer": "3",
            },
            {
                "prediction": "<think>\n就这样吧\n</think>\n<answer>5</answer>",
                "answer": "3",
            },
            {
                "prediction": "<answer>7</answer>",
                "answer": "3",
            },
        ]
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for d in demo:
            tmp.write(json.dumps(d, ensure_ascii=False) + "\n")
        tmp.close()
        report = eval_dataset(tmp.name)
        os.unlink(tmp.name)
    print_report(report)
