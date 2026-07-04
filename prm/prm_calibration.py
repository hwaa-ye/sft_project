"""
前沿实验2: PRM 分数与最优策略的 Gap 分析

核心问题: PRM 高分 = 推理质量高 ≠ 一定能得到正确答案。
量化 PRM 分数和"真正会做题"之间的 gap，识别系统性偏差来源。

不需要 GPU，纯数据分析。
"""
import json, sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from grpo_reward import check_answer


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "prm/prm_train_data.jsonl"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "prm/prm_calibration.json"

    samples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"加载 {len(samples)} 条标注数据")

    # ─── 1. 样本级分析: PRM 均分 vs 答案正确性 ───
    sample_results = []
    for item in samples:
        steps = item["steps"]
        if not steps:
            continue
        prm_avg = np.mean([s["score"] for s in steps])
        prm_std = np.std([s["score"] for s in steps])
        prm_min = min(s["score"] for s in steps)
        is_correct = check_answer(
            str(item.get("full_answer", "") or ""),
            str(item.get("answer", ""))
        )
        sample_results.append({
            "question": item["question"][:100],
            "n_steps": len(steps),
            "prm_avg": prm_avg,
            "prm_std": prm_std,
            "prm_min": prm_min,
            "is_correct": is_correct,
        })

    # ─── 2. 四象限分析 ───
    # 用 PRM 中位数作为高分/低分阈值
    prm_avgs = [r["prm_avg"] for r in sample_results]
    threshold = np.median(prm_avgs)

    tp = 0  # PRM高, 答案对 → 真优秀
    fp = 0  # PRM高, 答案错 → PRM被蒙骗 (假阳性)
    tn = 0  # PRM低, 答案错 → 真差
    fn = 0  # PRM低, 答案对 → PRM太严 (假阴性)

    fp_cases = []
    fn_cases = []

    for r in sample_results:
        prm_high = r["prm_avg"] >= threshold
        if prm_high and r["is_correct"]:
            tp += 1
        elif prm_high and not r["is_correct"]:
            fp += 1
            fp_cases.append(r)
        elif not prm_high and not r["is_correct"]:
            tn += 1
        elif not prm_high and r["is_correct"]:
            fn += 1
            fn_cases.append(r)

    total = len(sample_results)
    print(f"\n{'='*55}")
    print(f"PRM 分数 vs 答案正确性 — 四象限分析")
    print(f"{'='*55}")
    print(f"阈值 (PRM中位数): {threshold:.3f}")
    print(f"\n              答案正确        答案错误")
    print(f"PRM高分      TP={tp:4d} ({tp/total*100:4.1f}%)    FP={fp:4d} ({fp/total*100:4.1f}%)")
    print(f"PRM低分      FN={fn:4d} ({fn/total*100:4.1f}%)    TN={tn:4d} ({tn/total*100:4.1f}%)")

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # 假阳性率
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # 假阴性率
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"\n指标:")
    print(f"  假阳性率 (PRM高但答案错): {fpr:.2%}  ← PRM被蒙骗")
    print(f"  假阴性率 (PRM低但答案对): {fnr:.2%}  ← PRM太严")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")

    # ─── 3. 假阳性分析 (PRM高分但答案错) ───
    print(f"\n--- 假阳性 (FP) 分析: {len(fp_cases)} 个样本 ---")
    if fp_cases:
        print(f"  平均 PRM: {np.mean([c['prm_avg'] for c in fp_cases]):.3f}")
        print(f"  平均步数: {np.mean([c['n_steps'] for c in fp_cases]):.1f}")
        print(f"  PRM 方差:  {np.mean([c['prm_std'] for c in fp_cases]):.3f}")

        # 假阳性子类型
        high_std = [c for c in fp_cases if c["prm_std"] > 0.2]  # 分数波动大 → 某步有问题
        low_std = [c for c in fp_cases if c["prm_std"] <= 0.2]  # 全高分 → 隐蔽逻辑错误
        print(f"  高波动 (某步明显差): {len(high_std)}个")
        print(f"  低波动 (全部高分):   {len(low_std)}个 ← 隐蔽逻辑错误, API也没察觉")

        print(f"\n  典型 FP case (全部高分但答案错):")
        for c in sorted(low_std, key=lambda x: x["prm_avg"], reverse=True)[:3]:
            print(f"    PRM={c['prm_avg']:.2f} σ={c['prm_std']:.2f} | {c['question'][:80]}...")

    # ─── 4. 假阴性分析 (PRM低分但答案对) ───
    print(f"\n--- 假阴性 (FN) 分析: {len(fn_cases)} 个样本 ---")
    if fn_cases:
        print(f"  平均 PRM: {np.mean([c['prm_avg'] for c in fn_cases]):.3f}")
        print(f"  平均步数: {np.mean([c['n_steps'] for c in fn_cases]):.1f}")

        short = [c for c in fn_cases if c["n_steps"] <= 3]   # 简略但正确
        has_low = [c for c in fn_cases if c["prm_min"] < 0.5]  # 某步被严重低估
        print(f"  简略推理 (≤3步): {len(short)}个 ← PRM偏好长推理")
        print(f"  有低分步骤 (<0.5): {len(has_low)}个 ← 某步被误判")

        print(f"\n  典型 FN case (简略但正确):")
        for c in sorted(short, key=lambda x: x["prm_avg"])[:3]:
            print(f"    PRM={c['prm_avg']:.2f} steps={c['n_steps']} | {c['question'][:80]}...")

    # ─── 5. 按 PRM 分数分桶，看各桶的准确率 ───
    print(f"\n--- 分数分桶准确率 ---")
    buckets = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0)]
    print(f"  {'区间':<15} {'样本数':>6} {'准确率':>8} {'解读'}")
    for lo, hi in buckets:
        in_bucket = [r for r in sample_results if lo <= r["prm_avg"] < hi]
        if in_bucket:
            acc = np.mean([r["is_correct"] for r in in_bucket])
            note = "✓ 良好校准" if (lo < 0.5 and acc < 0.5) or (lo >= 0.5 and acc >= 0.5) else "⚠ 偏离"
            print(f"  [{lo:.1f}, {hi:.1f})        {len(in_bucket):>4}    {acc:>7.2%}  {note}")

    # ─── 6. 保存总结 ───
    result = {
        "n_samples": total,
        "threshold": float(threshold),
        "confusion": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "metrics": {
            "FPR": float(fpr), "FNR": float(fnr),
            "precision": float(precision), "recall": float(recall),
        },
        "fp_summary": {
            "count": len(fp_cases),
            "high_variance": len([c for c in fp_cases if c["prm_std"] > 0.2]),
            "low_variance": len([c for c in fp_cases if c["prm_std"] <= 0.2]),
        },
        "fn_summary": {
            "count": len(fn_cases),
            "short_reasoning": len([c for c in fn_cases if c["n_steps"] <= 3]),
            "has_low_step": len([c for c in fn_cases if c["prm_min"] < 0.5]),
        },
        "fp_cases": fp_cases,
        "fn_cases": fn_cases,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存至: {output_path}")

    # ─── 结论模板 ───
    print(f"\n{'='*55}")
    print("关键发现 (面试叙事)")
    print(f"{'='*55}")
    print(f"1. PRM 假阳性率 {fpr:.1%}: 约 {fp}/{fp+tp} 的高分样本实际答案错误")
    print(f"   → 这说明 PRM 存在系统性盲区，不能完全替代答案验证")
    print(f"2. PRM 假阴性率 {fnr:.1%}: 约 {fn}/{fn+tp} 的正确样本被 PRM 低估")
    print(f"   → 主要原因是推理简略但正确，PRM 偏好冗长推理")
    print(f"3. 这解释了为什么最终 reward 设计需要保留 accuracy 维度兜底")
    print(f"   → PRM 做过程监督 + accuracy 做结果监督 = 互补")


if __name__ == "__main__":
    main()
