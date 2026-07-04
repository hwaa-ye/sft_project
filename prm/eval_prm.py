"""
PRM vs 规则 Reward 对比评估
对同一批推理链，分别用规则 reward 和 PRM reward 打分，分析差异
"""
import json, sys, os, re
import numpy as np

# 加入项目脚本路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from grpo_reward import compute_reward, check_answer


def normalize_text(s):
    return re.sub(r"\s+", " ", str(s).strip())


def main():
    pred_file = sys.argv[1] if len(sys.argv) > 1 else "prm/gen_predictions.jsonl"
    prm_file = sys.argv[2] if len(sys.argv) > 2 else "prm/prm_train_data.jsonl"
    output = sys.argv[3] if len(sys.argv) > 3 else "prm/prm_comparison.json"

    # 加载规则 reward 打分
    samples = []
    with open(pred_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"加载 {len(samples)} 条预测数据")

    rule_rewards = []
    for item in samples:
        r = compute_reward(item["prediction"], str(item.get("answer", "")))
        rule_rewards.append(r)

    # 加载 PRM 打分
    prm_data = {}
    if os.path.exists(prm_file):
        with open(prm_file, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                key = normalize_text(item["question"])
                prm_data[key] = item

    # ─── 对比分析 ───
    n_matched = 0
    diffs = []
    for i, (item, rw) in enumerate(zip(samples, rule_rewards)):
        key = normalize_text(item.get("instruction", ""))
        prm = prm_data.get(key)
        if prm is None:
            continue
        n_matched += 1
        diffs.append({
            "idx": i,
            "rule_total": rw["total"],
            "rule_acc": rw["accuracy"],
            "rule_comp": rw["completeness"],
            "rule_eff": rw["efficiency"],
            "prm_avg": prm["avg_score"],
            "prm_steps": len(prm["steps"]),
            "delta": rw["total"] - prm["avg_score"],
        })

    print(f"匹配: {n_matched}/{len(samples)} 条")

    if not diffs:
        print("无匹配数据，退出")
        return

    # 统计
    deltas = [d["delta"] for d in diffs]
    prm_scores = [d["prm_avg"] for d in diffs]
    rule_totals = [d["rule_total"] for d in diffs]
    prm_steps = [d["prm_steps"] for d in diffs]

    print(f"\n{'='*50}")
    print(f"PRM vs 规则 Reward 对比")
    print(f"{'='*50}")
    print(f"规则 reward 均值: {np.mean(rule_totals):.3f} ± {np.std(rule_totals):.3f}")
    print(f"PRM 均值:         {np.mean(prm_scores):.3f} ± {np.std(prm_scores):.3f}")
    print(f"平均差异 (rule-PRM): {np.mean(deltas):.3f}")
    print(f"PRM 平均步骤数: {np.mean(prm_steps):.1f}")

    # 相关性
    corr = np.corrcoef(rule_totals, prm_scores)[0, 1]
    print(f"\n规则-PRM 相关性: {corr:.3f}")

    # 差异最大的 case（PRM > 规则：可能是正确但结构不完整）
    print(f"\n--- PRM >> 规则的 3 个 case (PRM 认出了规则漏掉的) ---")
    worst = sorted(diffs, key=lambda d: d["delta"])[:3]
    for d in worst:
        item = samples[d["idx"]]
        print(f"\n  idx={d['idx']} | PRM={d['prm_avg']:.2f} rule={d['rule_total']:.2f}")
        print(f"  Q: {item['instruction'][:150]}")
        print(f"  A: {item['prediction'][:200]}...")

    # 差异最小的 case（规则 > PRM：可能是答案蒙对但推理差）
    print(f"\n--- 规则 >> PRM 的 3 个 case (规则高分但推理有问题) ---")
    best = sorted(diffs, key=lambda d: d["delta"], reverse=True)[:3]
    for d in best:
        item = samples[d["idx"]]
        print(f"\n  idx={d['idx']} | PRM={d['prm_avg']:.2f} rule={d['rule_total']:.2f}")
        print(f"  Q: {item['instruction'][:150]}")
        print(f"  A: {item['prediction'][:200]}...")

    # 保存详细对比
    result = {
        "n_matched": n_matched,
        "rule_mean": float(np.mean(rule_totals)),
        "prm_mean": float(np.mean(prm_scores)),
        "correlation": float(corr),
        "mean_delta": float(np.mean(deltas)),
        "diffs": [{k: v for k, v in d.items()} for d in diffs],
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n详细对比保存至: {output}")


if __name__ == "__main__":
    main()
