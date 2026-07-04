"""
前沿实验1: MC Rollout vs API 标注 vs 结果反推 — 三种过程评分方法对比

Math-Shepherd (2024) 提出用 Monte Carlo rollout 自动估计步骤质量:
从某步出发随机采样N条后续推理, 最终正确的比例作为该步的"过程分数"。

本实验对比三种标注方法的相关性与分歧模式, 揭示标注可扩展性瓶颈。
"""
import json, os, sys, re, gc, torch, time
import numpy as np
from collections import defaultdict

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from grpo_reward import check_answer

# ─── 配置 ───
ANNOTATED_DATA = os.environ.get("MC_DATA", "prm/prm_train_data.jsonl")
OUTPUT = os.environ.get("MC_OUTPUT", "prm/mc_comparison.json")
N_SAMPLES = int(os.environ.get("MC_N_SAMPLES", "50"))     # 选多少条推理链
N_ROLLOUTS = int(os.environ.get("MC_N_ROLLOUTS", "5"))    # 每步采样几条后续
ROLLOUT_TOKENS = int(os.environ.get("MC_TOKENS", "512"))  # 后续推理最大 token 数

# ─── 加载 GRPO 模型（需要 GPU） ───
def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    BASE = os.path.join(os.path.expanduser("~"), "autodl-fs", "model_cache", "Qwen", "Qwen3-8B")
    LORA = os.environ.get("MC_LORA", "output/grpo_qwen3/step_200")

    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map=None,
        trust_remote_code=True, local_files_only=True,
    ).to("cuda:0")
    model = PeftModel.from_pretrained(model, LORA).to("cuda:0")
    model.eval()
    return model, tokenizer


def mc_score_step(model, tokenizer, question, prefix_steps, step_idx, gold_answer):
    """
    Math-Shepherd 风格: 从当前步骤之后, 采样 N 条后续推理,
    正确的比例作为该步骤的 MC 分数。

    prefix_steps: 到此步为止的所有步骤文本列表
    step_idx: 当前是第几步 (0-indexed)
    """
    # 构建 prefix: 问题 + 前面的推理
    prefix = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    for s in prefix_steps[:step_idx + 1]:
        prefix += s + "\n\n"

    correct = 0
    for _ in range(N_ROLLOUTS):
        inputs = tokenizer(prefix, return_tensors="pt", truncation=True,
                           max_length=1536).to("cuda:0")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=ROLLOUT_TOKENS, temperature=0.8,
                do_sample=True, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = (inputs.attention_mask[0] == 1).sum().item()
        completion = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)

        # 检查最终答案是否正确
        ans_m = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
        pred_ans = ans_m.group(1).strip() if ans_m else completion.strip()
        if check_answer(pred_ans, str(gold_answer)):
            correct += 1

    return correct / N_ROLLOUTS


def outcome_score(gold_answer, pred_answer):
    """结果反推: 最终答案正确 → 所有步骤标1, 错误 → 标0"""
    return 1.0 if check_answer(str(pred_answer or ""), str(gold_answer)) else 0.0


def main():
    # 加载标注数据
    samples = []
    with open(ANNOTATED_DATA, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"加载 {len(samples)} 条标注数据")

    # 选取子集（优先选有分歧的: API 分数方差大的）
    samples.sort(key=lambda s: np.std([st["score"] for st in s["steps"]]), reverse=True)
    samples = samples[:N_SAMPLES]
    print(f"选取 {len(samples)} 条（优先高方差样本）")

    # 准备模型
    print("加载模型...")
    model, tokenizer = load_model()
    print(f"显存: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    # ─── 三种方法打分 ───
    results = []
    t0 = time.time()

    for s_idx, item in enumerate(samples):
        question = item["question"]
        gold_answer = item.get("answer", "")
        pred_answer = item.get("full_answer", "")
        steps = item["steps"]

        api_scores = [st["score"] for st in steps]
        mc_scores = []
        outcome_scores = []

        for st in steps:
            step_texts = [s["step_text"] for s in steps]
            # MC rollout
            mc = mc_score_step(model, tokenizer, question, step_texts, st["step_idx"], gold_answer)
            mc_scores.append(mc)

            # 结果反推
            outcome_scores.append(outcome_score(gold_answer, pred_answer))

        api_mean = np.mean(api_scores)
        mc_mean = np.mean(mc_scores)
        out_mean = np.mean(outcome_scores)

        results.append({
            "question": question[:100],
            "n_steps": len(steps),
            "api_scores": api_scores,
            "mc_scores": mc_scores,
            "outcome_scores": outcome_scores,
            "api_mean": api_mean,
            "mc_mean": mc_mean,
            "outcome_mean": out_mean,
            "answer_correct": 1.0 if check_answer(pred_answer or "", str(gold_answer)) else 0.0,
        })

        if (s_idx + 1) % 10 == 0:
            elapsed = (time.time() - t0) / 60
            eta = elapsed / (s_idx + 1) * (len(samples) - s_idx - 1)
            print(f"  [{s_idx+1}/{len(samples)}] elapsed={elapsed:.0f}min eta={eta:.0f}min", flush=True)

    # ─── 分析 ───
    all_api = []
    all_mc = []
    all_out = []
    for r in results:
        all_api.extend(r["api_scores"])
        all_mc.extend(r["mc_scores"])
        all_out.extend(r["outcome_scores"])

    all_api = np.array(all_api)
    all_mc = np.array(all_mc)
    all_out = np.array(all_out)

    print(f"\n{'='*55}")
    print("MC Rollout vs API 标注 vs 结果反推 — 对比分析")
    print(f"{'='*55}")
    print(f"样本: {len(results)} 条推理链, {len(all_api)} 个步骤")
    print(f"MC rollout: N={N_ROLLOUTS}, max_tokens={ROLLOUT_TOKENS}")

    # 相关性
    corr_am = np.corrcoef(all_api, all_mc)[0, 1]
    corr_ao = np.corrcoef(all_api, all_out)[0, 1]
    corr_mo = np.corrcoef(all_mc, all_out)[0, 1]
    print(f"\n相关性矩阵:")
    print(f"  API-MC:      {corr_am:+.3f}")
    print(f"  API-Outcome: {corr_ao:+.3f}")
    print(f"  MC-Outcome:  {corr_mo:+.3f}")

    # 均值差异
    print(f"\n均值:")
    print(f"  API:     {all_api.mean():.3f} ± {all_api.std():.3f}")
    print(f"  MC:      {all_mc.mean():.3f} ± {all_mc.std():.3f}")
    print(f"  Outcome: {all_out.mean():.3f} ± {all_out.std():.3f}")

    # 分歧分析：API vs MC 差异最大的步骤
    delta_am = np.abs(all_api - all_mc)
    top_divergent = np.argsort(delta_am)[-10:]

    print(f"\nAPI vs MC 分歧最大的 5 个步骤:")
    flat_steps = []
    for r in results:
        for st_idx in range(r["n_steps"]):
            flat_steps.append({
                "question": r["question"],
                "api": r["api_scores"][st_idx],
                "mc": r["mc_scores"][st_idx],
            })
    for idx in top_divergent[-5:]:
        s = flat_steps[idx]
        print(f"  API={s['api']:.2f} MC={s['mc']:.2f} | Q: {s['question'][:80]}...")

    # 关键发现：按最终答案正确性分层
    correct_mask = np.array([r["answer_correct"] for r in results]) == 1.0
    correct_api_means = [r["api_mean"] for r, c in zip(results, correct_mask) if c]
    wrong_api_means = [r["api_mean"] for r, c in zip(results, correct_mask) if not c]
    correct_mc_means = [r["mc_mean"] for r, c in zip(results, correct_mask) if c]
    wrong_mc_means = [r["mc_mean"] for r, c in zip(results, correct_mask) if not c]

    print(f"\n按答案正确性分层:")
    print(f"  答案正确 ({sum(correct_mask)}条): API={np.mean(correct_api_means):.3f}, MC={np.mean(correct_mc_means):.3f}")
    print(f"  答案错误 ({sum(~correct_mask)}条): API={np.mean(wrong_api_means):.3f}, MC={np.mean(wrong_mc_means):.3f}")

    # 保存
    summary = {
        "n_samples": len(results),
        "n_steps": len(all_api),
        "n_rollouts": N_ROLLOUTS,
        "correlations": {"api_mc": float(corr_am), "api_outcome": float(corr_ao), "mc_outcome": float(corr_mo)},
        "api_mean": float(all_api.mean()), "api_std": float(all_api.std()),
        "mc_mean": float(all_mc.mean()), "mc_std": float(all_mc.std()),
        "outcome_mean": float(all_out.mean()), "outcome_std": float(all_out.std()),
    }
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存至: {OUTPUT}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
