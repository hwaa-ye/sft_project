"""
用 DeepSeek API 对推理链进行步骤级评分（PRM 训练数据标注）
输入：GRPO 模型生成的预测文件（含 <think> 标签分步）
输出：标注好的 (question, step_text, score, reason) 数据
"""
import json, os, re, sys, time, argparse
from openai import OpenAI


# ─── API 配置 ───
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"  # V4 Flash: 便宜够用

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# ─── 评分 Prompt ───
SCORING_SYSTEM = """你是一个数学推理步骤质量评估专家。你的任务是评估推理链中的每一步是否: (1)数学正确 (2)逻辑连贯 (3)不跳步。

评分标准（0~1连续值）:
- 0.9~1.0: 完全正确, 逻辑清晰, 步长合理
- 0.7~0.9: 大体正确, 有小瑕疵或表述略简略
- 0.5~0.7: 部分正确, 有逻辑跳跃或次要错误
- 0.3~0.5: 方向正确但推理有明显问题
- 0.0~0.3: 完全错误或无意义

仅返回JSON: {"score": <float>, "reason": "<一句话理由>"}"""


def split_steps(think_content):
    """将 <think> 内容按自然段拆分为推理步骤"""
    # 先按双换行分
    raw = think_content.strip()
    if not raw:
        return []
    steps = [s.strip() for s in re.split(r'\n\s*\n', raw) if s.strip()]
    # 如果只分出一个大段，尝试按单换行分
    if len(steps) <= 1:
        steps = [s.strip() for s in raw.split('\n') if s.strip()]
    # 如果还是太长，按句号分
    if len(steps) <= 1 and len(raw) > 200:
        steps = [s.strip() + '.' for s in raw.split('。') if s.strip()]
    return steps


def extract_think(response_text):
    """从 GRPO 输出中提取 <think> 和 <answer> 内容"""
    think_m = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
    answer_m = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
    return (think_m.group(1).strip() if think_m else None,
            answer_m.group(1).strip() if answer_m else None)


def score_step(question, all_steps, step_idx, step_text, max_retries=3):
    """调用 DeepSeek API 对单步打分"""
    # 构建上下文：前面的步骤 + 当前步骤
    context = ""
    if step_idx > 0:
        prev_text = "\n".join(f"步骤{i+1}: {s}" for i, s in enumerate(all_steps[:step_idx]))
        context = f"前面的推理步骤:\n{prev_text}\n\n"

    user_prompt = f"""问题: {question}

{context}当前要评估的步骤:
步骤{step_idx+1}: {step_text}

请对步骤{step_idx+1}的推理质量进行评分。"""

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SCORING_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content.strip()
            # 尝试解析 JSON
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            result = json.loads(raw)
            return float(result["score"]), str(result.get("reason", ""))
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None, f"API error: {e}"


def annotate_sample(item, idx, total):
    """标注单条推理链的所有步骤"""
    prediction = item.get("prediction", "")
    question = item.get("instruction", "")

    think, answer = extract_think(prediction)
    if not think:
        return None  # 没有 think 内容，跳过

    steps = split_steps(think)
    if len(steps) < 2:
        return None  # 步骤太少

    step_scores = []
    for s_idx, step_text in enumerate(steps):
        score, reason = score_step(question, steps, s_idx, step_text)
        if score is not None:
            step_scores.append({
                "step_idx": s_idx,
                "step_text": step_text,
                "score": score,
                "reason": reason,
            })
        else:
            print(f"  [{idx+1}/{total}] 步骤{s_idx+1} 打分失败: {reason}", flush=True)
        time.sleep(0.3)  # 限速

    if not step_scores:
        return None

    avg_score = sum(s["score"] for s in step_scores) / len(step_scores)
    print(f"  [{idx+1}/{total}] {len(step_scores)}步, avg={avg_score:.2f}", flush=True)

    return {
        "question": question,
        "answer": item.get("answer", ""),
        "full_think": think,
        "full_answer": answer,
        "steps": step_scores,
        "avg_score": avg_score,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="GRPO 预测 JSONL 文件")
    parser.add_argument("--output", default="prm/prm_train_data.jsonl")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()

    if not DEEPSEEK_API_KEY:
        print("错误: 请设置 DEEPSEEK_API_KEY 环境变量")
        sys.exit(1)

    samples = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"加载 {len(samples)} 条预测数据")
    samples = samples[args.skip:args.skip + args.max_samples]
    print(f"本次标注 {len(samples)} 条 (skip={args.skip}, max={args.max_samples})")

    results = []
    for i, item in enumerate(samples):
        result = annotate_sample(item, i, len(samples))
        if result:
            results.append(result)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n标注完成: {len(results)}/{len(samples)} 条有效, 保存至 {args.output}")

    if results:
        all_scores = [s["score"] for r in results for s in r["steps"]]
        all_avgs = [r["avg_score"] for r in results]
        print(f"步骤总分范围: {min(all_scores):.2f}~{max(all_scores):.2f}, "
              f"全局均值: {sum(all_scores)/len(all_scores):.3f}")
        print(f"样本均分范围: {min(all_avgs):.2f}~{max(all_avgs):.2f}")


if __name__ == "__main__":
    main()
