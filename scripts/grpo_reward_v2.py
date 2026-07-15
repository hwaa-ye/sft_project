"""
GRPO Reward V2: 针对截断问题优化

改进:
1. 截断=零分: 缺</answer>直接将reward置零（硬约束）
2. 正确+短=bonus: 正确且简洁的推理获得额外奖励
3. efficiency 分段更激进: 对长推理惩罚加重

与 train_grpo.py 配合: GEN_MAX_NEW=1200 (训练时缩短上下文)
"""
import re


def extract_answer(text: str) -> str | None:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def extract_think(text: str) -> str:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1) if m else ""


def normalize(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\boxed\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\dfrac", r"\\frac", s)
    s = s.replace("×", "错误").replace("✓", "正确").replace("√", "正确")
    s = re.sub(r"\s+", "", s)
    return s.strip('"\'$')


def check_answer(pred: str, gold: str) -> bool:
    p, g = normalize(pred), normalize(gold)
    if p == g:
        return True
    for s, t in [(p, g), (g, p)]:
        m = re.match(r"-?[\d,]+\.?\d*", s)
        if m:
            try:
                pn = float(m.group().replace(",", ""))
                m2 = re.match(r"-?[\d,]+\.?\d*", t)
                if m2:
                    gn = float(m2.group().replace(",", ""))
                    if abs(pn - gn) < 1e-6:
                        return True
            except ValueError:
                pass
    return False


def compute_reward(
    prediction: str,
    gold_answer: str,
    w_accuracy: float = 0.6,
    w_completeness: float = 0.15,
    w_efficiency: float = 0.25,
) -> dict:
    """
    V2 reward 设计:
    - 截断检测: 缺</answer> → total=0（硬约束，无论推理多好）
    - 准确性: 答案对不对
    - 完整性: think+answer 标签完整性（权重降低，因为有截断硬约束了）
    - 效率: 推理长度分段评分（权重提高）+ 正确且短的 bonus
    """

    has_think = "</think>" in prediction
    has_answer = "</answer>" in prediction
    think_text = extract_think(prediction)
    think_len = len(think_text)
    pred_ans = extract_answer(prediction)

    # ─── 硬约束: 截断 = 零分 ───
    if not has_answer:
        return {
            "total": 0.0,
            "accuracy": 0.0,
            "completeness": 0.0 if not has_think else 0.3,
            "efficiency": 0.0,
            "think_len": think_len,
            "is_truncated": True,
        }

    # ─── 1. 准确性 ───
    accuracy = 1.0 if (pred_ans and check_answer(pred_ans, gold_answer)) else 0.0

    # ─── 2. 完整性 ───
    completeness = 1.0 if (has_think and has_answer) else (0.5 if has_think else 0.0)

    # ─── 3. 效率 (激进分段) ───
    if think_len == 0:
        efficiency = 0.0
    elif think_len < 200:
        efficiency = 0.3   # 太短: 可能跳步（比V1更严）
    elif think_len < 800:
        efficiency = 1.0   # 理想长度: 够用但不冗余
    elif think_len < 1500:
        efficiency = 0.65  # 偏长
    elif think_len < 2500:
        efficiency = 0.3   # 过长，惩罚加重（V1=0.7）
    else:
        efficiency = 0.05  # 非常长，几乎零分（V1=0.3）

    # ─── 4. 正确+短 bonus ───
    bonus = 0.0
    if accuracy > 0 and think_len > 0:
        if think_len < 500:
            bonus = 0.15   # 正确且极简
        elif think_len < 1000:
            bonus = 0.08   # 正确且适中
        # >1000 无 bonus

    # ─── 总 reward ───
    total = (w_accuracy * accuracy +
             w_completeness * completeness +
             w_efficiency * efficiency +
             bonus)  # bonus 独立于权重体系

    # clamp 到 [0, 1]
    total = min(1.0, max(0.0, total))

    return {
        "total": total,
        "accuracy": accuracy,
        "completeness": completeness,
        "efficiency": efficiency,
        "bonus": bonus,
        "think_len": think_len,
        "is_truncated": False,
    }
