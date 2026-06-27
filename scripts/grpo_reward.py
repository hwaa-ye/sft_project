"""
GRPO Reward 函数：准确性 + 完整性 + 效率
基于 SFT badcase 分析设计：截断(20.7%)、计算错误(3.1%)、语义误解(1.9%)
"""
import re


def extract_answer(text: str) -> str | None:
    """从模型输出提取 <answer> 标签内的答案"""
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def extract_think(text: str) -> str:
    """提取 <think> 段落"""
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1) if m else ""


def normalize(s: str) -> str:
    """答案归一化"""
    s = str(s).strip().lower()
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\boxed\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\dfrac", r"\\frac", s)
    s = s.replace("×", "错误").replace("✓", "正确").replace("√", "正确")
    s = re.sub(r"\s+", "", s)
    return s.strip('"\'$')


def check_answer(pred: str, gold: str) -> bool:
    """答案比对（含数值容差）"""
    p, g = normalize(pred), normalize(gold)
    if p == g:
        return True
    # 数值比较
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
    w_accuracy: float = 0.7,
    w_completeness: float = 0.2,
    w_efficiency: float = 0.1,
) -> dict:
    """计算单条回答的 reward，返回总分和各维度分"""

    # 1. 准确性 (0.7): 答案对不对
    pred_ans = extract_answer(prediction)
    accuracy = 1.0 if (pred_ans and check_answer(pred_ans, gold_answer)) else 0.0

    # 2. 完整性 (0.2): 是否有完整的 think + answer 标签
    has_think = "</think>" in prediction
    has_answer = "</answer>" in prediction
    completeness = 1.0 if (has_think and has_answer) else (0.5 if has_think else 0.0)

    # 3. 效率 (0.1): 推理链不要太长（针对截断问题）
    think_text = extract_think(prediction)
    think_len = len(think_text)
    if think_len == 0:
        efficiency = 0.0
    elif accuracy > 0:
        # 正确的回答：越短越好（但不要太极端）
        if think_len < 200:
            efficiency = 0.5  # 太短可能跳步
        elif think_len < 1500:
            efficiency = 1.0  # 合理长度
        elif think_len < 2500:
            efficiency = 0.7  # 偏长
        else:
            efficiency = 0.3  # 过长
    else:
        # 错误回答：推理长度不是关键
        efficiency = 0.5

    total = w_accuracy * accuracy + w_completeness * completeness + w_efficiency * efficiency

    return {
        "total": total,
        "accuracy": accuracy,
        "completeness": completeness,
        "efficiency": efficiency,
        "think_len": think_len,
    }
