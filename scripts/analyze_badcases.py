"""Badcase 分析脚本：分类截断/计算错误/语义误解，输出典型样本"""
import json, re, sys

def norm(s):
    s = str(s).strip().lower()
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\boxed\{(.*?)\}", r"\1", s)
    s = s.replace("×","错误").replace("✓","正确").replace("√","正确")
    s = re.sub(r"\s+", "", s)
    s = s.strip('"\'$')
    return s

def get_num(s):
    s = re.sub(r"\s+", "", str(s))
    m = re.match(r"-?[\d,]+\.?\d*", s)
    if m:
        try: return float(m.group().replace(",",""))
        except: pass
    return None

def check(pred, gold):
    p, g = norm(pred), norm(gold)
    if p == g: return True
    pn, gn = get_num(p), get_num(g)
    if pn is not None and gn is not None and abs(pn-gn) < 1e-6:
        return True
    return False

pred_file = sys.argv[1] if len(sys.argv) > 1 else "data/test_predictions_v2.jsonl"

truncated = []
calc_errors = []
sem_errors = []
correct = 0

with open(pred_file) as f:
    for i, line in enumerate(f):
        item = json.loads(line)
        pred = item.get("prediction", "")
        gold = str(item.get("answer", "")).strip()

        has_close = "</think>" in pred
        has_ans = "<answer>" in pred

        if not has_close or not has_ans:
            truncated.append((i, item))
            continue

        ans_m = re.search(r"<answer>(.*?)</answer>", pred, re.DOTALL)
        if not ans_m:
            truncated.append((i, item))
            continue

        pred_ans = ans_m.group(1).strip()
        if check(pred_ans, gold):
            correct += 1
        else:
            has_num = bool(re.search(r"\d", norm(pred_ans)))
            if has_num:
                calc_errors.append((i, item, pred_ans))
            else:
                sem_errors.append((i, item, pred_ans))

print(f"正确: {correct}")
print(f"截断: {len(truncated)}")
print(f"计算错误: {len(calc_errors)}")
print(f"语义误解: {len(sem_errors)}")
print(f"总计: {correct + len(truncated) + len(calc_errors) + len(sem_errors)}")
print()

print("=" * 60)
print("一、截断样本（前5条）")
print("=" * 60)
for idx, (i, item) in enumerate(truncated[:5]):
    pred = item["prediction"]
    print(f"\n#{idx+1} (line {i}) | Gold: {item['answer']}")
    print(f"Q: {item['instruction'][:100]}")
    print(f"Pred start: {pred[:150]}")
    print(f"Pred end: ...{pred[-150:]}")

print()
print("=" * 60)
print("二、计算错误样本")
print("=" * 60)
for idx, (i, item, pred_ans) in enumerate(calc_errors):
    pred = item["prediction"]
    print(f"\n#{idx+1} (line {i}) | Gold: {item['answer']} | Pred: {pred_ans}")
    print(f"Q: {item['instruction'][:120]}")
    think_m = re.search(r"<think>(.*?)</think>", pred, re.DOTALL)
    if think_m:
        print(f"Think tail: ...{think_m.group(1)[-250:]}")

print()
print("=" * 60)
print("三、语义误解样本")
print("=" * 60)
for idx, (i, item, pred_ans) in enumerate(sem_errors):
    pred = item["prediction"]
    print(f"\n#{idx+1} (line {i}) | Gold: {item['answer']} | Pred: {pred_ans}")
    print(f"Q: {item['instruction'][:120]}")
    think_m = re.search(r"<think>(.*?)</think>", pred, re.DOTALL)
    if think_m:
        print(f"Think tail: ...{think_m.group(1)[-250:]}")
