"""
Step 1: 探索 Firefly 数据结构
- 看数据长什么样
- 统计任务类型分布
- 统计 input/target 长度分布
- 采样肉眼评估质量
"""
import json
from collections import Counter, defaultdict

DATA_PATH = "data/raw.jsonl"
ENCODING = "utf-8"  # 遇到编码错误时自动跳过坏字节

# ─── 1. 快速采样看格式 ───
print("=" * 60)
print("1. 采样 3 条看结构")
print("=" * 60)
with open(DATA_PATH, encoding=ENCODING, errors='ignore') as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        d = json.loads(line)
        print(f"\n--- 第 {i+1} 条 ---")
        print(f"kind:   {d['kind']}")
        print(f"input:  {d['input'][:150]}...")
        print(f"target: {d['target'][:150]}...")

# ─── 2. 全量扫描：统计分布 ───
print("\n" + "=" * 60)
print("2. 全量统计（逐行读取，不占内存）")
print("=" * 60)

kind_counter = Counter()
input_lens = []    # 存所有 input 长度
target_lens = []   # 存所有 target 长度
empty_input = 0
empty_target = 0
kind_samples = defaultdict(list)  # 每种 kind 存 3 条

with open(DATA_PATH, encoding=ENCODING, errors='ignore') as f:
    for i, line in enumerate(f):
        # 跳过空行和编码损坏的行
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue

        kind_counter[d["kind"]] += 1

        input_lens.append(len(d["input"]))
        target_lens.append(len(d["target"]))

        if not d["input"].strip():
            empty_input += 1
        if not d["target"].strip():
            empty_target += 1

        # 每类留 3 条样例
        if len(kind_samples[d["kind"]]) < 3:
            kind_samples[d["kind"]].append(d)

        # 进度提示
        if (i + 1) % 200000 == 0:
            print(f"  已扫描 {i+1} 条...")

total = i + 1
print(f"\n总条数: {total}")
print(f"空 input: {empty_input} ({100*empty_input/total:.2f}%)")
print(f"空 target: {empty_target} ({100*empty_target/total:.2f}%)")

# ─── 3. 任务类型分布 ───
print("\n" + "=" * 60)
print("3. 任务类型分布（Top 25）")
print("=" * 60)
for kind, count in kind_counter.most_common(25):
    pct = 100 * count / total
    bar = "█" * int(pct / 2)  # 每 2% 一个方块
    print(f"{kind:20s} {count:>8d} ({pct:5.2f}%) {bar}")

# ─── 4. 长度分布分析 ───
print("\n" + "=" * 60)
print("4. 长度分布")
print("=" * 60)

def analyze_lengths(lens, name):
    lens.sort()
    n = len(lens)
    print(f"\n{name}:")
    print(f"  min:     {lens[0]}")
    print(f"  max:     {lens[-1]}")
    print(f"  mean:    {sum(lens)/n:.0f}")
    print(f"  median:  {lens[n//2]}")
    print(f"  p10:     {lens[n//10]}   (10% 的数据短于这个值)")
    print(f"  p25:     {lens[n//4]}  ")
    print(f"  p75:     {lens[3*n//4]}  ")
    print(f"  p90:     {lens[9*n//10]}   (10% 的数据长于这个值)")
    # 看极端值
    short_count = sum(1 for x in lens if x < 10)
    long_count = sum(1 for x in lens if x > 2000)
    print(f"  input<10字:   {short_count} ({100*short_count/n:.1f}%)")
    print(f"  input>2000字: {long_count} ({100*long_count/n:.1f}%)")

analyze_lengths(input_lens, "INPUT 长度（字符数）")
analyze_lengths(target_lens, "TARGET 长度（字符数）")

# ─── 5. 采样看每种 kind 的数据质量 ───
print("\n" + "=" * 60)
print("5. 各类任务采样（每种 kind 看 1-3 条）")
print("=" * 60)
for kind in sorted(kind_samples.keys()):
    print(f"\n--- {kind} ({kind_counter[kind]} 条) ---")
    for d in kind_samples[kind][:2]:  # 每种看 2 条
        print(f"  input:  {d['input'][:200]}")
        print(f"  target: {d['target'][:200]}")
        print()

# ─── 6. 汇总输出 ───
print("\n" + "=" * 60)
print("探索完成，接下来可以基于以上分析定过滤策略")
print("=" * 60)
