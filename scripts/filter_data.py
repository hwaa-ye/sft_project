"""
过滤 Firefly 数据：去除非生成任务 + 长度过滤 + 按类采样
"""
import json
import random
from collections import defaultdict

DATA_PATH = "data/raw.jsonl"
OUTPUT_PATH = "data/train.jsonl"
SAMPLE_TOTAL = 130000  # 目标保留 13 万条

random.seed(42)

# ─── 1. 过滤配置 ───

# 非生成任务（去掉）
SKIP_KINDS = {"NER", "SentimentAnalyze", "TextMatching", "NLI"}

# 各类保留量（手动指定，保证多样性）
QUOTAS = {
    "Cot":                 65542,   # 推理链，全部保留
    "Translation":         15000,
    "OpenQA":              12000,
    "Composition":         12000,
    "ClassicalChinese":    10000,
    "Summary":             8000,
    "Program":             869,     # 代码，全部保留
    "AncientPoem":         5000,
    "Couplet":             3000,
    "LyricGeneration":     3000,
    "JinYongGeneration":   2000,
    "ProductDesc":         2000,
    "StoryGeneration":     2000,
    "MRC":                 5000,
    "MusicComment":        2000,
    "TextCorrection":      2000,
    "Dictionary":          3000,
    "KeywordRecognition":  2000,
    "ProseGeneration":     584,
}

# ─── 2. 第一轮：去掉非生成任务 + 长度过滤 ───

print("第一轮：任务类型过滤 + 长度过滤...")
passed = defaultdict(list)  # kind -> [条目]

with open(DATA_PATH, encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue

        kind = d["kind"]

        # 跳过非生成任务
        if kind in SKIP_KINDS:
            continue

        # 跳过不在配额表中的任务（不做多余处理）
        if kind not in QUOTAS:
            continue

        input_len = len(d["input"])
        target_len = len(d["target"])

        # 长度过滤
        if target_len < 20:      # 回答太短，无学习价值
            continue
        if target_len > 2000:    # 太长，截断保留
            d["target"] = d["target"][:2000]
        if input_len > 2000:
            d["input"] = d["input"][:2000]

        passed[kind].append(d)

    total_kept = sum(len(v) for v in passed.values())
    print(f"  第一轮后剩余: {total_kept} 条")

# ─── 3. 第二轮：按配额采样 ───

print("\n第二轮：按任务配额采样...")
final = []
stats = {}

for kind, quota in sorted(QUOTAS.items()):
    pool = passed.get(kind, [])
    actual = min(len(pool), quota)
    sampled = random.sample(pool, actual) if actual < len(pool) else pool
    final.extend(sampled)
    stats[kind] = (len(pool), actual, quota)

random.shuffle(final)  # 打乱，让训练时不会按任务类型排列

# ─── 4. 写入输出文件 ───

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for d in final:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

print(f"\n输出: {OUTPUT_PATH}")
print(f"总条数: {len(final)}")

# ─── 5. 打印统计 ───
print("\n各类保留量统计 (可用条数 / 实际保留 / 配额):")
print(f"{'Kind':25s} {'可用':>8s} {'保留':>8s} {'配额':>8s}")
print("-" * 55)
for kind, (avail, kept, quota) in sorted(stats.items()):
    flag = " ✓" if kept == min(avail, quota) else " ⚠"
    print(f"{kind:25s} {avail:>8d} {kept:>8d} {quota:>8d}{flag}")
print(f"\n{'总计':25s} {sum(v[0] for v in stats.values()):>8d} {len(final):>8d} -")
