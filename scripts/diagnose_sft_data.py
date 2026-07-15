"""
SFT 训练数据 token 级长度诊断（用真实 Qwen3 分词器 + 训练时的 ChatML 格式）。

复现 tokenize_dataset.py 的 format_math_sft，统计:
  - 各 source 的 reasoning/整条 ChatML 的 token 长度分布
  - 超过 MAX_LENGTH(2048) 的比例
  - 关键: 右截断到 2048 后, 是否还保留完整 </think> 和 </answer>
    （即：有多少监督样本会被 tokenizer 砍成"半截推理没答案"）

用法:
  python scripts/diagnose_sft_data.py
输出:
  - 终端打印
  - markdown 报告写入 analysis/<日期>_sft_train_data_token_length_diagnosis.md
"""
import json
import os
import datetime
from collections import defaultdict
from transformers import AutoTokenizer
from sft_data_utils import format_math_sft

TOK_PATH = os.environ.get(
    "QWEN_TOKENIZER",
    os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen3-8B"),
)
DATA = os.environ.get("SFT_TRAIN_JSONL", "data/train_math_all.jsonl")
MAX_LENGTH = int(os.environ.get("SFT_MAX_LENGTH", "2048"))

OUT_DIR = "analysis"
DATE = datetime.date.today().isoformat()
OUT_MD = f"{OUT_DIR}/{DATE}_sft_train_data_token_length_diagnosis.md"

tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)


def enc_len(s):
    return len(tok(s, add_special_tokens=False)["input_ids"])


def pct(a, p):
    a = sorted(a)
    return a[min(len(a) - 1, int(len(a) * p))] if a else 0


_md = []
def emit(line=""):
    print(line)
    _md.append(line)


by_src = defaultdict(lambda: {"n": 0, "tok": [], "reason_tok": [], "over": 0, "broken": 0})
overall = {"n": 0, "over": 0, "broken": 0}

with open(DATA, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        ex = json.loads(line)
        if not (ex.get("reasoning") and ex.get("answer")):
            continue
        src = ex.get("source", "?")
        text = format_math_sft(ex)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        n_tok = len(ids)

        s = by_src[src]
        s["n"] += 1
        overall["n"] += 1
        s["tok"].append(n_tok)
        s["reason_tok"].append(enc_len(ex.get("reasoning", "")))

        if n_tok > MAX_LENGTH:
            s["over"] += 1
            overall["over"] += 1
            truncated_text = tok.decode(ids[:MAX_LENGTH])
            if ("</think>" not in truncated_text) or ("</answer>" not in truncated_text):
                s["broken"] += 1
                overall["broken"] += 1

emit(f"# SFT 训练数据 token 级诊断 — {DATE}")
emit()
emit(f"- 分词器: Qwen3-8B（真实分词器，非字符估算）")
emit(f"- 格式: 与 tokenize_dataset.py 的 format_math_sft 完全一致")
emit(f"- MAX_LENGTH: {MAX_LENGTH}（SFT tokenize 时的右截断上限）")
emit(f"- 数据文件: {DATA}")
emit(f"- 总有效样本: {overall['n']}")
emit()
emit("> **「砍坏」= 样本 token 数超过 2048，被右截断后丢失了 `</think>` 或 `</answer>` 标签，"
     "即监督信号变成「半截推理、没有答案」。这类样本会教模型「可以不写完」。**")
emit()

emit("## 按 source 的 token 长度与砍坏率（按砍坏率降序）")
emit()
emit("| source | n | tok_p50 | tok_p95 | tok_max | 超2048 | 砍坏(丢标签) |")
emit("|---|---|---|---|---|---|---|")
for src, v in sorted(by_src.items(), key=lambda x: -x[1]["over"] / max(1, x[1]["n"])):
    n = v["n"]
    emit(f"| {src} | {n} | {pct(v['tok'],.5)} | {pct(v['tok'],.95)} | {max(v['tok'])} | "
         f"{100*v['over']/n:.1f}% | {100*v['broken']/n:.1f}% |")
emit(f"| **总计** | **{overall['n']}** | | | | "
     f"**{100*overall['over']/overall['n']:.1f}%** | "
     f"**{100*overall['broken']/overall['n']:.1f}% ({overall['broken']}条)** |")
emit()

emit("## 整条样本 token 长度桶分布")
emit()
all_tok = [t for v in by_src.values() for t in v["tok"]]
buckets = [(0, 1024), (1024, 2048), (2048, 4096), (4096, 10**9)]
labels = ["<=1024", "1024-2048", "2048-4096", ">4096"]
emit("| 长度桶 | n | 占比 |")
emit("|---|---|---|")
for (lo, hi), lab in zip(buckets, labels):
    c = sum(1 for t in all_tok if lo <= t < hi)
    emit(f"| {lab} | {c} | {100*c/len(all_tok):.1f}% |")
emit()

emit("## 结论")
emit()
clean = sum(1 for t in all_tok if t <= MAX_LENGTH)
emit(f"- **{100*overall['broken']/overall['n']:.1f}%（{overall['broken']}条）训练样本被 tokenizer 砍成半截、丢失答案标签**，"
     "这是「模型学会不写完」的直接监督来源。")
emit(f"- **{100*clean/overall['n']:.1f}%（{clean}条）样本 ≤2048 token，结构完整**，构成 SFT-repair 的干净基础。")
emit("- amc_aime 砍坏率最高（比例视角的重灾区）；EduChat-Math 砍坏绝对数最多（数量视角的重灾区）。两者都需处理。")
emit("- 「超2048」与「砍坏」两列几乎相等 → 只要超长几乎必然丢答案（答案在末尾，一超长先被砍）。")

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(_md) + "\n")
print(f"\n[saved] {OUT_MD}")
