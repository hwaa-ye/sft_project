"""
将过滤后的 JSONL 数据 tokenize 为模型可用的格式
输入: data/train.jsonl
输出: data/tokenized/ (arrow 格式，可用 datasets 加载)
"""
import json
import os
import pickle
import tempfile

# ModelScope 缓存：优先用 autodl-fs 持久化目录，避免每次新容器都重下模型
if "MODELSCOPE_CACHE" not in os.environ:
    _persist = os.path.join(os.path.expanduser("~"), "autodl-fs", "model_cache")
    _cache_dir = _persist if os.path.isdir(os.path.join(os.path.expanduser("~"), "autodl-fs")) \
                 else os.path.join(tempfile.gettempdir(), "modelscope")
    os.environ.setdefault("MODELSCOPE_CACHE", _cache_dir)
    os.environ.setdefault("MODELSCOPE_CREDENTIALS_PATH", os.path.join(_cache_dir, "credentials"))
    os.environ.setdefault("MS_CACHE_HOME", _cache_dir)

from modelscope import AutoTokenizer

# ─── 1. 加载 tokenizer ───
print("加载 Qwen2.5 tokenizer（从 ModelScope）...")
model_name = os.environ.get("SFT_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
)
# Qwen2.5 需要 pad token，默认没有
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

MAX_LENGTH = int(os.environ.get("SFT_MAX_LENGTH", "1024"))
DATA_PATH = os.environ.get("SFT_TRAIN_JSONL", "data/train.jsonl")
tokenized_dir = os.environ.get("SFT_TOKENIZED_DIR", "data/tokenized")

# ─── 2. 定义格式化函数 ───
def format_sft(example):
    """将 Firefly 的 kind/input/target 格式化为 Qwen 对话格式"""
    instruction = example["input"]
    response = example["target"]

    # Qwen ChatML 格式
    # <|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>
    text = (
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )
    return text

# ─── 3. 逐条 tokenize ───
print("读取数据并 tokenize...")
os.makedirs(tokenized_dir, exist_ok=True)

input_ids_list = []
labels_list = []
stats = {"total": 0, "skipped_too_long": 0}

with open(DATA_PATH, encoding="utf-8") as f:
    for i, line in enumerate(f):
        example = json.loads(line)
        text = format_sft(example)

        # tokenize 整段对话
        tokens = tokenizer(
            text,
            add_special_tokens=False,
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors=None,  # 返回 list
        )

        input_ids = tokens["input_ids"]

        # ─── 构造 labels：把 instruction 部分的 loss mask 掉 ───
        # 方法：重新 tokenize 只取 instruction 部分，找到它的长度
        instruction_text = f"<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n"
        instruction_tokens = tokenizer(
            instruction_text,
            add_special_tokens=False,
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors=None,
        )
        instruction_len = len(instruction_tokens["input_ids"])

        labels = input_ids.copy()
        if instruction_len >= len(labels):
            stats["skipped_too_long"] += 1
            continue
        labels[:instruction_len] = [-100] * instruction_len

        # 跳过过短的样本（整个都被截断了）
        if len(input_ids) < 10:
            stats["skipped_too_long"] += 1  # 实际是被截得太短
            continue

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        stats["total"] += 1

        if (i + 1) % 10000 == 0:
            print(f"  已处理 {i+1} 条...")

print(f"\n处理完成: 共 {stats['total']} 条, 跳过 {stats['skipped_too_long']} 条")

# ─── 4. 保存为 numpy 数组（比 pickle list-of-ints 内存效率高很多）───
print(f"\n保存到 {tokenized_dir}/ ...")
import numpy as np
input_ids_arr = np.array([np.array(ids, dtype=np.int32) for ids in input_ids_list], dtype=object)
labels_arr = np.array([np.array(lbl, dtype=np.int32) for lbl in labels_list], dtype=object)
np.save(f"{tokenized_dir}/input_ids.npy", input_ids_arr, allow_pickle=True)
np.save(f"{tokenized_dir}/labels.npy", labels_arr, allow_pickle=True)
# 同时保留 pkl 做兼容
with open(f"{tokenized_dir}/input_ids.pkl", "wb") as f:
    pickle.dump(input_ids_list, f)
with open(f"{tokenized_dir}/labels.pkl", "wb") as f:
    pickle.dump(labels_list, f)

# ─── 5. 打印一条验证 ───
print("\n--- 验证样例 ---")
print(f"input_ids 长度: {len(input_ids_list[0])}")
print(f"labels 长度:   {len(labels_list[0])}")
print(f"\ninput_ids 前 20 个: {input_ids_list[0][:20]}")
print(f"labels 前 20 个:   {labels_list[0][:20]}")
print(f"(前面应该都是 -100，说明 instruction 被 mask 了)")

# decode 验证
print("\ndecode 验证（只解码非 -100 部分）:")
response_ids = [id_ for id_, label in zip(input_ids_list[0], labels_list[0]) if label != -100]
print(tokenizer.decode(response_ids)[:200])
