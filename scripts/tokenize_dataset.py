"""
数学推理数据 tokenize：Qwen3 格式（含 think/answer 标签）
输入: data/train_math.jsonl（每条含 instruction, reasoning, answer）
输出: data/tokenized/
"""
import json
import os
import pickle
import tempfile

# ModelScope 缓存：优先用 autodl-fs 持久化目录
if "MODELSCOPE_CACHE" not in os.environ:
    _persist = os.path.join(os.path.expanduser("~"), "autodl-fs", "model_cache")
    _cache_dir = _persist if os.path.isdir(os.path.join(os.path.expanduser("~"), "autodl-fs")) \
                 else os.path.join(tempfile.gettempdir(), "modelscope")
    os.environ.setdefault("MODELSCOPE_CACHE", _cache_dir)
    os.environ.setdefault("MODELSCOPE_CREDENTIALS_PATH", os.path.join(_cache_dir, "credentials"))
    os.environ.setdefault("MS_CACHE_HOME", _cache_dir)

from modelscope import AutoTokenizer

# ─── 1. 加载 tokenizer ───
print("加载 Qwen3 tokenizer...")
model_name = os.environ.get("SFT_MODEL_NAME", "Qwen/Qwen3-8B")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

MAX_LENGTH = int(os.environ.get("SFT_MAX_LENGTH", "2048"))  # 推理链需要更长
DATA_PATH = os.environ.get("SFT_TRAIN_JSONL", "data/train_math.jsonl")
tokenized_dir = os.environ.get("SFT_TOKENIZED_DIR", "data/tokenized")

# ─── 2. 格式化函数 ───
def format_math_sft(example):
    """
    数学推理 SFT 格式（Qwen3 ChatML + think/answer 标签）
    example 包含: instruction（题目）, reasoning（推理过程）, answer（最终答案）
    """
    instruction = example["instruction"]
    reasoning = example.get("reasoning", "")
    answer = example.get("answer", "")

    response = f"<think>\n{reasoning}\n</think>\n\n<answer>\n{answer}\n</answer>"

    text = (
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )
    return text

def format_generic_sft(example):
    """通用 SFT 格式（兼容旧数据：input/target 字段）"""
    instruction = example.get("instruction") or example.get("input", "")
    target = example.get("target", "")

    # 如果 target 已经包含 think/answer 标签，直接用
    if "<think>" in target:
        response = target
    else:
        response = f"<think>\n{target}\n</think>"

    text = (
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )
    return text

# ─── 3. 逐条 tokenize ───
print(f"读取数据并 tokenize（max_length={MAX_LENGTH}）...")
os.makedirs(tokenized_dir, exist_ok=True)

input_ids_list = []
labels_list = []
stats = {"total": 0, "skipped_too_long": 0}

with open(DATA_PATH, encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        example = json.loads(line)

        # 自动选择格式化函数
        if "reasoning" in example and "answer" in example:
            text = format_math_sft(example)
        else:
            text = format_generic_sft(example)

        tokens = tokenizer(
            text,
            add_special_tokens=False,
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors=None,
        )
        input_ids = tokens["input_ids"]

        # 构造 labels：mask 掉 instruction 部分
        instruction_text = f"<|im_start|>user\n{example.get('instruction', example.get('input', ''))}<|im_end|>\n<|im_start|>assistant\n"
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

        if len(input_ids) < 10:
            stats["skipped_too_long"] += 1
            continue

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        stats["total"] += 1

        if (i + 1) % 10000 == 0:
            print(f"  已处理 {i+1} 条...")

print(f"\n处理完成: 共 {stats['total']} 条, 跳过 {stats['skipped_too_long']} 条")

# ─── 4. 保存 ───
print(f"\n保存到 {tokenized_dir}/ ...")
import numpy as np
input_ids_arr = np.array([np.array(ids, dtype=np.int32) for ids in input_ids_list], dtype=object)
labels_arr = np.array([np.array(lbl, dtype=np.int32) for lbl in labels_list], dtype=object)
np.save(f"{tokenized_dir}/input_ids.npy", input_ids_arr, allow_pickle=True)
np.save(f"{tokenized_dir}/labels.npy", labels_arr, allow_pickle=True)
with open(f"{tokenized_dir}/input_ids.pkl", "wb") as f:
    pickle.dump(input_ids_list, f)
with open(f"{tokenized_dir}/labels.pkl", "wb") as f:
    pickle.dump(labels_list, f)

# ─── 5. 验证 ───
print("\n--- 验证样例 ---")
print(f"input_ids 长度: {len(input_ids_list[0])}")
print(f"labels 长度:   {len(labels_list[0])}")
response_ids = [id_ for id_, label in zip(input_ids_list[0], labels_list[0]) if label != -100]
print("response 部分 decode:")
print(tokenizer.decode(response_ids)[:300])
