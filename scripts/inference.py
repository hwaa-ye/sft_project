"""
推理脚本：加载 SFT LoRA 模型，对测试集批量生成推理链
输入: data/test_math.jsonl
输出: data/test_predictions.jsonl
"""
import json, os, random, torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

BASE_MODEL = os.environ.get(
    "SFT_BASE_MODEL",
    os.path.join(os.path.expanduser("~"), "autodl-fs", "model_cache", "Qwen", "Qwen3-8B"),
)
LORA_PATH = os.environ.get("SFT_LORA_PATH", "output/sft_qwen3/final")
INIT_LORA = os.environ.get("SFT_INIT_LORA", "").strip()
TEST_PATH = os.environ.get("SFT_TEST_PATH", "data/test_math.jsonl")
OUTPUT_PATH = os.environ.get("SFT_PRED_PATH", "data/test_predictions.jsonl")

MAX_NEW_TOKENS = int(os.environ.get("SFT_MAX_NEW_TOKENS", "2048"))
TEMPERATURE = float(os.environ.get("SFT_TEMPERATURE", "0.7"))
BATCH_SIZE = int(os.environ.get("SFT_BATCH_SIZE", "4"))
SEED = int(os.environ.get("SFT_SEED", "42"))
MAX_SAMPLES = int(os.environ.get("SFT_MAX_SAMPLES", "0"))
RESUME = os.environ.get("SFT_RESUME", "0").strip().lower() in {"1", "true", "yes", "y"}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = "cuda:0"
torch.cuda.empty_cache()

print(f"设备: {device}")
print(f"模型: {BASE_MODEL}")
print(f"LoRA: {LORA_PATH}")
print(f"初始 LoRA: {INIT_LORA or '无'}")
print(f"batch_size: {BATCH_SIZE}, max_new_tokens: {MAX_NEW_TOKENS}")
print(f"temperature: {TEMPERATURE}, seed: {SEED}")

# 加载 tokenizer & 模型
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # 左填充，确保生成从右侧开始

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map=None,
    trust_remote_code=True, local_files_only=True,
)
if INIT_LORA:
    print(f"加载并合并初始 LoRA: {INIT_LORA}")
    model = PeftModel.from_pretrained(model, INIT_LORA, is_trainable=False)
    model = model.merge_and_unload()
model = PeftModel.from_pretrained(model, LORA_PATH)
model = model.to(device)
model.eval()
print(f"  显存: {torch.cuda.memory_allocated()/1024**3:.1f}GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

# 加载测试数据
test_data = []
with open(TEST_PATH, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            test_data.append(json.loads(line))
if MAX_SAMPLES > 0:
    test_data = test_data[:MAX_SAMPLES]
total_test_samples = len(test_data)
completed_samples = 0
if RESUME and os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, encoding="utf-8") as existing:
        completed_samples = sum(1 for line in existing if line.strip())
    if completed_samples > total_test_samples:
        raise ValueError(
            f"已有预测 {completed_samples} 条，超过测试集 {total_test_samples} 条"
        )
    test_data = test_data[completed_samples:]
print(f"测试数据: {total_test_samples} 条, 已完成: {completed_samples}, 本次: {len(test_data)} 条")

# 构建所有 prompt
prompts = [
    f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n"
    for item in test_data
]

# 批量推理（增量写入，防丢结果）
os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
out_f = open(OUTPUT_PATH, "a" if RESUME else "w", encoding="utf-8")

for i in range(0, len(test_data), BATCH_SIZE):
    batch_prompts = prompts[i:i + BATCH_SIZE]
    batch_items = test_data[i:i + BATCH_SIZE]

    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    for j, (item, out_ids) in enumerate(zip(batch_items, outputs)):
        # generate 返回的是统一 padded input 宽度 + 新生成 token。左 padding 时不能
        # 使用 real_prompt_len 切片，否则会把 prompt 尾部混入 prediction。
        input_width = inputs.input_ids.size(1)
        generated = tokenizer.decode(out_ids[input_width:], skip_special_tokens=True)
        item["prediction"] = generated
        out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
        out_f.flush()

    done = completed_samples + min(i + BATCH_SIZE, len(test_data))
    print(f"  {done}/{total_test_samples} 完成", flush=True)

out_f.close()
print(f"推理完成: {total_test_samples} 条 -> {OUTPUT_PATH}")
