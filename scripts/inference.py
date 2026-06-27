"""
推理脚本：加载 SFT LoRA 模型，对测试集批量生成推理链
输入: data/test_math.jsonl
输出: data/test_predictions.jsonl
"""
import json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

BASE_MODEL = os.path.join(os.path.expanduser("~"), "autodl-fs", "model_cache", "Qwen", "Qwen3-8B")
LORA_PATH = os.environ.get("SFT_LORA_PATH", "output/sft_qwen3/final")
TEST_PATH = os.environ.get("SFT_TEST_PATH", "data/test_math.jsonl")
OUTPUT_PATH = os.environ.get("SFT_PRED_PATH", "data/test_predictions.jsonl")

MAX_NEW_TOKENS = int(os.environ.get("SFT_MAX_NEW_TOKENS", "2048"))
TEMPERATURE = float(os.environ.get("SFT_TEMPERATURE", "0.7"))
BATCH_SIZE = int(os.environ.get("SFT_BATCH_SIZE", "4"))

device = "cuda:0"
torch.cuda.empty_cache()

print(f"设备: {device}")
print(f"模型: {BASE_MODEL}")
print(f"LoRA: {LORA_PATH}")
print(f"batch_size: {BATCH_SIZE}, max_new_tokens: {MAX_NEW_TOKENS}")

# 加载 tokenizer & 模型
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # 左填充，确保生成从右侧开始

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map=None,
    trust_remote_code=True, local_files_only=True,
)
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
print(f"测试数据: {len(test_data)} 条")

# 构建所有 prompt
prompts = [
    f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n"
    for item in test_data
]

# 批量推理（增量写入，防丢结果）
os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
out_f = open(OUTPUT_PATH, "w", encoding="utf-8")

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
        prompt_len = inputs.input_ids[j].size(0)  # 含 padding
        # 去掉左侧 padding 和 prompt 部分
        real_prompt_len = (inputs.attention_mask[j] == 1).sum().item()
        generated = tokenizer.decode(out_ids[real_prompt_len:], skip_special_tokens=True)
        item["prediction"] = generated
        out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
        out_f.flush()

    done = min(i + BATCH_SIZE, len(test_data))
    print(f"  {done}/{len(test_data)} 完成", flush=True)

out_f.close()
print(f"推理完成: {len(test_data)} 条 -> {OUTPUT_PATH}")
