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

MAX_NEW_TOKENS = int(os.environ.get("SFT_MAX_NEW_TOKENS", "1024"))
TEMPERATURE = float(os.environ.get("SFT_TEMPERATURE", "0.7"))

device = "cuda:0"
torch.cuda.empty_cache()

print(f"设备: {device}")
print(f"模型: {BASE_MODEL}")
print(f"LoRA: {LORA_PATH}")

# 加载 tokenizer & 模型
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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

# 批量推理
results = []
for i, item in enumerate(test_data):
    prompt = f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    item["prediction"] = generated
    results.append(item)

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(test_data)} 完成")

os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"推理完成: {len(results)} 条 -> {OUTPUT_PATH}")
