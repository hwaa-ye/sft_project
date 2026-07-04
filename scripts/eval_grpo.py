"""
批量 GRPO 评估推理：逐个 LoRA checkpoint 推理，每次重新加载 base 模型
"""
import json, os, sys, torch, gc, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

BASE_MODEL = os.path.join(os.path.expanduser("~"), "autodl-fs", "model_cache", "Qwen", "Qwen3-8B")
TEST_PATH = "data/test_math.jsonl"
OUTPUT_DIR = "output/grpo_qwen3"

# 只需要跑 GRPO checkpoint，SFT 已有 test_predictions_v2.jsonl
RUNS = [
    ("grpo_100", f"{OUTPUT_DIR}/step_100"),
    ("grpo_200", f"{OUTPUT_DIR}/step_200"),
    ("grpo_300", f"{OUTPUT_DIR}/step_300"),
]

BATCH_SIZE = 8
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.7

device = "cuda:0"

# 加载 tokenizer（一次）
print("加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# 加载测试数据
test_data = []
with open(TEST_PATH, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            test_data.append(json.loads(line))
print(f"测试数据: {len(test_data)} 条\n")

prompts = [
    f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n"
    for item in test_data
]

for name, lora_path in RUNS:
    pred_path = f"{OUTPUT_DIR}/predictions_{name}.jsonl"
    if os.path.exists(pred_path):
        print(f"[{name}] 已有预测文件, 跳过 (wc -l: {sum(1 for _ in open(pred_path))} 行)")
        continue

    print(f"[{name}] 加载 base 模型...")
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=None,
        trust_remote_code=True, local_files_only=True,
    ).to(device)

    print(f"[{name}] 加载 LoRA: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.to(device)
    model.eval()
    print(f"[{name}] 显存: {torch.cuda.memory_allocated()/1024**3:.1f}GB, 开始推理...")

    t0 = time.time()
    out_f = open(pred_path, "w", encoding="utf-8")
    for i in range(0, len(test_data), BATCH_SIZE):
        batch_prompts = prompts[i:i + BATCH_SIZE]
        batch_items = test_data[i:i + BATCH_SIZE]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
                do_sample=True, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for j, (item, out_ids) in enumerate(zip(batch_items, outputs)):
            real_prompt_len = (inputs.attention_mask[j] == 1).sum().item()
            generated = tokenizer.decode(out_ids[real_prompt_len:], skip_special_tokens=True)
            item["prediction"] = generated
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            out_f.flush()

        done = min(i + BATCH_SIZE, len(test_data))
        if done % 100 == 0 or done == len(test_data):
            elapsed = (time.time() - t0) / 60
            print(f"  [{name}] {done}/{len(test_data)} ({elapsed:.0f}min)", flush=True)

    out_f.close()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[{name}] 完成, 耗时 {(time.time()-t0)/60:.0f}min")

print("\n全部完成!")
for name, _ in RUNS:
    print(f"  {OUTPUT_DIR}/predictions_{name}.jsonl")
