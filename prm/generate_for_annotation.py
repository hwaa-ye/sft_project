"""
用 GRPO/SFT 模型批量生成推理链，供 PRM 标注使用
输出: JSONL 文件，每条含 instruction/answer/prediction (含 <think> 标签)
"""
import json, os, sys, torch, gc, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

BASE_MODEL = os.path.join(os.path.expanduser("~"), "autodl-fs", "model_cache", "Qwen", "Qwen3-8B")
LORA_PATH = os.environ.get("PRM_GEN_LORA", "output/grpo_qwen3/step_200")  # 使用最佳 GRPO checkpoint
DATA_PATH = os.environ.get("PRM_GEN_DATA", "data/train_math_all.jsonl")
OUTPUT_PATH = os.environ.get("PRM_GEN_OUTPUT", "prm/gen_predictions.jsonl")
MAX_SAMPLES = int(os.environ.get("PRM_GEN_MAX", "1200"))
BATCH_SIZE = int(os.environ.get("PRM_GEN_BATCH", "8"))
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.8  # 要有多样性，便于 PRM 学到不同质量的推理

device = "cuda:0"


def main():
    gc.collect()
    torch.cuda.empty_cache()

    print(f"输出: {OUTPUT_PATH}, 最多 {MAX_SAMPLES} 条")
    if os.path.exists(OUTPUT_PATH):
        existing = sum(1 for _ in open(OUTPUT_PATH))
        print(f"已有 {existing} 条, 跳过")
        return

    # 加载数据（只取前 MAX_SAMPLES 条有答案的）
    all_data = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("answer"):
                    all_data.append(item)
    all_data = all_data[:MAX_SAMPLES]
    print(f"数据: {len(all_data)} 条")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 加载模型
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=None,
        trust_remote_code=True, local_files_only=True,
    ).to(device)
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model = model.to(device)
    model.eval()
    print(f"显存: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    prompts = [
        f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n"
        for item in all_data
    ]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    t0 = time.time()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i in range(0, len(all_data), BATCH_SIZE):
            batch_prompts = prompts[i:i + BATCH_SIZE]
            batch_items = all_data[i:i + BATCH_SIZE]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
                    do_sample=True, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            for j, (item, out_ids) in enumerate(zip(batch_items, outputs)):
                prompt_len = (inputs.attention_mask[j] == 1).sum().item()
                pred = tokenizer.decode(out_ids[prompt_len:], skip_special_tokens=True)
                item["prediction"] = pred
                item.pop("instruction", None)
                item.pop("input", None)
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

            done = min(i + BATCH_SIZE, len(all_data))
            if done % 200 == 0:
                print(f"  {done}/{len(all_data)} ({(time.time()-t0)/60:.0f}min)", flush=True)

    print(f"完成! {len(all_data)} 条, 耗时 {(time.time()-t0)/60:.0f}min, 输出: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
