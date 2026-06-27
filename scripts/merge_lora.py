"""
Merge LoRA adapter 进 base model，产出完整权重供 vLLM 加载
"""
import os, torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

BASE_MODEL = os.path.join(os.path.expanduser("~"), "autodl-fs", "model_cache", "Qwen", "Qwen3-8B")
LORA_PATH = os.environ.get("SFT_LORA_PATH", "output/sft_qwen3/final")
MERGE_PATH = os.environ.get("SFT_MERGE_PATH", "output/sft_qwen3_merged")

print(f"Base: {BASE_MODEL}")
print(f"LoRA: {LORA_PATH}")
print(f"Output: {MERGE_PATH}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True, local_files_only=True,
)
model = PeftModel.from_pretrained(model, LORA_PATH)
merged = model.merge_and_unload()

os.makedirs(MERGE_PATH, exist_ok=True)
merged.save_pretrained(MERGE_PATH, safe_serialization=True)
tokenizer.save_pretrained(MERGE_PATH)

print(f"Merge done: {MERGE_PATH}")
print(f"Size: {sum(p.numel() for p in merged.parameters())/1e9:.1f}B params")
