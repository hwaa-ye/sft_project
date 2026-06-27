"""
vLLM 批量推理脚本：加载 merge 后的 SFT 模型，快速生成推理链
对比 HF generate: 速度 10-20x
"""
import json, os
from vllm import LLM, SamplingParams

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL_PATH = os.environ.get("SFT_MERGE_PATH", "output/sft_qwen3_merged")
TEST_PATH = os.environ.get("SFT_TEST_PATH", "data/test_math.jsonl")
OUTPUT_PATH = os.environ.get("SFT_PRED_PATH", "data/test_predictions.jsonl")
MAX_TOKENS = int(os.environ.get("SFT_MAX_NEW_TOKENS", "2048"))
TEMPERATURE = float(os.environ.get("SFT_TEMPERATURE", "0.7"))

print(f"Model: {MODEL_PATH}")
print(f"Max tokens: {MAX_TOKENS}, Temperature: {TEMPERATURE}")

# 加载测试数据
test_data = []
with open(TEST_PATH, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            test_data.append(json.loads(line))
print(f"测试数据: {len(test_data)} 条")

# 构建 prompt
prompts = [
    f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n"
    for item in test_data
]

# 加载 vLLM 模型
llm = LLM(model=MODEL_PATH, trust_remote_code=True, gpu_memory_utilization=0.85, max_model_len=4096)

sampling_params = SamplingParams(
    temperature=TEMPERATURE, top_p=0.95, max_tokens=MAX_TOKENS,
    stop=["<|im_end|>", "<|endoftext|>"],
)

# 批量推理
outputs = llm.generate(prompts, sampling_params)

# 写入结果
os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for i, (item, out) in enumerate(zip(test_data, outputs)):
        generated = out.outputs[0].text
        item["prediction"] = generated
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"推理完成: {len(test_data)} 条 -> {OUTPUT_PATH}")
