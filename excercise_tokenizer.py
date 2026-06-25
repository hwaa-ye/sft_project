import json
from modelscope import AutoTokenizer
from modelscope import AutoModelForCausalLM

#导入模型和分词器
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

input_file = "data/train.jsonl"
with open(input_file, "r", encoding="utf-8") as f:
    
        line = f.readline()
        example = json.loads(line)
    

instruction = example["input"]
response = example["target"]

instruction_tokens = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
full_text = instruction_tokens + response + "<|im_end|>"

max_length = 1024
full_tokens = tokenizer(full_text, max_length=max_length, truncation=True)

input_ids = full_tokens["input_ids"]

instruction_ids = tokenizer(instruction_tokens,add_special_tokens=False)
instruction_len = len(instruction_ids["input_ids"])

labels = input_ids.copy()
labels[:instruction_len+1] = [-100] * (instruction_len+1)

print("label 前20个:", labels[:20])  # 应该全是 -100                                                                                                                                                          
response_ids = [id_ for id_, lbl in zip(input_ids, labels) if lbl != -100]                                                                                                                                    
print("解码 response:", tokenizer.decode(response_ids)[:200])                