"""
PRM (Process Reward Model) 训练脚本
基座: Qwen3-1.5B + LoRA + 回归头
输入: 问题 + 推理步骤文本 → 输出: 步骤质量分数 (0~1)
损失: MSE(预测分, API标注分)
"""
import os, sys, json, gc
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ─── 配置 ───
BASE_MODEL = os.path.join(os.path.expanduser("~"), "autodl-fs", "model_cache", "Qwen", "Qwen3-1.5B")
DATA_PATH = os.environ.get("PRM_DATA", "prm/prm_train_data.jsonl")
OUTPUT_DIR = os.environ.get("PRM_OUTPUT", "output/prm_qwen3_1.5b")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = int(os.environ.get("PRM_BATCH", "8"))
LR = float(os.environ.get("PRM_LR", "2e-5"))
EPOCHS = int(os.environ.get("PRM_EPOCHS", "3"))
GRAD_ACCUM = int(os.environ.get("PRM_GRAD_ACCUM", "2"))
MAX_LENGTH = int(os.environ.get("PRM_MAX_LENGTH", "1024"))
LORA_R = int(os.environ.get("PRM_LORA_R", "8"))
LORA_ALPHA = int(os.environ.get("PRM_LORA_ALPHA", "16"))

device = torch.device("cuda:0")


# ─── 数据集 ───
class PRMDataset(Dataset):
    """每条样本: (问题, 单个推理步骤, API标注分数)"""
    def __init__(self, data_path):
        self.samples = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                question = item["question"]
                for step in item["steps"]:
                    self.samples.append({
                        "question": question,
                        "step_text": step["step_text"],
                        "score": step["score"],
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, tokenizer, max_length):
    """构建: <|im_start|>user\n评估以下推理步骤的质量。\n\n问题: {q}\n步骤: {step}\n<|im_end|>\n<|im_start|>assistant\n分数:"""
    texts = []
    for item in batch:
        prompt = (
            f"<|im_start|>user\n"
            f"评估以下数学推理步骤的质量。\n\n"
            f"问题: {item['question']}\n"
            f"步骤: {item['step_text']}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        texts.append(prompt)

    encoded = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=max_length,
    )

    # response 部分：只对 "assistant\n" 之后的部分计算 loss
    # 简化处理：取每个序列的最后 N 个 token 的 hidden state 平均
    scores = torch.tensor([item["score"] for item in batch], dtype=torch.float32)

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "scores": scores,
    }


# ─── 模型 ───
class ProcessRewardModel(nn.Module):
    """在 base 模型上加一个回归头"""
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        hidden_size = base_model.config.hidden_size
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),  # 约束输出 0~1
        )

    def forward(self, input_ids, attention_mask):
        # base 模型输出 hidden states
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # 取最后一层 hidden state
        last_hidden = outputs.hidden_states[-1]  # [B, seq, hidden]
        # 取每个序列最后一个非 padding token 的表示
        seq_lens = attention_mask.sum(dim=1) - 1  # [B]
        batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)
        last_token_hidden = last_hidden[batch_indices, seq_lens]  # [B, hidden]
        # 回归头预测分数
        score = self.score_head(last_token_hidden).squeeze(-1)  # [B]
        return score


def main():
    gc.collect()
    torch.cuda.empty_cache()

    print("=" * 50)
    print("PRM 训练配置")
    print(f"基座: {BASE_MODEL}")
    print(f"数据: {DATA_PATH}")
    print(f"Batch: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LR}")
    print(f"LoRA r={LORA_R}, alpha={LORA_ALPHA}")
    print("=" * 50)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 加载数据
    dataset = PRMDataset(DATA_PATH)
    print(f"训练样本: {len(dataset)} 个步骤")
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, MAX_LENGTH),
    )

    # 加载 base 模型
    print("\n加载 base 模型...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=None,
        trust_remote_code=True, local_files_only=True,
    ).to(device)

    # LoRA 配置
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    base = get_peft_model(base, lora_config)
    # 需要 hidden states 输出
    base.config.output_hidden_states = True

    model = ProcessRewardModel(base).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(dataloader) * EPOCHS // GRAD_ACCUM
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

    criterion = nn.MSELoss()

    # ─── 训练 ───
    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            scores = batch["scores"].to(device)

            preds = model(input_ids, attention_mask)
            loss = criterion(preds, scores) / GRAD_ACCUM
            loss.backward()
            epoch_loss += loss.item()

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if batch_idx % 50 == 0:
                # 计算一些调试指标
                mae = (preds - scores).abs().mean().item()
                print(f"  epoch {epoch+1} step {batch_idx:4d} | "
                      f"loss {loss.item()*GRAD_ACCUM:.4f} | mae {mae:.4f} | "
                      f"pred_mean {preds.mean().item():.3f} | true_mean {scores.mean().item():.3f}",
                      flush=True)

        avg_loss = epoch_loss / len(dataloader) * GRAD_ACCUM
        print(f"=== epoch {epoch+1}/{EPOCHS} avg_loss {avg_loss:.4f} ===", flush=True)

    # ─── 保存 ───
    final_path = f"{OUTPUT_DIR}/final"
    model.base.save_pretrained(final_path)
    # 单独保存回归头
    torch.save(model.score_head.state_dict(), f"{final_path}/score_head.pt")
    tokenizer.save_pretrained(final_path)
    print(f"\nPRM 训练完成! 模型: {final_path}")


if __name__ == "__main__":
    main()
