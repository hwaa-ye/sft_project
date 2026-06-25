"""
SFT 训练脚本（使用你手写的训练循环）
"""
import os, gc, pickle, json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ─── 配置 ───
MODEL_NAME = os.environ.get("SFT_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
BATCH_SIZE = int(os.environ.get("SFT_BATCH_SIZE", "2"))
GRAD_ACCUM = int(os.environ.get("SFT_GRAD_ACCUM", "4"))
LR = float(os.environ.get("SFT_LR", "2e-4"))
EPOCHS = int(os.environ.get("SFT_EPOCHS", "3"))
MAX_GRAD_NORM = float(os.environ.get("SFT_MAX_GRAD_NORM", "1.0"))
MAX_LENGTH = int(os.environ.get("SFT_MAX_LENGTH", "1024"))
DATA_DIR = os.environ.get("SFT_DATA_DIR", "data/tokenized")
OUTPUT_DIR = os.environ.get("SFT_OUTPUT_DIR", "output/sft")
TEST_MODE = os.environ.get("SFT_TEST_MODE", "").strip().lower() in {"1", "true", "yes", "y"}
TEST_SAMPLES = int(os.environ.get("SFT_TEST_SAMPLES", "100"))

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()
amp_dtype = (
    torch.bfloat16
    if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    else torch.float16
)
pad_token_id = 0

# ─── 数据集 ───
class SFTDataset(Dataset):
    def __init__(self, data_dir):
        npy_ids = f"{data_dir}/input_ids.npy"
        npy_labels = f"{data_dir}/labels.npy"
        pkl_ids = f"{data_dir}/input_ids.pkl"
        pkl_labels = f"{data_dir}/labels.pkl"
        if os.path.exists(npy_ids) and os.path.exists(npy_labels):
            import numpy as np
            self.input_ids = np.load(npy_ids, allow_pickle=True)
            self.labels = np.load(npy_labels, allow_pickle=True)
        elif os.path.exists(pkl_ids) and os.path.exists(pkl_labels):
            with open(pkl_ids, "rb") as f:
                self.input_ids = pickle.load(f)
            with open(pkl_labels, "rb") as f:
                self.labels = pickle.load(f)
        else:
            raise FileNotFoundError(f"数据文件未找到: {data_dir}/input_ids.[npy|pkl]")
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    max_len = min(max(ids.size(0) for ids in input_ids), MAX_LENGTH)
    padded_input_ids, padded_labels, attention_mask = [], [], []
    for i in range(len(input_ids)):
        ids = input_ids[i][:max_len]
        lbl = labels[i][:max_len]
        pad_len = max_len - ids.size(0)
        padded_input_ids.append(torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        padded_labels.append(torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)]))
        attention_mask.append(
            torch.cat(
                [
                    torch.ones(ids.size(0), dtype=torch.long),
                    torch.zeros(pad_len, dtype=torch.long),
                ]
            )
        )
    return {
        "input_ids": torch.stack(padded_input_ids),
        "labels": torch.stack(padded_labels),
        "attention_mask": torch.stack(attention_mask),
    }

# ─── 你手写的训练循环 ───
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler,
                    device,
                    grad_accum_steps=GRAD_ACCUM, max_grad_norm=MAX_GRAD_NORM, log_every=10):
    model.train()
    epoch_loss = 0.0
    running_loss = 0.0
    update_count = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        original_loss = loss.item()
        epoch_loss += original_loss

        loss = loss / grad_accum_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        del outputs

        if (step + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running_loss += original_loss
            update_count += 1
            if update_count % log_every == 0:
                avg_loss = running_loss / log_every
                print(f"  update {update_count} | avg loss {avg_loss:.4f}")
                running_loss = 0.0

    return epoch_loss / len(dataloader)

# ─── 主函数 ───
def main():
    from modelscope import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print(f"设备: {device}")
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    global pad_token_id
    pad_token_id = tokenizer.pad_token_id

    print("加载模型（先 CPU 后 GPU，避免显存碎片）...")
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=amp_dtype if use_amp else torch.float32,
        device_map=None,
        trust_remote_code=True,
    )
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.requires_grad_(False)
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if os.environ.get("SFT_GRAD_CHECKPOINT", "1").strip().lower() in {"1", "true", "yes", "y"}:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model = model.to(device)
    model.print_trainable_parameters()
    print(f"  移到 GPU 后显存: {torch.cuda.memory_allocated()/1024**3:.1f}GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    train_device = device

    print("加载数据...")
    dataset = SFTDataset(DATA_DIR)
    if TEST_MODE:
        dataset.input_ids = dataset.input_ids[:TEST_SAMPLES]
        dataset.labels = dataset.labels[:TEST_SAMPLES]
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(dataloader) * EPOCHS // GRAD_ACCUM
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.03), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda") if (use_amp and amp_dtype == torch.float16) else None

    if TEST_MODE:
        print(f"开始训练（测试模式：仅 {len(dataset)} 条数据）...")
    else:
        print(f"开始训练（全量数据：{len(dataset)} 条）...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        avg_loss = train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device=train_device)
        print(f"Epoch {epoch + 1} 平均 loss: {avg_loss:.4f}")

    # 保存
    save_path = f"{OUTPUT_DIR}/test" if TEST_MODE else f"{OUTPUT_DIR}/final"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"训练完成! 模型保存到 {save_path}")

if __name__ == "__main__":
    main()
