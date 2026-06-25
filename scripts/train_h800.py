"""
SFT 完整训练脚本 - AutoDL H800 优化版
修复了 OOM 问题，直接运行即可
"""
import os, gc, pickle, torch, json
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# ─── 环境优化 ───
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ─── 配置 ───
MODEL_NAME = os.environ.get("SFT_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
BATCH_SIZE = int(os.environ.get("SFT_BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("SFT_GRAD_ACCUM", "8"))
LR = float(os.environ.get("SFT_LR", "2e-4"))
EPOCHS = int(os.environ.get("SFT_EPOCHS", "3"))
MAX_GRAD_NORM = float(os.environ.get("SFT_MAX_GRAD_NORM", "1.0"))
MAX_LENGTH = int(os.environ.get("SFT_MAX_LENGTH", "1024"))
DATA_DIR = os.environ.get("SFT_DATA_DIR", "data/tokenized")
OUTPUT_DIR = os.environ.get("SFT_OUTPUT_DIR", "output/sft_final")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda:0")
torch.cuda.empty_cache()
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
        # 优先加载 .npy（内存效率高），兼容 .pkl
        npy_ids = f"{data_dir}/input_ids.npy"
        npy_labels = f"{data_dir}/labels.npy"
        pkl_ids = f"{data_dir}/input_ids.pkl"
        pkl_labels = f"{data_dir}/labels.pkl"
        if os.path.exists(npy_ids) and os.path.exists(npy_labels):
            self.input_ids = np.load(npy_ids, allow_pickle=True)
            self.labels = np.load(npy_labels, allow_pickle=True)
        elif os.path.exists(pkl_ids) and os.path.exists(pkl_labels):
            import pickle
            with open(pkl_ids, "rb") as f:
                self.input_ids = pickle.load(f)
            with open(pkl_labels, "rb") as f:
                self.labels = pickle.load(f)
        else:
            raise FileNotFoundError(f"数据文件未找到: {data_dir}/input_ids.[npy|pkl]")
        print(f"  数据加载完成: {len(self.input_ids)} 条")
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

# ─── 训练循环 ───
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler,
                    grad_accum_steps=GRAD_ACCUM, max_grad_norm=MAX_GRAD_NORM, log_every=10):
    model.train()
    epoch_loss = 0.0
    running_loss = 0.0
    update_count = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

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
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running_loss += original_loss
            update_count += 1
            if update_count % log_every == 0:
                avg_loss = running_loss / log_every
                lr = scheduler.get_last_lr()[0]
                print(f"  update {update_count} | loss {avg_loss:.4f} | lr {lr:.2e}")
                running_loss = 0.0

        # 每 500 步存一次
        if update_count > 0 and update_count % 500 == 0:
            ckpt = f"{OUTPUT_DIR}/checkpoint_update_{update_count}"
            model.save_pretrained(ckpt)
            print(f"  checkpoint 保存: {ckpt}")

    return epoch_loss / len(dataloader)

# ─── 主函数 ───
def main():
    gc.collect()
    torch.cuda.empty_cache()

    from modelscope import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print(f"设备: {device}")
    print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

    # 加载 tokenizer
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    global pad_token_id
    pad_token_id = tokenizer.pad_token_id

    # 加载模型 - 关键：先加载到 CPU，再手动移到 GPU
    print("加载模型（先 CPU 后到 GPU）...")
    gc.collect()
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=amp_dtype if use_amp else torch.float32,
        device_map=None,  # 不自动分配
        trust_remote_code=True,
    )
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    print(f"  模型加载后显存: {torch.cuda.memory_allocated()/1024**3:.1f}GB（应该在 0GB，因为还在 CPU）")

    # 冻结 + LoRA
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

    # 加载数据
    print("加载数据...")
    dataset = SFTDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(dataloader) * EPOCHS // GRAD_ACCUM
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.03),
        num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda") if (use_amp and amp_dtype == torch.float16) else None

    print(f"\n训练配置:")
    print(f"  总数据: {len(dataset)} 条")
    print(f"  batch: {BATCH_SIZE}, grad_accum: {GRAD_ACCUM}, 等效 batch: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  总更新步数: {total_steps}")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        avg_loss = train_one_epoch(model, dataloader, optimizer, scheduler, scaler)
        ckpt = f"{OUTPUT_DIR}/epoch_{epoch+1}"
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        print(f"Epoch {epoch+1} 完成 | avg loss: {avg_loss:.4f} | checkpoint 保存到 {ckpt}")

    final = f"{OUTPUT_DIR}/final"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"\n训练完成! 最终模型保存到 {final}")

if __name__ == "__main__":
    main()
