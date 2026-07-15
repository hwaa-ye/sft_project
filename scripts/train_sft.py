"""
SFT 训练脚本：Qwen3-8B + LoRA（5090 32GB 优化版）
Phase 1: 数学推理链冷启动 SFT → 产出 SFT 模型 → 接 Phase 2 GRPO
"""
import os, gc, pickle, random, torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# ─── 环境优化 ───
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ─── ModelScope 缓存：优先用 autodl-fs 持久化目录 ───
if "MODELSCOPE_CACHE" not in os.environ:
    _persist = os.path.join(os.path.expanduser("~"), "autodl-fs", "model_cache")
    _cache_dir = _persist if os.path.isdir(_persist) else os.path.join(os.path.expanduser("~"), ".cache", "modelscope")
    os.environ.setdefault("MODELSCOPE_CACHE", _cache_dir)
    os.environ.setdefault("MODELSCOPE_CREDENTIALS_PATH", os.path.join(_cache_dir, "credentials"))

# ─── 配置 ───
MODEL_NAME = os.environ.get("SFT_MODEL_NAME")
if not MODEL_NAME:
    _local = os.path.join(os.path.expanduser("~"), "autodl-fs", "model_cache", "Qwen", "Qwen3-8B")
    MODEL_NAME = _local if os.path.isdir(_local) else "Qwen/Qwen3-8B"
BATCH_SIZE = int(os.environ.get("SFT_BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("SFT_GRAD_ACCUM", "8"))
LR = float(os.environ.get("SFT_LR", "2e-4"))
EPOCHS = int(os.environ.get("SFT_EPOCHS", "3"))
SEED = int(os.environ.get("SFT_SEED", "42"))
MAX_UPDATES = int(os.environ.get("SFT_MAX_UPDATES", "0"))
MAX_SAMPLES = int(os.environ.get("SFT_MAX_SAMPLES", "0"))
MAX_GRAD_NORM = float(os.environ.get("SFT_MAX_GRAD_NORM", "1.0"))
MAX_LENGTH = int(os.environ.get("SFT_MAX_LENGTH", "2048"))  # 推理链需要更长
DATA_DIR = os.environ.get("SFT_DATA_DIR", "data/tokenized")
OUTPUT_DIR = os.environ.get("SFT_OUTPUT_DIR", "output/sft_qwen3")
INIT_LORA = os.environ.get("SFT_INIT_LORA", "").strip()
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
        npy_ids = f"{data_dir}/input_ids.npy"
        npy_labels = f"{data_dir}/labels.npy"
        pkl_ids = f"{data_dir}/input_ids.pkl"
        pkl_labels = f"{data_dir}/labels.pkl"
        if os.path.exists(npy_ids) and os.path.exists(npy_labels):
            self.input_ids = np.load(npy_ids, allow_pickle=True)
            self.labels = np.load(npy_labels, allow_pickle=True)
        elif os.path.exists(pkl_ids) and os.path.exists(pkl_labels):
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
            torch.cat([
                torch.ones(ids.size(0), dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long),
            ])
        )
    return {
        "input_ids": torch.stack(padded_input_ids),
        "labels": torch.stack(padded_labels),
        "attention_mask": torch.stack(attention_mask),
    }

# ─── 训练循环 ───
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, global_update,
                    grad_accum_steps=GRAD_ACCUM, max_grad_norm=MAX_GRAD_NORM, log_every=10):
    model.train()
    epoch_loss = 0.0
    running_loss = 0.0
    running_micro_batches = 0
    accum_count = 0
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
        running_loss += original_loss
        running_micro_batches += 1
        loss = loss / grad_accum_steps
        accum_count += 1

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        del outputs

        should_update = accum_count == grad_accum_steps or step + 1 == len(dataloader)
        if should_update:
            if scaler is not None:
                scaler.unscale_(optimizer)
            if accum_count < grad_accum_steps:
                correction = grad_accum_steps / accum_count
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1
            accum_count = 0
            if global_update % log_every == 0:
                avg_loss = running_loss / max(1, running_micro_batches)
                lr = scheduler.get_last_lr()[0]
                print(f"  update {global_update} | loss {avg_loss:.4f} | lr {lr:.2e}")
                running_loss = 0.0
                running_micro_batches = 0
            if global_update % 500 == 0:
                ckpt = f"{OUTPUT_DIR}/checkpoint_update_{global_update}"
                model.save_pretrained(ckpt)
                torch.save({
                    "global_update": global_update,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }, f"{ckpt}/trainer_state.pt")
                print(f"  checkpoint 保存: {ckpt}")
            if MAX_UPDATES > 0 and global_update >= MAX_UPDATES:
                break

    return epoch_loss / max(1, step + 1), global_update

# ─── 主函数 ───
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    gc.collect()
    torch.cuda.empty_cache()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model

    print(f"设备: {device}")
    print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

    # 强制离线模式，禁止联网
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    # 加载 tokenizer
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    global pad_token_id
    pad_token_id = tokenizer.pad_token_id

    # 加载模型 — 先 CPU 后 GPU，避免显存碎片
    print("加载模型（先 CPU 后 GPU）...")
    gc.collect()
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=amp_dtype if use_amp else torch.float32,
        device_map=None,
        trust_remote_code=True,
        local_files_only=True,
    )
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if INIT_LORA:
        print(f"加载并合并 Stage-1 LoRA: {INIT_LORA}")
        model = PeftModel.from_pretrained(model, INIT_LORA, is_trainable=False)
        model = model.merge_and_unload()
        print("  Stage-1 LoRA 已合并；将为本阶段初始化全新的 LoRA")

    print(f"  模型加载后显存: {torch.cuda.memory_allocated()/1024**3:.1f}GB（应为 0，模型在 CPU）")

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
    if MAX_SAMPLES > 0:
        dataset.input_ids = dataset.input_ids[:MAX_SAMPLES]
        dataset.labels = dataset.labels[:MAX_SAMPLES]
        print(f"  限制训练样本: {len(dataset)} 条")
    generator = torch.Generator()
    generator.manual_seed(SEED)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
        num_workers=0, generator=generator,
    )

    # 优化器
    optimizer = AdamW(model.parameters(), lr=LR)
    updates_per_epoch = (len(dataloader) + GRAD_ACCUM - 1) // GRAD_ACCUM
    total_steps = updates_per_epoch * EPOCHS
    if MAX_UPDATES > 0:
        total_steps = min(total_steps, MAX_UPDATES)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.03),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda") if (use_amp and amp_dtype == torch.float16) else None

    print(f"\n训练配置:")
    print(f"  模型: {MODEL_NAME}")
    print(f"  初始 LoRA: {INIT_LORA or '无（直接从基座）'}")
    print(f"  总数据: {len(dataset)} 条")
    print(f"  batch: {BATCH_SIZE}, grad_accum: {GRAD_ACCUM}, 等效 batch: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  max_length: {MAX_LENGTH}")
    print(f"  seed: {SEED}")
    print(f"  max_updates: {MAX_UPDATES or '未设置'}")
    print(f"  max_samples: {MAX_SAMPLES or '未设置'}")
    print(f"  总更新步数: {total_steps}")

    global_update = 0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        avg_loss, global_update = train_one_epoch(
            model, dataloader, optimizer, scheduler, scaler, global_update
        )
        ckpt = f"{OUTPUT_DIR}/epoch_{epoch+1}"
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        print(f"Epoch {epoch+1} 完成 | avg loss: {avg_loss:.4f} | checkpoint: {ckpt}")
        if MAX_UPDATES > 0 and global_update >= MAX_UPDATES:
            print(f"达到 max_updates={MAX_UPDATES}，停止训练")
            break

    final = f"{OUTPUT_DIR}/final"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"\nSFT 训练完成! 模型保存到 {final}")

if __name__ == "__main__":
    main()
