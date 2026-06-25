"""
SFT 训练脚本：Qwen2.5 + LoRA
流程: 加载 tokenized 数据 → 加载模型 → LoRA → 训练循环 → 保存
"""
import os, gc, pickle, math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ─── 配置 ───
class Config:
    # 模型
    model_name = os.environ.get("SFT_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")  # Mac 测试用 1.5B; 上 GPU 后改 7B

    # LoRA
    #一般取4，8，16，32
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # 训练
    batch_size = int(os.environ.get("SFT_BATCH_SIZE", "2"))
    grad_accum_steps = int(os.environ.get("SFT_GRAD_ACCUM", "4"))          # 等效 batch = batch_size * grad_accum
    learning_rate = float(os.environ.get("SFT_LR", "2e-4"))
    num_epochs = int(os.environ.get("SFT_EPOCHS", "3"))
    max_grad_norm = float(os.environ.get("SFT_MAX_GRAD_NORM", "1.0"))
    warmup_ratio = float(os.environ.get("SFT_WARMUP_RATIO", "0.03"))
    max_length = int(os.environ.get("SFT_MAX_LENGTH", "1024"))

    # 数据
    data_dir = os.environ.get("SFT_DATA_DIR", "data/tokenized")
    output_dir = os.environ.get("SFT_OUTPUT_DIR", "output/sft")

    # 混合精度
    use_amp = True

    # padding
    pad_token_id = 0

    # 保存
    save_every = 500              # 每 500 步保存一次
    log_every = 10                # 每 10 步打印一次 loss

config = Config()

def _get_amp_dtype():
    if not (config.use_amp and torch.cuda.is_available()):
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y"}

# ─── 数据集 ───
class SFTDataset(Dataset):
    """加载之前 tokenize 好的数据（优先 .npy，兼容 .pkl）"""
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
    """动态 padding：把一个 batch 内的序列 pad 到相同长度"""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    # 找 batch 内最大长度（不超过 max_length）
    max_len = min(max(ids.size(0) for ids in input_ids), config.max_length)

    # 截断并 padding（右 pad 到 max_len）
    padded_input_ids = []
    padded_labels = []
    attention_mask = []

    for i in range(len(input_ids)):
        ids = input_ids[i][:max_len]
        lbl = labels[i][:max_len]
        pad_len = max_len - ids.size(0)

        padded_input_ids.append(
            torch.cat([ids, torch.full((pad_len,), config.pad_token_id, dtype=torch.long)])
        )
        padded_labels.append(
            torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])
        )
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

# ─── 模型加载 ───
def load_model():
    from peft import LoraConfig, get_peft_model
    from modelscope import AutoModelForCausalLM, AutoTokenizer

    print(f"加载模型: {config.model_name}")
    amp_dtype = _get_amp_dtype()
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = tokenizer.pad_token_id

    # 先 CPU 后 GPU，避免 device_map="auto" 的显存碎片
    print(f"  可用显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=amp_dtype if amp_dtype is not None else torch.float32,
        device_map=None,
        trust_remote_code=True,
    )
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # 冻结基座模型
    model.requires_grad_(False)

    # LoRA 配置
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if _env_flag("SFT_GRAD_CHECKPOINT", True):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model = model.to("cuda")
    model.print_trainable_parameters()
    print(f"  移到 GPU 后显存: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    return model, tokenizer

# ─── 训练循环 ───
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    amp_dtype = _get_amp_dtype()
    use_amp = amp_dtype is not None

    # 加载数据和模型
    dataset = SFTDataset(config.data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Mac 上设 0，上 GPU 后可改 2-4
    )
    model, tokenizer = load_model()
    model.train()
    if not torch.cuda.is_available():
        model.to(device)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    # 总步数
    total_steps = len(dataloader) * config.num_epochs // config.grad_accum_steps
    warmup_steps = int(total_steps * config.warmup_ratio)

    # scheduler（线性 warmup + 线性衰减）
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 混合精度
    scaler = GradScaler("cuda") if (use_amp and amp_dtype == torch.float16) else None

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    # ─── 训练 loop ───
    global_step = 0
    total_loss = 0
    losses = []

    print(f"\n开始训练: {config.num_epochs} epoch, {total_steps} 步")
    print(f"  batch_size={config.batch_size}, grad_accum={config.grad_accum_steps}")
    print(f"  等效 batch size = {config.batch_size * config.grad_accum_steps}")
    print(f"  总数据量: {len(dataset)} 条")

    for epoch in range(config.num_epochs):
        for step, batch in enumerate(dataloader):
            # 数据移到 GPU
            batch = {k: v.to(device) for k, v in batch.items()}

            # 前向 + 反向
            with autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss / config.grad_accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            del outputs

            total_loss += loss.item()

            # 梯度累积：累积够 grad_accum_steps 步才更新参数
            if (step + 1) % config.grad_accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # 日志
                if global_step % config.log_every == 0:
                    avg_loss = total_loss / config.log_every
                    lr = scheduler.get_last_lr()[0]
                    print(f"  epoch {epoch+1} | step {global_step:>5d}/{total_steps} | loss {avg_loss:.4f} | lr {lr:.2e}")
                    losses.append(avg_loss)
                    total_loss = 0

                # 保存 checkpoint
                if global_step % config.save_every == 0:
                    save_path = f"{config.output_dir}/checkpoint_step_{global_step}"
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"  checkpoint 保存到 {save_path}")

    # ─── 最终保存 ───
    final_path = f"{config.output_dir}/final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n训练完成! 模型保存到 {final_path}")

    # 保存 loss 曲线
    with open(f"{config.output_dir}/losses.txt", "w") as f:
        for l in losses:
            f.write(f"{l}\n")

    return losses

if __name__ == "__main__":
    losses = train()
