"""
GRPO 训练脚本：基于 SFT 模型，用组内相对优势做强化学习微调
核心：rollout → reward → advantage → PPO clip loss + KL penalty → update
"""
import os, sys, gc, torch, json, pickle, re, copy
# 确保 scripts 目录在 path 中（AutoDL 上从项目根目录运行）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
from grpo_reward import compute_reward, extract_answer, check_answer

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ─── 配置（可通过环境变量覆盖） ───
BASE_MODEL = os.path.join(os.path.expanduser("~"), "autodl-fs", "model_cache", "Qwen", "Qwen3-8B")
SFT_LORA = os.environ.get("GRPO_SFT_LORA", "output/sft_qwen3/final")
DATA_PATH = os.environ.get("GRPO_DATA", "data/train_math_all.jsonl")
OUTPUT_DIR = os.environ.get("GRPO_OUTPUT_DIR", "output/grpo_qwen3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPTS_PER_STEP = int(os.environ.get("GRPO_PROMPTS", "4"))     # 每步多少道题
RESPONSES_PER_PROMPT = int(os.environ.get("GRPO_RESPONSES", "4"))  # 每道题生成几个回答
PPO_EPOCHS = int(os.environ.get("GRPO_PPO_EPOCHS", "2"))       # 同一批数据重复更新次数
GRAD_ACCUM = int(os.environ.get("GRPO_GRAD_ACCUM", "2"))
LR = float(os.environ.get("GRPO_LR", "5e-5"))
CLIP_EPS = float(os.environ.get("GRPO_CLIP", "0.2"))
KL_BETA = float(os.environ.get("GRPO_KL_BETA", "0.04"))
MAX_STEPS = int(os.environ.get("GRPO_MAX_STEPS", "500"))
MAX_LENGTH = int(os.environ.get("GRPO_MAX_LENGTH", "1536"))
GEN_MAX_NEW = int(os.environ.get("GRPO_GEN_MAX_NEW", "2048"))
GEN_TEMP = float(os.environ.get("GRPO_TEMP", "0.8"))

device = torch.device("cuda:0")
torch.cuda.empty_cache()
amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

print("=" * 55)
print("GRPO 训练配置")
print("=" * 55)
print(f"SFT LoRA: {SFT_LORA}")
print(f"每步题目: {PROMPTS_PER_STEP}, 每题生成: {RESPONSES_PER_PROMPT}")
print(f"PPO epochs/步: {PPO_EPOCHS}, grad_accum: {GRAD_ACCUM}")
print(f"LR: {LR}, clip: {CLIP_EPS}, KL beta: {KL_BETA}")
print(f"max_steps: {MAX_STEPS}")

# ─── 加载 tokenizer ───
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
pad_token_id = tokenizer.pad_token_id

# ─── 加载数据 ───
all_data = []
with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            all_data.append(json.loads(line))
# 过滤：只用有 answer 字段的数据
all_data = [d for d in all_data if d.get("answer") is not None and str(d["answer"]).strip()]
print(f"训练数据: {len(all_data)} 条（有答案）")

# ─── 辅助函数：log probability 计算（逐条处理，避免 logits 爆显存） ───
def _seq_logprob(model, input_ids_row, attn_mask_row, resp_mask_row):
    """计算单条序列的 response log probability（纯 bf16，避免 loss 函数 float 转换 OOM）"""
    labels = input_ids_row.clone()
    labels[~resp_mask_row] = -100
    with torch.amp.autocast("cuda", dtype=amp_dtype):
        out = model(input_ids=input_ids_row.unsqueeze(0),
                    attention_mask=attn_mask_row.unsqueeze(0))
    # 手动在 bf16 下算 CE loss，避免 logits.float() 爆显存
    logits = out.logits  # [1, seq, vocab] in bf16
    shift_logits = logits[:, :-1, :].squeeze(0)  # [seq-1, vocab]
    shift_labels = input_ids_row[1:]  # [seq-1]
    ce = nn.functional.cross_entropy(shift_logits, shift_labels, reduction="none")
    # 只取 response 部分的 loss
    resp_shift = resp_mask_row[1:]
    if resp_shift.sum() == 0:
        return torch.tensor(0.0, device=device)
    masked_ce = ce * resp_shift.float()
    total_ce = masked_ce.sum()
    n_tokens = resp_shift.sum().float()
    avg_ce = total_ce / n_tokens
    return -avg_ce * n_tokens  # total log probability


def compute_all_logprobs(model, input_ids, attention_mask, response_mask):
    """逐条计算 batch 中每条序列的 log probability"""
    logprobs = []
    for i in range(input_ids.size(0)):
        lp = _seq_logprob(model, input_ids[i], attention_mask[i], response_mask[i])
        logprobs.append(lp)
    return torch.stack(logprobs)


# ─── LoRA 权重管理 ───
def get_lora_state(model):
    """提取所有 LoRA 参数的 state_dict"""
    return {k: v.data.clone() for k, v in model.named_parameters() if "lora_" in k}


def set_lora_state(model, state):
    """将 state_dict 加载到模型的 LoRA 参数中"""
    for k, v in model.named_parameters():
        if k in state:
            v.data.copy_(state[k])


# ─── 主函数 ───
def main():
    gc.collect()
    torch.cuda.empty_cache()

    print("\n加载 base 模型...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=amp_dtype, device_map=None,
        trust_remote_code=True, local_files_only=True,
    )
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # 加载 SFT LoRA
    print("加载 SFT LoRA...")
    model = PeftModel.from_pretrained(model, SFT_LORA)
    model = model.to(device)

    # 保存 SFT 权重作为 reference
    sft_lora_state = get_lora_state(model)
    print(f"  SFT reference 权重已保存 ({len(sft_lora_state)} 个参数)")

    # 训练模式：LoRA 参数可训练，base 冻结
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 开启梯度检查点，节省训练显存
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    print(f"  显存: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    # 优化器（只优化 LoRA 参数）
    opt_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(opt_params, lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=20, num_training_steps=MAX_STEPS
    )

    # ─── 训练循环 ───
    step = 0
    while step < MAX_STEPS:
        # 采样 prompt
        indices = np.random.choice(len(all_data), PROMPTS_PER_STEP, replace=False)
        batch_data = [all_data[i] for i in indices]

        # ─── 1. Rollout：为每个 prompt 生成 N 个回答（分批生成，控制显存） ───
        prompts = []
        for item in batch_data:
            p = f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n"
            for _ in range(RESPONSES_PER_PROMPT):
                prompts.append(p)

        model.eval()
        model.config.use_cache = True  # 生成时开启 KV cache

        # 分批生成：每次生成一个 prompt 的 N 个回答
        all_responses = []
        all_rewards = []
        for p_idx in range(PROMPTS_PER_STEP):
            start = p_idx * RESPONSES_PER_PROMPT
            end = start + RESPONSES_PER_PROMPT
            batch_prompts = prompts[start:end]
            batch_items_text = batch_data[p_idx:p_idx+1] * RESPONSES_PER_PROMPT

            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=GEN_MAX_NEW, temperature=GEN_TEMP,
                    do_sample=True, pad_token_id=pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            for b_idx, (item, out_ids) in enumerate(zip(batch_items_text, outputs)):
                prompt_len = (inputs.attention_mask[b_idx] == 1).sum().item()
                response = tokenizer.decode(out_ids[prompt_len:], skip_special_tokens=True)
                all_responses.append(response)
                reward_info = compute_reward(response, str(item["answer"]))
                all_rewards.append(reward_info)

        responses = all_responses
        rewards_list = all_rewards

        # 清释放生成缓存，关闭 KV cache 准备训练
        model.config.use_cache = False
        del outputs, all_responses, all_rewards
        torch.cuda.empty_cache()
        gc.collect()

        # 重整为 [PROMPTS_PER_STEP, RESPONSES_PER_PROMPT]
        rewards_tensor = torch.tensor(
            [r["total"] for r in rewards_list], dtype=torch.float32, device=device
        ).view(PROMPTS_PER_STEP, RESPONSES_PER_PROMPT)

        # ─── 2. 计算 advantage（组内归一化） ───
        mean_r = rewards_tensor.mean(dim=1, keepdim=True)
        std_r = rewards_tensor.std(dim=1, keepdim=True) + 1e-8
        advantages = (rewards_tensor - mean_r) / std_r
        advantages = advantages.view(-1).to(device)  # flatten + to GPU

        # ─── 3. 构造训练 batch：prompt + response 拼接 ───
        full_texts = [
            p + r for p, r in zip(prompts, responses)
        ]
        encoded = tokenizer(
            full_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_LENGTH,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # 构建 response_mask：标记哪些 token 是 response 部分
        prompt_enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        prompt_lens = (prompt_enc["attention_mask"] == 1).sum(dim=1)

        response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for b_idx in range(input_ids.size(0)):
            p_len = min(prompt_lens[b_idx], input_ids.size(1))
            response_mask[b_idx, p_len:] = True
        # 对齐 padding
        response_mask = response_mask & (attention_mask == 1)

        # ─── 4. 计算 reference log probs ───
        current_lora = get_lora_state(model)
        set_lora_state(model, sft_lora_state)
        model.eval()

        with torch.no_grad():
            ref_logprobs = compute_all_logprobs(
                model, input_ids, attention_mask, response_mask
            )

        # 恢复 policy 权重
        set_lora_state(model, current_lora)

        # ─── 5. 计算 old_logprobs（policy 更新前的概率，no_grad） ───
        model.eval()
        with torch.no_grad():
            old_logprobs = compute_all_logprobs(
                model, input_ids, attention_mask, response_mask
            )  # [B]

        # ─── 6. PPO 更新（逐条 backward，释放中间激活） ───
        B = input_ids.size(0)
        ppo_loss_val = 0.0
        for ppo_epoch in range(PPO_EPOCHS):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss = 0.0
            for i in range(B):
                # 单条 log prob（with grad）
                pol_lp = _seq_logprob(
                    model, input_ids[i], attention_mask[i], response_mask[i]
                )  # scalar

                # PPO loss（单条）
                ratio_i = torch.exp(pol_lp - old_logprobs[i])
                loss_clip_i = -torch.min(
                    ratio_i * advantages[i],
                    torch.clamp(ratio_i, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages[i],
                )
                kl_i = ref_logprobs[i] - pol_lp
                loss_i = (loss_clip_i + KL_BETA * kl_i) / (B * GRAD_ACCUM)
                loss_i.backward()  # 立即释放本条计算图
                epoch_loss += loss_i.item()

            ppo_loss_val = epoch_loss

            if (ppo_epoch + 1) % GRAD_ACCUM == 0:
                nn.utils.clip_grad_norm_(opt_params, 1.0)
                optimizer.step()
                scheduler.step()

        step += 1
        if step % 10 == 0:
            avg_r = rewards_tensor.mean().item()
            acc = sum(r["accuracy"] for r in rewards_list) / len(rewards_list)
            comp = sum(r["completeness"] for r in rewards_list) / len(rewards_list)
            # KL 近似（用第一次 PPO epoch 的值）
            print(
                f"  step {step:4d}/{MAX_STEPS} | "
                f"reward {avg_r:.3f} | acc {acc:.2f} | comp {comp:.2f} | "
                f"loss {ppo_loss_val:.4f} | lr {scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )

        if step % 100 == 0:
            ckpt = f"{OUTPUT_DIR}/step_{step}"
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"  checkpoint: {ckpt}")

        if step >= MAX_STEPS:
            break

    # 保存最终模型
    final_path = f"{OUTPUT_DIR}/final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nGRPO 训练完成! 模型: {final_path}")


if __name__ == "__main__":
    main()
