# Stage2 Expanded SFT 统一评测 — 2026-07-15

## 模型与数据

- 基座：Qwen3-8B Base
- 起点：Stage1 clean SFT-repair LoRA
- Expanded hard：947 条
  - pilot hard：183
  - 新 DeepSeek V4 Pro teacher + verifier：764
  - 原模型 truncated：398
  - 原模型 wrong_complete：549
- source-balanced clean replay：1,555 条
- hard supervised-token fraction：19.98%
- 训练：1 epoch，313 updates，effective batch 8，LR 5e-5，seed 20260715
- 生成：Transformers，max_new_tokens 2048，temperature 0.7

## Hard validation（n=500）

该集合与所有 Stage2 训练和 teacher 数据隔离，使用可靠修复后的 gold。

| 模型 | Accuracy | Complete | Truncated |
|---|---:|---:|---:|
| Stage1 | 31.60% | 39.60% | 60.40% |
| Token-matched control | 34.60% | 44.00% | 56.00% |
| Stage2 v1（35% hard） | 43.40% | 85.60% | 14.40% |
| Stage2 v2（20% hard） | 43.60% | 77.00% | 23.00% |
| **Stage2 Expanded** | **47.60%** | **84.00%** | **16.00%** |

Expanded 相对 Stage1 修复 118 题、退化 38 题；McNemar exact/binomial
paired p=9.74e-11。相对 v2 修复 81 题、退化 61 题，单 seed paired
p=0.111，因此 +4.0pp 是正向趋势，但不应表述为已显著优于 v2。

### 按原始 SFT 长度

| 长度 | Stage1 | v1 | v2 | Expanded |
|---|---:|---:|---:|---:|
| <2.5k | 64.6% | 63.7% | 61.1% | **70.8%** |
| 2.5k–4k | 37.6% | 47.6% | 44.7% | **52.4%** |
| 4k–8k | 13.1% | 34.0% | 36.6% | **39.2%** |
| >=8k | 1.6% | 18.8% | **26.6%** | 14.1% |

主要收益覆盖 8k 以下；>=8k 是明确回退项，不能用总体 hard accuracy 掩盖。

## Mixed diagnostic（全量 n=484）

全量仍包含历史 gold 污染，只用于同口径诊断。

| 模型 | Accuracy | Complete | Truncated |
|---|---:|---:|---:|
| Stage1 | 66.12% | 82.02% | 17.98% |
| Token-matched control | 67.56% | 84.09% | 15.91% |
| v1 | 66.12% | 92.98% | 7.02% |
| v2 | 66.94% | 93.39% | 6.61% |
| **Expanded** | **67.15%** | **94.01%** | **5.99%** |

## Mixed reliable-gold subset（n=393）

只保留能从中间源数据最后一个 non-empty balanced `\\boxed{...}` 保守恢复
gold 的样本。

| 模型 | Accuracy | Truncated |
|---|---:|---:|
| 旧 SFT | 69.97% | 24.68% |
| Stage1 | 78.37% | 15.78% |
| Control | 79.64% | 13.99% |
| v1 | 77.86% | 6.11% |
| v2 | **79.90%** | 6.11% |
| Expanded | 79.39% | **5.60%** |

Expanded 与 v2 只差 2 题，属于单次采样噪声范围；相对 Stage1 的 mixed
accuracy 基本保持，完成性继续改善。

### Expanded 按来源（reliable subset）

| Source | n | Accuracy | Truncated |
|---|---:|---:|---:|
| EduChat-Math | 188 | 71.8% | 4.3% |
| Haijian/Advanced-Math | 2 | 50.0% | 50.0% |
| amc_aime | 46 | 58.7% | 19.6% |
| gavinluo/applied_math | 70 | 100.0% | 0.0% |
| meta-math/GSM8K_zh | 87 | 90.8% | 4.6% |

## 结论与后续

Expanded 是当前最均衡的 checkpoint：hard 从 Stage1 31.6% 提升至 47.6%，
reliable mixed 保持约 79%，mixed 截断率降至 5.6%。SFT 数据扩张开始出现边际
收益递减；下一阶段应以 Expanded 为 GRPO 起点，主要优化 182 条
complete-but-wrong hard failures。

进入 GRPO 前建议：

1. 固定 Expanded checkpoint 与当前 eval 文件 SHA256。
2. 使用程序可验证 answer reward；format reward 只占小权重。
3. 加 KL 与 easy anchor prompts，避免再次牺牲 mixed 能力。
4. 使用 vLLM 做 rollout，但所有 GRPO checkpoint 统一使用同一推理引擎评测。
5. 对 >=8k 单独建 probe；不要把它混在总体提升里。
