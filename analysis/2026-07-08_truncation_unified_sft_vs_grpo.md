# 截断诊断（统一口径） — 2026-07-08

**统一截断定义**: 缺 `</think>` / 缺 `<answer>` / 缺 `</answer>` / answer 抽取为空，命中任一即判为截断。

> 注: 旧 compare_eval.py 只在「`</think>` 和 `</answer>` 同时缺失」时才算截断（过于宽松），本脚本统一为「任一缺失即截断」，因此截断率高于旧记录，这是尺子变准，不是模型变差。

## SFT (test_predictions_v2)  (n=484)

- accuracy: **67.77%**
- truncation_rate: **27.48%** (133/484)
- complete_rate: 72.52%
- avg_output_len: 1967 字符
- output_len p50/p90/p95/p99: 1340 / 3300 / 5362 / 7972
- 截断原因分布: {'no_</think>': 100, 'empty_answer': 33}

### by_source

| source | n | trunc | acc |
|---|---|---|---|
| amc_aime | 48 | 56.2% | 43.8% |
| EduChat-Math | 236 | 39.0% | 54.7% |
| Haijian/Advanced-Math | 6 | 33.3% | 33.3% |
| meta-math/GSM8K_zh | 105 | 7.6% | 88.6% |
| gavinluo/applied_math | 89 | 4.5% | 93.3% |

### by_gold_reasoning_len (gold 参考解字符长度)

| 长度桶 | n | trunc |
|---|---|---|
| <1k | 153 | 7.8% |
| 1k-2k | 157 | 17.2% |
| 2k-4k | 80 | 28.8% |
| 4k-8k | 53 | 64.2% |
| >8k | 41 | 90.2% |

## GRPO_100  (n=484)

- accuracy: **68.39%**
- truncation_rate: **26.03%** (126/484)
- complete_rate: 73.97%
- avg_output_len: 1805 字符
- output_len p50/p90/p95/p99: 1171 / 3242 / 4732 / 7696
- 截断原因分布: {'no_</think>': 87, 'empty_answer': 39}

### by_source

| source | n | trunc | acc |
|---|---|---|---|
| amc_aime | 48 | 52.1% | 47.9% |
| EduChat-Math | 236 | 37.3% | 54.2% |
| Haijian/Advanced-Math | 6 | 16.7% | 33.3% |
| meta-math/GSM8K_zh | 105 | 9.5% | 87.6% |
| gavinluo/applied_math | 89 | 2.2% | 96.6% |

### by_gold_reasoning_len (gold 参考解字符长度)

| 长度桶 | n | trunc |
|---|---|---|
| <1k | 153 | 8.5% |
| 1k-2k | 157 | 15.9% |
| 2k-4k | 80 | 27.5% |
| 4k-8k | 53 | 60.4% |
| >8k | 41 | 82.9% |

## GRPO_200  (n=484)

- accuracy: **71.90%**
- truncation_rate: **21.90%** (106/484)
- complete_rate: 78.10%
- avg_output_len: 1676 字符
- output_len p50/p90/p95/p99: 1059 / 3197 / 4830 / 7820
- 截断原因分布: {'no_</think>': 68, 'empty_answer': 37, 'no_<answer>': 1}

### by_source

| source | n | trunc | acc |
|---|---|---|---|
| amc_aime | 48 | 50.0% | 50.0% |
| EduChat-Math | 236 | 33.5% | 58.9% |
| Haijian/Advanced-Math | 6 | 16.7% | 50.0% |
| meta-math/GSM8K_zh | 105 | 1.9% | 89.5% |
| gavinluo/applied_math | 89 | 0.0% | 98.9% |

### by_gold_reasoning_len (gold 参考解字符长度)

| 长度桶 | n | trunc |
|---|---|---|
| <1k | 153 | 7.2% |
| 1k-2k | 157 | 12.1% |
| 2k-4k | 80 | 20.0% |
| 4k-8k | 53 | 49.1% |
| >8k | 41 | 82.9% |

## GRPO_300  (n=484)

- accuracy: **70.25%**
- truncation_rate: **21.69%** (105/484)
- complete_rate: 78.31%
- avg_output_len: 1565 字符
- output_len p50/p90/p95/p99: 950 / 3129 / 4717 / 7435
- 截断原因分布: {'no_</think>': 66, 'empty_answer': 39}

### by_source

| source | n | trunc | acc |
|---|---|---|---|
| amc_aime | 48 | 45.8% | 54.2% |
| EduChat-Math | 236 | 33.1% | 55.1% |
| Haijian/Advanced-Math | 6 | 16.7% | 33.3% |
| meta-math/GSM8K_zh | 105 | 2.9% | 92.4% |
| gavinluo/applied_math | 89 | 1.1% | 95.5% |

### by_gold_reasoning_len (gold 参考解字符长度)

| 长度桶 | n | trunc |
|---|---|---|
| <1k | 153 | 7.8% |
| 1k-2k | 157 | 14.0% |
| 2k-4k | 80 | 17.5% |
| 4k-8k | 53 | 52.8% |
| >8k | 41 | 70.7% |

## 汇总对比（统一口径）

| 模型 | n | accuracy | truncation_rate |
|---|---|---|---|
| SFT (test_predictions_v2) | 484 | 67.77% | 27.48% |
| GRPO_100 | 484 | 68.39% | 26.03% |
| GRPO_200 | 484 | 71.90% | 21.90% |
| GRPO_300 | 484 | 70.25% | 21.69% |
