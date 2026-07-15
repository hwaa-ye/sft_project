# Stage2 SFT handoff（暂停至 2026-08）

## 当前推荐 checkpoint

远端：`output/stage2_expanded_experiment/final`

加载顺序不能省略：

1. 加载 Qwen3-8B Base。
2. 加载并 merge `output/sft_clean_repair_v1/final`。
3. 在 merged Stage1 上加载 `output/stage2_expanded_experiment/final`。

Stage2 adapter 不是直接相对 Base 训练的；若跳过 Stage1 merge，结果无效。

## Expanded 训练配置

- JSONL：`data/stage2_expanded/experiment_hard_replay.jsonl`
- hard：947
- clean replay：1,555
- hard supervised-token fraction：19.98%
- tokenized samples：2,502，0 skipped，max length 2046
- LR：5e-5
- epochs：1
- effective batch：8
- updates：313
- seed：20260715

## 核心结果

- Hard validation：47.60% accuracy，16.00% truncated
- Mixed reliable-gold subset：79.39% accuracy，5.60% truncated
- 详细报告：`analysis/2026-07-15_stage2_expanded_eval.md`

## 8 月恢复顺序

1. 校验本地归档 SHA256，并解压LoRA、teacher数据、最终JSONL、日志和预测。
2. 不再继续扩大同类 SFT；以 Expanded 为固定 SFT baseline。
3. 单独安装 vLLM 环境，避免改坏现有训练环境。
4. 建 GRPO pilot：可靠答案、exact/symbolic reward、轻量format reward、KL、easy anchors。
5. 先跑小规模 rollout/训练，使用 hard 500 + mixed reliable 393 做验收。
6. 再做 ablation：SFT Expanded、GRPO无KL、GRPO+KL、不同reward、Pass@1/Pass@k。

## 本地归档

目录：`/Users/yeouhan/sft_project_archive_2026-07-15`

远端归档被拆为 `stage2_essential.tar.gz.part-00` 至 `part-04`。下载完成后：

```bash
cd /Users/yeouhan/sft_project_archive_2026-07-15
cat stage2_essential.tar.gz.part-* > stage2_essential.tar.gz
shasum -a 256 stage2_essential.tar.gz
tar -tzf stage2_essential.tar.gz >/dev/null
```

SHA256 必须与目录中的 `SHA256SUMS` 一致，然后才解压。

## GitHub 完整恢复点

- commit：`79dd4cf015aacf6d48b4ef04956c67d8975bba2b`
- tag：`stage2-expanded-complete-2026-07-15`
- Stage1 adapter：`output/clean_sft_res/sft_clean_repair_v1/final`
- Expanded adapter：`output/stage2_expanded_experiment/final`
- 最终训练数据：`data/stage2_expanded/experiment_hard_replay.jsonl`
- teacher accepted：`data/hard_stage2_deepseek_v2_1156_min128/accepted.jsonl`
- 原始评测输出：`output/stage2_expanded_eval/{hard,mixed}.jsonl`

已逐项比较 AutoDL 与本地 SHA256：Expanded adapter、最终训练 JSONL、
teacher accepted、hard predictions、mixed predictions 均完全一致。

## 不应提交到普通 Git 的内容

- `adapter_model.safetensors`
- 完整 predictions JSONL
- teacher queue / accepted / rejected JSONL
- tokenized numpy/pickle
- 任何 API key 或 shell 环境快照

GitHub 只保存脚本、manifest、分析报告和本交接说明。
