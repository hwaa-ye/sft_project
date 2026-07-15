# Step 1: clean SFT-repair 数据构建

## 目标与边界

首个 repair 实验只修复右截断造成的坏监督：训练集中的每条样本都必须以完整
Qwen3 ChatML 形式落在 2048 token 内，且保留完整 `</think>`、`</answer>` 和
`<|im_end|>`。本阶段不压缩长推理、不修正答案语义、不改 GRPO。

## 构建策略

- 输入：`data/train_math_all.jsonl`
- 必需字段：非空字符串 `instruction`、`reasoning`、`answer`
- 长度口径：完整的 user + assistant ChatML，Qwen3-8B tokenizer，
  `add_special_tokens=False`
- 保留条件：完整格式化样本 `token_length <= 2048`
- 超长样本：原文写入 rejected 文件，供后续长难题压缩阶段使用；不截断
- 每条 clean 样本增加 `sft_token_length`，保留原始 `source`
- manifest 记录输入 SHA256、tokenizer、阈值、总量、来源和长度统计

## 执行命令

```bash
cd /root/autodl-tmp/sft_project

python3 scripts/build_clean_sft_repair.py \
  --input data/train_math_all.jsonl \
  --output-dir data/clean_sft_repair_2048 \
  --tokenizer /root/autodl-fs/model_cache/Qwen/Qwen3-8B \
  --max-length 2048 \
  --overwrite

SFT_TRAIN_JSONL=data/clean_sft_repair_2048/train.jsonl \
SFT_TOKENIZED_DIR=data/tokenized_clean_sft_repair_2048 \
SFT_MODEL_NAME=/root/autodl-fs/model_cache/Qwen/Qwen3-8B \
SFT_MAX_LENGTH=2048 \
python3 scripts/tokenize_dataset.py
```

本机 tokenizer 路径为
`/Users/yeouhan/.cache/modelscope/hub/models/Qwen/Qwen3-8B`；AutoDL 应按实际缓存
位置调整 `--tokenizer` 与 `SFT_MODEL_NAME`。

## 本机产物与结果

- `data/clean_sft_repair_2048/train.jsonl`: 26,474 条
- `data/clean_sft_repair_2048/rejected.jsonl`: 14,125 条
  - 缺必需文本：7,000 条
  - 完整格式超过 2048 token：7,125 条
- `data/clean_sft_repair_2048/manifest.json`: 构建审计信息
- `data/tokenized_clean_sft_repair_2048/`: 26,474 条 tokenized 样本
- tokenized 长度：min 177，max 2048
- tokenizer 跳过：0 条

在 33,599 条字段完整的有效样本中，26,474 条（78.8%）进入首版 repair 数据，
7,125 条（21.2%）进入长样本隔离集。

## 防回归约束

`scripts/tokenize_dataset.py` 已移除 `truncation=True`。如果输入出现任何超过
`SFT_MAX_LENGTH` 的样本，脚本会报出行号和真实长度并立即失败，不再静默生成
半截 target。训练时必须显式设置：

```bash
SFT_DATA_DIR=data/tokenized_clean_sft_repair_2048
SFT_MAX_LENGTH=2048
```

## Step 1 验收标准

- clean JSONL 条数与 manifest 的 kept 数一致
- clean + rejected 等于输入总行数
- tokenized 条数等于 clean JSONL 条数
- 所有 tokenized 长度不超过 2048
- input/label 逐条等长，每条至少有一个非 `-100` label
- tokenize 日志中 `跳过 0 条`

上述检查已在本机通过。

## 已知但不在本阶段处理的问题

抽样发现部分 `amc_aime` 的 `answer` 字段可能与 reasoning 末尾的 boxed 答案不符，
例如答案字段出现无意义短字符串。它属于答案语义质量问题，不能与右截断修复混为
同一个变量。进入训练前建议先做一次只读审计，再决定是否开展单独的答案重抽取实验。

## 2026-07-11：答案修复 v2

审计确认旧 `clean_answer` 使用了错误的字符集合前缀正则，并且旧 boxed 正则不支持
嵌套花括号。为避免重新下载整个 R1 数据集，使用 `source + instruction + reasoning`
将 clean v1 精确回连到 `train_math.jsonl` 和 `train_math_r1.jsonl`。

保守修复规则只接受中间文件 `answer` 字段中最后一个非空、括号平衡的
`\boxed{...}`；不从自由文本或 reasoning 猜测答案。

- clean v1 输入：26,474
- 接受：25,771
- 中间 answer 没有可靠 balanced box：699
- 修复答案后超过 2048 token：4
- join missing：0
- v2 tokenize：25,771，跳过 0

产物：

- `data/clean_sft_repair_v2_2048/train.jsonl`
- `data/clean_sft_repair_v2_2048/rejected.jsonl`
- `data/clean_sft_repair_v2_2048/manifest.json`
- `data/tokenized_clean_sft_repair_v2_2048/`

训练应使用 v2 tokenized 目录，不再使用 v1。

## 2026-07-11：最终一致性门

对 v2 中 reasoning 自带非空 boxed 的样本进行严格字符串一致性检查。67 条的最后
reasoning box 与修复 answer 不完全相同；即使其中包含数学等价表达，也统一保守剔除，
避免为极少量样本引入新的语义归一化规则。

- final JSONL：25,704 条
- 一致性门剔除：67 条
- final tokenize：25,704 条，跳过 0
- 最终目录：`data/tokenized_clean_sft_repair_final_2048`
