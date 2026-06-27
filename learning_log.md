# 学习日志

> 记录每日计划、收获、问题。由 Claude 根据对话整理。

---

## 2026-06-26（周四）

### 今日计划
- [x] 保研 PPT
- [x] 投递邮件
- [x] 项目：搭建评估环境、下载数据、启动 SFT 训练
- [ ] 面试题 1 道
- [ ] 健身

### 收获
- 完整跑通了 SFT 数据管线：ModelScope 拉取 MathR + R1 Distill（40,599条）→ 清洗 → ChatML 格式化 → tokenize（labels masking）
- 理解了公钥/私钥机制：非对称加密原理，Ed25519 vs RSA，SSH 认证流程，Git 协议选择
- 掌握了 AutoDL 离线环境模型加载的三道锁：`TRANSFORMERS_OFFLINE=1`（shell层）+ `local_files_only=True` + 绝对路径
- 理解了 from_pretrained 的真实行为：不是"读本地文件"，而是"检查云端更新后再决定"，离线环境必须显式禁用
- 学会了 AutoDL 三层存储架构：系统盘(30G overlay) / 数据盘(50G) / 网盘(持久化)，默认缓存往系统盘写是大坑
- 评估脚本 eval_math.py 完成：7维评估 + 5类错误分类（启发式规则版）
- Qwen3-8B LoRA SFT 训练已启动：15.3GB/31.4GB 显存，15,224步，后台跑中

### 遇到的问题
- HuggingFace/ModeScope 数据API 双双不可用 → 改用 HTTP 直接下载 ModelScope JSONL
- modelscope 1.37 与 datasets 5.0 API 不兼容 → 绕过 MsDataset，用 requests 直取
- 模型反复重下，系统盘爆满 → 4次迭代最终用 transformers+离线+本地路径解决
- 离线环境变量在 Python 层设太晚 → 改到 shell 脚本 nohup 前设置
- 旧进程 pkill 导致 SSH 会话退出 → 改用更安全的进程管理方式

### 明天要做
- [x] 验收 SFT 训练结果（loss 曲线、checkpoint）
- [x] 切测试集 + 模型推理 + 跑 eval_math.py 评估报告
- [x] Badcase 分析（为 Phase 2 GRPO reward 设计做准备）
- [ ] 面试题：Self-Attention 推导
- [ ] 健身

---

## 2026-06-27（周五）

### 今日计划
- [x] SFT 推理 + eval 基线评估
- [x] Badcase 分类分析（截断/计算错误/语义误解）
- [x] 修 eval 归一化假阴性（数值、LaTeX、中文符号）
- [x] 推理加速：batch inference (batch_size=16, ~4x 加速)
- [x] 尝试 vLLM 部署（系统盘满未成功，不阻塞主线）
- [x] Merge LoRA 产出完整模型权重
- [ ] 开始 GRPO 阶段

### 收获
- **推理速度的核心瓶颈**：自回归生成 = 逐 token 串行 forward（1024 token = 1024 次 forward），而训练是 teacher forcing 并行一次过。所以推理 forward 次数远超训练
- **KV Cache 原理**：旧 token 的 K/V 投影不变，存起来避免重算。vLLM 用 PagedAttention 把 KV cache 像 OS 分页管理（16 token/页），解决显存碎片，吞吐提 10-20x
- **Continuous Batching**：不等最慢的请求，谁先完谁先出，GPU 不空转
- **vLLM vs SGLang**：vLLM 强项通用 serving，SGLang 强项结构化生成+fork/join 多步编排。纯数学 GRPO 选 vLLM
- **Temperature 对推理一致性的影响**：T=0 时 argmax 不受浮点误差影响，vLLM/HF 输出完全一致；T>0 时采样会放大微小数值差异导致序列分叉
- **SFT 模型的真实水平**：完整输出中准确率 92.7%，截断 40%→20%（修 max_tokens 1024→2048），总体 74.38%
- **Badcase 分类方法**：不是看统计数字，是一条条读。发现大量"假阴性"（eval 太严格），真正推理错误很少
- **截断的本质不是 token 不够**：是模型不会控制推理长度，SFT 只教"正确"不教"简洁"，需要 GRPO 加效率 reward
- **增大 batch size 是最简单的推理加速**：HF generate 支持 batch 推理，16.3GB 显存 batch=16 只用一半，提速 4x
- **Merge LoRA = 一次性操作**：`merge_and_unload()` 把 adapter 权重融进 base model，产出标准权重文件，vLLM 可直接加载

### 遇到的问题
- HF generate batch=1 速度 25s/条，484 条要 3h → batch_size=16 压缩到 ~40min
- max_new_tokens=1024 导致 40% 输出截断 → 提到 2048 降到 20%
- vLLM 安装因系统盘满失败 → 暂不阻塞，用 batch HF 替代，GRPO 阶段再搞
- eval 归一化过于严格（21 vs 21.00 判为不同）→ 加数值比较和 LaTeX 清洗
- 第一次推理结果因缓冲全部丢失 → 改增量写入+flush，后两次推理都正常

### 明天要做
- [ ] 开始 GRPO 阶段：理解 nanoRL 代码结构
- [ ] 设计 GRPO reward function（准确性 + 完整性 + 效率）
- [ ] 准备 GRPO 八股：policy gradient、advantage、KL 散度、PPO vs GRPO
- [ ] 面试题：Self-Attention 推导

