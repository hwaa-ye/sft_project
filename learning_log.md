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
- [ ] 验收 SFT 训练结果（loss 曲线、checkpoint）
- [ ] 切测试集 + 模型推理 + 跑 eval_math.py 评估报告
- [ ] Badcase 分析（为 Phase 2 GRPO reward 设计做准备）
- [ ] 面试题：Self-Attention 推导
- [ ] 健身

