#!/bin/bash
set -e
# AutoDL 一键环境配置脚本
# 用法: bash scripts/setup_autodl.sh

echo "=== AutoDL SFT 环境配置 ==="

# 1. pip 换源加速（可选）
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ 2>/dev/null || true

# 2. 安装依赖
echo "安装 Python 依赖..."
pip install -r requirements.txt

# 3. 验证 GPU 和显存
echo ""
echo "=== GPU 信息 ==="
python3 -c "
import torch
print(f'CUDA 可用: {torch.cuda.is_available()}')
print(f'GPU 数量: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    prop = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {prop.name}, 显存 {prop.total_memory/1024**3:.1f}GB')
print(f'bf16 支持: {torch.cuda.is_bf16_supported()}')
"

# 4. 创建必要目录
mkdir -p data/tokenized output

echo ""
echo "=== 环境配置完成 ==="
echo ""
echo "接下来两步："
echo "  1. 生成数据:    python3 scripts/tokenize_dataset.py"
echo "  2. 启动训练:    bash scripts/fix_and_run.sh"
