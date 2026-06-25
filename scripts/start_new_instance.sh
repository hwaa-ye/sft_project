#!/bin/bash
set -e
# AutoDL 新容器一键启动脚本
# 用法: bash scripts/start_new_instance.sh

# ─── 把模型缓存放到持久化网盘，新容器不用重复下载 ───
PERSIST_DIR="${HOME}/autodl-fs"
CACHE_DIR="${PERSIST_DIR}/model_cache"

if [ -d "$PERSIST_DIR" ]; then
    mkdir -p "$CACHE_DIR"
    export MODELSCOPE_CACHE="$CACHE_DIR"
    # HuggingFace 缓存也指过去（有些依赖会用到）
    export HF_HOME="${PERSIST_DIR}/hf_cache"
    export HF_HUB_CACHE="${PERSIST_DIR}/hf_cache/hub"
    echo "模型缓存: $CACHE_DIR (持久化)"
else
    echo "⚠ autodl-fs 未挂载，缓存放在 /tmp（容器释放后会丢失）"
fi

# ─── 检查模型是否已下载 ───
MODEL_NAME="${SFT_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_PATH="${CACHE_DIR}/hub/modelscope/${MODEL_NAME}"
if [ -d "$MODEL_PATH" ]; then
    echo "模型已缓存: $MODEL_PATH"
else
    echo "首次运行，需要下载模型（约 14GB，只下这一次）..."
fi

# ─── 拉代码（如果还没拉） ───
# 建议把代码也放 autodl-fs 或用 git clone

# ─── 安装依赖 ───
pip install transformers peft modelscope numpy accelerate -q

# ─── tokenize（如果还没做） ───
if [ ! -f data/tokenized/input_ids.pkl ] && [ ! -f data/tokenized/input_ids.npy ]; then
    echo "生成 tokenize 数据..."
    python3 scripts/tokenize_dataset.py
fi

# ─── 启动训练 ───
echo "启动训练: $MODEL_NAME"
python3 scripts/train_h800.py
