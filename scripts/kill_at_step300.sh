#!/bin/bash
# 监控 GRPO 训练日志，到 step 300 时优雅终止
LOG_FILE="/root/autodl-tmp/sft_project/output/train_grpo.log"
PID=$(pgrep -f "train_grpo.py" | head -1)

if [ -z "$PID" ]; then
    echo "未找到 train_grpo.py 进程"
    exit 1
fi

echo "监控 PID $PID，等待 step 300..."

# tail -f 逐行读取，匹配到 "step 300/" 时杀进程
# step 300 % 100 == 0，checkpoint 会在打印前保存，所以看到日志即可杀
tail -f "$LOG_FILE" 2>/dev/null | while read line; do
    echo "$line" | grep -q "step.*300/" && {
        echo "检测到 step 300，保存 checkpoint 并终止..."
        kill -TERM "$PID" 2>/dev/null
        sleep 2
        # 如果 TERM 没杀掉，用 INT
        kill -INT "$PID" 2>/dev/null
        exit 0
    }
done
