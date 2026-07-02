#!/bin/bash
# 启动 RAG QA API 服务

set -e
source "$(dirname "$0")/env.sh"

export CONFIG_PATH="./configs/config.local.yaml"

echo "======================================"
echo "  RAG QA 系统 - 启动服务"
echo "  Python: $PROJECT_PYTHON"
echo "======================================"

# 检查 Redis 是否运行
if ! redis-cli ping &>/dev/null; then
    echo "⚠️  Redis 未运行，尝试启动..."
    redis-server --daemonize yes --maxmemory 256mb --maxmemory-policy allkeys-lru
    sleep 1
fi

echo "✅ Redis 已就绪"
echo "🚀 启动 FastAPI 服务（单 worker，避免重复加载模型）..."

"$PROJECT_PYTHON" -m uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    --access-log
