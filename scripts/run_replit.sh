#!/bin/bash
set -e  # 遇到错误立即退出

# 快速启动 Gunicorn（用户指定参数）
echo "=== 启动 Gunicorn 服务 ==="
exec gunicorn config.wsgi:application \
    --bind 0.0.0.0:8000 \
    --workers 2 \
    --worker-class sync \
    --timeout 60 \
    --graceful-timeout 10 \
    --access-logfile - \
    --error-logfile -