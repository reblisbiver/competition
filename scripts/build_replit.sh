#!/bin/bash
set -e  # 遇到错误立即退出，保证构建可靠性

# 1. 设置构建阶段环境变量（用户要求）
export REPLIT_BUILD_PHASE=1
echo "=== 开始构建阶段，设置 REPLIT_BUILD_PHASE=$REPLIT_BUILD_PHASE ==="

# 2. 安装 Python 依赖（Django/Gunicorn 核心依赖）
echo "=== 安装项目依赖 ==="
pip install --upgrade pip
pip install -r requirements.txt || pip install django gunicorn  # 兜底：无requirements.txt时直接装核心包

# 3. 配置 LibreOffice + 字体（Replit 中通过 Nix 已安装，此处做软链接/验证）
echo "=== 配置 LibreOffice 环境 ==="
if command -v libreoffice &>/dev/null; then
    export LIBREOFFICE_PATH=$(which libreoffice)
    echo "LibreOffice 已配置：$LIBREOFFICE_PATH"
else
    echo "警告：LibreOffice 未安装，若需使用请在 .replit 的 nix.packages 中添加 libreoffice"
fi

# 4. 数据库迁移（Django 项目核心步骤）
echo "=== 执行 Django 数据库迁移 ==="
python manage.py migrate --noinput || echo "数据库迁移暂未执行（可能未初始化项目）"

# 5. 构建完成提示
echo "=== 构建阶段完成 ==="