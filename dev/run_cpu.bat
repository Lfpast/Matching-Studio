@echo off
REM 快速启动脚本 - CPU 版本
REM 用法: 双击运行或 cmd 中执行

title Professor Matching System - CPU Mode
cd /d %~dp0

echo 激活虚拟环境...
call venv\Scripts\activate.bat

if errorlevel 1 (
    echo 错误: 虚拟环境不存在，请先运行 setup_win10_venv.bat
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Professor Matching System - API Server
echo  Mode: CPU
echo ============================================
echo.

REM 检查config文件
if not exist config\config.yaml (
    echo 错误: 找不到 config\config.yaml
    pause
    exit /b 1
)

echo 配置提示:
echo - batch_size: 应设置为 2 (CPU模式)
echo - device: 应设置为 cpu
echo.

REM 启动API服务
echo 正在启动 FastAPI 服务...
echo 访问地址: http://localhost:8000
echo 文档地址: http://localhost:8000/docs
echo 按 Ctrl+C 停止服务
echo.

python api/app.py --config config/config.yaml

pause
