@echo off
REM 快速启动脚本 - GPU 版本
REM 用法: 双击运行或 cmd 中执行
REM 注意: GTX 1060 3GB 显存可能不足，易出现 Out of Memory

title Professor Matching System - GPU Mode
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
echo  Mode: GPU (CUDA 11.8)
echo  WARNING: GTX 1060 3GB 显存受限
echo ============================================
echo.

REM 检查config文件
if not exist config\config.yaml (
    echo 错误: 找不到 config\config.yaml
    pause
    exit /b 1
)

echo 配置检查:
echo Batch Size 应该设置为 2-4 (推荐 2)
echo Device 应该设置为 cuda
echo.

REM 启动API服务
echo 正在启动 FastAPI 服务 (GPU模式)...
echo 访问地址: http://localhost:8000
echo 文档地址: http://localhost:8000/docs
echo 按 Ctrl+C 停止服务
echo.
echo 如出现 Out of Memory 错误，改用 run_cpu.bat
echo.

python api/app.py --config config/config.yaml

pause
