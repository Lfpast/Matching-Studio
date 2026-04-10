# Win10 GTX 1060 3GB 快速参考卡片

## 🚀 快速开始 (5分钟)

### 第一次设置
```cmd
# 1. 下载 Python 3.10.13 并安装 https://www.python.org/downloads/
# 2. 打开项目文件夹，右键选 "在此处打开 PowerShell 窗口"
# 3. 运行
setup_win10_venv.bat
# 4. 按提示选择 1 (CPU版本，推荐) 或 2 (GPU版本)
# 5. 等待安装完成（5-15分钟）
```

### 后续启动
```cmd
# CPU版本 (推荐)
run_cpu.bat

# 或 GPU版本 (如果显存够)
run_gpu.bat
```

## ⚙️ 配置调整 (config/config.yaml)

### CPU版本
```yaml
embedding:
  batch_size: 2          # ⚠️ 小批量，避免OOM
  device: cpu            # 使用CPU
```

### GPU版本 (如出现OOM，降低batch_size)
```yaml
embedding:
  batch_size: 4          # 显存溢出时改为 2
  device: cuda           # 使用GPU
```

## 🔍 诊断问题

| 问题 | 解决方案 |
|------|---------|
| `ModuleNotFoundError: No module named 'torch'` | 虚拟环境未激活，运行 `venv\Scripts\activate.bat` |
| `CUDA out of memory` | 这是显存不足，改用CPU版本或降低batch_size |
| 安装很慢 | 正常（3GB文件），或使用代理加速 |
| 模型下载失败 | PyTorch和模型会自动下载，首次可能 5-10分钟 |

## 📊 性能期望

| 任务 | CPU版 | GPU版 |
|------|-------|-------|
| 启动时间 | ~30s | ~20s |
| 单次查询 | 2-3s | 1s |
| 大数据处理 | 慢 | 快 |
| 稳定性 | ✅ 优秀 | ⚠️ 可能OOM |

## 💾 虚拟环境管理

```cmd
# 激活虚拟环境
venv\Scripts\activate.bat

# 查看已安装包
pip list

# 升级某个包
pip install --upgrade torch

# 删除虚拟环境（释放空间）
rmdir venv /s /q

# 重建虚拟环境
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements_cpu.txt
```

## 🌐 访问应用

启动脚本后，访问:
- **Web界面**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## ⚡ 性能优化建议

如果CPU版本太慢，可以尝试:

1. **增加CPU线程** (config/config.yaml):
```yaml
embedding:
  num_workers: 4  # 增加处理线程
```

2. **批量处理** - 修改Python代码用大的batch_size

3. **使用较小的模型**:
```yaml
embedding:
  model_name: sentence-transformers/all-MiniLM-L6-v2  # 更小更快
```

## 📚 相关文件

- `requirements_cpu.txt` - CPU版本依赖
- `requirements_gpu.txt` - GPU版本依赖
- `DEPLOYMENT_WIN10_GUIDE.md` - 详细部署指南
- `setup_win10_venv.bat` - 自动安装脚本

## 🆘 寻求帮助

问题排查顺序:
1. 检查 Python 版本: `python --version` (应该是 3.10)
2. 检查虚拟环境: `where python` (应指向 venv 文件夹)
3. 检查PyTorch: `python -c "import torch; print(torch.__version__)"`
4. 查看详细指南: README_DEPLOYMENT_WIN10_GUIDE.md
