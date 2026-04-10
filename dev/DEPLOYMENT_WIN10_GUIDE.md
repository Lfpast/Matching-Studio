# Win10 + GTX 1060 3GB 部署指南

## ⚠️ 关键问题分析

### 1. **显存不足的核心问题**
- **all-mpnet-base-v2 模型大小**: ~430MB
- **运行内存需求**: 至少需要 2-3GB 显存（batch processing）
- **GTX 1060 3GB 的实际可用**: ~2.5-2.8GB (扣除系统开销)
- **结论**: ❌ **非常紧张，容易 Out of Memory**

### 2. **requirements.txt 的版本问题**

| 库 | 问题 | 解决方案 |
|---|---|---|
| `torch>=2.0.0` | 对CUDA版本要求苛刻，GTX1060最多支持CUDA 11.8 | 需要精确指定版本 |
| `sentence-transformers>=2.2.0` | 会自动下载大型预训练模型 | 可接受，但需要大显存 |
| `numpy>=1.23.0` | Windows上可能有兼容性问题 | 建议降至 `numpy>=1.21.0, <1.24.0` |
| `openpyxl>=3.1.0` | 过新，可能导致依赖冲突 | 改为 `openpyxl>=3.0.0, <3.1.0` |

---

## 🔧 部署方案对比

### 方案 A: GPU加速（推荐但有风险）
**优点**: 速度快
**缺点**: 显存溢出风险高，需要严格控制batch_size

### 方案 B: CPU-only（强烈推荐）⭐
**优点**: 稳定可靠，不会显存溢出
**缺点**: 速度慢（10-20倍），但胜在稳定

### 方案 C: GPU+CPU混合（折中）
**优点**: 平衡速度和稳定性
**缺点**: 配置复杂

---

## 📋 具体部署步骤

### 第1步: Python 环境准备

#### 在 Win10 上安装 Python
1. **下载 Python 3.10** (不要用3.12+，依赖兼容性差)
   - 官网: https://www.python.org/downloads/
   - 推荐版本: **Python 3.10.13** (稳定版)

2. **安装时勾选**:
   - ✅ "Add Python to PATH"
   - ✅ "pip"
   - ✅ "py launcher"

3. **验证安装**:
   ```cmd
   python --version
   pip --version
   ```

---

### 第2步: 创建虚拟环境 (使用 venv)

```cmd
# 进入项目目录
cd D:\path\to\Professor_Matching_System

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# 在 Windows CMD:
venv\Scripts\activate.bat

# 或在 PowerShell:
venv\Scripts\Activate.ps1
```

验证激活成功（cmd前面会显示 `(venv)`）

---

### 第3步: 修改和安装依赖

#### 选项 A: **GPU方案** (如果你坚持用GPU)

**创建新的 `requirements_gpu.txt`**:
```txt
# PyTorch with CUDA 11.8 support (for GTX 1060)
torch==2.0.2
torchvision==0.15.2
torchaudio==2.0.2

numpy>=1.21.0,<1.24.0
pandas>=1.5.0
pydantic>=2.0.0
openpyxl>=3.0.0,<3.1.0
playwright>=1.41.0
pyyaml>=6.0
scikit-learn>=1.2.0
sentence-transformers==2.2.2
transformers>=4.30.0,<4.35.0
networkx>=3.0
uvicorn>=0.23.0

# Optional: memory management
# pympler>=1.1  # for memory profiling
```

安装步骤：
```cmd
# 激活虚拟环境后
pip install torch==2.0.2 torchvision==0.15.2 torchaudio==0.2.2 -i https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements_gpu.txt -i https://mirrors.aliyun.com/pypi/simple
```

#### 选项 B: **CPU-only方案** (强烈推荐) ⭐

**创建新的 `requirements_cpu.txt`**:
```txt
# PyTorch CPU-only version
torch==2.0.2 --index-url https://download.pytorch.org/whl/cpu
torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

numpy>=1.21.0,<1.24.0
pandas>=1.5.0
pydantic>=2.0.0
openpyxl>=3.0.0,<3.1.0
playwright>=1.41.0
pyyaml>=6.0
scikit-learn>=1.2.0
sentence-transformers==2.2.2
transformers>=4.30.0,<4.35.0
networkx>=3.0
uvicorn>=0.23.0
```

安装步骤：
```cmd
# 激活虚拟环境后
pip install torch==2.0.2 torchvision==0.15.2 torchaudio==2.0.2 -i https://download.pytorch.org/whl/cpu

pip install -r requirements_cpu.txt -i https://mirrors.aliyun.com/pypi/simple
```

---

### 第4步: 优化配置文件

编辑 `config/config.yaml`:

#### GPU 方案优化:
```yaml
embedding:
  model_name: sentence-transformers/all-mpnet-base-v2
  batch_size: 4          # ⬇️ 从32降低到4，减少显存占用
  device: cuda            # 使用GPU
  
  attribute_weights:
    deeptech_projects: 5.0
    paper: 2.5
    research_interests: 1.5
    leading_project: 1.0
    department: 0.2
    title: 0.1
    other: 0.1
```

#### CPU-only 方案优化:
```yaml
embedding:
  model_name: sentence-transformers/all-mpnet-base-v2
  batch_size: 2          # ⬇️ 更小的batch_size
  device: cpu             # 使用CPU
  num_workers: 2          # CPU线程数
  
  attribute_weights:
    deeptech_projects: 5.0
    paper: 2.5
    research_interests: 1.5
    leading_project: 1.0
    department: 0.2
    title: 0.1
    other: 0.1
```

---

### 第5步: 创建启动脚本 (可选但推荐)

**创建 `run_gpu.bat` (GPU方案)**:
```batch
@echo off
REM 启动GPU模式的API服务
cd /d %~dp0
call venv\Scripts\activate.bat
python api/app.py --device cuda
pause
```

**创建 `run_cpu.bat` (CPU方案)**:
```batch
@echo off
REM 启动CPU模式的API服务
cd /d %~dp0
call venv\Scripts\activate.bat
python api/app.py --device cpu
pause
```

---

## ⚡ 运行时注意事项

### GPU 方案出现 Out of Memory 时的解决步骤:

1. **降低 batch_size** (在 config.yaml 中):
   ```yaml
   batch_size: 2    # 继续降低
   ```

2. **清除缓存**:
   ```python
   import torch
   torch.cuda.empty_cache()  # 在Python代码中添加
   ```

3. **使用内存优化**:
   ```python
   # 在 embedding_model.py 中改进
   import torch
   model.to("cuda:0")
   torch.cuda.set_per_process_memory_fraction(0.8)  # 限制显存使用到80%
   ```

### CPU 方案的性能期望:

| 操作 | GPU (3GB) | CPU | 备注 |
|---|---|---|---|
| 数据加载 | ~2s | ~5s | 相差不大 |
| 嵌入生成 | ~5-10s | ~60-120s | CPU慢10-15倍 |
| 图构建 | ~3s | ~10s | 取决于数据量 |
| 单次查询 | ~1s | ~2-3s | 可接受 |

---

## 🐛 常见问题排查

### 问题1: ImportError: No module named 'torch'
```cmd
# 检查虚拟环境是否激活
venv\Scripts\activate.bat

# 检查安装
pip list | find "torch"

# 重新安装
pip install --upgrade torch==2.0.2
```

### 问题2: CUDA out of memory
```
❌ 这是正常的
✅ 解决方案:
  1. 改用CPU方案
  2. 或继续降低 batch_size (1)
  3. 或分批处理数据
```

### 问题3: 模型下载太慢/失败
```cmd
# 手动下载并缓存
# 在代码中指定本地路径
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2', 
                           cache_folder='./models')
```

### 问题4: Windows Defender 阻止
- 添加虚拟环境文件夹到 Windows Defender 排除列表
- 或在运行前临时禁用实时保护

---

## 📊 最终推荐方案总结

| 方案 | 推荐度 | 理由 |
|---|---|---|
| **CPU-only** | ⭐⭐⭐⭐⭐ | 稳定可靠，零显存问题，GTX1060形同虚设 |
| **GPU with batch_size=4** | ⭐⭐⭐ | 速度快但不稳定，需持续监控 |
| **GPU with batch_size=2** | ⭐⭐ | 可能工作，但性能和稳定性都不理想 |

---

## 📝 部署清单

```
□ 安装 Python 3.10.13
□ 创建虚拟环境 (venv)
□ 复制 requirements_cpu.txt (推荐) 或 requirements_gpu.txt
□ 安装依赖包
□ 修改 config/config.yaml 中的 batch_size 和 device
□ 测试导入 (python -c "import torch; print(torch.__version__)")
□ 运行单元测试
□ 启动 API 服务
□ 访问 http://localhost:8000/docs
```

---

## 🔗 有用的链接

- PyTorch 下载: https://pytorch.org/get-started/locally/
- CUDA 11.8 工具包: https://developer.nvidia.com/cuda-11-8-0-download-archive
- cuDNN (可选): https://developer.nvidia.com/cudnn

---

## 💡 最后建议

**强烈建议:**
1. **第一次部署选择 CPU-only 方案** - 保证稳定性
2. 一切正常后再尝试GPU优化
3. 保存好这个 requirements_cpu.txt，以便后续重装环境
4. 定期清理 pip 缓存: `pip cache purge`
