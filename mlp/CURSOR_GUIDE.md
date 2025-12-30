# Cursor IDE 使用指南 - NumPy MLP 实验

本指南说明如何在 Cursor IDE 中完成 NumPy 实现 MLP 的实验。

## 1. Cursor IDE 简介

Cursor 是基于 VS Code 的 AI 代码编辑器，支持智能代码补全、代码生成和调试功能。本实验完全可以在 Cursor 中完成。

## 2. 环境配置

### 2.1 打开项目

1. 启动 Cursor IDE
2. 点击 `File` → `Open Folder`，选择项目目录 `/Users/bytedance/Documents/mlp`
3. 项目结构将在左侧文件浏览器中显示

### 2.2 创建虚拟环境

在 Cursor 中打开集成终端：

1. 点击顶部菜单 `Terminal` → `New Terminal`，或使用快捷键：
   - **Mac**: `Ctrl + `` ` `` (反引号)
   - **Windows/Linux**: `` Ctrl + ` ``

2. 在终端中执行以下命令：

```bash
# 创建虚拟环境
python -m venv mlp-env

# Mac/Linux 激活
source mlp-env/bin/activate

# Windows 激活
mlp-env\Scripts\activate
```

3. 验证环境激活：终端提示符前应显示 `(mlp-env)`

### 2.3 安装依赖

在激活的虚拟环境中执行：

```bash
pip install -r requirements.txt
```

或者手动安装：

```bash
pip install numpy==1.24.3 pandas==1.5.3 scikit-learn==1.2.2 matplotlib==3.7.1
```

### 2.4 配置 Python 解释器

1. 按 `Cmd + Shift + P` (Mac) 或 `Ctrl + Shift + P` (Windows/Linux) 打开命令面板
2. 输入 `Python: Select Interpreter`
3. 选择虚拟环境中的 Python 解释器：
   - `./mlp-env/bin/python` (Mac/Linux)
   - `.\mlp-env\Scripts\python.exe` (Windows)

## 3. 项目结构说明

```
lab1-mlp/
├── mlp/                    # 核心模块目录
│   ├── __init__.py        # 模块接口
│   ├── activations.py     # 激活函数
│   ├── datasets.py        # 数据集加载
│   ├── layers.py          # 神经网络层
│   ├── losses.py          # 损失函数
│   ├── model.py           # MLP 模型
│   └── optimizers.py      # 优化器
├── demo.py                # 基础演示脚本
├── demo_advanced.py       # 高级演示脚本
├── requirements.txt       # 依赖列表
├── README.md              # 项目说明
└── results/               # 结果目录（自动生成）
```

## 4. 运行代码

### 4.1 运行基础演示

1. 打开 `demo.py` 文件
2. 点击右上角的运行按钮（▶️），或使用快捷键：
   - **Mac**: `F5` 或 `Cmd + F5`
   - **Windows/Linux**: `F5` 或 `Ctrl + F5`
3. 选择运行配置（选择 Python 解释器）
4. 查看输出：运行日志显示在底部 `TERMINAL` 标签页

### 4.2 运行高级演示

同样方式运行 `demo_advanced.py`

### 4.3 使用终端运行

在集成终端中：

```bash
# 确保虚拟环境已激活
source mlp-env/bin/activate  # Mac/Linux
# 或
mlp-env\Scripts\activate     # Windows

# 运行脚本
python demo.py
python demo_advanced.py
```

## 5. 查看结果

### 5.1 查看运行日志

运行脚本后，在底部终端查看：
- 数据集加载信息
- 训练过程日志（每 10 个 epoch 输出一次）
- 模型评估结果（MSE、MAE、R²）

### 5.2 查看可视化结果

1. 脚本运行完成后，会在 `results/` 目录生成图片：
   - `training_curve.png` - 训练曲线
   - `predictions.png` - 预测结果散点图

2. 在 Cursor 中查看图片：
   - 在左侧文件浏览器中展开 `results/` 目录
   - 双击图片文件即可在 Cursor 中预览

## 6. 调试代码

### 6.1 设置断点

1. 在代码行号左侧点击，设置断点（红色圆点）
2. 点击调试按钮（🐛）或按 `F5` 启动调试
3. 程序会在断点处暂停，可以：
   - 查看变量值（鼠标悬停或使用调试面板）
   - 单步执行（F10: 单步跳过, F11: 单步进入）
   - 继续执行（F5）

### 6.2 调试面板

调试时，左侧会显示：
- **Variables**: 当前作用域的变量
- **Watch**: 监视的表达式
- **Call Stack**: 调用栈
- **Breakpoints**: 断点列表

## 7. Cursor 实用功能

### 7.1 AI 代码补全

Cursor 支持 AI 代码补全：
- 输入代码时自动提示
- 按 `Tab` 接受建议
- 使用 `Cmd + K` (Mac) 或 `Ctrl + K` (Windows) 进行 AI 编辑

### 7.2 代码导航

- **跳转到定义**: `Cmd + 点击` (Mac) 或 `Ctrl + 点击` (Windows)
- **查找引用**: 右键点击符号 → `Find All References`
- **文件搜索**: `Cmd + P` (Mac) 或 `Ctrl + P` (Windows)

### 7.3 代码格式化

- **格式化文档**: `Shift + Option + F` (Mac) 或 `Shift + Alt + F` (Windows)
- 或右键 → `Format Document`

### 7.4 终端快捷操作

- **新建终端**: `Ctrl + Shift + `` ` ``
- **分割终端**: 点击终端右上角的 `+` 按钮
- **清屏**: 输入 `clear` (Mac/Linux) 或 `cls` (Windows)

## 8. 常见问题解决

### 8.1 虚拟环境激活失败

**问题**: 终端提示找不到 `activate` 脚本

**解决**:
- 检查虚拟环境是否创建成功：`ls mlp-env/bin/` (Mac/Linux) 或 `dir mlp-env\Scripts\` (Windows)
- 重新创建虚拟环境：删除 `mlp-env` 文件夹后重新执行创建命令

### 8.2 Python 解释器配置失败

**问题**: Cursor 无法识别虚拟环境中的 Python

**解决**:
1. 确保虚拟环境已创建
2. 使用命令面板 (`Cmd/Ctrl + Shift + P`) 选择正确的解释器
3. 重启 Cursor IDE

### 8.3 模块导入错误

**问题**: `ModuleNotFoundError: No module named 'mlp'`

**解决**:
- 确保在项目根目录运行脚本
- 检查 Python 解释器是否选择了虚拟环境
- 确保 `mlp/` 目录下有 `__init__.py` 文件

### 8.4 数据集加载失败

**问题**: 无法从网络下载数据集

**解决**:
- 检查网络连接
- 代码会自动尝试多个数据源（openml 和 UCI）
- 如果持续失败，可以手动下载数据文件并修改 `datasets.py` 中的加载路径

### 8.5 图片无法显示

**问题**: 运行后没有生成图片文件

**解决**:
- 检查 `results/` 目录是否已创建
- 查看终端错误日志
- 确保 matplotlib 已正确安装：`pip list | grep matplotlib`

## 9. 实验步骤（Cursor 版本）

### 步骤 1: 准备环境
- ✅ 在 Cursor 中打开项目
- ✅ 创建并激活虚拟环境
- ✅ 安装依赖包
- ✅ 配置 Python 解释器

### 步骤 2: 运行基础演示
- ✅ 运行 `demo.py`
- ✅ 查看训练日志
- ✅ 检查生成的可视化结果

### 步骤 3: 理解核心模块
- 在 Cursor 中打开各个模块文件，阅读代码
- 使用代码导航功能跳转到相关定义
- 设置断点调试，观察变量变化

### 步骤 4: 分析实验结果
- 查看 `results/training_curve.png` 分析训练过程
- 查看 `results/predictions.png` 分析预测准确性
- 在终端查看评估指标（MSE、MAE、R²）

### 步骤 5: 超参数调优
- 修改 `demo.py` 中的超参数
- 重新运行并对比结果
- 或运行 `demo_advanced.py` 进行网格搜索

## 10. 与 TRAE IDE 的对比

| 功能 | TRAE IDE | Cursor IDE |
|------|----------|------------|
| 代码编辑 | ✅ | ✅ |
| 终端支持 | ✅ | ✅ |
| 调试功能 | ✅ | ✅ |
| AI 代码补全 | ❌ | ✅ |
| 代码导航 | 基础 | 强大 |
| 扩展支持 | 有限 | 丰富 |
| 跨平台 | ✅ | ✅ |

**优势**: Cursor 提供更强大的 AI 辅助功能和更丰富的扩展生态。

## 11. 下一步

完成基础实验后，可以尝试：

1. **实现 L2 正则化**: 修改 `layers.py` 中的 `Dense` 层
2. **实现 Dropout**: 在 `layers.py` 中添加 `Dropout` 类
3. **超参数网格搜索**: 使用 `demo_advanced.py` 中的 `grid_search()` 函数
4. **尝试不同的网络结构**: 修改隐藏层数量和神经元数量
5. **尝试不同的优化器**: 对比 SGD 和 Adam 的效果

## 12. 获取帮助

- 查看 `README.md` 了解项目详情
- 查看代码注释了解实现细节
- 使用 Cursor 的 AI 功能询问代码相关问题
- 查看终端错误日志定位问题

祝实验顺利！🎉

