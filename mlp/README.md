# NumPy 实现 MLP 神经网络

基于纯 NumPy 实现的多层感知机（MLP）神经网络，用于解决 Boston Housing 房价回归问题。

## 项目简介

本项目完全使用 NumPy 实现神经网络的所有核心组件，不依赖任何深度学习框架（如 TensorFlow、PyTorch），旨在深入理解神经网络的前向传播、反向传播和梯度下降等核心机制。

## 技术栈

- **Python**: 3.8+
- **NumPy**: 1.21+（数组运算核心）
- **Pandas**: 1.3+（数据读取与预处理辅助）
- **Scikit-learn**: 0.24~1.3（数据集加载、分割、标准化、指标计算）
- **Matplotlib**: 3.4+（训练曲线、预测结果可视化）

## 项目结构

```
lab1-mlp/
├── mlp/                    # 核心模块
│   ├── __init__.py        # 模块暴露接口
│   ├── activations.py     # 激活函数（ReLU/Tanh/Sigmoid/Linear）
│   ├── datasets.py        # 数据集加载（Boston 替代方案+预处理）
│   ├── layers.py          # 神经网络层（全连接层 Dense）
│   ├── losses.py          # 损失函数（MSE）
│   ├── model.py           # MLP 模型核心（训练/预测/评估）
│   └── optimizers.py      # 优化器（SGD/Adam）
├── demo.py                # 基础演示脚本（快速运行）
├── demo_advanced.py       # 高级演示（超参数调优+正则化）
├── requirements.txt       # 依赖列表
├── README.md              # 项目说明
└── results/               # 结果保存目录（自动生成，存储可视化图）
```

## 安装说明

### 1. 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv mlp-env

# Windows 激活
mlp-env\Scripts\activate

# Mac/Linux 激活
source mlp-env/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

或者手动安装：

```bash
pip install numpy==1.24.3 pandas==1.5.3 scikit-learn==1.2.2 matplotlib==3.7.1
```

## 快速开始

### 运行基础演示

```bash
python demo.py
```

这将执行以下步骤：
1. 加载并预处理 Boston Housing 数据集
2. 构建 MLP 模型（13→64(ReLU)→32(ReLU)→1(Linear)）
3. 训练模型（100 轮，批次大小 32）
4. 评估模型性能（MSE、MAE、R²）
5. 可视化训练曲线和预测结果
6. 进行示例预测

### 运行高级演示

```bash
python demo_advanced.py
```

展示不同的模型构建方式和超参数调优示例。

## 模型构建与训练指南

### 基本使用

```python
from mlp.model import MLP
from mlp.layers import Dense
from mlp.activations import ReLU, Linear
from mlp.losses import MSE
from mlp.optimizers import Adam
from mlp.datasets import BostonHousingLoader

# 1. 加载数据
data_loader = BostonHousingLoader(test_size=0.2, random_state=42)
X_train, y_train, X_test, y_test = data_loader.load_data()

# 2. 构建模型
model = MLP()
model.add_layer(Dense(input_dim=13, output_dim=64, weight_initializer='he'))
model.add_layer(ReLU())
model.add_layer(Dense(input_dim=64, output_dim=32, weight_initializer='he'))
model.add_layer(ReLU())
model.add_layer(Dense(input_dim=32, output_dim=1, weight_initializer='he'))
model.add_layer(Linear())

# 3. 设置损失函数和优化器
model.set_loss(MSE())
model.set_optimizer(Adam(learning_rate=0.001))

# 4. 训练模型
history = model.train(
    X_train=X_train,
    y_train=y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# 5. 评估模型
test_loss, metrics = model.evaluate(X_test, y_test)
print(f"MSE: {test_loss:.4f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}")

# 6. 预测
y_pred = model.predict(X_test)

# 7. 可视化
model.plot_history(save_path='results/training_curve.png')
model.plot_predictions(X_test, y_test, save_path='results/predictions.png')
```

## API 参考

### MLP 类

- `add_layer(layer)`: 添加层到模型
- `set_loss(loss)`: 设置损失函数
- `set_optimizer(optimizer)`: 设置优化器
- `train(X_train, y_train, epochs, batch_size, validation_data)`: 训练模型
- `evaluate(X_test, y_test)`: 评估模型性能
- `predict(X)`: 预测新数据
- `plot_history(save_path)`: 绘制训练曲线
- `plot_predictions(X_test, y_test, save_path)`: 绘制预测结果散点图

### 层（Layers）

- `Dense(input_dim, output_dim, weight_initializer='he')`: 全连接层
  - `weight_initializer`: 'he'（适合 ReLU）或 'xavier'（适合 Tanh/Sigmoid）

### 激活函数（Activations）

- `ReLU()`: ReLU 激活函数（隐藏层首选）
- `Tanh()`: Tanh 激活函数
- `Sigmoid()`: Sigmoid 激活函数（二分类输出层）
- `Linear()`: 线性激活函数（回归输出层）

### 损失函数（Losses）

- `MSE()`: 均方误差损失（回归任务）

### 优化器（Optimizers）

- `SGD(learning_rate=0.01, momentum=0.0)`: 随机梯度下降
- `Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)`: Adam 优化器

### 数据集加载器

- `BostonHousingLoader(test_size=0.2, random_state=42)`: Boston Housing 数据集加载器
  - `load_data()`: 加载并预处理数据，返回 X_train, y_train, X_test, y_test

## 常见问题

### 1. 数据集加载失败

如果出现数据集加载失败，请检查网络连接。代码会自动尝试从 openml 或 UCI 下载数据。

### 2. 维度不匹配错误

确保输入数据的形状正确：
- 特征矩阵：`(n_samples, n_features)`
- 标签向量：`(n_samples,)` 或 `(n_samples, 1)`

### 3. 训练损失不下降

- 检查学习率是否合适（建议范围：0.0001~0.01）
- 尝试不同的权重初始化方法（He 初始化适合 ReLU）
- 增加训练轮数或调整批次大小

### 4. 过拟合问题

- 减小模型复杂度（减少隐藏层神经元数）
- 添加正则化（L2 正则化或 Dropout）
- 使用早停策略

## 实验目标

- 深入理解神经网络核心机制：前向传播、反向传播、梯度下降
- 掌握纯 NumPy 构建多层感知机的完整流程
- 学会数据预处理、模型训练、性能评估的标准化方法
- 理解超参数对模型性能的影响机制
- 具备超参数调优、模型改进的实践能力

## 性能指标

- 测试集 MSE 损失应低于 15
- R² 评分通常能达到 0.8 以上
- 平均绝对误差（MAE）通常在 2-3 千美元范围内

## 扩展实验

### L2 正则化

在 `Dense` 层的 `backward` 方法中添加权重衰减，详见实验文档。

### Dropout 层

实现 Dropout 层用于正则化，详见实验文档。

### 网格搜索超参数

使用 `demo_advanced.py` 中的 `grid_search()` 函数进行超参数调优。

## 许可证

本项目仅用于学习和教育目的。

## 贡献

欢迎提交 Issue 和 Pull Request！

