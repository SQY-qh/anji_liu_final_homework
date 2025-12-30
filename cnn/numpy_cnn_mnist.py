"""
NumPy 实现 CNN 实验 - MNIST 手写数字分类
使用纯 NumPy 实现卷积神经网络，不使用任何深度学习框架
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import gzip
import urllib.request
import pickle
from tqdm import tqdm

# 设置中文字体（用于matplotlib显示中文）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 激活函数 ====================

def relu(x):
    """ReLU 激活函数"""
    return np.maximum(0, x)

def relu_backward(x, dout):
    """ReLU 反向传播"""
    return dout * (x > 0)

def softmax(x):
    """Softmax 激活函数（数值稳定实现）"""
    # 减去最大值避免数值溢出
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-8)

# ==================== 卷积层 ====================

class Conv2D:
    """2D 卷积层"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        初始化卷积层
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数（卷积核数量）
            kernel_size: 卷积核大小（整数或元组）
            stride: 步幅
            padding: 填充大小
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        
        # 初始化权重：使用Xavier初始化
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        fan_out = out_channels * self.kernel_size[0] * self.kernel_size[1]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        self.kernel = np.random.uniform(-limit, limit, 
                                       (self.kernel_size[0], self.kernel_size[1], 
                                        in_channels, out_channels))
        self.bias = np.zeros(out_channels)
        
        # 用于反向传播的缓存
        self.x = None
        self.d_kernel = None
        self.d_bias = None
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (batch_size, height, width, in_channels)
        
        返回:
            输出张量，形状为 (batch_size, out_height, out_width, out_channels)
        """
        self.x = x
        batch_size, h_in, w_in, c_in = x.shape
        
        # 添加填充
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (self.padding, self.padding), 
                                  (self.padding, self.padding), (0, 0)), 
                             mode='constant', constant_values=0)
        else:
            x_padded = x
        
        # 计算输出维度
        h_out = (h_in + 2 * self.padding - self.kernel_size[0]) // self.stride + 1
        w_out = (w_in + 2 * self.padding - self.kernel_size[1]) // self.stride + 1
        
        # 初始化输出
        out = np.zeros((batch_size, h_out, w_out, self.out_channels))
        
        # 执行卷积操作
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride
                w_end = w_start + self.kernel_size[1]
                
                # 提取局部区域
                x_local = x_padded[:, h_start:h_end, w_start:w_end, :]
                
                # 卷积运算：对每个输出通道
                for c_out in range(self.out_channels):
                    out[:, i, j, c_out] = np.sum(
                        x_local * self.kernel[:, :, :, c_out], 
                        axis=(1, 2, 3)
                    ) + self.bias[c_out]
        
        return out
    
    def backward(self, dout):
        """
        反向传播
        
        参数:
            dout: 上游梯度，形状为 (batch_size, h_out, w_out, out_channels)
        
        返回:
            dx: 输入梯度，形状为 (batch_size, h_in, w_in, in_channels)
        """
        batch_size, h_out, w_out, c_out = dout.shape
        _, h_in, w_in, c_in = self.x.shape
        
        # 初始化梯度
        dx = np.zeros_like(self.x)
        self.d_kernel = np.zeros_like(self.kernel)
        self.d_bias = np.zeros(self.out_channels)
        
        # 添加填充（用于计算输入梯度）
        if self.padding > 0:
            x_padded = np.pad(self.x, ((0, 0), (self.padding, self.padding), 
                                       (self.padding, self.padding), (0, 0)), 
                             mode='constant', constant_values=0)
            dx_padded = np.zeros_like(x_padded)
        else:
            x_padded = self.x
            dx_padded = dx
        
        # 计算梯度
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride
                w_end = w_start + self.kernel_size[1]
                
                x_local = x_padded[:, h_start:h_end, w_start:w_end, :]
                
                # 对每个输出通道
                for c_out_idx in range(c_out):
                    # 计算权重梯度
                    self.d_kernel[:, :, :, c_out_idx] += np.sum(
                        x_local * dout[:, i:i+1, j:j+1, c_out_idx:c_out_idx+1], 
                        axis=0
                    )
                    
                    # 计算输入梯度
                    dx_padded[:, h_start:h_end, w_start:w_end, :] += \
                        np.sum(self.kernel[:, :, :, c_out_idx] * 
                              dout[:, i:i+1, j:j+1, c_out_idx:c_out_idx+1], 
                              axis=3, keepdims=True)
                
                # 计算偏置梯度
                self.d_bias += np.sum(dout[:, i, j, :], axis=0)
        
        # 移除填充（如果有）
        if self.padding > 0:
            dx = dx_padded[:, self.padding:-self.padding, 
                          self.padding:-self.padding, :]
        else:
            dx = dx_padded
        
        return dx

# ==================== 池化层 ====================

class MaxPool2D:
    """2D 最大池化层"""
    
    def __init__(self, pool_size=2, stride=2):
        """
        初始化池化层
        
        参数:
            pool_size: 池化窗口大小
            stride: 步幅
        """
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride
        self.x = None
        self.max_indices = None  # 记录最大值位置，用于反向传播
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (batch_size, height, width, channels)
        
        返回:
            输出张量，形状为 (batch_size, out_height, out_width, channels)
        """
        self.x = x
        batch_size, h_in, w_in, c_in = x.shape
        
        # 计算输出维度
        h_out = (h_in - self.pool_size[0]) // self.stride + 1
        w_out = (w_in - self.pool_size[1]) // self.stride + 1
        
        # 初始化输出和最大值位置记录
        out = np.zeros((batch_size, h_out, w_out, c_in))
        self.max_indices = np.zeros((batch_size, h_out, w_out, c_in, 2), dtype=np.int32)
        
        # 执行最大池化
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.pool_size[0]
                w_start = j * self.stride
                w_end = w_start + self.pool_size[1]
                
                x_local = x[:, h_start:h_end, w_start:w_end, :]
                
                # 找到每个通道的最大值及其位置
                for c in range(c_in):
                    x_channel = x_local[:, :, :, c]
                    max_vals = np.max(x_channel.reshape(batch_size, -1), axis=1)
                    out[:, i, j, c] = max_vals
                    
                    # 记录最大值位置（用于反向传播）
                    for b in range(batch_size):
                        flat_idx = np.argmax(x_channel[b].flatten())
                        h_idx = flat_idx // self.pool_size[1]
                        w_idx = flat_idx % self.pool_size[1]
                        self.max_indices[b, i, j, c] = [h_start + h_idx, w_start + w_idx]
        
        return out
    
    def backward(self, dout):
        """
        反向传播
        
        参数:
            dout: 上游梯度，形状为 (batch_size, h_out, w_out, channels)
        
        返回:
            dx: 输入梯度，形状为 (batch_size, h_in, w_in, channels)
        """
        batch_size, h_out, w_out, c_in = dout.shape
        _, h_in, w_in, _ = self.x.shape
        
        # 初始化输入梯度（只有最大值位置有梯度）
        dx = np.zeros_like(self.x)
        
        # 将梯度传递到最大值位置
        for i in range(h_out):
            for j in range(w_out):
                for c in range(c_in):
                    for b in range(batch_size):
                        h_idx, w_idx = self.max_indices[b, i, j, c]
                        dx[b, h_idx, w_idx, c] += dout[b, i, j, c]
        
        return dx

# ==================== 全连接层 ====================

class Dense:
    """全连接层"""
    
    def __init__(self, in_features, out_features):
        """
        初始化全连接层
        
        参数:
            in_features: 输入特征数
            out_features: 输出特征数
        """
        self.in_features = in_features
        self.out_features = out_features
        
        # 初始化权重：使用Xavier初始化
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.w = np.random.uniform(-limit, limit, (in_features, out_features))
        self.b = np.zeros(out_features)
        
        # 用于反向传播的缓存
        self.x = None
        self.d_w = None
        self.d_b = None
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (batch_size, in_features)
        
        返回:
            输出张量，形状为 (batch_size, out_features)
        """
        self.x = x
        return x @ self.w + self.b
    
    def backward(self, dout):
        """
        反向传播
        
        参数:
            dout: 上游梯度，形状为 (batch_size, out_features)
        
        返回:
            dx: 输入梯度，形状为 (batch_size, in_features)
        """
        # 计算权重和偏置的梯度
        self.d_w = self.x.T @ dout
        self.d_b = np.sum(dout, axis=0)
        
        # 计算输入梯度
        dx = dout @ self.w.T
        
        return dx

# ==================== 损失函数 ====================

class CrossEntropyLoss:
    """交叉熵损失函数"""
    
    def __init__(self):
        self.y_pred = None
        self.y_true = None
    
    def forward(self, y_pred, y_true):
        """
        前向传播
        
        参数:
            y_pred: 预测概率，形状为 (batch_size, num_classes)
            y_true: 真实标签（独热编码），形状为 (batch_size, num_classes)
        
        返回:
            损失值（标量）
        """
        self.y_pred = y_pred
        self.y_true = y_true
        
        # 数值稳定性：避免log(0)
        y_pred_clipped = np.clip(y_pred, 1e-8, 1.0 - 1e-8)
        loss = -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
        
        return loss
    
    def backward(self):
        """
        反向传播
        
        返回:
            梯度，形状为 (batch_size, num_classes)
        """
        # 交叉熵损失对softmax输出的梯度是 (y_pred - y_true) / batch_size
        # 但通常我们直接返回 (y_pred - y_true)，因为batch_size会在权重更新时处理
        return (self.y_pred - self.y_true) / self.y_pred.shape[0]

# ==================== CNN 模型 ====================

class CNN:
    """卷积神经网络模型"""
    
    def __init__(self, learning_rate=0.01):
        """
        初始化CNN模型
        
        参数:
            learning_rate: 学习率
        """
        self.lr = learning_rate
        
        # 构建网络结构
        # 输入: (batch_size, 28, 28, 1)
        # Conv2D: 1 -> 16 channels, kernel_size=3, stride=1, padding=1
        self.conv1 = Conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # MaxPool2D: pool_size=2, stride=2
        # 输出: (batch_size, 14, 14, 16)
        self.pool = MaxPool2D(pool_size=2, stride=2)
        
        # Flatten: (batch_size, 14*14*16) = (batch_size, 3136)
        # Dense: 3136 -> 128
        self.fc1 = Dense(in_features=14*14*16, out_features=128)
        
        # Dense: 128 -> 10 (输出层)
        self.fc2 = Dense(in_features=128, out_features=10)
        
        # 损失函数
        self.criterion = CrossEntropyLoss()
        
        # 用于反向传播的缓存
        self.x_conv = None
        self.x_relu1 = None
        self.x_pool = None
        self.x_fc1 = None
        self.x_relu2 = None
    
    def forward(self, x, training=True):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (batch_size, 28, 28, 1)
            training: 是否为训练模式
        
        返回:
            输出概率，形状为 (batch_size, 10)
        """
        # Conv2D -> ReLU -> MaxPool2D
        self.x_conv = self.conv1.forward(x)
        self.x_relu1 = relu(self.x_conv)
        self.x_pool = self.pool.forward(self.x_relu1)
        
        # Flatten
        batch_size = self.x_pool.shape[0]
        x_flat = self.x_pool.reshape(batch_size, -1)
        
        # Dense -> ReLU
        self.x_fc1 = self.fc1.forward(x_flat)
        self.x_relu2 = relu(self.x_fc1)
        
        # Dense -> Softmax
        x = self.fc2.forward(self.x_relu2)
        x = softmax(x)
        
        return x
    
    def backward(self, dout):
        """
        反向传播
        
        参数:
            dout: 损失函数的梯度，形状为 (batch_size, 10)
        """
        # 反向传播通过各层
        # Dense2 -> ReLU2 -> Dense1
        dout = self.fc2.backward(dout)
        dout = relu_backward(self.x_relu2, dout)
        dout = self.fc1.backward(dout)
        
        # 恢复形状用于池化层反向传播
        batch_size = self.x_pool.shape[0]
        dout = dout.reshape(self.x_pool.shape)
        
        # Pool -> ReLU1 -> Conv1
        dout = self.pool.backward(dout)
        dout = relu_backward(self.x_relu1, dout)
        dout = self.conv1.backward(dout)
        
        return dout
    
    def update(self):
        """更新权重"""
        # 更新卷积层权重
        self.conv1.kernel -= self.lr * self.conv1.d_kernel
        self.conv1.bias -= self.lr * self.conv1.d_bias
        
        # 更新全连接层权重
        self.fc1.w -= self.lr * self.fc1.d_w
        self.fc1.b -= self.lr * self.fc1.d_b
        self.fc2.w -= self.lr * self.fc2.d_w
        self.fc2.b -= self.lr * self.fc2.d_b
    
    def train(self, x_train, y_train, batch_size=32, epochs=10):
        """
        训练模型
        
        参数:
            x_train: 训练数据，形状为 (N, 28, 28, 1)
            y_train: 训练标签（独热编码），形状为 (N, 10)
            batch_size: 批量大小
            epochs: 训练轮数
        
        返回:
            history: 训练历史（损失和准确率）
        """
        num_samples = x_train.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
        
        for epoch in range(epochs):
            # 打乱数据
            indices = np.random.permutation(num_samples)
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]
            
            epoch_loss = 0.0
            epoch_correct = 0
            
            # 批量训练
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                
                x_batch = x_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # 前向传播
                y_pred = self.forward(x_batch, training=True)
                
                # 计算损失
                loss = self.criterion.forward(y_pred, y_batch)
                epoch_loss += loss
                
                # 计算准确率
                pred_labels = np.argmax(y_pred, axis=1)
                true_labels = np.argmax(y_batch, axis=1)
                epoch_correct += np.sum(pred_labels == true_labels)
                
                # 反向传播
                dout = self.criterion.backward()
                self.backward(dout)
                
                # 更新权重
                self.update()
            
            # 计算平均损失和准确率
            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_correct / num_samples
            
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(avg_acc)
            
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}")
        
        return history
    
    def evaluate(self, x_test, y_test):
        """
        评估模型
        
        参数:
            x_test: 测试数据，形状为 (N, 28, 28, 1)
            y_test: 测试标签（独热编码），形状为 (N, 10)
        
        返回:
            loss: 平均损失
            accuracy: 准确率
        """
        # 前向传播
        y_pred = self.forward(x_test, training=False)
        
        # 计算损失
        loss = self.criterion.forward(y_pred, y_test)
        
        # 计算准确率
        pred_labels = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_test, axis=1)
        accuracy = np.mean(pred_labels == true_labels)
        
        return loss, accuracy
    
    def predict(self, x):
        """
        预测
        
        参数:
            x: 输入数据，形状为 (N, 28, 28, 1) 或 (28, 28, 1)
        
        返回:
            预测类别
        """
        if x.ndim == 3:
            x = x[np.newaxis, :, :, :]
        
        y_pred = self.forward(x, training=False)
        return np.argmax(y_pred, axis=1)

# ==================== 工具函数 ====================

def one_hot_encode(labels, num_classes=10):
    """将标签转换为独热编码"""
    num_samples = labels.shape[0]
    one_hot = np.zeros((num_samples, num_classes))
    one_hot[np.arange(num_samples), labels] = 1
    return one_hot

def load_mnist_images(filename):
    """加载MNIST图像文件"""
    with gzip.open(filename, 'rb') as f:
        # 跳过文件头（16字节）
        f.read(16)
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(-1, 28, 28)
    return data

def load_mnist_labels(filename):
    """加载MNIST标签文件"""
    with gzip.open(filename, 'rb') as f:
        # 跳过文件头（8字节）
        f.read(8)
        buf = f.read()
        labels = np.frombuffer(buf, dtype=np.uint8)
    return labels

def download_mnist():
    """下载MNIST数据集"""
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    data_dir = 'mnist_data'
    os.makedirs(data_dir, exist_ok=True)
    
    downloaded_files = {}
    for key, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"正在下载 {filename}...")
            url = base_url + filename
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"  {filename} 下载完成")
            except Exception as e:
                print(f"  下载失败: {e}")
                # 如果下载失败，使用模拟数据
                return None
        downloaded_files[key] = filepath
    
    return downloaded_files

def load_mnist_data(use_reduced_data=True, train_samples=1000, test_samples=100):
    """
    加载MNIST数据集（不依赖TensorFlow）
    
    参数:
        use_reduced_data: 是否使用减少的数据量（用于快速测试）
        train_samples: 训练样本数（如果use_reduced_data=True）
        test_samples: 测试样本数（如果use_reduced_data=True）
    
    返回:
        x_train, y_train, x_test, y_test
    """
    print("Loading MNIST dataset...")
    start_time = time.time()
    
    try:
        # 尝试下载并加载真实MNIST数据
        files = download_mnist()
        if files:
            x_train = load_mnist_images(files['train_images'])
            y_train = load_mnist_labels(files['train_labels'])
            x_test = load_mnist_images(files['test_images'])
            y_test = load_mnist_labels(files['test_labels'])
        else:
            raise Exception("无法下载MNIST数据，使用模拟数据")
    except Exception as e:
        print(f"警告: {e}")
        print("使用模拟MNIST数据（随机生成）...")
        # 生成模拟数据用于测试
        np.random.seed(42)
        x_train = np.random.randint(0, 256, (60000, 28, 28), dtype=np.uint8)
        y_train = np.random.randint(0, 10, 60000, dtype=np.uint8)
        x_test = np.random.randint(0, 256, (10000, 28, 28), dtype=np.uint8)
        y_test = np.random.randint(0, 10, 10000, dtype=np.uint8)
    
    load_time = time.time() - start_time
    print(f"Dataset loaded successfully! (Time: {load_time:.2f}s)")
    
    # 显示原始数据形状
    print(f"Original - x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"Original - x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
    
    # 数据预处理
    # 归一化
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    # 添加通道维度
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    
    # 转换为独热编码
    y_train = one_hot_encode(y_train, num_classes=10)
    y_test = one_hot_encode(y_test, num_classes=10)
    
    # 可选：减少数据量（用于快速测试）
    if use_reduced_data:
        x_train = x_train[:train_samples]
        y_train = y_train[:train_samples]
        x_test = x_test[:test_samples]
        y_test = y_test[:test_samples]
        print(f"Using reduced dataset - x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
        print(f"Using reduced dataset - x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
    
    return x_train, y_train, x_test, y_test

def visualize_predictions(model, x_test, y_test, num_samples=10):
    """
    可视化测试样本的预测结果
    
    参数:
        model: CNN模型
        x_test: 测试数据
        y_test: 测试标签（独热编码）
        num_samples: 要显示的样本数
    """
    # 随机选择样本
    indices = np.random.choice(x_test.shape[0], num_samples, replace=False)
    x_samples = x_test[indices]
    y_samples = y_test[indices]
    
    # 预测
    pred_labels = model.predict(x_samples)
    true_labels = np.argmax(y_samples, axis=1)
    
    # 创建图像
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle('测试样本预测结果', fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            img = x_samples[i].squeeze()
            ax.imshow(img, cmap='gray')
            
            # 判断预测是否正确
            is_correct = pred_labels[i] == true_labels[i]
            color = 'green' if is_correct else 'red'
            title = f'真实: {true_labels[i]}, 预测: {pred_labels[i]}'
            
            ax.set_title(title, color=color, fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    # 直接保存图片到本地
    plt.savefig('test_predictions.png', dpi=150, bbox_inches='tight')
    print("✓ 测试结果图已保存: test_predictions.png")
    try:
        plt.show()
    except:
        pass
    plt.close()

def visualize_training_history(history):
    """
    可视化训练曲线
    
    参数:
        history: 训练历史字典
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    ax1.plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
    ax1.plot(epochs, history['test_loss'], 'r-', label='测试损失', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('训练和测试损失', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(epochs, history['train_acc'], 'b-', label='训练准确率', linewidth=2)
    ax2.plot(epochs, history['test_acc'], 'r-', label='测试准确率', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('训练和测试准确率', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # 直接保存图片到本地
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("✓ 训练曲线图已保存: training_curves.png")
    try:
        plt.show()
    except:
        pass
    plt.close()

# ==================== 主函数 ====================

def main():
    """主函数"""
    print("=" * 60)
    print("NumPy 实现 CNN 实验 - MNIST 手写数字分类")
    print("=" * 60)
    
    # 1. 加载数据
    x_train, y_train, x_test, y_test = load_mnist_data(
        use_reduced_data=True,  # 设置为False使用完整数据集
        train_samples=5000,     # 训练样本数（增加到5000以提升效果）
        test_samples=100      # 测试样本数（增加到1000以更好评估）
    )
    
    # 2. 创建模型
    print("\n创建CNN模型...")
    cnn = CNN(learning_rate=0.01)
    print("模型创建完成！")
    
    # 3. 训练模型
    print("\n开始训练...")
    epochs = 5  # 训练轮数（增加到10轮以提升效果）
    batch_size = 64  # 批量大小（增加到64以加快训练）
    learning_rate = 0.01  # 学习率（稍微降低以提高稳定性）
    
    # 更新模型学习率
    cnn.lr = learning_rate
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    start_time = time.time()
    
    # 使用tqdm显示epoch进度
    epoch_pbar = tqdm(range(epochs), desc="训练进度", unit="epoch", ncols=100)
    
    for epoch in epoch_pbar:
        # 训练
        num_samples = x_train.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        # 打乱数据
        indices = np.random.permutation(num_samples)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]
        
        epoch_loss = 0.0
        epoch_correct = 0
        
        # 使用tqdm显示batch进度
        batch_pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}", 
                        leave=False, unit="batch", ncols=100)
        
        for i in batch_pbar:
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            
            x_batch = x_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            # 前向传播
            y_pred = cnn.forward(x_batch, training=True)
            
            # 计算损失
            loss = cnn.criterion.forward(y_pred, y_batch)
            epoch_loss += loss
            
            # 计算准确率
            pred_labels = np.argmax(y_pred, axis=1)
            true_labels = np.argmax(y_batch, axis=1)
            epoch_correct += np.sum(pred_labels == true_labels)
            
            # 反向传播
            dout = cnn.criterion.backward()
            cnn.backward(dout)
            
            # 更新权重
            cnn.update()
            
            # 更新batch进度条
            batch_pbar.set_postfix({'Loss': f'{loss:.4f}', 
                                   'Avg Loss': f'{epoch_loss/(i+1):.4f}'})
        
        batch_pbar.close()
        
        # 计算平均损失和准确率
        avg_train_loss = epoch_loss / num_batches
        avg_train_acc = epoch_correct / num_samples
        
        # 评估
        test_loss, test_acc = cnn.evaluate(x_test, y_test)
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # 更新epoch进度条
        epoch_pbar.set_postfix({
            'Train Loss': f'{avg_train_loss:.4f}',
            'Train Acc': f'{avg_train_acc:.4f}',
            'Test Loss': f'{test_loss:.4f}',
            'Test Acc': f'{test_acc:.4f}'
        })
        
        print(f"\nEpoch {epoch+1}/{epochs} 完成: Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {avg_train_acc:.4f}, Test Loss: {test_loss:.4f}, "
              f"Test Acc: {test_acc:.4f}")
    
    epoch_pbar.close()
    
    total_time = time.time() - start_time
    avg_time_per_epoch = total_time / epochs
    
    print(f"\n总训练时间: {total_time:.2f} 秒")
    print(f"每轮平均训练时间: {avg_time_per_epoch:.2f} 秒")
    
    # 4. 可视化结果
    print("\n生成可视化结果...")
    visualize_predictions(cnn, x_test, y_test, num_samples=10)
    visualize_training_history(history)
    
    print("\n实验完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()

