"""
快速测试脚本 - 使用模拟数据快速验证CNN实现
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示问题
import matplotlib.pyplot as plt
import time
import sys
import os

# 导入主模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from numpy_cnn_mnist import (
    CNN, one_hot_encode, visualize_predictions, visualize_training_history
)

def generate_mock_data(num_train=1000, num_test=100):
    """生成模拟MNIST数据"""
    np.random.seed(42)
    
    # 生成模拟图像数据（28x28）
    x_train = np.random.rand(num_train, 28, 28, 1).astype(np.float32)
    x_test = np.random.rand(num_test, 28, 28, 1).astype(np.float32)
    
    # 生成模拟标签
    y_train = np.random.randint(0, 10, num_train)
    y_test = np.random.randint(0, 10, num_test)
    
    # 转换为独热编码
    y_train = one_hot_encode(y_train, num_classes=10)
    y_test = one_hot_encode(y_test, num_classes=10)
    
    return x_train, y_train, x_test, y_test

def main():
    """快速测试主函数"""
    print("=" * 60)
    print("NumPy CNN 快速测试 - 使用模拟数据")
    print("=" * 60)
    
    # 1. 生成模拟数据
    print("\n生成模拟数据...")
    x_train, y_train, x_test, y_test = generate_mock_data(
        num_train=1000,
        num_test=100
    )
    print(f"训练数据: {x_train.shape}, {y_train.shape}")
    print(f"测试数据: {x_test.shape}, {y_test.shape}")
    
    # 2. 创建模型
    print("\n创建CNN模型...")
    cnn = CNN(learning_rate=0.01)
    print("模型创建完成！")
    
    # 3. 训练模型
    print("\n开始训练...")
    epochs = 2
    batch_size = 32
    learning_rate = 0.01
    
    cnn.lr = learning_rate
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        num_samples = x_train.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        
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
            
            # 显示进度
            if (i + 1) % 10 == 0 or (i + 1) == num_batches:
                print(f"  Batch {i+1}/{num_batches}, Loss: {loss:.4f}")
        
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
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {avg_train_acc:.4f}, Test Loss: {test_loss:.4f}, "
              f"Test Acc: {test_acc:.4f}")
    
    total_time = time.time() - start_time
    avg_time_per_epoch = total_time / epochs
    
    print(f"\n总训练时间: {total_time:.2f} 秒")
    print(f"每轮平均训练时间: {avg_time_per_epoch:.2f} 秒")
    
    # 4. 保存可视化结果（不显示）
    print("\n保存可视化结果...")
    try:
        visualize_predictions(cnn, x_test, y_test, num_samples=10)
        plt.savefig('test_predictions.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  测试结果图已保存: test_predictions.png")
    except Exception as e:
        print(f"  保存测试结果图失败: {e}")
    
    try:
        visualize_training_history(history)
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  训练曲线图已保存: training_curves.png")
    except Exception as e:
        print(f"  保存训练曲线图失败: {e}")
    
    print("\n实验完成！")
    print("=" * 60)
    
    # 返回结果摘要
    return {
        'final_train_loss': history['train_loss'][-1],
        'final_train_acc': history['train_acc'][-1],
        'final_test_loss': history['test_loss'][-1],
        'final_test_acc': history['test_acc'][-1],
        'total_time': total_time
    }

if __name__ == "__main__":
    results = main()
    print("\n结果摘要:")
    print(f"  最终训练损失: {results['final_train_loss']:.4f}")
    print(f"  最终训练准确率: {results['final_train_acc']:.4f}")
    print(f"  最终测试损失: {results['final_test_loss']:.4f}")
    print(f"  最终测试准确率: {results['final_test_acc']:.4f}")
    print(f"  总训练时间: {results['total_time']:.2f} 秒")

