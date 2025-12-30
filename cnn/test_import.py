"""
快速测试脚本 - 验证所有模块能否正确导入
"""

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    print("✓ 所有依赖库导入成功")
    
    # 测试导入主模块
    from numpy_cnn_mnist import (
        Conv2D, MaxPool2D, Dense, CNN,
        relu, softmax, CrossEntropyLoss,
        one_hot_encode, load_mnist_data
    )
    print("✓ 所有核心组件导入成功")
    
    # 测试创建模型
    cnn = CNN(learning_rate=0.01)
    print("✓ CNN 模型创建成功")
    
    print("\n所有测试通过！可以运行 numpy_cnn_mnist.py")
    
except ImportError as e:
    print(f"✗ 导入错误: {e}")
    print("请确保已安装所有依赖: pip install numpy matplotlib tensorflow")
except Exception as e:
    print(f"✗ 错误: {e}")

