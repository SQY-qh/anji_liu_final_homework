import numpy as np
import itertools
from mlp.model import MLP
from mlp.layers import Dense
from mlp.activations import ReLU, Linear
from mlp.losses import MSE
from mlp.optimizers import Adam, SGD
from mlp.datasets import BostonHousingLoader

def grid_search():
    """网格搜索超参数"""
    # 定义超参数网格
    param_grid = {
        'hidden_sizes': [(64, 32), (128, 64), (32, 16)],
        'learning_rate': [0.001, 0.0005],
        'batch_size': [32, 64],
        'optimizer': ['Adam', 'SGD']
    }
    
    # 加载数据
    data_loader = BostonHousingLoader(random_state=42)
    X_train, y_train, X_test, y_test = data_loader.load_data()
    
    best_r2 = -np.inf
    best_params = None
    best_model = None
    
    # 遍历所有超参数组合
    for hidden_sizes in param_grid['hidden_sizes']:
        for lr in param_grid['learning_rate']:
            for batch_size in param_grid['batch_size']:
                for opt_name in param_grid['optimizer']:
                    # 创建优化器
                    if opt_name == 'Adam':
                        optimizer = Adam(learning_rate=lr)
                    else:
                        optimizer = SGD(learning_rate=lr, momentum=0.9)
                    
                    # 构建模型
                    model = MLP()
                    model.add_layer(Dense(13, hidden_sizes[0], weight_initializer='he'))
                    model.add_layer(ReLU())
                    model.add_layer(Dense(hidden_sizes[0], hidden_sizes[1], weight_initializer='he'))
                    model.add_layer(ReLU())
                    model.add_layer(Dense(hidden_sizes[1], 1, weight_initializer='he'))
                    model.add_layer(Linear())
                    model.set_loss(MSE())
                    model.set_optimizer(optimizer)
                    
                    # 训练模型
                    print(f"训练参数：hidden_sizes={hidden_sizes}, lr={lr}, batch_size={batch_size}, optimizer={opt_name}")
                    model.train(X_train, y_train, epochs=80, batch_size=batch_size, validation_data=(X_test, y_test))
                    
                    # 评估模型
                    _, metrics = model.evaluate(X_test, y_test)
                    print(f"R²评分：{metrics['r2']:.4f}\n")
                    
                    # 更新最佳模型
                    if metrics['r2'] > best_r2:
                        best_r2 = metrics['r2']
                        best_params = {
                            'hidden_sizes': hidden_sizes,
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'optimizer': opt_name
                        }
                        best_model = model
    
    # 输出最佳结果
    print("=" * 50)
    print(f"最佳超参数：{best_params}")
    print(f"最佳 R²评分：{best_r2:.4f}")
    best_model.plot_history(save_path='results/best_training_curve.png')
    best_model.plot_predictions(X_test, y_test, save_path='results/best_predictions.png')
    print("=" * 50)

def demo_way1():
    """方式 1：分别设置各组件"""
    print("\n" + "=" * 50)
    print("方式 1：分别设置各组件")
    print("=" * 50)
    
    # 加载数据
    data_loader = BostonHousingLoader(random_state=42)
    X_train, y_train, X_test, y_test = data_loader.load_data()
    
    # 构建模型
    model = MLP()
    model.add_layer(Dense(13, 64, weight_initializer='he'))
    model.add_layer(ReLU())
    model.add_layer(Dense(64, 32, weight_initializer='he'))
    model.add_layer(ReLU())
    model.add_layer(Dense(32, 1, weight_initializer='he'))
    model.add_layer(Linear())
    
    # 分别设置损失函数和优化器
    model.set_loss(MSE())
    model.set_optimizer(Adam(learning_rate=0.001))
    
    # 训练模型
    print("开始训练（10 轮，用于快速演示）...")
    model.train(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    # 评估模型
    test_loss, metrics = model.evaluate(X_test, y_test)
    print(f"\n测试集 MSE：{test_loss:.4f}")
    print(f"MAE：{metrics['mae']:.2f}")
    print(f"R²：{metrics['r2']:.4f}")

def demo_way2():
    """方式 2：使用不同的优化器和学习率"""
    print("\n" + "=" * 50)
    print("方式 2：使用 SGD 优化器")
    print("=" * 50)
    
    # 加载数据
    data_loader = BostonHousingLoader(random_state=42)
    X_train, y_train, X_test, y_test = data_loader.load_data()
    
    # 构建模型
    model = MLP()
    model.add_layer(Dense(13, 64, weight_initializer='he'))
    model.add_layer(ReLU())
    model.add_layer(Dense(64, 32, weight_initializer='he'))
    model.add_layer(ReLU())
    model.add_layer(Dense(32, 1, weight_initializer='he'))
    model.add_layer(Linear())
    
    # 使用 SGD 优化器
    model.set_loss(MSE())
    model.set_optimizer(SGD(learning_rate=0.01, momentum=0.9))
    
    # 训练模型
    print("开始训练（10 轮，用于快速演示）...")
    model.train(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    # 评估模型
    test_loss, metrics = model.evaluate(X_test, y_test)
    print(f"\n测试集 MSE：{test_loss:.4f}")
    print(f"MAE：{metrics['mae']:.2f}")
    print(f"R²：{metrics['r2']:.4f}")

def main():
    print("=" * 50)
    print("高级演示：超参数调优与模型改进")
    print("=" * 50)
    
    # 演示两种构建方式
    demo_way1()
    demo_way2()
    
    # 可选：运行网格搜索（耗时较长，可注释掉）
    # print("\n" + "=" * 50)
    # print("网格搜索超参数（可选，耗时较长）")
    # print("=" * 50)
    # grid_search()

if __name__ == "__main__":
    main()

