#!/usr/bin/env python3
"""快速验证项目设置是否正确"""

import sys
import os

def check_imports():
    """检查所有模块是否可以正常导入"""
    print("=" * 50)
    print("检查模块导入...")
    print("=" * 50)
    
    try:
        from mlp.layers import Layer, Dense
        print("✅ layers 模块导入成功")
    except Exception as e:
        print(f"❌ layers 模块导入失败: {e}")
        return False
    
    try:
        from mlp.activations import Activation, ReLU, Tanh, Sigmoid, Linear
        print("✅ activations 模块导入成功")
    except Exception as e:
        print(f"❌ activations 模块导入失败: {e}")
        return False
    
    try:
        from mlp.losses import Loss, MSE
        print("✅ losses 模块导入成功")
    except Exception as e:
        print(f"❌ losses 模块导入失败: {e}")
        return False
    
    try:
        from mlp.optimizers import Optimizer, SGD, Adam
        print("✅ optimizers 模块导入成功")
    except Exception as e:
        print(f"❌ optimizers 模块导入失败: {e}")
        return False
    
    try:
        from mlp.model import MLP
        print("✅ model 模块导入成功")
    except Exception as e:
        print(f"❌ model 模块导入失败: {e}")
        return False
    
    try:
        from mlp.datasets import BostonHousingLoader
        print("✅ datasets 模块导入成功")
    except Exception as e:
        print(f"❌ datasets 模块导入失败: {e}")
        return False
    
    try:
        # 测试统一导入
        from mlp import MLP, Dense, ReLU, Linear, MSE, Adam, BostonHousingLoader
        print("✅ 统一导入接口正常")
    except Exception as e:
        print(f"❌ 统一导入接口失败: {e}")
        return False
    
    return True

def check_dependencies():
    """检查依赖包是否已安装"""
    print("\n" + "=" * 50)
    print("检查依赖包...")
    print("=" * 50)
    
    required_packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib'
    }
    
    all_ok = True
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"✅ {package_name} 已安装")
        except ImportError:
            print(f"❌ {package_name} 未安装，请运行: pip install -r requirements.txt")
            all_ok = False
    
    return all_ok

def check_project_structure():
    """检查项目结构是否完整"""
    print("\n" + "=" * 50)
    print("检查项目结构...")
    print("=" * 50)
    
    required_files = [
        'mlp/__init__.py',
        'mlp/layers.py',
        'mlp/activations.py',
        'mlp/losses.py',
        'mlp/optimizers.py',
        'mlp/model.py',
        'mlp/datasets.py',
        'demo.py',
        'demo_advanced.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_ok = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} 不存在")
            all_ok = False
    
    # 检查 results 目录（如果不存在会自动创建）
    if not os.path.exists('results'):
        print("ℹ️  results/ 目录不存在（运行 demo.py 时会自动创建）")
    
    return all_ok

def main():
    """主函数"""
    print("\n" + "=" * 50)
    print("NumPy MLP 项目设置验证")
    print("=" * 50)
    
    # 检查项目结构
    structure_ok = check_project_structure()
    
    # 检查依赖
    deps_ok = check_dependencies()
    
    # 检查导入
    imports_ok = check_imports()
    
    # 总结
    print("\n" + "=" * 50)
    print("验证结果总结")
    print("=" * 50)
    
    if structure_ok and deps_ok and imports_ok:
        print("✅ 所有检查通过！项目设置正确，可以开始实验。")
        print("\n下一步：运行 python demo.py 开始训练模型")
        return 0
    else:
        print("❌ 部分检查未通过，请根据上述提示修复问题。")
        if not deps_ok:
            print("\n建议：运行 pip install -r requirements.txt 安装依赖")
        return 1

if __name__ == "__main__":
    sys.exit(main())

