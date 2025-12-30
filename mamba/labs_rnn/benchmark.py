import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import labs_rnn.test_model as tm


def _set_chinese_font():
    candidates = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Songti SC",
        "Heiti SC",
        "STHeiti",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
    ]
    installed = {f.name.lower() for f in fm.fontManager.ttflist}
    for name in candidates:
        if name.lower() in installed:
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["axes.unicode_minus"] = False
            break


if __name__ == "__main__":
    _set_chinese_font()
    min_gru_results = tm.train_min_gru()
    old_lr = tm.learning_rate
    tm.learning_rate = 0.005
    mamba_results = tm.train_mamba()
    tm.learning_rate = old_lr
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, tm.n_epochs + 1), min_gru_results["train_losses"], label="MinGRU")
    plt.plot(range(1, tm.n_epochs + 1), mamba_results["train_losses"], label="Mamba")
    plt.title("训练损失对比")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, tm.n_epochs + 1), min_gru_results["valid_losses"], label="MinGRU")
    plt.plot(range(1, tm.n_epochs + 1), mamba_results["valid_losses"], label="Mamba")
    plt.title("验证损失对比")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mamba_vs_mingru.png")
    print("可视化结果已保存到 mamba_vs_mingru.png")
