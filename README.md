多数据集分类对比实验指南（MNIST/CIFAR-10/SMS Spam；MLP/CNN/MinGRU/Mamba/Transformer）

一、项目简介
- 覆盖图像与文本两类任务：`mnist`、`cifar10`、`sms_spam`
- 统一数据加载、训练与评估；提供断点续跑与实验汇总，生成横向（跨模型）与纵向（模型消融）对比图
- 模型家族：`mlp`、`cnn`、`mingru`、`mamba`、`transformer`（图像用 `SimpleTransformer`，序列用 `SequenceTransformer`）

二、环境准备
- 依赖：`torch`、`numpy`、`matplotlib`；图像数据在相关分支中按需导入 `torchvision`
- 安装：`pip install -r classify/requirements.txt`
- 设备：默认 `cpu`；如具备 GPU，运行时传入 `--device cuda`

三、快速开始（推荐 CLI）
- 预取并断点续跑单数据集：
  - `python -m classify.experiments --resume --prefetch --epochs 8  --datasets mnist`
  - `python -m classify.experiments --resume --prefetch --epochs 15 --datasets cifar10`
  - `python -m classify.experiments --resume --prefetch --epochs 15 --datasets sms_spam`
- 批量运行多个数据集：
  - `python -m classify.experiments --resume --prefetch --epochs 15 --datasets mnist,cifar10,sms_spam`
- 单模型交互示例：
  - `from classify.train import train_one`
  - `hist = train_one('cnn', epochs=3, batch_size=64, lr=1e-3, device='cpu', params={'in_chans':1,'image_size':28})`

四、数据集与下载
- `mnist`：自动下载灰度手写数字（28×28，10 类）
- `cifar10`：自动下载彩色图像（32×32，10 类），内置镜像与解压，提升下载稳定性
- `sms_spam`：UCI SMS Spam Collection（词级）；自动下载 zip 并解析
  - 词表：Top-K（默认 10000），`PAD=0`、`UNK=1`，实际 `vocab_size=10002`
  - 序列长度：`max_len=50`（截断与填充）

五、模型与主要变体
- `mlp`：`base(w=256,128)`、`wide(w=512,256)`、`tanh`
- `cnn`（图像）：`base(32,64)`、`wide_bn(64,128,BN)`、`k5_do(kernel=5,dropout)`
- `cnn`（文本）：TextCNN（Conv1d + `Embedding(vocab_size, embed_dim=128)`）
- `mingru`：`base`、`hidden256/512`、`time_dim1/2`
- `mamba`：`base`、`bigger_state`、`wider`
- `transformer`（图像）：`base`、`larger`、`more_tokens`
- `transformer`（文本）：`SequenceTransformer`，可设 `vocab_size`、`embed_dim`、`num_heads`、`depth`

六、训练与断点续跑
- 优化：`Adam`（MLP/CNN/MinGRU/Mamba）、`AdamW+Cosine`（Transformer）
- 序列任务启用梯度裁剪：`clip_grad_norm_ = 1.0`
- 断点续跑与跳过已完成：
  - 根据 `classify/results/ablation_summary.json` 与曲线文件是否存在，自动跳过已完成子实验
  - 传入 `--resume` 开启续跑；`--prefetch` 预取数据，避免首次迭代阻塞

七、输出与可视化（PNG）
- 目录：`classify/results/`
- 单模型曲线：`{dataset}_{model}_{variant}_curves.png`
- 横向对比：
  - 测试准确率：`{dataset}_cross_model_test_acc.png`
  - 测试损失：`{dataset}_cross_model_test_loss.png`
- 纵向消融：
  - 测试准确率：`{dataset}_{model}_ablation_test_acc.png`
  - 测试损失：`{dataset}_{model}_ablation_test_loss.png`
- 指标汇总：`ablation_summary.json`（逐 epoch 的 `train_loss/test_loss/train_acc/test_acc`）

八、实验结论（基于最新摘要）
- MNIST：`cnn/transformer` 最强；`mlp` 稳健；序列模型略逊于具备空间先验的架构
- CIFAR-10：`cnn` 与随容量提升的 `transformer` 表现最佳；缺少空间结构的 `mlp/序列模型` 相对弱
- SMS Spam（词级）：`mingru/mamba/transformer` 显著优于 `mlp/cnn`；充足 `embed_dim/depth` 与梯度裁剪带来稳定提升

九、论文导出（可选）
- 生成图表（按 `ablation_summary.json` 重绘，含动态纵轴与图例）：`python paper/make_figs.py`
- 编译论文（XeLaTeX）：`xelatex -interaction=nonstopmode -halt-on-error -output-directory paper paper/main.tex`
- 论文位置：`paper/main.pdf`；图目录：`paper/figs/`

十、常见问题
- 下载缓慢：使用 `--prefetch` 提前拉取；`cifar10` 内置镜像与自动解压
- 曲线不全或出框：重新运行 `python paper/make_figs.py`；该脚本会动态调节坐标范围并添加图例
- 训练重启：传入 `--resume`；确保未手动清理 `classify/results/` 现有文件
- 文本模型报错：确保 `vocab_size/embed_dim` 传参正确（已在 `train.py` 自动设置）

十一、文件与命令速览
- 入口：`classify/experiments.py`（CLI）、`classify/train.py`（训练与汇总）
- 数据：`classify/data.py`（下载与解析、词级管线）
- 模型：`classify/models/`（`mlp/cnn/mingru/mamba/transformer`）
- 运行示例：
  - `python -m classify.experiments --resume --prefetch --epochs 15 --datasets sms_spam`
  - `python -m classify.experiments --resume --prefetch --epochs 8  --datasets mnist`
  - `python -m classify.experiments --resume --prefetch --epochs 15 --datasets cifar10`
anji_liu_final_homework
