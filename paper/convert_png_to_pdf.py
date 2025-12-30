import os
import matplotlib.pyplot as plt

pairs = [
    ('paper/figs/cifar10_cross_model_test_acc.png', 'paper/figs/cifar10_cross_model_test_acc.pdf'),
    ('paper/figs/cifar10_cross_model_test_loss.png', 'paper/figs/cifar10_cross_model_test_loss.pdf')
]

pairs += [
    ('paper/figs/mnist_cross_model_test_acc.png', 'paper/figs/mnist_cross_model_test_acc.pdf'),
    ('paper/figs/mnist_cross_model_test_loss.png', 'paper/figs/mnist_cross_model_test_loss.pdf')
]

# CIFAR-10 ablation pngs
for name in ['cnn','mlp','mingru','mamba','transformer']:
    for kind in ['acc','loss']:
        src = f'paper/figs/cifar10_{name}_ablation_test_{kind}.png'
        dst = f'paper/figs/cifar10_{name}_ablation_test_{kind}.pdf'
        pairs.append((src, dst))

# MNIST ablation pngs
for name in ['cnn','mlp','mingru','mamba','transformer']:
    for kind in ['acc','loss']:
        src = f'paper/figs/mnist_{name}_ablation_test_{kind}.png'
        dst = f'paper/figs/mnist_{name}_ablation_test_{kind}.pdf'
        pairs.append((src, dst))

for png, pdf in pairs:
    if os.path.exists(png):
        img = plt.imread(png)
        plt.figure(figsize=(6,4))
        plt.axis('off')
        plt.imshow(img)
        plt.savefig(pdf, bbox_inches='tight')
        plt.close()
