import os
import shutil

src = 'classify/results'
dst = 'paper/figs'
os.makedirs(dst, exist_ok=True)

files = [
    'mnist_cross_model_test_acc.png',
    'mnist_cross_model_test_loss.png',
    'cifar10_cross_model_test_acc.png',
    'cifar10_cross_model_test_loss.png',
    'sms_spam_cross_model_test_acc.png',
    'sms_spam_cross_model_test_loss.png'
]

# Add MNIST and CIFAR-10 ablation pngs
for ds in ['mnist','cifar10']:
    for name in ['cnn','mlp','mingru','mamba','transformer']:
        for kind in ['acc','loss']:
            files.append(f'{ds}_{name}_ablation_test_{kind}.png')

for f in files:
    s = os.path.join(src, f)
    if os.path.exists(s):
        shutil.copy2(s, os.path.join(dst, f))
