import json
import os
import matplotlib.pyplot as plt
from typing import List, Dict

def load_summary(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _set_limits_from_series(series_list: List[List[float]], kind: str):
    xs = [list(range(1, len(s)+1)) for s in series_list]
    max_x = max(len(s) for s in series_list) if series_list else 1
    plt.xlim(1, max_x)
    if kind == 'acc':
        ymin = min((min(s) for s in series_list), default=0.0)
        ymax = max((max(s) for s in series_list), default=1.0)
        pad = 0.05
        ymin = max(0.0, ymin - pad)
        ymax = min(1.0, ymax + pad)
        plt.ylim(ymin, ymax)
    else:
        ymin = min((min(s) for s in series_list), default=0.0)
        ymax = max((max(s) for s in series_list), default=1.0)
        y_range = max(ymax - ymin, 1e-8)
        pad = max(0.1 * y_range, 0.05 * max(ymax, 1.0))
        lower = max(0.0, ymin - pad)
        upper = ymax + pad
        if lower >= upper:
            lower = max(0.0, ymin - 0.1)
            upper = ymax + 0.1
        plt.ylim(lower, upper)

def plot_cross_model(dataset: str, summary: Dict, out_dir: str, epochs: int):
    names = [k for k in summary.get(dataset, {}).keys()]
    if not names:
        return
    # Accuracy
    acc_series = []
    plt.figure(figsize=(6,4))
    for name in names:
        acc = summary[dataset][name]['base']['test_acc']
        acc_series.append(acc)
        plt.plot(range(1, len(acc)+1), acc, linewidth=2, label=name)
    _set_limits_from_series(acc_series, kind='acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{dataset}_cross_model_test_acc.pdf'))
    plt.close()
    # Loss
    loss_series = []
    plt.figure(figsize=(6,4))
    for name in names:
        loss = summary[dataset][name]['base']['test_loss']
        loss_series.append(loss)
        plt.plot(range(1, len(loss)+1), loss, linewidth=2, label=name)
    _set_limits_from_series(loss_series, kind='loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{dataset}_cross_model_test_loss.pdf'))
    plt.close()

def plot_per_model(dataset: str, summary: Dict, out_dir: str):
    if dataset not in summary:
        return
    for name, variants in summary[dataset].items():
        # Accuracy
        acc_series = []
        plt.figure(figsize=(6,4))
        for tag, hist in variants.items():
            acc = hist['test_acc']
            acc_series.append(acc)
            plt.plot(range(1, len(acc)+1), acc, linewidth=2, label=tag)
        _set_limits_from_series(acc_series, kind='acc')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.grid(True)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{dataset}_{name}_ablation_test_acc.pdf'))
        plt.close()
        # Loss
        loss_series = []
        plt.figure(figsize=(6,4))
        for tag, hist in variants.items():
            loss = hist['test_loss']
            loss_series.append(loss)
            plt.plot(range(1, len(loss)+1), loss, linewidth=2, label=tag)
        _set_limits_from_series(loss_series, kind='loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid(True)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{dataset}_{name}_ablation_test_loss.pdf'))
        plt.close()

def main():
    summary = load_summary('classify/results/ablation_summary.json')
    out_dir = 'paper/figs'
    ensure_dir(out_dir)
    datasets = ['mnist', 'cifar10', 'sms_spam']
    for ds in datasets:
        plot_cross_model(ds, summary, out_dir, epochs=15)
        plot_per_model(ds, summary, out_dir)

if __name__ == '__main__':
    main()
