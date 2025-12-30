import json
import os
import sys
sys.path.append(os.getcwd())
from classify.train import run_ablation

def load(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def save(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)

def merge_run(dataset, epochs, batch_size=64, lr=1e-3, device='cpu'):
    path = 'classify/results/ablation_summary.json'
    existing = load(path)
    summaries = run_ablation(epochs=epochs, batch_size=batch_size, lr=lr, device=device, dataset=dataset, resume=True)
    existing[dataset] = summaries
    save(path, existing)

def main():
    merge_run('mnist', epochs=8)
    merge_run('cifar10', epochs=15)
    merge_run('sms_spam', epochs=15)

if __name__ == '__main__':
    main()
