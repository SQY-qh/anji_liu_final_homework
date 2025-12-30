import json
from classify.train import run_all

if __name__ == '__main__':
    models = ['mlp', 'cnn', 'mingru', 'mamba', 'transformer']
    results = run_all(models)
    with open('classify/results/summary.json', 'w') as f:
        json.dump(results, f)
    print('done')
