import json
import os
import argparse
from classify.train import run_ablation
from classify.data import prefetch_datasets

def clean_results():
    d = 'classify/results'
    if os.path.isdir(d):
        for f in os.listdir(d):
            try:
                os.remove(os.path.join(d, f))
            except Exception:
                pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--datasets', type=str, default='mnist,cifar10,seqmnist')
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--clean', action='store_true', default=False)
    parser.add_argument('--prefetch', action='store_true', default=True)
    args = parser.parse_args()

    if args.clean:
        clean_results()

    os.makedirs('classify/results', exist_ok=True)
    if args.prefetch:
        print(f'Prefetching datasets: {args.datasets}')
        prefetch_datasets(args.datasets)
    all_summaries = {}
    for ds in args.datasets.split(','):
        print(f'Running dataset={ds} (resume={args.resume})')
        summaries = run_ablation(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            dataset=ds,
            resume=args.resume
        )
        all_summaries[ds] = summaries
    with open('classify/results/ablation_summary.json', 'w') as f:
        json.dump(all_summaries, f)
    print('ablation done')

if __name__ == '__main__':
    main()
