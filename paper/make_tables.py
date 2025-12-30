import json
import os

def load_summary(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def write_tables(summary, out_tex='paper/tables.tex'):
    lines = []
    def table_header(caption, label, cols):
        lines.append('')
        lines.append(r'\begin{table}[h]')
        lines.append(r'\centering')
        lines.append(r'\caption{%s}' % caption)
        lines.append(r'\label{%s}' % label)
        lines.append(r'\begin{tabular}{%s}' % cols)
        lines.append(r'\toprule')
    def table_footer():
        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append(r'\end{table}')
    # Cross-model base variant final metrics
    title_map = {'mnist': 'MNIST', 'cifar10': 'CIFAR-10', 'sms_spam': 'SMS Spam'}
    for ds in ['mnist', 'cifar10', 'sms_spam']:
        if ds not in summary:
            continue
        ds_title = title_map.get(ds, ds.replace('_', ' '))
        table_header(f'{ds_title} Base Variants: Final Test Metrics', f'tab:{ds}-base', 'lcc')
        lines.append(r'Model & Test Acc & Test Loss \\')
        lines.append(r'\midrule')
        for name, variants in summary[ds].items():
            hist = variants.get('base')
            if not hist:
                continue
            acc = hist['test_acc'][-1]
            loss = hist['test_loss'][-1]
            safe_name = name.replace('_', r'\_')
            lines.append(r'%s & %.4f & %.4f \\' % (safe_name, acc, loss))
        table_footer()
    # SMS Spam detailed ablations
    ds = 'sms_spam'
    if ds in summary:
        for name, variants in summary[ds].items():
            table_header(f'SMS Spam {name} Ablations: Final Test Metrics', f'tab:sms-{name}', 'lcc')
            lines.append(r'Variant & Test Acc & Test Loss \\')
            lines.append(r'\midrule')
            for tag, hist in variants.items():
                acc = hist['test_acc'][-1]
                loss = hist['test_loss'][-1]
                safe_tag = tag.replace('_', r'\_')
                lines.append(r'%s & %.4f & %.4f \\' % (safe_tag, acc, loss))
            table_footer()
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, 'w') as f:
        f.write('\n'.join(lines))

def main():
    summary = load_summary('classify/results/ablation_summary.json')
    write_tables(summary)

if __name__ == '__main__':
    main()
