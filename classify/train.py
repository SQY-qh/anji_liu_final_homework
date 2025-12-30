import torch
import torch.nn as nn
import torch.optim as optim
from classify.data import get_loaders
from classify.models.mlp import MLP
from classify.models.cnn import CNN
from classify.models.mingru import MinGRU
from classify.models.mamba import Mamba
from classify.models.transformer import SimpleTransformer
from classify.models.transformer import SequenceTransformer
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def get_model(name, params=None):
    params = params or {}
    if name == 'mlp':
        return MLP(**params)
    if name == 'cnn':
        return CNN(**params)
    if name == 'mingru':
        return MinGRU(**params)
    if name == 'mamba':
        return Mamba(**params)
    if name == 'transformer':
        if params.get('seq', False):
            p = dict(params)
            p.pop('seq', None)
            return SequenceTransformer(**p)
        return SimpleTransformer(**params)
    raise ValueError('unknown model')

def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total

def train_one(name, epochs=3, batch_size=64, lr=1e-3, device='cpu', params=None, tag=None, loss_ylim=(0.0, 1.0), acc_ylim=(0.0, 1.0), dataset='mnist'):
    train_loader, test_loader = get_loaders(batch_size=batch_size, dataset=dataset)
    model = get_model(name, params).to(device)
    criterion = nn.CrossEntropyLoss()
    if name == 'transformer':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = None
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    pbar_epochs = tqdm(range(1, epochs+1), desc=f'{name}{"-"+tag if tag else ""} epochs')
    for epoch in pbar_epochs:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar_batches = tqdm(train_loader, desc=f'epoch {epoch} train', leave=False)
        for x, y in pbar_batches:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if dataset in ('seqmnist','synthetic_ts','har','ecg200','sms_spam','longdep'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        pbar_epochs.set_postfix(train_loss=f'{train_loss:.4f}', test_loss=f'{test_loss:.4f}', train_acc=f'{train_acc:.4f}', test_acc=f'{test_acc:.4f}')
        print(f'{dataset} {name}{"-"+tag if tag else "-base"} epoch {epoch}: train_loss={train_loss:.4f} test_loss={test_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}')
        if scheduler is not None:
            scheduler.step()
    os.makedirs('classify/results', exist_ok=True)
    epochs_range = range(1, epochs+1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, history['train_loss'], label='train')
    plt.plot(epochs_range, history['test_loss'], label='test')
    plt.title(f'{name}{"-"+tag if tag else ""} loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    plt.ylim(*loss_ylim)
    plt.subplot(1,2,2)
    plt.plot(epochs_range, history['train_acc'], label='train')
    plt.plot(epochs_range, history['test_acc'], label='test')
    plt.title(f'{name}{"-"+tag if tag else ""} acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend()
    plt.grid(True)
    plt.ylim(*acc_ylim)
    plt.tight_layout()
    fname = f'{dataset}_{name}{"_"+tag if tag else ""}_curves.png'
    plt.savefig(f'classify/results/{fname}')
    return history

def run_all(models, epochs=3, batch_size=64, lr=1e-3, device='cpu', dataset='mnist'):
    results = {}
    for m in models:
        results[m] = train_one(m, epochs=epochs, batch_size=batch_size, lr=lr, device=device, dataset=dataset)
    return results

def _curves_path(dataset, name, tag):
    return f'classify/results/{dataset}_{name}{"_"+tag if tag else ""}_curves.png'

def _load_existing_summary():
    import json
    try:
        with open('classify/results/ablation_summary.json', 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def _save_summary(all_summaries):
    import json
    os.makedirs('classify/results', exist_ok=True)
    with open('classify/results/ablation_summary.json', 'w') as f:
        json.dump(all_summaries, f)

def run_ablation(epochs=8, batch_size=64, lr=1e-3, device='cpu', dataset='mnist', resume=True):
    # dataset defaults
    if dataset == 'mnist' or dataset == 'seqmnist':
        in_chans = 1
        image_size = 28
        input_dim = 28*28
    elif dataset == 'cifar10':
        in_chans = 3
        image_size = 32
        input_dim = 32*32*3
    elif dataset == 'har':
        in_chans = None
        image_size = None
        input_dim = 128*9
    elif dataset == 'synthetic_ts':
        in_chans = None
        image_size = None
        input_dim = 128*3
    elif dataset == 'ecg200':
        in_chans = None
        image_size = None
        input_dim = 200*1
    elif dataset == 'gunpoint':
        in_chans = None
        image_size = None
        input_dim = 150*1
    elif dataset == 'longdep':
        in_chans = None
        image_size = None
        input_dim = 200*2
    elif dataset == 'sms_spam':
        in_chans = None
        image_size = None
        input_dim = 50
    else:
        raise ValueError('unknown dataset')
    # derived sizes for RNN/SSM and transformer
    if dataset == 'seqmnist':
        rnn_input_size = 1
        mamba_state = 256
    elif dataset == 'mnist':
        rnn_input_size = 1
        mamba_state = 128
    elif dataset == 'har':
        rnn_input_size = 9
        mamba_state = 128
    elif dataset == 'synthetic_ts':
        rnn_input_size = 3
        mamba_state = 128
    elif dataset == 'ecg200':
        rnn_input_size = 1
        mamba_state = 128
    elif dataset == 'gunpoint':
        rnn_input_size = 1
        mamba_state = 128
    elif dataset == 'longdep':
        rnn_input_size = 2
        mamba_state = 128
    elif dataset == 'sms_spam':
        rnn_input_size = 64
        mamba_state = 128
    else:
        rnn_input_size = in_chans * image_size
        mamba_state = 128
    # number of classes
    if dataset in ('ecg200', 'gunpoint', 'longdep', 'sms_spam'):
        num_classes = 2
    elif dataset == 'synthetic_ts':
        num_classes = 3
    elif dataset == 'har':
        num_classes = 6
    else:
        num_classes = 10
    configs = {
        'mlp': {
            'base': {'widths': (256, 128), 'activation': 'relu', 'input_dim': input_dim, 'num_classes': num_classes},
            'wide': {'widths': (512, 256), 'activation': 'relu', 'input_dim': input_dim, 'num_classes': num_classes},
            'tanh': {'widths': (256, 128), 'activation': 'tanh', 'input_dim': input_dim, 'num_classes': num_classes}
        }
    }
    if image_size is not None:
        configs['cnn'] = {
            'base': {'channels': (32, 64), 'ksize': 3, 'use_bn': False, 'dropout': 0.0, 'in_chans': in_chans, 'image_size': image_size},
            'wide_bn': {'channels': (64, 128), 'ksize': 3, 'use_bn': True, 'dropout': 0.0, 'in_chans': in_chans, 'image_size': image_size},
            'k5_do': {'channels': (32, 64), 'ksize': 5, 'use_bn': False, 'dropout': 0.5, 'in_chans': in_chans, 'image_size': image_size}
        }
    elif dataset == 'sms_spam':
        configs['cnn'] = {
            'base': {'channels': (64, 128), 'ksize': 3, 'use_bn': True, 'dropout': 0.1, 'vocab_size': 10002, 'embed_dim': 128, 'num_classes': num_classes},
            'wide_bn': {'channels': (128, 256), 'ksize': 3, 'use_bn': True, 'dropout': 0.1, 'vocab_size': 10002, 'embed_dim': 128, 'num_classes': num_classes},
            'k5_do': {'channels': (64, 128), 'ksize': 5, 'use_bn': False, 'dropout': 0.3, 'vocab_size': 10002, 'embed_dim': 128, 'num_classes': num_classes}
        }
    configs['mingru'] = {
        'base': {'input_size': rnn_input_size, 'hidden_size': 256 if dataset == 'seqmnist' else 128, 'time_dim': 2, 'num_classes': num_classes},
        'hidden256': {'input_size': rnn_input_size, 'hidden_size': 512 if dataset == 'seqmnist' else 256, 'time_dim': 2, 'num_classes': num_classes},
        'time_dim1': {'input_size': rnn_input_size, 'hidden_size': 256 if dataset == 'seqmnist' else 128, 'time_dim': 1, 'num_classes': num_classes}
    }
    configs['mamba'] = {
        'base': {'input_size': rnn_input_size, 'hidden_size': 256 if dataset == 'seqmnist' else 128, 'state_size': mamba_state, 'num_classes': num_classes},
        'bigger_state': {'input_size': rnn_input_size, 'hidden_size': 256 if dataset == 'seqmnist' else 128, 'state_size': max(mamba_state*2, 256), 'num_classes': num_classes},
        'wider': {'input_size': rnn_input_size, 'hidden_size': 512 if dataset == 'seqmnist' else 256, 'state_size': mamba_state, 'num_classes': num_classes}
    }
    if dataset == 'seqmnist':
        configs['transformer'] = {
            'base': {'embed_dim': 128, 'num_heads': 8, 'depth': 4, 'patch_size': 1, 'dropout': 0.1, 'in_chans': 1}
        }
    elif dataset in ('har','synthetic_ts','ecg200', 'longdep', 'sms_spam'):
        t_num_classes = 6 if dataset == 'har' else (3 if dataset == 'synthetic_ts' else 2)
        configs['transformer'] = {
            'base': {'input_size': rnn_input_size, 'num_classes': t_num_classes, 'embed_dim': 128, 'num_heads': 8, 'depth': 4, 'dropout': 0.1, 'seq': True},
            'larger': {'input_size': rnn_input_size, 'num_classes': t_num_classes, 'embed_dim': 256, 'num_heads': 8, 'depth': 6, 'dropout': 0.1, 'seq': True},
            'more_tokens': {'input_size': rnn_input_size, 'num_classes': t_num_classes, 'embed_dim': 128, 'num_heads': 8, 'depth': 6, 'dropout': 0.1, 'seq': True}
        }
        if dataset == 'sms_spam':
            for k in configs['transformer'].keys():
                configs['transformer'][k]['vocab_size'] = 10002
            for k in configs['mingru'].keys():
                configs['mingru'][k]['vocab_size'] = 10002
                configs['mingru'][k]['embed_dim'] = 128
            for k in configs['mamba'].keys():
                configs['mamba'][k]['vocab_size'] = 10002
                configs['mamba'][k]['embed_dim'] = 128
    elif dataset == 'gunpoint':
        num_classes = 2
        configs['transformer'] = {
            'base': {'input_size': rnn_input_size, 'num_classes': num_classes, 'embed_dim': 128, 'num_heads': 8, 'depth': 4, 'dropout': 0.1, 'seq': True},
            'larger': {'input_size': rnn_input_size, 'num_classes': num_classes, 'embed_dim': 256, 'num_heads': 8, 'depth': 6, 'dropout': 0.1, 'seq': True},
            'more_tokens': {'input_size': rnn_input_size, 'num_classes': num_classes, 'embed_dim': 128, 'num_heads': 8, 'depth': 6, 'dropout': 0.1, 'seq': True}
        }
    else:
        configs['transformer'] = {
            'base': {'embed_dim': 64, 'num_heads': 4, 'depth': 2, 'patch_size': 4, 'dropout': 0.0, 'in_chans': in_chans},
            'larger': {'embed_dim': 128, 'num_heads': 8, 'depth': 4, 'patch_size': 4, 'dropout': 0.1, 'in_chans': in_chans},
            'more_tokens': {'embed_dim': 64, 'num_heads': 4, 'depth': 2, 'patch_size': 2, 'dropout': 0.0, 'in_chans': in_chans}
        }
    summaries = {}
    existing = _load_existing_summary() if resume else {}
    for name, variants in configs.items():
        summaries[name] = {}
        for tag, params in variants.items():
            curves_path = _curves_path(dataset, name, tag)
            if resume and os.path.exists(curves_path) and existing.get(dataset, {}).get(name, {}).get(tag):
                hist = existing[dataset][name][tag]
            else:
                hist = train_one(name, epochs=epochs, batch_size=batch_size, lr=lr, device=device, params=params, tag=tag, loss_ylim=(0.0, 1.0), acc_ylim=(0.0, 1.0), dataset=dataset)
                # save partial summary incrementally
                partial = existing if resume else {}
                if dataset not in partial:
                    partial[dataset] = {}
                if name not in partial[dataset]:
                    partial[dataset][name] = {}
                partial[dataset][name][tag] = hist
                _save_summary(partial)
            summaries[name][tag] = hist
    os.makedirs('classify/results', exist_ok=True)
    # Cross-model comparison using base variant
    plt.figure(figsize=(8,5))
    for name in configs.keys():
        acc = summaries[name]['base']['test_acc']
        plt.plot(range(1, epochs+1), acc, label=name)
    plt.title(f'{dataset} Cross-model Test Accuracy (base)')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(f'classify/results/{dataset}_cross_model_test_acc.png')
    plt.figure(figsize=(8,5))
    for name in configs.keys():
        loss = summaries[name]['base']['test_loss']
        plt.plot(range(1, epochs+1), loss, label=name)
    plt.title(f'{dataset} Cross-model Test Loss (base)')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(f'classify/results/{dataset}_cross_model_test_loss.png')
    # Per-model ablation comparison
    for name, variants in configs.items():
        plt.figure(figsize=(8,5))
        for tag in variants.keys():
            acc = summaries[name][tag]['test_acc']
            plt.plot(range(1, epochs+1), acc, label=tag)
        plt.title(f'{dataset} {name} Ablation Test Accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.grid(True)
        plt.ylim(0.0, 1.0)
        plt.tight_layout()
        plt.savefig(f'classify/results/{dataset}_{name}_ablation_test_acc.png')
        plt.figure(figsize=(8,5))
        for tag in variants.keys():
            loss = summaries[name][tag]['test_loss']
            plt.plot(range(1, epochs+1), loss, label=tag)
        plt.title(f'{dataset} {name} Ablation Test Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.grid(True)
        plt.ylim(0.0, 1.0)
        plt.tight_layout()
        plt.savefig(f'classify/results/{dataset}_{name}_ablation_test_loss.png')
    return summaries
