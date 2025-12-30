import torch
import os
import tarfile
import urllib.request
import ssl
import requests

_perm = torch.randperm(28 * 28)

def _permute_img_tensor(t):
    # t: (1, 28, 28) or (28, 28)
    if t.dim() == 3:
        t = t.squeeze(0)
    b = t.view(28 * 28)
    b = b[_perm]
    out = b.view(28, 28).unsqueeze(0)
    return out

def prefetch_datasets(datasets):
    ds_list = datasets if isinstance(datasets, (list, tuple)) else str(datasets).split(',')
    ds_list = [ds.strip() for ds in ds_list if ds.strip()]
    for ds in ds_list:
        try:
            print(f'Prefetching {ds}...')
            if ds == 'synthetic_ts':
                print('Synthetic dataset does not require download')
                continue
            # Trigger dataset initialization to ensure files are present (download if needed)
            _train_loader, _test_loader = get_loaders(batch_size=1, dataset=ds)
        except Exception as e:
            print(f'Prefetch {ds} failed: {e}')

def get_loaders(batch_size=64, dataset='mnist'):
    if dataset == 'mnist':
        from torchvision import datasets, transforms
        def ensure_mnist(root='./data'):
            raw_dir = os.path.join(root, 'MNIST', 'raw')
            os.makedirs(raw_dir, exist_ok=True)
            files = {
                'train-images-idx3-ubyte.gz': ['https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'],
                'train-labels-idx1-ubyte.gz': ['https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'],
                't10k-images-idx3-ubyte.gz': ['https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'],
                't10k-labels-idx1-ubyte.gz': ['https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']
            }
            ctx = ssl.create_default_context()
            for fname, mirrors in files.items():
                path = os.path.join(raw_dir, fname)
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    continue
                for url in mirrors:
                    try:
                        print(f'Downloading MNIST {fname} from {url}')
                        with urllib.request.urlopen(url, context=ctx, timeout=60) as resp:
                            with open(path, 'wb') as f:
                                while True:
                                    chunk = resp.read(1024 * 1024)
                                    if not chunk:
                                        break
                                    f.write(chunk)
                        break
                    except Exception as e:
                        print(f'Failed {fname} from {url}: {e}')
            return raw_dir

        ensure_mnist('./data')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'cifar10':
        from torchvision import datasets, transforms
        def ensure_cifar10(root='./data'):
            target = os.path.join(root, 'cifar-10-batches-py')
            if os.path.isdir(target):
                return
            os.makedirs(root, exist_ok=True)
            tar_path = os.path.join(root, 'cifar-10-python.tar.gz')
            urls = [
                'https://ossci-datasets.s3.amazonaws.com/cifar-10-python.tar.gz',
                'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
            ]
            ctx = ssl.create_default_context()
            for url in urls:
                try:
                    print(f'Downloading CIFAR-10 from {url}')
                    with urllib.request.urlopen(url, context=ctx, timeout=60) as resp:
                        with open(tar_path, 'wb') as f:
                            while True:
                                chunk = resp.read(1024 * 1024)
                                if not chunk:
                                    break
                                f.write(chunk)
                    print('Extracting CIFAR-10...')
                    with tarfile.open(tar_path, 'r:gz') as tf:
                        tf.extractall(root)
                    break
                except Exception as e:
                    print(f'Failed to fetch {url}: {e}')
            if not os.path.isdir(target):
                raise RuntimeError('CIFAR-10 download failed; please check network or download manually to ./data')

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        ensure_cifar10('./data')
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    elif dataset == 'seqmnist':
        from torchvision import datasets, transforms
        # Permuted Sequential MNIST: apply a fixed permutation to pixels, then reshape back to image
        def ensure_mnist(root='./data'):
            raw_dir = os.path.join(root, 'MNIST', 'raw')
            os.makedirs(raw_dir, exist_ok=True)
            files = {
                'train-images-idx3-ubyte.gz': ['https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'],
                'train-labels-idx1-ubyte.gz': ['https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'],
                't10k-images-idx3-ubyte.gz': ['https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'],
                't10k-labels-idx1-ubyte.gz': ['https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']
            }
            ctx = ssl.create_default_context()
            for fname, mirrors in files.items():
                path = os.path.join(raw_dir, fname)
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    continue
                for url in mirrors:
                    try:
                        print(f'Downloading MNIST {fname} from {url}')
                        with urllib.request.urlopen(url, context=ctx, timeout=60) as resp:
                            with open(path, 'wb') as f:
                                while True:
                                    chunk = resp.read(1024 * 1024)
                                    if not chunk:
                                        break
                                    f.write(chunk)
                        break
                    except Exception as e:
                        print(f'Failed {fname} from {url}: {e}')
            return raw_dir

        ensure_mnist('./data')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(_permute_img_tensor),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'har':
        def ensure_har(root='./data'):
            root_dir = os.path.join(root, 'UCI HAR Dataset')
            if os.path.isdir(root_dir):
                return root_dir
            os.makedirs(root, exist_ok=True)
            urls = [
                'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip',
                'https://raw.githubusercontent.com/guillaume-chevalier/HAR_dataset/master/UCI%20HAR%20Dataset.zip'
            ]
            zip_path = os.path.join(root, 'UCI_HAR_Dataset.zip')
            for url in urls:
                try:
                    print(f'Downloading HAR from {url}')
                    with requests.get(url, stream=True, timeout=60) as r:
                        r.raise_for_status()
                        # basic content-type check
                        ctype = r.headers.get('Content-Type', '')
                        with open(zip_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    f.write(chunk)
                    # verify zip signature
                    import zipfile
                    if not zipfile.is_zipfile(zip_path):
                        print('Downloaded file is not a valid zip, trying next mirror')
                        continue
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extractall(root)
                    break
                except Exception as e:
                    print(f'Failed HAR fetch {url}: {e}')
            if not os.path.isdir(root_dir):
                raise RuntimeError('HAR download failed')
            return root_dir

        har_dir = ensure_har('./data')
        import numpy as np
        def load_split(split):
            sig_dir = os.path.join(har_dir, split, 'Inertial Signals')
            files = [
                f'body_acc_x_{split}.txt', f'body_acc_y_{split}.txt', f'body_acc_z_{split}.txt',
                f'body_gyro_x_{split}.txt', f'body_gyro_y_{split}.txt', f'body_gyro_z_{split}.txt',
                f'total_acc_x_{split}.txt', f'total_acc_y_{split}.txt', f'total_acc_z_{split}.txt'
            ]
            mats = [np.loadtxt(os.path.join(sig_dir, fn)) for fn in files]
            # mats: list of (n_samples, 128)
            X = np.stack(mats, axis=-1)  # (n_samples, 128, 9)
            y = np.loadtxt(os.path.join(har_dir, split, f'y_{split}.txt')).astype(np.int64) - 1  # 0..5
            return X, y
        X_train, y_train = load_split('train')
        X_test, y_test = load_split('test')
        # normalize by train stats
        mean = X_train.mean(axis=(0,1), keepdims=True)
        std = X_train.std(axis=(0,1), keepdims=True) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        train_t = torch.tensor(X_train, dtype=torch.float32)
        test_t = torch.tensor(X_test, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        y_test_t = torch.tensor(y_test, dtype=torch.long)
        train_dataset = torch.utils.data.TensorDataset(train_t, y_train_t)
        test_dataset = torch.utils.data.TensorDataset(test_t, y_test_t)
    elif dataset == 'synthetic_ts':
        import numpy as np
        def make_split(n_samples):
            T = 128
            F = 3
            X = np.zeros((n_samples, T, F), dtype=np.float32)
            y = np.zeros((n_samples,), dtype=np.int64)
            for i in range(n_samples):
                cls = np.random.randint(0, 3)
                y[i] = cls
                t = np.linspace(0, 1, T)
                if cls == 0:
                    X[i, :, 0] = np.sin(2 * np.pi * (3 + np.random.rand()*2) * t)
                    X[i, :, 1] = np.sin(2 * np.pi * (5 + np.random.rand()*2) * t + 0.5)
                    X[i, :, 2] = np.sin(2 * np.pi * (7 + np.random.rand()*2) * t + 1.0)
                elif cls == 1:
                    X[i, :, 0] = np.sign(np.sin(2 * np.pi * (3 + np.random.rand()*2) * t))
                    X[i, :, 1] = np.sign(np.sin(2 * np.pi * (5 + np.random.rand()*2) * t + 0.5))
                    X[i, :, 2] = np.sign(np.sin(2 * np.pi * (7 + np.random.rand()*2) * t + 1.0))
                else:
                    X[i, :, 0] = 2*(t - np.floor(t + 0.5))
                    X[i, :, 1] = 2*(t*2 - np.floor(t*2 + 0.5))
                    X[i, :, 2] = 2*(t*3 - np.floor(t*3 + 0.5))
                X[i] += 0.1 * np.random.randn(T, F).astype(np.float32)
            mean = X.mean(axis=(0,1), keepdims=True)
            std = X.std(axis=(0,1), keepdims=True) + 1e-8
            X = (X - mean) / std
            return X, y
        X_train, y_train = make_split(2000)
        X_test, y_test = make_split(500)
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    elif dataset == 'ecg200':
        import numpy as np
        import requests
        def ensure_ecg200(root='./data'):
            base_dir = os.path.join(root, 'ECG200')
            os.makedirs(base_dir, exist_ok=True)
            files = {
                'ECG200_TRAIN.txt': [
                    'https://raw.githubusercontent.com/hfawaz/UCR-Timeseries-Classification-Deep-Learning/master/datasets/ECG200/ECG200_TRAIN.txt',
                    'http://www.timeseriesclassification.com/Downloads/ECG200/ECG200_TRAIN.txt'
                ],
                'ECG200_TEST.txt': [
                    'https://raw.githubusercontent.com/hfawaz/UCR-Timeseries-Classification-Deep-Learning/master/datasets/ECG200/ECG200_TEST.txt',
                    'http://www.timeseriesclassification.com/Downloads/ECG200/ECG200_TEST.txt'
                ]
            }
            for fname, urls in files.items():
                fpath = os.path.join(base_dir, fname)
                if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
                    continue
                ok = False
                for url in urls:
                    try:
                        print(f'Downloading {fname} from {url}')
                        r = requests.get(url, timeout=60)
                        r.raise_for_status()
                        with open(fpath, 'wb') as f:
                            f.write(r.content)
                        ok = True
                        break
                    except Exception as e:
                        print(f'Failed {url}: {e}')
                if not ok:
                    raise RuntimeError(f'ECG200 file {fname} download failed')
            return os.path.join(base_dir, 'ECG200_TRAIN.txt'), os.path.join(base_dir, 'ECG200_TEST.txt')

        train_txt, test_txt = ensure_ecg200('./data')
        def load_txt(path):
            rows = []
            with open(path, 'r') as f:
                for line in f:
                    toks = line.strip().split(',')
                    if len(toks) <= 1:
                        toks = line.strip().split()  # fallback space-separated
                    rows.append([float(t) for t in toks])
            arr = np.array(rows, dtype=np.float32)
            y = arr[:, 0].astype(np.int64)
            # labels are -1 and 1 → map to 0/1
            y = ((y + 1) // 2).astype(np.int64)
            X = arr[:, 1:]
            X = X.reshape(X.shape[0], X.shape[1], 1)
            return X, y
        X_train, y_train = load_txt(train_txt)
        X_test, y_test = load_txt(test_txt)
        mean = X_train.mean(axis=(0,1), keepdims=True)
        std = X_train.std(axis=(0,1), keepdims=True) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    elif dataset == 'gunpoint':
        import numpy as np
        import requests
        def ensure_gunpoint(root='./data'):
            base_dir = os.path.join(root, 'GunPoint')
            os.makedirs(base_dir, exist_ok=True)
            files = {
                'GunPoint_TRAIN.txt': [
                    'https://raw.githubusercontent.com/hfawaz/UCR-Timeseries-Classification-Deep-Learning/master/datasets/GunPoint/GunPoint_TRAIN.txt'
                ],
                'GunPoint_TEST.txt': [
                    'https://raw.githubusercontent.com/hfawaz/UCR-Timeseries-Classification-Deep-Learning/master/datasets/GunPoint/GunPoint_TEST.txt'
                ]
            }
            for fname, urls in files.items():
                fpath = os.path.join(base_dir, fname)
                if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
                    continue
                ok = False
                for url in urls:
                    try:
                        print(f'Downloading {fname} from {url}')
                        r = requests.get(url, timeout=60)
                        r.raise_for_status()
                        with open(fpath, 'wb') as f:
                            f.write(r.content)
                        ok = True
                        break
                    except Exception as e:
                        print(f'Failed {url}: {e}')
                if not ok:
                    raise RuntimeError(f'GunPoint file {fname} download failed')
            return os.path.join(base_dir, 'GunPoint_TRAIN.txt'), os.path.join(base_dir, 'GunPoint_TEST.txt')

        train_txt, test_txt = ensure_gunpoint('./data')
        def load_txt(path):
            rows = []
            with open(path, 'r') as f:
                for line in f:
                    toks = line.strip().split(',')
                    if len(toks) <= 1:
                        toks = line.strip().split()
                    rows.append([float(t) for t in toks])
            arr = np.array(rows, dtype=np.float32)
            y = arr[:, 0].astype(np.int64)
            y = ((y + 1) // 2).astype(np.int64)
            X = arr[:, 1:]
            X = X.reshape(X.shape[0], X.shape[1], 1)
            return X, y
        X_train, y_train = load_txt(train_txt)
        X_test, y_test = load_txt(test_txt)
        mean = X_train.mean(axis=(0,1), keepdims=True)
        std = X_train.std(axis=(0,1), keepdims=True) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    elif dataset == 'longdep':
        import numpy as np
        def make_split(n_samples, T=200, F=2):
            X = np.random.randn(n_samples, T, F).astype(np.float32) * 0.1
            y = np.zeros((n_samples,), dtype=np.int64)
            for i in range(n_samples):
                p1 = np.random.randint(10, T//2 - 10)
                p2 = np.random.randint(T//2 + 10, T - 10)
                b1 = np.random.randint(0, 2)
                b2 = np.random.randint(0, 2)
                # feature 0: signal value (0/1), feature 1: marker flag
                X[i, p1, 0] = float(b1)
                X[i, p1, 1] = 1.0
                X[i, p2, 0] = float(b2)
                X[i, p2, 1] = 1.0
                y[i] = (b1 ^ b2)
            # normalize signal feature across train
            return X, y
        X_train, y_train = make_split(4000)
        X_test, y_test = make_split(1000)
        # standardize using train stats
        mean = X_train.mean(axis=(0,1), keepdims=True)
        std = X_train.std(axis=(0,1), keepdims=True) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    elif dataset == 'sms_spam':
        import numpy as np
        import zipfile
        def ensure_sms(root='./data'):
            import requests
            base = os.path.join(root, 'sms_spam')
            os.makedirs(base, exist_ok=True)
            txt_path = os.path.join(base, 'SMSSpamCollection')
            if not os.path.exists(txt_path):
                url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
                zip_path = os.path.join(base, 'smsspamcollection.zip')
                print(f'Downloading SMS Spam Collection from {url}')
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                with open(zip_path, 'wb') as f:
                    f.write(r.content)
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(base)
            return txt_path
        txt_path = ensure_sms('./data')
        # Build word vocab (top-K)
        import re
        max_len = 50
        top_k = 10000
        texts = []
        labels = []
        with open(txt_path, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue
                label, text = parts
                labels.append(1 if label.lower()=='spam' else 0)
                texts.append(text.lower())
        # shuffle
        idxs = np.arange(len(texts))
        np.random.seed(42)
        np.random.shuffle(idxs)
        texts = [texts[i] for i in idxs]
        labels = np.array([labels[i] for i in idxs], dtype=np.int64)
        # tokenize
        tokens_list = [re.findall(r"[a-z0-9]+", t) for t in texts]
        # build vocab
        from collections import Counter
        counter = Counter([tok for toks in tokens_list for tok in toks])
        vocab = [tok for tok,_ in counter.most_common(top_k)]
        PAD = 0
        UNK = 1
        tok2id = {tok:i+2 for i,tok in enumerate(vocab)}
        def toks_to_ids(toks):
            ids = np.full((max_len,), PAD, dtype=np.int64)
            j = 0
            for tok in toks:
                if j >= max_len:
                    break
                ids[j] = tok2id.get(tok, UNK)
                j += 1
            return ids
        X = np.stack([toks_to_ids(toks) for toks in tokens_list], axis=0)
        n_train = int(0.8*X.shape[0])
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = labels[:n_train], labels[n_train:]
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long))
    else:
        raise ValueError('unknown dataset')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader
