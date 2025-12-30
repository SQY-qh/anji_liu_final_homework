import os
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim

from tokenizers import BasicBPETokenizer, SentencePieceTokenizer, PAD
from transformer_model import create_model as create_transformer
from mamba_model import create_model as create_mamba

def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

class ParallelDataset(torch.utils.data.Dataset):
    def __init__(self, src_lines, tgt_lines, src_tok, tgt_tok, max_len=32):
        self.src = src_lines
        self.tgt = tgt_lines
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok
        self.max_len = max_len

    def __len__(self):
        return len(self.src)

    def pad(self, ids):
        ids = ids[: self.max_len]
        ids = ids + [2]
        if len(ids) < self.max_len:
            ids = ids + [0] * (self.max_len - len(ids))
        return ids

    def __getitem__(self, idx):
        src_ids = self.src_tok.encode(self.src[idx])
        tgt_ids = self.tgt_tok.encode(self.tgt[idx])
        return torch.tensor(self.pad(src_ids), dtype=torch.long), torch.tensor(self.pad(tgt_ids), dtype=torch.long)

def run_experiment(name, device, tok_kind, pos_encoding, model_kind, data_dir, epochs=5, batch_size=8, lr=1e-4):
    src_train = load_lines(os.path.join(data_dir, "train.de"))
    tgt_train = load_lines(os.path.join(data_dir, "train.en"))
    src_test = load_lines(os.path.join(data_dir, "test.de"))
    tgt_test = load_lines(os.path.join(data_dir, "test.en"))

    if tok_kind == "sentencepiece":
        tok_de = SentencePieceTokenizer(os.path.join(data_dir, "spm_de"), vocab_size=200)
        tok_en = SentencePieceTokenizer(os.path.join(data_dir, "spm_en"), vocab_size=200)
        if not os.path.exists(os.path.join(data_dir, "spm_de", "spm.model")):
            tok_de.train([os.path.join(data_dir, "train.de")])
        else:
            tok_de.load()
        if not os.path.exists(os.path.join(data_dir, "spm_en", "spm.model")):
            tok_en.train([os.path.join(data_dir, "train.en")])
        else:
            tok_en.load()
    else:
        tok_de = BasicBPETokenizer(max_vocab_size=1000)
        tok_en = BasicBPETokenizer(max_vocab_size=1000)
        tok_de.fit(src_train)
        tok_en.fit(tgt_train)

    train_ds = ParallelDataset(src_train, tgt_train, tok_de, tok_en)
    test_ds = ParallelDataset(src_test, tgt_test, tok_de, tok_en)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)

    if model_kind == "mamba":
        model = create_mamba(tok_de.vocab_size, tok_en.vocab_size)
    else:
        model = create_transformer(tok_de.vocab_size, tok_en.vocab_size, pos_encoding=pos_encoding)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start = time.time()
    for _ in range(epochs):
        model.train()
        for src, tgt in train_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            optimizer.zero_grad()
            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()
    model.eval()
    total = 0.0
    with torch.no_grad():
        for src, tgt in test_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            total += loss.item()
    elapsed = time.time() - start
    return {"name": name, "train_time": round(elapsed, 2), "final_loss": round(total / max(1, len(test_loader)), 4)}

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root, "data")
    device = torch.device("cpu")
    results = []
    results.append(run_experiment("Transformer", device, "bpe", "sin", "transformer", data_dir))
    results.append(run_experiment("Transformer with RoPE", device, "bpe", "rope", "transformer", data_dir))
    results.append(run_experiment("Transformer with SentencePiece", device, "sentencepiece", "sin", "transformer", data_dir))
    results.append(run_experiment("Mamba Transformer", device, "bpe", "learned", "mamba", data_dir))
    out_path = os.path.join(root, "experiment_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved_results", out_path)

if __name__ == "__main__":
    main()
