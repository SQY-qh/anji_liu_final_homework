import os
import time
import argparse
import math

import torch
import torch.nn as nn
import torch.optim as optim

from tokenizers import BasicBPETokenizer, SentencePieceTokenizer, PAD, SOS, EOS
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
        ids = ids + [EOS]
        if len(ids) < self.max_len:
            ids = ids + [PAD] * (self.max_len - len(ids))
        return ids

    def __getitem__(self, idx):
        src_ids = self.src_tok.encode(self.src[idx])
        tgt_ids = self.tgt_tok.encode(self.tgt[idx])
        return torch.tensor(self.pad(src_ids), dtype=torch.long), torch.tensor(self.pad(tgt_ids), dtype=torch.long)

def build_tokenizers(data_dir, kind):
    src_train = load_lines(os.path.join(data_dir, "train.de"))
    tgt_train = load_lines(os.path.join(data_dir, "train.en"))
    if kind == "sentencepiece":
        sp_dir_de = os.path.join(data_dir, "spm_de")
        sp_dir_en = os.path.join(data_dir, "spm_en")
        tok_de = SentencePieceTokenizer(sp_dir_de, vocab_size=200)
        tok_en = SentencePieceTokenizer(sp_dir_en, vocab_size=200)
        if not os.path.exists(os.path.join(sp_dir_de, "spm.model")):
            tok_de.train([os.path.join(data_dir, "train.de")])
        else:
            tok_de.load()
        if not os.path.exists(os.path.join(sp_dir_en, "spm.model")):
            tok_en.train([os.path.join(data_dir, "train.en")])
        else:
            tok_en.load()
        return tok_de, tok_en
    else:
        tok_de = BasicBPETokenizer(max_vocab_size=1000)
        tok_en = BasicBPETokenizer(max_vocab_size=1000)
        tok_de.fit(src_train)
        tok_en.fit(tgt_train)
        return tok_de, tok_en

def train_one(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        optimizer.zero_grad()
        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))

def eval_one(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            total += loss.item()
    return total / max(1, len(loader))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default="transformer")
    parser.add_argument("--pos_encoding", type=str, default="sin")
    parser.add_argument("--tokenizer", type=str, default="bpe")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--quantize", type=str, default="none")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root, "data")
    device = torch.device(args.device)

    src_train = load_lines(os.path.join(data_dir, "train.de"))
    tgt_train = load_lines(os.path.join(data_dir, "train.en"))
    src_test = load_lines(os.path.join(data_dir, "test.de"))
    tgt_test = load_lines(os.path.join(data_dir, "test.en"))

    tok_kind = "sentencepiece" if args.tokenizer == "sentencepiece" else "bpe"
    tok_de, tok_en = build_tokenizers(data_dir, tok_kind)

    train_ds = ParallelDataset(src_train, tgt_train, tok_de, tok_en)
    test_ds = ParallelDataset(src_test, tgt_test, tok_de, tok_en)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size)

    if args.model == "mamba":
        model = create_mamba(tok_de.vocab_size, tok_en.vocab_size)
    elif args.model == "transformer_sp":
        model = create_transformer(tok_de.vocab_size, tok_en.vocab_size, pos_encoding="sin")
    elif args.model == "transformer_rope":
        model = create_transformer(tok_de.vocab_size, tok_en.vocab_size, pos_encoding="rope")
    else:
        model = create_transformer(tok_de.vocab_size, tok_en.vocab_size, pos_encoding=args.pos_encoding)

    model = model.to(device)

    if args.quantize == "int8" and args.device == "cpu":
        model = torch.quantization.quantize_dynamic(model, {nn.Linear, nn.GRU, nn.LSTM}, dtype=torch.qint8)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start = time.time()
    for epoch in range(args.epochs):
        train_loss = train_one(model, train_loader, optimizer, criterion, device)
    final_loss = eval_one(model, test_loader, criterion, device)
    elapsed = time.time() - start
    print("train_time", round(elapsed, 2))
    print("final_loss", round(final_loss, 4))
    if args.quantize == "int8":
        with torch.no_grad():
            for src, tgt in test_loader:
                src = src.to(device)
                tgt_in = tgt[:, :-1].to(device)
                t0 = time.time()
                _ = model(src, tgt_in)
                dt = (time.time() - t0) * 1000.0
                print("inference_latency_ms", round(dt, 2))
                break

if __name__ == "__main__":
    main()
