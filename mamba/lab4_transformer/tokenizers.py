import os
import re
from collections import Counter

spm = None

PAD = 0
SOS = 1
EOS = 2
UNK = 3

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-zäöüß0-9\s\-\.]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class BasicBPETokenizer:
    def __init__(self, max_vocab_size=1000):
        self.max_vocab_size = max_vocab_size
        self.stoi = {"<pad>": PAD, "<s>": SOS, "</s>": EOS, "<unk>": UNK}
        self.itos = {PAD: "<pad>", SOS: "<s>", EOS: "</s>", UNK: "<unk>"}

    def fit(self, texts):
        counter = Counter()
        for t in texts:
            t = normalize(t)
            toks = t.split(" ")
            counter.update(toks)
        most = counter.most_common(self.max_vocab_size - len(self.stoi))
        for tok, _ in most:
            if tok not in self.stoi:
                idx = len(self.stoi)
                self.stoi[tok] = idx
                self.itos[idx] = tok

    def encode(self, text, add_special=True):
        text = normalize(text)
        toks = text.split(" ") if text else []
        ids = [self.stoi.get(t, UNK) for t in toks]
        if add_special:
            return [SOS] + ids + [EOS]
        return ids

    def decode(self, ids):
        ids = [i for i in ids if i not in (PAD, SOS, EOS)]
        toks = [self.itos.get(i, "<unk>") for i in ids]
        return " ".join(toks)

    @property
    def vocab_size(self):
        return len(self.stoi)

class SentencePieceTokenizer:
    def __init__(self, model_dir, vocab_size=200):
        self.model_dir = model_dir
        self._target_vocab_size = vocab_size
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "spm.model")
        global spm
        if spm is None:
            import importlib
            spm = importlib.import_module("sentencepiece")
        self.sp = spm.SentencePieceProcessor()

    def train(self, corpus_paths):
        input_file = os.path.join(self.model_dir, "corpus.txt")
        with open(input_file, "w", encoding="utf-8") as f:
            for p in corpus_paths:
                with open(p, "r", encoding="utf-8") as fp:
                    for line in fp:
                        f.write(normalize(line.strip()) + "\n")
        spm.SentencePieceTrainer.Train(
            input=input_file,
            model_prefix=os.path.join(self.model_dir, "spm"),
            vocab_size=self._target_vocab_size,
            character_coverage=1.0,
            model_type="bpe",
        )
        self.sp.Load(self.model_path)

    def load(self):
        self.sp.Load(self.model_path)

    def encode(self, text, add_special=True):
        ids = self.sp.EncodeAsIds(normalize(text))
        if add_special:
            return [SOS] + ids + [EOS]
        return ids

    def decode(self, ids):
        ids = [i for i in ids if i not in (PAD, SOS, EOS)]
        return self.sp.DecodeIds(ids)

    @property
    def vocab_size(self):
        return self.sp.GetPieceSize() + 4
