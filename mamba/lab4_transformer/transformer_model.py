import math
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)

def rotate_half(x):
    x1 = x[..., : x.size(-1) // 2]
    x2 = x[..., x.size(-1) // 2 :]
    return torch.cat([-x2, x1], dim=-1)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        t = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos().unsqueeze(1))
        self.register_buffer("sin", emb.sin().unsqueeze(1))

    def forward(self, x):
        l = x.size(0)
        cos = self.cos[:l]
        sin = self.sin[:l]
        return x * cos + rotate_half(x) * sin

class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, pos_encoding="sin", max_len=128):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.generator = nn.Linear(d_model, tgt_vocab)
        self.pos_encoding_type = pos_encoding
        if pos_encoding == "sin":
            self.pe = SinusoidalPositionalEncoding(d_model, dropout, max_len)
        elif pos_encoding == "rope":
            self.rope = RotaryPositionalEmbedding(d_model, max_len)
        else:
            self.pe = None

    def forward(self, src, tgt):
        src_pad_mask = src.eq(0)
        tgt_pad_mask = tgt.eq(0)
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        src_emb = self.src_embed(src)
        tgt_emb = self.tgt_embed(tgt)
        if self.pos_encoding_type == "sin":
            src_emb = self.pe(src_emb)
            tgt_emb = self.pe(tgt_emb)
        elif self.pos_encoding_type == "rope":
            src_emb = self.rope(src_emb)
            tgt_emb = self.rope(tgt_emb)
        src_key_padding_mask = src_pad_mask
        tgt_key_padding_mask = tgt_pad_mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(0)).to(tgt_emb.device)
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        logits = self.generator(out)
        return logits.transpose(0, 1)

def create_model(src_vocab, tgt_vocab, pos_encoding="sin"):
    return Seq2SeqTransformer(src_vocab=src_vocab, tgt_vocab=tgt_vocab, pos_encoding=pos_encoding)
