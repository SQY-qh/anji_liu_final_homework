import torch
import torch.nn as nn

class MambaModel(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=64, num_layers=2, dropout=0.1, max_len=128):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.encoder = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=num_layers, dropout=dropout, bidirectional=False)
        self.decoder = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=num_layers, dropout=dropout, bidirectional=False)
        self.out = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt):
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        s_pos = torch.arange(src.size(0), device=src.device).unsqueeze(1).expand(src.size(0), src.size(1))
        t_pos = torch.arange(tgt.size(0), device=tgt.device).unsqueeze(1).expand(tgt.size(0), tgt.size(1))
        src_in = self.src_embed(src) + self.pos_embed(s_pos)
        tgt_in = self.tgt_embed(tgt) + self.pos_embed(t_pos)
        enc_out, h = self.encoder(src_in)
        dec_out, _ = self.decoder(tgt_in, h)
        logits = self.out(dec_out)
        return logits.transpose(0, 1)

def create_model(src_vocab, tgt_vocab):
    return MambaModel(src_vocab, tgt_vocab)
