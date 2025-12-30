import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=1, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SimpleTransformer(nn.Module):
    def __init__(self, num_classes=10, embed_dim=64, num_heads=4, depth=2, patch_size=4, dropout=0.0, in_chans=1):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size=patch_size, embed_dim=embed_dim, in_chans=in_chans)
        self.pos = PositionalEncoding(embed_dim, max_len=1024)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=128, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.fc = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)

class TSPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024, dropout=0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class SequenceTransformer(nn.Module):
    def __init__(self, input_size=9, num_classes=6, embed_dim=128, num_heads=8, depth=4, dropout=0.1, vocab_size=None):
        super().__init__()
        if vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.proj = None
        else:
            self.embedding = None
            self.proj = nn.Linear(input_size, embed_dim)
        self.pos = TSPositionalEncoding(embed_dim, max_len=1024, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*2, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.fc = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        # x: (B, seq_len, input_size) or (B, seq_len) token ids
        if self.embedding is not None and x.dim() == 2:
            x = self.embedding(x)
        else:
            x = self.proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)
