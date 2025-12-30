import torch
import torch.nn as nn

class MinGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.Wz = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wr = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wh = nn.Linear(input_size + hidden_size, hidden_size)
    def forward(self, x, h):
        combined = torch.cat([h, x], dim=-1)
        z = torch.sigmoid(self.Wz(combined))
        r = torch.sigmoid(self.Wr(combined))
        combined_r = torch.cat([r * h, x], dim=-1)
        h_tilde = torch.tanh(self.Wh(combined_r))
        h = (1 - z) * h + z * h_tilde
        return h

class MinGRU(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_classes=10, time_dim=2, vocab_size=None, embed_dim=None):
        super().__init__()
        if vocab_size is not None and embed_dim is not None:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.cell = MinGRUCell(embed_dim, hidden_size)
        else:
            self.embedding = None
            self.cell = MinGRUCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.time_dim = time_dim
    def forward(self, x):
        b = x.size(0)
        if self.embedding is not None and x.dim() == 2 and x.dtype in (torch.long, torch.int64):
            # token ids: (B, T)
            x = self.embedding(x)  # (B, T, embed_dim)
        # Pixel-sequence mode: 1x28x28 or 28x28 → seq_len=784, input_size=1
        if (x.dim() == 4 and x.size(1) == 1 and x.size(2) == 28 and x.size(3) == 28) or (x.dim() == 3 and x.size(1) == 28 and x.size(2) == 28):
            img = x.squeeze(1) if x.dim() == 4 else x
            seq = img.reshape(b, 28 * 28)
            h = torch.zeros(b, self.cell.hidden_size, device=x.device)
            for t in range(seq.size(1)):
                xt = seq[:, t].unsqueeze(1)
                h = self.cell(xt, h)
        elif x.dim() == 3:
            # Generic sequence mode: x shape (B, T, F)
            T, F = x.size(1), x.size(2)
            h = torch.zeros(b, self.cell.hidden_size, device=x.device)
            for t in range(T):
                xt = x[:, t, :]
                h = self.cell(xt, h)
        else:
            # Column/row scan fallback
            if x.dim() == 4:
                c, h_img, w_img = x.size(1), x.size(2), x.size(3)
                steps = w_img if self.time_dim == 2 else h_img
                h = torch.zeros(b, self.cell.hidden_size, device=x.device)
                for t in range(steps):
                    if self.time_dim == 2:
                        xt = x[:, :, :, t].reshape(b, c * h_img)
                    else:
                        xt = x[:, :, t, :].reshape(b, c * w_img)
                    h = self.cell(xt, h)
            else:
                h_img, w_img = x.size(1), x.size(2)
                steps = w_img if self.time_dim == 2 else h_img
                h = torch.zeros(b, self.cell.hidden_size, device=x.device)
                for t in range(steps):
                    xt = x[:, :, t] if self.time_dim == 2 else x[:, t, :]
                    h = self.cell(xt, h)
        return self.fc(h)
