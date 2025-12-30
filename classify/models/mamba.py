import torch
import torch.nn as nn

class MambaCell(nn.Module):
    def __init__(self, input_size, hidden_size, state_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = hidden_size if state_size is None else state_size
        self.W_in = nn.Linear(input_size, hidden_size)
        self.decay = nn.Parameter(torch.randn(self.state_size))
        self.Bg = nn.Linear(hidden_size, self.state_size, bias=False)
        self.Ch = nn.Linear(self.state_size, hidden_size, bias=False)
        self.D = nn.Linear(input_size, hidden_size)
    def forward(self, x, s):
        gate = torch.nn.functional.silu(self.W_in(x))
        decay = torch.sigmoid(self.decay).unsqueeze(0)
        s = s * decay + self.Bg(gate)
        y = gate * (self.Ch(s) + self.D(x))
        return y, s

class Mamba(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_classes=10, state_size=None, vocab_size=None, embed_dim=None):
        super().__init__()
        if vocab_size is not None and embed_dim is not None:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.cell = MambaCell(embed_dim, hidden_size, state_size)
        else:
            self.embedding = None
            self.cell = MambaCell(input_size, hidden_size, state_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.state_size = hidden_size if state_size is None else state_size
    def forward(self, x):
        b = x.size(0)
        if self.embedding is not None and x.dim() == 2 and x.dtype in (torch.long, torch.int64):
            x = self.embedding(x)
        s = torch.zeros(b, self.state_size, device=x.device)
        h = None
        # Pixel-sequence mode: 1x28x28 or 28x28 → seq_len=784, input_size=1
        if (x.dim() == 4 and x.size(1) == 1 and x.size(2) == 28 and x.size(3) == 28) or (x.dim() == 3 and x.size(1) == 28 and x.size(2) == 28):
            img = x.squeeze(1) if x.dim() == 4 else x
            seq = img.reshape(b, 28 * 28)
            for t in range(seq.size(1)):
                xt = seq[:, t].unsqueeze(1)
                h, s = self.cell(xt, s)
        elif x.dim() == 3:
            # Generic sequence mode: (B, T, F)
            T = x.size(1)
            for t in range(T):
                xt = x[:, t, :]
                h, s = self.cell(xt, s)
        else:
            if x.dim() == 4:
                c, h_img, w_img = x.size(1), x.size(2), x.size(3)
                steps = w_img
                for t in range(steps):
                    xt = x[:, :, :, t].reshape(b, c * h_img)
                    h, s = self.cell(xt, s)
            else:
                steps = x.size(2)
                for t in range(steps):
                    h, s = self.cell(x[:, :, t], s)
        return self.fc(h)
