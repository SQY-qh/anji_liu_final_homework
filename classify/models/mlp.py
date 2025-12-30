import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, widths=(256, 128), activation='relu', input_dim=28*28, num_classes=10):
        super().__init__()
        act = nn.ReLU() if activation == 'relu' else nn.Tanh()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, widths[0]),
            act,
            nn.Linear(widths[0], widths[1]),
            act,
            nn.Linear(widths[1], num_classes)
        )
    def forward(self, x):
        x = x.float()
        return self.net(x)
