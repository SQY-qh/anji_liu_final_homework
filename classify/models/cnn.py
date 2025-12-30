import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, channels=(32, 64), ksize=3, use_bn=False, dropout=0.0, in_chans=1, image_size=28, num_classes=10, vocab_size=None, embed_dim=None):
        super().__init__()
        c1, c2 = channels
        pad = ksize // 2
        if vocab_size is not None and embed_dim is not None:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            blocks = []
            blocks += [nn.Conv1d(embed_dim, c1, kernel_size=ksize, padding=pad)]
            if use_bn:
                blocks += [nn.BatchNorm1d(c1)]
            blocks += [nn.ReLU(), nn.MaxPool1d(2)]
            blocks += [nn.Conv1d(c1, c2, kernel_size=ksize, padding=pad)]
            if use_bn:
                blocks += [nn.BatchNorm1d(c2)]
            blocks += [nn.ReLU(), nn.MaxPool1d(2)]
            if dropout > 0:
                blocks += [nn.Dropout(dropout)]
            blocks += [nn.AdaptiveMaxPool1d(1)]
            self.features = nn.Sequential(*blocks)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(c2, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
            self.is_text = True
        else:
            self.embedding = None
            blocks = []
            blocks += [nn.Conv2d(in_chans, c1, kernel_size=ksize, padding=pad)]
            if use_bn:
                blocks += [nn.BatchNorm2d(c1)]
            blocks += [nn.ReLU(), nn.MaxPool2d(2)]
            blocks += [nn.Conv2d(c1, c2, kernel_size=ksize, padding=pad)]
            if use_bn:
                blocks += [nn.BatchNorm2d(c2)]
            blocks += [nn.ReLU(), nn.MaxPool2d(2)]
            if dropout > 0:
                blocks += [nn.Dropout(dropout)]
            self.features = nn.Sequential(*blocks)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(c2*((image_size//4)**2), 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
            self.is_text = False
    def forward(self, x):
        if self.is_text and x.dim() == 2 and x.dtype in (torch.long, torch.int64):
            x = self.embedding(x)
            x = x.transpose(1, 2)
            x = self.features(x)
            return self.classifier(x)
        x = self.features(x)
        return self.classifier(x)
