
import torch
import torch.nn as nn


class CLIPClassifier(nn.Module):
    def __init__(self, in_dim=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),   # layer 0
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),      # layer 3
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 2)         # layer 6
        )

    def forward(self, x):
        return self.net(x)