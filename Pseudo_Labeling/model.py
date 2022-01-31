import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_channels, size):
        super(Model, self).__init__()

        self.size = size // 4

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * (self.size**2), 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
