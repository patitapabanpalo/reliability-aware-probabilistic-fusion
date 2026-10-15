"""Exact encoder architectures from the final Symile-MIMIC training script."""
import torch
from torch import nn

class CXREncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,7,stride=2,padding=3), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(3,stride=2,padding=1),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)))
        self.projection = nn.Linear(128, embedding_dim)
    def forward(self, x):
        return self.projection(torch.flatten(self.features(x), start_dim=1))

class ECGEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(12,32,15,stride=2,padding=7), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32,64,9,stride=2,padding=4), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64,128,7,stride=2,padding=3), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.projection = nn.Linear(128, embedding_dim)
    def forward(self, x):
        x = x.squeeze(1).transpose(1,2)
        return self.projection(torch.flatten(self.features(x), start_dim=1))

class LabEncoder(nn.Module):
    def __init__(self, input_dim=100, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, embedding_dim))
    def forward(self, x):
        return self.net(x)
