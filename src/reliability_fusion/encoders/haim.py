"""Exact HAIM feature-family projection used in the final experiments."""
from torch import nn
MODALITY_DIMS = {"structured":1542, "timeseries":451, "cxr":2084, "text":2304}
class HAIMModalityEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=128):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim,256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.20), nn.Linear(256,embedding_dim), nn.LayerNorm(embedding_dim), nn.ReLU())
    def forward(self, x): return self.network(x)
