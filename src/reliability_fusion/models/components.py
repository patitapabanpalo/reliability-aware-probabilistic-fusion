"""Canonical probabilistic components, preserving final experiment parameter names."""
import torch
from torch import nn

MIN_LOGVAR = -6.0
MAX_LOGVAR = 4.0

class GaussianEvidence(nn.Module):
    """Diagonal Gaussian evidence head used in the final Symile and HAIM runs."""
    def __init__(self, input_dim=128, latent_dim=32, min_logvar=MIN_LOGVAR, max_logvar=MAX_LOGVAR):
        super().__init__()
        # Names intentionally preserve original checkpoints: mu.*, logvar.*
        self.mu = nn.Linear(input_dim, latent_dim)
        self.logvar = nn.Linear(input_dim, latent_dim)
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
    def forward(self, x):
        return {"mu": self.mu(x), "logvar": self.logvar(x).clamp(self.min_logvar, self.max_logvar)}

class ReliabilityHead(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=32):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2 * latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
    def forward(self, mu, logvar):
        return torch.sigmoid(self.network(torch.cat([mu, logvar], dim=1)))

class PredictionHead(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=32):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
    def forward(self, z):
        return self.network(z).squeeze(-1)
