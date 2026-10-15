"""Fusion rules used in the paper and its controlled ablations."""

import torch


def mean_fusion(experts):
    """Deterministic mean fusion of modality latent means."""
    if not experts:
        raise ValueError("At least one expert is required.")
    mu = torch.stack([expert["mu"] for expert in experts], dim=0).mean(dim=0)
    return {"mu": mu, "variance": torch.ones_like(mu)}


def gaussian_poe(experts, reliability_weighted=False, eps=1e-6):
    """Fuse diagonal Gaussian experts with a unit-Gaussian prior.

    When ``reliability_weighted`` is true, modality precision is scaled by
    the scalar reliability score r_m, giving effective precision r_m/sigma_m^2.
    """
    if not experts:
        raise ValueError("At least one expert is required.")

    reference = experts[0]["mu"]
    precision_sum = torch.ones_like(reference)  # N(0, I) prior
    precision_mean_sum = torch.zeros_like(reference)

    for expert in experts:
        variance = torch.exp(expert["logvar"]).clamp_min(eps)
        precision = variance.reciprocal()

        if reliability_weighted:
            reliability = expert["reliability"]
            if reliability.ndim == 1:
                reliability = reliability.unsqueeze(-1)
            precision = precision * reliability.clamp(0.0, 1.0)

        precision_sum = precision_sum + precision
        precision_mean_sum = precision_mean_sum + precision * expert["mu"]

    variance = precision_sum.clamp_min(eps).reciprocal()
    mu = variance * precision_mean_sum
    return {"mu": mu, "variance": variance, "logvar": torch.log(variance.clamp_min(eps))}


def reliability_weighted_poe(experts, eps=1e-6):
    """Reliability-aware Gaussian product-of-experts used by the main method."""
    return gaussian_poe(experts, reliability_weighted=True, eps=eps)
