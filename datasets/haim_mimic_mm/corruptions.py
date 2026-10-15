"""Representation-space corruptions used in the HAIM robustness evaluation."""
import torch

CORRUPTION_LEVELS = {
    "clean": {"severity": 0, "mask_prob": 0.00, "noise_std": 0.00},
    "mild": {"severity": 1, "mask_prob": 0.10, "noise_std": 0.10},
    "moderate": {"severity": 2, "mask_prob": 0.30, "noise_std": 0.30},
    "severe": {"severity": 3, "mask_prob": 0.50, "noise_std": 0.50},
}


def corrupt_tensor(x, mask_prob, noise_std, generator=None):
    """Mask features and add Gaussian noise to a standardized representation."""
    if mask_prob == 0.0 and noise_std == 0.0:
        return x

    corrupted = x.clone()
    if mask_prob > 0:
        keep = torch.rand(
            x.shape, device=x.device, dtype=x.dtype, generator=generator
        ) >= mask_prob
        corrupted = corrupted * keep.to(x.dtype)

    if noise_std > 0:
        noise = torch.randn(
            x.shape, device=x.device, dtype=x.dtype, generator=generator
        )
        corrupted = corrupted + noise_std * noise

    return corrupted
