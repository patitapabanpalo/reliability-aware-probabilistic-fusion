"""Input-space corruptions used for the Symile-MIMIC robustness experiments."""
import torch
import torch.nn.functional as F

SEVERITIES = (0, 1, 2, 3)


def corrupt_cxr(x, severity):
    if severity == 0:
        return x
    kernel = {1: 3, 2: 7, 3: 13}[severity]
    strength = {1: 0.25, 2: 0.55, 3: 0.85}[severity]
    noise = {1: 0.02, 2: 0.05, 3: 0.10}[severity]
    blurred = F.avg_pool2d(x, kernel_size=kernel, stride=1, padding=kernel // 2)
    return (1.0 - strength) * x + strength * blurred + noise * torch.randn_like(x)


def corrupt_ecg(x, severity):
    if severity == 0:
        return x
    strength = {1: 0.05, 2: 0.15, 3: 0.30}[severity]
    return (x + strength * torch.randn_like(x)).clamp(-1.0, 1.0)


def corrupt_labs(x, severity):
    if severity == 0:
        return x
    alpha = {1: 0.25, 2: 0.60, 3: 1.00}[severity]
    return (1.0 - alpha) * x + alpha * torch.rand_like(x)


def corrupt_input(x, modality, severity):
    functions = {"cxr": corrupt_cxr, "ecg": corrupt_ecg, "labs": corrupt_labs}
    try:
        return functions[modality.lower()](x, severity)
    except KeyError as exc:
        raise ValueError(f"Unknown modality: {modality}") from exc
