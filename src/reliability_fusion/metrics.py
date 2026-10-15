"""Common binary-classification metrics."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics(targets, probabilities):
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    return {
        "auroc": float(roc_auc_score(targets, probabilities)),
        "auprc": float(average_precision_score(targets, probabilities)),
    }
