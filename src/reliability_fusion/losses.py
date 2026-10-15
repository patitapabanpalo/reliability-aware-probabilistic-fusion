"""Objectives used in the final experiments. Dataset-specific differences are explicit."""
import torch
import torch.nn.functional as F

def binary_prediction_loss(logits, targets): return F.binary_cross_entropy_with_logits(logits, targets.float())
def reliability_ranking_loss(higher, lower, margin=0.05): return torch.relu(margin-higher+lower).mean()
def monotonic_reliability_loss(levels, margin=0.05):
    if len(levels)!=4: raise ValueError("Expected four reliability levels (clean + 3 corruption severities).")
    return sum(reliability_ranking_loss(levels[i],levels[i+1],margin) for i in range(3))
def consistency_loss(incomplete_logits, full_logits, space="logit"):
    """HAIM uses logit MSE; final Symile uses probability-space MSE."""
    teacher=full_logits.detach(); student=incomplete_logits
    if space=="probability": teacher=torch.sigmoid(teacher); student=torch.sigmoid(student)
    elif space!="logit": raise ValueError("space must be 'logit' or 'probability'")
    return F.mse_loss(student,teacher)
