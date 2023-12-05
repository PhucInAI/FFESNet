import torch
import torch.nn.functional as F

def tversky_loss(pred, mask, alpha=0.5, beta=0.5, gamma=2):
    pred = torch.sigmoid(pred)       

    #flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)

    #True Positives, False Positives & False Negatives
    TP = (pred * mask).sum()    
    FP = ((1 - mask) * pred).sum()
    FN = (mask * (1 - pred)).sum()

    Tversky = (TP + 1) / (TP + alpha * FP + beta * FN + 1)  

    return (1 - Tversky) ** gamma