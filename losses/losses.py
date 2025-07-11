import torch

def MSE(y_pred: torch.tensor, y_true: torch.tensor):
    return torch.mean((y_true - y_pred)**2)