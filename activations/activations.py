import numpy as np
from scipy.stats import norm
import torch


def ReLu(x: np.array):
    return torch.maximum(x, torch.tensor(0))

def sigmoid(x: np.array):
    with torch.no_grad():   
        return 1 / (1 + torch.exp(-x))

def tanH(x: np.array):
    num = np.exp(x) - np.exp(-x)
    den = np.exp(x) + np.exp(-x)
    return num / den

def LeakyReLu(x: np.array, alpha=0.01):
    return np.maximum(alpha*x, x)

def softmax(x: np.array):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def GELU(x: np.array):
    return x * norm.cdf(x)