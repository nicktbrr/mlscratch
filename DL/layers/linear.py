import torch
import torch.nn as nn
import torch.nn.functional as F

class linear(nn.Module):
    def __init__(self,input, output):
        super(linear).__init__()
        self.weights = nn.Parameter(torch.randn(input, output))
        self.bias = nn.Parameter(torch.randn(input))
    
    def forward(self, input):
        return F.linear(input, self.weights, self.bias)
