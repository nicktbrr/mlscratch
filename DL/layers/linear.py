import torch
import torch.nn as nn
import torch.nn.functional as F

class linear(nn.Module):
    def __init__(self, input: int, output: int):
        super(linear, self).__init__()
        self.weights = nn.Parameter(torch.randn(output, input))
        self.bias = nn.Parameter(torch.randn(output))
    
    def forward(self, input):
        return F.linear(input, self.weights, self.bias)
    


if __name__ == "__main__":
    input = torch.randn(10)
    print(input.shape)
    f = linear(input.shape[0],20)
    print(f.weights.shape)
    print(f(input).shape)
