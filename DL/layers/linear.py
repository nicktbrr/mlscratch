import os
import sys
import torch.nn as nn
import torch
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from weight_inits.he import he



class linear(nn.Module):
    def __init__(self, input: int, output: int):
        super(linear, self).__init__()
        self.weights = nn.Parameter(torch.randn(output, input))
        self.bias = nn.Parameter(torch.randn(output))

        self.weights = he(self.weights)

    def forward(self, input):
        return torch.matmul(input, self.weights.T) + self.bias


if __name__ == "__main__":
    input = torch.randn(10)
    print(input.shape)
    f = linear(input.shape[0], 20)
    print(f.weights.shape)
    print(f(input).shape)
