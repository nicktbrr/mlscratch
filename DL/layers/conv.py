# C, H, W


import torch
import torch.nn as nn
from typing import Union, Optional
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from weight_inits.he import he
from weight_inits.xavier import xavier


class Conv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, tuple],
                 stride: Optional[Union[int, tuple]] = 1,
                 padding: Optional[Union[int, tuple]] = 0,
                 bias: Optional[bool] = True):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        # stride = H, W
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        # padding = H, W
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        self.bias_on = bias

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        self.filter = nn.Parameter(torch.empty(out_channels,
                                   in_channels,
                                   self.kernel_size[0],
                                   self.kernel_size[1]))
        # initialize the filter
        self.filter.data = he(self.filter)

    # x: (B, in_channels, H, W) -> (B, out_channels, H, W)
    def forward(self, x):
        pad = (self.padding[1], self.padding[1],
               self.padding[0], self.padding[0])
        x_padded = nn.functional.pad(x, pad)
        flattened_filter = self.filter.view(self.out_channels, -1)
        unfolded_x = nn.functional.unfold(x_padded,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride)
        temp_res = torch.matmul(flattened_filter, unfolded_x)
        if self.bias_on:
            temp_res = temp_res + self.bias.unsqueeze(1)
        H_out = (x_padded.shape[2] - self.kernel_size[0]) // self.stride[0] + 1
        W_out = (x_padded.shape[3] - self.kernel_size[1]) // self.stride[1] + 1

        return temp_res.view(x.shape[0], self.out_channels, H_out, W_out)


if __name__ == "__main__":
    torch.manual_seed(42)
    
    conv = Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
    x = torch.randn(2, 3, 32, 32)
    output = conv(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    conv_no_bias = Conv2d(in_channels=1, out_channels=2, kernel_size=5, bias=False)
    x2 = torch.randn(1, 1, 28, 28)
    output2 = conv_no_bias(x2)
    print(f"Input shape: {x2.shape}")
    print(f"Output shape: {output2.shape}")
    
    conv_stride = Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=2)
    x3 = torch.randn(4, 2, 16, 16)
    output3 = conv_stride(x3)
    print(f"Input shape: {x3.shape}")
    print(f"Output shape: {output3.shape}")
