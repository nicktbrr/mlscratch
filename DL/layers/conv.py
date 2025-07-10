import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional


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

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

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

    def forward(self, x):
        x = nn.functional.pad(x, )

        for row in range(self.kernel_size[0] // 2, x.shape[0], self.stride[0]):
            for col in range(self.kernel_size[1] // 2, x.shape[1], self.stride[1]):
                pass
