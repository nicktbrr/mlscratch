# good for relu varieties

import torch


def he(x: torch.Tensor):
    fan_in = x.shape[1]
    receptive_field = 1
    if x.dim() > 2:
        for s in x.shape[2:]:
            receptive_field *= s
    fan_in *= receptive_field
    bound = torch.sqrt(6 / fan_in)
    return x.uniform_(-bound, bound)
