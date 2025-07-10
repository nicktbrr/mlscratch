import torch

# good for sigmoid and tahn inits


def xavier(x: torch.Tensor):
    input_maps = x.shape[1]
    output_maps = x.shape[0]

    receptive_field_size = 1
    if x.dim() > 2:
        for s in x.shape[2:]:
            receptive_field_size *= s
    fan_in = input_maps * receptive_field_size
    fan_out = output_maps * receptive_field_size
    bound = torch.sqrt(6 / (fan_in + fan_out))
    x.uniform_(-bound, bound)
