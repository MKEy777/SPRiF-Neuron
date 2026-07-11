import math
import torch


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        def gaussian(value, mu, sigma):
            return torch.exp(-((value - mu) ** 2) / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
        lens, scale, height = 0.5, 6.0, 0.15
        surrogate = (gaussian(x, 0.0, lens) * (1 + height)
                     - gaussian(x, lens, scale * lens) * height
                     - gaussian(x, -lens, scale * lens) * height)
        return grad * surrogate * 0.5


spike_fn = SurrogateSpike.apply
