import torch
from ..kernels.loader import naive_attn

class NaiveAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, scaling):
        q,k,v = query.contiguous(), key.contiguous(), value.contiguous()
        return naive_attn(q,k,v,scaling)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass for Naive V1 not implemented yet.")
    
