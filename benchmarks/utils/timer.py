import torch

class benchmark_kernels:
    def __init__(self,fn,warmup_iters,timed_iters):
        self.fn = fn
        self.warmup_iters = warmup_iters
        self.timed_iters = timed_iters
        