from utils import Timer
from PyCuAttention.kernels.loader import naive_attn,sdpa_attn
import torch

for N in [64,128,512,1024,2048,4096]:
    Q = torch.randn(1,8,N,64,device = "cuda")
    K = torch.randn(1,8,N,64, device = "cuda")
    V = torch.randn(1,8,N,64, device = "cuda")

    kv_size_per_head = 2 * N * 64 * 4
    kv_size_all_heads = kv_size_per_head * 8

    timer_custom = Timer(
        fn = lambda: naive_attn(
            q = Q,
            k = K,
            v = V,
            scaling = 0.125
        ),
        warmup_iters=25,
        timed_iters=100
    )

    timer_sdpa = Timer(
        fn = lambda: sdpa_attn(
            q = Q,
            k = K,
            v = V,
            scaling = 0.125
        ),
        warmup_iters = 25,
        timed_iters=100
    )

    timestats_custom = timer_custom.benchmark_time()
    timestats_sdpa = timer_sdpa.benchmark_time()



    print(f"Benchmark for N = {N}")
    print(f"KV_Per_Head: {kv_size_per_head / 1e6} MB ", end = "")
    print(f"KV_all_head: {kv_size_all_heads / 1e6} MB")
    print(f"Custom Kernel:\n {timestats_custom}")
    print(f"SDPA Kernel:\n {timestats_sdpa}\n")

