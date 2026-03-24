#!/usr/bin/env python3
"""
_ncu_target.py — minimal ncu attach target for CUDA Attention Suite.

ncu launches this process, attaches, and collects hardware counters.
Do NOT add warmup loops here — ncu instruments the FIRST kernel launch.
Do NOT add timing here — ncu timing is invalid (kernel replayed per counter group).

Arguments (all required, passed by run_ncu.sh):
    --kernel   naive_v1 | sdpa | fused_v2
    --B --H --N --D  shape
    --dtype    float32 | float16 | bfloat16
"""
import argparse
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

parser = argparse.ArgumentParser()
parser.add_argument('--kernel', required=True)
parser.add_argument('--B', type=int, required=True)
parser.add_argument('--H', type=int, required=True)
parser.add_argument('--N', type=int, required=True)
parser.add_argument('--D', type=int, required=True)
parser.add_argument('--dtype', default='float32')
args = parser.parse_args()

DTYPE_MAP = {
    'float32':  torch.float32,
    'float16':  torch.float16,
    'bfloat16': torch.bfloat16,
}
dtype = DTYPE_MAP[args.dtype]
device = 'cuda'

torch.manual_seed(42)
q = torch.randn(args.B, args.H, args.N, args.D, dtype=dtype, device=device)
k = torch.randn(args.B, args.H, args.N, args.D, dtype=dtype, device=device)
v = torch.randn(args.B, args.H, args.N, args.D, dtype=dtype, device=device)

# Single synchronised launch — ncu instruments this.
torch.cuda.synchronize()

if args.kernel == 'naive_v1':
    from PyCuAttention.kernels.loader import naive_attn
    out = naive_attn(q, k, v)

elif args.kernel == 'sdpa':
    from torch.nn.functional import scaled_dot_product_attention
    scale = args.D ** -0.5
    out = scaled_dot_product_attention(q, k, v, scale=scale)

elif args.kernel == 'fused_v2':
    from PyCuAttention.kernels.loader import fused_attn
    scale = args.D ** -0.5
    out = fused_attn(q, k, v, scale)

else:
    print(f"Unknown kernel: {args.kernel}", file=sys.stderr)
    sys.exit(1)

torch.cuda.synchronize()
print(f"kernel={args.kernel} B={args.B} H={args.H} N={args.N} D={args.D} "
      f"dtype={args.dtype} output_shape={list(out.shape)} DONE")
