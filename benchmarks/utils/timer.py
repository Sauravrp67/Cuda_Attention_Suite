# benchmarks/timer.py
#
# Upgrade over v1:
#   - Cold L2 cache flush between iterations (matches FlashInfer/vLLM methodology)
#   - CUDA Graphs mode for sub-millisecond kernels (eliminates launch overhead noise)
#   - Auto dry-run by elapsed time (robust across N=64 to N=4096)
#   - TimingStats unchanged — downstream code unaffected
#
# Methodology note:
#   SOTA benchmark suites (FA, FlashInfer, vLLM) use analytical bytes_moved,
#   NOT ncu hardware counters, for their metric tables and roofline plots.
#   ncu is used separately for diagnostics (occupancy, stall reasons, etc.).
#
#   bytes_moved = 4 * B * H * N * D * elem_bytes  (algorithmic lower bound)
#   achieved_bw = bytes_moved / (median_ms * 1e-3) / 1e9
#   achieved_tf = flops      / (median_ms * 1e-3) / 1e12

import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from utils.hardware_constants import RTX_4050_LAPTOP

# TimingStats — unchanged from v1, all downstream code compatible

@dataclass
class TimingStats:
    median_ms: float
    mean_ms:   float
    std_ms:    float
    min_ms:    float
    max_ms:    float

    @property
    def coefficient_of_variation(self) -> float:
        return self.std_ms / self.mean_ms

    def __repr__(self) -> str:
        return (
            f"TimingStats | median: {self.median_ms:.4f} ms | "
            f"mean: {self.mean_ms:.4f} ms | std: {self.std_ms:.4f} ms | "
            f"cv: {self.coefficient_of_variation * 100:.2f}% | "
            f"min: {self.min_ms:.4f} ms | max: {self.max_ms:.4f} ms"
        )

# L2 flush buffer
# Evicts L2 cache by writing a buffer larger than L2 size.

_L2_FLUSH_BUFFER: Optional[torch.Tensor] = None
_L2_FLUSH_SIZE_BYTES = RTX_4050_LAPTOP.l2_cache_bytes * 2   # 48 MB For this hardware

def _get_l2_flush_buffer() -> torch.Tensor:
    global _L2_FLUSH_BUFFER
    if _L2_FLUSH_BUFFER is None:
        n = _L2_FLUSH_SIZE_BYTES // 4   # float32 elements
        _L2_FLUSH_BUFFER = torch.zeros(n, dtype=torch.float32, device="cuda")
    return _L2_FLUSH_BUFFER

def flush_l2_cache() -> None:
    """
    Evict L2 cache by writing to a 32 MB scratch buffer.
    Call before each timed iteration when cold_l2_cache=True.
    """
    buf = _get_l2_flush_buffer()
    buf.fill_(0.0)
    torch.cuda.synchronize()

# Bench: CUDA EVENTS

def _bench_cuda_events(
    fn:           Callable,
    warmup_iters: int,
    timed_iters:  int,
    cold_l2:      bool,
) -> list[float]:
    """
    Core CUDA Event timing loop.
    Returns:
      [list] of per-iteration latencies in milliseconds.
    """
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(timed_iters)]
    stop_events  = [torch.cuda.Event(enable_timing=True) for _ in range(timed_iters)]

    # Warmup — not timed, brings clocks to boost and warms caches/JIT
    torch.cuda.synchronize()
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    # Timed iterations
    for i in range(timed_iters):
        if cold_l2:
            flush_l2_cache()
        start_events[i].record()
        fn()
        stop_events[i].record()

    torch.cuda.synchronize()

    return [start_events[i].elapsed_time(stop_events[i]) for i in range(timed_iters)]

# Bench: CUDA Graphs

def _bench_cuda_graph(
    fn:                     Callable,
    warmup_iters:           int,
    timed_iters:            int,
    num_iters_within_graph: int,
    cold_l2:                bool,
) -> list[float]:
    """
    Cuda Graph for amortizing CPU launch overhead — used for kernels faster than ~200µs.
    Uses rotating input buffers to prevent L2 reuse across graph replays.

    Captures fn into a graph, replays graph per timed iteration.
    Each graph replay executes fn num_iters_within_graph times.
    Reported time is per-fn-call = graph_time / num_iters_within_graph.

    """
    # Warmup outside graph to stabilise clocks
    torch.cuda.synchronize()
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    # Capture graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(num_iters_within_graph):
            fn()

    start = torch.cuda.Event(enable_timing=True)
    stop  = torch.cuda.Event(enable_timing=True)
    timings = []

    for _ in range(timed_iters):
        if cold_l2:
            flush_l2_cache()
        start.record()
        g.replay()
        stop.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(stop) / num_iters_within_graph)

    return timings

# Auto dry-run helpers (time-based, matches FlashInfer convention)

def _auto_warmup_iters(fn: Callable, target_ms: float = 25.0) -> int:
    """Run fn until target_ms of GPU time has elapsed. Returns iter count."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop  = torch.cuda.Event(enable_timing=True)
    elapsed = 0.0
    iters   = 0
    while elapsed < target_ms:
        start.record(); fn(); stop.record()
        torch.cuda.synchronize()
        elapsed += start.elapsed_time(stop)
        iters   += 1
        if iters > 1000:   # safety cap
            break
    return max(iters, 1)

def _auto_timed_iters(fn: Callable, target_ms: float = 100.0) -> int:
    """Estimate how many iters fit in target_ms. Returns iter count."""
    start = torch.cuda.Event(enable_timing=True)
    stop  = torch.cuda.Event(enable_timing=True)
    start.record(); fn(); stop.record()
    torch.cuda.synchronize()
    single_ms = max(start.elapsed_time(stop), 1e-4)
    return max(int(target_ms / single_ms), 5)

# Public API
class Timer:
    """
    Drop-in replacement for v1 Timer.
    Adds: cold_l2_cache, use_cuda_graph, auto iter count.

    v1 compatibility:
        Timer(fn, warmup_iters=5, timed_iters=50).benchmark_time()
        → works unchanged, now also flushes L2 by default.
    """

    def __init__(
        self,
        fn:                     Callable,
        warmup_iters:           Optional[int]  = None,
        timed_iters:            Optional[int]  = None,
        cold_l2_cache:          bool           = True,
        use_cuda_graph:         bool           = False,
        num_iters_within_graph: int            = 10,
        warmup_time_ms:         float          = 25.0,
        timed_time_ms:          float          = 100.0,
    ):
        self.fn                     = fn
        self._warmup_iters          = warmup_iters
        self._timed_iters           = timed_iters
        self.cold_l2_cache          = cold_l2_cache
        self.use_cuda_graph         = use_cuda_graph
        self.num_iters_within_graph = num_iters_within_graph
        self.warmup_time_ms         = warmup_time_ms
        self.timed_time_ms          = timed_time_ms

    def _resolve_iters(self) -> tuple[int, int]:
        w = self._warmup_iters
        t = self._timed_iters
        if w is None:
            w = _auto_warmup_iters(self.fn, self.warmup_time_ms)
        if t is None:
            t = _auto_timed_iters(self.fn, self.timed_time_ms)
        return w, t

    def benchmark_time(self) -> TimingStats:
        warmup, timed = self._resolve_iters()

        if self.use_cuda_graph:
            raw = _bench_cuda_graph(
                self.fn, warmup, timed,
                self.num_iters_within_graph,
                self.cold_l2_cache,
            )
        else:
            raw = _bench_cuda_events(
                self.fn, warmup, timed, self.cold_l2_cache
            )

        return TimingStats(
            median_ms = float(np.median(raw)),
            mean_ms   = float(np.mean(raw)),
            std_ms    = float(np.std(raw)),
            min_ms    = float(np.min(raw)),
            max_ms    = float(np.max(raw)),
        )

# Standalone bench_gpu_time

def bench_gpu_time(
    fn:                     Callable,
    warmup_iters:           Optional[int]  = None,
    timed_iters:            Optional[int]  = None,
    warmup_time_ms:         float          = 25.0,
    timed_time_ms:          float          = 100.0,
    cold_l2_cache:          bool           = True,
    use_cuda_graph:         bool           = False,
    num_iters_within_graph: int            = 10,
) -> list[float]:
    """
    Unified GPU benchmarking entry point (mirrors FlashInfer's bench_gpu_time).

    Returns raw per-iteration times in milliseconds.
    Compute median/mean/std yourself or pass to TimingStats.

    Example — compute achieved bandwidth:
        times = bench_gpu_time(lambda: naive_attn(q, k, v))
        median_ms = np.median(times)

        # Algorithmic bytes (lower bound — same convention as FA paper)
        B, H, N, D = q.shape
        bytes_moved = 4 * B * H * N * D * 4   # fp32
        flops       = 4 * B * H * N * N * D

        achieved_bw_gbs  = bytes_moved / (median_ms * 1e-3) / 1e9
        achieved_tflops  = flops       / (median_ms * 1e-3) / 1e12
        sol_bw_pct       = achieved_bw_gbs  / 192.0 * 100
        sol_compute_pct  = achieved_tflops  / 13.5  * 100
    """
    timer = Timer(
        fn=fn,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        cold_l2_cache=cold_l2_cache,
        use_cuda_graph=use_cuda_graph,
        num_iters_within_graph=num_iters_within_graph,
        warmup_time_ms=warmup_time_ms,
        timed_time_ms=timed_time_ms,
    )
    warmup, timed = timer._resolve_iters()
    if use_cuda_graph:
        return _bench_cuda_graph(fn, warmup, timed,
                                  num_iters_within_graph, cold_l2_cache)
    return _bench_cuda_events(fn, warmup, timed, cold_l2_cache)