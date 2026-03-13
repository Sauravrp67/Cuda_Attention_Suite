import torch
from dataclasses import dataclass
import numpy as np

@dataclass
class TimingStats:
    median_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float

    @property
    def coefficient_of_variation(self):
        return self.std_ms / self.mean_ms
    
    def __repr__(self):
        return f"TimingStats | median: {self.median_ms } ms | mean: {self.mean_ms} ms | std_ms: {self.std_ms} ms |  cv: {self.coefficient_of_variation * 100}% | min: {self.min_ms} ms | max: {self.max_ms} ms"

class Timer:
    def __init__(self,fn,warmup_iters,timed_iters):
        self.fn = fn
        self.warmup_iters = warmup_iters
        self.timed_iters = timed_iters

    def benchmark_time(self) -> TimingStats:
        cuda_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.timed_iters)]
        cuda_stop_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.timed_iters)]
        timings = []

        torch.cuda.synchronize()
        for _ in range(self.warmup_iters):
            self.fn()
        torch.cuda.synchronize()

        for i in range(self.timed_iters):
            cuda_start_events[i].record()
            self.fn()
            cuda_stop_events[i].record()
        
        torch.cuda.synchronize()

        for i in range(self.timed_iters):
            timings.append(cuda_start_events[i].elapsed_time(cuda_stop_events[i]))

        median_ms = np.median(timings).item()
        mean_ms = np.mean(timings).item()
        std_ms = np.std(timings).item()
        min_ms = np.min(timings).item()
        max_ms = np.max(timings).item()

        return TimingStats(
            median_ms = median_ms,
            mean_ms = mean_ms,
            std_ms = std_ms,
            min_ms = min_ms,
            max_ms = max_ms
        )
    
