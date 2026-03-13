from dataclasses import dataclass

@dataclass(frozen=True)
class HardwareSpecs:
    """
    Hardware specification for roofline model analysis.
    peak_flops_fp32_tflops : theoretical peak in TFLOPS (FP32, boost clock)
    peak_bandwidth_gbs     : theoretical peak memory bandwidth in GB/s
    ridge_point            : FLOPs/byte at which kernel transitions from 
                            memory-bound to compute-bound
    """
    name: str
    architecture: str
    num_sms: int
    peak_flops_fp32_tflops: float
    peak_bandwidth_gbs: float
    l2_cache_bytes: int
    vram_gb: float
    compute_capability: str

    @property
    def ridge_point(self):
        """Calculates the Ridge Point in Roofline Model Analysis"""
        return (self.peak_flops_fp32_tflops * 1e12) / (self.peak_bandwidth_gbs * 1e9)

RTX_4050_LAPTOP = HardwareSpecs(
    name="RTX 4050 Laptop GPU",
    architecture="Ada Lovelace",
    num_sms=20,
    peak_flops_fp32_tflops=13.5,  # NVIDIA spec sheet: https://www.nvidia.com/...
    peak_bandwidth_gbs=192.0,      # 2 × 8.001 GHz × 96-bit bus / 8
    l2_cache_bytes=24 * 1024 * 1024,
    vram_gb=6.0,
    compute_capability="sm_89",
)
