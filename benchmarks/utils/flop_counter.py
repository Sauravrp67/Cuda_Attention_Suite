from dataclasses import dataclass

@dataclass
class KernelStats:
    flops: float
    algorithmic_bytes: int
    pessimistic_bytes: int

    @property
    def algorithmic_ai(self) -> float:
        """Arithmetic Intensity assuming perfect caching (FLOPs/byte)"""
        return self.flops / self.algorithmic_bytes
    
    @property
    def pessimistic_ai(self) -> float:
        """FLOPs/byte assuming no caching — worst case roofline position"""
        return self.flops / self.pessimistic_bytes
    
    