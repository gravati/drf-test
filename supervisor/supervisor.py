from __future__ import annotations
import numpy as np
from dataclasses import dataclass

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(1.0 - np.dot(a, b) / (na * nb))

@dataclass
class Baseline:
    dim: int
    count: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    def partial_fit(self, x: np.ndarray) -> None:
        x = x.astype(np.float64, copy=False)
        if self.mean is None:
            self.mean = np.zeros(self.dim, dtype=np.float64)
            self.m2 = np.zeros(self.dim, dtype=np.float64)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    @property
    def std(self) -> np.ndarray:
        if self.count < 2:
            return np.ones(self.dim, dtype=np.float64)
        return np.sqrt(np.maximum(self.m2 / (self.count - 1), 1e-12))
    
def zscores(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / (std + 1e-12)
    
@dataclass
class Scorer:
    w_cos: float = 0.6
    w_z: float = 0.4
    agg: str = "max" # mean

    def score(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict:
        cos = cosine_distance(x, mean)
        z = np.abs(zscores(x, mean, std))
        z_agg = np.max(z) if self.agg == "max" else float(np.mean(z))
        # bounded fusion, cos and z mapped to [0,1], soft ramps
        s_cos = 1.0 - np.exp(-3.0 * cos)
        s_z = 1.0 - np.exp(-0.5 * z_agg)
        s = self.w_cos * s_cos + self.w_z * s_z
        return{"cos": float(s_cos), "z": float(s_z), "score": float(s)}