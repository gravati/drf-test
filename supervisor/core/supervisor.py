# supervisor/supervisor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence, Optional, Dict, Any
import numpy as np

from supervisor.utils.utils import make_explanation
from supervisor.utils.config import SystemConfig, SupervisorConfig


# CORE ---
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
    agg: str = "max"  # "max" or "mean"

    def score(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict:
        cos = cosine_distance(x, mean)
        z = np.abs(zscores(x, mean, std))
        z_agg = np.max(z) if self.agg == "max" else float(np.mean(z))
        # bounded fusion, cos and z mapped to [0,1], soft ramps
        s_cos = 1.0 - np.exp(-3.0 * cos)
        s_z = 1.0 - np.exp(-0.5 * z_agg)
        s = self.w_cos * s_cos + self.w_z * s_z
        return {"cos": float(s_cos), "z": float(s_z), "score": float(s), "z_full": z}


# ORCHESTRATION ---
@dataclass
class SupervisorConfig:
    warn: float = 0.35
    fail: float = 0.65
    warmup_windows: int = 200
    w_cos: float = 0.6
    w_z: float = 0.4
    agg: str = "max"
    center_freq_hz: float = 1_024_000_000.4842604  # from digital metadata

class Supervisor:
    """
    Drives the pipeline:
      - warmup baseline fitting
      - per-window scoring
      - OK/WARN/ALERT policy
      - attribution (peak freq, top features, entropy, peak_snr)
      - explanation string
    """

    def __init__(self,
                 cfg: SupervisorConfig,
                 feature_names: Sequence[str]):
        self.cfg = cfg
        self.feature_names = list(feature_names)
        self.scorer = Scorer(w_cos=cfg.w_cos, w_z=cfg.w_z, agg=cfg.agg)
        self.baseline: Optional[Baseline] = None
        self._i: int = 0

    @classmethod
    def from_system_config(cls,
                           sys_cfg: SystemConfig,
                           feature_names: Sequence[str]) -> Supervisor:
        return cls(cfg=sys_cfg.sup, feature_names=feature_names)

    @staticmethod
    def _classify(score: float, warn: float, fail: float) -> str:
        if score < warn:
            return "OK"
        if score < fail:
            return "WARN"
        return "ALERT"

    @staticmethod
    def _fft_freqs(n: int, fs: float, fftshifted: bool = True) -> np.ndarray:
        freqs = np.fft.fftfreq(n, d=1.0 / fs)
        return np.fft.fftshift(freqs) if fftshifted else freqs

    def _build_attribution(self,
                           feat: Dict[str, Any],
                           fv: np.ndarray,
                           z_full: np.ndarray,
                           sample_rate_hz: float,
                           top_k: int = 3) -> Dict[str, Any]:
        # Top-k contributors
        idxs = np.argsort(z_full)[::-1][:top_k]
        top = [{"feature": self.feature_names[j],
                "z": float(z_full[j]),
                "value": float(fv[j])} for j in idxs]

        peak_bin = int(feat["peak_bin"])
        N = int(feat.get("nfft", feat.get("psd_len", len(fv))))
        fftshifted = bool(feat.get("fftshifted", True))
        freqs = self._fft_freqs(N, sample_rate_hz, fftshifted)

        # guard
        peak_bin = max(0, min(peak_bin, len(freqs) - 1))
        peak_freq_hz = float(freqs[peak_bin])
        abs_rf_hz = float(self.cfg.center_freq_hz + peak_freq_hz)

        return {
            "top_features": top,
            "peak_bin": peak_bin,
            "peak_freq_hz": peak_freq_hz,       # signed offset from center
            "peak_freq_rf_hz": abs_rf_hz,       # absolute RF frequency
            "entropy": float(feat["entropy"]),
            "peak_snr_db": float(feat["peak_snr"]),
        }

    def process_window(self,
                       x_win: np.ndarray,
                       k_start: int,
                       k_end: int,
                       *,
                       extract_features: Callable[[np.ndarray], Dict[str, Any]],
                       stack_for_model: Callable[[Dict[str, Any]], np.ndarray],
                       k_to_unix: Callable[[int], float],
                       sample_rate_hz: float) -> Dict[str, Any]:
        """
        Run one window through the supervisor and return a self-contained row dict.
        """
        feat = extract_features(x_win)
        fv = stack_for_model(feat)

        if self.baseline is None:
            self.baseline = Baseline(dim=fv.shape[0])

        # Warmup: only fit baseline, no scoring
        if self._i < self.cfg.warmup_windows:
            self.baseline.partial_fit(fv)
            state, score, cos, z, z_full = "WARMUP", None, None, None, None
        else:
            s = self.scorer.score(fv, self.baseline.mean, self.baseline.std)
            score, cos, z, z_full = s["score"], s["cos"], s["z"], s["z_full"]

            # slow baseline adaptation
            self.baseline.partial_fit(fv)

            state = self._classify(score, self.cfg.warn, self.cfg.fail)

        row: Dict[str, Any] = {
            "i": self._i,
            "k_start": int(k_start),
            "k_end": int(k_end),
            "t_start": float(k_to_unix(k_start)),
            "t_end": float(k_to_unix(k_end)),
            "state": state,
            "score": score,
            "cos": cos,
            "z": z,
        }

        if state in ("WARN", "ALERT") and z_full is not None:
            attr = self._build_attribution(
                feat=feat,
                fv=fv,
                z_full=z_full,
                sample_rate_hz=sample_rate_hz,
            )
            row["attribution"] = attr
            row["explanation"] = make_explanation(row)

        self._i += 1
        return row
