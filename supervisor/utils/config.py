from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

# DRF ---
@dataclass(frozen=True)
class DRFConfig:
    root: Path                  # DigitalRF dataset root
    channel: str | None = None  # None => auto-pick first
    ns_read: int = 1_000_000    # samples per disk read
    win: int = 4096             # window size (samples)
    hop: int = 2048             # hop size (samples)
    max_windows: int | None = None  # None => run full stream

# CONTEXT ---
@dataclass(frozen=True)
class Context:
    sample_rate: float          # Hz (e.g., 2_500_000.0)
    epoch_unix: float = 0.0     # unix epoch offset for k_to_unix()

# SUPERVISOR ---
@dataclass(frozen=True)
class SupervisorConfig:
    # Threshold policy
    warn: float = 0.35
    fail: float = 0.65
    warmup_windows: int = 200

    # Scorer weights
    w_cos: float = 0.6
    w_z: float = 0.4
    agg: str = "max"  # or "mean"

    # Domain constant (from metadata)
    center_freq_hz: float = 1_024_000_000.4842604

# SYSTEM ---
@dataclass(frozen=True)
class SystemConfig:
    drf: DRFConfig
    ctx: Context
    sup: SupervisorConfig
