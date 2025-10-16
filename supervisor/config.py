from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DRFConfig:
    root: Path                   # DRF dataset root
    channel: str | None = None   # if None - autopick first
    ns_read: int = 1_000_000     # samples pulled per disk read
    # windowing :
    win: int = 4096
    hop: int = 2048              # 50 % overlap
    # safety :
    max_windows: int | None = None

@dataclass(frozen=True)
class Context:
    sample_rate: float           # hz
    epoch_unix: float = 0.0      # sec since origin