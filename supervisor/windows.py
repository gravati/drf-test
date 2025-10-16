from __future__ import annotations
from typing import Iterator, Tuple, Dict
import numpy as np

def window_iter(
        iq_block: np.ndarray,
        k0: int,
        win: int,
        hop: int,
) -> Iterator[Tuple[np.ndarray, int, int]]:
    """
    Slice block into overlapping windows.
    YIELDS: (window_iq, k_start, k_end)
    """

    n = len(iq_block)
    for i in range(0, max(0, n - win + 1), hop):
        yield iq_block[i:i+win], (k0 + i), (k0 + i + win)

def unit_power(x: np.ndarray) -> np.ndarray:
    """
    OPT.: Norm window to unit avg power
    """
    p = np.mean(np.abs(x)**2) + 1e-12
    return x / np.sqrt(p)