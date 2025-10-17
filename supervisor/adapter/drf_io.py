# adapter script

from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Tuple, Dict
import numpy as np

import digital_rf as drf

from supervisor.utils.config import DRFConfig, Context
from supervisor.utils.windows import window_iter, unit_power

class DigitalRFLoader:
    """
    Testable wrapper for DigitalRFReader.
    FOR: Reading contiguous sample blocks, attaching basic context.
    """

    def __init__(self, cfg: DRFConfig, ctx: Context):
        self.cfg = cfg
        self.ctx = ctx

        self._r = drf.DigitalRFReader(str(cfg.root))
        chans = self._r.get_channels()
        if not chans:
            raise RuntimeError("No channels in dataset!")
        self.ch = cfg.channel or chans[0]

        self.start, self.end = self._r.get_bounds(self.ch)
        if self.end <= self.start:
            raise RuntimeError(f"Channel {self.ch} is empty.")
        
    @property
    def meta(self) -> Dict:
        """
        Static metadata for windows.
        """
        return{
            "channel": self.ch,
            "start_sample": int(self.start),
            "end_sample": int(self.end),
            "config": asdict(self.cfg),
            "context": asdict(self.ctx),
        }
    
    def read_blocks(self) -> Iterator[Tuple[np.ndarray, int, int]]:
        """
        Stream blocks of samples.
        YIELDS: (iq_block, k0, k1)
        """
        sr = self.ctx.sample_rate
        n = self.cfg.ns_read
        k = int(self.start)
        while k < self.end:
            take = min(n, self.end - k)
            x = self._r.read_vector(k, take, self.ch)
            # norm dtype contracts
            if np.iscomplexobj(x) and x.dtype != np.complex64:
                x = x.astype(np.complex64, copy=False)
            elif not np.iscomplexobj(x):
                x = x.astype(np.float32, copy=False)
            yield x, k, k + take
            k += take

    def stream_windows(
            self,
            win: int, 
            hop: int, 
            normalize_power: bool = False,
    ) -> Iterator[Tuple[np.ndarray, int, int]]:
        """
        Overlapping windows across block bounds.
        """
        carry = None
        carry_len = max(0, win - hop)
        emitted = 0

        for block, k0, k1 in self.read_blocks():
            if carry is not None and len(carry) > 0:
                k0 = k0 - len(carry)
                block = np.concatenate((carry, block))

            for w, ks, ke in window_iter(block, k0, win, hop):
                if normalize_power:
                    w = unit_power(w)
                yield w, ks, ke
                emitted += 1
                if self.cfg.max_windows and emitted >= self.cfg.max_windows:
                    return
                
            if carry_len > 0:
                carry = block[-carry_len:] if len(block) >= carry_len else block
            else:
                carry = None

    def k_to_unix(self, k: int | np.ndarray) -> np.ndarray:
        """
        Convert sample index to Unix time (sec).
        """
        return self.ctx.epoch_unix + (np.asarray(k, dtype=np.float64) / self.ctx.sample_rate)