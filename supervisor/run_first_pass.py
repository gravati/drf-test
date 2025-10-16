from __future__ import annotations
from pathlib import Path
import json
import numpy as np

from config import DRFConfig, Context
from io_digitalrf import DigitalRFLoader
from features import feature_vector
from supervisor import Baseline, Scorer

ROOT = r"C:\Users\hatti\Downloads\netbeans-26-bin\sds-code\sdk\tests\integration\data\captures\drf\westford-vpol"

cfg = DRFConfig(root=ROOT, win=4096, hop=2048, ns_read=262_144, max_windows=None)
ctx = Context(sample_rate=2_500_000.0, epoch_unix=0.0)

ldr = DigitalRFLoader(cfg, ctx)
print("META:", ldr.meta)

# warmup windows to learn baseline (e.g., first 200 windows)
WARMUP = 200
baseline = None
scorer = Scorer(w_cos=0.6, w_z=0.4, agg="max")

rows = []
i = 0
for w, ks, ke in ldr.stream_windows(cfg.win, cfg.hop, normalize_power=True):
    fv = feature_vector(w)  # shape (64 + 6 + 3,) = 73 dims
    if baseline is None:
        baseline = Baseline(dim=fv.shape[0])

    if i < WARMUP:
        baseline.partial_fit(fv)
        state = "WARMUP"
        s = {"cos": None, "z": None, "score": None}
    else:
        s = scorer.score(fv, baseline.mean, baseline.std)
        # optional slow update to track normal drift (comment out to freeze)
        baseline.partial_fit(fv)
        state = "OK" if s["score"] < 0.35 else ("WARN" if s["score"] < 0.65 else "ALERT")

    t0 = float(ldr.k_to_unix(ks))
    t1 = float(ldr.k_to_unix(ke))

    row = {
        "i": i,
        "k_start": int(ks),
        "k_end": int(ke),
        "t_start": t0,
        "t_end": t1,
        "state": state,
        "score": s["score"],
        "cos": s["cos"],
        "z": s["z"],
    }
    print(json.dumps(row))
    rows.append(row)

    i += 1
    # (optional) break early for smoke runs
    # if i > 600: break
