from __future__ import annotations
from pathlib import Path
import json
import numpy as np

from config import DRFConfig, Context
from io_digitalrf import DigitalRFLoader
from features import extract_features, stack_for_model
from supervisor import Baseline, Scorer

FEATURE_NAMES = (
    [f"psd_comp_{i}" for i in range(64)] +
    [f"band_{i}" for i in range(6)] +
    ["entropy", "kurtosis", "peak_snr"]
)

ROOT = r"C:\Users\hatti\Downloads\netbeans-26-bin\sds-code\sdk\tests\integration\data\captures\drf\westford-vpol"

cfg = DRFConfig(root=ROOT, win=4096, hop=2048, ns_read=262_144, max_windows=None)
ctx = Context(sample_rate=2_500_000.0, epoch_unix=0.0)

ldr = DigitalRFLoader(cfg, ctx)
print("META:", ldr.meta)

# warmup windows to learn baseline (e.g., first 200 windows)
WARMUP = 200
baseline = None
scorer = Scorer(w_cos=0.6, w_z=0.4, agg="max")

# MAIN LOOP ---
i = 0
for w, ks, ke in ldr.stream_windows(cfg.win, cfg.hop, normalize_power=True):
    feat = extract_features(w)
    fv = stack_for_model(feat)

    if baseline is None:
        baseline = Baseline(dim=fv.shape[0])

    if i < WARMUP:
        baseline.partial_fit(fv)
        state = "WARMUP"
        score = None
        cos = None
        z = None
        z_full = None
    else:
        s = scorer.score(fv, baseline.mean, baseline.std)
        score = s["score"]
        cos = s["cos"]
        z = s["z"]
        z_full = s["z_full"]

        # opt. slow bl update
        baseline.partial_fit(fv)

        # determine state
        if score < 0.35:
            state = "OK"
        elif score < 0.65:
            state = "WARN"
        else:
            state = "ALERT"

    t0 = float(ldr.k_to_unix(ks))
    t1 = float(ldr.k_to_unix(ke))

    row = {
        "i": i,
        "k_start": int(ks),
        "k_end": int(ke),
        "t_start": t0,
        "t_end": t1,
        "state": state,
        "score": score,
        "cos": cos,
        "z": z,
    }
    
    # Attribution if WARN or ALERT
if state in ("WARN", "ALERT") and z_full is not None:
    # Top-k features
    top_k = []
    idxs = np.argsort(z_full)[::-1]
    for j in idxs[:3]:
        top_k.append({
            "feature": FEATURE_NAMES[j],
            "z": float(z_full[j]),
            "value": float(fv[j]),
        })

    # Peak frequency using exactly the same nfft & shift as features
    peak_bin = int(feat["peak_bin"])
    N = int(feat.get("nfft", feat.get("psd_len", cfg.win)))
    fs = ctx.sample_rate

    if bool(feat.get("fftshifted", True)):
        freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / fs))
    else:
        freqs = np.fft.fftfreq(N, d=1.0 / fs)

    # Tiny guard in case an old module version was cached
    if peak_bin >= len(freqs):
        peak_bin = len(freqs) - 1
    elif peak_bin < 0:
        peak_bin = 0

    peak_freq_hz = float(freqs[peak_bin])

    row["attribution"] = {
        "top_features": top_k,
        "peak_freq_hz": peak_freq_hz,
    }

    print(json.dumps(row))
    i += 1