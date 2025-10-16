from __future__ import annotations
import numpy as np

def psd_db(x: np.ndarray, nfft: int | None = None) -> np.ndarray:
    """
    Windowed FFT-based PSD in dB.
    - Uses full complex FFT for complex IQ.
    - Falls back to rFFT for real signals.
    """
    if nfft is None:
        nfft = len(x)

    w = np.hanning(len(x)).astype(np.float64, copy=False)
    xw = x * w

    if np.iscomplexobj(xw):
        X = np.fft.fft(xw, n=nfft)
    else:
        X = np.fft.rfft(xw, n=nfft)

    P = (np.abs(X) ** 2) / nfft
    return 10.0 * np.log10(P + 1e-12)

def compress_bins(psd: np.ndarray, n_bins: int = 64) -> np.ndarray:
    L = len(psd)
    edges = np.linspace(0, L, n_bins + 1, dtype=int)
    out = []
    for i in range(n_bins):
        a, b = edges[i], edges[i+1]
        if b > a:
            out.append(psd[a:b].mean())
        else:
            out.append(psd[a])
    return np.array(out, dtype=np.float32)

def spectral_entropy(psd_db_vals: np.ndarray) -> float:
    p = np.clip(10 ** (psd_db_vals / 10.0), 1e-18, None)
    p = p / p.sum()
    h = -(p * np.log(p + 1e-18)).sum()
    return float(h / np.log(len(p)))

def kurtosis_mag(x: np.ndarray) -> float:
    m = np.abs(x)
    m = m - m.mean()
    v = (m**2).mean() + 1e-12
    k = (m**4).mean() / (v**2) - 3.0
    return float(k)

def peak_snr_db(psd_db_vals: np.ndarray) -> float:
    med = np.median(psd_db_vals)
    peak = psd_db_vals.max()
    return float(peak - med)

def band_energies(psd_db_vals: np.ndarray, bands: int = 6) -> np.ndarray:
    L = len(psd_db_vals)
    edges = np.linspace(0, L, bands + 1, dtype=int)
    return np.array([psd_db_vals[edges[i]:edges[i+1]].mean() for i in range(bands)], dtype=np.float32)

def feature_vector(x: np.ndarray, nfft: int | None = None) -> np.ndarray:
    # nfft == len(x) for simple 1:1 reasoning
    if nfft is None:
        nfft = len(x)

    psd = psd_db(x, nfft=nfft)
    comp = compress_bins(psd, n_bins=64)
    ent  = spectral_entropy(psd)
    kur  = kurtosis_mag(x)
    snr  = peak_snr_db(psd)
    bands = band_energies(psd, bands=6)

    return np.concatenate([comp, bands, np.array([ent, kur, snr], dtype=np.float32)])
