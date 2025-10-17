from __future__ import annotations
import numpy as np


# DEFINE FEATURES ---
N_PSD_COMP = 64
N_BANDS = 6

def get_feature_names(n_psd: int = N_PSD_COMP,
                      n_bands: int = N_BANDS) -> list[str]:
    return (
        [f"psd_comp_{i}" for i in range(n_psd)] +
        [f"band_{i}" for i in range(n_bands)] +
        ["entropy", "kurtosis", "peak_snr"]
    )

FEATURE_NAMES = tuple(get_feature_names())

# PSD CALC ---
def psd_db(x: np.ndarray, nfft: int | None = None) -> np.ndarray:
    """
    Windowed FFT-based PSD in dB for IQ
    """
    if nfft is None:
        nfft = len(x)

    w = np.hanning(len(x)).astype(np.float64, copy=False)
    xw = w * x.astype(np.complex128, copy=False)

    X = np.fft.fft(xw, n=nfft)
    X = np.fft.fftshift(X)

    P = (np.abs(X) ** 2) / nfft
    return 10.0 * np.log10(P + 1e-12)

def compress_bins(psd: np.ndarray, n_bins: int = 64) -> np.ndarray:
    L = len(psd)
    edges = np.linspace(0, L, n_bins + 1, dtype=int)
    out = []
    for i in range(n_bins):
        a, b = edges[i], edges[i+1]
        out.append(psd[a:b].mean() if b > a else psd[a])
    return np.array(out, dtype=np.float32)

def band_energies(psd: np.ndarray, bands: int = 6) -> np.ndarray:
    L = len(psd)
    edges = np.linspace(0, L, bands + 1, dtype=int)
    return np.array([psd[edges[i]:edges[i+1]].mean() for i in range(bands)], dtype=np.float32)

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

def peak_info(psd_db_vals: np.ndarray) -> tuple[float, int]:
    peak_idx = int(np.argmax(psd_db_vals))
    med = np.median(psd_db_vals)
    return float(psd_db_vals[peak_idx] - med), peak_idx

# EXTRACTOR ---
def extract_features(x: np.ndarray) -> dict:
    N = len(x)
    psd = psd_db(x, nfft=N)                # length == N (fftshifted)

    comp  = compress_bins(psd, 64)
    bands = band_energies(psd, 6)
    ent   = spectral_entropy(psd)
    kur   = kurtosis_mag(x)
    snr, peak_idx = peak_info(psd)

    return {
        "psd_comp": comp,
        "bands": bands,
        "entropy": ent,
        "kurtosis": kur,
        "peak_snr": snr,
        "peak_bin": peak_idx,      # index into fftshifted PSD of length N
        "psd_len": int(len(psd)),  # == N
        "nfft": int(N),            # <— explicit
        "fftshifted": True
    }

def stack_for_model(f: dict) -> np.ndarray:
    return np.concatenate([
        f["psd_comp"],
        f["bands"],
        np.array([f["entropy"], f["kurtosis"], f["peak_snr"]], dtype=np.float32),
    ])
