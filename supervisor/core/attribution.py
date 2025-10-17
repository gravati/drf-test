import numpy as np

def build_attribution(
    feat: dict,
    fv: np.ndarray,
    z_full: np.ndarray,
    feature_names: list[str],
    sample_rate_hz: float,
    center_freq_hz: float,
    n_top: int = 3,
) -> dict:
    idxs = np.argsort(z_full)[::-1][:n_top]
    top_k = [{"feature": feature_names[j], "z": float(z_full[j]), "value": float(fv[j])} for j in idxs]

    peak_bin = int(feat["peak_bin"])
    N = int(feat.get("nfft", feat.get("psd_len", len(fv))))  # fallback
    fftshifted = bool(feat.get("fftshifted", True))

    freqs = np.fft.fftfreq(N, d=1.0 / sample_rate_hz)
    if fftshifted:
        freqs = np.fft.fftshift(freqs)

    peak_bin = max(0, min(peak_bin, len(freqs) - 1))
    peak_freq_hz = float(freqs[peak_bin])
    abs_rf_hz = float(center_freq_hz + peak_freq_hz)

    return {
        "top_features": top_k,
        "peak_bin": peak_bin,
        "peak_freq_hz": peak_freq_hz,
        "peak_freq_rf_hz": abs_rf_hz,
        "entropy": float(feat["entropy"]),
        "peak_snr_db": float(feat["peak_snr"]),
    }
