def make_explanation(row):
    """
    Turn a WARN/ALERT row into a short, human-readable sentence.
    Assumes row has 'state', 'score', and 'attribution' keys.
    """
    state = row["state"]
    score = row["score"]
    att = row.get("attribution", {})

    # Peak frequency (Hz offset and absolute if available)
    peak_freq_hz = att.get("peak_freq_hz", None)
    abs_rf_hz = att.get("peak_freq_rf_hz", None)

    # Optional extras
    peak_snr = att.get("peak_snr_db", None)
    entropy = att.get("entropy", None)

    # Start building sentence
    parts = []

    # State and score
    parts.append(f"{state} event (score={score:.3f}):")

    # Tone location
    if peak_freq_hz is not None:
        parts.append(f"peak at {peak_freq_hz:+.0f} Hz offset")
        if abs_rf_hz is not None:
            parts.append(f"(~{abs_rf_hz/1e6:.6f} MHz RF)")
        parts.append(",")

    # Tone strength
    if peak_snr is not None:
        parts.append(f"SNR +{peak_snr:.1f} dB,")

    # Entropy context
    if entropy is not None:
        if entropy < 0.4:
            parts.append(f"low entropy ({entropy:.2f}, tonal).")
        else:
            parts.append(f"entropy {entropy:.2f}.")

    return " ".join(parts)
