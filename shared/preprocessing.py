"""Shared EEG preprocessing helpers for offline training and realtime prediction."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt


def attenuate_slow_eye_artifacts(
    data_v: np.ndarray,
    sfreq: float,
    low_freq: float = 0.5,
    high_freq: float = 4.0,
    attenuation: float = 0.75,
) -> np.ndarray:
    """Subtract a low-frequency eye-movement-like component from EEG data.

    This is a conservative dual-channel fallback, not ICA. It should be used
    only when the model was trained with the same preprocessing.
    """
    data = np.asarray(data_v, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] < int(sfreq):
        return data

    nyquist = sfreq / 2.0
    low = max(0.01, float(low_freq))
    high = min(float(high_freq), nyquist * 0.95)
    strength = float(np.clip(attenuation, 0.0, 1.0))
    if low >= high or strength <= 0.0:
        return data

    sos = butter(2, [low, high], btype="bandpass", fs=sfreq, output="sos")
    slow_component = sosfiltfilt(sos, data, axis=1)
    return (data - strength * slow_component).astype(np.float32)


def epoch_artifact_flags(epoch_v: np.ndarray, ptp_limit_uv: float) -> Tuple[bool, str]:
    """Return whether an epoch should be rejected by a simple amplitude rule."""
    if ptp_limit_uv <= 0:
        return False, ""
    epoch_uv = np.asarray(epoch_v, dtype=np.float32) * 1e6
    centered = epoch_uv - np.median(epoch_uv, axis=1, keepdims=True)
    ptp = np.ptp(centered, axis=1)
    if np.any(ptp > ptp_limit_uv):
        return True, f"ptp>{ptp_limit_uv:g}uV"
    return False, ""
