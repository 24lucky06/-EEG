"""
shared/feature_definitions.py

离线训练与实时预测共用的 EEG 人工特征计算函数。

此文件是 offline_training/01_extract_features_LIGHT_FIR_vscode.py
与 realtime_system/realtime_feature_extractor.py 的唯一真相来源。
修改特征计算公式时只需改这一处。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew


BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "sigma": (11, 16),
    "beta": (13, 30),
}


def trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    """兼容 NumPy 新旧版本的梯形积分。"""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def calculate_hjorth_parameters(signal: np.ndarray) -> Tuple[float, float, float]:
    """计算 Hjorth 三个特征：Activity、Mobility、Complexity。"""
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)
    var_zero = float(np.var(signal))
    var_d1 = float(np.var(first_deriv))
    var_d2 = float(np.var(second_deriv))

    activity = var_zero
    mobility = float(np.sqrt(var_d1 / var_zero)) if var_zero > 1e-20 else 0.0
    complexity = float(np.sqrt(var_d2 / var_d1) / mobility) if (var_d1 > 1e-20 and mobility > 1e-20) else 0.0
    return activity, mobility, complexity


def calculate_pfd(signal: np.ndarray) -> float:
    """计算 Petrosian Fractal Dimension，用于描述信号复杂度。"""
    diff = np.diff(signal)
    if len(diff) < 2:
        return 0.0
    n_delta = int(np.sum(diff[1:] * diff[:-1] < 0))
    n = len(signal)
    if n_delta == 0 or n <= 1:
        return 0.0
    return float(np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_delta))))


def calculate_shannon_entropy(signal: np.ndarray, num_bins: int = 10) -> float:
    """计算 Shannon entropy。这里使用直方图概率，而不是 density，避免熵值异常。"""
    counts, _ = np.histogram(signal, bins=num_bins, density=False)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


def extract_one_channel_features(ch_data: np.ndarray, sfreq: float) -> Tuple[List[float], List[str]]:
    """对单个通道、单个 30 秒 epoch 提取一组人工 EEG 特征。"""
    features: List[float] = []
    names: List[str] = []

    ch_var = float(np.var(ch_data))
    if ch_var > 1e-20:
        skew_value = float(skew(ch_data, nan_policy="omit"))
        kurtosis_value = float(kurtosis(ch_data, nan_policy="omit"))
    else:
        skew_value = 0.0
        kurtosis_value = 0.0

    time_features = {
        "mean": float(np.mean(ch_data)),
        "var": ch_var,
        "skew": skew_value,
        "kurtosis": kurtosis_value,
        "ptp": float(np.ptp(ch_data)),
    }
    for k, v in time_features.items():
        names.append(k)
        features.append(v)

    nperseg = max(8, min(len(ch_data), int(sfreq * 2)))
    freqs, psd = welch(ch_data, fs=sfreq, nperseg=nperseg)

    band_powers: Dict[str, float] = {}
    for band_name, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs <= high)
        band_powers[band_name] = trapezoid_integral(psd[mask], freqs[mask]) if np.any(mask) else 0.0

    total_power = float(sum(band_powers.values()) + 1e-12)

    for band_name in BANDS:
        names.append(f"{band_name}_abs")
        features.append(band_powers[band_name])

    for band_name in BANDS:
        names.append(f"{band_name}_rel")
        features.append(band_powers[band_name] / total_power)

    act, mob, comp = calculate_hjorth_parameters(ch_data)
    names.extend(["hjorth_activity", "hjorth_mobility", "hjorth_complexity"])
    features.extend([act, mob, comp])

    names.extend(["shannon_entropy", "petrosian_fd"])
    features.extend([calculate_shannon_entropy(ch_data), calculate_pfd(ch_data)])

    # 防止常数信号导致 skew/kurtosis 为 NaN。
    features = [float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for v in features]
    return features, names
