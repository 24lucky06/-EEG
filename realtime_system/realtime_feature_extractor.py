"""
realtime_feature_extractor.py

实时端特征提取模块。

它的目标是：让硬件实时采集到的双导联 EEG，在每 30 秒形成一个 epoch 后，
提取出与 offline_training/01_extract_features_LIGHT_FIR_vscode.py 尽量一致的人工特征。

注意：
- 离线 01 脚本使用 MNE 对连续信号做 FIR 滤波后再切 epoch；
- 实时端通常只能拿到当前窗口，因此这里对当前窗口做 FIR 近似处理；
- 真正接硬件时，必须保证通道顺序、采样率、滤波范围、特征顺序与训练模型一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.signal import filtfilt, firwin, welch
from scipy.stats import kurtosis, skew


BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "sigma": (11, 16),
    "beta": (13, 30),
}


@dataclass
class RealtimeFeatureConfig:
    channel_names: Sequence[str]
    sfreq: float = 100.0
    epoch_seconds: int = 30
    low_freq: float = 0.3
    high_freq: float = 35.0
    clip_uv: float = 150.0
    apply_fir: bool = True


def calculate_hjorth_parameters(signal: np.ndarray) -> Tuple[float, float, float]:
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
    diff = np.diff(signal)
    if len(diff) < 2:
        return 0.0
    n_delta = int(np.sum(diff[1:] * diff[:-1] < 0))
    n = len(signal)
    if n_delta == 0 or n <= 1:
        return 0.0
    return float(np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_delta))))


def calculate_shannon_entropy(signal: np.ndarray, num_bins: int = 10) -> float:
    counts, _ = np.histogram(signal, bins=num_bins, density=False)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


def extract_one_channel_features(ch_data: np.ndarray, sfreq: float) -> Tuple[List[float], List[str]]:
    features: List[float] = []
    names: List[str] = []

    time_features = {
        "mean": float(np.mean(ch_data)),
        "var": float(np.var(ch_data)),
        "skew": float(skew(ch_data, nan_policy="omit")),
        "kurtosis": float(kurtosis(ch_data, nan_policy="omit")),
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
        band_powers[band_name] = float(np.trapz(psd[mask], freqs[mask])) if np.any(mask) else 0.0

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

    features = [float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for v in features]
    return features, names


class RealtimeFeatureExtractor:
    def __init__(self, config: RealtimeFeatureConfig):
        self.config = config
        self.channel_names = list(config.channel_names)
        self.expected_samples = int(config.sfreq * config.epoch_seconds)
        self._fir_b = self._make_fir_filter() if config.apply_fir else None

    def _make_fir_filter(self) -> np.ndarray:
        nyq = self.config.sfreq / 2.0
        low = self.config.low_freq / nyq
        high = self.config.high_freq / nyq
        # 101 taps 对 30 秒窗口较稳妥；硬件端可以后续换成流式滤波状态。
        return firwin(numtaps=101, cutoff=[low, high], pass_zero=False)

    def preprocess_epoch(self, epoch_data: np.ndarray) -> np.ndarray:
        """输入 shape=(n_channels, n_samples)，单位建议为 V。"""
        data = np.asarray(epoch_data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError("epoch_data 必须是二维数组，shape=(n_channels, n_samples)")
        if data.shape[0] != len(self.channel_names):
            raise ValueError(f"通道数不匹配：收到 {data.shape[0]}，期望 {len(self.channel_names)}")
        if data.shape[1] < self.expected_samples:
            raise ValueError(f"当前窗口太短：收到 {data.shape[1]} 点，至少需要 {self.expected_samples} 点")

        data = data[:, : self.expected_samples]
        clip_v = self.config.clip_uv * 1e-6
        data = np.clip(data, -clip_v, clip_v)

        if self._fir_b is not None:
            filtered = []
            for ch in data:
                filtered.append(filtfilt(self._fir_b, [1.0], ch).astype(np.float32))
            data = np.vstack(filtered)
        return data

    def extract_epoch_features(self, epoch_data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        data = self.preprocess_epoch(epoch_data)
        all_features: List[float] = []
        all_names: List[str] = []
        for ch_idx, ch_name in enumerate(self.channel_names):
            ch_features, ch_feature_names = extract_one_channel_features(data[ch_idx], self.config.sfreq)
            all_features.extend(ch_features)
            all_names.extend([f"{ch_name}__{name}" for name in ch_feature_names])
        return np.asarray(all_features, dtype=np.float32), all_names
