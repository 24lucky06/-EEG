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
from typing import List, Sequence, Tuple

import sys
from pathlib import Path

import numpy as np
from scipy.signal import filtfilt, firwin

_shared_root = Path(__file__).resolve().parent.parent
if str(_shared_root) not in sys.path:
    sys.path.insert(0, str(_shared_root))
from shared.feature_definitions import (
    extract_one_channel_features,
)
from shared.preprocessing import attenuate_slow_eye_artifacts
from config.settings import CLIP_UV, EPOCH_SECONDS, HIGH_FREQ, LOW_FREQ, SFREQ


@dataclass
class RealtimeFeatureConfig:
    channel_names: Sequence[str]
    sfreq: float = SFREQ
    epoch_seconds: int = EPOCH_SECONDS
    low_freq: float = LOW_FREQ
    high_freq: float = HIGH_FREQ
    clip_uv: float = CLIP_UV
    apply_fir: bool = True
    suppress_eye_artifacts: bool = False
    eye_low_freq: float = 0.5
    eye_high_freq: float = 4.0
    eye_attenuation: float = 0.75


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

        if self._fir_b is not None:
            filtered = []
            for ch in data:
                filtered.append(filtfilt(self._fir_b, [1.0], ch).astype(np.float32))
            data = np.vstack(filtered)
        if self.config.suppress_eye_artifacts:
            data = attenuate_slow_eye_artifacts(
                data,
                self.config.sfreq,
                low_freq=self.config.eye_low_freq,
                high_freq=self.config.eye_high_freq,
                attenuation=self.config.eye_attenuation,
            )
        clip_v = self.config.clip_uv * 1e-6
        data = np.clip(data, -clip_v, clip_v)
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
