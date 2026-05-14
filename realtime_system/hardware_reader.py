"""
hardware_reader.py

硬件采集占位模块。

你们后面真正接硬件时，把这里的 SimulatedEEGReader 替换成串口、蓝牙、USB 或 SDK 读取即可。
关键要求：每次输出一个 30 秒 epoch，shape=(n_channels, n_samples)，单位尽量统一为 V。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np


@dataclass
class HardwareConfig:
    channel_names: Sequence[str]
    sfreq: float = 100.0
    epoch_seconds: int = 30


class SimulatedEEGReader:
    """模拟双导联 EEG，用于在没有硬件时测试实时流程是否能跑通。"""

    def __init__(self, config: HardwareConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.samples = int(config.sfreq * config.epoch_seconds)
        self.t = np.arange(self.samples) / config.sfreq

    def iter_epochs(self) -> Iterator[np.ndarray]:
        while True:
            epoch = []
            for idx, _ in enumerate(self.config.channel_names):
                alpha = 25e-6 * np.sin(2 * np.pi * (9 + idx) * self.t)
                theta = 12e-6 * np.sin(2 * np.pi * 6 * self.t)
                noise = self.rng.normal(0, 8e-6, size=self.samples)
                epoch.append(alpha + theta + noise)
            yield np.asarray(epoch, dtype=np.float32)
