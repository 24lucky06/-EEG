"""
hardware_reader.py

硬件采集占位模块。

你们后面真正接硬件时，把这里的 SimulatedEEGReader 替换成串口、蓝牙、USB 或 SDK 读取即可。
关键要求：每次输出一个 30 秒 epoch，shape=(n_channels, n_samples)，单位尽量统一为 V。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence
import time

import numpy as np
from scipy.signal import resample_poly
import serial


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


class CustomSerialEEGReader:
    """Read the project's CP210x serial EEG board.

    The currently observed 19-byte packet layout is:
    A0, sequence, four signed 24-bit channel values, four auxiliary bytes, C0.
    Output data uses volts and is resampled to the configured target rate.
    """

    FRAME_LEN = 19
    FRAME_HEAD = 0xA0
    FRAME_TAIL = 0xC0
    HARDWARE_CHANNELS = 4

    def __init__(
        self,
        config: HardwareConfig,
        serial_port: str,
        baud_rate: int = 230400,
        source_sfreq: float = 250.0,
        channel_indices: Sequence[int] = (1, 0),
        volts_per_count: float = 4.5 / 24 / (2**23 - 1),
        start_command: bytes = b"b",
        stop_command: bytes = b"s",
    ) -> None:
        self.config = config
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.source_sfreq = float(source_sfreq)
        self.target_sfreq = float(config.sfreq)
        self.channel_indices = list(channel_indices)
        self.volts_per_count = float(volts_per_count)
        self.start_command = start_command
        self.stop_command = stop_command

        for idx in self.channel_indices:
            if idx < 0 or idx >= self.HARDWARE_CHANNELS:
                raise ValueError(
                    f"硬件通道索引 {idx} 超出范围；当前帧包含 {self.HARDWARE_CHANNELS} 路通道"
                )

        if len(self.channel_indices) != len(config.channel_names):
            raise ValueError(
                f"选择了 {len(self.channel_indices)} 个硬件通道，但配置需要 "
                f"{len(config.channel_names)} 个通道：{list(config.channel_names)}"
            )

    @staticmethod
    def _decode_signed_24(data: bytes) -> int:
        value = int.from_bytes(data, byteorder="big", signed=False)
        return value - (1 << 24) if value & (1 << 23) else value

    def decode_frame(self, frame: bytes) -> np.ndarray:
        if (
            len(frame) != self.FRAME_LEN
            or frame[0] != self.FRAME_HEAD
            or frame[-1] != self.FRAME_TAIL
        ):
            raise ValueError("无效的 19 字节 EEG 数据帧")

        values = [
            self._decode_signed_24(frame[2 + 3 * idx : 5 + 3 * idx])
            for idx in self.channel_indices
        ]
        return np.asarray(values, dtype=np.float64) * self.volts_per_count

    def _iter_frames(self, ser: serial.Serial) -> Iterator[bytes | None]:
        buffer = bytearray()
        while True:
            chunk = ser.read(512)
            if not chunk:
                yield None
                continue
            buffer.extend(chunk)

            while True:
                try:
                    start = buffer.index(self.FRAME_HEAD)
                except ValueError:
                    buffer.clear()
                    break
                if start:
                    del buffer[:start]
                if len(buffer) < self.FRAME_LEN:
                    break

                frame = bytes(buffer[: self.FRAME_LEN])
                del buffer[: self.FRAME_LEN]
                if frame[-1] == self.FRAME_TAIL:
                    yield frame
                else:
                    buffer[:0] = frame[1:]

    def iter_chunks(
        self, chunk_seconds: float = 1.0, timeout_seconds: float = 3.0
    ) -> Iterator[np.ndarray | None]:
        source_chunk_samples = max(1, int(self.source_sfreq * chunk_seconds))
        with serial.Serial(
            self.serial_port,
            self.baud_rate,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=0.2,
        ) as ser:
            ser.reset_input_buffer()
            ser.write(self.start_command)
            ser.flush()
            print(
                f"自定义 EEG 板已连接：{self.serial_port} @ {self.baud_rate}，"
                f"{self.source_sfreq:g}Hz -> {self.target_sfreq:g}Hz"
            )

            frame_iter = self._iter_frames(ser)
            samples: list[np.ndarray] = []
            resample_history = np.empty(
                (len(self.channel_indices), 0), dtype=np.float64
            )
            resample_history_samples = max(1, int(self.source_sfreq))
            last_frame_at = time.monotonic()
            try:
                while True:
                    frame = next(frame_iter)
                    if frame is None:
                        if time.monotonic() - last_frame_at >= timeout_seconds:
                            yield None
                            last_frame_at = time.monotonic()
                        continue

                    last_frame_at = time.monotonic()
                    samples.append(self.decode_frame(frame))
                    if len(samples) < source_chunk_samples:
                        continue

                    data = np.asarray(samples[:source_chunk_samples]).T
                    del samples[:source_chunk_samples]
                    if self.source_sfreq != self.target_sfreq:
                        combined = np.concatenate([resample_history, data], axis=1)
                        resampled = resample_poly(
                            combined,
                            int(self.target_sfreq),
                            int(self.source_sfreq),
                            axis=1,
                        )
                        output_samples = max(
                            1,
                            int(round(data.shape[1] * self.target_sfreq / self.source_sfreq)),
                        )
                        data = resampled[:, -output_samples:]
                        resample_history = combined[:, -resample_history_samples:]
                    yield data.astype(np.float32)
            finally:
                ser.write(self.stop_command)
                ser.flush()
