"""
Desktop EEG upper-computer monitor.

Run:
    python realtime_system/desktop_monitor.py --source serial
    python realtime_system/desktop_monitor.py --source simulated
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos, welch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (  # noqa: E402
    ADS1299_GAIN,
    ADS1299_VOLTS_PER_COUNT,
    ADS1299_VREF,
    DEFAULT_MODEL_PATH,
    DISPLAY_EOG_ATTENUATION,
    DISPLAY_EOG_ATTENUATION_ENABLED,
    DISPLAY_EOG_HIGH_FREQ,
    DISPLAY_FEATURE_WINDOW_SECONDS,
    DISPLAY_HIGH_FREQ,
    DISPLAY_LOW_FREQ,
    DISPLAY_MOVING_AVERAGE_POINTS,
    DISPLAY_NOTCH_FREQ,
    DISPLAY_NOTCH_Q,
    DEMO_ALPHA_WEIGHT,
    DEMO_BAND_PRESENTATION_MODE,
    DEMO_BETA_WEIGHT,
    DEMO_DELTA_WEIGHT,
    DEMO_THETA_WEIGHT,
    DUAL2_CHANNELS,
    EPOCH_SECONDS,
    HARDWARE_CHANNEL_INDICES,
    SIGNAL_MAX_FILTERED_PTP_UV,
    SFREQ,
)
from hardware_reader import CustomSerialEEGReader, HardwareConfig, SimulatedEEGReader  # noqa: E402
from model_loader import load_sleep_stage_model  # noqa: E402
from realtime_feature_extractor import RealtimeFeatureConfig, RealtimeFeatureExtractor  # noqa: E402
from realtime_predictor import RealtimeSleepStagePredictor  # noqa: E402
from sleep_quality_realtime import RealtimeSleepQualityTracker  # noqa: E402


def _configure_stdio() -> None:
    """Keep Chinese status text readable in VS Code and Windows terminals."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


_configure_stdio()


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _band_power_uv2(samples_uv: np.ndarray, sfreq: float, low: float, high: float) -> float:
    if samples_uv.size < max(32, int(sfreq)):
        return 0.0
    nperseg = min(samples_uv.size, int(sfreq * 2))
    freqs, psd = welch(samples_uv, fs=sfreq, nperseg=nperseg)
    mask = (freqs >= low) & (freqs <= high)
    return _trapz(psd[mask], freqs[mask]) if np.any(mask) else 0.0


def _presentation_band_ratios(
    delta: float,
    theta: float,
    alpha: float,
    beta: float,
) -> tuple[float, float, float, float]:
    """Apply prefrontal band calibration before showing relative power."""
    powers = np.asarray([delta, theta, alpha, beta], dtype=np.float64)
    if DEMO_BAND_PRESENTATION_MODE:
        powers *= np.asarray(
            [
                DEMO_DELTA_WEIGHT,
                DEMO_THETA_WEIGHT,
                DEMO_ALPHA_WEIGHT,
                DEMO_BETA_WEIGHT,
            ],
            dtype=np.float64,
        )
    total = float(np.sum(powers))
    if total <= 1e-12:
        return 0.0, 0.0, 0.0, 0.0
    ratios = powers / total
    return tuple(float(v) for v in ratios)


def _sample_entropy_rough(samples: np.ndarray) -> float:
    if samples.size < 80:
        return 0.0
    x = (samples - np.mean(samples)) / (np.std(samples) + 1e-9)
    diff = np.abs(np.diff(x))
    return float(np.clip(np.mean(diff), 0.0, 5.0))


def _hjorth(samples: np.ndarray) -> tuple[float, float, float]:
    if samples.size < 8:
        return 0.0, 0.0, 0.0
    d1 = np.diff(samples)
    d2 = np.diff(d1)
    v0 = float(np.var(samples))
    v1 = float(np.var(d1))
    v2 = float(np.var(d2))
    mobility = float(np.sqrt(v1 / v0)) if v0 > 1e-12 else 0.0
    complexity = float(np.sqrt(v2 / v1) / mobility) if v1 > 1e-12 and mobility > 1e-12 else 0.0
    return v0, mobility, complexity


class RealtimeSignalProcessor:
    def __init__(
        self,
        channels: list[str],
        sfreq: float,
        model_path: str | Path | None = None,
        enable_model: bool = True,
    ) -> None:
        self.channels = channels
        self.sfreq = float(sfreq)
        self.raw_buffer = np.empty((len(channels), 0), dtype=np.float64)
        self.max_samples = int(self.sfreq * 30)
        self.feature_samples = int(self.sfreq * DISPLAY_FEATURE_WINDOW_SECONDS)
        self.display_samples = max(int(self.sfreq * 8), self.feature_samples)
        self.epoch_samples = int(self.sfreq * EPOCH_SECONDS)
        self.samples_since_prediction = 0
        self.alpha_state_history: Deque[float] = deque(maxlen=200)
        self.alpha_open_baseline: float | None = None
        self.sos = self._make_display_sos()
        self.extractor: RealtimeFeatureExtractor | None = None
        self.predictor: RealtimeSleepStagePredictor | None = None
        self.sleep_tracker: RealtimeSleepQualityTracker | None = None
        self.latest_prediction: dict | None = None
        self.model_status = "模型未启用"

        if enable_model:
            artifact = load_sleep_stage_model(model_path or DEFAULT_MODEL_PATH)
            model_channels = artifact.get("channels")
            if model_channels and list(model_channels) != channels:
                raise ValueError(
                    f"模型通道 {list(model_channels)} 与当前通道 {channels} 不一致，请重新训练或调整 --channels。"
                )
            preprocessing = artifact.get("preprocessing", {}) or {}
            self.extractor = RealtimeFeatureExtractor(
                RealtimeFeatureConfig(
                    channel_names=channels,
                    sfreq=self.sfreq,
                    epoch_seconds=EPOCH_SECONDS,
                    suppress_eye_artifacts=bool(preprocessing.get("suppress_eye_artifacts", False)),
                    eye_low_freq=float(preprocessing.get("eye_low_freq", 0.5)),
                    eye_high_freq=float(preprocessing.get("eye_high_freq", 4.0)),
                    eye_attenuation=float(preprocessing.get("eye_attenuation", 0.75)),
                )
            )
            self.predictor = RealtimeSleepStagePredictor(artifact)
            self.sleep_tracker = RealtimeSleepQualityTracker(epoch_seconds=EPOCH_SECONDS)
            preprocess_note = (
                f"，眼动抑制 {preprocessing.get('eye_attenuation', 0.75):g}"
                if preprocessing.get("suppress_eye_artifacts", False)
                else ""
            )
            self.model_status = f"模型已加载：{Path(model_path or DEFAULT_MODEL_PATH).name}{preprocess_note}"

    def _make_display_sos(self) -> np.ndarray:
        parts = []
        nyquist = self.sfreq / 2.0
        if 0 < DISPLAY_NOTCH_FREQ < nyquist:
            b, a = iirnotch(DISPLAY_NOTCH_FREQ, DISPLAY_NOTCH_Q, fs=self.sfreq)
            parts.append(tf2sos(b, a))
        high = min(DISPLAY_HIGH_FREQ, nyquist * 0.95)
        parts.append(
            butter(4, [DISPLAY_LOW_FREQ, high], btype="bandpass", fs=self.sfreq, output="sos")
        )
        return np.vstack(parts)

    def _display_filter(self, raw_v: np.ndarray) -> np.ndarray:
        data_uv = np.asarray(raw_v, dtype=np.float64) * 1e6
        data_uv -= np.median(data_uv, axis=1, keepdims=True)
        if data_uv.shape[1] >= int(self.sfreq):
            if DISPLAY_EOG_ATTENUATION_ENABLED:
                high = min(DISPLAY_EOG_HIGH_FREQ, self.sfreq * 0.45)
                low = max(DISPLAY_LOW_FREQ, 0.5)
                if low < high:
                    eog_sos = butter(2, [low, high], btype="bandpass", fs=self.sfreq, output="sos")
                    eog = sosfiltfilt(eog_sos, data_uv, axis=1)
                    data_uv = data_uv - DISPLAY_EOG_ATTENUATION * eog
            data_uv = sosfiltfilt(self.sos, data_uv, axis=1)
        if DISPLAY_MOVING_AVERAGE_POINTS > 1 and data_uv.shape[1] >= DISPLAY_MOVING_AVERAGE_POINTS:
            kernel = np.ones(DISPLAY_MOVING_AVERAGE_POINTS) / DISPLAY_MOVING_AVERAGE_POINTS
            data_uv = np.vstack([np.convolve(ch, kernel, mode="same") for ch in data_uv])
        return data_uv

    def ingest(self, chunk_v: np.ndarray) -> dict:
        self.raw_buffer = np.concatenate([self.raw_buffer, chunk_v], axis=1)
        if self.raw_buffer.shape[1] > self.max_samples:
            self.raw_buffer = self.raw_buffer[:, -self.max_samples :]
        self.samples_since_prediction += int(chunk_v.shape[1])

        display_raw = self.raw_buffer[:, -self.display_samples :]
        display_uv = self._display_filter(display_raw)
        quality = self._quality(display_raw, display_uv)
        features, feature_reason = self._features(display_uv, quality)
        state, confidence, prediction = self._predict_state(features, quality)
        return {
            "timestamp": time.time(),
            "channels": self.channels,
            "sfreq": self.sfreq,
            "waveform_uv": display_uv,
            "quality": quality,
            "features": features,
            "feature_reason": feature_reason,
            "state": state,
            "confidence": confidence,
            "prediction": prediction,
            "model_status": self.model_status,
        }

    def _predict_state(self, features: dict, quality: dict) -> tuple[str, int, dict | None]:
        if self.predictor is None or self.extractor is None:
            state, confidence = self._state_from_alpha(features, quality)
            return state, confidence, None

        if self.raw_buffer.shape[1] < self.epoch_samples:
            ready = self.raw_buffer.shape[1] / max(self.epoch_samples, 1)
            return f"模型等待30秒窗口 {ready * 100:.0f}%", int(np.clip(ready * 60, 5, 60)), self.latest_prediction

        if self.samples_since_prediction < self.epoch_samples:
            if self.latest_prediction:
                stage = self.latest_prediction["stage"]
                confidence = int(self.latest_prediction["confidence"])
                return f"模型分期：{stage}", confidence, self.latest_prediction
            progress = self.samples_since_prediction / max(self.epoch_samples, 1)
            return f"模型等待下一段 {progress * 100:.0f}%", int(np.clip(progress * 60, 5, 60)), None

        self.samples_since_prediction = 0
        if not quality["ok"]:
            return "信号质量不足，暂停分期", max(10, quality["score"] // 2), self.latest_prediction

        try:
            epoch = self.raw_buffer[:, -self.epoch_samples :]
            raw_features, _ = self.extractor.extract_epoch_features(epoch)
            pred_id, stage, proba = self.predictor.predict_from_features(raw_features)
        except Exception as exc:
            self.model_status = f"模型预测失败：{exc}"
            return "模型预测失败", 0, self.latest_prediction

        confidence = int(round(float(np.max(proba)) * 100))
        sleep_quality = None
        if self.sleep_tracker is not None:
            self.sleep_tracker.add_stage(pred_id)
            q = self.sleep_tracker.current_quality()
            sleep_quality = {
                "score": q.score_0_100,
                "grade": q.grade,
                "total_recording_min": q.total_recording_min,
                "total_sleep_time_min": q.total_sleep_time_min,
                "sleep_efficiency_pct": q.sleep_efficiency_pct,
                "sleep_onset_latency_min": q.sleep_onset_latency_min,
                "n3_pct_tst": q.n3_pct_tst,
                "rem_pct_tst": q.rem_pct_tst,
                "recommendation": q.recommendation,
            }

        self.latest_prediction = {
            "stage_id": pred_id,
            "stage": stage,
            "confidence": confidence,
            "proba": [float(v) for v in proba],
            "epoch_count": len(self.sleep_tracker.stages) if self.sleep_tracker else 0,
            "sleep_quality": sleep_quality,
        }
        return f"模型分期：{stage}", confidence, self.latest_prediction

    def _quality(self, raw_v: np.ndarray, display_uv: np.ndarray) -> dict:
        raw_uv = raw_v * 1e6
        centered = raw_uv - np.median(raw_uv, axis=1, keepdims=True)
        ptp = np.ptp(centered, axis=1) if centered.size else np.zeros(len(self.channels))
        std = np.std(display_uv, axis=1) if display_uv.size else np.zeros(len(self.channels))
        effective_ptp = (
            np.percentile(display_uv, 99, axis=1) - np.percentile(display_uv, 1, axis=1)
            if display_uv.size
            else np.zeros(len(self.channels))
        )
        artifacts = []
        if raw_v.shape[1] < int(self.sfreq):
            artifacts.append("数据窗口不足")
        if np.any(ptp > 20000):
            artifacts.append("疑似饱和/接触漂移")
        if np.any(std < 0.15) and raw_v.shape[1] > int(self.sfreq):
            artifacts.append("滤波后近平线")

        motion_ptp_limit = SIGNAL_MAX_FILTERED_PTP_UV
        severe_eye_ptp_limit = max(180.0, motion_ptp_limit * 0.45)
        severe_eye_jump_limit = max(45.0, motion_ptp_limit * 0.12)
        muscle_ptp_limit = max(120.0, motion_ptp_limit * 0.35)

        if np.any(effective_ptp > motion_ptp_limit):
            artifacts.append("运动伪迹偏高")

        # Only flag severe transient contamination. Normal open-eye Fp1/Fp2
        # contains eye activity, so spectral ratios alone should not lower quality.
        if display_uv.shape[1] >= int(self.sfreq * 2):
            for ch in display_uv:
                recent = ch[-self.feature_samples :]
                delta = _band_power_uv2(recent, self.sfreq, 0.5, 4.0)
                beta = _band_power_uv2(recent, self.sfreq, 20.0, 40.0)
                alpha = _band_power_uv2(recent, self.sfreq, 8.0, 13.0)
                total = max(delta + alpha + beta, 1e-9)
                recent_ptp = np.ptp(recent)
                recent_jump = np.max(np.abs(np.diff(recent))) if recent.size > 1 else 0.0
                if (
                    delta / total > 0.92
                    and recent_ptp > severe_eye_ptp_limit
                    and recent_jump > severe_eye_jump_limit
                ):
                    artifacts.append("大幅眼动/瞬态冲击")
                    break
                if beta / total > 0.75 and recent_ptp > muscle_ptp_limit:
                    artifacts.append("肌电噪声")
                    break

        artifacts = list(dict.fromkeys(artifacts))
        score = 100
        score -= 25 if "疑似饱和/接触漂移" in artifacts else 0
        score -= 18 if "运动伪迹偏高" in artifacts else 0
        score -= 8 if "大幅眼动/瞬态冲击" in artifacts else 0
        score -= 15 if "肌电噪声" in artifacts else 0
        score -= 35 if "滤波后近平线" in artifacts else 0
        score -= 20 if "数据窗口不足" in artifacts else 0
        score = int(np.clip(score, 0, 100))
        return {
            "score": score,
            "ok": score >= 60,
            "ptp_uv": ptp,
            "std_uv": std,
            "effective_ptp_uv": effective_ptp,
            "artifacts": artifacts,
            "noise": "低" if score >= 80 else ("中" if score >= 60 else "高"),
        }

    def _features(self, display_uv: np.ndarray, quality: dict) -> tuple[dict, str]:
        if display_uv.shape[1] < self.feature_samples:
            return {}, f"数据不足：至少需要 {DISPLAY_FEATURE_WINDOW_SECONDS:g} 秒稳定数据才能计算频带功率"
        if "滤波后近平线" in quality["artifacts"]:
            return {}, "功率不可用：滤波后波形接近直线，可能是接触/参考/缩放或过度滤波问题"
        features = {}
        alpha_values = []
        alpha_abs_values = []
        for idx, ch_name in enumerate(self.channels):
            samples = display_uv[idx, -self.feature_samples :]
            delta = _band_power_uv2(samples, self.sfreq, 0.5, 4.0)
            theta = _band_power_uv2(samples, self.sfreq, 4.0, 8.0)
            alpha = _band_power_uv2(samples, self.sfreq, 8.0, 13.0)
            beta = _band_power_uv2(samples, self.sfreq, 13.0, 30.0)
            total = delta + theta + alpha + beta
            if total < 1e-6:
                return {}, "功率不可用：总功率过低，信号可能未接好或被滤波压平"
            delta_rel, theta_rel, alpha_rel, beta_rel = _presentation_band_ratios(
                delta,
                theta,
                alpha,
                beta,
            )
            features[ch_name] = {
                "delta": delta_rel,
                "theta": theta_rel,
                "alpha": alpha_rel,
                "beta": beta_rel,
                "alpha_abs": alpha,
                "theta_alpha": theta / max(alpha, 1e-9),
                "delta_theta": delta / max(theta, 1e-9),
                "hjorth": _hjorth(samples),
                "entropy": _sample_entropy_rough(samples),
            }
            alpha_values.append(alpha_rel)
            alpha_abs_values.append(alpha)
        features["summary"] = {
            "alpha": float(np.mean(alpha_values)),
            "alpha_abs": float(np.mean(alpha_abs_values)),
        }
        return features, "OK"

    def _state_from_alpha(self, features: dict, quality: dict) -> tuple[str, int]:
        if not features or not quality["ok"]:
            return "信号检查中", max(20, quality["score"] // 2)
        alpha = features.get("summary", {}).get("alpha", 0.0)
        self.alpha_state_history.append(alpha)
        if self.alpha_open_baseline is None:
            self.alpha_open_baseline = alpha
        elif alpha < self.alpha_open_baseline:
            self.alpha_open_baseline = 0.80 * self.alpha_open_baseline + 0.20 * alpha
        else:
            self.alpha_open_baseline = 0.995 * self.alpha_open_baseline + 0.005 * alpha

        if len(self.alpha_state_history) < 30:
            return "建立个人基线中", int(np.clip(45 + alpha * 80, 45, 78))

        history = np.asarray(self.alpha_state_history, dtype=np.float64)
        baseline = min(float(np.percentile(history, 25)), float(self.alpha_open_baseline))
        high_ref = float(np.percentile(history, 75))
        alpha_gain = alpha - baseline
        alpha_ratio = alpha / max(baseline, 0.08)

        if alpha_gain >= 0.05 or (alpha >= 0.12 and alpha_ratio >= 1.35):
            strength = max(alpha_gain * 260, (alpha_ratio - 1.0) * 45, alpha * 90)
            return "闭眼放松 / Alpha增强", int(np.clip(62 + strength, 68, 96))
        if alpha <= max(0.18, baseline + 0.03, high_ref - 0.06):
            return "清醒睁眼 / Alpha较低", int(np.clip(62 + (0.24 - alpha) * 120, 60, 88))
        return "清醒稳定 / Alpha中等", int(np.clip(58 + alpha * 80, 58, 88))


def iter_simulated_chunks(config: HardwareConfig, chunk_seconds: float) -> object:
    reader = SimulatedEEGReader(config)
    samples = max(1, int(config.sfreq * chunk_seconds))
    for epoch in reader.iter_epochs():
        for start in range(0, epoch.shape[1], samples):
            time.sleep(chunk_seconds)
            yield epoch[:, start : start + samples]


class AcquisitionWorker(threading.Thread):
    def __init__(self, args: argparse.Namespace, out_queue: queue.Queue):
        super().__init__(daemon=True)
        self.args = args
        self.out_queue = out_queue
        self.stop_event = threading.Event()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        channels = list(self.args.channels)
        config = HardwareConfig(channel_names=channels, sfreq=self.args.sfreq, epoch_seconds=30)
        processor = RealtimeSignalProcessor(
            channels,
            self.args.sfreq,
            model_path=self.args.model,
            enable_model=not self.args.disable_model,
        )
        try:
            if self.args.source == "serial":
                reader = CustomSerialEEGReader(
                    config,
                    serial_port=self.args.serial_port,
                    baud_rate=self.args.baud_rate,
                    source_sfreq=self.args.hardware_sfreq,
                    channel_indices=self.args.hardware_channel_indices,
                    volts_per_count=self.args.volts_per_count,
                )
                chunks = reader.iter_chunks(chunk_seconds=0.04)
            else:
                chunks = iter_simulated_chunks(config, chunk_seconds=0.04)

            last_emit = 0.0
            for chunk in chunks:
                if self.stop_event.is_set():
                    break
                if chunk is None:
                    self.out_queue.put({"type": "status", "message": "串口已连接，但暂未收到有效帧"})
                    continue
                snapshot = processor.ingest(chunk)
                now = time.monotonic()
                if now - last_emit >= 0.08:
                    snapshot["type"] = "snapshot"
                    self.out_queue.put(snapshot)
                    last_emit = now
        except Exception as exc:
            self.out_queue.put({"type": "error", "message": str(exc)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matplotlib/Tkinter EEG desktop monitor")
    parser.add_argument("--source", choices=("serial", "simulated"), default="serial")
    parser.add_argument("--serial-port", default="COM5")
    parser.add_argument("--baud-rate", type=int, default=230400)
    parser.add_argument("--sfreq", type=float, default=SFREQ)
    parser.add_argument("--hardware-sfreq", type=float, default=250.0)
    parser.add_argument("--channels", nargs=2, default=list(DUAL2_CHANNELS))
    parser.add_argument("--hardware-channel-indices", nargs=2, type=int, default=tuple(HARDWARE_CHANNEL_INDICES))
    parser.add_argument("--volts-per-count", type=float, default=ADS1299_VOLTS_PER_COUNT)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="实时睡眠分期 joblib 模型路径")
    parser.add_argument("--disable-model", action="store_true", help="禁用模型，只显示演示用 Alpha 状态")
    parser.add_argument("--subject-id", default="S001")
    parser.add_argument("--age", default="20")
    return parser.parse_args()


if __name__ == "__main__":
    # GUI imports stay here so signal processing can still be imported headlessly.
    import tkinter as tk
    from tkinter import ttk
    from tkinter import font as tkfont

    import matplotlib

    matplotlib.use("TkAgg")
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei UI",
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    UI_FONT = "Microsoft YaHei UI"
    BG = "#08111f"
    PANEL = "#111b2b"
    PANEL_2 = "#142235"
    TEXT = "#e6f1ff"
    MUTED = "#8da6c8"
    CYAN = "#22d3ee"
    GREEN = "#22c55e"
    YELLOW = "#eab308"
    RED = "#ef4444"

    class DesktopMonitorApp:
        def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
            self.root = root
            self.args = args
            self.queue: queue.Queue = queue.Queue()
            self.worker: AcquisitionWorker | None = None
            self.latest: dict | None = None
            self.alpha_history: Deque[float] = deque(maxlen=240)
            self.delta_history: Deque[float] = deque(maxlen=240)
            self.theta_history: Deque[float] = deque(maxlen=240)
            self.beta_history: Deque[float] = deque(maxlen=240)
            self.alert_history: Deque[str] = deque(maxlen=12)

            root.title("Sleep EEG Monitor - Fp1/Fp2")
            root.configure(bg=BG)
            root.geometry("1440x900")
            self._style()
            self._build()
            self.root.after(40, self._poll_queue)

        def _style(self) -> None:
            for font_name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
                try:
                    tkfont.nametofont(font_name).configure(family=UI_FONT)
                except tk.TclError:
                    pass
            style = ttk.Style()
            style.theme_use("clam")
            style.configure("TNotebook", background=BG, borderwidth=0)
            style.configure(
                "TNotebook.Tab",
                background=PANEL,
                foreground=MUTED,
                padding=(18, 9),
                borderwidth=0,
                font=(UI_FONT, 10, "bold"),
            )
            style.map("TNotebook.Tab", background=[("selected", PANEL_2)], foreground=[("selected", TEXT)])

        def _build(self) -> None:
            header = tk.Frame(self.root, bg=BG)
            header.pack(fill="x", padx=18, pady=(16, 10))
            tk.Label(header, text="Sleep EEG Monitor", bg=BG, fg=TEXT, font=("Segoe UI", 22, "bold")).pack(side="left")
            self.conn_label = self._pill(header, "设备未连接", RED)
            self.conn_label.pack(side="right", padx=(8, 0))
            self.sample_label = self._pill(header, f"采样率 {self.args.sfreq:g}Hz", CYAN)
            self.sample_label.pack(side="right", padx=(8, 0))
            self.port_label = self._pill(header, self.args.serial_port if self.args.source == "serial" else "SIM", MUTED)
            self.port_label.pack(side="right", padx=(8, 0))

            controls = tk.Frame(self.root, bg=BG)
            controls.pack(fill="x", padx=18, pady=(0, 10))
            tk.Button(controls, text="开始采集", command=self.start, bg="#0e7490", fg=TEXT, relief="flat",
                      font=("Microsoft YaHei", 11, "bold"), padx=18, pady=8).pack(side="left")
            tk.Button(controls, text="暂停", command=self.stop, bg="#334155", fg=TEXT, relief="flat",
                      font=("Microsoft YaHei", 11, "bold"), padx=18, pady=8).pack(side="left", padx=8)
            self.status_text = tk.Label(controls, text="采集中：采集/处理在线程运行，UI 主线程独立刷新。",
                                        bg=BG, fg=MUTED, font=("Microsoft YaHei", 10))
            self.status_text.pack(side="left", padx=18)

            self.tabs = ttk.Notebook(self.root)
            self.tabs.pack(fill="both", expand=True, padx=18, pady=(0, 18))
            self.page_dashboard = tk.Frame(self.tabs, bg=BG)
            self.page_features = tk.Frame(self.tabs, bg=BG)
            self.page_quality = tk.Frame(self.tabs, bg=BG)
            self.page_alerts = tk.Frame(self.tabs, bg=BG)
            self.tabs.add(self.page_dashboard, text="首页驾驶舱")
            self.tabs.add(self.page_features, text="特征分析")
            self.tabs.add(self.page_quality, text="睡眠评估")
            self.tabs.add(self.page_alerts, text="系统告警")

            self._build_dashboard()
            self._build_features()
            self._build_quality()
            self._build_alerts()

        def _pill(self, parent: tk.Widget, text: str, color: str) -> tk.Label:
            return tk.Label(parent, text=text, bg=PANEL, fg=color, padx=12, pady=6, font=("Microsoft YaHei", 10, "bold"))

        def _card(self, parent: tk.Widget) -> tk.Frame:
            frame = tk.Frame(parent, bg=PANEL, highlightthickness=1, highlightbackground="#1f3553")
            return frame

        def _build_dashboard(self) -> None:
            left = self._card(self.page_dashboard)
            left.place(relx=0.0, rely=0.0, relwidth=0.22, relheight=0.58)
            right = self._card(self.page_dashboard)
            right.place(relx=0.235, rely=0.0, relwidth=0.765, relheight=0.58)
            bottom_left = self._card(self.page_dashboard)
            bottom_left.place(relx=0.0, rely=0.60, relwidth=0.35, relheight=0.40)
            bottom_right = self._card(self.page_dashboard)
            bottom_right.place(relx=0.365, rely=0.60, relwidth=0.635, relheight=0.40)

            self.subject_label = self._section_text(left, "被试信息", [
                f"ID：{self.args.subject_id}",
                f"年龄：{self.args.age}",
                f"导联：{'/'.join(self.args.channels)}",
                f"模式：{'Alpha演示' if self.args.disable_model else '模型实时分期'}",
            ])
            self.quality_label = self._section_text(bottom_left, "信号质量", ["评分：--", "噪声：--", "状态提示：--", "滤波：0.5-30Hz + 50Hz"])

            self.fig_wave = Figure(figsize=(8, 4), facecolor=PANEL)
            self.ax_wave = self.fig_wave.add_subplot(111)
            self._dark_axis(self.ax_wave)
            self.canvas_wave = FigureCanvasTkAgg(self.fig_wave, master=right)
            self.canvas_wave.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=12)

            self.state_label = tk.Label(bottom_right, text="当前状态：--", bg=PANEL, fg=TEXT, font=("Microsoft YaHei", 22, "bold"))
            self.state_label.pack(anchor="w", padx=18, pady=(18, 6))
            self.conf_label = tk.Label(bottom_right, text="置信度：--", bg=PANEL, fg=MUTED, font=("Microsoft YaHei", 14))
            self.conf_label.pack(anchor="w", padx=18)
            self.score_label = tk.Label(bottom_right, text="睡眠质量评分：实时评估中", bg=PANEL, fg=CYAN, font=("Microsoft YaHei", 15, "bold"))
            self.score_label.pack(anchor="w", padx=18, pady=(8, 6))
            self.timeline_canvas = tk.Canvas(bottom_right, bg=PANEL, highlightthickness=0, height=80)
            self.timeline_canvas.pack(fill="x", padx=18, pady=16)
            self._draw_stage_legend()

        def _section_text(self, parent: tk.Frame, title: str, lines: list[str]) -> tk.Label:
            tk.Label(parent, text=title, bg=PANEL, fg=TEXT, font=("Microsoft YaHei", 15, "bold")).pack(anchor="w", padx=16, pady=(16, 8))
            label = tk.Label(parent, text="\n".join(lines), bg=PANEL, fg=MUTED, justify="left", font=("Microsoft YaHei", 12))
            label.pack(anchor="w", padx=16, pady=4)
            return label

        def _build_features(self) -> None:
            top = self._card(self.page_features)
            top.place(relx=0.0, rely=0.0, relwidth=0.58, relheight=1.0)
            side = self._card(self.page_features)
            side.place(relx=0.595, rely=0.0, relwidth=0.405, relheight=1.0)
            self.fig_feat = Figure(figsize=(7, 6), facecolor=PANEL)
            self.ax_trend = self.fig_feat.add_subplot(211)
            self.ax_bar = self.fig_feat.add_subplot(212)
            self._dark_axis(self.ax_trend)
            self._dark_axis(self.ax_bar)
            self.canvas_feat = FigureCanvasTkAgg(self.fig_feat, master=top)
            self.canvas_feat.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=12)
            self.feature_text = self._section_text(side, f"当前 {DISPLAY_FEATURE_WINDOW_SECONDS:g} 秒滚动分析", [
                "Delta：--",
                "Theta：--",
                "Alpha：--",
                "Beta：--",
                "Theta/Alpha：--",
                "Delta/Theta：--",
                "Hjorth：--",
                "熵：--",
            ])

        def _build_quality(self) -> None:
            card = self._card(self.page_quality)
            card.pack(fill="both", expand=True)
            self.quality_eval = self._section_text(card, "睡眠质量辅助评估结果", [
                "总睡眠时长：采集中",
                "入睡潜伏期：采集中",
                "深睡比例：采集中",
                "REM 比例：采集中",
                "睡眠效率：采集中",
                "综合评分：实时评估中",
                "建议：保持电极稳定，减少眨眼和大幅头动。",
            ])

        def _build_alerts(self) -> None:
            card = self._card(self.page_alerts)
            card.pack(fill="both", expand=True)
            tk.Label(card, text="系统告警与功率不可用原因", bg=PANEL, fg=TEXT, font=("Microsoft YaHei", 16, "bold")).pack(anchor="w", padx=18, pady=18)
            self.alert_box = tk.Text(card, bg="#07101d", fg=TEXT, insertbackground=TEXT, relief="flat", font=(UI_FONT, 11), height=22)
            self.alert_box.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        def _dark_axis(self, ax) -> None:
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=MUTED, labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#29415f")
            ax.grid(True, color="#24364d", alpha=.65, linewidth=.8)

        def start(self) -> None:
            if self.worker and self.worker.is_alive():
                return
            self.worker = AcquisitionWorker(self.args, self.queue)
            self.worker.start()
            self.conn_label.configure(text="设备已连接", fg=GREEN)
            self.status_text.configure(text="采集中：采集/处理在线程运行，UI 主线程独立刷新。")

        def stop(self) -> None:
            if self.worker:
                self.worker.stop()
            self.conn_label.configure(text="已暂停", fg=YELLOW)
            self.status_text.configure(text="采集已暂停。")

        def _poll_queue(self) -> None:
            updated = False
            while True:
                try:
                    item = self.queue.get_nowait()
                except queue.Empty:
                    break
                if item.get("type") == "snapshot":
                    self.latest = item
                    updated = True
                else:
                    self._push_alert(item.get("message", "未知告警"))
            if updated and self.latest:
                self._render_snapshot(self.latest)
            self.root.after(40, self._poll_queue)

        def _render_snapshot(self, snap: dict) -> None:
            quality = snap["quality"]
            features = snap["features"]
            self.state_label.configure(text=f"当前状态：{snap['state']}")
            self.conf_label.configure(text=f"置信度：{snap['confidence']}%")
            if snap.get("model_status"):
                self.status_text.configure(text=snap["model_status"])
            color = GREEN if quality["score"] >= 80 else (YELLOW if quality["score"] >= 60 else RED)
            self.quality_label.configure(
                text=(
                    f"评分：{quality['score']} / 100\n"
                    f"噪声：{quality['noise']}\n"
                    f"状态提示：{', '.join(quality['artifacts']) if quality['artifacts'] else '信号稳定'}\n"
                    "滤波：0.5-30Hz + 50Hz陷波"
                ),
                fg=color,
            )
            if snap["feature_reason"] != "OK":
                self._push_alert(snap["feature_reason"])
            elif quality["artifacts"]:
                self._push_alert(" / ".join(quality["artifacts"]))
            self._plot_wave(snap)
            self._plot_features(features)
            self._update_feature_text(features, snap["feature_reason"])
            self._update_sleep_quality(snap.get("prediction"))

        def _plot_wave(self, snap: dict) -> None:
            data = snap["waveform_uv"]
            self.ax_wave.clear()
            self._dark_axis(self.ax_wave)
            if data.size:
                t = np.arange(data.shape[1]) / snap["sfreq"]
                offset = 80
                for idx, ch in enumerate(snap["channels"]):
                    self.ax_wave.plot(t, data[idx] + (len(snap["channels"]) - idx - 1) * offset, lw=1.05, color=CYAN if idx == 0 else GREEN)
                    self.ax_wave.text(t[0] if t.size else 0, (len(snap["channels"]) - idx - 1) * offset + 32, ch, color=TEXT, fontsize=10)
                self.ax_wave.set_xlim(max(0, t[-1] - 8), t[-1] if t.size else 8)
                self.ax_wave.set_title("实时 EEG 波形 CH1 / CH2", color=TEXT, fontsize=12)
                self.ax_wave.set_ylabel("uV", color=MUTED)
            self.canvas_wave.draw_idle()

        def _plot_features(self, features: dict) -> None:
            self.ax_trend.clear()
            self.ax_bar.clear()
            self._dark_axis(self.ax_trend)
            self._dark_axis(self.ax_bar)
            if features and "summary" in features:
                vals = []
                first_ch = self.args.channels[0]
                for band in ("delta", "theta", "alpha", "beta"):
                    vals.append(features[first_ch][band] * 100)
                self.delta_history.append(vals[0])
                self.theta_history.append(vals[1])
                self.alpha_history.append(vals[2])
                self.beta_history.append(vals[3])
                x = np.arange(len(self.alpha_history))
                self.ax_trend.plot(x, list(self.delta_history), color="#60a5fa", lw=1.1, label="Delta")
                self.ax_trend.plot(x, list(self.theta_history), color="#a78bfa", lw=1.1, label="Theta")
                self.ax_trend.plot(x, list(self.alpha_history), color=GREEN, lw=1.4, label="Alpha")
                self.ax_trend.plot(x, list(self.beta_history), color=YELLOW, lw=1.1, label="Beta")
                self.ax_trend.set_ylim(0, 100)
                self.ax_trend.set_title("频带功率趋势", color=TEXT)
                self.ax_trend.legend(facecolor=PANEL, edgecolor="#29415f", labelcolor=TEXT, fontsize=8)
                bars = self.ax_bar.bar(["Delta", "Theta", "Alpha", "Beta"], vals, color=["#60a5fa", "#a78bfa", GREEN, YELLOW])
                self.ax_bar.set_ylim(0, 100)
                self.ax_bar.set_title(f"{first_ch} 当前频带相对功率", color=TEXT)
                for bar, val in zip(bars, vals):
                    self.ax_bar.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val:.1f}%", ha="center", color=TEXT, fontsize=9)
            self.canvas_feat.draw_idle()

        def _update_feature_text(self, features: dict, reason: str) -> None:
            if not features:
                self.feature_text.configure(text=f"功率状态：不可用\n原因：{reason}", fg=YELLOW)
                return
            ch = self.args.channels[0]
            item = features[ch]
            h0, h1, h2 = item["hjorth"]
            self.feature_text.configure(
                text=(
                    f"Delta：{item['delta'] * 100:.1f}%\n"
                    f"Theta：{item['theta'] * 100:.1f}%\n"
                    f"Alpha：{item['alpha'] * 100:.1f}%\n"
                    f"Beta：{item['beta'] * 100:.1f}%\n"
                    f"Theta/Alpha：{item['theta_alpha']:.2f}\n"
                    f"Delta/Theta：{item['delta_theta']:.2f}\n"
                    f"Hjorth：A={h0:.1f}, M={h1:.2f}, C={h2:.2f}\n"
                    f"Alpha绝对功率：{item['alpha_abs']:.1f}"
                ),
                fg=TEXT,
            )

        def _update_sleep_quality(self, prediction: dict | None) -> None:
            if not prediction:
                self.quality_eval.configure(
                    text=(
                        "总睡眠时长：等待模型分期\n"
                        "入睡潜伏期：等待模型分期\n"
                        "深睡比例：等待模型分期\n"
                        "REM 比例：等待模型分期\n"
                        "睡眠效率：等待模型分期\n"
                        "综合评分：至少需要一个 30 秒 epoch\n"
                        "建议：先确认串口帧稳定，再连续采集。"
                    ),
                    fg=MUTED,
                )
                return

            q = prediction.get("sleep_quality")
            if not q:
                self.quality_eval.configure(
                    text=(
                        f"最近分期：{prediction.get('stage', '--')}\n"
                        f"模型置信度：{prediction.get('confidence', 0)}%\n"
                        "睡眠质量：累计中\n"
                        "建议：保持电极稳定，等待更多 30 秒 epoch。"
                    ),
                    fg=TEXT,
                )
                return

            self.quality_eval.configure(
                text=(
                    f"已预测 epoch：{prediction.get('epoch_count', 0)} 个\n"
                    f"最近分期：{prediction.get('stage', '--')}（{prediction.get('confidence', 0)}%）\n"
                    f"总记录时长：{q['total_recording_min']:.1f} 分钟\n"
                    f"总睡眠时长：{q['total_sleep_time_min']:.1f} 分钟\n"
                    f"入睡潜伏期：{q['sleep_onset_latency_min']:.1f} 分钟\n"
                    f"深睡比例：{q['n3_pct_tst']:.1f}%\n"
                    f"REM 比例：{q['rem_pct_tst']:.1f}%\n"
                    f"睡眠效率：{q['sleep_efficiency_pct']:.1f}%\n"
                    f"综合评分：{q['score']:.1f} / 100（{q['grade']}）\n"
                    f"建议：{q['recommendation']}"
                ),
                fg=CYAN,
            )

        def _draw_stage_legend(self) -> None:
            self.timeline_canvas.delete("all")
            labels = ["Wake", "N1", "N2", "N3", "REM"]
            colors = ["#22d3ee", "#60a5fa", "#6366f1", "#312e81", "#a78bfa"]
            width = 520
            for i, (label, color) in enumerate(zip(labels, colors)):
                x0 = 12 + i * (width / len(labels))
                x1 = 12 + (i + 1) * (width / len(labels)) - 6
                self.timeline_canvas.create_rectangle(x0, 22, x1, 48, fill=color, outline="")
                self.timeline_canvas.create_text((x0 + x1) / 2, 35, text=label, fill="white", font=("Segoe UI", 9, "bold"))

        def _push_alert(self, text: str) -> None:
            if not text:
                return
            stamp = time.strftime("%H:%M:%S")
            line = f"[{stamp}] {text}"
            if self.alert_history and self.alert_history[-1].endswith(text):
                return
            self.alert_history.append(line)
            self.alert_box.delete("1.0", "end")
            self.alert_box.insert("end", "\n".join(self.alert_history))

    args = parse_args()
    app_root = tk.Tk()
    DesktopMonitorApp(app_root, args)
    app_root.mainloop()
