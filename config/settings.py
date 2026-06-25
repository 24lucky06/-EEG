"""
config/settings.py

Project-wide configuration for offline training and realtime prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

# Project paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
OFFLINE_DIR: Path = PROJECT_ROOT / "offline_training"

# Signal processing
SFREQ: float = 250.0
EPOCH_SECONDS: int = 30
LOW_FREQ: float = 0.3
HIGH_FREQ: float = 35.0
CLIP_UV: float = 150.0
DISPLAY_LOW_FREQ: float = 0.5
DISPLAY_HIGH_FREQ: float = 30.0
DISPLAY_NOTCH_FREQ: float = 50.0
DISPLAY_NOTCH_Q: float = 30.0
DISPLAY_COMMON_MODE_REJECTION: bool = True
DISPLAY_DESPIKE_MAD_MULTIPLIER: float = 8.0
DISPLAY_DESPIKE_MIN_UV: float = 120.0
DISPLAY_MOVING_AVERAGE_POINTS: int = 5
DISPLAY_EOG_ATTENUATION_ENABLED: bool = True
DISPLAY_EOG_LOW_FREQ: float = 0.1
DISPLAY_EOG_HIGH_FREQ: float = 4.0
DISPLAY_EOG_ATTENUATION: float = 0.75
DISPLAY_CLIP_UV: float = 300.0
SIGNAL_MAX_CENTERED_PTP_UV: float = 1000.0
SIGNAL_MAX_FILTERED_PTP_UV: float = 600.0
SIGNAL_MAX_ABS_MEAN_UV: float = 100000.0
REALTIME_WARMUP_EPOCHS: int = 2
ENABLE_SLEEP_STAGE_PREDICTION: bool = False
WAVEFORM_BROADCAST_INTERVAL_SEC: float = 0.10
DISPLAY_FEATURE_WINDOW_SECONDS: float = 10.0
DEMO_BAND_PRESENTATION_MODE: bool = False
DEMO_DELTA_WEIGHT: float = 0.35
DEMO_THETA_WEIGHT: float = 0.85
DEMO_ALPHA_WEIGHT: float = 1.80
DEMO_BETA_WEIGHT: float = 0.90

# Channel settings
CHANNEL_MODE: str = "dual2"
DUAL2_CHANNELS: List[str] = ["Fp1", "Fp2"]

# Custom ADS1299 board
# Default to two prefrontal electrodes in hardware input order.
HARDWARE_CHANNEL_INDICES: List[int] = [0, 1]
ADS1299_VREF: float = 4.5
ADS1299_GAIN: int = 24
ADS1299_SCALE_CORRECTION: float = 0.25
ADS1299_VOLTS_PER_COUNT: float = (
    ADS1299_VREF / ADS1299_GAIN / (2**23 - 1) * ADS1299_SCALE_CORRECTION
)

# Offline training output paths
FEATURE_DIR: Path = OFFLINE_DIR / "features"
RESULT_DIR: Path = OFFLINE_DIR / "results"
MODEL_DIR: Path = OFFLINE_DIR / "saved_models"

# Training settings
WITHIN_TRAIN_RATIO: float = 0.7
CALIBRATION_RATIO: float = 0.2
CONTEXT_MODE: str = "causal"
STAGE_NAMES: List[str] = ["W", "N1", "N2", "N3", "REM"]
MAX_EPOCHS_PER_SUBJECT = None

# Realtime system
DEFAULT_MODEL_PATH: Path = MODEL_DIR / f"sleep_stage_{CHANNEL_MODE}_{CONTEXT_MODE}_global.joblib"
