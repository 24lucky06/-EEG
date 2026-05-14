"""
app.py

实时系统入口示例。

当前是“可跑通流程”的骨架：
1. 模拟硬件产生双导联 EEG；
2. 每 30 秒提取一次特征；
3. 加载 offline_training/saved_models 中的 dual2 causal 模型；
4. 输出当前预测睡眠阶段。

真正接硬件时，替换 hardware_reader.py 即可。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hardware_reader import HardwareConfig, SimulatedEEGReader
from model_loader import load_sleep_stage_model
from realtime_feature_extractor import RealtimeFeatureConfig, RealtimeFeatureExtractor
from realtime_predictor import RealtimeSleepStagePredictor
from sleep_quality_realtime import RealtimeSleepQualityTracker

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT.parent / "offline_training" / "saved_models" / "sleep_stage_dual2_causal_global.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="双导联 EEG 实时预测流程示例")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="02 脚本保存的 joblib 模型路径")
    parser.add_argument("--channels", nargs=2, default=["C3", "O1"], help="双导联通道名，必须和训练模型一致")
    parser.add_argument("--sfreq", type=float, default=100.0, help="实时采样率")
    parser.add_argument("--demo-epochs", type=int, default=5, help="模拟运行多少个 epoch")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact = load_sleep_stage_model(args.model)

    hw_config = HardwareConfig(channel_names=args.channels, sfreq=args.sfreq)
    reader = SimulatedEEGReader(hw_config)

    extractor = RealtimeFeatureExtractor(
        RealtimeFeatureConfig(channel_names=args.channels, sfreq=args.sfreq)
    )
    predictor = RealtimeSleepStagePredictor(artifact)
    tracker = RealtimeSleepQualityTracker(epoch_seconds=30)

    print("实时流程启动。当前使用模拟硬件数据。")
    print(f"加载模型：{args.model}")

    for i, epoch in zip(range(args.demo_epochs), reader.iter_epochs()):
        features, _ = extractor.extract_epoch_features(epoch)
        pred_id, stage, proba = predictor.predict_from_features(features)
        tracker.add_stage(pred_id)
        quality = tracker.current_quality()
        print(
            f"Epoch {i + 1:03d}: stage={stage}, "
            f"confidence={proba[pred_id]:.3f}, current_score={quality.score_0_100}, grade={quality.grade}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
