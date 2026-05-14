"""sleep_quality_realtime.py：实时累计预测阶段，并动态计算睡眠质量评分。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# 允许从 realtime_system 单独运行，也允许复用 offline_training 的评分逻辑。
try:
    from sleep_quality_score import SleepQualityResult, compute_sleep_quality
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1] / "offline_training"))
    from sleep_quality_score import SleepQualityResult, compute_sleep_quality


@dataclass
class RealtimeSleepQualityTracker:
    epoch_seconds: int = 30
    stages: List[int] = field(default_factory=list)

    def add_stage(self, stage_id: int) -> None:
        self.stages.append(int(stage_id))

    def current_quality(self) -> SleepQualityResult:
        return compute_sleep_quality(self.stages, epoch_seconds=self.epoch_seconds)
