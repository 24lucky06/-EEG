"""
sleep_quality_score.py

基于 W / N1 / N2 / N3 / REM 睡眠分期序列计算睡眠质量指标和 0-100 分评分。

注意：
1. 如果没有 PSQI、医生评分等真实睡眠质量标签，本模块不是监督学习模型；
2. 它是一个规则评分模块，用分期序列计算睡眠效率、入睡潜伏期、WASO、深睡比例等指标；
3. 可用于离线标签、离线预测结果，也可用于实时系统累计整晚预测后输出睡眠质量评分。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


@dataclass
class SleepQualityResult:
    total_recording_min: float
    total_sleep_time_min: float
    sleep_efficiency_pct: float
    sleep_onset_latency_min: float
    waso_min: float
    awakening_count: int
    n1_pct_tst: float
    n2_pct_tst: float
    n3_pct_tst: float
    rem_pct_tst: float
    stage_transition_count: int
    transition_per_hour: float
    score_0_100: float
    grade: str
    recommendation: str


def _safe_pct(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator * 100.0)


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def stage_sequence_to_names(stages: Sequence[int]) -> List[str]:
    names: List[str] = []
    for s in stages:
        idx = int(s)
        names.append(STAGE_NAMES[idx] if 0 <= idx < len(STAGE_NAMES) else "UNKNOWN")
    return names


def compute_sleep_quality(stages: Sequence[int], epoch_seconds: int = 30) -> SleepQualityResult:
    """根据一整段睡眠分期序列计算睡眠质量。

    参数
    ----
    stages:
        分期序列，编码固定为 W=0, N1=1, N2=2, N3=3, REM=4。
    epoch_seconds:
        每个分期片段的时长，睡眠分期常用 30 秒。

    返回
    ----
    SleepQualityResult
        包含睡眠效率、入睡潜伏期、WASO、N3比例、REM比例、评分和建议。
    """
    y = np.asarray(stages, dtype=int)
    y = y[np.isfinite(y)] if y.size else y
    if y.size == 0:
        return SleepQualityResult(
            total_recording_min=0.0,
            total_sleep_time_min=0.0,
            sleep_efficiency_pct=0.0,
            sleep_onset_latency_min=0.0,
            waso_min=0.0,
            awakening_count=0,
            n1_pct_tst=0.0,
            n2_pct_tst=0.0,
            n3_pct_tst=0.0,
            rem_pct_tst=0.0,
            stage_transition_count=0,
            transition_per_hour=0.0,
            score_0_100=0.0,
            grade="无数据",
            recommendation="没有可用的睡眠分期序列，无法评分。",
        )

    epoch_min = epoch_seconds / 60.0
    total_recording_min = float(len(y) * epoch_min)
    sleep_mask = y != 0
    sleep_epochs = int(np.sum(sleep_mask))
    total_sleep_time_min = float(sleep_epochs * epoch_min)
    sleep_efficiency = _safe_pct(total_sleep_time_min, total_recording_min)

    sleep_indices = np.flatnonzero(sleep_mask)
    if len(sleep_indices) == 0:
        sleep_onset_idx = len(y)
        sol_min = total_recording_min
        after_onset = np.array([], dtype=int)
    else:
        sleep_onset_idx = int(sleep_indices[0])
        sol_min = float(sleep_onset_idx * epoch_min)
        after_onset = y[sleep_onset_idx:]

    waso_epochs = int(np.sum(after_onset == 0)) if after_onset.size else 0
    waso_min = float(waso_epochs * epoch_min)

    awakening_count = 0
    if after_onset.size >= 2:
        prev_sleep = after_onset[:-1] != 0
        curr_wake = after_onset[1:] == 0
        awakening_count = int(np.sum(prev_sleep & curr_wake))

    n1_pct = _safe_pct(np.sum(y == 1), sleep_epochs)
    n2_pct = _safe_pct(np.sum(y == 2), sleep_epochs)
    n3_pct = _safe_pct(np.sum(y == 3), sleep_epochs)
    rem_pct = _safe_pct(np.sum(y == 4), sleep_epochs)

    transition_count = int(np.sum(y[1:] != y[:-1])) if len(y) >= 2 else 0
    hours = total_recording_min / 60.0 if total_recording_min > 0 else 0.0
    transition_per_hour = float(transition_count / hours) if hours > 0 else 0.0

    # 规则评分：不是临床诊断，只是工程项目中的睡眠质量综合指标。
    penalty = 0.0
    penalty += _clip((85.0 - sleep_efficiency) * 1.2, 0.0, 25.0)
    penalty += _clip((sol_min - 20.0) / 2.0, 0.0, 15.0)
    penalty += _clip((waso_min - 30.0) / 3.0, 0.0, 20.0)
    penalty += _clip((awakening_count - 2.0) * 1.5, 0.0, 10.0)

    if sleep_epochs > 0:
        if n3_pct < 12.0:
            penalty += _clip((12.0 - n3_pct) * 0.8, 0.0, 10.0)
        elif n3_pct > 30.0:
            penalty += _clip((n3_pct - 30.0) * 0.3, 0.0, 6.0)

        if rem_pct < 15.0:
            penalty += _clip((15.0 - rem_pct) * 0.5, 0.0, 8.0)
        elif rem_pct > 30.0:
            penalty += _clip((rem_pct - 30.0) * 0.4, 0.0, 8.0)

    penalty += _clip((transition_per_hour - 18.0) * 0.4, 0.0, 5.0)

    score = round(_clip(100.0 - penalty, 0.0, 100.0), 1)
    if score >= 85:
        grade = "优秀"
    elif score >= 70:
        grade = "良好"
    elif score >= 60:
        grade = "一般"
    else:
        grade = "较差"

    tips: List[str] = []
    if sleep_efficiency < 85:
        tips.append("睡眠效率偏低")
    if sol_min > 20:
        tips.append("入睡潜伏期偏长")
    if waso_min > 30:
        tips.append("入睡后觉醒时间偏多")
    if awakening_count > 2:
        tips.append("夜间觉醒次数偏多")
    if sleep_epochs > 0 and n3_pct < 12:
        tips.append("深睡比例偏低")
    if sleep_epochs > 0 and rem_pct < 15:
        tips.append("REM比例偏低")
    if not tips:
        tips.append("主要睡眠结构指标较稳定")
    recommendation = "；".join(tips) + "。"

    return SleepQualityResult(
        total_recording_min=round(total_recording_min, 2),
        total_sleep_time_min=round(total_sleep_time_min, 2),
        sleep_efficiency_pct=round(sleep_efficiency, 2),
        sleep_onset_latency_min=round(sol_min, 2),
        waso_min=round(waso_min, 2),
        awakening_count=awakening_count,
        n1_pct_tst=round(n1_pct, 2),
        n2_pct_tst=round(n2_pct, 2),
        n3_pct_tst=round(n3_pct, 2),
        rem_pct_tst=round(rem_pct, 2),
        stage_transition_count=transition_count,
        transition_per_hour=round(transition_per_hour, 2),
        score_0_100=score,
        grade=grade,
        recommendation=recommendation,
    )


def result_to_dict(result: SleepQualityResult) -> Dict[str, object]:
    return asdict(result)


def build_quality_table(records: Iterable[Dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(list(records))
