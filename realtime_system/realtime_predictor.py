"""
realtime_predictor.py

实时睡眠分期预测器：
- 输入当前 30 秒 epoch 的原始特征；
- 使用离线训练保存的 scaler 标准化；
- 按 causal 模式拼接前两段 + 当前段；
- 输出 W / N1 / N2 / N3 / REM。
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np


class RealtimeSleepStagePredictor:
    def __init__(self, artifact: Dict[str, object]):
        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.context_mode = str(artifact.get("context_mode", "causal"))
        self.stage_names: List[str] = list(artifact.get("stage_names", ["W", "N1", "N2", "N3", "REM"]))
        self.raw_feature_dim = int(artifact["raw_feature_dim"])
        self.history: Deque[np.ndarray] = deque(maxlen=2)

    def reset(self) -> None:
        self.history.clear()

    def _make_context(self, current_scaled: np.ndarray) -> np.ndarray:
        if self.context_mode == "none":
            return current_scaled.reshape(1, -1)
        if self.context_mode != "causal":
            raise ValueError("实时预测只能使用 causal 或 none。centered 会偷看未来，不适合实时。")

        zero = np.zeros(self.raw_feature_dim, dtype=np.float32)
        if len(self.history) == 0:
            prev2, prev1 = zero, zero
        elif len(self.history) == 1:
            prev2, prev1 = zero, self.history[-1]
        else:
            prev2, prev1 = self.history[0], self.history[1]
        return np.concatenate([prev2, prev1, current_scaled]).reshape(1, -1)

    def predict_from_features(self, raw_features: np.ndarray) -> Tuple[int, str, np.ndarray]:
        raw_features = np.asarray(raw_features, dtype=np.float32).reshape(1, -1)
        if raw_features.shape[1] != self.raw_feature_dim:
            raise ValueError(f"特征维度不匹配：收到 {raw_features.shape[1]}，模型需要 {self.raw_feature_dim}")

        current_scaled = self.scaler.transform(raw_features).reshape(-1).astype(np.float32)
        x_ctx = self._make_context(current_scaled)
        proba = self.model.predict_proba(x_ctx)[0]
        pred = int(np.argmax(proba))
        stage = self.stage_names[pred] if 0 <= pred < len(self.stage_names) else str(pred)

        self.history.append(current_scaled)
        return pred, stage, proba
