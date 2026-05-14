"""model_loader.py：加载 offline_training/saved_models 中保存的 joblib 模型。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib


def load_sleep_stage_model(model_path: str | Path) -> Dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到模型文件：{path}")
    artifact = joblib.load(path)
    required = ["model", "scaler", "context_mode", "stage_names", "raw_feature_dim"]
    missing = [k for k in required if k not in artifact]
    if missing:
        raise ValueError(f"模型文件缺少必要字段：{missing}")
    return artifact
