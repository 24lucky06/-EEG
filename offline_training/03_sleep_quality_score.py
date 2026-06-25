"""
03_sleep_quality_score.py

离线睡眠质量评分脚本。

用法示例：
  python 03_sleep_quality_score.py --feature-dir features --mode dual2

它会读取 features/*_y.npy，基于真实分期标签计算每个被试的睡眠质量指标和 0-100 分评分。
如果你想对模型预测结果评分，可以在 02_train_compare_models_vscode.py 运行后查看
results/sleep_quality_from_global_model_predictions_<mode>_<context>.csv。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from sleep_quality_score import compute_sleep_quality, result_to_dict


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据离线睡眠分期标签计算睡眠质量评分")
    parser.add_argument("--feature-dir", default="features", help="01 脚本生成的 features 文件夹")
    parser.add_argument("--result-dir", default="results", help="结果输出文件夹")
    parser.add_argument("--mode", choices=["all32", "dual2"], default="dual2", help="通道模式")
    parser.add_argument("--epoch-seconds", type=int, default=30, help="每个 epoch 秒数")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_dir = resolve_path(args.feature_dir)
    result_dir = resolve_path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    y_files = sorted(feature_dir.glob(f"*_{args.mode}_y.npy"))
    if not y_files:
        print(f"❌ 没有找到 {feature_dir}/*_{args.mode}_y.npy，请先运行 01 脚本。")
        return 1

    for y_file in y_files:
        prefix = y_file.name.replace("_y.npy", "")
        y = np.load(y_file).astype(int)
        result = compute_sleep_quality(y, epoch_seconds=args.epoch_seconds)
        row = {"subject": prefix, "source": "label_y", **result_to_dict(result)}
        rows.append(row)
        print(f"{prefix}: 睡眠质量 {result.score_0_100} 分，等级：{result.grade}，{result.recommendation}")

    out_csv = result_dir / f"sleep_quality_from_labels_{args.mode}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ 已保存：{out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
