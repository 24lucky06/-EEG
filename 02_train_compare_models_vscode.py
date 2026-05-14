"""
02_train_compare_models_vscode.py

VSCode 友好版模型训练与对比脚本。

功能：
1. 读取 01_extract_features_LIGHT_FIR_vscode.py 生成的 features/*_X.npy、*_y.npy、*_meta.json；
2. 使用 XGBoost 完成 W / N1 / N2 / N3 / REM 五分类睡眠分期；
3. 支持同被试时间切分、LOSO、个体化校准、特征重要性分析；
4. 默认使用 causal 时序上下文：前两段 + 当前段，便于后续接硬件实时预测；
5. 输出 results/model_results_summary.csv、分类报告、feature_importance_ranking.csv、saved_models/*.joblib。

推荐 VSCode 运行方式：
- 先运行 01 脚本生成 features 文件夹；
- 在 VSCode 终端运行：
  python 02_train_compare_models_vscode.py --mode dual2 --context-mode causal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parent


CONFIG: Dict[str, Any] = {
    "FEATURE_DIR": "features",
    "RESULT_DIR": "results",
    "CHANNEL_MODE": "dual2",
    "WITHIN_TRAIN_RATIO": 0.7,
    "CALIBRATION_RATIO": 0.2,
    "STAGE_NAMES": ["W", "N1", "N2", "N3", "REM"],
    # causal = 前两段 + 当前段，适合实时；centered = 上一段 + 当前段 + 下一段，适合离线复现原版。
    "CONTEXT_MODE": "causal",
}


def resolve_path(path_like: str | Path) -> Path:
    """把相对路径统一解释为相对于脚本所在文件夹。"""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def add_temporal_context(X: np.ndarray, mode: str = "causal") -> np.ndarray:
    """
    注入简单时序上下文。

    causal：前两段 + 前一段 + 当前段，即 t-2, t-1, t。适合实时预测。
    centered：上一段 + 当前段 + 下一段，即 t-1, t, t+1。适合离线分析，但实时不可用。
    none：不拼接上下文。
    """
    if mode == "none":
        return X

    df = pd.DataFrame(X)
    if mode == "causal":
        return pd.concat(
            [df.shift(2).fillna(0), df.shift(1).fillna(0), df],
            axis=1,
        ).values
    if mode == "centered":
        return pd.concat(
            [df.shift(1).fillna(0), df, df.shift(-1).fillna(0)],
            axis=1,
        ).values
    raise ValueError("context_mode 只能是 causal、centered 或 none")


def make_extended_feature_names(feature_names: Optional[List[str]], raw_dim: int, context_mode: str) -> List[str]:
    """生成拼接时序上下文之后的特征名。"""
    if feature_names is None or len(feature_names) != raw_dim:
        if context_mode == "none":
            return [f"Feature_{i}" for i in range(raw_dim)]
        return [f"Feature_{i}" for i in range(raw_dim * 3)]

    if context_mode == "none":
        return feature_names
    if context_mode == "causal":
        prefixes = ["Prev60s_", "Prev30s_", "Current_"]
    elif context_mode == "centered":
        prefixes = ["Prev30s_", "Current_", "Next30s_"]
    else:
        raise ValueError("context_mode 只能是 causal、centered 或 none")

    extended = []
    for prefix in prefixes:
        extended.extend([prefix + fn for fn in feature_names])
    return extended


def smooth_predictions(preds: Sequence[int]) -> np.ndarray:
    """简单生理平滑：如果前后相同、中间突变，则修正中间。"""
    preds_arr = np.asarray(preds).copy()
    for i in range(1, len(preds_arr) - 1):
        if preds_arr[i - 1] == preds_arr[i + 1] and preds_arr[i] != preds_arr[i - 1]:
            preds_arr[i] = preds_arr[i - 1]
    return preds_arr


def make_model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=5,
        n_jobs=-1,
    )


def load_available_subjects(feature_dir: Path, mode: str) -> List[Dict[str, Any]]:
    """读取 features 文件夹中所有当前模式的被试特征。"""
    items: List[Dict[str, Any]] = []
    feature_dir = Path(feature_dir)

    for x_file in sorted(feature_dir.glob(f"*_{mode}_X.npy")):
        prefix = x_file.name.replace("_X.npy", "")
        y_file = feature_dir / f"{prefix}_y.npy"
        meta_file = feature_dir / f"{prefix}_meta.json"
        if not y_file.exists():
            print(f"⚠️ 找到 {x_file.name}，但缺少对应 y 文件，已跳过。")
            continue

        meta: Dict[str, Any] = {}
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

        items.append({
            "prefix": prefix,
            "x_path": x_file,
            "y_path": y_file,
            "meta_path": meta_file if meta_file.exists() else None,
            "meta": meta,
        })
    return items


def load_xy(item: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """读取单个被试的 X/y，并删除非有限数值行。"""
    x = np.load(item["x_path"])
    y = np.load(item["y_path"]).astype(int)

    n = min(len(x), len(y))
    if len(x) != len(y):
        print(f"⚠️ {item['prefix']} 的 X/y 长度不一致：X={len(x)}, y={len(y)}，已截断到 {n}")
        x = x[:n]
        y = y[:n]

    good = np.isfinite(x).all(axis=1)
    if not np.all(good):
        print(f"⚠️ {item['prefix']} 删除非有限数值 epoch：{np.sum(~good)} 个")
    return x[good], y[good]


def evaluate_and_save(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    result_dir: Path,
    stage_names: Sequence[str],
) -> Dict[str, Any]:
    """计算指标并保存分类报告。"""
    result_dir.mkdir(parents=True, exist_ok=True)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        target_names=list(stage_names),
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

    safe_title = title.replace("/", "_").replace("\\", "_").replace(" ", "_")
    report_path = result_dir / f"{safe_title}_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(title + "\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro-F1: {macro_f1:.4f}\n")
        f.write(f"Weighted-F1: {weighted_f1:.4f}\n")
        f.write(f"Cohen Kappa: {kappa:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix, labels=[W,N1,N2,N3,REM]:\n")
        f.write(str(cm))

    return {
        "experiment": title,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "kappa": kappa,
        "n_test": len(y_true),
        "report_path": str(report_path),
    }


def train_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    context_mode: str,
) -> Tuple[np.ndarray, xgb.XGBClassifier, StandardScaler]:
    """
    标准化 + 时序上下文拼接 + XGBoost 训练预测。
    scaler 只在训练集上 fit，避免数据泄露。
    """
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    x_train_ctx = add_temporal_context(x_train_s, mode=context_mode)
    x_test_ctx = add_temporal_context(x_test_s, mode=context_mode)

    weights = compute_sample_weight(class_weight="balanced", y=y_train)
    model = make_model()
    model.fit(x_train_ctx, y_train, sample_weight=weights)

    y_pred = model.predict(x_test_ctx)
    y_pred = smooth_predictions(y_pred)
    return y_pred, model, scaler


def run_within_subject_time_split(
    subjects: List[Dict[str, Any]],
    result_dir: Path,
    within_train_ratio: float,
    context_mode: str,
    stage_names: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    print("\n" + "=" * 90)
    print("实验1：同被试时间切分")

    for item in subjects:
        x, y = load_xy(item)
        n = len(y)
        split = int(n * within_train_ratio)

        if split < 30 or n - split < 30:
            print(f"⚠️ {item['prefix']} 数据太少，跳过。n={n}, split={split}")
            continue

        x_train, y_train = x[:split], y[:split]
        x_test, y_test = x[split:], y[split:]

        y_pred, _, _ = train_predict(x_train, y_train, x_test, context_mode=context_mode)

        title = f"within_subject_time_split__{item['prefix']}"
        row = evaluate_and_save(y_test, y_pred, title, result_dir, stage_names)
        row["subject"] = item["prefix"]
        row["context_mode"] = context_mode
        rows.append(row)

        print(f"{item['prefix']}: acc={row['accuracy']:.3f}, macroF1={row['macro_f1']:.3f}")
    return rows


def run_loso(
    subjects: List[Dict[str, Any]],
    result_dir: Path,
    context_mode: str,
    stage_names: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    print("\n" + "=" * 90)
    print("实验2：留一被试交叉验证 LOSO")

    if len(subjects) < 2:
        print("⚠️ 至少需要 2 个被试才能做 LOSO，已跳过。")
        return rows

    for test_item in subjects:
        train_items = [s for s in subjects if s["prefix"] != test_item["prefix"]]

        x_train_list, y_train_list = [], []
        for item in train_items:
            x_i, y_i = load_xy(item)
            x_train_list.append(x_i)
            y_train_list.append(y_i)

        x_train = np.vstack(x_train_list)
        y_train = np.concatenate(y_train_list)
        x_test, y_test = load_xy(test_item)

        y_pred, _, _ = train_predict(x_train, y_train, x_test, context_mode=context_mode)

        title = f"leave_one_subject_out__test_{test_item['prefix']}"
        row = evaluate_and_save(y_test, y_pred, title, result_dir, stage_names)
        row["test_subject"] = test_item["prefix"]
        row["context_mode"] = context_mode
        rows.append(row)

        print(f"test={test_item['prefix']}: acc={row['accuracy']:.3f}, macroF1={row['macro_f1']:.3f}")
    return rows


def run_personal_calibration(
    subjects: List[Dict[str, Any]],
    result_dir: Path,
    calibration_ratio: float,
    context_mode: str,
    stage_names: Sequence[str],
    save_models: bool = True,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    print("\n" + "=" * 90)
    print("实验3：个体化校准")

    if len(subjects) < 2:
        print("⚠️ 至少需要 2 个被试才能做个体化校准，已跳过。")
        return rows

    for test_item in subjects:
        train_items = [s for s in subjects if s["prefix"] != test_item["prefix"]]

        x_train_list, y_train_list = [], []
        for item in train_items:
            x_i, y_i = load_xy(item)
            x_train_list.append(x_i)
            y_train_list.append(y_i)

        x_target, y_target = load_xy(test_item)
        n = len(y_target)
        cal_n = int(n * calibration_ratio)

        if cal_n < 20 or n - cal_n < 30:
            print(f"⚠️ {test_item['prefix']} 可校准/测试数据太少，跳过。n={n}, cal_n={cal_n}")
            continue

        x_train_list.append(x_target[:cal_n])
        y_train_list.append(y_target[:cal_n])

        x_train = np.vstack(x_train_list)
        y_train = np.concatenate(y_train_list)
        x_test = x_target[cal_n:]
        y_test = y_target[cal_n:]

        y_pred, model, scaler = train_predict(x_train, y_train, x_test, context_mode=context_mode)

        title = f"personal_calibration_{int(calibration_ratio * 100)}pct__test_{test_item['prefix']}"
        row = evaluate_and_save(y_test, y_pred, title, result_dir, stage_names)
        row["test_subject"] = test_item["prefix"]
        row["calibration_ratio"] = calibration_ratio
        row["context_mode"] = context_mode
        rows.append(row)

        if save_models:
            model_dir = result_dir / "saved_models"
            model_dir.mkdir(parents=True, exist_ok=True)

            # 保存模型时同时保存上下文模式、特征名，后面硬件实时接入时可以直接复用。
            feature_names = test_item.get("meta", {}).get("feature_names")
            raw_dim = x_target.shape[1]
            extended_feature_names = make_extended_feature_names(feature_names, raw_dim, context_mode)
            joblib.dump(
                {
                    "model": model,
                    "scaler": scaler,
                    "mode": test_item.get("meta", {}).get("channel_mode"),
                    "context_mode": context_mode,
                    "stage_names": list(stage_names),
                    "feature_names": feature_names,
                    "extended_feature_names": extended_feature_names,
                    "raw_feature_dim": int(raw_dim),
                },
                model_dir / f"{title}.joblib",
            )

        print(f"test={test_item['prefix']}: acc={row['accuracy']:.3f}, macroF1={row['macro_f1']:.3f}")
    return rows


def run_feature_importance_analysis(
    subjects: List[Dict[str, Any]],
    result_dir: Path,
    context_mode: str,
) -> None:
    """训练一个全局模型，输出特征重要性排序。"""
    print("\n" + "=" * 90)
    print("实验4：全局特征重要性分析")

    if not subjects:
        print("⚠️ 没有可用被试，跳过特征重要性分析。")
        return

    x_all_list, y_all_list = [], []
    feature_names: Optional[List[str]] = None
    raw_dim: Optional[int] = None

    for item in subjects:
        x, y = load_xy(item)
        x_all_list.append(x)
        y_all_list.append(y)
        if feature_names is None and item.get("meta") and "feature_names" in item["meta"]:
            feature_names = item["meta"]["feature_names"]
            raw_dim = x.shape[1]

    x_all = np.vstack(x_all_list)
    y_all = np.concatenate(y_all_list)
    raw_dim = raw_dim or x_all.shape[1]

    extended_names = make_extended_feature_names(feature_names, raw_dim, context_mode)

    scaler = StandardScaler()
    x_s = scaler.fit_transform(x_all)
    x_ctx = add_temporal_context(x_s, mode=context_mode)

    weights = compute_sample_weight(class_weight="balanced", y=y_all)
    model = make_model()
    model.fit(x_ctx, y_all, sample_weight=weights)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n✅ 核心贡献特征 Top 20：")
    top_features = []
    for rank, idx in enumerate(indices):
        fname = extended_names[idx] if idx < len(extended_names) else f"Feature_{idx}"
        imp = float(importances[idx])
        top_features.append({
            "Rank": rank + 1,
            "Feature_Name": fname,
            "Importance_Score": round(imp, 6),
        })
        if rank < 20:
            print(f"Top {rank + 1:02d} | 贡献度: {imp:.4f} | 特征: {fname}")

    result_dir.mkdir(parents=True, exist_ok=True)
    out_csv = result_dir / "feature_importance_ranking.csv"
    pd.DataFrame(top_features).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n-> 已保存完整特征排行表到：{out_csv}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VSCode 友好版睡眠分期模型训练脚本")
    parser.add_argument("--feature-dir", default=CONFIG["FEATURE_DIR"], help="01 脚本生成的 features 文件夹")
    parser.add_argument("--result-dir", default=CONFIG["RESULT_DIR"], help="结果输出文件夹")
    parser.add_argument("--mode", choices=["all32", "dual2"], default=CONFIG["CHANNEL_MODE"], help="通道模式，必须和 01 脚本一致")
    parser.add_argument("--context-mode", choices=["causal", "centered", "none"], default=CONFIG["CONTEXT_MODE"], help="时序上下文模式")
    parser.add_argument("--within-train-ratio", type=float, default=CONFIG["WITHIN_TRAIN_RATIO"], help="同被试时间切分训练比例")
    parser.add_argument("--calibration-ratio", type=float, default=CONFIG["CALIBRATION_RATIO"], help="个体化校准比例")
    parser.add_argument("--skip-within", action="store_true", help="跳过同被试时间切分")
    parser.add_argument("--skip-loso", action="store_true", help="跳过 LOSO")
    parser.add_argument("--skip-calibration", action="store_true", help="跳过个体化校准")
    parser.add_argument("--skip-importance", action="store_true", help="跳过特征重要性分析")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    feature_dir = resolve_path(args.feature_dir)
    result_dir = resolve_path(args.result_dir)

    print("\n当前脚本目录：", PROJECT_ROOT)
    print("当前模式：", args.mode)
    print("时序上下文：", args.context_mode)
    print("特征目录：", feature_dir)
    print("结果目录：", result_dir)

    subjects = load_available_subjects(feature_dir, args.mode)

    print(f"找到 {len(subjects)} 个特征文件：")
    for s in subjects:
        x_tmp = np.load(s["x_path"], mmap_mode="r")
        print(f" - {s['prefix']}, X维度={x_tmp.shape}, 文件={s['x_path']}")

    if len(subjects) == 0:
        print(
            f"\n❌ 没有找到 {feature_dir}/*_{args.mode}_X.npy。\n"
            "请先运行 01_extract_features_LIGHT_FIR_vscode.py，并确认 --mode 一致。\n"
            "例如：python 01_extract_features_LIGHT_FIR_vscode.py --subject-id day1 --edf data/day1.edf --label data/day1.txt --mode portable6"
        )
        return 1

    all_rows: List[Dict[str, Any]] = []
    stage_names = CONFIG["STAGE_NAMES"]

    if not args.skip_within:
        all_rows.extend(run_within_subject_time_split(subjects, result_dir, args.within_train_ratio, args.context_mode, stage_names))

    if not args.skip_loso:
        all_rows.extend(run_loso(subjects, result_dir, args.context_mode, stage_names))

    if not args.skip_calibration:
        all_rows.extend(run_personal_calibration(subjects, result_dir, args.calibration_ratio, args.context_mode, stage_names))

    if not args.skip_importance:
        run_feature_importance_analysis(subjects, result_dir, args.context_mode)

    result_dir.mkdir(parents=True, exist_ok=True)
    out_csv = result_dir / "model_results_summary.csv"
    pd.DataFrame(all_rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print("全部实验完成")
    print(f"模型评估汇总表已保存：{out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
