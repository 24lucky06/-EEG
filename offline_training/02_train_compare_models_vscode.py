"""
02_train_compare_models_vscode.py

VSCode 友好版离线模型训练与对比脚本。

定位：
1. 本脚本属于 offline_training 离线训练端；
2. 它不直接读取硬件，也不负责实时 UI；
3. 它负责训练睡眠分期模型，并输出可供 realtime_system 加载的模型文件；
4. 如果没有真实睡眠质量标签，本脚本不会伪造“睡眠质量监督训练”，而是基于分期序列计算规则型睡眠质量评分。

功能：
1. 读取 01_extract_features_LIGHT_FIR_vscode.py 生成的 features/*_X.npy、*_y.npy、*_meta.json；
2. 使用 XGBoost 完成 W / N1 / N2 / N3 / REM 五分类睡眠分期；
3. 支持同被试时间切分、LOSO、个体化校准、特征重要性分析；
4. 默认使用 causal 时序上下文：前两段 + 当前段，便于后续接硬件实时预测；
5. 保存 global 模型到 saved_models/，dual2 + causal 模型就是后续实时系统优先加载的模型；
6. 输出睡眠质量评分 CSV：基于真实标签，以及基于 global 模型预测序列。

推荐 VSCode 运行方式：
- 先运行 01 脚本生成 features 文件夹；
- 在 VSCode 终端运行：
  python 02_train_compare_models_vscode.py --mode dual2 --context-mode causal

重要提醒：
- all32 模型用于离线研究和特征重要性分析；
- dual2 + causal 模型才适合后续双导联硬件实时预测；
- 32 导联模型不能直接给双导联硬件用，因为输入特征维度不同。
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

from sleep_quality_score import compute_sleep_quality, result_to_dict, stage_sequence_to_names

PROJECT_ROOT = Path(__file__).resolve().parent


CONFIG: Dict[str, Any] = {
    "FEATURE_DIR": "features",
    "RESULT_DIR": "results",
    "MODEL_DIR": "saved_models",
    "CHANNEL_MODE": "dual2",
    "WITHIN_TRAIN_RATIO": 0.7,
    "CALIBRATION_RATIO": 0.2,
    "STAGE_NAMES": ["W", "N1", "N2", "N3", "REM"],
    # causal = 前两段 + 当前段，适合实时；centered = 上一段 + 当前段 + 下一段，适合离线分析。
    "CONTEXT_MODE": "causal",
    "EPOCH_SECONDS": 30,
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
    model_dir: Path,
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
            model_dir.mkdir(parents=True, exist_ok=True)
            feature_names = test_item.get("meta", {}).get("feature_names")
            raw_dim = x_target.shape[1]
            extended_feature_names = make_extended_feature_names(feature_names, raw_dim, context_mode)
            joblib.dump(
                {
                    "model": model,
                    "scaler": scaler,
                    "model_role": "personal_calibration",
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


def train_global_model(
    subjects: List[Dict[str, Any]],
    context_mode: str,
) -> Tuple[xgb.XGBClassifier, StandardScaler, List[str], int, Optional[List[str]], np.ndarray, np.ndarray]:
    """训练一个使用全部被试数据的全局模型，用于保存给 realtime_system。"""
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
    scaler.fit(x_all)

    # 为了避免不同被试边界之间产生错误的前两段上下文，这里每个被试单独拼接 temporal context。
    x_ctx_list, y_ctx_list = [], []
    for item in subjects:
        x_i, y_i = load_xy(item)
        x_i_s = scaler.transform(x_i)
        x_ctx_list.append(add_temporal_context(x_i_s, mode=context_mode))
        y_ctx_list.append(y_i)

    x_ctx_all = np.vstack(x_ctx_list)
    y_ctx_all = np.concatenate(y_ctx_list)

    weights = compute_sample_weight(class_weight="balanced", y=y_ctx_all)
    model = make_model()
    model.fit(x_ctx_all, y_ctx_all, sample_weight=weights)
    return model, scaler, extended_names, int(raw_dim), feature_names, x_ctx_all, y_ctx_all


def save_global_model_and_prediction_quality(
    subjects: List[Dict[str, Any]],
    result_dir: Path,
    model_dir: Path,
    mode: str,
    context_mode: str,
    stage_names: Sequence[str],
    epoch_seconds: int,
) -> None:
    """保存全局模型，并用该模型对每个被试全序列预测，输出预测序列的睡眠质量评分。"""
    print("\n" + "=" * 90)
    print("实验5：保存全局模型，并生成预测序列睡眠质量评分")

    if not subjects:
        print("⚠️ 没有可用被试，跳过全局模型保存。")
        return

    model, scaler, extended_names, raw_dim, feature_names, _, _ = train_global_model(subjects, context_mode)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"sleep_stage_{mode}_{context_mode}_global.joblib"
    first_meta = subjects[0].get("meta", {}) if subjects else {}
    artifact = {
        "model": model,
        "scaler": scaler,
        "model_role": "global_realtime_candidate",
        "mode": mode,
        "context_mode": context_mode,
        "stage_names": list(stage_names),
        "feature_names": feature_names,
        "extended_feature_names": extended_names,
        "raw_feature_dim": raw_dim,
        "epoch_seconds": epoch_seconds,
        "channels": first_meta.get("channels"),
        "n_channels": first_meta.get("n_channels"),
        "note": "dual2 + causal 模型适合后续双导联硬件实时系统；all32 模型仅适合离线研究，不能直接用于双导联硬件。",
    }
    joblib.dump(artifact, model_path)
    print(f"✅ 全局模型已保存：{model_path}")

    quality_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    for item in subjects:
        x, y_true = load_xy(item)
        x_s = scaler.transform(x)
        x_ctx = add_temporal_context(x_s, mode=context_mode)
        y_pred = smooth_predictions(model.predict(x_ctx))

        q_pred = compute_sleep_quality(y_pred, epoch_seconds=epoch_seconds)
        q_true = compute_sleep_quality(y_true, epoch_seconds=epoch_seconds)
        quality_rows.append({
            "subject": item["prefix"],
            "source": "global_model_prediction_trainset_reference",
            **result_to_dict(q_pred),
        })
        quality_rows.append({
            "subject": item["prefix"],
            "source": "label_y_reference",
            **result_to_dict(q_true),
        })

        true_names = stage_sequence_to_names(y_true)
        pred_names = stage_sequence_to_names(y_pred)
        for i, (yt, yp, ytn, ypn) in enumerate(zip(y_true, y_pred, true_names, pred_names)):
            pred_rows.append({
                "subject": item["prefix"],
                "epoch_index": i,
                "y_true": int(yt),
                "y_pred": int(yp),
                "stage_true": ytn,
                "stage_pred": ypn,
            })

        print(f"{item['prefix']}: 预测评分={q_pred.score_0_100}({q_pred.grade})；标签参考评分={q_true.score_0_100}({q_true.grade})")

    quality_path = result_dir / f"sleep_quality_from_global_model_predictions_{mode}_{context_mode}.csv"
    pd.DataFrame(quality_rows).to_csv(quality_path, index=False, encoding="utf-8-sig")
    print(f"✅ 睡眠质量评分已保存：{quality_path}")

    pred_path = result_dir / f"sleep_stage_predictions_global_model_{mode}_{context_mode}.csv"
    pd.DataFrame(pred_rows).to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"✅ 全序列预测结果已保存：{pred_path}")


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

    model, _, extended_names, _, _, _, _ = train_global_model(subjects, context_mode)
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


def run_sleep_quality_from_labels(
    subjects: List[Dict[str, Any]],
    result_dir: Path,
    mode: str,
    epoch_seconds: int,
) -> None:
    """基于真实 y 标签计算离线睡眠质量评分。"""
    print("\n" + "=" * 90)
    print("实验6：基于标签序列计算睡眠质量评分")

    rows: List[Dict[str, Any]] = []
    for item in subjects:
        _, y = load_xy(item)
        result = compute_sleep_quality(y, epoch_seconds=epoch_seconds)
        rows.append({"subject": item["prefix"], "source": "label_y", **result_to_dict(result)})
        print(f"{item['prefix']}: 标签评分={result.score_0_100}，等级={result.grade}，{result.recommendation}")

    out_csv = result_dir / f"sleep_quality_from_labels_{mode}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 标签睡眠质量评分已保存：{out_csv}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VSCode 友好版离线睡眠分期模型训练脚本")
    parser.add_argument("--feature-dir", default=CONFIG["FEATURE_DIR"], help="01 脚本生成的 features 文件夹")
    parser.add_argument("--result-dir", default=CONFIG["RESULT_DIR"], help="结果输出文件夹")
    parser.add_argument("--model-dir", default=CONFIG["MODEL_DIR"], help="模型输出文件夹，默认 saved_models")
    parser.add_argument("--mode", choices=["all32", "dual2"], default=CONFIG["CHANNEL_MODE"], help="通道模式，必须和 01 脚本一致")
    parser.add_argument("--context-mode", choices=["causal", "centered", "none"], default=CONFIG["CONTEXT_MODE"], help="时序上下文模式")
    parser.add_argument("--within-train-ratio", type=float, default=CONFIG["WITHIN_TRAIN_RATIO"], help="同被试时间切分训练比例")
    parser.add_argument("--calibration-ratio", type=float, default=CONFIG["CALIBRATION_RATIO"], help="个体化校准比例")
    parser.add_argument("--epoch-seconds", type=int, default=CONFIG["EPOCH_SECONDS"], help="每个 epoch 秒数")
    parser.add_argument("--skip-within", action="store_true", help="跳过同被试时间切分")
    parser.add_argument("--skip-loso", action="store_true", help="跳过 LOSO")
    parser.add_argument("--skip-calibration", action="store_true", help="跳过个体化校准")
    parser.add_argument("--skip-importance", action="store_true", help="跳过特征重要性分析")
    parser.add_argument("--skip-quality", action="store_true", help="跳过睡眠质量评分输出")
    parser.add_argument("--skip-save-global-model", action="store_true", help="跳过保存全局模型")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    feature_dir = resolve_path(args.feature_dir)
    result_dir = resolve_path(args.result_dir)
    model_dir = resolve_path(args.model_dir)

    print("\n当前脚本目录：", PROJECT_ROOT)
    print("当前模式：", args.mode)
    print("时序上下文：", args.context_mode)
    print("特征目录：", feature_dir)
    print("结果目录：", result_dir)
    print("模型目录：", model_dir)

    if args.mode == "all32":
        print("\n⚠️ 当前是 all32：适合离线研究和特征重要性分析，不适合直接部署到双导联硬件。")
    if args.mode == "dual2" and args.context_mode == "causal":
        print("\n✅ 当前是 dual2 + causal：适合后续 realtime_system 加载并做双导联实时预测。")

    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    subjects = load_available_subjects(feature_dir, args.mode)

    print(f"找到 {len(subjects)} 个特征文件：")
    for s in subjects:
        x_tmp = np.load(s["x_path"], mmap_mode="r")
        print(f" - {s['prefix']}, X维度={x_tmp.shape}, 文件={s['x_path']}")

    if len(subjects) == 0:
        print(
            f"\n❌ 没有找到 {feature_dir}/*_{args.mode}_X.npy。\n"
            "请先运行 01_extract_features_LIGHT_FIR_vscode.py，并确认 --mode 一致。\n"
            "例如：python 01_extract_features_LIGHT_FIR_vscode.py --subject-id day1 --edf data/day1.edf --label data/day1.txt --mode dual2"
        )
        return 1

    all_rows: List[Dict[str, Any]] = []
    stage_names = CONFIG["STAGE_NAMES"]

    if not args.skip_within:
        all_rows.extend(run_within_subject_time_split(subjects, result_dir, args.within_train_ratio, args.context_mode, stage_names))

    if not args.skip_loso:
        all_rows.extend(run_loso(subjects, result_dir, args.context_mode, stage_names))

    if not args.skip_calibration:
        all_rows.extend(run_personal_calibration(subjects, result_dir, model_dir, args.calibration_ratio, args.context_mode, stage_names))

    if not args.skip_importance:
        run_feature_importance_analysis(subjects, result_dir, args.context_mode)

    if not args.skip_save_global_model:
        save_global_model_and_prediction_quality(
            subjects=subjects,
            result_dir=result_dir,
            model_dir=model_dir,
            mode=args.mode,
            context_mode=args.context_mode,
            stage_names=stage_names,
            epoch_seconds=args.epoch_seconds,
        )

    if not args.skip_quality:
        run_sleep_quality_from_labels(subjects, result_dir, args.mode, args.epoch_seconds)

    out_csv = result_dir / "model_results_summary.csv"
    pd.DataFrame(all_rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print("全部离线实验完成")
    print(f"模型评估汇总表已保存：{out_csv}")
    print(f"可部署候选模型目录：{model_dir}")
    if args.mode == "dual2" and args.context_mode == "causal":
        print("后续 realtime_system 应优先加载：sleep_stage_dual2_causal_global.joblib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
