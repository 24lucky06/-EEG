"""
01_extract_features_LIGHT_FIR_vscode.py

VSCode 友好版 EEG 离线特征提取脚本。

功能：
0. 本脚本属于 offline_training 离线训练端，不直接连接硬件；
1. 读取 EDF 脑电文件和 TXT 睡眠分期标签；
2. 支持 all32 / dual2 两种通道模式；
3. 支持自由控制处理长度；
4. 使用 FIR 带通滤波、降采样、30 秒 epoch 切片；
5. 提取时域、频域功率、Hjorth、Shannon entropy、Petrosian FD 等特征；
6. 输出 X.npy / y.npy / meta.json，供 02_train_compare_models_vscode.py 使用。

推荐 VSCode 运行方式：
- 把 EDF 和 TXT 放到本脚本同级目录下的 data 文件夹；
- 在 VSCode 终端运行：
  python 01_extract_features_LIGHT_FIR_vscode.py --subject-id day1 --edf data/day1.edf --label data/day1.txt --mode dual2 --max-epochs 120

  默认 dual2 选择 C3 和 O1。也可以自定义两个通道，例如：
  python 01_extract_features_LIGHT_FIR_vscode.py --subject-id day1 --edf data/day1.edf --label data/day1.txt --mode dual2 --dual2-channels C4 O2 --max-epochs 120

也可以批量运行：
- 复制 subjects_template.json 为 subjects.json；
- 填写多个被试；
- 运行：
  python 01_extract_features_LIGHT_FIR_vscode.py --subjects-file subjects.json --mode dual2 --max-epochs 120
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, skew

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

PROJECT_ROOT = Path(__file__).resolve().parent


# =========================
# 默认配置。VSCode 里也可以直接改这里。
# =========================
CONFIG: Dict[str, Any] = {
    # all32：尽量选择 32 个 EEG 通道；dual2：默认 C3 和 O1 两个通道，可通过 --dual2-channels 自定义。
    "CHANNEL_MODE": "dual2",

    # 输出特征文件夹，默认是当前脚本所在目录下的 features。
    "OUTPUT_DIR": "features",

    # 默认被试列表。也可以通过命令行参数或 subjects.json 覆盖。
    "subjects": [
        {"subject_id": "day1", "edf": "data/day1.edf", "label": "data/day1.txt"},
    ],

    # FIR 带通滤波范围。
    "LOW_FREQ": 0.3,
    "HIGH_FREQ": 35.0,

    # 降采样频率。
    "RESAMPLE_FREQ": 100.0,

    # 睡眠分期常用 30 秒一个 epoch。
    "EPOCH_SECONDS": 30,

    # 振幅裁剪阈值，单位微伏。
    "CLIP_UV": 150,

    # 120 = 前 60 分钟；240 = 前 120 分钟；None = 尽量处理整晚。
    "MAX_EPOCHS_PER_SUBJECT": 120,

    # dual2 模式默认选用的两个通道。
    "DUAL2_CHANNELS": ["C3", "O1"],
}

DUAL2_DEFAULT_CHANNELS = ["C3", "O1"]

STANDARD_32_CHANNELS = [
    "FP1", "FP2",
    "F7", "F3", "FZ", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "CZ", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6",
    "P7", "P3", "PZ", "P4", "P8",
    "PO3", "POZ", "PO4",
    "O1", "OZ", "O2", "IZ",
]

STAGE_MAP = {
    "W": 0, "WAKE": 0, "0": 0, "0.0": 0,
    "N1": 1, "S1": 1, "1": 1, "1.0": 1,
    "N2": 2, "S2": 2, "2": 2, "2.0": 2,
    "N3": 3, "S3": 3, "S4": 3, "3": 3, "3.0": 3,
    "R": 4, "REM": 4, "4": 4, "4.0": 4, "5": 4, "5.0": 4,
}


def resolve_path(path_like: str | os.PathLike[str]) -> Path:
    """把相对路径统一解释为：相对于脚本所在文件夹，而不是 VSCode 当前工作目录。"""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_channel_name(ch: str) -> str:
    """统一通道名称格式，方便匹配不同 EDF 里的命名方式。"""
    ch = str(ch).upper()
    for token in ["EEG", "REF", "LE", "RE", "-M1", "-M2", "-A1", "-A2"]:
        ch = ch.replace(token, "")
    for token in ["-", "_", ".", " ", ":"]:
        ch = ch.replace(token, "")
    return ch


def is_bad_non_eeg_channel(ch: str) -> bool:
    """判断一个通道是否明显不是 EEG 通道，例如眼电、肌电、心电、呼吸等。"""
    u = str(ch).upper()
    bad_keywords = [
        "EOG", "EMG", "ECG", "EKG", "CHIN", "LEG", "RESP", "AIRFLOW",
        "SNORE", "SAO2", "SPO2", "PLETH", "POSITION", "LIGHT", "MARKER",
        "TRIG", "EVENT", "PRESSURE", "FLOW", "THOR", "ABDO",
    ]
    return any(k in u for k in bad_keywords)


def pick_channels(raw: mne.io.BaseRaw, mode: str, dual2_channels: Optional[Sequence[str]] = None) -> List[str]:
    """根据 all32 / dual2 选择 EEG 通道。

    dual2 模式下使用 dual2_channels 指定的两个通道，默认 C3 + O1。
    """
    raw_names = list(raw.ch_names)
    norm_to_raw = {normalize_channel_name(ch): ch for ch in raw_names}

    if mode == "dual2":
        targets = list(dual2_channels) if dual2_channels else list(DUAL2_DEFAULT_CHANNELS)
        if len(targets) != 2:
            raise ValueError(
                f"dual2 模式必须指定两个通道，但收到 {len(targets)} 个：{targets}"
            )

        selected: List[str] = []
        missing: List[str] = []
        for target in targets:
            target_norm = normalize_channel_name(target)
            matched = norm_to_raw.get(target_norm)
            if matched is None:
                for raw_ch in raw_names:
                    if target_norm == normalize_channel_name(raw_ch):
                        matched = raw_ch
                        break
            if matched is None:
                for raw_ch in raw_names:
                    if target_norm in normalize_channel_name(raw_ch):
                        matched = raw_ch
                        break
            if matched is None:
                missing.append(target)
            elif matched not in selected:
                selected.append(matched)

        if missing:
            raise ValueError(
                f"dual2 模式找不到以下通道：{missing}\n"
                f"目标通道：{targets}\n"
                f"EDF 通道名如下：{raw_names}\n"
                "解决方法：1）确认 EDF 通道名拼写；2）通过 --dual2-channels 指定实际存在的两个通道。"
            )
        if len(selected) != 2:
            raise ValueError(
                f"dual2 模式需要正好 2 个通道，但匹配到 {len(selected)} 个：{selected}\n"
                f"目标通道：{targets}\n"
                f"EDF 通道名如下：{raw_names}"
            )
        return selected

    if mode == "all32":
        selected = []
        for target in STANDARD_32_CHANNELS:
            target_norm = normalize_channel_name(target)
            matched = norm_to_raw.get(target_norm)
            if matched is None:
                for raw_ch in raw_names:
                    raw_norm = normalize_channel_name(raw_ch)
                    if target_norm == raw_norm or target_norm in raw_norm:
                        matched = raw_ch
                        break
            if matched is not None and matched not in selected:
                selected.append(matched)

        # 如果标准通道没有凑够 32 个，则补充看起来像 EEG 的通道。
        for ch in raw_names:
            if ch in selected:
                continue
            if is_bad_non_eeg_channel(ch):
                continue
            selected.append(ch)
            if len(selected) >= 32:
                break

        if len(selected) < 32:
            raise ValueError(
                f"all32 模式要求至少 32 个 EEG 通道，但只找到 {len(selected)} 个：{selected}\n"
                f"EDF 全部通道名如下：{raw_names}\n"
                "解决方法：1）确认 EDF 是否为多导数据；2）先使用 --mode dual2 跑通流程。"
            )
        return selected[:32]

    raise ValueError("CHANNEL_MODE 只能是 all32 或 dual2")


def load_custom_labels(txt_path: Path) -> np.ndarray:
    """读取 TXT 睡眠分期标签，并统一映射为数字：W=0, N1=1, N2=2, N3=3, REM=4。"""
    if not txt_path.exists():
        raise FileNotFoundError(f"找不到标签文件：{txt_path}")

    try:
        df = pd.read_csv(txt_path, sep=r"\s+", header=None)
        raw_labels = df.iloc[:, 0].values
    except Exception:
        raw_labels = np.loadtxt(txt_path, dtype=str)

    labels = []
    unknown: Dict[str, int] = {}
    for s in raw_labels:
        key = str(s).strip().upper()
        if key in STAGE_MAP:
            labels.append(STAGE_MAP[key])
        else:
            unknown[key] = unknown.get(key, 0) + 1
            labels.append(0)

    if unknown:
        print(f"⚠️ 未识别标签已临时按 W 处理：{unknown}")

    return np.asarray(labels, dtype=int)


def calculate_hjorth_parameters(signal: np.ndarray) -> Tuple[float, float, float]:
    """计算 Hjorth 三个特征：Activity、Mobility、Complexity。"""
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)
    var_zero = float(np.var(signal))
    var_d1 = float(np.var(first_deriv))
    var_d2 = float(np.var(second_deriv))

    activity = var_zero
    mobility = float(np.sqrt(var_d1 / var_zero)) if var_zero > 1e-20 else 0.0
    complexity = float(np.sqrt(var_d2 / var_d1) / mobility) if (var_d1 > 1e-20 and mobility > 1e-20) else 0.0
    return activity, mobility, complexity


def calculate_pfd(signal: np.ndarray) -> float:
    """计算 Petrosian Fractal Dimension，用于描述信号复杂度。"""
    diff = np.diff(signal)
    if len(diff) < 2:
        return 0.0
    n_delta = int(np.sum(diff[1:] * diff[:-1] < 0))
    n = len(signal)
    if n_delta == 0 or n <= 1:
        return 0.0
    return float(np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_delta))))


def calculate_shannon_entropy(signal: np.ndarray, num_bins: int = 10) -> float:
    """计算 Shannon entropy。这里使用直方图概率，而不是 density，避免熵值异常。"""
    counts, _ = np.histogram(signal, bins=num_bins, density=False)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


def extract_one_channel_features(ch_data: np.ndarray, sfreq: float) -> Tuple[List[float], List[str]]:
    """对单个通道、单个 30 秒 epoch 提取一组人工 EEG 特征。"""
    features: List[float] = []
    names: List[str] = []

    time_features = {
        "mean": float(np.mean(ch_data)),
        "var": float(np.var(ch_data)),
        "skew": float(skew(ch_data, nan_policy="omit")),
        "kurtosis": float(kurtosis(ch_data, nan_policy="omit")),
        "ptp": float(np.ptp(ch_data)),
    }
    for k, v in time_features.items():
        names.append(k)
        features.append(v)

    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "sigma": (11, 16),
        "beta": (13, 30),
    }

    nperseg = max(8, min(len(ch_data), int(sfreq * 2)))
    freqs, psd = welch(ch_data, fs=sfreq, nperseg=nperseg)

    band_powers: Dict[str, float] = {}
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        band_powers[band_name] = float(np.trapz(psd[mask], freqs[mask])) if np.any(mask) else 0.0

    total_power = float(sum(band_powers.values()) + 1e-12)

    for band_name in bands:
        names.append(f"{band_name}_abs")
        features.append(band_powers[band_name])

    for band_name in bands:
        names.append(f"{band_name}_rel")
        features.append(band_powers[band_name] / total_power)

    act, mob, comp = calculate_hjorth_parameters(ch_data)
    names.extend(["hjorth_activity", "hjorth_mobility", "hjorth_complexity"])
    features.extend([act, mob, comp])

    names.extend(["shannon_entropy", "petrosian_fd"])
    features.extend([calculate_shannon_entropy(ch_data), calculate_pfd(ch_data)])

    # 防止常数信号导致 skew/kurtosis 为 NaN。
    features = [float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for v in features]
    return features, names


def extract_features_from_edf(
    edf_path: Path,
    label_path: Path,
    subject_id: str,
    channel_mode: str,
    output_dir: Path,
    max_epochs_per_subject: Optional[int],
    low_freq: float,
    high_freq: float,
    resample_freq: float,
    epoch_seconds: int,
    clip_uv: float,
    dual2_channels: Optional[Sequence[str]] = None,
) -> None:
    """读取单个 EDF 和标签文件，完成预处理、切片、特征提取和保存。"""
    print("\n" + "=" * 90)
    print(f"开始处理：{subject_id}")
    print(f"EDF：{edf_path}")
    print(f"标签：{label_path}")
    print(f"模式：{channel_mode}")
    if channel_mode == "dual2":
        print(f"dual2 目标通道：{list(dual2_channels) if dual2_channels else list(DUAL2_DEFAULT_CHANNELS)}")

    if not edf_path.exists():
        raise FileNotFoundError(f"找不到 EDF 文件：{edf_path}")

    labels = load_custom_labels(label_path)

    if max_epochs_per_subject is not None:
        if not isinstance(max_epochs_per_subject, int) or max_epochs_per_subject <= 0:
            raise ValueError("max_epochs_per_subject 必须是正整数，或者设为 None。")
        labels = labels[:max_epochs_per_subject]
        print(f"⚠️ 自定义长度：只处理前 {len(labels)} 个 epoch，也就是 {len(labels) * epoch_seconds / 60:.1f} 分钟")
    else:
        print(f"✅ 全夜模式：不限制 epoch 数量，将尽量处理标签中的全部 {len(labels)} 个 epoch")

    if len(labels) == 0:
        raise ValueError(f"标签文件为空或无法读取：{label_path}")

    required_seconds = len(labels) * epoch_seconds

    # 第一次只读头信息，便于检查通道，不加载整晚数据。
    temp_raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    target_channels = pick_channels(temp_raw, mode=channel_mode, dual2_channels=dual2_channels)
    print(f"✅ 选中通道数：{len(target_channels)}")
    print(f"✅ 选中通道：{target_channels}")

    raw = mne.io.read_raw_edf(edf_path, include=target_channels, preload=False, verbose="ERROR")

    # 关键：先裁剪，再 load_data，降低内存压力。
    tmax = min(float(required_seconds), float(raw.times[-1]))
    raw.crop(tmin=0.0, tmax=tmax, include_tmax=False)

    print("-> 正在加载裁剪后的数据，不是整晚数据...")
    raw.load_data(verbose="ERROR")

    print("-> 正在进行 FIR 带通滤波和降采样...")
    raw.filter(
        l_freq=low_freq,
        h_freq=high_freq,
        method="fir",
        fir_design="firwin",
        verbose="ERROR",
    )
    raw.resample(resample_freq, npad="auto", verbose="ERROR")

    data = raw.get_data(picks=target_channels)
    clip_v = clip_uv * 1e-6
    data = np.clip(data, -clip_v, clip_v).astype(np.float32)

    sfreq = float(raw.info["sfreq"])
    epoch_len = int(epoch_seconds * sfreq)
    max_epochs = min(len(labels), data.shape[1] // epoch_len)
    if max_epochs <= 0:
        raise ValueError("可用数据长度不足 1 个 epoch，请检查 EDF 时长、标签长度或 epoch_seconds 设置。")

    data = data[:, :max_epochs * epoch_len]
    x_raw = data.reshape(len(target_channels), max_epochs, epoch_len).transpose(1, 0, 2)
    y = labels[:max_epochs]

    x_features: List[List[float]] = []
    feature_names: Optional[List[str]] = None

    print("-> 开始提取特征...")
    for i in range(max_epochs):
        epoch_features: List[float] = []
        epoch_feature_names: List[str] = []

        for ch_idx, ch_name in enumerate(target_channels):
            ch_features, ch_feature_names = extract_one_channel_features(x_raw[i, ch_idx, :], sfreq)
            epoch_features.extend(ch_features)
            epoch_feature_names.extend([f"{ch_name}__{name}" for name in ch_feature_names])

        if feature_names is None:
            feature_names = epoch_feature_names

        x_features.append(epoch_features)

        if (i + 1) % 20 == 0 or (i + 1) == max_epochs:
            print(f"已完成 {i + 1}/{max_epochs} 个 epoch")

    x = np.asarray(x_features, dtype=np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{subject_id}_{channel_mode}"
    x_path = output_dir / f"{prefix}_X.npy"
    y_path = output_dir / f"{prefix}_y.npy"
    meta_path = output_dir / f"{prefix}_meta.json"

    np.save(x_path, x)
    np.save(y_path, y)

    meta = {
        "subject_id": subject_id,
        "channel_mode": channel_mode,
        "edf_path": str(edf_path),
        "label_path": str(label_path),
        "channels": target_channels,
        "n_channels": len(target_channels),
        "n_epochs": int(max_epochs),
        "epoch_seconds": epoch_seconds,
        "sfreq": sfreq,
        "feature_dim": int(x.shape[1]),
        "feature_names": feature_names,
        "max_epochs_per_subject": max_epochs_per_subject,
        "filter_method": "FIR",
        "fir_design": "firwin",
        "low_freq": low_freq,
        "high_freq": high_freq,
        "resample_freq": resample_freq,
        "clip_uv": clip_uv,
        "label_meaning": {"0": "W", "1": "N1", "2": "N2", "3": "N3", "4": "REM"},
    }
    if channel_mode == "dual2":
        meta["dual2_target_channels"] = (
            list(dual2_channels) if dual2_channels else list(DUAL2_DEFAULT_CHANNELS)
        )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"🎉 保存完成：{x_path}")
    print(f"标签文件：{y_path}")
    print(f"元信息：{meta_path}")
    print(f"特征矩阵维度：{x.shape}")


def load_subjects_from_file(subjects_file: Path) -> List[Dict[str, str]]:
    """从 subjects.json 读取被试列表。"""
    if not subjects_file.exists():
        raise FileNotFoundError(f"找不到 subjects 文件：{subjects_file}")
    with open(subjects_file, "r", encoding="utf-8") as f:
        subjects = json.load(f)
    if not isinstance(subjects, list):
        raise ValueError("subjects 文件格式应为列表，例如 [{\"subject_id\":\"day1\", \"edf\":\"data/day1.edf\", \"label\":\"data/day1.txt\"}]")
    for item in subjects:
        if not all(k in item for k in ["subject_id", "edf", "label"]):
            raise ValueError(f"subjects 中每项都需要 subject_id / edf / label：{item}")
    return subjects


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VSCode 友好版 EEG 特征提取脚本")
    parser.add_argument("--mode", choices=["all32", "dual2"], default=CONFIG["CHANNEL_MODE"], help="通道模式")
    parser.add_argument("--output-dir", default=CONFIG["OUTPUT_DIR"], help="特征输出文件夹")
    parser.add_argument("--subject-id", default=None, help="单个被试 ID，例如 day1")
    parser.add_argument("--edf", default=None, help="单个 EDF 文件路径，例如 data/day1.edf")
    parser.add_argument("--label", default=None, help="单个标签 TXT 文件路径，例如 data/day1.txt")
    parser.add_argument("--subjects-file", default=None, help="批量被试 JSON 文件，例如 subjects.json")
    parser.add_argument("--max-epochs", type=str, default=str(CONFIG["MAX_EPOCHS_PER_SUBJECT"]), help="最大 epoch 数。填 None 表示整晚")
    parser.add_argument("--low-freq", type=float, default=CONFIG["LOW_FREQ"], help="FIR 带通低截止频率")
    parser.add_argument("--high-freq", type=float, default=CONFIG["HIGH_FREQ"], help="FIR 带通高截止频率")
    parser.add_argument("--resample-freq", type=float, default=CONFIG["RESAMPLE_FREQ"], help="重采样频率")
    parser.add_argument("--epoch-seconds", type=int, default=CONFIG["EPOCH_SECONDS"], help="每个 epoch 秒数")
    parser.add_argument("--clip-uv", type=float, default=CONFIG["CLIP_UV"], help="振幅裁剪阈值，单位微伏")
    parser.add_argument(
        "--dual2-channels",
        nargs=2,
        default=list(CONFIG["DUAL2_CHANNELS"]),
        metavar=("CH1", "CH2"),
        help="dual2 模式选用的两个通道名，默认 C3 O1。例如：--dual2-channels C4 O2",
    )
    return parser.parse_args(argv)


def parse_max_epochs(value: str) -> Optional[int]:
    if str(value).strip().lower() in {"none", "null", "all", "full", "整晚"}:
        return None
    max_epochs = int(value)
    if max_epochs <= 0:
        raise ValueError("--max-epochs 必须是正整数，或者填 None")
    return max_epochs


def build_subjects(args: argparse.Namespace) -> List[Dict[str, str]]:
    """根据命令行参数、subjects.json 或 CONFIG 生成被试列表。"""
    if args.subjects_file:
        return load_subjects_from_file(resolve_path(args.subjects_file))

    if args.edf or args.label or args.subject_id:
        if not (args.edf and args.label and args.subject_id):
            raise ValueError("使用单个被试模式时，--subject-id、--edf、--label 三个参数必须都提供。")
        return [{"subject_id": args.subject_id, "edf": args.edf, "label": args.label}]

    return CONFIG["subjects"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = resolve_path(args.output_dir)
    max_epochs = parse_max_epochs(args.max_epochs)
    subjects = build_subjects(args)

    dual2_channels = list(args.dual2_channels) if args.mode == "dual2" else None

    print("\n当前脚本目录：", PROJECT_ROOT)
    print("当前通道模式：", args.mode)
    if args.mode == "dual2":
        print("dual2 通道：", dual2_channels)
    print("输出目录：", output_dir)
    print("被试数量：", len(subjects))

    success_count = 0
    for item in subjects:
        try:
            extract_features_from_edf(
                edf_path=resolve_path(item["edf"]),
                label_path=resolve_path(item["label"]),
                subject_id=str(item["subject_id"]),
                channel_mode=args.mode,
                output_dir=output_dir,
                max_epochs_per_subject=max_epochs,
                low_freq=args.low_freq,
                high_freq=args.high_freq,
                resample_freq=args.resample_freq,
                epoch_seconds=args.epoch_seconds,
                clip_uv=args.clip_uv,
                dual2_channels=dual2_channels,
            )
            success_count += 1
        except Exception as exc:
            print("\n❌ 处理失败：", item)
            print("错误原因：", exc)
            print("提示：如果 all32 找不到 32 个通道，先试试 --mode dual2；如果路径报错，把 EDF/TXT 放到 data 文件夹。")

    print("\n" + "=" * 90)
    print(f"特征提取结束：成功 {success_count}/{len(subjects)} 个被试")
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
