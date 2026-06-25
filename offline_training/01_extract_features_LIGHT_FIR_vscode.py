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
  python 01_extract_features_LIGHT_FIR_vscode.py --subject-id day1 --edf ../data/day1.edf --label ../data/day1.txt --mode dual2 --max-epochs 120

  默认 dual2 选择 config/settings.py 中的 DUAL2_CHANNELS（当前为 Fp1 和 Fp2）。
  也可以自定义两个通道，例如：
  python 01_extract_features_LIGHT_FIR_vscode.py --subject-id day1 --edf ../data/day1.edf --label ../data/day1.txt --mode dual2 --dual2-channels Fp2 O2 --max-epochs 120

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
import re
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))
from shared.feature_definitions import (
    calculate_hjorth_parameters,
    calculate_pfd,
    calculate_shannon_entropy,
    extract_one_channel_features,
)
from shared.preprocessing import attenuate_slow_eye_artifacts, epoch_artifact_flags
from config.settings import (
    CHANNEL_MODE,
    CLIP_UV,
    DUAL2_CHANNELS,
    EPOCH_SECONDS,
    FEATURE_DIR,
    HIGH_FREQ,
    LOW_FREQ,
    MAX_EPOCHS_PER_SUBJECT,
    SFREQ,
)

# 默认被试列表（仅命令行无参数时使用）。信号参数见 config/settings.py。
_DEFAULT_SUBJECTS = [
    {"subject_id": "day1", "edf": "../data/day1.edf", "label": "../data/day1.txt"},
]

DUAL2_DEFAULT_CHANNELS = list(DUAL2_CHANNELS)

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
    compact = re.sub(r"[^A-Z0-9]+", "", str(ch).upper())
    return strip_channel_affixes(compact)


def strip_channel_affixes(compact_name: str) -> str:
    """去掉 EDF 常见的 EEG/POL 前缀和 M1/M2/A1/A2/REF 等参考后缀。"""
    name = compact_name
    prefixes = ("EEG", "POL", "CHANNEL", "CHAN")
    suffixes = ("REF", "AVG", "LE", "RE", "M1", "M2", "A1", "A2")

    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix) and len(name) > len(prefix):
                name = name[len(prefix):]
                changed = True
        for suffix in suffixes:
            if name.endswith(suffix) and len(name) > len(suffix):
                name = name[:-len(suffix)]
                changed = True
    return name


def channel_aliases(ch: str) -> List[str]:
    """生成一个 EDF 通道名的多个候选别名，例如 EEG C3-M2 -> C3。"""
    raw = str(ch).upper().strip()
    aliases = set()

    compact_full = re.sub(r"[^A-Z0-9]+", "", raw)
    if compact_full:
        aliases.add(compact_full)
        aliases.add(strip_channel_affixes(compact_full))

    for part in re.split(r"[\s_\-.:/\\()]+", raw):
        compact_part = re.sub(r"[^A-Z0-9]+", "", part)
        if compact_part:
            aliases.add(compact_part)
            aliases.add(strip_channel_affixes(compact_part))

    return [a for a in aliases if a]


def find_best_channel_match(target: str, raw_names: Sequence[str], selected: Sequence[str]) -> Optional[str]:
    """按别名和包含关系给通道匹配打分，返回最佳 EDF 原始通道名。"""
    target_norm = normalize_channel_name(target)
    if not target_norm:
        return None

    selected_set = set(selected)
    best_name: Optional[str] = None
    best_score = -1

    for raw_ch in raw_names:
        if raw_ch in selected_set:
            continue
        aliases = channel_aliases(raw_ch)
        score = -1
        if target_norm in aliases:
            score = 100
        else:
            for alias in aliases:
                if alias == target_norm:
                    score = max(score, 95)
                elif alias.endswith(target_norm) or alias.startswith(target_norm):
                    score = max(score, 80)
                elif target_norm in alias:
                    score = max(score, 70)

        if score > best_score:
            best_score = score
            best_name = raw_ch

    return best_name if best_score >= 70 else None


def read_edf_channel_names(edf_path: Path) -> List[str]:
    """只读取 EDF 头部的通道标签，避免为了选通道而加载巨大注释通道。"""
    with open(edf_path, "rb") as f:
        header = f.read(256)
        if len(header) < 256:
            raise ValueError(f"EDF 文件头不完整：{edf_path}")
        try:
            n_channels = int(header[252:256].decode("ascii").strip())
        except ValueError as exc:
            raise ValueError(f"无法从 EDF 文件头读取通道数：{edf_path}") from exc
        labels = [
            f.read(16).decode("latin1", errors="ignore").strip()
            for _ in range(n_channels)
        ]
    return labels


def is_bad_non_eeg_channel(ch: str) -> bool:
    """判断一个通道是否明显不是 EEG 通道，例如眼电、肌电、心电、呼吸等。"""
    u = str(ch).upper()
    bad_keywords = [
        "EOG", "EMG", "ECG", "EKG", "CHIN", "LEG", "RESP", "AIRFLOW",
        "SNORE", "SAO2", "SPO2", "PLETH", "POSITION", "LIGHT", "MARKER",
        "TRIG", "EVENT", "PRESSURE", "FLOW", "THOR", "ABDO",
    ]
    return any(k in u for k in bad_keywords)


def pick_channels(raw_or_names: mne.io.BaseRaw | Sequence[str], mode: str, dual2_channels: Optional[Sequence[str]] = None) -> List[str]:
    """根据 all32 / dual2 选择 EEG 通道。

    dual2 模式下使用 dual2_channels 指定的两个通道，默认来自 config/settings.py。
    """
    raw_names = list(raw_or_names.ch_names) if hasattr(raw_or_names, "ch_names") else list(raw_or_names)

    if mode == "dual2":
        targets = list(dual2_channels) if dual2_channels else list(DUAL2_DEFAULT_CHANNELS)
        if len(targets) != 2:
            raise ValueError(
                f"dual2 模式必须指定两个通道，但收到 {len(targets)} 个：{targets}"
            )

        selected: List[str] = []
        missing: List[str] = []
        for target in targets:
            matched = find_best_channel_match(target, raw_names, selected)
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
            matched = find_best_channel_match(target, raw_names, selected)
            if matched is not None and matched not in selected:
                selected.append(matched)

        # 如果标准通道没有凑够 32 个，则补充看起来像 EEG 的通道。
        for ch in raw_names:
            if ch in selected:
                continue
            if is_bad_non_eeg_channel(ch):
                continue
            if normalize_channel_name(ch) == "EDFANNOTATIONS":
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
    suppress_eye_artifacts: bool = False,
    eye_low_freq: float = 0.5,
    eye_high_freq: float = 4.0,
    eye_attenuation: float = 0.75,
    artifact_ptp_uv: float = 0.0,
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

    # 只读 EDF 文件头选择通道，避免 MNE 为 EDF Annotations 分配大量内存。
    edf_channels = read_edf_channel_names(edf_path)
    target_channels = pick_channels(edf_channels, mode=channel_mode, dual2_channels=dual2_channels)
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

    data = raw.get_data(picks=target_channels).astype(np.float32)
    if suppress_eye_artifacts:
        print(
            f"-> 正在进行低频眼动抑制：{eye_low_freq:g}-{eye_high_freq:g}Hz, "
            f"attenuation={eye_attenuation:g}"
        )
        data = attenuate_slow_eye_artifacts(
            data,
            float(raw.info["sfreq"]),
            low_freq=eye_low_freq,
            high_freq=eye_high_freq,
            attenuation=eye_attenuation,
        )
    artifact_check_data = data.copy()
    clip_v = clip_uv * 1e-6
    data = np.clip(data, -clip_v, clip_v).astype(np.float32)

    sfreq = float(raw.info["sfreq"])
    epoch_len = int(epoch_seconds * sfreq)
    max_epochs = min(len(labels), data.shape[1] // epoch_len)
    if max_epochs <= 0:
        raise ValueError("可用数据长度不足 1 个 epoch，请检查 EDF 时长、标签长度或 epoch_seconds 设置。")

    data = data[:, :max_epochs * epoch_len]
    artifact_check_data = artifact_check_data[:, :max_epochs * epoch_len]
    x_raw = data.reshape(len(target_channels), max_epochs, epoch_len).transpose(1, 0, 2)
    x_artifact = artifact_check_data.reshape(len(target_channels), max_epochs, epoch_len).transpose(1, 0, 2)
    y = labels[:max_epochs]

    x_features: List[List[float]] = []
    y_kept: List[int] = []
    skipped_epochs = 0
    feature_names: Optional[List[str]] = None

    print("-> 开始提取特征...")
    for i in range(max_epochs):
        rejected, reason = epoch_artifact_flags(x_artifact[i], artifact_ptp_uv)
        if rejected:
            skipped_epochs += 1
            if skipped_epochs <= 10:
                print(f"跳过 epoch {i + 1}: {reason}")
            continue

        epoch_features: List[float] = []
        epoch_feature_names: List[str] = []

        for ch_idx, ch_name in enumerate(target_channels):
            ch_features, ch_feature_names = extract_one_channel_features(x_raw[i, ch_idx, :], sfreq)
            epoch_features.extend(ch_features)
            epoch_feature_names.extend([f"{ch_name}__{name}" for name in ch_feature_names])

        if feature_names is None:
            feature_names = epoch_feature_names

        x_features.append(epoch_features)
        y_kept.append(int(y[i]))

        if (i + 1) % 20 == 0 or (i + 1) == max_epochs:
            print(f"已完成 {i + 1}/{max_epochs} 个 epoch")

    x = np.asarray(x_features, dtype=np.float32)
    y = np.asarray(y_kept, dtype=int)
    if x.size == 0 or y.size == 0:
        raise ValueError("所有 epoch 都被伪迹规则跳过了，请调大 --artifact-ptp-uv 或关闭伪迹跳过。")
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
        "n_epochs_kept": int(len(y)),
        "n_epochs_skipped_artifact": int(skipped_epochs),
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
        "preprocessing": {
            "suppress_eye_artifacts": bool(suppress_eye_artifacts),
            "eye_low_freq": float(eye_low_freq),
            "eye_high_freq": float(eye_high_freq),
            "eye_attenuation": float(eye_attenuation),
            "artifact_ptp_uv": float(artifact_ptp_uv),
        },
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
        raise ValueError("subjects 文件格式应为列表，例如 [{\"subject_id\":\"day1\", \"edf\":\"../data/day1.edf\", \"label\":\"../data/day1.txt\"}]")
    for item in subjects:
        if not all(k in item for k in ["subject_id", "edf", "label"]):
            raise ValueError(f"subjects 中每项都需要 subject_id / edf / label：{item}")
    return subjects


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VSCode 友好版 EEG 特征提取脚本")
    parser.add_argument("--mode", choices=["all32", "dual2"], default=CHANNEL_MODE, help="通道模式")
    parser.add_argument("--output-dir", default=str(FEATURE_DIR), help="特征输出文件夹")
    parser.add_argument("--subject-id", default=None, help="单个被试 ID，例如 day1")
    parser.add_argument("--edf", default=None, help="单个 EDF 文件路径，例如 ../data/day1.edf")
    parser.add_argument("--label", default=None, help="单个标签 TXT 文件路径，例如 ../data/day1.txt")
    parser.add_argument("--subjects-file", default=None, help="批量被试 JSON 文件，例如 subjects.json")
    parser.add_argument("--max-epochs", type=str, default=str(MAX_EPOCHS_PER_SUBJECT), help="最大 epoch 数。填 None 表示整晚")
    parser.add_argument("--low-freq", type=float, default=LOW_FREQ, help="FIR 带通低截止频率")
    parser.add_argument("--high-freq", type=float, default=HIGH_FREQ, help="FIR 带通高截止频率")
    parser.add_argument("--resample-freq", type=float, default=SFREQ, help="重采样频率")
    parser.add_argument("--epoch-seconds", type=int, default=EPOCH_SECONDS, help="每个 epoch 秒数")
    parser.add_argument("--clip-uv", type=float, default=CLIP_UV, help="振幅裁剪阈值，单位微伏")
    parser.add_argument("--suppress-eye-artifacts", action="store_true", help="启用双导低频眼动抑制，训练端和实时端必须保持一致")
    parser.add_argument("--eye-low-freq", type=float, default=0.5, help="眼动抑制低截止频率")
    parser.add_argument("--eye-high-freq", type=float, default=4.0, help="眼动抑制高截止频率")
    parser.add_argument("--eye-attenuation", type=float, default=0.75, help="眼动低频成分衰减强度，0-1")
    parser.add_argument("--artifact-ptp-uv", type=float, default=0.0, help="epoch 峰峰值超过该阈值则跳过；0 表示不跳过")
    parser.add_argument(
        "--dual2-channels",
        nargs=2,
        default=list(DUAL2_CHANNELS),
        metavar=("CH1", "CH2"),
        help="dual2 模式选用的两个通道名，默认来自 config/settings.py。例如：--dual2-channels Fp2 O2",
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
    """根据命令行参数、subjects.json 或默认列表生成被试列表。"""
    if args.subjects_file:
        return load_subjects_from_file(resolve_path(args.subjects_file))

    if args.edf or args.label or args.subject_id:
        if not (args.edf and args.label and args.subject_id):
            raise ValueError("使用单个被试模式时，--subject-id、--edf、--label 三个参数必须都提供。")
        return [{"subject_id": args.subject_id, "edf": args.edf, "label": args.label}]

    return _DEFAULT_SUBJECTS


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
                suppress_eye_artifacts=args.suppress_eye_artifacts,
                eye_low_freq=args.eye_low_freq,
                eye_high_freq=args.eye_high_freq,
                eye_attenuation=args.eye_attenuation,
                artifact_ptp_uv=args.artifact_ptp_uv,
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
