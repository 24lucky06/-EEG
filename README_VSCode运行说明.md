# EEG 睡眠分期代码：VSCode 运行版

## 1. 文件放置

建议目录结构如下：

```text
vscode_eeg_sleep_project/
├─ 01_extract_features_LIGHT_FIR_vscode.py
├─ 02_train_compare_models_vscode.py
├─ requirements.txt
├─ subjects_template.json
├─ data/
│  ├─ day1.edf
│  └─ day1.txt
├─ features/
└─ results/
```

把你的 `.edf` 和 `.txt` 标签文件放进 `data` 文件夹。

---

## 2. 安装依赖

在 VSCode 终端输入：

```bash
pip install -r requirements.txt
```

如果你用的是 Anaconda，建议先新建环境：

```bash
conda create -n eeg_sleep python=3.10 -y
conda activate eeg_sleep
pip install -r requirements.txt
```

---

## 3. 运行第一个脚本：提取特征

先用双导联模式跑通：

```bash
python 01_extract_features_LIGHT_FIR_vscode.py --subject-id day1 --edf data/day1.edf --label data/day1.txt --mode dual2 --max-epochs 120
```

`dual2` 默认选用 `C3` 和 `O1`。如需自定义两个通道，例如改成 C4 和 O2：

```bash
python 01_extract_features_LIGHT_FIR_vscode.py --subject-id day1 --edf data/day1.edf --label data/day1.txt --mode dual2 --dual2-channels C4 O2 --max-epochs 120
```

如果你的 EDF 是 32 导或更多，再跑：

```bash
python 01_extract_features_LIGHT_FIR_vscode.py --subject-id day1 --edf data/day1.edf --label data/day1.txt --mode all32 --max-epochs 120
```

`--max-epochs 120` 表示只处理前 120 个 30 秒片段，也就是前 60 分钟。整晚运行可以改成：

```bash
python 01_extract_features_LIGHT_FIR_vscode.py --subject-id day1 --edf data/day1.edf --label data/day1.txt --mode dual2 --max-epochs None
```

---

## 4. 批量处理多个被试

复制 `subjects_template.json`，重命名为 `subjects.json`，然后修改里面的 EDF 和 TXT 文件名。

运行：

```bash
python 01_extract_features_LIGHT_FIR_vscode.py --subjects-file subjects.json --mode dual2 --max-epochs 120
```

---

## 5. 运行第二个脚本：训练模型

特征提取完成后，运行：

```bash
python 02_train_compare_models_vscode.py --mode dual2 --context-mode causal
```

如果你前面用的是 all32，这里也必须用 all32：

```bash
python 02_train_compare_models_vscode.py --mode all32 --context-mode causal
```

`--context-mode causal` 表示使用“前两段 + 当前段”，适合后续接实时硬件。

如果想复现原来离线代码的“上一段 + 当前段 + 下一段”，可以运行：

```bash
python 02_train_compare_models_vscode.py --mode dual2 --context-mode centered
```

---

## 6. 输出结果

第一个脚本输出在 `features/`：

```text
subject_id_channel_mode_X.npy
subject_id_channel_mode_y.npy
subject_id_channel_mode_meta.json
```

第二个脚本输出在 `results/`：

```text
model_results_summary.csv
feature_importance_ranking.csv
*_report.txt
saved_models/*.joblib
```

---

## 7. 常见问题

### 问题一：找不到 EDF/TXT 文件

把数据文件放到 `data` 文件夹，并确认命令里的文件名完全一致。

### 问题二：dual2 找不到指定的两个通道

说明你的 EDF 通道名可能不是 `C3` / `O1`。可以打开报错信息看实际通道名，然后用 `--dual2-channels` 指定实际存在的两个通道，例如 `--dual2-channels C4 O2`。

### 问题三：all32 找不到 32 个通道

先用 `--mode dual2` 跑通流程。all32 只适合多导 EEG 数据。

### 问题四：训练脚本找不到特征文件

检查 01 和 02 的 `--mode` 是否一致。比如 01 用 dual2，02 也必须用 dual2。
