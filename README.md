# EEG Sleep Project 重构版

这个版本把项目拆成两端：

```text
offline_training/   离线算法训练端
realtime_system/    硬件采集 + 实时预测 + UI 原型端
```

## 1. 离线训练端 offline_training

### 01_extract_features_LIGHT_FIR_vscode.py

作用：读取 EDF + TXT 标签，完成 FIR 滤波、降采样、30 秒切片和特征提取。

双导联示例：

```bash
cd offline_training
python 01_extract_features_LIGHT_FIR_vscode.py --subject-id day1 --edf data/day1.edf --label data/day1.txt --mode dual2 --max-epochs 120
```

32 导联示例：

```bash
cd offline_training
python 01_extract_features_LIGHT_FIR_vscode.py --subject-id day1 --edf data/day1.edf --label data/day1.txt --mode all32 --max-epochs 120
```

输出：

```text
features/day1_dual2_X.npy
features/day1_dual2_y.npy
features/day1_dual2_meta.json
```

### 02_train_compare_models_vscode.py

作用：训练 W / N1 / N2 / N3 / REM 五分类睡眠分期模型，并输出睡眠质量评分。

双导联实时候选模型：

```bash
cd offline_training
python 02_train_compare_models_vscode.py --mode dual2 --context-mode causal
```

输出重点：

```text
saved_models/sleep_stage_dual2_causal_global.joblib
results/model_results_summary.csv
results/feature_importance_ranking.csv
results/sleep_quality_from_labels_dual2.csv
results/sleep_quality_from_global_model_predictions_dual2_causal.csv
```

### 03_sleep_quality_score.py

作用：只基于已有 y 标签计算睡眠质量评分。

```bash
cd offline_training
python 03_sleep_quality_score.py --mode dual2
```

## 2. 实时系统端 realtime_system

当前提供的是可跑通流程的骨架：

```text
hardware_reader.py             硬件采集接口占位，目前用模拟 EEG
realtime_feature_extractor.py  实时特征提取，尽量对齐 01 的特征顺序
model_loader.py                加载 joblib 模型
realtime_predictor.py          实时分期预测
sleep_quality_realtime.py      动态睡眠质量评分
app.py                         命令行模拟运行入口
ui/                            前端 UI 原型
```

模拟运行：

```bash
python realtime_system/app.py --model offline_training/saved_models/sleep_stage_dual2_causal_global.joblib --demo-epochs 5
```

## 3. 重要原则

1. all32 模型用于离线研究、特征重要性分析和论文/比赛展示。
2. dual2 + causal 模型用于双导联硬件实时系统。
3. 32 导联模型不能直接给双导联硬件用，因为输入维度不同。
4. 实时特征提取必须和 01 的特征顺序、通道顺序、采样率、epoch 长度保持一致。
5. 睡眠质量评分当前是基于分期序列的规则评分，不是有 PSQI 标签的监督学习模型。
