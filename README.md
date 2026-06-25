# EEG Sleep Project

项目分为两部分：

```text
offline_training/   离线特征提取、模型训练、结果评估
realtime_system/    硬件采集、桌面实时监测、实时特征分析
```

## 离线训练

提取特征示例：

```bash
cd offline_training
python 01_extract_features_LIGHT_FIR_vscode.py --subject-id day1 --edf ../data/day1.edf --label ../data/day1.txt --mode dual2 --max-epochs 120
```

训练双导联模型示例：

```bash
cd offline_training
python 02_train_compare_models_vscode.py --mode dual2 --context-mode causal
```

输出重点：

```text
offline_training/saved_models/sleep_stage_dual2_causal_global.joblib
offline_training/results/model_results_summary.csv
offline_training/results/feature_importance_ranking.csv
```

## 实时系统

桌面实时监测入口：

```bash
python realtime_system/desktop_monitor.py --source serial --serial-port COM5
```

模拟数据运行：

```bash
python realtime_system/desktop_monitor.py --source simulated
```

相关文件：

```text
desktop_monitor.py             桌面实时监测界面
hardware_reader.py             串口硬件读取与模拟数据读取
realtime_feature_extractor.py  30 秒 epoch 实时特征提取
model_loader.py                加载 joblib 模型
realtime_predictor.py          实时分期预测
sleep_quality_realtime.py      动态睡眠质量评分
app.py                         命令行模拟流程入口
```

## 一键训练

```powershell
.\run_training_pipeline.ps1 -Mode dual2 -ContextMode causal -MaxEpochs 120
```

训练完整数据：

```powershell
.\run_training_pipeline.ps1 -Mode dual2 -ContextMode causal -MaxEpochs None
```

## 注意

1. `all32` 模型用于离线研究和特征重要性分析。
2. `dual2 + causal` 模型才适合双导联硬件实时使用。
3. 实时特征提取必须和离线训练保持通道顺序、采样率、滤波范围和特征顺序一致。
4. 当前实时展示以桌面端 `desktop_monitor.py` 为准，网页端代码已移除。
