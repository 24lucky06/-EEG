# 32导联模型训练交接说明

把整个项目文件夹发给训练的人，至少要包含这些目录和文件：

- `data/`
- `offline_training/`
- `config/`
- `shared/`
- `requirements.txt`
- `run_training_pipeline.ps1`

训练前安装依赖：

```powershell
pip install -r requirements.txt
```

先测试 120 个 epoch 是否能跑通：

```powershell
.\run_training_pipeline.ps1 -Mode all32 -ContextMode causal -MaxEpochs 120
```

确认没问题后训练完整 32 导联模型：

```powershell
.\run_training_pipeline.ps1 -Mode all32 -ContextMode causal -MaxEpochs None
```

训练完成后，模型文件在：

```text
offline_training/saved_models/sleep_stage_all32_causal_global.joblib
```

结果表和报告在：

```text
offline_training/results/
```

注意：`all32` 模型主要用于离线研究和特征重要性分析，不能直接给当前双导联实时硬件使用。当前实时系统仍然使用：

```text
offline_training/saved_models/sleep_stage_dual2_causal_global.joblib
```
