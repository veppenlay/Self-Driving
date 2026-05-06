# 云端实验工作流

本文档用于固定后续协作方式：GPT 给出实验方案，云端服务器负责训练，本地项目负责归档与结果分析。

## 1. 推荐流程

1. 提出实验问题

   明确这次实验要回答的问题，例如：

   - 某种风格增强是否提升 external domain MAE。
   - 蒸馏权重是否需要重新搜索。
   - 只增强 train 是否优于 train+val 同步增强。

2. 生成实验方案

   每个实验至少记录：

   - `experiment_id`：建议格式 `YYYYMMDD_short_goal_variant`。
   - baseline checkpoint。
   - 数据来源和 split。
   - 风格增强策略。
   - 训练命令和关键超参。
   - 主要评价集和主指标。
   - 预期风险。

3. 云端训练

   每个实验独立输出到：

   ```text
   runs/<experiment_id>/
   ```

   不同实验不要复用同一个 `output_dir`。

4. 打包训练产物

   训练结束后在云端执行：

   ```bash
   python scripts/package_run_artifacts.py \
     --run-dir runs/<experiment_id> \
     --output-dir artifacts/<experiment_id>
   ```

   如果同时有评估输出：

   ```bash
   python scripts/package_run_artifacts.py \
     --run-dir runs/<experiment_id> \
     --output-dir artifacts/<experiment_id> \
     --extra output/<experiment_id> \
     --extra experiments/<experiment_id>_metrics.json
   ```

5. 拉回本地分析

   优先拉回 `artifacts/<experiment_id>`，本地分析时把它放到：

   ```text
   experiments/cloud_runs/<experiment_id>/
   ```

   本地只需要复核指标、读日志、画对比表和总结结论，不需要重新下载完整训练数据。

## 2. 必须保存并拉回本地的产物

这些文件缺一项，后续分析和复现实验都会变弱。

| 类型 | 建议文件 | 用途 |
|---|---|---|
| 最佳权重 | `best_*.pth` | 后续评估、导出、继续训练和对照 |
| 训练摘要 | `training_summary.json` | 保存 best epoch、MAE、loss、preprocess、numFrames、distill 配置 |
| 实验命令 | `command.txt` 或 `run_command.sh` | 确认训练入口和参数 |
| 环境信息 | `env.txt`、`pip_freeze.txt`、`git_commit.txt` | 解释依赖差异和代码版本 |
| 数据摘要 | `dataset_summary.json` | 确认数据来源、样本数量和 split 策略 |
| 指标结果 | `*_metrics.json`、`*.csv`、`*.md` | 直接用于横向对比 |
| 日志 | `train.log`、`stdout.log`、`stderr.log` | 排查早停、学习率、异常和过拟合 |
| 打包清单 | `artifact_manifest.json` | 文件大小、hash、来源路径和打包时间 |

## 3. 建议保存但按需拉回的产物

| 类型 | 何时需要 |
|---|---|
| `last_*.pth` 或普通 `*.pth` | 需要排查 best 与最终 epoch 差异，或继续训练 |
| TensorBoard 事件文件 | 需要看曲线形态、过拟合过程、loss 是否震荡 |
| ONNX 文件 | 需要做部署验证或跨运行时评估 |
| 可视化图表 | 需要写报告或快速比较 |
| 预测明细 CSV | 需要做误差分桶、失败样本分析、角度段分析 |
| 少量失败样本图片 | 只在需要人工判断风格/场景问题时拉回 |

## 4. 不建议拉回本地的产物

这些通常很大，且不直接增加分析价值。

- 完整训练数据集图片。
- 完整增强后图片目录。
- 所有 TensorBoard 中间文件。
- Python 虚拟环境。
- `__pycache__`、临时缓存、下载缓存。
- 中间 epoch 的大量 checkpoint，除非当前实验就是研究训练过程。

## 5. 最小可分析包

如果云端空间或带宽很紧，最低限度拉回：

```text
artifacts/<experiment_id>/
  artifact_manifest.json
  training_summary.json
  best_*.pth
  *_metrics.json
  train.log
  command.txt
  git_commit.txt
  pip_freeze.txt
  dataset_summary.json
```

没有 `best_*.pth` 仍可做指标总结，但无法复评和导出；没有 `training_summary.json` 则很难确认训练配置。

## 6. 本项目当前主线

当前建议把以下两个模型作为固定对照：

- `checkpoints/baseline_3frame_temporal_v2/best_baseline_3frame_temporal_v2.pth`
- `checkpoints/distill_temporal3_w010/best_distill_temporal3_w010.pth`

后续域风格实验优先比较：

- clean test MAE/RMSE。
- data_aug 或目标域验证集 MAE/RMSE。
- external test MAE/RMSE。
- latency/FPS 是否明显退化。

如果 clean 小幅退化但 external 明显提升，应把结论写成“泛化优先候选”，不要直接覆盖 baseline。

## 7. 云端命令建议

训练时建议同时保存命令和环境：

```bash
mkdir -p runs/<experiment_id>
git rev-parse HEAD > runs/<experiment_id>/git_commit.txt
python -m pip freeze > runs/<experiment_id>/pip_freeze.txt
printf '%s\n' '<paste training command here>' > runs/<experiment_id>/command.txt
```

训练命令建议通过 `tee` 保存日志：

```bash
bash configs/train_baseline_temporal3.sh 2>&1 | tee runs/<experiment_id>/train.log
```

如果使用自定义命令，确保 `--output_dir runs/<experiment_id>` 与日志目录一致。
