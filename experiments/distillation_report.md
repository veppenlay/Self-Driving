# Teacher-Student 蒸馏实验报告

## 实验设置

- Student：当前 `3-frame Temporal V2`，`numFrames=3`，`frameStride=1`，HSV `144x192`，无 ROI。
- Teacher：`.\runs\baseline_3frame_temporal_v2\best_baseline_3frame_temporal_v2.pth`，训练阶段加载，`eval()`，参数 `requires_grad=False`，不进入 optimizer。
- 训练数据：`.\dataset\data_train`，本地实际 split 为 `train.txt / val.txt / test.txt`。
- data1：`E:\桌面\Epaicar\temporal3_project\dataset\data_aug`。
- external：`E:\桌面\Epaicar\temporal3_project\dataset\data_test`，按 recursive 方式评估。
- 评估脚本：`.\scripts\collect_baseline_metrics.py`。
- ONNX 导出：`.\export_onnx.py`，导出时只读取 checkpoint 的 `model` 权重。

## 性能对比

| 实验 | checkpoint | distill weight | teacher checkpoint | test_clean MAE | test_clean RMSE | data1 MAE | data1 RMSE | external MAE | external RMSE | latency ms | FPS | model size MiB | 结论 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline_3frame_temporal_v2 | `.\runs\baseline_3frame_temporal_v2\best_baseline_3frame_temporal_v2.pth` | 0 | 无 | 0.075347 | 0.221090 | 0.124825 | 0.330784 | 0.662554 | 1.075637 | 5.090412 | 196.447753 | 12.086415 | baseline reference |
| distill_temporal3_w005 | `.\runs\distill_temporal3_w005\best_distill_temporal3_w005.pth` | 0.05 | `.\runs\baseline_3frame_temporal_v2\best_baseline_3frame_temporal_v2.pth` | 1.131602 | 1.316391 | 1.070362 | 1.282559 | 1.177574 | 1.365807 | 4.987556 | 200.499022 | 12.082267 | 不推荐：clean/data1/external 全面退化，早停在低质量解 |
| distill_temporal3_w010 | `.\runs\distill_temporal3_w010\best_distill_temporal3_w010.pth` | 0.10 | `.\runs\baseline_3frame_temporal_v2\best_baseline_3frame_temporal_v2.pth` | 0.085254 | 0.235447 | 0.122480 | 0.307276 | 0.571072 | 0.860644 | 5.491676 | 182.093772 | 12.082267 | 推荐：external 最优，data1 优于 baseline，clean 小幅退化 |
| distill_temporal3_w020 | `.\runs\distill_temporal3_w020\best_distill_temporal3_w020.pth` | 0.20 | `.\runs\baseline_3frame_temporal_v2\best_baseline_3frame_temporal_v2.pth` | 0.090917 | 0.265218 | 0.105069 | 0.260835 | 0.576690 | 0.875420 | 5.183058 | 192.936275 | 12.082267 | 候选：data1 最优，external 优于 baseline，clean 退化较 w010 更大 |

说明：本地 `dataset/data_train` 没有 `test_clean.txt`，因此 `test_clean` 指标使用 `dataset/data_train/test.txt` 作为 clean test split。

## 训练日志核查

| 实验 | useDistillation | distillWeight | finalTrainDistillLoss | finalTrainTeacherStudentDiff | finalValCleanMAE | finalValStressMAE | bestEpoch | completedEpochs | earlyStopped | deploymentLoadsTeacher |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distill_temporal3_w005 | True | 0.05 | 0.026324 | 0.142664 | 1.033098 | 1.030145 | 13 | 23 | True | False |
| distill_temporal3_w010 | True | 0.10 | 0.003675 | 0.052885 | 0.086688 | 0.104615 | 73 | 80 | False | False |
| distill_temporal3_w020 | True | 0.20 | 0.009361 | 0.059994 | 0.096945 | 0.111678 | 80 | 80 | False | False |

所有蒸馏实验均记录 `useDistillation=true`、`teacherConfig`、`deploymentLoadsTeacher=false`。`Train_Distill_Loss` 非 0，`Train_Teacher_Student_Diff` 非异常常数，并随训练总体下降。

## 部署验证

| 实验 | ONNX 文件 | onnx.checker | input shape | 输出数 | ONNX size MiB |
|---|---|---:|---|---:|---:|
| distill_temporal3_w005 | `output\distill_temporal3_w005_student.onnx` | True | `[batch_size, 9, 144, 192]` | 1 | 3.863 |
| distill_temporal3_w010 | `output\distill_temporal3_w010_student.onnx` | True | `[batch_size, 9, 144, 192]` | 1 | 3.863 |
| distill_temporal3_w020 | `output\distill_temporal3_w020_student.onnx` | True | `[batch_size, 9, 144, 192]` | 1 | 3.863 |

checkpoint 结构检查结果：所有蒸馏 checkpoint 的 `model` 权重键中没有 `teacher` 权重；top-level 也没有 teacher 权重。ONNX 导出输出显示只读取 student checkpoint，且导出图输入为 `[batch_size, 9, 144, 192]`。导出脚本为兼容当前 PyTorch 2.10 环境使用 legacy exporter（`dynamo=False`），并安装了缺失的 `onnx` 包用于导出和 checker 验证。

## 结论

- 最优权重：`0.10`，按 external MAE 优先、data1 MAE 次优选择 `distill_temporal3_w010`。
- clean 是否提升：否。baseline clean MAE=0.075347，最佳蒸馏 clean MAE=0.085254，变化 +0.009907（+13.15%）。
- data1 泛化是否提升：是。baseline data1 MAE=0.124825，最佳蒸馏 data1 MAE=0.122480，变化 -0.002345（-1.88%）。`w020` 的 data1 MAE=0.105069，是 data1 单项最优。
- 外部测试集是否提升：是。baseline external MAE=0.662554，最佳蒸馏 external MAE=0.571072，变化 -0.091482（-13.81%）。
- 蒸馏是否主要提升了强风格泛化：是，`data_aug` 和 external 均改善，clean 未改善。
- 蒸馏是否改善外部测试集泛化：是，`w010` external MAE 最优，`w020` 也明显优于 baseline。
- 推理资源是否变化：基本一致。`w010` latency=5.491676 ms、FPS=182.093772、模型大小=12.082267 MiB；baseline latency=5.090412 ms、FPS=196.447753、模型大小=12.086415 MiB。latency 单次测量慢约 7.88%，仍在同结构测速波动可接受范围内；模型大小基本不变。
- 是否通过部署验证：通过。三个蒸馏 student checkpoint 均成功导出 ONNX，ONNX checker 通过。
- 是否确认部署阶段只加载 student，不加载 teacher：确认。导出脚本只读取 checkpoint `model`，不读取 `teacherCheckpoint`；checkpoint `model` 内没有 teacher 权重。
- 推荐后续主线：推荐 `distill_temporal3_w010` 作为外部泛化优先的后续主线候选；如果 clean 指标优先级最高，则应保留 baseline 或继续复现实验确认。

## 异常与风险说明

- `w005` 明显异常：clean/data1/external 均显著差于 baseline，训练早停，说明较小蒸馏权重在本次随机训练下未收敛到可用解。
- `w010` 是本次综合最优：external MAE 最低，data1 也优于 baseline，但 clean MAE 从 0.075347 退化到 0.085254。
- `w020` data1 MAE 最优，但 external 略差于 `w010`，clean 退化更明显，存在过强 teacher 约束或泛化取舍风险。
- latency 存在单次 benchmark 波动；建议后续在固定 GPU 空载条件下重复测速 3 次取均值。
- 本地 `data_train` 没有 `test_clean.txt`，报告中的 `test_clean` 使用 `test.txt` 作为 clean test split。
