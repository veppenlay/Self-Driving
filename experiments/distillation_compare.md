# Baseline vs Distillation 对比表

| 实验 | checkpoint | distill weight | teacher checkpoint | test_clean MAE | test_clean RMSE | data1 MAE | data1 RMSE | external MAE | external RMSE | latency ms | FPS | model size MiB | 结论 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline_3frame_temporal_v2 | `.\runs\baseline_3frame_temporal_v2\best_baseline_3frame_temporal_v2.pth` | 0 | 无 | 0.075347 | 0.221090 | 0.124825 | 0.330784 | 0.662554 | 1.075637 | 5.090412 | 196.447753 | 12.086415 | baseline reference |
| distill_temporal3_w005 | `.\runs\distill_temporal3_w005\best_distill_temporal3_w005.pth` | 0.05 | `.\runs\baseline_3frame_temporal_v2\best_baseline_3frame_temporal_v2.pth` | 1.131602 | 1.316391 | 1.070362 | 1.282559 | 1.177574 | 1.365807 | 4.987556 | 200.499022 | 12.082267 | 不推荐：clean/data1/external 全面退化，早停在低质量解 |
| distill_temporal3_w010 | `.\runs\distill_temporal3_w010\best_distill_temporal3_w010.pth` | 0.10 | `.\runs\baseline_3frame_temporal_v2\best_baseline_3frame_temporal_v2.pth` | 0.085254 | 0.235447 | 0.122480 | 0.307276 | 0.571072 | 0.860644 | 5.491676 | 182.093772 | 12.082267 | 推荐：external 最优，data1 优于 baseline，clean 小幅退化 |
| distill_temporal3_w020 | `.\runs\distill_temporal3_w020\best_distill_temporal3_w020.pth` | 0.20 | `.\runs\baseline_3frame_temporal_v2\best_baseline_3frame_temporal_v2.pth` | 0.090917 | 0.265218 | 0.105069 | 0.260835 | 0.576690 | 0.875420 | 5.183058 | 192.936275 | 12.082267 | 候选：data1 最优，external 优于 baseline，clean 退化较 w010 更大 |

说明：本地 `dataset/data_train` 没有 `test_clean.txt`，因此 `test_clean` 指标使用 `dataset/data_train/test.txt` 作为 clean test split。
