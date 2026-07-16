# 锁存模型闭环

## 已锁存方案

| 项目 | 固定值 |
|---|---|
| 网络 | Seq-CfC Temporal3，输出 `[steering, e_y]` |
| 输入 | 连续 3 帧、HSV、`144×192`、保留图片底部 70% |
| 训练/验证 | **`dataset/26合并`（强制；各实验共用）** |
| 独立测试 | **`dataset/25地下室`（强制；只测不调参）** |
| 后处理 | steering-only HMM online，`e_y` 不参与部署控制 |
| RAW / HMM MAE | `0.204611` / `0.167248`（ve GPU 复核） |

## 保留文件

- `checkpoints/best_seq_cfc_temporal3_2d.pth`：唯一锁存权重。
- `labels/current/labels_2d.csv`：训练、验证和独立测试的二维标签；图片路径为绝对路径，请保持 `dataset/` 位置不变，或重新生成标签。
- `evaluation/baseline/`：锁存权重的 train/val/test 原始预测与指标。
- `epaicar_deploy/steering_only_hmm_online_model.json`：车端与离线评测共用的固定 HMM 参数。

## 复现

重新生成标签（会覆盖指定输出目录）：

```powershell
python locked_model/generate_2d_labels.py `
  --dataset-root dataset/26合并 `
  --test-root dataset/25地下室 `
  --output-dir locked_model/labels/current
```

评测锁存权重：

```powershell
python locked_model/evaluate.py `
  --ckpt locked_model/checkpoints/best_seq_cfc_temporal3_2d.pth `
  --label-csv locked_model/labels/current/labels_2d.csv `
  --split test --output-dir locked_model/evaluation/recheck
```

将已锁存 HMM 应用于预测：

```powershell
python locked_model/apply_current_hmm_postprocess.py `
  --model-json epaicar_deploy/steering_only_hmm_online_model.json `
  --input-pred-csv locked_model/evaluation/baseline/predictions_test.csv `
  --output-dir locked_model/evaluation/hmm_recheck
```

训练、导出和单图 ONNX 推理分别使用 `train.py`、`export_onnx.py` 和 `onnx_inference.py`。训练产物请写到 `locked_model/runs/`，不要覆盖 `checkpoints/` 的锁存权重。
