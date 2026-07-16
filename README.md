# v-Net：锁存的三帧转向模型

本仓库只保留已在独立 `25地下室` 测试集验证过的主线：

```text
原始图像 → 底部 70% ROI → HSV / 144×192 → Seq-CfC Temporal3 2D 网络
        → steering-only HMM 在线滤波 → 车辆转向命令
```

锁存权重的 RAW MAE 为 `0.204611`，HMM 在线滤波后为 `0.167248`（ve 远端 GPU 复核）。SoftTarget、TCN、Temporal5、旧 CNN、旧单帧、蒸馏和旧后处理试验均未能改善独立地下室测试，已从工作区移除。

## 目录

```text
locked_model/     # 标注、训练、评测、ONNX 导出与锁存权重
dataset/          # 采集图片；26合并为训练/验证，25地下室为独立测试
scripts/          # 合并采集数据并设置独立测试集
epaicar_deploy/   # 车端 TensorRT + 锁存 HMM 部署代码和 ONNX
docs/             # 当前方案与数据说明
```

详细的可复现命令、文件角色与指标见 [locked_model/README.md](locked_model/README.md)。

## 关键约束

- **各实验强制数据边界**：`dataset/26合并` = 训练/验证；`dataset/25地下室` = 独立测试（只测不调参）。
- `dataset/26合并/train_clean.txt`、`val_clean.txt` 是训练/验证边界。
- `dataset/26合并/test_clean.txt` 指向 `dataset/25地下室`，只用于最终测试，不能用于调参。
- 车端必须使用 `epaicar_deploy/steering_only_hmm_online_model.json`；该滤波器只读取网络的 `steering`，不使用 `e_y` 或测试真值。
- 实验细节与挑战枪约定见 [mp_cursor/网络侧鲁棒性实验指导.md](mp_cursor/网络侧鲁棒性实验指导.md)。
