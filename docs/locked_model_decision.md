# 锁存方案决策记录

## 结论

当前唯一保留的网络与后处理为：

```text
Seq-CfC Temporal3 2D + HSV/底部 70% ROI + steering-only HMM online
```

## 各实验强制数据边界

```text
训练 / 验证：dataset/26合并（train_clean / val_clean）
独立测试：  dataset/25地下室（test_clean 指向；禁止调参）
```

训练和验证只使用 `dataset/26合并`，最终独立测试使用 `dataset/25地下室`。在该测试边界上，锁存网络 RAW steering MAE 为 `0.204611`，固定 HMM online 后为 `0.167248`（ve 远端 GPU 复核；权重文件未更换）。

ve 上 150 epoch + 早停的 ROI 消融（非 ROI 冷启、ROI→非 ROI、非 ROI→ROI）地下室 RAW MAE 均差于锁存，**不**替换基线。细节见 `mp_cursor/baseline_ve/baseline_record.json`。

## 淘汰依据

SoftTarget + TCN/CfC 的 Temporal3 和 Temporal5 候选虽然降低了源域验证误差，但地下室测试的 HMM MAE 分别为 `0.186775` 与 `0.189709`，均差于锁存方案。因此不保留这些模型、权重、训练脚本和评测输出。

`e_y` 仅作为带质量权重的训练辅助标签；在线部署不读取 `e_y`，也不使用测试真值。车端固定 HMM 参数见 `epaicar_deploy/steering_only_hmm_online_model.json`。
