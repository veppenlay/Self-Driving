# ve 远端训练（AutoDL 5090）

与 VGGT-Long / Codex **旁路共存**：Cursor 工作区目录带 `_cursor` 后缀，勿与 Codex 目录混用。

## 布局

```text
/root/autodl-tmp/
  VGGT-Long/              # Codex 项目，勿改
  gpv16_3dgs_env/         # 既有 venv，勿用
  v-Net_cursor/           # Cursor：本项目（代码 + dataset + 远端标签）
  envs/vnet_cursor/       # Cursor：专用 conda 前缀
```

本机实验产出：`E:\桌面\v-Net\mp_cursor\`

## 各实验强制数据边界

```text
训练 / 验证：dataset/26合并（train_clean / val_clean）
独立测试：  dataset/25地下室（只测不调参）
```

远端标签：`locked_model/labels/remote/labels_2d.csv`（含上述 split）。  
正式对照只认地下室 RAW + 固定 HMM；禁止用 `26合并` 内部分割伪 test 冒充正式基线。

## 激活环境

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/vnet_cursor
cd /root/autodl-tmp/v-Net_cursor
```

## 切到有卡后：一键开训

```bash
bash /root/autodl-tmp/v-Net_cursor/scripts/ve_start_train.sh
```

默认：`max_epochs=150`，早停 `min_delta=0.005`、连续 10 epoch 无足够改进则停。

训练产物写到 `locked_model/runs/`，**不要覆盖** `locked_model/checkpoints/` 锁存权重。

## 辅助脚本（均在 `v-Net_cursor/scripts/`）

| 脚本 | 作用 |
|---|---|
| `ve_remap_labels.py` | 锁存 CSV 路径 → Linux |
| `ve_prepare_labels.sh` | 改写标签 + DataLoader 冒烟 |
| `ve_start_train.sh` | 有卡正式开训 |
| `ve_ablation_roi_chain.sh` | ROI 消融链 |

## 隔离约定

1. 只用 `conda activate /root/autodl-tmp/envs/vnet_cursor`。
2. 勿修改 `/root/autodl-tmp/VGGT-Long`。
3. Cursor 新增目录/环境一律用 `_cursor` 后缀，避免与 Codex 冲突。
