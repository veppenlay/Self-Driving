# ve 远端训练（AutoDL 5090）

与 VGGT-Long **旁路共存**：项目与独立 Python 环境都在数据盘，互不覆盖。

## 布局

```text
/root/autodl-tmp/
  VGGT-Long/          # 既有项目，勿改
  gpv16_3dgs_env/     # 既有 venv，勿用
  v-Net/              # 本项目（代码 + dataset + 远端标签）
  envs/vnet/          # 本项目专用 conda 前缀
```

## 当前就绪状态（无卡已完成）

- 代码、`envs/vnet`、`dataset/26合并`、`dataset/25地下室` 均已就位
- 标签：`locked_model/labels/remote/labels_2d.csv`（由锁存 CSV 路径改写为 Linux，**标签值不变**）
  - split：train 2654 / val 569 / test 423
- DataLoader 冒烟通过（`READY_FOR_GPU_TRAIN`）
- 无卡 cgroup 内存约 **2GB**，勿在无卡模式跑全量 `generate_2d_labels.py`（易被 OOM kill）

## 切到有卡后：一键开训

```bash
bash /root/autodl-tmp/v-Net/scripts/ve_start_train.sh
```

等价手动命令：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/vnet
cd /root/autodl-tmp/v-Net

python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

python locked_model/train.py \
  --label-csv locked_model/labels/remote/labels_2d.csv \
  --output-dir locked_model/runs/ve_seq_cfc_temporal3_2d \
  --model-variant seq_cfc_temporal3 \
  --epochs 80 --batch-size 32 --num-workers 8 --device cuda
```

训练产物写到 `locked_model/runs/`，**不要覆盖** `locked_model/checkpoints/` 锁存权重。

建议用 `tmux`/`screen`，避免 SSH 断开中断训练。

## 辅助脚本

| 脚本 | 作用 |
|---|---|
| `scripts/ve_remap_labels.py` | 把锁存 CSV 路径改写为 Linux |
| `scripts/ve_prepare_labels.sh` | 改写标签 + DataLoader 冒烟 |
| `scripts/ve_smoke_dataloader.sh` | 仅冒烟 |
| `scripts/ve_start_train.sh` | 有卡后正式开训 |

## 评测（训练后可选）

```bash
python locked_model/evaluate.py \
  --ckpt locked_model/runs/ve_seq_cfc_temporal3_2d/best_seq_cfc_temporal3_2d.pth \
  --label-csv locked_model/labels/remote/labels_2d.csv \
  --split test --output-dir locked_model/evaluation/ve_recheck

python locked_model/apply_current_hmm_postprocess.py \
  --model-json epaicar_deploy/steering_only_hmm_online_model.json \
  --input-pred-csv locked_model/evaluation/ve_recheck/predictions_test.csv \
  --output-dir locked_model/evaluation/ve_hmm_recheck
```

## 隔离约定

1. 只用 `conda activate /root/autodl-tmp/envs/vnet`，勿向 `vggt-long` / `gpv16_3dgs_env` / base 装包。
2. 勿修改 `/root/autodl-tmp/VGGT-Long` 内任何文件。
3. 勿设置会指向 VGGT 的全局 `PYTHONPATH`。
4. 大文件与 conda 环境放在 `/root/autodl-tmp`，避免撑满 30G 系统盘。
