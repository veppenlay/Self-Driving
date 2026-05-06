# 云端环境与训练说明

## 1. 克隆项目

```bash
git clone https://github.com/veppenlay/Self-Driving.git
cd Self-Driving
```

## 2. 安装环境

建议 Python 3.10 或 3.11。CUDA 版本以云端服务器为准。

CUDA 12.1 示例：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

CPU 或已预装 PyTorch 的环境：

```bash
python -m pip install -r requirements.txt
```

## 3. 上传或挂载数据

不要把大规模图片数据提交到 Git。建议把数据放在云盘、对象存储或服务器本地目录，然后用脚本生成 split：

```bash
python scripts/prepare_real_dataset.py \
  --src /mnt/data/raw_images \
  --dst-root dataset \
  --name temporal3_data \
  --label-shift 0
```

## 4. 训练主线模型

baseline：

```bash
bash configs/train_baseline_temporal3.sh
```

蒸馏 0.10：

```bash
bash configs/train_distill_temporal3_w010.sh
```

如果数据目录不是 `dataset/temporal3_data`，请直接修改脚本里的 `DATA_DIR`。

## 5. 查看结果

训练完成后重点看：

- `runs/<run_name>/training_summary.json`
- `runs/<run_name>/best_*.pth`
- TensorBoard 日志

```bash
tensorboard --logdir runs
```
