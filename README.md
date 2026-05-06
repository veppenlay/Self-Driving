# v-Net: 3-frame Temporal Steering Project

本仓库是从毕业设计工程中拆出的干净训练项目，后续主线聚焦数据域风格实验，而不是前后端展示或车端页面工程。

## 保留内容

- 3-frame Temporal V2 转向角回归模型代码。
- baseline 3-frame 最佳权重。
- distillation weight = 0.10 的 student 最佳权重。
- 数据整理、离线风格增强、合并数据集、评估与 ONNX 导出脚本。
- baseline 与蒸馏实验的关键指标记录。

## 不包含内容

- 前端/后端页面项目。
- 原始数据集图片与增强后数据集图片。
- TensorBoard 日志、完整训练输出、临时缓存和虚拟环境。
- CycleGAN 源码包本体。当前仓库保留的是域风格实验接口和离线增强入口；如果后续要接 CycleGAN 生成图像，建议把生成结果作为外部数据目录输入，不把 CycleGAN 工程合入本仓库。

## 目录结构

```text
v-Net/
  train.py                         # 训练入口，默认 temporal3
  models.py                        # checkpoint 加载与模型封装
  steering_models.py               # MobileNet temporal student/backbone
  datasets.py                      # 3-frame 时序数据集
  steering_preprocess.py           # HSV/resize/角度标签处理
  steering_augmentations.py        # 在线训练增强
  scripts/                         # 数据准备、风格增强、评估脚本
  checkpoints/                     # 当前保留的两个最佳权重
  experiments/                     # 关键实验指标和报告
  configs/                         # 云端训练命令模板
  docs/云端实验工作流_CN.md          # GPT 方案、云端训练、本地分析的交接规范
  dataset/                         # 本地/云端数据放置目录，不提交图片
  runs/                            # 训练输出目录，不提交
  output/                          # 评估/导出输出目录，不提交
```

## 快速开始

建议云端使用 Python 3.10 或 3.11。CUDA 版 PyTorch 请按服务器 CUDA 版本安装；下面以 CUDA 12.1 为例：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

如果云端镜像已经预装 PyTorch，可以直接执行：

```bash
python -m pip install -r requirements.txt
```

## 数据格式

训练图片文件名需要能解析转向角，推荐格式：

```text
000001_xxx_0.12.jpg
000002_xxx_0.10.jpg
000003_xxx_-0.05.jpg
```

最后一个下划线后的数值会作为转向角标签。3-frame 输入会按帧编号查找历史帧；序列开头缺少历史帧时，代码会用当前帧补齐，保证样本可训练。

## 准备数据

把原始图片目录整理为 `dataset/temporal3_data`：

```bash
python scripts/prepare_real_dataset.py \
  --src /path/to/raw_images \
  --dst-root dataset \
  --name temporal3_data \
  --label-shift 0
```

如需先做安全的离线风格增强：

```bash
python scripts/augment_dataset_offline.py \
  --src /path/to/raw_images \
  --dst dataset/data_aug \
  --copies-per-source-image 2 \
  --overwrite
```

如需把原始训练集、增强训练集和独立测试集合并成一个训练目录：

```bash
python scripts/build_merged_temporal_dataset.py \
  --base-root dataset/data_train \
  --aug-root dataset/data_aug \
  --test-root dataset/data_test \
  --output-root dataset/merged_temporal3
```

## 训练 baseline 3-frame

```bash
python train.py \
  --data_folder dataset/temporal3_data \
  --model_variant temporal3 \
  --num_frames 3 \
  --frame_stride 1 \
  --epochs 80 \
  --batch_size 16 \
  --output_dir runs/baseline_3frame_temporal_v2 \
  --save_name baseline_3frame_temporal_v2.pth \
  --best_save_name best_baseline_3frame_temporal_v2.pth
```

## 训练蒸馏模型 distill weight = 0.10

```bash
python train.py \
  --data_folder dataset/temporal3_data \
  --model_variant temporal3 \
  --num_frames 3 \
  --frame_stride 1 \
  --epochs 80 \
  --batch_size 16 \
  --use_distillation \
  --teacher_ckpt checkpoints/baseline_3frame_temporal_v2/best_baseline_3frame_temporal_v2.pth \
  --distill_weight 0.10 \
  --output_dir runs/distill_temporal3_w010 \
  --save_name distill_temporal3_w010.pth \
  --best_save_name best_distill_temporal3_w010.pth
```

蒸馏只在训练阶段加载 teacher，导出和部署阶段只读取 student checkpoint 中的 `model` 权重。

## 评估 checkpoint

```bash
python scripts/compare_regression_models.py \
  --model baseline=checkpoints/baseline_3frame_temporal_v2/best_baseline_3frame_temporal_v2.pth \
  --model distill_w010=checkpoints/distill_temporal3_w010/best_distill_temporal3_w010.pth \
  --flat-dataset test=dataset/temporal3_data \
  --output-dir output/eval
```

## 云端实验交接

后续推荐采用固定协作方式：GPT 给出实验方案，云端训练，本地拉回轻量产物做分析。详细规范见：

```text
docs/云端实验工作流_CN.md
```

云端训练完成后，建议打包每次实验的必要产物：

```bash
python scripts/package_run_artifacts.py \
  --run-dir runs/<experiment_id> \
  --output-dir artifacts/<experiment_id>
```

如果已经生成评估结果，可一起带回：

```bash
python scripts/package_run_artifacts.py \
  --run-dir runs/<experiment_id> \
  --output-dir artifacts/<experiment_id> \
  --extra output/<experiment_id> \
  --extra experiments/<experiment_id>_metrics.json
```

最低限度应拉回 `best_*.pth`、`training_summary.json`、评估指标、训练日志、命令记录、代码提交号、依赖冻结文件和 `artifact_manifest.json`。

## 导出 ONNX

```bash
python export_onnx.py \
  --ckpt checkpoints/distill_temporal3_w010/best_distill_temporal3_w010.pth \
  --out output/distill_temporal3_w010_student.onnx
```

## 当前保留的模型

| 模型 | 路径 | 说明 |
|---|---|---|
| baseline_3frame_temporal_v2 | `checkpoints/baseline_3frame_temporal_v2/best_baseline_3frame_temporal_v2.pth` | 3-frame Temporal V2 基线 |
| distill_temporal3_w010 | `checkpoints/distill_temporal3_w010/best_distill_temporal3_w010.pth` | 蒸馏权重 0.10 的 student 模型 |

关键指标见 `experiments/distillation_report.md`。
