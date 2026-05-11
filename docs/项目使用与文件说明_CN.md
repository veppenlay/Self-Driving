# 项目使用与文件说明

本文面向第一次接手本仓库的训练、评估和交接人员，说明项目目标、推荐运行流程、数据约定、常见命令，以及仓库中每个文件/目录的作用。

## 1. 项目定位

本项目是一个 3-frame temporal steering（3 帧时序转向角回归）训练工程，主线目标是基于连续图像帧预测转向角。当前仓库保留了：

- 训练入口、模型定义、数据集读取、预处理与增强代码。
- baseline 3-frame 模型与 distillation weight = 0.10 student 模型的最佳 checkpoint。
- 数据整理、离线增强、合并 split、模型评估、ONNX 导出和云端产物打包脚本。
- 已整理的关键实验指标和报告。

仓库不包含原始图片数据、增强后的大规模图片、TensorBoard 全量日志、虚拟环境、前后端页面工程或 CycleGAN 工程本体。

## 2. 环境安装

建议使用 Python 3.10 或 3.11。CUDA 版 PyTorch 请按服务器 CUDA 版本安装；下面以 CUDA 12.1 为例：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

如果云端镜像已经预装 PyTorch，可以只执行：

```bash
python -m pip install -r requirements.txt
```

Linux 服务器如运行 OpenCV/Albumentations 时提示 `libGL.so.1` 缺失，需要在系统层补装 OpenGL 运行库，或把 `opencv-python` 换成适合无显示环境的 `opencv-python-headless` 后再验证训练和评估脚本。

## 3. 数据格式约定

训练图片文件名必须能解析出转向角，推荐格式如下：

```text
000001_xxx_0.12.jpg
000002_xxx_0.10.jpg
000003_xxx_-0.05.jpg
```

规则：

- 文件名前缀的数字用于按时间顺序查找历史帧。
- 最后一个下划线后的数值作为转向角标签。
- 3-frame 输入会按 `num_frames` 和 `frame_stride` 查找历史帧；序列开头缺帧时，用当前帧补齐，保证样本仍可训练。
- 大规模图片数据默认放在 `dataset/` 或外部挂载目录，不应提交到 Git。

推荐生成后的数据目录：

```text
dataset/temporal3_data/
  train_clean.txt
  val_clean.txt
  test_clean.txt
  dataset_summary.json
  *.jpg
```

## 4. 推荐使用流程

### 4.1 准备原始数据

```bash
python scripts/prepare_real_dataset.py \
  --src /path/to/raw_images \
  --dst-root dataset \
  --name temporal3_data \
  --label-shift 0
```

该命令会复制图片、生成训练/验证/测试 split，并输出 `dataset_summary.json`。

### 4.2 可选：离线风格增强

```bash
python scripts/augment_dataset_offline.py \
  --src /path/to/raw_images \
  --dst dataset/data_aug \
  --copies-per-source-image 2 \
  --overwrite
```

如果已有 base/aug/test 三套数据，可合并为一个 temporal3 数据集：

```bash
python scripts/build_merged_temporal_dataset.py \
  --base-root dataset/data_train \
  --aug-root dataset/data_aug \
  --test-root dataset/data_test \
  --output-root dataset/merged_temporal3
```

### 4.3 训练 baseline

推荐直接使用配置脚本：

```bash
bash configs/train_baseline_temporal3.sh
```

等价的显式命令：

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

### 4.4 训练蒸馏 student

```bash
bash configs/train_distill_temporal3_w010.sh
```

等价的显式命令：

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

蒸馏 teacher 只在训练阶段加载；部署、导出和推理时使用 student checkpoint 中的 `model` 权重。

### 4.5 评估 checkpoint

```bash
python scripts/compare_regression_models.py \
  --model baseline=checkpoints/baseline_3frame_temporal_v2/best_baseline_3frame_temporal_v2.pth \
  --model distill_w010=checkpoints/distill_temporal3_w010/best_distill_temporal3_w010.pth \
  --flat-dataset test=dataset/temporal3_data \
  --output-dir output/eval
```

评估结果、图表和 Markdown 报告默认写入 `output/`，长期保留的结论建议整理到 `experiments/`。

### 4.6 导出 ONNX 与单图推理

导出：

```bash
python export_onnx.py \
  --ckpt checkpoints/distill_temporal3_w010/best_distill_temporal3_w010.pth \
  --out output/distill_temporal3_w010_student.onnx
```

ONNX 推理：

```bash
python onnx_inference.py \
  --onnx output/distill_temporal3_w010_student.onnx \
  --image dataset/temporal3_data/000003_xxx_0.10.jpg \
  --num-frames 3 \
  --frame-stride 1
```

### 4.7 云端训练产物打包

```bash
python scripts/package_run_artifacts.py \
  --run-dir runs/<experiment_id> \
  --output-dir artifacts/<experiment_id> \
  --extra output/<experiment_id> \
  --extra experiments/<experiment_id>_metrics.json
```

建议至少带回：`best_*.pth`、`training_summary.json`、评估指标、训练日志、命令记录、代码提交号、依赖冻结文件和 `artifact_manifest.json`。

## 5. 当前保留模型

| 模型 | 路径 | 用途 |
|---|---|---|
| baseline_3frame_temporal_v2 | `checkpoints/baseline_3frame_temporal_v2/best_baseline_3frame_temporal_v2.pth` | 3-frame Temporal V2 基线模型 |
| distill_temporal3_w010 | `checkpoints/distill_temporal3_w010/best_distill_temporal3_w010.pth` | distill weight = 0.10 的 student 模型 |

对应训练摘要位于各自 checkpoint 目录下的 `training_summary.json`，实验结论见 `experiments/distillation_report.md`。

## 6. 文件和目录作用总览

### 6.1 根目录代码文件

| 文件 | 作用 |
|---|---|
| `README.md` | 项目总览、快速开始、核心命令和保留内容说明。 |
| `requirements.txt` | Python 依赖清单。 |
| `.gitignore` | Git 忽略规则，排除本地环境、数据集、训练输出、导出模型和云端临时包，同时保留发布 checkpoint。 |
| `.gitattributes` | Git 属性配置，用于控制文本/二进制文件在仓库中的处理方式。 |
| `train.py` | 主训练入口，负责解析命令行/环境变量、构建数据集、模型、优化器、损失、EMA、蒸馏训练和 checkpoint 保存。 |
| `datasets.py` | `AutoDriveDataset` 数据集实现，读取 split 或图片目录，构建 3-frame temporal 输入并处理缺帧补齐。 |
| `steering_models.py` | MobileNetV2 temporal steering backbone/student 等模型结构定义。 |
| `models.py` | checkpoint 兼容加载、模型变体选择和历史模型封装。 |
| `steering_preprocess.py` | 图像读取、ROI、颜色空间转换、resize、CHW tensor 转换、角度词表/软标签编码和预处理配置序列化。 |
| `augmentations.py` | 训练/验证/压力测试增强流水线的兼容导出层。 |
| `steering_augmentations.py` | Albumentations 在线增强实现，包括颜色、噪声、压缩、局部曝光和时序帧同步增强。 |
| `sampler_utils.py` | 按转向角分桶计算样本权重并构建 weighted sampler。 |
| `utils.py` | 训练指标累计工具 `AverageMeter`。 |
| `debug_utils.py` | 调试/诊断辅助工具。 |
| `benchmark_utils.py` | 离线评估通用工具，包括模型加载、样本收集、指标计算、参数量/文件大小统计和单样本延迟测试。 |
| `export_onnx.py` | 根据 checkpoint 中保存的预处理元数据导出 ONNX。 |
| `onnx_inference.py` | 加载 ONNX 模型，对单张图片自动组装 temporal frame stack 并输出预测转向角。 |

### 6.2 `configs/`

| 文件 | 作用 |
|---|---|
| `configs/baseline_temporal3.env` | baseline temporal3 训练参数模板，便于云端记录配置。 |
| `configs/distill_temporal3_w010.env` | distill weight = 0.10 student 训练参数模板。 |
| `configs/train_baseline_temporal3.sh` | baseline temporal3 一键训练脚本。 |
| `configs/train_distill_temporal3_w010.sh` | distill student 一键训练脚本，会默认使用 baseline 最佳 checkpoint 作为 teacher。 |

### 6.3 `scripts/`

| 文件 | 作用 |
|---|---|
| `scripts/prepare_real_dataset.py` | 从原始图片目录复制并划分 train/val/test split，可用 `--label-shift` 生成滞后/超前标签。 |
| `scripts/prepare_formal_view_run_dataset.py` | 按 run 目录组织正式视角数据集，支持 train/val/test/val_style 多 split，可选择复制图片或引用原路径。 |
| `scripts/augment_dataset_offline.py` | 对数据集执行较安全的离线增强，保留转向角标签并生成增强图片/split。 |
| `scripts/augment_style_strong.py` | 生成更强的风格扰动图片，用于鲁棒性或域风格实验。 |
| `scripts/build_merged_temporal_dataset.py` | 合并 base、增强和独立 test 数据，生成统一 split 目录。 |
| `scripts/compare_regression_models.py` | 对多个 checkpoint 在 flat/recursive 数据集上做回归评估，输出指标、CSV、图表和报告。 |
| `scripts/compare_temporal_ablations.py` | 面向 temporal ablation 的多模型多数据集对比，统一输出 JSON/Markdown 结论。 |
| `scripts/collect_baseline_metrics.py` | 冻结 baseline 指标、参数量、文件大小和延迟基准，便于后续实验横向对比。 |
| `scripts/visualize_temporal3_model.py` | 追踪 temporal3 模型结构，输出层级记录、CSV、架构图和 Markdown 说明。 |
| `scripts/package_run_artifacts.py` | 打包云端训练产物，复制 checkpoint、summary、日志、额外文件并可生成 zip。 |

### 6.4 数据、输出和实验目录

| 路径 | 作用 |
|---|---|
| `dataset/README.md` | 数据目录说明；真实图片和 split 默认不提交。 |
| `runs/README.md` | 训练输出目录说明；运行产生的 checkpoint、summary 和 TensorBoard 日志默认不提交。 |
| `output/README.md` | 评估结果、图表和 ONNX 导出目录说明；生成物默认不提交。 |
| `artifacts/README.md` | 云端训练后轻量实验包的本地存放说明；生成物默认不提交。 |
| `checkpoints/baseline_3frame_temporal_v2/best_baseline_3frame_temporal_v2.pth` | baseline 发布权重文件，可直接用于评估、蒸馏 teacher 或 ONNX 导出。 |
| `checkpoints/baseline_3frame_temporal_v2/training_summary.json` | baseline 训练摘要，记录训练配置和关键结果。 |
| `checkpoints/distill_temporal3_w010/best_distill_temporal3_w010.pth` | distillation student 发布权重文件，可直接用于评估或部署导出。 |
| `checkpoints/distill_temporal3_w010/training_summary.json` | distillation student 训练摘要，记录训练配置和关键结果。 |
| `experiments/` | 已整理的实验指标、对比表和报告，适合提交到 Git 长期保存。 |
| `experiments/cloud_runs/README.md` | 临时放置云端拉回实验包的说明，具体实验包默认不提交。 |
| `docs/云端实验工作流_CN.md` | GPT 方案、云端训练、本地分析的协作交接规范。 |
| `docs/使用说明_CN.md` | 简版云端环境与训练说明。 |
| `docs/项目使用与文件说明_CN.md` | 本文档，提供更完整的使用流程和文件作用说明。 |

### 6.5 `experiments/` 已提交结果文件

| 文件 | 作用 |
|---|---|
| `experiments/baseline_3frame_temporal_v2.json` | baseline 实验配置/摘要记录。 |
| `experiments/baseline_3frame_temporal_v2_metrics.json` | baseline 关键评估指标。 |
| `experiments/distill_temporal3_w010_metrics.json` | distill student 关键评估指标。 |
| `experiments/distillation_compare.csv` | baseline 与 distill student 对比表。 |
| `experiments/distillation_compare.md` | baseline 与 distill student 的 Markdown 对比说明。 |
| `experiments/distillation_report.md` | 蒸馏实验报告和结论。 |

## 7. 审查建议

- 训练、数据准备和导出链路基本完整，适合继续做云端实验。
- 大文件管控合理：数据、运行输出、导出物、云端包默认被 `.gitignore` 排除，发布 checkpoint 被显式保留。
- 建议后续固定记录每次实验的命令、Git commit、依赖冻结文件和数据版本，避免云端/本地结果不可复现。
- 如果评估脚本在无显示 Linux 环境报 `libGL.so.1`，优先修复系统依赖或使用 headless OpenCV，再跑完整评估。
