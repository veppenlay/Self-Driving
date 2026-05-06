#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "runs" / "temporal3"
DEFAULT_CKPT = DEFAULT_RUN_DIR / "best_temporal3.pth"
DEFAULT_OUT = ROOT / "output" / "model_visualization" / "temporal3_mobilenet_v2"


def _load_net_module():
    module_path = ROOT / "models.py"
    spec = importlib.util.spec_from_file_location("autodrive_net_models", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shape(tensor: torch.Tensor) -> str:
    return "x".join(str(v) for v in tensor.shape)


def _params(module: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters())


def _load_model(checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model") or ckpt.get("emaState") or ckpt.get("state_dict") or ckpt
    module = _load_net_module()
    model = module.build_model_for_checkpoint(state_dict, ckpt.get("modelVariant") if isinstance(ckpt, dict) else None)
    model.eval()
    return model, ckpt


def trace_model(model: torch.nn.Module, ckpt: dict[str, Any]) -> list[dict[str, Any]]:
    preprocess = ckpt.get("preprocess", {}) if isinstance(ckpt, dict) else {}
    input_size = preprocess.get("inputSize", [144, 192])
    height, width = int(input_size[0]), int(input_size[1])
    num_frames = int(ckpt.get("numFrames", 3)) if isinstance(ckpt, dict) else 3
    channels = 3 * num_frames

    records: list[dict[str, Any]] = []
    with torch.no_grad():
        stacked = torch.zeros(1, channels, height, width)
        adapted = model._adapt_temporal_input(stacked)
        records.append(
            {
                "stage": "input",
                "module": "3-frame stacked HSV input",
                "operation": "concat [t-2, t-1, t] on channel axis",
                "input_shape": "-",
                "output_shape": _shape(stacked),
                "params": 0,
                "role": "三帧 HSV 图像按通道拼接，保留短时时序上下文。",
            }
        )
        records.append(
            {
                "stage": "temporal_adapter",
                "module": repr(model.temporal_adapter).replace("\n", " "),
                "operation": "grouped 1x1 conv + BN + Hardswish",
                "input_shape": _shape(stacked),
                "output_shape": _shape(adapted),
                "params": _params(model.temporal_adapter),
                "role": "把 9 通道时序输入压回 3 通道，使后续预训练 MobileNet stem 可复用。",
            }
        )

        x = adapted
        mid_map = None
        for idx, layer in enumerate(model.backbone):
            in_shape = _shape(x)
            x = layer(x)
            out_shape = _shape(x)
            role = "MobileNetV3-Small backbone feature extraction"
            if idx == 8:
                mid_map = x
                role = "中间高分辨率特征，保留更多赛道线空间位置。"
            if idx == len(model.backbone) - 1:
                role = "最终语义特征，提供更强的全局/抽象表征。"
            records.append(
                {
                    "stage": f"backbone.{idx}",
                    "module": layer.__class__.__name__,
                    "operation": str(layer).split("\n", 1)[0],
                    "input_shape": in_shape,
                    "output_shape": out_shape,
                    "params": _params(layer),
                    "role": role,
                }
            )
        final_map = x
        if mid_map is None:
            raise RuntimeError("failed to capture backbone.8 mid feature map")

        mid_reduce = model.mid_reduce(mid_map)
        records.append(
            {
                "stage": "mid_reduce",
                "module": "Conv1x1(48->48)+BN+Hardswish",
                "operation": "channel projection",
                "input_shape": _shape(mid_map),
                "output_shape": _shape(mid_reduce),
                "params": _params(model.mid_reduce),
                "role": "压缩/整理中层空间特征，保留车道线几何细节。",
            }
        )

        final_reduce = model.final_reduce(final_map)
        records.append(
            {
                "stage": "final_reduce",
                "module": "Conv1x1(576->80)+BN+Hardswish",
                "operation": "channel projection",
                "input_shape": _shape(final_map),
                "output_shape": _shape(final_reduce),
                "params": _params(model.final_reduce),
                "role": "降低最终语义特征通道数，减少融合计算量。",
            }
        )

        final_up = F.interpolate(final_reduce, size=mid_reduce.shape[-2:], mode="bilinear", align_corners=False)
        records.append(
            {
                "stage": "upsample",
                "module": "bilinear interpolate",
                "operation": "resize final feature to mid spatial size",
                "input_shape": _shape(final_reduce),
                "output_shape": _shape(final_up),
                "params": 0,
                "role": "把低分辨率语义特征对齐到中层空间网格，准备融合。",
            }
        )

        concat = torch.cat([mid_reduce, final_up], dim=1)
        records.append(
            {
                "stage": "concat",
                "module": "torch.cat([mid, final], dim=1)",
                "operation": "channel concat",
                "input_shape": f"{_shape(mid_reduce)} + {_shape(final_up)}",
                "output_shape": _shape(concat),
                "params": 0,
                "role": "合并空间几何信息和高层语义信息。",
            }
        )

        fused = model.fuse(concat)
        records.append(
            {
                "stage": "fuse",
                "module": "DepthwiseSeparableConv(128->112)",
                "operation": "DW 3x3 + PW 1x1",
                "input_shape": _shape(concat),
                "output_shape": _shape(fused),
                "params": _params(model.fuse),
                "role": "低成本融合多尺度特征，控制推理开销。",
            }
        )

        pooled = model.spatial_pool(fused)
        records.append(
            {
                "stage": "spatial_pool",
                "module": "SpatialAttentionPool(112)",
                "operation": "attention softmax over H*W + weighted sum",
                "input_shape": _shape(fused),
                "output_shape": _shape(pooled),
                "params": _params(model.spatial_pool),
                "role": "让模型学习关注赛道关键区域，而不是简单平均所有位置。",
            }
        )

        angle = model.reg_head(pooled)
        records.append(
            {
                "stage": "reg_head",
                "module": "MLP 112->96->24->1",
                "operation": "continuous steering regression",
                "input_shape": _shape(pooled),
                "output_shape": _shape(angle),
                "params": _params(model.reg_head),
                "role": "输出最终连续转向角，推理阶段只使用这个分支。",
            }
        )

        aux = model.aux_head(pooled)
        records.append(
            {
                "stage": "aux_head",
                "module": "MLP 112->96->num_angle_vocab",
                "operation": "training-only soft angle vocab prediction",
                "input_shape": _shape(pooled),
                "output_shape": _shape(aux),
                "params": _params(model.aux_head),
                "role": "训练期辅助约束角度词表连续性；推理阶段不参与输出。",
            }
        )
    return records


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stage", "module", "operation", "input_shape", "output_shape", "params", "role"],
        )
        writer.writeheader()
        writer.writerows(records)


def draw_architecture(records: list[dict[str, Any]], ckpt: dict[str, Any], output_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1800, 1280
    img = Image.new("RGB", (width, height), "#f5f7fb")
    draw = ImageDraw.Draw(img)

    def font(size: int, bold: bool = False):
        candidates = [
            "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
        for item in candidates:
            if Path(item).exists():
                return ImageFont.truetype(item, size)
        return ImageFont.load_default()

    title_font = font(42, True)
    subtitle_font = font(24)
    h_font = font(24, True)
    text_font = font(19)
    small_font = font(16)

    navy = "#18314f"
    teal = "#16867a"
    blue = "#2f6fa8"
    green = "#238b45"
    orange = "#d9871c"
    gray = "#6b7a90"
    border = "#d8e1ec"

    draw.rectangle([0, 0, width, 105], fill=navy)
    draw.text((60, 25), "3-frame Temporal V2 / MobileNet 回归网络可视化", font=title_font, fill="white")
    preprocess = ckpt.get("preprocess", {})
    sub = f"输入契约：3 帧 HSV，Resize {preprocess.get('inputSize', [144, 192])}，Full Image；输出：连续转向角"
    draw.text((60, 72), sub, font=subtitle_font, fill="#d7e4f2")

    def box(x, y, w, h, title, lines, color):
        draw.rounded_rectangle([x, y, x + w, y + h], radius=18, fill="white", outline=border, width=3)
        draw.rectangle([x, y, x + 9, y + h], fill=color)
        draw.text((x + 24, y + 18), title, font=h_font, fill=color)
        yy = y + 58
        for line in lines:
            draw.text((x + 24, yy), line, font=text_font, fill="#233143")
            yy += 30

    def arrow(x1, y1, x2, y2, color="#9aaabd"):
        draw.line([x1, y1, x2, y2], fill=color, width=4)
        draw.polygon([(x2, y2), (x2 - 14, y2 - 8), (x2 - 14, y2 + 8)], fill=color)

    box(70, 150, 300, 155, "输入", ["t-2 / t-1 / t", "HSV 每帧 3 通道", "拼接为 1x9x144x192"], blue)
    box(430, 150, 310, 155, "Temporal Adapter", ["Grouped 1x1 Conv", "9 通道压回 3 通道", "输出 1x3x144x192"], teal)
    box(800, 150, 340, 155, "MobileNetV3-Small", ["ImageNet 预训练骨干", "backbone[8] 取中层特征", "backbone[12] 取最终特征"], blue)
    box(1200, 150, 330, 155, "双尺度特征", ["mid: 1x48x9x12", "final: 1x576x5x6", "语义 + 空间几何"], green)
    arrow(370, 227, 430, 227)
    arrow(740, 227, 800, 227)
    arrow(1140, 227, 1200, 227)

    box(160, 395, 330, 170, "中层分支", ["mid_reduce", "1x1 Conv: 48 -> 48", "保留赛道线空间位置"], green)
    box(560, 395, 360, 170, "最终分支", ["final_reduce", "1x1 Conv: 576 -> 80", "上采样到 9x12"], green)
    box(990, 395, 370, 170, "融合模块", ["Concat: 48 + 80 = 128", "Depthwise Separable Conv", "输出 1x112x9x12"], teal)
    arrow(490, 480, 560, 480)
    arrow(920, 480, 990, 480)

    box(270, 675, 385, 165, "空间注意力池化", ["对 9*12 空间位置做 softmax", "学习关注关键赛道区域", "输出 112维特征"], orange)
    box(755, 675, 330, 165, "回归头", ["MLP: 112 -> 96 -> 24 -> 1", "输出连续 steering angle", "推理阶段唯一输出"], navy)
    box(1185, 675, 360, 165, "辅助头（训练期）", ["MLP: 112 -> 96 -> 11", "角度词表软标签约束", "推理阶段不使用"], gray)
    arrow(1360, 480, 1420, 480)
    draw.line([1420, 480, 1420, 757, 655, 757], fill="#9aaabd", width=4)
    draw.polygon([(655, 757), (669, 749), (669, 765)], fill="#9aaabd")
    arrow(655, 757, 755, 757)
    arrow(1085, 757, 1185, 757, "#c4a06b")

    draw.rounded_rectangle([70, 910, 1730, 1145], radius=20, fill="#eef8f5", outline="#b6ded7", width=2)
    draw.text((105, 940), "工作流核心解释", font=h_font, fill=teal)
    bullets = [
        "1. 三帧输入提供短时连续性，缓解“上一帧还是直行、下一帧突然急转”的单帧歧义。",
        "2. Temporal Adapter 只用极小代价把 9 通道时序信息压回 3 通道，继续复用预训练 MobileNet。",
        "3. 中层特征保留车道线几何位置，最终特征提供更强语义；融合后再用空间注意力池化。",
        "4. 推理只走连续角度回归头；辅助角度词表头只在训练时稳定表征，不增加部署输出复杂度。",
    ]
    yy = 980
    for item in bullets:
        draw.text((105, yy), item, font=text_font, fill="#233143")
        yy += 38

    total_params = sum(int(p.numel()) for p in ckpt.get("_model_ref").parameters()) if "_model_ref" in ckpt else None
    footer = "实际 checkpoint：mobilenet_v2_temporal3；teacherEnabled=false；bestEpoch=66；test_clean MAE=0.068950"
    draw.text((70, 1190), footer, font=small_font, fill=gray)
    if total_params is not None:
        draw.text((1340, 1190), f"参数量：{total_params:,}", font=small_font, fill=gray)

    img.save(output_path)


def write_markdown(records: list[dict[str, Any]], ckpt: dict[str, Any], out_path: Path, image_path: Path, csv_path: Path) -> None:
    preprocess = ckpt.get("preprocess", {})
    angle_vocab = ckpt.get("angleVocab", [])
    total_params = ckpt.get("_total_params")
    trainable_note = "推理阶段仅使用回归头输出连续转向角；辅助头只服务训练约束。"

    rows = []
    for item in records:
        rows.append(
            f"| `{item['stage']}` | `{item['input_shape']}` | `{item['output_shape']}` | `{int(item['params']):,}` | {item['role']} |"
        )

    text = f"""# 3-frame Temporal V2 网络逐层拆解与工作流说明

## 1. 名称与实际结构说明

当前项目中常说的 **3-frame MobileV2 / 3-frame Temporal V2**，这里的 `V2` 指的是项目里的第二版 MobileNet 回归主线，并不是 torchvision 的 `MobileNetV2` 网络。代码中的实际骨干为 **MobileNetV3-Small**，外层增加了轻量 3 帧时序适配器、多尺度特征融合和空间注意力池化。

- 模型变体：`{ckpt.get('modelVariant', 'unknown')}`
- 输入颜色空间：`{preprocess.get('colorSpace', 'hsv')}`
- 输入尺寸：`{preprocess.get('inputSize', [144, 192])}`
- ROI：`{preprocess.get('useRoi', False)}`
- 帧数：`{ckpt.get('numFrames', 3)}`
- 帧间隔：`{ckpt.get('frameStride', 1)}`
- Legacy CNN 教师蒸馏：`teacherEnabled=false`，当前最终 3 帧模型未使用 Legacy CNN 作为教师
- 参数量：`{total_params:,}` 参数

## 2. 总体结构图

![3-frame Temporal V2 architecture]({image_path.as_posix()})

## 3. 输入输出契约

模型输入不是单张 RGB 图，而是连续三帧图像经过同一预处理后拼接：

1. 读取 `t-2`、`t-1`、`t` 三帧。
2. 每帧转为 HSV。
3. 每帧 Resize 到 `144x192`。
4. 三帧按通道维拼接，形成 `[B, 9, 144, 192]`。
5. 模型输出 `[B, 1]`，表示连续转向角。

这种设计的意义是：在车辆高速行驶或弯道入口处，单帧图像可能无法判断接下来是否需要急转；三帧输入可以给模型提供短时运动趋势，降低单帧歧义。

## 4. 逐层 Shape Trace

完整表格已导出到：`{csv_path}`

| 阶段 | 输入尺寸 | 输出尺寸 | 参数量 | 作用 |
|------|----------|----------|--------|------|
{chr(10).join(rows)}

## 5. 模块级工作流拆解

### 5.1 三帧输入与 Temporal Adapter

三帧 HSV 图像拼接后是 `[B, 9, 144, 192]`。如果直接修改 MobileNet 第一层为 9 通道，会破坏预训练 stem 的输入分布。当前模型采用更轻量的做法：先用 `Temporal Adapter` 将 9 通道压回 3 通道，再送入预训练 MobileNet。

`Temporal Adapter` 的核心是 `groups=3` 的 `1x1 Conv`：

- H 通道只融合三帧中的 H；
- S 通道只融合三帧中的 S；
- V 通道只融合三帧中的 V；
- 不在适配层混合不同颜色语义，避免引入过强扰动。

因此它的计算量很小，但可以学习“当前帧”和“历史帧”之间的融合权重。

### 5.2 MobileNetV3-Small 预训练骨干

适配后的 `[B, 3, 144, 192]` 输入进入 MobileNetV3-Small 的 `features`。模型会保留两类特征：

- `backbone[8]` 的中层特征：分辨率更高，更适合表达赛道线位置、车道边界和局部几何。
- `backbone[12]` 的最终特征：语义更强，但空间分辨率更低。

这样做是为了避免纯全局平均池化过早抹掉赛道线的空间位置。

### 5.3 多尺度特征融合

中层特征经过 `mid_reduce: 48 -> 48`，最终特征经过 `final_reduce: 576 -> 80`，然后最终特征被上采样到中层特征的空间尺寸。两者拼接后得到 `128` 通道，再通过深度可分离卷积融合为 `112` 通道。

这个模块的设计目标是：

- 保留空间几何信息；
- 引入高层语义上下文；
- 用深度可分离卷积控制推理成本。

### 5.4 空间注意力池化

融合后的特征不是直接 `AdaptiveAvgPool2d(1)`，而是经过空间注意力池化。它会对 `H*W` 个空间位置生成权重，然后做加权求和。

这样可以让模型更关注赛道线、弯道趋势、关键边界等位置，而不是把背景区域和赛道区域平均对待。

### 5.5 回归头与辅助头

回归头为 `112 -> 96 -> 24 -> 1`，输出连续转向角，是部署时真正使用的输出。

辅助头为 `112 -> 96 -> 11`，对应当前数据中的真实角度词表：

`{angle_vocab}`

辅助头只在训练时使用，用来约束特征对角度分布的表达；推理时不使用，不增加实际部署输出复杂度。

## 6. 推理阶段完整流程

1. 从视频流或图像序列中取连续三帧。
2. 对每帧执行同样的 HSV + Resize 预处理。
3. 拼接成 `[1, 9, 144, 192]`。
4. Temporal Adapter 压缩为 `[1, 3, 144, 192]`。
5. MobileNetV3-Small 提取中层和最终特征。
6. 多尺度融合得到兼顾空间几何与语义的特征图。
7. 空间注意力池化得到 `[1, 112]` 的全局特征向量。
8. 回归头输出最终连续转向角。

{trainable_note}

## 7. 相比 Legacy CNN 的关键变化

| 维度 | Legacy CNN | 3-frame Temporal V2 |
|------|------------|---------------------|
| 输入 | 单帧图像 | 三帧时序输入 |
| 骨干 | 从零训练的小 CNN | ImageNet 预训练 MobileNetV3-Small |
| 空间信息 | 卷积后直接拉平/全连接 | 中层 + 最终特征融合 |
| 池化方式 | 无显式注意力 | 空间注意力加权池化 |
| 输出 | 单连续角度 | 单连续角度，训练期附加角度词表辅助头 |
| 部署成本 | 很低 | 小幅增加，但仍是轻量单模型推理 |
| 主要收益 | 同源数据拟合强 | 更好的风格鲁棒性和短时连续理解 |

## 8. 为什么这版结构适合当前任务

当前任务的核心不是识别任意开放道路，而是在固定赛道/近似固定轨迹下稳定输出转向角。真正困难点主要来自光照、显示风格、摄像头画面差异以及单帧急转弯前兆不足。因此当前结构的收益集中在三点：

- 预训练 MobileNet 提供更稳健的底层视觉表征；
- 三帧输入提供短时运动趋势，降低单帧标签跳变带来的不确定性；
- 多尺度融合和空间注意力尽量保留赛道线几何位置，提高对转向控制关键区域的敏感性。

"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize and trace the 3-frame Temporal V2 steering model.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    model, ckpt = _load_model(args.checkpoint)
    total_params = sum(int(p.numel()) for p in model.parameters())
    ckpt["_total_params"] = total_params
    ckpt["_model_ref"] = model

    records = trace_model(model, ckpt)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "temporal3_mobilenet_v2_layer_shapes.csv"
    png_path = out_dir / "temporal3_mobilenet_v2_architecture.png"
    md_path = out_dir / "temporal3_mobilenet_v2_workflow.md"
    json_path = out_dir / "temporal3_mobilenet_v2_layer_shapes.json"

    write_csv(records, csv_path)
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_architecture(records, ckpt, png_path)
    write_markdown(records, ckpt, md_path, png_path, csv_path)

    print(f"wrote {png_path}")
    print(f"wrote {md_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"total_params={total_params}")


if __name__ == "__main__":
    main()
