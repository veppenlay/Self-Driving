#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from augmentations import AugConfig, build_eval_transforms, build_stress_transforms, build_train_transform_bundle
from datasets import AutoDriveDataset
from models import AutoDriveLegacyNet, AutoDriveNet, AutoDriveNetTemporal, AutoDriveNetV1, build_model_for_checkpoint
from sampler_utils import SamplerConfig, build_weighted_sampler, compute_sample_weights
from steering_preprocess import (
    PreprocessConfig,
    build_angle_vocab,
    preprocess_config_from_dict,
    preprocess_config_to_dict,
    soft_encode_angles_to_vocab,
)
from utils import AverageMeter


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float_tuple(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    value = os.getenv(name)
    if not value:
        return default
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if len(parts) != len(default):
        return default
    return tuple(float(item) for item in parts)


def _arg_or_env(args: argparse.Namespace, arg_name: str, env_name: str, default, cast):
    value = getattr(args, arg_name, None)
    if value is not None:
        return cast(value)
    env_value = os.getenv(env_name)
    if env_value in (None, ""):
        return default
    return cast(env_value)


def _arg_or_env_str(args: argparse.Namespace, arg_name: str, env_name: str, default: str) -> str:
    value = getattr(args, arg_name, None)
    if value not in (None, ""):
        return str(value)
    env_value = os.getenv(env_name)
    if env_value in (None, ""):
        return default
    return str(env_value)


def _arg_or_env_bool(args: argparse.Namespace, arg_name: str, env_name: str, default: bool) -> bool:
    value = getattr(args, arg_name, None)
    if value is not None:
        return bool(value)
    return _env_bool(env_name, default)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train steering regression models with optional temporal inputs.")
    parser.add_argument("--data_folder", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--use_distillation", action="store_true", default=None)
    parser.add_argument("--teacher_ckpt", default=None)
    parser.add_argument("--distill_weight", type=float, default=None)
    parser.add_argument("--model_variant", default=None)
    parser.add_argument("--num_frames", "--num-frames", dest="num_frames", type=int, default=None)
    parser.add_argument("--frame_stride", "--frame-stride", dest="frame_stride", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--save_name", default=None)
    parser.add_argument("--best_save_name", default=None)
    parser.add_argument("--log_dir", default=None)
    return parser.parse_args()


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _count_trainable_params(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.ema_model = copy.deepcopy(_unwrap(model)).eval()
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        src = _unwrap(model)
        for ema_value, value in zip(self.ema_model.state_dict().values(), src.state_dict().values()):
            if not torch.is_floating_point(ema_value):
                ema_value.copy_(value)
            else:
                ema_value.mul_(self.decay).add_(value, alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu() for key, value in self.ema_model.state_dict().items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.ema_model.load_state_dict(state_dict, strict=True)


class SoftTargetCrossEntropy(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        return -(target * log_probs).sum(dim=1).mean()


def _configure_optimizer(base_model: nn.Module, *, lr: float, weight_decay: float, backbone_lr_factor: float, freeze_backbone: bool):
    if not hasattr(base_model, "set_backbone_trainable"):
        optimizer = torch.optim.AdamW(base_model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(2, int(os.getenv("VENET_LR_PATIENCE", "4"))),
            min_lr=1e-6,
        )
        return optimizer, scheduler

    base_model.set_backbone_trainable(not freeze_backbone)
    if freeze_backbone:
        optimizer = torch.optim.AdamW(base_model.head_parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": list(base_model.head_parameters()), "lr": lr},
                {"params": list(base_model.backbone_parameters()), "lr": lr * backbone_lr_factor},
            ],
            weight_decay=weight_decay,
        )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, int(os.getenv("VENET_LR_PATIENCE", "4"))),
        min_lr=1e-6,
    )
    return optimizer, scheduler


def _extract_batch(batch):
    if len(batch) == 3:
        imgs, labels, meta = batch
    else:
        imgs, labels = batch
        meta = {}
    return imgs, labels, meta


def _forward_model(model: nn.Module, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    try:
        pred, aux_logits = model(imgs, return_aux=True)
    except TypeError:
        pred = model(imgs)
        aux_logits = None
    return pred, aux_logits


def _forward_teacher(model: nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    pred = model(imgs)
    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    return pred


def _load_teacher_model(
    teacher_ckpt: Path,
    device: torch.device,
    *,
    expected_preprocess: PreprocessConfig,
    expected_num_frames: int,
    expected_frame_stride: int,
) -> tuple[nn.Module, dict[str, Any]]:
    if not teacher_ckpt.is_file():
        raise FileNotFoundError(f"distillation teacher checkpoint not found: {teacher_ckpt}")

    ckpt = torch.load(str(teacher_ckpt), map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError(
            "Distillation teacher checkpoint must include metadata fields "
            "'model', 'preprocess', 'numFrames' and 'frameStride' so input compatibility can be verified."
        )

    missing_metadata = [key for key in ("model", "preprocess", "numFrames", "frameStride") if key not in ckpt]
    if missing_metadata:
        raise ValueError(f"distillation teacher checkpoint is missing metadata: {missing_metadata}")

    state = ckpt["model"]
    teacher_model = build_model_for_checkpoint(state, ckpt.get("modelVariant")).to(device)
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)

    teacher_preprocess = preprocess_config_from_dict(ckpt.get("preprocess"))
    teacher_num_frames = int(ckpt.get("numFrames", getattr(teacher_model, "num_frames", 1)))
    teacher_frame_stride = int(ckpt.get("frameStride", 1))

    mismatches: list[str] = []
    if teacher_preprocess != expected_preprocess:
        mismatches.append(
            f"preprocess teacher={preprocess_config_to_dict(teacher_preprocess)} "
            f"student={preprocess_config_to_dict(expected_preprocess)}"
        )
    if teacher_num_frames != int(expected_num_frames):
        mismatches.append(f"numFrames teacher={teacher_num_frames} student={expected_num_frames}")
    if teacher_frame_stride != int(expected_frame_stride):
        mismatches.append(f"frameStride teacher={teacher_frame_stride} student={expected_frame_stride}")
    if mismatches:
        raise ValueError(
            "Refusing to run distillation with teacher/student input mismatch. "
            + "; ".join(mismatches)
        )

    teacher_config = {
        "checkpoint": str(teacher_ckpt),
        "modelVariant": ckpt.get("modelVariant"),
        "preprocess": preprocess_config_to_dict(teacher_preprocess),
        "numFrames": teacher_num_frames,
        "frameStride": teacher_frame_stride,
        "trainableParams": _count_trainable_params(teacher_model),
    }
    return teacher_model, teacher_config


def _run_epoch(
    *,
    model: nn.Module,
    loader,
    device: torch.device,
    reg_criterion: nn.Module,
    aux_criterion: nn.Module,
    angle_vocab: list[float],
    aux_cls_weight: float,
    soft_label_temperature: float,
    soft_label_neighbors: int,
    teacher_model: nn.Module | None = None,
    distill_weight: float = 0.0,
    optimizer=None,
    grad_clip: float = 0.0,
    ema: ModelEMA | None = None,
):
    is_train = optimizer is not None
    model.train(is_train)
    if teacher_model is not None:
        teacher_model.eval()

    total_meter = AverageMeter("TotalLoss")
    reg_meter = AverageMeter("RegLoss")
    aux_meter = AverageMeter("AuxLoss")
    distill_meter = AverageMeter("DistillLoss")
    teacher_student_diff_meter = AverageMeter("TeacherStudentDiff")
    mae_meter = AverageMeter("MAE")

    for batch in loader:
        imgs, labels, meta = _extract_batch(batch)
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        soft_aux_targets = soft_encode_angles_to_vocab(
            labels.view(-1),
            angle_vocab,
            device=device,
            temperature=soft_label_temperature,
            neighbor_count=soft_label_neighbors,
        )

        with torch.set_grad_enabled(is_train):
            angle_pred, aux_logits = _forward_model(model, imgs)
            reg_loss = reg_criterion(angle_pred, labels)
            aux_loss = aux_criterion(aux_logits, soft_aux_targets) if aux_logits is not None else reg_loss.new_tensor(0.0)
            distill_loss = reg_loss.new_tensor(0.0)
            teacher_student_diff = reg_loss.new_tensor(0.0)

            if teacher_model is not None and distill_weight > 0:
                with torch.no_grad():
                    teacher_pred = _forward_teacher(teacher_model, imgs)
                if teacher_pred.shape != angle_pred.shape:
                    raise ValueError(
                        f"teacher/student prediction shape mismatch: teacher={tuple(teacher_pred.shape)} "
                        f"student={tuple(angle_pred.shape)}"
                    )
                teacher_pred = teacher_pred.detach()
                distill_loss = reg_criterion(angle_pred, teacher_pred)
                teacher_student_diff = (angle_pred.detach() - teacher_pred).abs().mean()

            total_loss = reg_loss + aux_cls_weight * aux_loss + distill_weight * distill_loss
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                if ema is not None:
                    ema.update(model)

        mae = (angle_pred - labels).abs().mean().item()
        batch_size = imgs.size(0)
        total_meter.update(total_loss.item(), batch_size)
        reg_meter.update(reg_loss.item(), batch_size)
        aux_meter.update(aux_loss.item(), batch_size)
        distill_meter.update(distill_loss.item(), batch_size)
        teacher_student_diff_meter.update(teacher_student_diff.item(), batch_size)
        mae_meter.update(mae, batch_size)

    return {
        "total_loss": total_meter.avg,
        "reg_loss": reg_meter.avg,
        "aux_loss": aux_meter.avg,
        "distill_loss": distill_meter.avg,
        "teacher_student_diff": teacher_student_diff_meter.avg,
        "mae": mae_meter.avg,
    }


def _write_summary(summary_path: Path, summary: dict):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_dataset_summary(data_folder: str) -> dict[str, Any]:
    summary_path = Path(data_folder) / "dataset_summary.json"
    if not summary_path.is_file():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_aug_config(*, num_frames: int = 1) -> AugConfig:
    preprocess = PreprocessConfig(
        color_space=os.getenv("VENET_PREPROCESS_COLOR_SPACE", "hsv").strip().lower(),
        input_size=(
            int(os.getenv("VENET_INPUT_HEIGHT", "144")),
            int(os.getenv("VENET_INPUT_WIDTH", "192")),
        ),
        use_roi=False,
    )
    return AugConfig(
        preprocess=preprocess,
        style_mix_ratio=_env_float_tuple("VENET_STYLE_MIX_RATIO", (0.6, 0.25, 0.15)),
        num_frames=num_frames,
    )


def _build_sampler_config(train_size: int) -> SamplerConfig:
    return SamplerConfig(
        bin_mode=os.getenv("VENET_SAMPLER_BIN_MODE", "uniform").strip().lower(),
        num_bins=int(os.getenv("VENET_SAMPLER_NUM_BINS", "9")),
        use_abs_angle=_env_bool("VENET_SAMPLER_USE_ABS_ANGLE", True),
        smoothing=float(os.getenv("VENET_SAMPLER_SMOOTHING", "1.0")),
        max_weight=float(os.getenv("VENET_SAMPLER_MAX_WEIGHT", "8.0")),
        downweight_straight=_env_bool("VENET_SAMPLER_DOWNWEIGHT_STRAIGHT", True),
        straight_threshold=float(os.getenv("VENET_SAMPLER_STRAIGHT_THRESHOLD", "0.08")),
        straight_weight_scale=float(os.getenv("VENET_SAMPLER_STRAIGHT_WEIGHT_SCALE", "0.45")),
        replacement=_env_bool("VENET_SAMPLER_REPLACEMENT", True),
        num_samples=int(os.getenv("VENET_SAMPLER_NUM_SAMPLES", str(train_size))),
    )


def _resolve_num_frames(model_variant: str, explicit_num_frames: int | None = None) -> int:
    if explicit_num_frames is not None:
        return max(1, int(explicit_num_frames))
    explicit = os.getenv("VENET_NUM_FRAMES")
    if explicit:
        return max(1, int(explicit))
    match = re.search(r"temporal(\d+)", model_variant)
    if match:
        return max(2, int(match.group(1)))
    if "temporal" in model_variant:
        return 3
    return 1


def _print_temporal_dataset_summary(label: str, dataset: AutoDriveDataset) -> None:
    summary = getattr(dataset, "temporal_summary", None)
    if not isinstance(summary, dict):
        return
    print(
        f"TemporalDataset split={label} num_frames={summary['numFrames']} frame_stride={summary['frameStride']} "
        f"padding_mode={summary['paddingMode']} remapped_sample_paths={summary['remappedSamplePaths']} "
        f"samples_with_missing_history={summary['samplesWithMissingHistory']} "
        f"total_missing_history_frames={summary['totalMissingHistoryFrames']} "
        f"max_missing_history_frames_per_sample={summary['maxMissingHistoryFramesPerSample']}"
    )
    if summary["numFrames"] > 1:
        print(f"TemporalDataset split={label} padding_description={summary['paddingDescription']}")


def _load_training_checkpoint(
    *,
    checkpoint_path: Path,
    model: nn.Module,
    ema: ModelEMA,
    device: torch.device,
    requested_num_frames: int,
) -> int:
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    ckpt_num_frames = int(ckpt.get("numFrames", getattr(_unwrap(model), "num_frames", 1))) if isinstance(ckpt, dict) else int(getattr(_unwrap(model), "num_frames", 1))
    if requested_num_frames != ckpt_num_frames:
        raise ValueError(
            f"Refusing to resume from checkpoint with mismatched numFrames: requested={requested_num_frames}, "
            f"checkpoint={ckpt_num_frames}. Create a fresh 2-frame/3-frame branch or use a matching checkpoint."
        )
    start_epoch = int(ckpt.get("epoch", 0)) + 1 if isinstance(ckpt, dict) else 1
    raw_state = ckpt.get("modelRaw", ckpt.get("model", {})) if isinstance(ckpt, dict) else ckpt
    _unwrap(model).load_state_dict(raw_state, strict=True)
    ema_state = ckpt.get("emaState", ckpt.get("model", {})) if isinstance(ckpt, dict) else raw_state
    ema.load_state_dict(ema_state)
    print(f"ResumeCheckpoint path={checkpoint_path} num_frames={ckpt_num_frames} start_epoch={start_epoch}")
    return start_epoch


def _make_checkpoint_payload(
    *,
    epoch: int,
    best_epoch: int,
    best_metric: float,
    angle_vocab: list[float],
    preprocess: PreprocessConfig,
    model_variant: str,
    base_model: nn.Module,
    ema: ModelEMA,
    num_frames: int,
    frame_stride: int,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model": ema.state_dict(),
        "modelRaw": {key: value.detach().cpu() for key, value in base_model.state_dict().items()},
        "emaState": ema.state_dict(),
        "bestEpoch": best_epoch,
        "bestSelectionMetric": best_metric,
        "angleVocab": angle_vocab,
        "modelVariant": model_variant,
        "preprocess": preprocess_config_to_dict(preprocess),
        "pretrainedLoaded": bool(getattr(base_model, "pretrained_loaded", False)),
        "numFrames": int(num_frames),
        "frameStride": int(frame_stride),
    }


def main():
    args = _parse_args()

    data_folder = _arg_or_env_str(args, "data_folder", "VENET_DATA_FOLDER", "dataset/temporal3_data")
    dataset_summary = _load_dataset_summary(data_folder)
    checkpoint = _arg_or_env_str(args, "checkpoint", "VENET_CHECKPOINT", "") or None
    teacher_ckpt = _arg_or_env_str(args, "teacher_ckpt", "VENET_TEACHER_CKPT", "") or None
    batch_size = _arg_or_env(args, "batch_size", "VENET_BATCH_SIZE", 16, int)
    start_epoch = 1
    epochs = _arg_or_env(args, "epochs", "VENET_EPOCHS", 80, int)
    lr = float(os.getenv("VENET_LR", "1e-4"))
    weight_decay = float(os.getenv("VENET_WEIGHT_DECAY", "1e-4"))
    grad_clip = float(os.getenv("VENET_GRAD_CLIP", "3.0"))
    early_stop_patience = int(os.getenv("VENET_EARLY_STOP_PATIENCE", "10"))
    early_stop_min_delta = float(os.getenv("VENET_EARLY_STOP_MIN_DELTA", "1e-4"))
    enable_train_aug = not _env_bool("VENET_DISABLE_TRAIN_AUG", False)
    use_weighted_sampler = _env_bool("VENET_USE_WEIGHTED_SAMPLER", True)
    use_pretrained = _env_bool("VENET_USE_PRETRAINED", True)
    model_variant = _arg_or_env_str(args, "model_variant", "VENET_MODEL_VARIANT", "temporal3").strip().lower()
    num_frames = _resolve_num_frames(model_variant, explicit_num_frames=args.num_frames)
    if args.num_frames is not None and args.model_variant is None and os.getenv("VENET_MODEL_VARIANT") in (None, "", "temporal3", "temporal2", "temporal"):
        if "temporal" in model_variant:
            model_variant = f"temporal{num_frames}"
    frame_stride = _arg_or_env(args, "frame_stride", "VENET_FRAME_STRIDE", 1, int)
    freeze_backbone_epochs = int(os.getenv("VENET_FREEZE_BACKBONE_EPOCHS", "3"))
    backbone_lr_factor = float(os.getenv("VENET_BACKBONE_LR_FACTOR", "0.2"))
    aux_cls_weight = float(os.getenv("VENET_AUX_CLS_WEIGHT", "0.15"))
    use_distillation = _arg_or_env_bool(args, "use_distillation", "VENET_USE_DISTILLATION", False)
    distill_weight = _arg_or_env(args, "distill_weight", "VENET_DISTILL_WEIGHT", 0.20, float)
    soft_label_temperature = float(os.getenv("VENET_SOFT_LABEL_TEMPERATURE", "0.03"))
    soft_label_neighbors = int(os.getenv("VENET_SOFT_LABEL_NEIGHBORS", "3"))
    ema_decay = float(os.getenv("VENET_EMA_DECAY", "0.999"))
    log_dir = _arg_or_env_str(args, "log_dir", "VENET_LOG_DIR", "")

    output_dir = Path(_arg_or_env_str(args, "output_dir", "VENET_OUTPUT_DIR", "runs/temporal3"))
    save_name = _arg_or_env_str(args, "save_name", "VENET_SAVE_NAME", "temporal3.pth")
    best_save_name = _arg_or_env_str(args, "best_save_name", "VENET_BEST_SAVE_NAME", f"best_{save_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")

    cudnn.benchmark = True
    writer = SummaryWriter(log_dir=log_dir if log_dir else None)

    aug_config = _build_aug_config(num_frames=num_frames)
    train_transform = build_train_transform_bundle(aug_config) if enable_train_aug else build_eval_transforms(aug_config)
    eval_transform = build_eval_transforms(aug_config)
    stress_transform = build_stress_transforms(aug_config)

    teacher_model = None
    teacher_config = None
    if use_distillation:
        if not teacher_ckpt:
            raise ValueError("--use_distillation requires --teacher_ckpt")
        if distill_weight <= 0:
            raise ValueError(f"--distill_weight must be > 0 when distillation is enabled, got {distill_weight}")
        teacher_path = Path(teacher_ckpt).expanduser().resolve()
        teacher_model, teacher_config = _load_teacher_model(
            teacher_path,
            device,
            expected_preprocess=aug_config.preprocess,
            expected_num_frames=num_frames,
            expected_frame_stride=frame_stride,
        )
        print(
            f"DistillationConfig enabled=1 ckpt={teacher_path} distill_weight={distill_weight:.3f} "
            f"teacher_num_frames={teacher_config['numFrames']} teacher_frame_stride={teacher_config['frameStride']} "
            f"teacher_trainable_params={teacher_config['trainableParams']} same_input=1"
        )
    else:
        distill_weight = 0.0
        ignored_teacher = f" teacher_ckpt_ignored={teacher_ckpt}" if teacher_ckpt else ""
        print(f"DistillationConfig enabled=0 distill_weight=0.000{ignored_teacher}")

    dataset_kwargs = {"num_frames": num_frames, "frame_stride": frame_stride}
    train_dataset = AutoDriveDataset(
        data_folder,
        mode="train",
        split_name="train_clean",
        transform=train_transform,
        return_meta=False,
        **dataset_kwargs,
    )
    val_dataset = AutoDriveDataset(data_folder, mode="val", split_name="val_clean", transform=eval_transform, **dataset_kwargs)
    test_dataset = AutoDriveDataset(data_folder, mode="test", split_name="test_clean", transform=eval_transform, **dataset_kwargs)
    _print_temporal_dataset_summary("train", train_dataset)
    _print_temporal_dataset_summary("val", val_dataset)
    _print_temporal_dataset_summary("test", test_dataset)

    data_folder_path = Path(data_folder)
    has_explicit_val_style_real = any(
        (data_folder_path / candidate).is_file() for candidate in ("val_style_real.txt", "val_style.txt")
    )
    if has_explicit_val_style_real:
        val_style_real_dataset = AutoDriveDataset(
            data_folder,
            mode="val_style_real",
            split_name="val_style_real",
            transform=eval_transform,
            **dataset_kwargs,
        )
        has_val_style_real = True
    else:
        val_style_real_dataset = None
        has_val_style_real = False

    val_stress_dataset = AutoDriveDataset(
        data_folder,
        mode="val",
        split_name="val_clean",
        transform=stress_transform,
        deterministic_seed=20260418,
        **dataset_kwargs,
    )

    angle_vocab = build_angle_vocab(train_dataset.angles)
    if model_variant in {"legacy", "original", "baseline"}:
        model = AutoDriveLegacyNet().to(device)
        use_pretrained = False
        freeze_backbone_epochs = 0
        aux_cls_weight = 0.0
        num_frames = 1
    elif model_variant in {"mobilenet_v1", "current", "baseline_mobilenet"}:
        model = AutoDriveNetV1(num_aux_classes=len(angle_vocab), use_pretrained=use_pretrained).to(device)
    elif "temporal" in model_variant:
        if num_frames < 2:
            raise ValueError(f"temporal model requires num_frames >= 2, got {num_frames}")
        model = AutoDriveNetTemporal(
            num_aux_classes=len(angle_vocab),
            use_pretrained=use_pretrained,
            num_frames=num_frames,
        ).to(device)
    else:
        model = AutoDriveNet(num_aux_classes=len(angle_vocab), use_pretrained=use_pretrained).to(device)

    print(
        f"ModelConfig variant={model_variant} use_pretrained={int(use_pretrained)} pretrained_loaded={int(getattr(model, 'pretrained_loaded', False))} "
        f"freeze_backbone_epochs={freeze_backbone_epochs} aux_cls_weight={aux_cls_weight:.3f} distill_weight={distill_weight:.3f} "
        f"ema_decay={ema_decay:.4f} preprocess={preprocess_config_to_dict(aug_config.preprocess)} "
        f"num_frames={num_frames} frame_stride={frame_stride} angle_vocab={angle_vocab}"
    )
    print(f"ModelTrainableParams { _count_trainable_params(model) }")

    ema = ModelEMA(model, ema_decay)
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        start_epoch = _load_training_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            ema=ema,
            device=device,
            requested_num_frames=num_frames,
        )

    train_sampler = None
    sampler_debug = None
    if use_weighted_sampler:
        sampler_config = _build_sampler_config(len(train_dataset))
        _, raw_sampler_debug = compute_sample_weights(train_dataset.angles, sampler_config, return_debug=True)
        sampler_debug = {
            "edges": [float(x) for x in raw_sampler_debug["edges"].tolist()],
            "counts": [int(x) for x in raw_sampler_debug["counts"].tolist()],
            "useAbsAngle": bool(raw_sampler_debug["useAbsAngle"]),
            "weightsMin": float(raw_sampler_debug["weightsMin"]),
            "weightsMax": float(raw_sampler_debug["weightsMax"]),
            "weightsMean": float(raw_sampler_debug["weightsMean"]),
        }
        train_sampler = build_weighted_sampler(train_dataset.angles, sampler_config)
        print(
            "WeightedRandomSampler enabled | "
            f"bins={sampler_config.num_bins} use_abs_angle={sampler_config.use_abs_angle} "
            f"weight_min={sampler_debug['weightsMin']:.4f} weight_max={sampler_debug['weightsMax']:.4f} weight_mean={sampler_debug['weightsMean']:.4f}"
        )
    else:
        print("WeightedRandomSampler disabled; train DataLoader will use shuffle=True")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    val_stress_loader = torch.utils.data.DataLoader(val_stress_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    val_style_real_loader = (
        torch.utils.data.DataLoader(val_style_real_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
        if val_style_real_dataset is not None
        else None
    )

    reg_criterion = nn.SmoothL1Loss().to(device)
    aux_criterion = SoftTargetCrossEntropy().to(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / save_name
    best_path = output_dir / best_save_name
    summary_path = output_dir / "training_summary.json"

    selection_metric_name = "val_style_real_mae" if has_val_style_real else "val_stress_mae_fallback"
    best_selection_metric = float("inf")
    best_epoch = 0
    no_improve_count = 0
    completed_epochs = start_epoch - 1
    stopped_epoch = None
    early_stopped = False
    last_train_metrics = {
        "total_loss": 0.0,
        "reg_loss": 0.0,
        "aux_loss": 0.0,
        "distill_loss": 0.0,
        "teacher_student_diff": 0.0,
        "mae": 0.0,
    }
    last_val_metrics = {
        "total_loss": 0.0,
        "reg_loss": 0.0,
        "aux_loss": 0.0,
        "distill_loss": 0.0,
        "teacher_student_diff": 0.0,
        "mae": 0.0,
    }
    last_val_style_real_metrics = None
    last_val_stress_metrics = {
        "total_loss": 0.0,
        "reg_loss": 0.0,
        "aux_loss": 0.0,
        "distill_loss": 0.0,
        "teacher_student_diff": 0.0,
        "mae": 0.0,
    }

    current_stage = None
    base_model = _unwrap(model)
    optimizer = None
    scheduler = None

    for epoch in range(start_epoch, epochs + 1):
        desired_stage = "head" if freeze_backbone_epochs > 0 and epoch <= freeze_backbone_epochs else "full"
        if desired_stage != current_stage:
            optimizer, scheduler = _configure_optimizer(
                base_model,
                lr=lr,
                weight_decay=weight_decay,
                backbone_lr_factor=backbone_lr_factor,
                freeze_backbone=(desired_stage == "head"),
            )
            current_stage = desired_stage
            print(f"TrainingStage epoch={epoch} stage={current_stage} backbone_trainable={int(current_stage == 'full')}")

        last_train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            reg_criterion=reg_criterion,
            aux_criterion=aux_criterion,
            angle_vocab=angle_vocab,
            aux_cls_weight=aux_cls_weight,
            soft_label_temperature=soft_label_temperature,
            soft_label_neighbors=soft_label_neighbors,
            teacher_model=teacher_model,
            distill_weight=distill_weight,
            optimizer=optimizer,
            grad_clip=grad_clip,
            ema=ema,
        )

        eval_model = ema.ema_model
        last_val_metrics = _run_epoch(
            model=eval_model,
            loader=val_loader,
            device=device,
            reg_criterion=reg_criterion,
            aux_criterion=aux_criterion,
            angle_vocab=angle_vocab,
            aux_cls_weight=aux_cls_weight,
            soft_label_temperature=soft_label_temperature,
            soft_label_neighbors=soft_label_neighbors,
            optimizer=None,
        )
        if val_style_real_loader is not None:
            last_val_style_real_metrics = _run_epoch(
                model=eval_model,
                loader=val_style_real_loader,
                device=device,
                reg_criterion=reg_criterion,
                aux_criterion=aux_criterion,
                angle_vocab=angle_vocab,
                aux_cls_weight=aux_cls_weight,
                soft_label_temperature=soft_label_temperature,
                soft_label_neighbors=soft_label_neighbors,
                optimizer=None,
            )
        last_val_stress_metrics = _run_epoch(
            model=eval_model,
            loader=val_stress_loader,
            device=device,
            reg_criterion=reg_criterion,
            aux_criterion=aux_criterion,
            angle_vocab=angle_vocab,
            aux_cls_weight=aux_cls_weight,
            soft_label_temperature=soft_label_temperature,
            soft_label_neighbors=soft_label_neighbors,
            optimizer=None,
        )

        completed_epochs = epoch
        selection_metric = (last_val_style_real_metrics or last_val_stress_metrics)["mae"]
        scheduler.step(selection_metric)
        current_lr = max(group["lr"] for group in optimizer.param_groups)

        writer.add_scalar("Train_Total_Loss", last_train_metrics["total_loss"], epoch)
        writer.add_scalar("Train_Reg_Loss", last_train_metrics["reg_loss"], epoch)
        writer.add_scalar("Train_Aux_Loss", last_train_metrics["aux_loss"], epoch)
        writer.add_scalar("Train_Distill_Loss", last_train_metrics["distill_loss"], epoch)
        writer.add_scalar("Train_Teacher_Student_Diff", last_train_metrics["teacher_student_diff"], epoch)
        writer.add_scalar("Train_MAE", last_train_metrics["mae"], epoch)
        writer.add_scalar("Val_Clean_Total_Loss", last_val_metrics["total_loss"], epoch)
        writer.add_scalar("Val_Clean_MAE", last_val_metrics["mae"], epoch)
        writer.add_scalar("Val_Stress_Total_Loss", last_val_stress_metrics["total_loss"], epoch)
        writer.add_scalar("Val_Stress_MAE", last_val_stress_metrics["mae"], epoch)
        if last_val_style_real_metrics is not None:
            writer.add_scalar("Val_Style_Real_MAE", last_val_style_real_metrics["mae"], epoch)
        writer.add_scalar("LR", current_lr, epoch)

        val_style_text = f"{last_val_style_real_metrics['mae']:.6f}" if last_val_style_real_metrics is not None else "NA"
        print(
            f"epoch:{epoch}"
            f"  Train_Total_Loss:{last_train_metrics['total_loss']:.6f}"
            f"  Train_Supervised_Loss:{last_train_metrics['reg_loss']:.6f}"
            f"  Train_Aux_Loss:{last_train_metrics['aux_loss']:.6f}"
            f"  Train_Distill_Loss:{last_train_metrics['distill_loss']:.6f}"
            f"  Train_Teacher_Student_Diff:{last_train_metrics['teacher_student_diff']:.6f}"
            f"  Train_MAE:{last_train_metrics['mae']:.6f}"
            f"  Val_Clean_MAE:{last_val_metrics['mae']:.6f}"
            f"  Val_Style_Real_MAE:{val_style_text}"
            f"  Val_Stress_MAE:{last_val_stress_metrics['mae']:.6f}"
            f"  LR:{current_lr:.6f}"
        )

        payload = _make_checkpoint_payload(
            epoch=epoch,
            best_epoch=best_epoch,
            best_metric=best_selection_metric,
            angle_vocab=angle_vocab,
            preprocess=aug_config.preprocess,
            model_variant=model_variant,
            base_model=base_model,
            ema=ema,
            num_frames=num_frames,
            frame_stride=frame_stride,
        )
        torch.save(payload, latest_path)

        if selection_metric < best_selection_metric - early_stop_min_delta:
            best_selection_metric = selection_metric
            best_epoch = epoch
            no_improve_count = 0
            payload["bestEpoch"] = best_epoch
            payload["bestSelectionMetric"] = best_selection_metric
            torch.save(payload, best_path)
        else:
            no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                early_stopped = True
                stopped_epoch = epoch
                print(
                    f"EarlyStopping triggered at epoch {epoch} | best_epoch {best_epoch} | best_{selection_metric_name} {best_selection_metric:.6f}"
                )
                break

    if best_epoch == 0:
        best_epoch = completed_epochs
        best_selection_metric = (last_val_style_real_metrics or last_val_stress_metrics)["mae"]
        if latest_path.is_file() and not best_path.is_file():
            torch.save(torch.load(latest_path, map_location="cpu"), best_path)

    best_checkpoint = torch.load(best_path if best_path.is_file() else latest_path, map_location=device)
    eval_model = copy.deepcopy(base_model).to(device)
    eval_model.load_state_dict(best_checkpoint["model"], strict=True)
    eval_model.eval()
    test_metrics = _run_epoch(
        model=eval_model,
        loader=test_loader,
        device=device,
        reg_criterion=reg_criterion,
        aux_criterion=aux_criterion,
        angle_vocab=angle_vocab,
        aux_cls_weight=aux_cls_weight,
        soft_label_temperature=soft_label_temperature,
        soft_label_neighbors=soft_label_neighbors,
        optimizer=None,
    )
    writer.add_scalar("Test_Clean_MAE", test_metrics["mae"], 0)

    summary = {
        "dataFolder": data_folder,
        "datasetLabelShiftFrames": dataset_summary.get("labelShiftFrames"),
        "requestedEpochs": epochs,
        "completedEpochs": completed_epochs,
        "bestEpoch": best_epoch,
        "stoppedEpoch": stopped_epoch,
        "earlyStopped": early_stopped,
        "modelSelectionMetric": selection_metric_name,
        "selectionMetricValue": float(best_selection_metric),
        "modelVariant": model_variant,
        "usePretrained": use_pretrained,
        "pretrainedLoaded": bool(getattr(base_model, "pretrained_loaded", False)),
        "freezeBackboneEpochs": freeze_backbone_epochs,
        "auxClsWeight": aux_cls_weight,
        "useDistillation": bool(use_distillation),
        "distillWeight": distill_weight,
        "teacherWeight": distill_weight,
        "teacherEnabled": teacher_model is not None and distill_weight > 0,
        "teacherCheckpoint": teacher_ckpt,
        "teacherConfig": teacher_config,
        "deploymentLoadsTeacher": False,
        "emaDecay": ema_decay,
        "softLabelTemperature": soft_label_temperature,
        "softLabelNeighbors": soft_label_neighbors,
        "preprocess": preprocess_config_to_dict(aug_config.preprocess),
        "numFrames": num_frames,
        "frameStride": frame_stride,
        "angleVocab": angle_vocab,
        "hasValStyleRealSplit": has_val_style_real,
        "temporalPadding": getattr(train_dataset, "temporal_summary", None),
        "finalTrainLoss": float(last_train_metrics["total_loss"]),
        "finalTrainSupervisedLoss": float(last_train_metrics["reg_loss"]),
        "finalTrainAuxLoss": float(last_train_metrics["aux_loss"]),
        "finalTrainDistillLoss": float(last_train_metrics["distill_loss"]),
        "finalTrainTeacherStudentDiff": float(last_train_metrics["teacher_student_diff"]),
        "finalValCleanLoss": float(last_val_metrics["total_loss"]),
        "finalValCleanMAE": float(last_val_metrics["mae"]),
        "finalValStyleRealMAE": float(last_val_style_real_metrics["mae"]) if last_val_style_real_metrics is not None else None,
        "finalValStressMAE": float(last_val_stress_metrics["mae"]),
        "steeringError": float(best_selection_metric),
        "finalTrainMAE": float(last_train_metrics["mae"]),
        "finalTestCleanLoss": float(test_metrics["total_loss"]),
        "finalTestCleanMAE": float(test_metrics["mae"]),
        "finalTestLoss": float(test_metrics["total_loss"]),
        "finalTestMAE": float(test_metrics["mae"]),
        "usedDedicatedTestSplit": True,
        "sampler": sampler_debug,
    }
    _write_summary(summary_path, summary)
    print(
        f"TrainingSummary requested_epochs={epochs} completed_epochs={completed_epochs} "
        f"best_epoch={best_epoch} early_stopped={int(early_stopped)} stopped_epoch={stopped_epoch or 0} "
        f"best_{selection_metric_name}={best_selection_metric:.6f}"
    )
    writer.close()


if __name__ == "__main__":
    main()
