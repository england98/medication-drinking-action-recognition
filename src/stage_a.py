"""Phase 5 AI-Hub Stage A dataset, model, evaluation, and checkpoints."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src.pilot_manifest import validate_ai_hub_manifest
from src.roi_preflight import Detection, LandmarkDetector, RoiPreflightConfig, extract_roi


CLASS_NAMES = ("복약", "음수", "기타")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class StageAError(RuntimeError):
    """Raised when a Phase 5 invariant is violated."""


@dataclass(frozen=True)
class StageAConfig:
    seed: int
    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    optimizer: str
    weight_decay: float
    num_workers: int
    device: str
    fine_tuning_mode: str
    last_n_blocks: int
    horizontal_flip_probability: float
    color_jitter: float
    output_relative_path: str
    mlflow_enabled: bool
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    mlflow_artifact_relative_path: str
    roi_config_path: str
    manifest_summary_relative_path: str


def load_stage_a_config(path: Path) -> StageAConfig:
    try:
        root = yaml.safe_load(path.read_text(encoding="utf-8"))["phase5"]
        fine, aug, mlflow, provenance = root["fine_tuning"], root["augmentation"], root["mlflow"], root["provenance"]
        config = StageAConfig(
            seed=int(root["seed"]), image_size=int(root["image_size"]), batch_size=int(root["batch_size"]),
            epochs=int(root["epochs"]), learning_rate=float(root["learning_rate"]),
            optimizer=str(root["optimizer"]), weight_decay=float(root["weight_decay"]),
            num_workers=int(root["num_workers"]), device=str(root["device"]),
            fine_tuning_mode=str(fine["mode"]), last_n_blocks=int(fine["last_n_blocks"]),
            horizontal_flip_probability=float(aug["horizontal_flip_probability"]),
            color_jitter=float(aug["color_jitter"]), output_relative_path=str(root["output_relative_path"]),
            mlflow_enabled=bool(mlflow["enabled"]), mlflow_tracking_uri=str(mlflow["tracking_uri"]),
            mlflow_experiment_name=str(mlflow["experiment_name"]), roi_config_path=str(provenance["roi_config"]),
            mlflow_artifact_relative_path=str(mlflow["artifact_relative_path"]),
            manifest_summary_relative_path=str(provenance["manifest_summary_relative_path"]),
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise StageAError(f"Invalid Stage A config {path}: {exc}") from exc
    if config.image_size <= 0 or config.batch_size <= 0 or config.epochs <= 0 or config.learning_rate <= 0:
        raise StageAError("Stage A size, batch, epochs, and learning rate must be positive")
    if config.optimizer not in {"adam", "adamw", "sgd"} or config.fine_tuning_mode not in {"all", "classifier_only", "last_n_blocks"}:
        raise StageAError("Unsupported optimizer or fine-tuning mode")
    if config.num_workers != 0:
        raise StageAError("Stage A requires num_workers=0 because one MediaPipe detector is owned by the main process")
    return config


def read_stage_a_manifest(path: Path) -> list[dict[str, Any]]:
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        validate_ai_hub_manifest(rows)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise StageAError(f"Cannot read Stage A manifest {path}: {exc}") from exc
    if hashlib.sha256(path.read_bytes()).hexdigest() != before:
        raise StageAError("Fixed Pilot Manifest changed while being read")
    selected = [row for row in rows if row.get("dataset") == "ai_hub" and row.get("pilot_selected") is True and row.get("valid") is True]
    if not selected or any(row.get("viewpoint") != 3 for row in selected):
        raise StageAError("Stage A requires selected valid AI-Hub viewpoint_3 rows")
    if {row.get("target_class") for row in selected} - set(CLASS_NAMES):
        raise StageAError("Manifest contains an unknown target_class")
    return rows


def build_transform(config: StageAConfig, training: bool) -> transforms.Compose:
    operations: list[Any] = [transforms.ToPILImage(), transforms.Resize((config.image_size, config.image_size))]
    if training:
        operations.extend([transforms.RandomHorizontalFlip(config.horizontal_flip_probability),
                           transforms.ColorJitter(config.color_jitter, config.color_jitter,
                                                  config.color_jitter, config.color_jitter / 2)])
    operations.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(operations)


class StageAFrameDataset(Dataset):
    """Expand each selected manifest video into its three frames without mutation."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], split: str, roots: Mapping[str, Path],
                 roi_config: RoiPreflightConfig, detector: LandmarkDetector,
                 transform: Callable[[np.ndarray], torch.Tensor]) -> None:
        if split not in {"train", "val"}: raise StageAError(f"Unsupported split: {split}")
        self.roots, self.roi_config, self.detector, self.transform = roots, roi_config, detector, transform
        self.frames: list[dict[str, Any]] = []
        actors = defaultdict(set)
        for row in rows:
            if row.get("dataset") != "ai_hub": continue
            actors[str(row.get("split"))].add(str(row.get("actor")))
            if row.get("pilot_selected") is True and row.get("valid") is not True:
                raise StageAError("Selected Stage A row must be valid")
            if not (row.get("pilot_selected") is True and row.get("valid") is True and row.get("split") == split): continue
            if row.get("viewpoint") != 3 or row.get("target_class") not in CLASS_TO_INDEX:
                raise StageAError("Selected Stage A row violates viewpoint/class mapping")
            paths = row.get("frame_relative_paths")
            if not isinstance(paths, list) or len(paths) != 3: raise StageAError("Each selected video must have three frames")
            for index, relative in enumerate(paths):
                self.frames.append({"root_key": row["root_key"], "frame_relative_path": relative,
                    "target_class": row["target_class"], "video_id": str(row["video_id"]),
                    "actor": str(row["actor"]), "frame_key": f"{row['video_id']}:{index}",
                    "source_sample_key": str(row.get("relative_path", row["video_id"])), "split": split})
        if actors["train"] & actors["val"]: raise StageAError("AI-Hub train/val actors overlap")
        if not self.frames: raise StageAError(f"No Stage A frames for split={split}")

    def __len__(self) -> int: return len(self.frames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.frames[index]
        root = self.roots.get(str(item["root_key"])); relative = Path(str(item["frame_relative_path"]))
        if root is None or relative.is_absolute() or ".." in relative.parts: raise StageAError("Unsafe frame path")
        path = (root / relative).resolve(strict=False)
        try: path.relative_to(root.resolve(strict=True))
        except ValueError as exc: raise StageAError(f"Frame escapes configured root: {path}") from exc
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None: raise StageAError(f"Cannot read frame: {path}")
        result = extract_roi(image, self.detector.detect(image), self.roi_config)
        rgb = cv2.cvtColor(result.crop, cv2.COLOR_BGR2RGB)
        return {"image": self.transform(rgb), "target": CLASS_TO_INDEX[str(item["target_class"])],
                "target_class": item["target_class"], "video_id": item["video_id"], "actor": item["actor"],
                "frame_path": str(relative), "frame_key": item["frame_key"], "roi_status": result.roi_status,
                "source_sample_key": item["source_sample_key"], "split": item["split"]}


class StageAMobileNetV3(nn.Module):
    embedding_dim = 1024

    def __init__(self, pretrained: bool = True, num_classes: int = 3) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = mobilenet_v3_small(weights=weights)
        self.features, self.avgpool = base.features, base.avgpool
        self.embedding_head = nn.Sequential(*list(base.classifier.children())[:-1])
        self.classifier = nn.Linear(self.embedding_dim, num_classes)
        self.frozen_batchnorm_policy = "not_configured"

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        values = self.avgpool(self.features(images)); values = torch.flatten(values, 1)
        return self.embedding_head(values)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(images))

    def configure_fine_tuning(self, mode: str, last_n_blocks: int = 2) -> None:
        for parameter in self.parameters(): parameter.requires_grad = mode == "all"
        if mode in {"classifier_only", "last_n_blocks"}:
            for parameter in self.embedding_head.parameters(): parameter.requires_grad = True
            for parameter in self.classifier.parameters(): parameter.requires_grad = True
        if mode == "last_n_blocks":
            if not 0 < last_n_blocks <= len(self.features): raise StageAError("Invalid last_n_blocks")
            for block in list(self.features.children())[-last_n_blocks:]:
                for parameter in block.parameters(): parameter.requires_grad = True
        self.frozen_batchnorm_policy = "freeze_running_stats"

    def train(self, mode: bool = True) -> StageAMobileNetV3:
        super().train(mode)
        if mode and self.frozen_batchnorm_policy == "freeze_running_stats":
            for block in self.features.children():
                if not any(parameter.requires_grad for parameter in block.parameters()):
                    block.eval()
        return self


def trainable_parameter_summary(model: StageAMobileNetV3, fine_tuning_mode: str,
                                last_n_blocks: int) -> dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    modules: list[str] = []
    for index, block in enumerate(model.features.children()):
        if any(parameter.requires_grad for parameter in block.parameters()): modules.append(f"features.{index}")
    if any(parameter.requires_grad for parameter in model.embedding_head.parameters()): modules.append("embedding_head")
    if any(parameter.requires_grad for parameter in model.classifier.parameters()): modules.append("classifier")
    return {"total_parameter_count": total, "trainable_parameter_count": trainable,
            "trainable_ratio": trainable / total if total else 0.0, "trainable_modules": modules,
            "fine_tuning_mode": fine_tuning_mode, "last_n_blocks": last_n_blocks,
            "frozen_batchnorm_policy": model.frozen_batchnorm_policy}


def classification_metrics(targets: Sequence[int], logits: torch.Tensor) -> dict[str, Any]:
    predictions = logits.argmax(1).cpu().numpy(); truth = np.asarray(targets, dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(truth, predictions, labels=range(3), zero_division=0)
    return {"macro_f1": float(f1.mean()), "per_class": {name: {"precision": float(precision[i]),
            "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(CLASS_NAMES)}, "confusion_matrix": confusion_matrix(truth, predictions, labels=range(3)).tolist()}


def evaluate_predictions(targets: Sequence[int], logits: torch.Tensor, video_ids: Sequence[str]) -> dict[str, Any]:
    frame = classification_metrics(targets, logits)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, video_id in enumerate(video_ids): groups[str(video_id)].append(index)
    video_targets, video_logits = [], []
    for video_id in sorted(groups):
        indexes = groups[video_id]
        labels = {int(targets[index]) for index in indexes}
        if len(labels) != 1 or len(indexes) != 3: raise StageAError(f"Video {video_id} must have three frames with one label")
        video_targets.append(labels.pop()); video_logits.append(logits[indexes].mean(0))
    return {"frame": frame, "video": classification_metrics(video_targets, torch.stack(video_logits)),
            "note": "Stage A 음수는 AI-Hub auxiliary visual class이며 최종 물 마시기 성능이 아닙니다."}


def git_commit_hash(project_root: Path) -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def git_provenance(project_root: Path) -> dict[str, Any]:
    commit = git_commit_hash(project_root)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=project_root,
                            capture_output=True, text=True, check=False)
    dirty = status.returncode != 0 or bool(status.stdout.strip())
    diff_hash = None
    if dirty:
        tracked = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=project_root,
                                 capture_output=True, check=False).stdout
        digest = hashlib.sha256(tracked)
        untracked_result = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "-z"],
                                          cwd=project_root, capture_output=True, check=False)
        untracked = sorted(value.decode(errors="surrogateescape")
                           for value in untracked_result.stdout.split(b"\0") if value)
        for relative in untracked:
            path = project_root / Path(relative)
            digest.update(relative.encode())
            if path.is_file(): digest.update(path.read_bytes())
        diff_hash = digest.hexdigest()
    return {"git_commit_hash": commit, "git_dirty": dirty, "git_diff_sha256": diff_hash}


def save_checkpoint(path: Path, model: StageAMobileNetV3, provenance: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format_version": 1, "model_state_dict": model.state_dict(), "provenance": dict(provenance)}, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[StageAMobileNetV3, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    provenance = payload.get("provenance", {})
    if provenance.get("backbone") != "mobilenet_v3_small" or provenance.get("class_mapping") != CLASS_TO_INDEX:
        raise StageAError("Checkpoint provenance is incompatible with Stage A")
    model = StageAMobileNetV3(pretrained=False); model.load_state_dict(payload["model_state_dict"]); model.to(device)
    return model, provenance


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
