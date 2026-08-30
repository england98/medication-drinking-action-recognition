"""Phase 7 Stage B models, immutable cache data, metrics, OOF, and checkpoints."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import Dataset

from src.etri_embedding import (EtriEmbeddingError, deterministic_cache_name,
    load_embedding_config, read_selected_etri_manifest, source_clip_key, validate_cache_payload)
from src.stage_a import CLASS_NAMES, CLASS_TO_INDEX


EXPERIMENTS = {
    "A": {"encoder_key": "encoder_a", "encoder": "imagenet_only", "stage_b": "mean"},
    "B": {"encoder_key": "encoder_b", "encoder": "ai_hub_fine_tuned", "stage_b": "mean"},
    "C": {"encoder_key": "encoder_a", "encoder": "imagenet_only", "stage_b": "gru"},
    "D": {"encoder_key": "encoder_b", "encoder": "ai_hub_fine_tuned", "stage_b": "gru"},
}
EXPECTED_FOLDS = frozenset(range(5))


class StageBError(RuntimeError):
    """Raised when a Phase 7 invariant is violated."""


@dataclass(frozen=True)
class GruConfig:
    hidden_size: int
    num_layers: int
    bidirectional: bool
    dropout: float
    batch_first: bool
    sequence_representation: str


@dataclass(frozen=True)
class StageBConfig:
    seed: int
    sequence_length: int
    embedding_dimension: int
    num_classes: int
    batch_size: int
    epochs: int
    learning_rate: float
    optimizer: str
    weight_decay: float
    num_workers: int
    device: str
    loss: str
    best_monitor: str
    manifest_sha256: str
    phase6_config_path: str
    output_relative_path: str
    gru: GruConfig
    mlflow_enabled: bool
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    mlflow_artifact_relative_path: str


def load_stage_b_config(path: Path) -> StageBConfig:
    try:
        root = yaml.safe_load(path.read_text(encoding="utf-8"))["phase7"]
        gru, mlflow = root["gru"], root["mlflow"]
        config = StageBConfig(
            seed=int(root["seed"]), sequence_length=int(root["sequence_length"]),
            embedding_dimension=int(root["embedding_dimension"]), num_classes=int(root["num_classes"]),
            batch_size=int(root["batch_size"]), epochs=int(root["epochs"]),
            learning_rate=float(root["learning_rate"]), optimizer=str(root["optimizer"]),
            weight_decay=float(root["weight_decay"]), num_workers=int(root["num_workers"]),
            device=str(root["device"]), loss=str(root["loss"]), best_monitor=str(root["best_monitor"]),
            manifest_sha256=str(root["manifest_sha256"]), phase6_config_path=str(root["phase6_config"]),
            output_relative_path=str(root["output_relative_path"]),
            gru=GruConfig(hidden_size=int(gru["hidden_size"]), num_layers=int(gru["num_layers"]),
                bidirectional=bool(gru["bidirectional"]), dropout=float(gru["dropout"]),
                batch_first=bool(gru["batch_first"]),
                sequence_representation=str(gru["sequence_representation"])),
            mlflow_enabled=bool(mlflow["enabled"]), mlflow_tracking_uri=str(mlflow["tracking_uri"]),
            mlflow_experiment_name=str(mlflow["experiment_name"]),
            mlflow_artifact_relative_path=str(mlflow["artifact_relative_path"]),
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise StageBError(f"Invalid Phase 7 config {path}: {exc}") from exc
    if (config.sequence_length, config.embedding_dimension, config.num_classes) != (64, 1024, 3):
        raise StageBError("Phase 7 requires T=64, D=1024, and 3 classes")
    if min(config.batch_size, config.epochs, config.gru.hidden_size, config.gru.num_layers) <= 0:
        raise StageBError("Phase 7 batch/epoch/GRU sizes must be positive")
    if config.learning_rate <= 0 or config.weight_decay < 0 or config.num_workers != 0:
        raise StageBError("Invalid Phase 7 training values; num_workers must be 0")
    if config.optimizer not in {"adam", "adamw", "sgd"} or config.loss != "cross_entropy":
        raise StageBError("Phase 7 requires a supported optimizer and Standard CrossEntropy")
    if config.best_monitor != "val_macro_f1": raise StageBError("Phase 7 best monitor must be val_macro_f1")
    if (config.gru.bidirectional or not config.gru.batch_first or config.gru.dropout != 0.0
            or config.gru.sequence_representation != "final_hidden"):
        raise StageBError("Phase 7 GRU must be unidirectional batch-first final-hidden with dropout=0")
    if len(config.manifest_sha256) != 64: raise StageBError("Phase 7 manifest SHA-256 is invalid")
    for relative in (config.phase6_config_path, config.output_relative_path,
                     config.mlflow_artifact_relative_path):
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise StageBError(f"Phase 7 path must be safe and relative: {relative}")
    return config


def validate_experiment_fairness(config: StageBConfig) -> None:
    expected = {
        "A": ("encoder_a", "imagenet_only", "mean"),
        "B": ("encoder_b", "ai_hub_fine_tuned", "mean"),
        "C": ("encoder_a", "imagenet_only", "gru"),
        "D": ("encoder_b", "ai_hub_fine_tuned", "gru"),
    }
    actual = {key: (value["encoder_key"], value["encoder"], value["stage_b"])
              for key, value in EXPERIMENTS.items()}
    if actual != expected: raise StageBError("A/B/C/D experiment definition is not canonical")
    common = {key: {"T": config.sequence_length, "D": config.embedding_dimension,
        "class_mapping": CLASS_TO_INDEX, "loss": config.loss, "optimizer": config.optimizer,
        "learning_rate": config.learning_rate, "weight_decay": config.weight_decay,
        "batch_size": config.batch_size, "epochs": config.epochs, "seed": config.seed,
        "folds": sorted(EXPECTED_FOLDS), "manifest_sha256": config.manifest_sha256,
        "metric": config.best_monitor} for key in EXPERIMENTS}
    if len({json.dumps(value, ensure_ascii=False, sort_keys=True) for value in common.values()}) != 1:
        raise StageBError("A/B/C/D common training policy differs")


def validate_fold_contract(rows: Sequence[Mapping[str, Any]], validation_fold: int | None = None) -> None:
    if not rows: raise StageBError("No selected valid ETRI rows")
    keys, participant_folds = set(), {}
    folds = set()
    for row in rows:
        if row.get("pilot_selected") is not True or row.get("valid") is not True or row.get("dataset") != "etri":
            raise StageBError("Phase 7 received a non-selected/invalid/non-ETRI row")
        if row.get("target_class") not in CLASS_TO_INDEX:
            raise StageBError(f"Unknown target class: {row.get('target_class')}")
        key = source_clip_key(row)
        if key in keys: raise StageBError(f"Duplicate clip key: {key}")
        keys.add(key); fold = row.get("fold")
        if not isinstance(fold, int) or fold not in EXPECTED_FOLDS: raise StageBError(f"Invalid fold for {key}")
        folds.add(fold); participant = str(row.get("participant"))
        previous = participant_folds.setdefault(participant, fold)
        if previous != fold: raise StageBError(f"Participant assigned to multiple folds: {participant}")
    if folds != EXPECTED_FOLDS: raise StageBError(f"Expected folds 0..4, found {sorted(folds)}")
    if validation_fold is not None:
        if validation_fold not in EXPECTED_FOLDS: raise StageBError(f"Invalid validation fold: {validation_fold}")
        train = [row for row in rows if row["fold"] != validation_fold]
        val = [row for row in rows if row["fold"] == validation_fold]
        if not train or not val: raise StageBError("Train/validation split is empty")
        train_keys = {source_clip_key(row) for row in train}; val_keys = {source_clip_key(row) for row in val}
        train_people = {str(row["participant"]) for row in train}; val_people = {str(row["participant"]) for row in val}
        if train_keys & val_keys or train_people & val_people:
            raise StageBError("Participant/clip leakage between train and validation")


def load_phase6_inventory(manifest: Path, config: StageBConfig, work_root: Path,
                          cache_root: Path | None = None) -> tuple[list[dict[str, Any]], Path, dict[str, Any], Any]:
    phase6 = load_embedding_config(Path(config.phase6_config_path))
    rows, manifest_hash = read_selected_etri_manifest(manifest, Path(phase6.pilot_config_path))
    if manifest_hash != config.manifest_sha256:
        raise StageBError(f"Frozen manifest hash mismatch: {manifest_hash}")
    validate_fold_contract(rows)
    root = cache_root or work_root / phase6.full_cache_relative_path / f"manifest-{manifest_hash[:8]}"
    root = root.expanduser().resolve(strict=False)
    try: root.relative_to(work_root.resolve(strict=True))
    except ValueError as exc: raise StageBError("Phase 7 cache root must be inside work_root") from exc
    summary_path = root / "summary.json"
    try: summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise StageBError(f"Cannot read Phase 6 summary: {summary_path}") from exc
    if (summary.get("status") != "PASS" or summary.get("validation_mode") != "full_239"
            or summary.get("total_cache_success") != len(rows)
            or summary.get("manifest_sha256") != manifest_hash
            or summary.get("manifest_cache_clip_key_set_parity") is not True):
        raise StageBError("Phase 6 full-cache summary is not a compatible PASS artifact")
    return rows, root, summary, phase6


def _validate_selected_embedding(payload: Mapping[str, Any], row: Mapping[str, Any], encoder_key: str,
                                 config: StageBConfig) -> torch.Tensor:
    if encoder_key not in {"encoder_a", "encoder_b"}: raise StageBError(f"Unknown encoder cache: {encoder_key}")
    record = payload.get("encoders", {}).get(encoder_key, {})
    embedding = record.get("embedding"); provenance = record.get("provenance", {})
    if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2:
        raise StageBError(f"Embedding rank invalid for {source_clip_key(row)}/{encoder_key}")
    if tuple(embedding.shape) != (config.sequence_length, config.embedding_dimension):
        raise StageBError(f"Embedding shape invalid for {source_clip_key(row)}/{encoder_key}: {tuple(embedding.shape)}")
    if not torch.isfinite(embedding).all(): raise StageBError(f"Embedding contains NaN/Inf for {source_clip_key(row)}")
    expected_type = "imagenet_only" if encoder_key == "encoder_a" else "ai_hub_fine_tuned"
    if provenance.get("encoder_type") != expected_type:
        raise StageBError(f"Wrong encoder provenance for {source_clip_key(row)}/{encoder_key}")
    if provenance.get("source_clip_key") != source_clip_key(row): raise StageBError("Cache clip key mismatch")
    for field in ("participant", "fold", "target_class"):
        if provenance.get(field) != row.get(field): raise StageBError(f"Cache {field} mismatch")
    return embedding.to(dtype=torch.float32)


def validate_cache_inventory(rows: Sequence[Mapping[str, Any]], cache_root: Path, summary: Mapping[str, Any],
                             phase6: Any, config: StageBConfig) -> dict[str, Any]:
    clips = cache_root / "clips"; expected_files = set(); manifest_hash = config.manifest_sha256
    encoder_a_identifier = str(summary["encoder_a"]["identifier"])
    encoder_b_identifier = str(summary["encoder_b"]["checkpoint_identifier"])
    for row in rows:
        path = clips / deterministic_cache_name(source_clip_key(row)); expected_files.add(path.name)
        if not path.is_file(): raise StageBError(f"Required Phase 6 cache is missing: {path}")
        try: payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc: raise StageBError(f"Cannot load Phase 6 cache: {path}: {exc}") from exc
        try:
            validate_cache_payload(payload, row, manifest_hash, encoder_a_identifier, encoder_b_identifier,
                                   phase6, str(summary["roi_config_sha256"]), str(summary["stage_a_config_sha256"]))
        except EtriEmbeddingError as exc: raise StageBError(f"Phase 6 cache contract failed: {path}: {exc}") from exc
        _validate_selected_embedding(payload, row, "encoder_a", config)
        _validate_selected_embedding(payload, row, "encoder_b", config)
    actual_files = {path.name for path in clips.glob("*.pt")}
    if actual_files != expected_files: raise StageBError("Phase 6 cache/manifest filename set differs")
    return {"status": "PASS", "clips": len(rows), "encoders": ["encoder_a", "encoder_b"],
            "T": config.sequence_length, "D": config.embedding_dimension,
            "manifest_sha256": manifest_hash, "cache_root": str(cache_root)}


class EtriEmbeddingDataset(Dataset):
    def __init__(self, rows: Sequence[Mapping[str, Any]], cache_root: Path, encoder_key: str,
                 config: StageBConfig, fold: int | None, split: str) -> None:
        if split not in {"train", "val", "full"}: raise StageBError(f"Invalid split: {split}")
        if split == "full":
            if fold is not None: raise StageBError("Full-pilot dataset must not apply a fold filter")
            validate_fold_contract(rows)
            self.rows = [dict(row) for row in rows]
        else:
            if fold is None: raise StageBError(f"Phase 7 {split} split requires a validation fold")
            validate_fold_contract(rows, fold)
            self.rows = [dict(row) for row in rows if (row["fold"] != fold) == (split == "train")]
        self.cache_root, self.encoder_key, self.config = cache_root / "clips", encoder_key, config
        if not self.rows: raise StageBError(f"Empty Stage B split: {split}/fold={fold}")

    def __len__(self) -> int: return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]; key = source_clip_key(row)
        path = self.cache_root / deterministic_cache_name(key)
        if not path.is_file(): raise StageBError(f"Required Phase 6 cache is missing: {path}")
        try: payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc: raise StageBError(f"Cannot load Phase 6 cache {path}: {exc}") from exc
        if payload.get("source_clip_key") != key: raise StageBError(f"Cache clip key mismatch: {key}")
        for field in ("participant", "fold", "target_class"):
            if payload.get(field) != row.get(field): raise StageBError(f"Cache {field} mismatch: {key}")
        embedding = _validate_selected_embedding(payload, row, self.encoder_key, self.config)
        target_class = str(row.get("target_class"))
        if target_class not in CLASS_TO_INDEX: raise StageBError(f"Unknown target class: {target_class}")
        roi = payload.get("roi_counts", {})
        roi_status = "fallback" if roi.get("fallback", 0) else "partial" if roi.get("partial", 0) else "success"
        return {"embedding": embedding, "target": CLASS_TO_INDEX[target_class], "target_class": target_class,
            "participant": str(row["participant"]), "fold": int(row["fold"]), "clip_key": key,
            "encoder_key": self.encoder_key, "cache_path": str(path), "provenance": payload["encoders"][self.encoder_key]["provenance"],
            "roi_status": roi_status, "roi_counts": dict(roi)}


def _validate_model_input(values: torch.Tensor, input_size: int) -> None:
    if values.ndim != 3 or values.shape[1] != 64 or values.shape[2] != input_size:
        raise StageBError(f"Stage B input must be [B,64,{input_size}], got {tuple(values.shape)}")


class MeanPoolingClassifier(nn.Module):
    def __init__(self, input_size: int = 1024, num_classes: int = 3) -> None:
        super().__init__(); self.input_size = input_size; self.classifier = nn.Linear(input_size, num_classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _validate_model_input(values, self.input_size)
        return self.classifier(values.mean(dim=1))


class GruClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int, config: GruConfig) -> None:
        super().__init__(); self.input_size = input_size; self.gru_config = config
        self.gru = nn.GRU(input_size=input_size, hidden_size=config.hidden_size,
            num_layers=config.num_layers, batch_first=config.batch_first,
            bidirectional=config.bidirectional, dropout=config.dropout)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _validate_model_input(values, self.input_size)
        _, hidden = self.gru(values)
        return self.classifier(hidden[-1])


def build_model(stage_b: str, config: StageBConfig) -> nn.Module:
    if stage_b == "mean": return MeanPoolingClassifier(config.embedding_dimension, config.num_classes)
    if stage_b == "gru": return GruClassifier(config.embedding_dimension, config.num_classes, config.gru)
    raise StageBError(f"Unknown Stage B architecture: {stage_b}")


def stage_b_metrics(targets: Sequence[int], logits: torch.Tensor) -> dict[str, Any]:
    truth = np.asarray(targets, dtype=np.int64)
    if len(truth) == 0 or logits.ndim != 2 or logits.shape != (len(truth), 3):
        raise StageBError("Metric input shape mismatch")
    if set(truth.tolist()) - set(range(3)): raise StageBError("Metric target outside canonical class mapping")
    predictions = logits.argmax(1).cpu().numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=range(3), zero_division=0)
    return {"macro_f1": float(f1.mean()), "per_class": {name: {"precision": float(precision[index]),
        "recall": float(recall[index]), "f1": float(f1[index]), "support": int(support[index])}
        for index, name in enumerate(CLASS_NAMES)},
        "confusion_matrix": confusion_matrix(truth, predictions, labels=range(3)).tolist(),
        "label_order": list(CLASS_NAMES), "probability_columns": [f"prob_{name}" for name in CLASS_NAMES]}


def make_oof_rows(batch_metadata: Sequence[Mapping[str, Any]], targets: Sequence[int], logits: torch.Tensor,
                  experiment: str) -> list[dict[str, Any]]:
    probabilities = torch.softmax(logits, dim=1).cpu().numpy(); predictions = probabilities.argmax(1)
    definition = EXPERIMENTS[experiment]; result = []
    for index, metadata in enumerate(batch_metadata):
        result.append({"clip_key": metadata["clip_key"], "participant": metadata["participant"],
            "fold": int(metadata["fold"]), "true_label": int(targets[index]), "pred_label": int(predictions[index]),
            "prob_medication": float(probabilities[index, CLASS_TO_INDEX["복약"]]),
            "prob_drinking": float(probabilities[index, CLASS_TO_INDEX["음수"]]),
            "prob_other": float(probabilities[index, CLASS_TO_INDEX["기타"]]),
            "experiment": experiment, "encoder": definition["encoder"], "stage_b": definition["stage_b"],
            "roi_status": metadata["roi_status"]})
    return result


def validate_oof(rows: Sequence[Mapping[str, Any]], manifest_rows: Sequence[Mapping[str, Any]], experiment: str) -> None:
    if experiment not in EXPERIMENTS: raise StageBError(f"Unknown experiment: {experiment}")
    expected = {source_clip_key(row): row for row in manifest_rows}; seen = set()
    for row in rows:
        key = str(row.get("clip_key"))
        if key in seen: raise StageBError(f"Duplicate OOF prediction: {key}")
        seen.add(key); manifest = expected.get(key)
        if manifest is None: raise StageBError(f"Unknown OOF clip: {key}")
        if (row.get("participant") != manifest.get("participant") or row.get("fold") != manifest.get("fold")
                or row.get("true_label") != CLASS_TO_INDEX.get(str(manifest.get("target_class")))):
            raise StageBError(f"OOF manifest mismatch: {key}")
        if row.get("experiment") != experiment: raise StageBError(f"OOF experiment mismatch: {key}")
    if seen != set(expected): raise StageBError(f"OOF missing clips: {len(set(expected) - seen)}")


def aggregate_fold_results(fold_metrics: Sequence[Mapping[str, Any]], oof_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if {int(item["fold"]) for item in fold_metrics} != EXPECTED_FOLDS: raise StageBError("Aggregation requires folds 0..4")
    macro = np.asarray([item["macro_f1"] for item in fold_metrics], dtype=float)
    recall = {name: np.asarray([item["per_class"][name]["recall"] for item in fold_metrics], dtype=float)
              for name in CLASS_NAMES}
    truth = [int(row["true_label"]) for row in oof_rows]; predictions = [int(row["pred_label"]) for row in oof_rows]
    return {"fold_macro_f1": macro.tolist(), "macro_f1_mean": float(macro.mean()),
        "macro_f1_std": float(macro.std(ddof=0)),
        "class_recall": {name: {"fold_values": values.tolist(), "mean": float(values.mean()),
            "std": float(values.std(ddof=0))} for name, values in recall.items()},
        "fold_confusion_matrices": {str(item["fold"]): item["confusion_matrix"] for item in fold_metrics},
        "oof_confusion_matrix": confusion_matrix(truth, predictions, labels=range(3)).tolist(),
        "std_definition": "population_ddof_0", "label_order": list(CLASS_NAMES)}


def save_stage_b_checkpoint(path: Path, model: nn.Module, provenance: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format_version": 1, "model_state_dict": model.state_dict(),
                "provenance": dict(provenance)}, temporary); temporary.replace(path)


def load_stage_b_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False); provenance = payload.get("provenance", {})
    if (payload.get("format_version") != 1 or provenance.get("phase") not in {7, 9}
            or provenance.get("class_mapping") != CLASS_TO_INDEX):
        raise StageBError("Stage B checkpoint provenance is incompatible")
    if provenance.get("phase") == 9 and (provenance.get("role") != "deployment_check"
            or provenance.get("selected_experiment") != "D"):
        raise StageBError("Phase 9 checkpoint role/selection is incompatible")
    model_config = provenance.get("model_config", {}); stage_b = provenance.get("stage_b")
    config = load_stage_b_config(Path(provenance["config_path"]))
    if model_config != model_config_for(stage_b, config): raise StageBError("Stage B checkpoint model config mismatch")
    model = build_model(str(stage_b), config); model.load_state_dict(payload["model_state_dict"]); model.to(device); model.eval()
    return model, dict(provenance)


def model_config_for(stage_b: str, config: StageBConfig) -> dict[str, Any]:
    if stage_b == "mean": return {"type": "mean", "input_size": 1024, "num_classes": 3,
                                   "sequence_representation": "temporal_mean"}
    if stage_b == "gru": return {"type": "gru", "input_size": 1024, "num_classes": 3, **asdict(config.gru)}
    raise StageBError(f"Unknown Stage B architecture: {stage_b}")


def write_oof_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows: raise StageBError("Cannot write empty OOF artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
