"""Phase 9 full-pilot Stage B deployment/check training contracts."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.etri_embedding import source_clip_key
from src.etri_stage_b import (EXPERIMENTS, EtriEmbeddingDataset, StageBConfig,
                              StageBError, model_config_for, validate_fold_contract)
from src.stage_a import CLASS_NAMES, CLASS_TO_INDEX


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_phase9_selection(path: Path, config: StageBConfig) -> dict[str, Any]:
    try:
        root = yaml.safe_load(path.read_text(encoding="utf-8"))["phase8"]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise StageBError(f"Invalid Phase 8 selection artifact {path}: {exc}") from exc
    definition = EXPERIMENTS["D"]
    expected_model = model_config_for("gru", config)
    checks = (
        (root.get("status") == "complete", "Phase 8 selection is not complete"),
        (root.get("selected_experiment") == "D", "Phase 9 requires selected Experiment D"),
        (root.get("encoder", {}).get("type") == definition["encoder"], "Phase 9 requires Encoder B"),
        (root.get("encoder", {}).get("cache_key") == definition["encoder_key"], "Encoder cache key mismatch"),
        (root.get("encoder", {}).get("frozen") is True, "Encoder must remain frozen"),
        (root.get("stage_b", {}).get("type") == definition["stage_b"], "Phase 9 requires GRU Stage B"),
        (root.get("stage_b", {}).get("model_config") == expected_model, "Selected GRU config mismatch"),
        (root.get("provenance", {}).get("manifest_sha256") == config.manifest_sha256,
         "Selection/Phase 7 manifest hash mismatch"),
        (root.get("provenance", {}).get("T") == config.sequence_length, "Selection T mismatch"),
        (root.get("provenance", {}).get("embedding_dimension") == config.embedding_dimension,
         "Selection embedding dimension mismatch"),
        (root.get("provenance", {}).get("class_mapping") == CLASS_TO_INDEX, "Selection class mapping mismatch"),
        (root.get("phase9", {}).get("retrain_scope") == "etri_pilot_all_selected_valid",
         "Selection Phase 9 scope mismatch"),
        (root.get("phase9", {}).get("cv_fold_checkpoint_reuse") is False,
         "Phase 9 must reinitialize Stage B"),
    )
    for passed, message in checks:
        if not passed:
            raise StageBError(message)
    return root


def full_pilot_diagnostics(rows: Sequence[Mapping[str, Any]], dataset: EtriEmbeddingDataset) -> dict[str, Any]:
    validate_fold_contract(rows)
    manifest_keys = [source_clip_key(row) for row in rows]
    dataset_keys = [source_clip_key(row) for row in dataset.rows]
    if dataset_keys != manifest_keys or len(set(dataset_keys)) != len(dataset_keys):
        raise StageBError("Full-pilot dataset coverage/order differs from the frozen manifest")
    if any(row.get("pilot_selected") is not True or row.get("valid") is not True for row in dataset.rows):
        raise StageBError("Full-pilot dataset contains an unselected or invalid sample")
    participants = Counter(str(row["participant"]) for row in dataset.rows)
    classes = Counter(str(row["target_class"]) for row in dataset.rows)
    folds = Counter(int(row["fold"]) for row in dataset.rows)
    if set(classes) != set(CLASS_NAMES):
        raise StageBError("Full-pilot dataset does not contain the canonical three classes")
    return {"status": "PASS", "fold_filter_applied": False, "total_samples": len(dataset),
            "participant_count": len(participants),
            "class_counts": {name: classes[name] for name in CLASS_NAMES},
            "fold_counts": {str(fold): folds[fold] for fold in sorted(folds)},
            "duplicate_clip_keys": len(dataset_keys) - len(set(dataset_keys)),
            "invalid_samples": 0, "unselected_samples": 0, "cache_encoder_key": dataset.encoder_key}


def validate_selected_cache_identity(selection: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    expected = selection.get("encoder", {}).get("identifier")
    actual = summary.get("encoder_b", {}).get("checkpoint_identifier")
    if not expected or actual != expected:
        raise StageBError("Phase 6 Encoder B cache identity differs from the Phase 8 selection")


def phase9_checkpoint_provenance(*, selection_path: Path, selection: Mapping[str, Any],
        config_path: Path, config: StageBConfig, manifest: Path, cache_root: Path,
        summary: Mapping[str, Any], diagnostics: Mapping[str, Any], project_root: Path,
        completed_epochs: int, smoke: bool, mlflow_run_id: str | None,
        git: Mapping[str, Any]) -> dict[str, Any]:
    encoder = selection["encoder"]
    return {"phase": 9, "role": "deployment_check", "evaluation_role": "none_training_diagnostic_only",
        "selected_experiment": "D", "training_scope": "all_valid_pilot_samples",
        "stage_b": "gru", "model_config": model_config_for("gru", config),
        "encoder": encoder["type"], "encoder_key": encoder["cache_key"],
        "encoder_checkpoint": encoder["checkpoint_relative_path"],
        "encoder_cache_identifier": encoder["identifier"], "encoder_frozen": True,
        "T": config.sequence_length, "D": config.embedding_dimension,
        "num_classes": config.num_classes, "class_mapping": CLASS_TO_INDEX,
        "manifest_path": str(manifest), "manifest_sha256": config.manifest_sha256,
        "cache_root": str(cache_root), "roi_config_sha256": summary["roi_config_sha256"],
        "stage_a_config_sha256": summary["stage_a_config_sha256"],
        "sampling": summary["sampling"], "normalization_identity": "torchvision_imagenet_default",
        "seed": config.seed, "seed_policy": "fixed_phase7_base_seed",
        "loss": config.loss, "optimizer": config.optimizer, "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay, "batch_size": config.batch_size,
        "configured_epochs": config.epochs, "completed_epochs": completed_epochs,
        "epoch_policy": "fixed_phase7_config_epochs_final_checkpoint",
        "sample_counts": dict(diagnostics), "config_path": str(config_path),
        "config": asdict(config), "selection_path": str(selection_path),
        "selection_sha256": file_sha256(selection_path), "smoke": smoke,
        "mlflow_run_id": mlflow_run_id, "cv_fold_checkpoint_reuse": False,
        "stage_b_reinitialized": True, **dict(git)}
