"""Train, evaluate, smoke-test, or verify Phase 5 Stage A encoders."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.path_config import load_project_paths
from src.roi_preflight import MediaPipeLandmarkDetector, load_roi_config
from src.stage_a import (CLASS_TO_INDEX, IMAGENET_MEAN, IMAGENET_STD, StageAError,
    StageAFrameDataset, StageAMobileNetV3, build_transform, git_provenance, load_checkpoint,
    load_stage_a_config, read_stage_a_manifest, save_checkpoint, set_seed, trainable_parameter_summary)
from src.stage_a_training import is_better_checkpoint, make_optimizer, run_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "train", "evaluate", "verify-encoders"))
    parser.add_argument("--paths-config", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/stage_a.yaml"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_effective_config(config: Any, batch_size: int | None, epochs: int | None,
                             device: torch.device) -> dict[str, Any]:
    if batch_size is not None and batch_size <= 0: raise StageAError("--batch-size must be positive")
    if epochs is not None and epochs <= 0: raise StageAError("--epochs must be positive")
    overrides = {key: value for key, value in {"batch_size": batch_size, "epochs": epochs}.items()
                 if value is not None}
    return {**asdict(config), "effective_batch_size": batch_size or config.batch_size,
            "effective_epochs": epochs or config.epochs, "effective_device": str(device),
            "effective_num_workers": config.num_workers, "cli_overrides": overrides}


def make_run_id(effective: dict[str, Any], now: datetime | None = None) -> str:
    core = json.dumps({key: effective[key] for key in ("seed", "effective_batch_size", "effective_epochs",
                                                       "fine_tuning_mode", "last_n_blocks")}, sort_keys=True)
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S%fZ")
    return f"stage-a-{stamp}-{hashlib.sha256(core.encode()).hexdigest()[:8]}"


def _provenance(config: Any, effective: dict[str, Any], manifest: Path, project_root: Path,
                roi_path: Path, run_id: str, run_name: str, parameter_summary: dict[str, Any],
                mlflow_run_id: str | None) -> dict[str, Any]:
    return {"encoder_type": "ai_hub_fine_tuned", "backbone": "mobilenet_v3_small",
        "model_config": effective, "effective_batch_size": effective["effective_batch_size"],
        "effective_epochs": effective["effective_epochs"], "effective_device": effective["effective_device"],
        "effective_num_workers": effective["effective_num_workers"], "cli_overrides": effective["cli_overrides"],
        "class_count": 3, "class_mapping": CLASS_TO_INDEX,
        "manifest_path": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "roi_config": str(roi_path), "roi_config_sha256": hashlib.sha256(roi_path.read_bytes()).hexdigest(),
        "image_size": config.image_size, "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "augmentation": {"horizontal_flip_probability": config.horizontal_flip_probability,
                         "color_jitter": config.color_jitter}, "seed": config.seed,
        "embedding_dimension": StageAMobileNetV3.embedding_dim, "best_monitor_name": "val_video_macro_f1",
        "frozen_batchnorm_policy": "freeze_running_stats", "trainable_parameter_summary": parameter_summary,
        "run_id": run_id, "run_name": run_name, "mlflow_run_id": mlflow_run_id,
        **git_provenance(project_root)}


def _loaders(config: Any, effective: dict[str, Any], paths: Any, manifest: Path):
    rows = read_stage_a_manifest(manifest); roi_config = load_roi_config(Path(config.roi_config_path))
    detector = MediaPipeLandmarkDetector(paths.work_root / roi_config.face_model_relative_path,
        paths.work_root / roi_config.hand_model_relative_path, paths.work_root / roi_config.pose_model_relative_path,
        roi_config)
    roots = {"ai_hub_raw_root": paths.ai_hub_raw_root}
    # MediaPipe task instances are intentionally kept in the main process.
    train = StageAFrameDataset(rows, "train", roots, roi_config, detector, build_transform(config, True))
    val = StageAFrameDataset(rows, "val", roots, roi_config, detector, build_transform(config, False))
    generator = torch.Generator().manual_seed(config.seed)
    return (DataLoader(train, batch_size=effective["effective_batch_size"], shuffle=True,
                       num_workers=effective["effective_num_workers"], generator=generator),
            DataLoader(val, batch_size=effective["effective_batch_size"], shuffle=False,
                       num_workers=effective["effective_num_workers"]), detector)


def _start_mlflow_run(config: Any, project_root: Path, run_name: str):
    import mlflow
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    artifact_root = (project_root / config.mlflow_artifact_relative_path).resolve(strict=False)
    artifact_root.mkdir(parents=True, exist_ok=True)
    expected = artifact_root.as_uri()
    experiment = mlflow.get_experiment_by_name(config.mlflow_experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(config.mlflow_experiment_name, artifact_location=expected)
    else:
        experiment_id = experiment.experiment_id
        if Path(experiment.artifact_location.removeprefix("file://")).resolve(strict=False) != artifact_root:
            raise StageAError(f"MLflow experiment artifact root mismatch: {experiment.artifact_location} != {expected}")
    return mlflow.start_run(experiment_id=experiment_id, run_name=run_name)


def best_final_metrics(best_epoch: int, best_result: dict[str, Any], final_epoch: int,
                       final_result: dict[str, Any], best_monitor_value: float) -> dict[str, float]:
    metrics = {"best_epoch": float(best_epoch), "best_monitor_value": best_monitor_value,
        "best_val_frame_macro_f1": best_result["metrics"]["frame"]["macro_f1"],
        "best_val_video_macro_f1": best_result["metrics"]["video"]["macro_f1"],
        "final_epoch": float(final_epoch),
        "final_val_frame_macro_f1": final_result["metrics"]["frame"]["macro_f1"],
        "final_val_video_macro_f1": final_result["metrics"]["video"]["macro_f1"]}
    for prefix, result in (("best", best_result), ("final", final_result)):
        for level in ("frame", "video"):
            for class_name, values in result["metrics"][level]["per_class"].items():
                for metric in ("precision", "recall", "f1"):
                    metrics[f"{prefix}_val_{level}_{CLASS_TO_INDEX[class_name]}_{metric}"] = values[metric]
    return metrics


def main() -> int:
    args = parse_args(); config = load_stage_a_config(args.config); set_seed(config.seed); device = _device(config.device)
    effective = resolve_effective_config(config, args.batch_size, args.epochs, device)
    if args.command == "verify-encoders":
        encoder_a = StageAMobileNetV3(pretrained=True).eval().to(device)
        if args.checkpoint is None: raise StageAError("--checkpoint is required for Encoder B verification")
        encoder_b, provenance = load_checkpoint(args.checkpoint, device); encoder_b.eval()
        sample = torch.zeros(2, 3, config.image_size, config.image_size, device=device)
        with torch.no_grad(): shape_a = list(encoder_a.extract_features(sample).shape); shape_b = list(encoder_b.extract_features(sample).shape)
        if shape_a != shape_b: raise StageAError(f"Encoder interfaces differ: {shape_a} vs {shape_b}")
        print(json.dumps({"encoder_a_embedding": shape_a, "encoder_b_embedding": shape_b,
                          "embedding_dimension": provenance["embedding_dimension"]}, indent=2)); return 0
    paths = load_project_paths(args.paths_config)
    manifests_root = paths.paths.get("manifests_root", paths.work_root / "manifests")
    manifest = args.manifest or manifests_root / "pilot" / "ai_hub_pilot_manifest.jsonl"
    train_loader, val_loader, detector = _loaders(config, effective, paths, manifest)
    try:
        if args.command == "evaluate":
            if args.checkpoint is None: raise StageAError("--checkpoint is required for evaluation")
            model, _ = load_checkpoint(args.checkpoint, device); result = run_epoch(model, val_loader, device)
            print(json.dumps(result, ensure_ascii=False, indent=2)); return 0
        model = StageAMobileNetV3(pretrained=True).to(device)
        model.configure_fine_tuning(config.fine_tuning_mode, config.last_n_blocks)
        parameter_summary = trainable_parameter_summary(model, config.fine_tuning_mode, config.last_n_blocks)
        print(json.dumps({"trainable_parameter_summary": parameter_summary}, ensure_ascii=False, indent=2))
        optimizer = make_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
        if args.command == "smoke":
            train_result = run_epoch(model, train_loader, device, optimizer, max_batches=2)
            val_result = run_epoch(model, val_loader, device, max_batches=1)
            print(json.dumps({"device": str(device), "train": train_result, "val": val_result}, indent=2)); return 0
        epochs = effective["effective_epochs"]; run_id = make_run_id(effective); run_name = run_id
        output = paths.work_root / config.output_relative_path / run_id
        output.mkdir(parents=True, exist_ok=False); best_path = output / "best.pt"
        effective_path = output / "effective_config.json"; history_path = output / "history.json"
        effective_path.write_text(json.dumps(effective, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        mlflow_enabled = config.mlflow_enabled and not args.no_mlflow
        run_context = _start_mlflow_run(config, paths.project_root, run_name) if mlflow_enabled else contextlib.nullcontext(None)
        with run_context as active_run:
            mlflow_run_id = active_run.info.run_id if active_run is not None else None
            provenance = _provenance(config, effective, manifest, paths.project_root,
                                     Path(config.roi_config_path), run_id, run_name,
                                     parameter_summary, mlflow_run_id)
            summary_path = paths.work_root / config.manifest_summary_relative_path
            if summary_path.is_file(): provenance["manifest_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
            history, best_value, best_loss, best_epoch, best_result = [], -1.0, float("inf"), 0, None
            if mlflow_enabled:
                import mlflow
                mlflow.log_params({"run_id": run_id, "seed": config.seed, "backbone": "mobilenet_v3_small",
                    "encoder_type": "ai_hub_fine_tuned", "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay, "optimizer": config.optimizer,
                    "batch_size": effective["effective_batch_size"], "epochs": epochs,
                    "num_workers": effective["effective_num_workers"], "device": effective["effective_device"],
                    "fine_tuning_mode": config.fine_tuning_mode, "last_n_blocks": config.last_n_blocks,
                    "total_parameter_count": parameter_summary["total_parameter_count"],
                    "trainable_parameter_count": parameter_summary["trainable_parameter_count"],
                    "trainable_ratio": parameter_summary["trainable_ratio"],
                    "trainable_modules": json.dumps(parameter_summary["trainable_modules"]),
                    "frozen_batchnorm_policy": "freeze_running_stats",
                    "manifest_sha256": provenance["manifest_sha256"], "roi_config_sha256": provenance["roi_config_sha256"],
                    "git_commit_hash": provenance["git_commit_hash"], "git_dirty": provenance["git_dirty"]})
            for epoch in range(1, epochs + 1):
                train_result = run_epoch(model, train_loader, device, optimizer)
                val_result = run_epoch(model, val_loader, device); value = val_result["metrics"]["video"]["macro_f1"]
                loss = val_result["loss"]; history.append({"epoch": epoch, "train": train_result, "val": val_result})
                if is_better_checkpoint(value, loss, best_value, best_loss):
                    best_value, best_loss, best_epoch, best_result = value, loss, epoch, val_result
                    current = {**provenance, "best_epoch": epoch, "best_monitor_value": value,
                               "best_val_loss": loss, "frame_metrics": val_result["metrics"]["frame"],
                               "video_metrics": val_result["metrics"]["video"]}
                    save_checkpoint(best_path, model, current)
                if mlflow_enabled:
                    mlflow.log_metrics({"train_loss": train_result["loss"], "val_loss": loss,
                        "val_frame_macro_f1": val_result["metrics"]["frame"]["macro_f1"],
                        "val_video_macro_f1": value}, step=epoch)
                print(f"epoch={epoch} train_loss={train_result['loss']:.6f} val_loss={loss:.6f} val_video_macro_f1={value:.6f}")
            history_document = {"run_id": run_id, "effective_config": effective, "epochs": history}
            history_path.write_text(json.dumps(history_document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            assert best_result is not None
            summary_metrics = best_final_metrics(best_epoch, best_result, epochs, history[-1]["val"], best_value)
            if mlflow_enabled:
                mlflow.log_metrics(summary_metrics)
                mlflow.log_dict(best_result["metrics"]["frame"]["confusion_matrix"], "metrics/best_frame_confusion_matrix.json")
                mlflow.log_dict(best_result["metrics"]["video"]["confusion_matrix"], "metrics/best_video_confusion_matrix.json")
                mlflow.log_artifact(history_path, "training")
                mlflow.log_artifact(effective_path, "config")
                mlflow.log_artifact(best_path, "checkpoints")
                mlflow.log_param("best_checkpoint", str(best_path))
        print(f"run output: {output}"); print(f"best checkpoint: {best_path}"); return 0
    finally:
        detector.close()


if __name__ == "__main__":
    try: sys.exit(main())
    except (StageAError, OSError, ValueError) as exc:
        print(f"Stage A failed: {exc}", file=sys.stderr); sys.exit(1)
