"""Validate or run Phase 7 Stage B 2x2 participant-disjoint ablation folds."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_etri_embedding_cache import validate_manifest_hash_unchanged
from src.etri_embedding import EtriEmbeddingError
from src.etri_stage_b import (EXPECTED_FOLDS, EXPERIMENTS, EtriEmbeddingDataset, StageBError,
    aggregate_fold_results, build_model, load_phase6_inventory, load_stage_b_checkpoint,
    load_stage_b_config, make_oof_rows, model_config_for, save_stage_b_checkpoint,
    validate_cache_inventory, validate_experiment_fairness, validate_fold_contract,
    validate_oof, write_oof_csv)
from src.etri_stage_b_training import run_stage_b_epoch
from src.path_config import load_project_paths
from src.stage_a import CLASS_TO_INDEX, git_provenance, set_seed
from src.stage_a_training import is_better_checkpoint, make_optimizer


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/phase7_ablation.yaml"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS))
    parser.add_argument("--fold", type=int, choices=tuple(sorted(EXPECTED_FOLDS)))
    parser.add_argument("--all-experiments", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run a bounded training smoke, never a production CV run")
    parser.add_argument("--epochs", type=int, default=None, help="Allowed only with --smoke")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing experiment/fold output")
    args = parser.parse_args(argv)
    if args.validate_only and any((args.experiment, args.all_experiments, args.fold is not None, args.smoke, args.epochs)):
        parser.error("--validate-only cannot be combined with training selection/override arguments")
    if not args.validate_only and not args.experiment and not args.all_experiments:
        parser.error("select --experiment, --all-experiments, or --validate-only")
    if args.experiment and args.all_experiments: parser.error("--experiment and --all-experiments are exclusive")
    if args.fold is not None and args.experiment is None: parser.error("--fold requires --experiment")
    if args.epochs is not None and not args.smoke: parser.error("--epochs is allowed only with --smoke")
    if args.smoke and args.fold is None: parser.error("--smoke requires one --experiment and --fold")
    if args.epochs is not None and args.epochs <= 0: parser.error("--epochs must be positive")
    return args


def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def _safe_output_root(requested: Path | None, work_root: Path, relative: str) -> Path:
    root = (requested or work_root / relative).expanduser().resolve(strict=False)
    try: root.relative_to(work_root.resolve(strict=True))
    except ValueError as exc: raise StageBError("Phase 7 output root must be inside work_root") from exc
    return root


def _start_mlflow(config: Any, project_root: Path, run_name: str):
    import mlflow
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    artifacts = (project_root / config.mlflow_artifact_relative_path).resolve(strict=False); artifacts.mkdir(parents=True, exist_ok=True)
    expected = artifacts.as_uri(); experiment = mlflow.get_experiment_by_name(config.mlflow_experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(config.mlflow_experiment_name, artifact_location=expected)
    else:
        experiment_id = experiment.experiment_id
        if Path(experiment.artifact_location.removeprefix("file://")).resolve(strict=False) != artifacts:
            raise StageBError("Phase 7 MLflow artifact root mismatch")
    return mlflow.start_run(experiment_id=experiment_id, run_name=run_name)


def _checkpoint_provenance(config_path: Path, config: Any, definition: Mapping[str, str], experiment: str,
                           fold: int, manifest: Path, cache_root: Path, summary: Mapping[str, Any],
                           best_epoch: int, best_result: Mapping[str, Any], project_root: Path,
                           smoke: bool, mlflow_run_id: str | None) -> dict[str, Any]:
    encoder_identifier = (summary["encoder_a"]["identifier"] if definition["encoder_key"] == "encoder_a"
                          else summary["encoder_b"]["checkpoint_identifier"])
    return {"phase": 7, "experiment": experiment, "fold": fold, "stage_b": definition["stage_b"],
        "model_config": model_config_for(definition["stage_b"], config), "encoder": definition["encoder"],
        "encoder_key": definition["encoder_key"], "encoder_cache_identifier": encoder_identifier,
        "T": config.sequence_length, "D": config.embedding_dimension, "class_mapping": CLASS_TO_INDEX,
        "manifest_path": str(manifest), "manifest_sha256": config.manifest_sha256,
        "cache_root": str(cache_root), "roi_config_sha256": summary["roi_config_sha256"],
        "stage_a_config_sha256": summary["stage_a_config_sha256"], "sampling": summary["sampling"],
        "normalization_identity": "torchvision_imagenet_default",
        "seed": config.seed + fold, "seed_policy": "base_seed_plus_fold",
        "base_seed": config.seed, "loss": config.loss, "optimizer": config.optimizer,
        "learning_rate": config.learning_rate, "weight_decay": config.weight_decay,
        "batch_size": config.batch_size, "configured_epochs": config.epochs,
        "best_monitor": config.best_monitor, "best_epoch": best_epoch,
        "best_val_loss": best_result["loss"], "best_val_macro_f1": best_result["metrics"]["macro_f1"],
        "config_path": str(config_path), "config": asdict(config), "smoke": smoke,
        "mlflow_run_id": mlflow_run_id, **git_provenance(project_root)}


def run_fold(experiment: str, fold: int, config_path: Path, config: Any, rows: Sequence[Mapping[str, Any]],
             manifest: Path, cache_root: Path, summary: Mapping[str, Any], output_root: Path,
             project_root: Path, device: torch.device, epochs: int, smoke: bool,
             mlflow_enabled: bool, overwrite: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    definition = EXPERIMENTS[experiment]; effective_seed = config.seed + fold; set_seed(effective_seed)
    validate_fold_contract(rows, fold)
    train_dataset = EtriEmbeddingDataset(rows, cache_root, definition["encoder_key"], config, fold, "train")
    val_dataset = EtriEmbeddingDataset(rows, cache_root, definition["encoder_key"], config, fold, "val")
    generator = torch.Generator().manual_seed(config.seed + fold)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    suffix = "_smoke" if smoke else ""; run_name = f"phase7_{experiment}_fold{fold}{suffix}"
    output = output_root / run_name
    if output.exists() and not overwrite: raise StageBError(f"Output exists; use --overwrite explicitly: {output}")
    output.mkdir(parents=True, exist_ok=True)
    effective_config_path = output / "effective_config.json"
    effective_config_path.write_text(json.dumps({"phase7": asdict(config), "experiment": experiment,
        "fold": fold, "effective_seed": effective_seed, "epochs": epochs, "smoke": smoke},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    model = build_model(definition["stage_b"], config).to(device)
    optimizer = make_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
    context = _start_mlflow(config, project_root, run_name) if mlflow_enabled else contextlib.nullcontext(None)
    history, best_value, best_loss, best_epoch, best_result = [], -1.0, float("inf"), 0, None
    with context as active_run:
        if mlflow_enabled:
            import mlflow
            mlflow.log_params({"phase": 7, "experiment": experiment, "fold": fold,
                "encoder": definition["encoder"], "stage_b": definition["stage_b"], "T": 64, "D": 1024,
                "seed": effective_seed, "seed_policy": "base_seed_plus_fold", "loss": config.loss, "optimizer": config.optimizer,
                "learning_rate": config.learning_rate, "weight_decay": config.weight_decay,
                "batch_size": config.batch_size, "epochs": epochs, "manifest_sha256": config.manifest_sha256,
                "encoder_cache_identifier": summary["encoder_a"]["identifier"] if definition["encoder_key"] == "encoder_a" else summary["encoder_b"]["checkpoint_identifier"],
                "roi_config_sha256": summary["roi_config_sha256"], "sampling": summary["sampling"],
                "gru_config": json.dumps(asdict(config.gru), sort_keys=True) if definition["stage_b"] == "gru" else "not_applicable",
                **git_provenance(project_root)})
        for epoch in range(1, epochs + 1):
            train_result = run_stage_b_epoch(model, train_loader, device, optimizer,
                                             max_batches=2 if smoke else None)
            val_result = run_stage_b_epoch(model, val_loader, device,
                                           max_batches=1 if smoke else None)
            history.append({"epoch": epoch, "train_loss": train_result["loss"],
                "val_loss": val_result["loss"], "val_metrics": val_result["metrics"]})
            value, loss = val_result["metrics"]["macro_f1"], val_result["loss"]
            if is_better_checkpoint(value, loss, best_value, best_loss):
                best_value, best_loss, best_epoch, best_result = value, loss, epoch, val_result
                provenance = _checkpoint_provenance(config_path, config, definition, experiment, fold,
                    manifest, cache_root, summary, epoch, val_result, project_root, smoke,
                    active_run.info.run_id if active_run is not None else None)
                save_stage_b_checkpoint(output / "best.pt", model, provenance)
            if mlflow_enabled:
                mlflow.log_metrics({"train_loss": train_result["loss"], "val_loss": loss,
                                    "val_macro_f1": value}, step=epoch)
        assert best_result is not None
        best_model, provenance = load_stage_b_checkpoint(output / "best.pt", device)
        best_result = run_stage_b_epoch(best_model, val_loader, device, max_batches=1 if smoke else None)
        oof = make_oof_rows(best_result["metadata"], best_result["targets"], best_result["logits"], experiment)
        fold_metrics = {"fold": fold, "best_epoch": best_epoch, "validation_loss": best_result["loss"],
                        **best_result["metrics"]}
        (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output / "fold_metrics.json").write_text(json.dumps(fold_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_oof_csv(output / "oof_predictions.csv", oof)
        if mlflow_enabled:
            mlflow.log_metrics({"best_epoch": best_epoch, "best_val_loss": best_result["loss"],
                                "best_val_macro_f1": best_result["metrics"]["macro_f1"]})
            for name, values in best_result["metrics"]["per_class"].items():
                index = CLASS_TO_INDEX[name]; mlflow.log_metrics({f"best_val_class_{index}_precision": values["precision"],
                    f"best_val_class_{index}_recall": values["recall"]})
            mlflow.log_artifact(output / "fold_metrics.json", "metrics")
            mlflow.log_artifact(output / "oof_predictions.csv", "predictions")
            mlflow.log_artifact(output / "history.json", "training")
            mlflow.log_artifact(effective_config_path, "config")
            mlflow.log_artifact(output / "best.pt", "checkpoints")
    return fold_metrics, oof


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv); config_path = args.config.resolve(strict=True); config = load_stage_b_config(config_path)
    validate_experiment_fairness(config); paths = load_project_paths(args.paths_config)
    manifest = args.manifest or paths.paths.get("manifests_root", paths.work_root / "manifests") / "pilot" / "etri_pilot_manifest.jsonl"
    rows, cache_root, summary, phase6 = load_phase6_inventory(manifest, config, paths.work_root, args.cache_root)
    cache_result = validate_cache_inventory(rows, cache_root, summary, phase6, config)
    for fold in EXPECTED_FOLDS: validate_fold_contract(rows, fold)
    validation = {"status": "PASS", "manifest": "PASS", "cache": cache_result,
        "fold_leakage": "PASS", "fairness": "PASS", "provenance": "PASS",
        "selected_valid_clips": len(rows), "folds": sorted(EXPECTED_FOLDS),
        "experiments": EXPERIMENTS, "class_mapping": CLASS_TO_INDEX}
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if args.validate_only: return 0
    output_root = _safe_output_root(args.output_root, paths.work_root, config.output_relative_path)
    experiments = list(EXPERIMENTS) if args.all_experiments else [args.experiment]
    folds = [args.fold] if args.fold is not None else sorted(EXPECTED_FOLDS)
    device = _device(config.device); epochs = args.epochs or (1 if args.smoke else config.epochs)
    for experiment in experiments:
        metrics, oof = [], []
        for fold in folds:
            fold_metrics, fold_oof = run_fold(experiment, fold, config_path, config, rows, manifest,
                cache_root, summary, output_root, paths.project_root, device, epochs, args.smoke,
                config.mlflow_enabled and not args.no_mlflow, args.overwrite)
            metrics.append(fold_metrics); oof.extend(fold_oof)
        if not args.smoke and set(folds) == EXPECTED_FOLDS:
            validate_oof(oof, rows, experiment); aggregate = aggregate_fold_results(metrics, oof)
            experiment_root = output_root / f"phase7_{experiment}_aggregate"; experiment_root.mkdir(exist_ok=args.overwrite)
            write_oof_csv(experiment_root / "oof_predictions.csv", oof)
            (experiment_root / "summary.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validate_manifest_hash_unchanged(manifest, config.manifest_sha256)
    return 0


if __name__ == "__main__":
    try: sys.exit(main())
    except (StageBError, EtriEmbeddingError, OSError, ValueError) as exc:
        print(f"Phase 7 failed: {exc}", file=sys.stderr); sys.exit(1)
