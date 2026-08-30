"""Validate or train the selected Phase 9 full-pilot deployment/check Stage B model."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_etri_embedding_cache import validate_manifest_hash_unchanged
from scripts.run_phase7_ablation import _device, _safe_output_root
from src.etri_embedding import EtriEmbeddingError
from src.etri_stage_b import (EtriEmbeddingDataset, StageBError, build_model,
    load_phase6_inventory, load_stage_b_checkpoint, load_stage_b_config,
    save_stage_b_checkpoint, validate_cache_inventory, validate_experiment_fairness)
from src.etri_stage_b_training import run_stage_b_epoch
from src.path_config import load_project_paths
from src.phase9_deployment import (full_pilot_diagnostics, load_phase9_selection,
                                   phase9_checkpoint_provenance, validate_selected_cache_identity)
from src.stage_a import git_provenance, set_seed
from src.stage_a_training import make_optimizer


OUTPUT_RELATIVE_PATH = "checkpoints/phase9_deployment"
MLFLOW_EXPERIMENT = "phase9_pilot_deployment_check"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=None)
    parser.add_argument("--phase7-config", type=Path, default=Path("configs/phase7_ablation.yaml"))
    parser.add_argument("--selection", type=Path, default=Path("configs/phase8_selected_model.yaml"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Validate all inputs without training or outputs")
    mode.add_argument("--train", action="store_true", help="Run the fixed full-data Phase 9 training")
    mode.add_argument("--verify-checkpoint", type=Path, help="Reload and verify a Phase 9 checkpoint")
    parser.add_argument("--smoke", action="store_true", help="Bounded 1-epoch/2-batch training in a separate output")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke and not args.train: parser.error("--smoke requires --train")
    if args.overwrite and not args.train: parser.error("--overwrite requires --train")
    return args


def _phase9_mlflow(config: Any, project_root: Path, run_name: str):
    import mlflow
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    artifacts = (project_root / "runtime/mlflow/artifacts/phase9").resolve(strict=False)
    artifacts.mkdir(parents=True, exist_ok=True); expected = artifacts.as_uri()
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=expected)
    else:
        experiment_id = experiment.experiment_id
        if Path(experiment.artifact_location.removeprefix("file://")).resolve(strict=False) != artifacts:
            raise StageBError("Phase 9 MLflow artifact root mismatch")
    return mlflow.start_run(experiment_id=experiment_id, run_name=run_name)


def _verify(path: Path, device: torch.device) -> dict[str, Any]:
    model, provenance = load_stage_b_checkpoint(path.resolve(strict=True), device)
    if provenance.get("phase") != 9 or provenance.get("smoke") is True:
        raise StageBError("Deployment verification requires a non-smoke Phase 9 checkpoint")
    with torch.no_grad(): logits = model(torch.zeros(2, provenance["T"], provenance["D"], device=device))
    if tuple(logits.shape) != (2, 3) or not torch.isfinite(logits).all():
        raise StageBError("Reloaded Phase 9 checkpoint output contract failed")
    return {"status": "PASS", "checkpoint": str(path), "phase": 9, "role": provenance["role"],
            "selected_experiment": provenance["selected_experiment"], "logits_shape": list(logits.shape)}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv); paths = load_project_paths(args.paths_config)
    config_path = args.phase7_config.resolve(strict=True); config = load_stage_b_config(config_path)
    selection_path = args.selection.resolve(strict=True); selection = load_phase9_selection(selection_path, config)
    validate_experiment_fairness(config); device = _device(config.device)
    if args.verify_checkpoint:
        print(json.dumps(_verify(args.verify_checkpoint, device), ensure_ascii=False, indent=2)); return 0
    manifest = args.manifest or paths.paths.get("manifests_root", paths.work_root / "manifests") / "pilot" / "etri_pilot_manifest.jsonl"
    rows, cache_root, summary, phase6 = load_phase6_inventory(manifest, config, paths.work_root, args.cache_root)
    validate_selected_cache_identity(selection, summary)
    cache_validation = validate_cache_inventory(rows, cache_root, summary, phase6, config)
    dataset = EtriEmbeddingDataset(rows, cache_root, "encoder_b", config, None, "full")
    diagnostics = full_pilot_diagnostics(rows, dataset)
    validation = {"status": "PASS", "phase": 9, "mode": "full_pilot_no_fold_filter",
        "selected_experiment": "D", "encoder": "ai_hub_fine_tuned", "encoder_frozen": True,
        "stage_b": "gru", "dataset": diagnostics, "cache": cache_validation,
        "T": config.sequence_length, "D": config.embedding_dimension,
        "epochs": config.epochs, "batch_size": config.batch_size,
        "learning_rate": config.learning_rate, "optimizer": config.optimizer,
        "loss": config.loss, "seed": config.seed}
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if args.dry_run: return 0

    smoke = args.smoke; run_name = "phase9_deployment_smoke" if smoke else "phase9_deployment_full_pilot"
    output_root = _safe_output_root(args.output_root, paths.work_root, OUTPUT_RELATIVE_PATH)
    output = output_root / run_name
    if output.exists() and not args.overwrite: raise StageBError(f"Output exists; use --overwrite explicitly: {output}")
    output.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed); generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                        num_workers=config.num_workers, generator=generator)
    model = build_model("gru", config).to(device)
    optimizer = make_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
    epochs = 1 if smoke else config.epochs
    context = (_phase9_mlflow(config, paths.project_root, run_name)
               if config.mlflow_enabled and not args.no_mlflow else contextlib.nullcontext(None))
    history = []
    with context as active_run:
        if active_run is not None:
            import mlflow
            mlflow.log_params({"phase": 9, "role": "deployment_check", "selected_experiment": "D",
                "training_scope": "full_pilot", "encoder": "ai_hub_fine_tuned", "encoder_frozen": True,
                "stage_b": "gru", "T": 64, "D": 1024, "manifest_sha256": config.manifest_sha256,
                "total_samples": diagnostics["total_samples"], "participant_count": diagnostics["participant_count"],
                "class_counts": json.dumps(diagnostics["class_counts"], ensure_ascii=False, sort_keys=True),
                "seed": config.seed, "learning_rate": config.learning_rate, "optimizer": config.optimizer,
                "batch_size": config.batch_size, "epochs": epochs, "loss": config.loss,
                "encoder_cache_identifier": selection["encoder"]["identifier"],
                "encoder_checkpoint": selection["encoder"]["checkpoint_relative_path"],
                "roi_config_sha256": summary["roi_config_sha256"], "sampling": summary["sampling"],
                "normalization": "torchvision_imagenet_default", **git_provenance(paths.project_root)})
        for epoch in range(1, epochs + 1):
            result = run_stage_b_epoch(model, loader, device, optimizer, max_batches=2 if smoke else None)
            record = {"epoch": epoch, "training_diagnostic_loss": result["loss"],
                      "training_diagnostic_macro_f1": result["metrics"]["macro_f1"]}
            history.append(record)
            if active_run is not None:
                mlflow.log_metrics(record, step=epoch)
        provenance = phase9_checkpoint_provenance(selection_path=selection_path, selection=selection,
            config_path=config_path, config=config, manifest=manifest, cache_root=cache_root,
            summary=summary, diagnostics=diagnostics, project_root=paths.project_root,
            completed_epochs=epochs, smoke=smoke,
            mlflow_run_id=active_run.info.run_id if active_run is not None else None,
            git=git_provenance(paths.project_root))
        checkpoint = output / ("smoke.pt" if smoke else "deployment_check.pt")
        save_stage_b_checkpoint(checkpoint, model, provenance)
        (output / "training_diagnostics.json").write_text(json.dumps({"role": "training_diagnostic_only",
            "history": history, "dataset": diagnostics}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        loaded, restored = load_stage_b_checkpoint(checkpoint, device)
        loaded_result = loaded(next(iter(DataLoader(dataset, batch_size=2)))["embedding"].to(device))
        if tuple(loaded_result.shape) != (2, 3) or restored.get("phase") != 9:
            raise StageBError("Saved Phase 9 checkpoint reload contract failed")
        if active_run is not None:
            mlflow.log_artifact(checkpoint, "checkpoints")
            mlflow.log_artifact(output / "training_diagnostics.json", "training")
    validate_manifest_hash_unchanged(manifest, config.manifest_sha256)
    print(json.dumps({"status": "PASS", "checkpoint": str(checkpoint), "reload": "PASS",
                      "smoke": smoke, "epochs": epochs}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try: sys.exit(main())
    except (StageBError, EtriEmbeddingError, OSError, ValueError) as exc:
        print(f"Phase 9 failed: {exc}", file=sys.stderr); sys.exit(1)
