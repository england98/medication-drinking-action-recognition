"""Evaluate all 239 ETRI Pilot raw videos with fold-specific Experiment D checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from src.etri_embedding import (load_embedding_config, read_selected_etri_manifest,
                                resolve_video_path, source_clip_key)
from src.etri_stage_b import load_stage_b_checkpoint, load_stage_b_config, model_config_for
from src.path_config import load_project_paths
from src.phase10_inference import (VideoInferencePipeline, file_sha256, resolve_device)
from src.phase10_oof_evaluation import (PREDICTION_FIELDS, RawVideoOofError, aggregate_metrics,
    compare_cached_raw, roi_statistics, sha256_file, validate_oof_universe, validate_predictions,
    write_confusion_csv, write_csv, write_json)
from src.roi_preflight import MediaPipeLandmarkDetector, load_roi_config
from src.stage_a import CLASS_TO_INDEX, build_transform, load_checkpoint, load_stage_a_config


OUTPUT_RELATIVE = Path("evaluations/phase10_raw_video_oof")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None, help="Functional smoke only; never emits final metrics")
    return parser.parse_args(argv)


def checkpoint_paths(work_root: Path) -> dict[int, Path]:
    return {fold: work_root / f"checkpoints/phase7_ablation/phase7_D_fold{fold}/best.pt" for fold in range(5)}


def validate_fold_checkpoints(paths: dict[int, Path], config: Any, manifest_hash: str,
                              encoder_identifier: str, device: torch.device) -> tuple[dict[int, torch.nn.Module], dict[str, Any]]:
    models, records = {}, {}
    for fold in range(5):
        path = paths[fold].resolve(strict=True); model, provenance = load_stage_b_checkpoint(path, device)
        checks = ((provenance.get("phase") == 7, "phase"), (provenance.get("experiment") == "D", "experiment"),
            (provenance.get("fold") == fold, "fold"), (provenance.get("stage_b") == "gru", "architecture"),
            (provenance.get("model_config") == model_config_for("gru", config), "model_config"),
            (provenance.get("encoder") == "ai_hub_fine_tuned", "encoder"),
            (provenance.get("encoder_key") == "encoder_b", "encoder_key"),
            (provenance.get("encoder_cache_identifier") == encoder_identifier, "encoder_identity"),
            (provenance.get("manifest_sha256") == manifest_hash, "manifest"),
            (provenance.get("T") == config.sequence_length, "T"),
            (provenance.get("D") == config.embedding_dimension, "D"),
            (provenance.get("class_mapping") == CLASS_TO_INDEX, "class_mapping"),
            (provenance.get("smoke") is False, "production_checkpoint"))
        for passed, name in checks:
            if not passed: raise RawVideoOofError(f"Fold {fold} checkpoint {name} mismatch")
        with torch.inference_mode(): output = model(torch.zeros(1, config.sequence_length, config.embedding_dimension, device=device))
        if tuple(output.shape) != (1, 3): raise RawVideoOofError(f"Fold {fold} forward smoke failed")
        digest = sha256_file(path); models[fold] = model
        records[str(fold)] = {"path": str(path), "sha256": digest, "restore": "PASS",
            "forward_shape": list(output.shape), "experiment": "D", "stage_b": "gru",
            "validation_fold": fold, "model_config": provenance["model_config"]}
    return models, records


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit is not None and args.limit <= 0: raise RawVideoOofError("--limit must be positive")
    project_paths = load_project_paths(args.paths_config); device = resolve_device(args.device)
    embedding_path = Path("configs/etri_embedding.yaml").resolve(strict=True)
    phase7_path = Path("configs/phase7_ablation.yaml").resolve(strict=True)
    embedding_config = load_embedding_config(embedding_path); stage_b_config = load_stage_b_config(phase7_path)
    manifest = args.manifest or project_paths.paths.get("manifests_root", project_paths.work_root / "manifests") / "pilot/etri_pilot_manifest.jsonl"
    rows, manifest_hash = read_selected_etri_manifest(manifest, Path(embedding_config.pilot_config_path))
    universe = validate_oof_universe(rows)
    encoder_path = (project_paths.work_root / embedding_config.encoder_b_checkpoint_relative_path).resolve(strict=True)
    encoder_hash = sha256_file(encoder_path)
    encoder_identifier = f"stage-a-20260829T170113531051Z-adf86795:{encoder_hash}"
    encoder, encoder_provenance = load_checkpoint(encoder_path, device)
    if encoder_provenance.get("embedding_dimension") != stage_b_config.embedding_dimension:
        raise RawVideoOofError("Encoder embedding dimension mismatch")
    models, checkpoint_records = validate_fold_checkpoints(checkpoint_paths(project_paths.work_root),
        stage_b_config, manifest_hash, encoder_identifier, device)
    roi_path = Path(embedding_config.roi_config_path).resolve(strict=True); roi_config = load_roi_config(roi_path)
    stage_a_path = Path(embedding_config.stage_a_config_path).resolve(strict=True); stage_a_config = load_stage_a_config(stage_a_path)
    roi_hash, preprocessing_hash = file_sha256(roi_path), file_sha256(stage_a_path)
    for fold, record in checkpoint_records.items():
        checkpoint = torch.load(record["path"], map_location="cpu", weights_only=False)["provenance"]
        if checkpoint.get("roi_config_sha256") != roi_hash or checkpoint.get("stage_a_config_sha256") != preprocessing_hash:
            raise RawVideoOofError(f"Fold {fold} preprocessing provenance mismatch")
    output_root = (args.output_root or project_paths.work_root / OUTPUT_RELATIVE).expanduser().resolve(strict=False)
    try: output_root.relative_to(project_paths.work_root.resolve(strict=True))
    except ValueError as exc: raise RawVideoOofError("Evaluation output must remain inside work_root") from exc
    if output_root.exists(): raise RawVideoOofError(f"Refusing to overwrite existing evaluation: {output_root}")
    output_root.mkdir(parents=True)
    timestamp = datetime.now(timezone.utc).isoformat(); git_commit = _git_commit()
    detector = MediaPipeLandmarkDetector(project_paths.work_root / roi_config.face_model_relative_path,
        project_paths.work_root / roi_config.hand_model_relative_path,
        project_paths.work_root / roi_config.pose_model_relative_path, roi_config)
    pipelines = {fold: VideoInferencePipeline(encoder=encoder, stage_b=models[fold], detector=detector,
        roi_config=roi_config, transform=build_transform(stage_a_config, False), embedding_config=embedding_config,
        device=device, encoder_batch_size=embedding_config.inference_batch_size,
        metadata={"selected_encoder": "ai_hub_fine_tuned", "selected_stage_b": "gru",
            "encoder_checkpoint": str(encoder_path), "stage_b_checkpoint": checkpoint_records[str(fold)]["path"],
            "T": stage_b_config.sequence_length, "D": stage_b_config.embedding_dimension,
            "image_size": stage_a_config.image_size}) for fold in range(5)}
    selected_rows = sorted(rows, key=lambda row: (int(row["fold"]), source_clip_key(row)))
    if args.limit is not None: selected_rows = selected_rows[:args.limit]
    predictions, failures = [], []
    try:
        for number, row in enumerate(selected_rows, 1):
            key, fold = source_clip_key(row), int(row["fold"])
            try:
                video = resolve_video_path(row, {"etri_raw_root": project_paths.etri_raw_root})
                result = pipelines[fold].predict(video); probabilities, roi = result["probabilities"], result["roi_counts"]
                prediction = {"dataset": "etri", "clip_key": key, "participant": row["participant"],
                    "fold": fold, "root_key": row["root_key"], "relative_path": row["relative_path"],
                    "action": row["action"], "ground_truth_index": CLASS_TO_INDEX[row["target_class"]],
                    "ground_truth_label": row["target_class"], "prediction_index": result["predicted_index"],
                    "prediction_label": result["predicted_class"], "prob_medication": probabilities["복약"],
                    "prob_drinking": probabilities["음수"], "prob_other": probabilities["기타"],
                    "confidence": result["confidence"], "correct": result["predicted_class"] == row["target_class"],
                    "roi_success": roi["success"], "roi_partial": roi["partial"], "roi_fallback": roi["fallback"],
                    "n_sampled_frames": result["sampled_frame_count"], "encoder_checkpoint": str(encoder_path),
                    "encoder_checkpoint_sha256": encoder_hash, "stage_b_checkpoint": checkpoint_records[str(fold)]["path"],
                    "stage_b_checkpoint_sha256": checkpoint_records[str(fold)]["sha256"],
                    "T": result["T"], "D": result["D"], "roi_config_sha256": roi_hash,
                    "preprocessing_config_sha256": preprocessing_hash, "manifest_sha256": manifest_hash,
                    "git_commit": git_commit, "evaluation_timestamp": timestamp,
                    "inference_success": True, "error": ""}
                predictions.append(prediction); print(f"[{number}/{len(selected_rows)}] PASS fold={fold} {key}", flush=True)
            except Exception as exc:
                failure = {"clip_key": key, "participant": row["participant"], "fold": fold,
                           "error_type": type(exc).__name__, "error": str(exc)}
                failures.append(failure); print(f"[{number}/{len(selected_rows)}] FAIL {key}: {exc}", file=sys.stderr, flush=True)
    finally: detector.close()
    write_json(output_root / "inference_failures.json", {"failures": failures})
    if args.limit is not None:
        write_csv(output_root / "raw_video_oof_predictions_smoke.csv", predictions, PREDICTION_FIELDS)
        print(json.dumps({"status": "SMOKE_PASS" if not failures else "SMOKE_FAIL", "evaluated": len(predictions),
                          "failed": len(failures), "output_root": str(output_root)}, ensure_ascii=False)); return 0 if not failures else 1
    completeness = validate_predictions(predictions, rows)
    metrics = aggregate_metrics(predictions); roi = roi_statistics(predictions)
    cached_csv = project_paths.work_root / "checkpoints/phase7_ablation/phase7_D_aggregate/oof_predictions.csv"
    consistency, comparison = compare_cached_raw(predictions, cached_csv)
    summary = {"evaluation_name": "ETRI Batch B Pilot participant-disjoint 5-fold raw-video OOF End-to-End evaluation",
        "status": "PASS", "total_samples": len(predictions), "class_counts": universe["class_counts"],
        "participant_count": universe["participant_count"], "folds": list(range(5)), "completeness": completeness,
        **metrics, "roi": roi, "manifest_path": str(manifest), "manifest_sha256": manifest_hash,
        "encoder": {"type": "ai_hub_fine_tuned", "path": str(encoder_path), "sha256": encoder_hash,
                    "identifier": encoder_identifier, "frozen": True},
        "fold_checkpoints": checkpoint_records, "preprocessing": {"sampling": embedding_config.sampling_method,
            "T": embedding_config.sequence_length, "D": stage_b_config.embedding_dimension,
            "image_size": stage_a_config.image_size, "normalization": "torchvision_imagenet_default",
            "roi_config_sha256": roi_hash, "stage_a_config_sha256": preprocessing_hash},
        "class_mapping": CLASS_TO_INDEX, "git_commit": git_commit, "evaluation_timestamp": timestamp,
        "device": str(device), "participant_leakage": universe["leakage"]}
    write_csv(output_root / "raw_video_oof_predictions.csv", predictions, PREDICTION_FIELDS)
    write_json(output_root / "raw_video_oof_metrics.json", summary)
    write_json(output_root / "fold_metrics.json", metrics["fold_metrics"])
    write_confusion_csv(output_root / "raw_video_oof_confusion_matrix.csv", metrics["overall"]["confusion_matrix"])
    for fold in range(5): write_confusion_csv(output_root / f"fold{fold}_confusion_matrix.csv",
                                               metrics["fold_metrics"][str(fold)]["confusion_matrix"])
    write_json(output_root / "raw_vs_cached_oof_consistency.json", consistency)
    write_csv(output_root / "raw_vs_cached_oof_predictions.csv", comparison, tuple(comparison[0]))
    print(json.dumps({"status": "PASS", "evaluated": len(predictions), "failed": 0,
        "macro_f1": metrics["overall"]["macro_f1"], "consistency": consistency["final_consistency_verdict"],
        "output_root": str(output_root)}, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    try: sys.exit(main())
    except (RawVideoOofError, OSError, RuntimeError, ValueError) as exc:
        print(f"Phase 10 raw-video OOF failed: {exc}", file=sys.stderr); sys.exit(1)
