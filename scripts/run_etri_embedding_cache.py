"""Generate and validate the frozen Encoder A/B cache for the 239-clip ETRI Pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.etri_embedding import (EtriEmbeddingConfig, EtriEmbeddingError, cache_statistics,
    deterministic_cache_name, encoder_b_training_provenance, freeze_encoder, load_embedding_config,
    load_valid_cache, process_clip, read_selected_etri_manifest, resolve_video_path,
    save_and_reload_cache, source_clip_key, validate_cache_payload)
from src.path_config import load_project_paths
from src.roi_preflight import MediaPipeLandmarkDetector, load_roi_config
from src.stage_a import StageAMobileNetV3, build_transform, load_checkpoint, load_stage_a_config


EXPECTED_SELECTED_CLIPS = 239
ENCODER_A_IDENTIFIER = "torchvision:MobileNet_V3_Small_Weights.DEFAULT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="Reuse only fully validated existing caches")
    parser.add_argument("--limit", type=int, default=None, help="Multi-clip preflight only; never use for the final run")
    return parser.parse_args()


def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def _safe_output_root(requested: Path | None, work_root: Path, config: EtriEmbeddingConfig,
                      manifest_hash: str) -> Path:
    base = (requested or work_root / config.full_cache_relative_path).expanduser().resolve(strict=False)
    try: base.relative_to(work_root.resolve(strict=True))
    except ValueError as exc: raise EtriEmbeddingError("Full cache output root must be inside work_root") from exc
    return base / f"manifest-{manifest_hash[:8]}"


def build_summary(rows: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]],
                  success: int, failed: Sequence[Mapping[str, Any]], resumed: int,
                  manifest_hash: str, encoder_b_identifier: str, config: EtriEmbeddingConfig,
                  roi_hash: str, stage_a_hash: str, limited: bool) -> dict[str, Any]:
    fold_counts = Counter(str(row["fold"]) for row in rows); class_counts = Counter(str(row["target_class"]) for row in rows)
    roi = Counter(); a_nan = a_inf = b_nan = b_inf = 0
    for record in records:
        roi.update(record["roi_counts"]); a_nan += int(record["encoder_a_nan"]); a_inf += int(record["encoder_a_inf"])
        b_nan += int(record["encoder_b_nan"]); b_inf += int(record["encoder_b_inf"])
    keys = {str(record["source_clip_key"]) for record in records}
    manifest_keys = {source_clip_key(row) for row in rows}
    key_set_hash = lambda values: hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()
    attempted = len(records) + len(failed)
    common_pass = (not failed and len(keys) == len(records)
                   and all(record["encoder_a_shape"] == [64, 1024] and record["encoder_b_shape"] == [64, 1024]
                           and record["encoder_a_nan"] == record["encoder_a_inf"] == 0
                           and record["encoder_b_nan"] == record["encoder_b_inf"] == 0
                           and sum(record["roi_counts"].values()) == 64
                           and record["clip_key_parity"] and record["frame_index_parity"] for record in records))
    full_pass = (not limited and len(rows) == EXPECTED_SELECTED_CLIPS and len(records) == EXPECTED_SELECTED_CLIPS
                 and keys == manifest_keys and common_pass)
    preflight_pass = limited and attempted > 0 and common_pass
    return {"status": "PASS" if full_pass else "PREFLIGHT_PASS" if preflight_pass else "FAIL",
        "validation_mode": "limited_preflight" if limited else "full_239",
        "total_manifest_selected_valid": len(rows), "total_attempted": attempted,
        "total_cache_success": len(records), "total_cache_created": success, "total_cache_failed": len(failed),
        "total_cache_skipped_resumed": resumed, "failures": list(failed),
        "manifest_fold_clip_counts": dict(sorted(fold_counts.items())),
        "manifest_class_clip_counts": dict(sorted(class_counts.items())),
        "roi": {"success_total": roi["success"], "partial_total": roi["partial"],
                "fallback_total": roi["fallback"]},
        "encoder_a": {"identifier": ENCODER_A_IDENTIFIER, "embedding_shape_contract": [64, 1024],
                      "nan_count": a_nan, "inf_count": a_inf},
        "encoder_b": {"checkpoint_identifier": encoder_b_identifier,
                      "embedding_shape_contract": [64, 1024], "nan_count": b_nan, "inf_count": b_inf},
        "a_b": {"clip_key_parity": all(record["clip_key_parity"] for record in records),
                "frame_index_parity": all(record["frame_index_parity"] for record in records),
                "encoder_a_clip_key_count": len(keys), "encoder_b_clip_key_count": len(keys),
                "manifest_clip_key_count": len(manifest_keys),
                "encoder_a_clip_key_set_sha256": key_set_hash(keys),
                "encoder_b_clip_key_set_sha256": key_set_hash(keys),
                "manifest_clip_key_set_sha256": key_set_hash(manifest_keys)},
        "manifest_cache_clip_key_set_parity": keys == manifest_keys if not limited else None,
        "manifest_sha256": manifest_hash, "T": config.sequence_length, "D": 1024,
        "sampling": config.sampling_method, "inference_batch_size": config.inference_batch_size,
        "roi_config_sha256": roi_hash, "stage_a_config_sha256": stage_a_hash}


def validate_summary_gate(summary: Mapping[str, Any], limited: bool) -> None:
    expected = "PREFLIGHT_PASS" if limited else "PASS"
    if summary.get("status") != expected:
        raise EtriEmbeddingError(f"Full cache validation gate failed: status={summary.get('status')}")


def validate_manifest_hash_unchanged(manifest: Path, expected_sha256: str) -> None:
    current_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if current_sha256 != expected_sha256:
        raise EtriEmbeddingError("Manifest hash changed during run")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0: raise EtriEmbeddingError("--limit must be positive")
    paths = load_project_paths(args.paths_config); config = load_embedding_config(Path("configs/etri_embedding.yaml"))
    manifest = args.manifest or paths.paths.get("manifests_root", paths.work_root / "manifests") / "pilot" / "etri_pilot_manifest.jsonl"
    rows, manifest_hash = read_selected_etri_manifest(manifest, Path(config.pilot_config_path))
    rows = sorted(rows, key=source_clip_key); selected_rows = rows[:args.limit] if args.limit is not None else rows
    output_root = _safe_output_root(args.output_root, paths.work_root, config, manifest_hash)
    clips_root = output_root / "clips"; summary_path = output_root / "summary.json"
    roi_path = Path(config.roi_config_path); stage_a_path = Path(config.stage_a_config_path)
    roi_config = load_roi_config(roi_path); stage_a_config = load_stage_a_config(stage_a_path)
    roi_hash = hashlib.sha256(roi_path.read_bytes()).hexdigest(); stage_a_hash = hashlib.sha256(stage_a_path.read_bytes()).hexdigest()
    checkpoint = paths.work_root / config.encoder_b_checkpoint_relative_path
    encoder_b_identifier = f"stage-a-20260829T170113531051Z-adf86795:{hashlib.sha256(checkpoint.read_bytes()).hexdigest()}"
    device = _device(config.device); encoder_a = freeze_encoder(StageAMobileNetV3(pretrained=True).to(device))
    encoder_b, checkpoint_provenance = load_checkpoint(checkpoint, device); freeze_encoder(encoder_b)
    encoder_b_training = encoder_b_training_provenance(checkpoint_provenance)
    detector = MediaPipeLandmarkDetector(paths.work_root / roi_config.face_model_relative_path,
        paths.work_root / roi_config.hand_model_relative_path, paths.work_root / roi_config.pose_model_relative_path,
        roi_config)
    records: list[dict[str, Any]] = []; failures: list[dict[str, Any]] = []; success = resumed = 0
    try:
        for number, row in enumerate(selected_rows, 1):
            key = source_clip_key(row); cache_path = clips_root / deterministic_cache_name(key)
            try:
                if cache_path.exists():
                    if not args.resume: raise EtriEmbeddingError(f"Cache exists; use --resume: {cache_path}")
                    payload = load_valid_cache(cache_path, row, manifest_hash, ENCODER_A_IDENTIFIER,
                        encoder_b_identifier, config, roi_hash, stage_a_hash); resumed += 1
                else:
                    video = resolve_video_path(row, {"etri_raw_root": paths.etri_raw_root})
                    payload = process_clip(row, video, config, roi_config, build_transform(stage_a_config, False),
                        detector, encoder_a, encoder_b, device, ENCODER_A_IDENTIFIER, encoder_b_identifier,
                        encoder_b_training, roi_hash, stage_a_hash, stage_a_config.image_size, manifest_hash)
                    save_and_reload_cache(cache_path, payload)
                    validate_cache_payload(payload, row, manifest_hash, ENCODER_A_IDENTIFIER,
                                           encoder_b_identifier, config, roi_hash, stage_a_hash); success += 1
                records.append(cache_statistics(payload)); print(f"[{number}/{len(selected_rows)}] PASS {key}")
            except Exception as exc:
                failures.append({"source_clip_key": key, "error": str(exc)}); print(f"[{number}/{len(selected_rows)}] FAIL {key}: {exc}", file=sys.stderr)
    finally: detector.close()
    try:
        validate_manifest_hash_unchanged(manifest, manifest_hash)
    except EtriEmbeddingError as exc:
        failures.append({"source_clip_key": None, "error": str(exc)})
    summary = build_summary(rows, records, success, failures, resumed, manifest_hash,
                            encoder_b_identifier, config, roi_hash, stage_a_hash, args.limit is not None)
    _write_json_atomic(summary_path, summary); print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"summary: {summary_path}"); validate_summary_gate(summary, args.limit is not None); return 0


if __name__ == "__main__":
    try: sys.exit(main())
    except (EtriEmbeddingError, OSError, ValueError) as exc:
        print(f"ETRI full embedding cache failed: {exc}", file=sys.stderr); sys.exit(1)
