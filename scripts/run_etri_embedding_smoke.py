"""Run the Phase 6 one-clip Encoder A/B embedding-cache smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from src.etri_embedding import (EtriEmbeddingError, encoder_b_training_provenance, freeze_encoder,
    load_embedding_config, process_clip, read_selected_etri_manifest, resolve_video_path,
    save_and_reload_cache, select_smoke_clip, source_clip_key)
from src.path_config import load_project_paths
from src.roi_preflight import MediaPipeLandmarkDetector, load_roi_config
from src.stage_a import StageAMobileNetV3, build_transform, load_checkpoint, load_stage_a_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/etri_embedding.yaml"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--sample-key", default=None)
    parser.add_argument("--output", type=Path, required=True, help="Temporary smoke cache .pt path")
    return parser.parse_args()


def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def main() -> int:
    args = parse_args(); config = load_embedding_config(args.config); paths = load_project_paths(args.paths_config)
    manifest = args.manifest or paths.paths.get("manifests_root", paths.work_root / "manifests") / "pilot" / "etri_pilot_manifest.jsonl"
    output = args.output.expanduser().resolve(strict=False)
    for raw_root in (paths.ai_hub_raw_root, paths.etri_raw_root):
        if output == raw_root or raw_root in output.parents: raise EtriEmbeddingError("Refusing to write smoke cache inside Raw root")
    if output.suffix != ".pt": raise EtriEmbeddingError("Smoke cache output must use a .pt suffix")
    rows, manifest_hash = read_selected_etri_manifest(manifest, Path(config.pilot_config_path))
    row = select_smoke_clip(rows, args.sample_key); video = resolve_video_path(row, {"etri_raw_root": paths.etri_raw_root})
    roi_config_path = Path(config.roi_config_path); roi_config = load_roi_config(roi_config_path)
    stage_a_config_path = Path(config.stage_a_config_path); stage_a_config = load_stage_a_config(stage_a_config_path)
    detector = MediaPipeLandmarkDetector(paths.work_root / roi_config.face_model_relative_path,
        paths.work_root / roi_config.hand_model_relative_path, paths.work_root / roi_config.pose_model_relative_path,
        roi_config)
    device = _device(config.device)
    encoder_a = freeze_encoder(StageAMobileNetV3(pretrained=True).to(device))
    checkpoint = paths.work_root / config.encoder_b_checkpoint_relative_path
    encoder_b, checkpoint_provenance = load_checkpoint(checkpoint, device); freeze_encoder(encoder_b)
    if checkpoint_provenance.get("embedding_dimension") != 1024: raise EtriEmbeddingError("Encoder B checkpoint D is not 1024")
    roi_hash = hashlib.sha256(roi_config_path.read_bytes()).hexdigest()
    stage_a_hash = hashlib.sha256(stage_a_config_path.read_bytes()).hexdigest()
    encoder_a_id = "torchvision:MobileNet_V3_Small_Weights.DEFAULT"
    checkpoint_id = f"stage-a-20260829T170113531051Z-adf86795:{hashlib.sha256(checkpoint.read_bytes()).hexdigest()}"
    try:
        payload = process_clip(row, video, config, roi_config, build_transform(stage_a_config, False), detector,
            encoder_a, encoder_b, device, encoder_a_id, checkpoint_id,
            encoder_b_training_provenance(checkpoint_provenance), roi_hash, stage_a_hash,
            stage_a_config.image_size, manifest_hash)
    finally: detector.close()
    loaded = save_and_reload_cache(output, payload)
    embedding_a = payload["encoders"]["encoder_a"]["embedding"]
    embedding_b = payload["encoders"]["encoder_b"]["embedding"]
    indices = payload["sampled_frame_indices"]; roi_counts = payload["roi_counts"]
    report = {"source_clip_key": source_clip_key(row), "participant": row["participant"], "fold": row["fold"],
        "target_class": row["target_class"], "original_video_frame_count": payload["original_video_frame_count"],
        "sampled_frame_count": len(indices), "sampled_frame_indices": list(indices),
        "roi_success_count": roi_counts["success"], "roi_partial_count": roi_counts["partial"],
        "roi_fallback_count": roi_counts["fallback"], "encoder_a_embedding_shape": list(embedding_a.shape),
        "encoder_b_embedding_shape": list(embedding_b.shape), "dtype": str(embedding_a.dtype),
        "device": str(device), "encoder_a_nan_count": int(torch.isnan(embedding_a).sum()),
        "encoder_a_inf_count": int(torch.isinf(embedding_a).sum()),
        "encoder_b_nan_count": int(torch.isnan(embedding_b).sum()),
        "encoder_b_inf_count": int(torch.isinf(embedding_b).sum()), "cache_save_path": str(output),
        "encoder_a_frozen": all(not parameter.requires_grad for parameter in encoder_a.parameters()),
        "encoder_b_frozen": all(not parameter.requires_grad for parameter in encoder_b.parameters()),
        "a_b_shared_frame_indices": "PASS" if loaded["encoders"]["encoder_a"]["sampled_frame_indices"]
            == loaded["encoders"]["encoder_b"]["sampled_frame_indices"] else "FAIL",
        "cache_reload_result": "PASS" if loaded["source_clip_key"] == source_clip_key(row) else "FAIL"}
    print(json.dumps(report, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    try: sys.exit(main())
    except (EtriEmbeddingError, OSError, ValueError) as exc:
        print(f"ETRI embedding smoke failed: {exc}", file=sys.stderr); sys.exit(1)
