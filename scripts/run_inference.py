"""Run the fixed Phase 9 model on one MP4 through the full Phase 10 pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.etri_embedding import EtriEmbeddingError, load_embedding_config
from src.etri_stage_b import StageBError, load_stage_b_checkpoint, load_stage_b_config
from src.path_config import PathConfigError, load_project_paths
from src.phase10_inference import (Phase10InferenceError, VideoInferencePipeline,
    resolve_device, validate_checkpoint_provenance, validate_video_path)
from src.phase9_deployment import load_phase9_selection
from src.roi_preflight import MediaPipeLandmarkDetector, load_roi_config
from src.stage_a import StageAError, build_transform, load_checkpoint, load_stage_a_config


DEFAULT_STAGE_B = Path("checkpoints/phase9_deployment/phase9_deployment_full_pilot/deployment_check.pt")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--paths-config", type=Path, default=None)
    parser.add_argument("--embedding-config", type=Path, default=Path("configs/etri_embedding.yaml"))
    parser.add_argument("--phase7-config", type=Path, default=Path("configs/phase7_ablation.yaml"))
    parser.add_argument("--selection", type=Path, default=Path("configs/phase8_selected_model.yaml"))
    parser.add_argument("--encoder-checkpoint", type=Path, default=None)
    parser.add_argument("--stage-b-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv); video_path = validate_video_path(args.video)
    paths = load_project_paths(args.paths_config)
    embedding_path = args.embedding_config.resolve(strict=True)
    phase7_path = args.phase7_config.resolve(strict=True)
    selection_path = args.selection.resolve(strict=True)
    embedding_config = load_embedding_config(embedding_path)
    stage_b_config = load_stage_b_config(phase7_path)
    selection = load_phase9_selection(selection_path, stage_b_config)
    stage_a_config_path = Path(embedding_config.stage_a_config_path).resolve(strict=True)
    roi_config_path = Path(embedding_config.roi_config_path).resolve(strict=True)
    stage_a_config = load_stage_a_config(stage_a_config_path)
    roi_config = load_roi_config(roi_config_path)
    encoder_checkpoint = (args.encoder_checkpoint or
        paths.work_root / embedding_config.encoder_b_checkpoint_relative_path).resolve(strict=True)
    stage_b_checkpoint = (args.stage_b_checkpoint or paths.work_root / DEFAULT_STAGE_B).resolve(strict=True)
    device = resolve_device(args.device)
    encoder, encoder_provenance = load_checkpoint(encoder_checkpoint, device)
    stage_b, stage_b_provenance = load_stage_b_checkpoint(stage_b_checkpoint, device)
    validate_checkpoint_provenance(stage_b_provenance=stage_b_provenance,
        stage_a_provenance=encoder_provenance, selection=selection,
        embedding_config=embedding_config, stage_b_config=stage_b_config,
        stage_a_config=stage_a_config, stage_a_checkpoint=encoder_checkpoint,
        roi_config_path=roi_config_path, stage_a_config_path=stage_a_config_path)
    detector = MediaPipeLandmarkDetector(paths.work_root / roi_config.face_model_relative_path,
        paths.work_root / roi_config.hand_model_relative_path,
        paths.work_root / roi_config.pose_model_relative_path, roi_config)
    pipeline = VideoInferencePipeline(encoder=encoder, stage_b=stage_b, detector=detector,
        roi_config=roi_config, transform=build_transform(stage_a_config, False),
        embedding_config=embedding_config, device=device,
        encoder_batch_size=embedding_config.inference_batch_size,
        metadata={"selected_encoder": selection["encoder"]["type"],
            "selected_stage_b": selection["stage_b"]["type"],
            "encoder_checkpoint": str(encoder_checkpoint), "stage_b_checkpoint": str(stage_b_checkpoint),
            "T": embedding_config.sequence_length, "D": stage_b_config.embedding_dimension,
            "image_size": stage_a_config.image_size})
    try:
        result = pipeline.predict(video_path)
    finally:
        pipeline.close()
    if args.output_json:
        output = args.output_json.expanduser().resolve(strict=False)
        if output.suffix.lower() != ".json":
            raise Phase10InferenceError("--output-json must use a .json suffix")
        if output == paths.etri_raw_root or paths.etri_raw_root in output.parents:
            raise Phase10InferenceError("Refusing to write output inside Raw ETRI root")
        if output == paths.ai_hub_raw_root or paths.ai_hub_raw_root in output.parents:
            raise Phase10InferenceError("Refusing to write output inside Raw AI-Hub root")
        if output.exists():
            raise Phase10InferenceError(f"Refusing to overwrite existing inference result: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (Phase10InferenceError, PathConfigError, EtriEmbeddingError, StageAError,
            StageBError, OSError, RuntimeError, ValueError) as exc:
        print(f"Inference failed: {exc}", file=sys.stderr); sys.exit(1)
