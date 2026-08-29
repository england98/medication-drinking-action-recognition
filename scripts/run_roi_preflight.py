"""Run Phase 4 ROI Preflight on deterministic Fixed Pilot representatives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.path_config import load_project_paths
from src.roi_preflight import (
    MediaPipeLandmarkDetector, RoiPreflightError, download_model, load_roi_config,
    read_manifest, run_preflight, select_representatives, write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/roi_preflight.yaml"))
    parser.add_argument("--ai-hub-manifest", type=Path, default=None)
    parser.add_argument("--etri-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--download-models", action="store_true", help="Download official MediaPipe task models into Working cache when absent.")
    parser.add_argument("--max-samples", type=int, default=None, help="Smoke-test only; deterministic prefix after representative selection.")
    parser.add_argument("--sample-key", action="append", default=[], help="Smoke-test only; process an exact representative source_sample_key (repeatable).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")
    if args.max_samples is not None and args.sample_key:
        raise SystemExit("Use either --max-samples or --sample-key, not both")
    try:
        paths = load_project_paths(args.paths_config)
        config = load_roi_config(args.config)
        manifests_root = paths.paths.get("manifests_root", paths.work_root / "manifests")
        ai_path = args.ai_hub_manifest or manifests_root / "pilot" / "ai_hub_pilot_manifest.jsonl"
        etri_path = args.etri_manifest or manifests_root / "pilot" / "etri_pilot_manifest.jsonl"
        output_root = (args.output_dir or paths.work_root / config.output_relative_path).resolve(strict=False)
        for raw_root in (paths.ai_hub_raw_root, paths.etri_raw_root):
            if output_root == raw_root or raw_root in output_root.parents:
                raise RoiPreflightError("Refusing to write ROI output inside a Raw root")
        face_model = paths.work_root / config.face_model_relative_path
        hand_model = paths.work_root / config.hand_model_relative_path
        pose_model = paths.work_root / config.pose_model_relative_path
        if args.download_models:
            if not face_model.is_file(): download_model(config.face_model_url, face_model)
            if not hand_model.is_file(): download_model(config.hand_model_url, hand_model)
            if not pose_model.is_file(): download_model(config.pose_model_url, pose_model)
        samples = select_representatives(
            read_manifest(ai_path, "ai_hub"), read_manifest(etri_path, "etri"), config,
        )
        if args.sample_key:
            requested = set(args.sample_key)
            samples = [sample for sample in samples if sample.source_sample_key in requested]
            missing = requested - {sample.source_sample_key for sample in samples}
            if missing: raise RoiPreflightError(f"Requested representative sample key(s) not found: {sorted(missing)}")
        if args.max_samples is not None: samples = samples[:args.max_samples]
        detector = MediaPipeLandmarkDetector(face_model, hand_model, pose_model, config)
        records, summary = run_preflight(
            samples, {"ai_hub_raw_root": paths.ai_hub_raw_root, "etri_raw_root": paths.etri_raw_root},
            detector, config, output_root,
        )
        report_paths = write_report(records, summary, output_root)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        print(f"representative frames: {len(samples)}")
        for path in report_paths: print(path)
        print("PASS decision: pending user visual review (no automatic threshold applied)")
        return 0
    except RoiPreflightError as exc:
        print(f"ROI Preflight failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
