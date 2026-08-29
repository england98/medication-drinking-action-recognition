"""Build and validate Phase 3 Fixed Pilot Manifests from Phase 2 inventories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.path_config import load_project_paths
from src.pilot_manifest import (
    build_ai_hub_manifest, build_etri_manifest, load_pilot_config, read_jsonl,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=None)
    parser.add_argument("--pilot-config", type=Path, default=Path("configs/pilot_manifest.yaml"))
    parser.add_argument("--ai-hub-inventory", type=Path, default=None)
    parser.add_argument("--etri-inventory", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = load_project_paths(args.paths_config)
    manifests_root = paths.paths.get("manifests_root", paths.work_root / "manifests")
    ai_input = args.ai_hub_inventory or manifests_root / "ai_hub" / "ai_hub_full_candidate_inventory.jsonl"
    etri_input = args.etri_inventory or manifests_root / "etri" / "etri_batch_b_full_candidate_inventory.jsonl"
    output = (args.output_dir or manifests_root / "pilot").resolve(strict=False)
    for raw_root in (paths.ai_hub_raw_root, paths.etri_raw_root):
        if output == raw_root or raw_root in output.parents:
            raise SystemExit("Refusing to write Fixed Pilot Manifest inside a Raw root")
    config = load_pilot_config(args.pilot_config)
    ai_rows, ai_summary = build_ai_hub_manifest(read_jsonl(ai_input), config)
    etri_rows, etri_summary = build_etri_manifest(read_jsonl(etri_input), config)
    ai_paths = write_manifest(ai_rows, ai_summary, output, "ai_hub_pilot_manifest")
    etri_paths = write_manifest(etri_rows, etri_summary, output, "etri_pilot_manifest")
    print(json.dumps({"ai_hub": ai_summary, "etri": etri_summary}, ensure_ascii=False, indent=2, sort_keys=True))
    for path in (*ai_paths, *etri_paths): print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
