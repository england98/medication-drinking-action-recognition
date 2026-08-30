"""Independently validate Phase 7 evidence and write the Phase 8 selection artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from src.path_config import load_project_paths
from src.etri_stage_b import StageBError
from src.phase8_selection import Phase8SelectionError, run_selection, write_selection_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=None)
    parser.add_argument("--phase7-config", type=Path, default=Path("configs/phase7_ablation.yaml"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("configs/phase8_selected_model.yaml"))
    return parser.parse_args()


def main() -> int:
    args = parse_args(); paths = load_project_paths(args.paths_config)
    manifest = args.manifest or paths.paths.get("manifests_root", paths.work_root / "manifests") / "pilot" / "etri_pilot_manifest.jsonl"
    artifact = run_selection(paths.project_root, paths.work_root, manifest, args.phase7_config)
    state = write_selection_artifact(args.output, artifact)
    selected = artifact["phase8"]
    print(json.dumps({"status": "PASS", "artifact": str(args.output), "artifact_state": state,
        "selected_experiment": selected["selected_experiment"], "selected_encoder": selected["encoder"]["type"],
        "selected_stage_b": selected["stage_b"]["type"], "primary_metric": selected["selection"]["primary_metric"],
        "primary_value": selected["selection"]["primary_value"],
        "secondary_tie_break_used": selected["selection"]["secondary_tie_break_used"],
        "phase9": selected["phase9"]["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try: sys.exit(main())
    except (Phase8SelectionError, StageBError, OSError, ValueError) as exc:
        print(f"Phase 8 selection failed: {exc}", file=sys.stderr); sys.exit(1)
