"""Build ETRI Batch B RGB Full Candidate Inventory.

Run from the repository root with ``python -m scripts.build_etri_inventory``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.etri_inventory import build_inventory, write_inventory
from src.path_config import load_project_paths


def exit_code_for_status(status: str) -> int:
    return 0 if status in {"PASS", "PASS_WITH_WARNINGS"} else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, default=None, help="Smoke-test limit; disables full-count checks.")
    parser.add_argument("--skip-rgb-metadata", action="store_true", help="Mock/diagnostic use only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_candidates is not None and args.max_candidates <= 0:
        raise SystemExit("--max-candidates must be positive")
    paths = load_project_paths(args.paths_config)
    output_dir = (args.output_dir or paths.work_root / "manifests" / "etri").expanduser().resolve(strict=False)
    for raw_label, raw_root in (("AI-Hub", paths.ai_hub_raw_root), ("ETRI", paths.etri_raw_root)):
        if output_dir == raw_root or raw_root in output_dir.parents:
            raise SystemExit(f"Refusing to write inventory inside {raw_label} Raw root.")
    result = build_inventory(
        paths.etri_raw_root,
        max_candidates=args.max_candidates,
        validate_rgb_metadata=not args.skip_rgb_metadata,
    )
    inventory_path, summary_path = write_inventory(result, output_dir)
    print(json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"inventory: {inventory_path}")
    print(f"summary: {summary_path}")
    return exit_code_for_status(result.summary["status"])


if __name__ == "__main__":
    sys.exit(main())
