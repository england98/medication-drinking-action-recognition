"""Build and validate the AI-Hub Full Candidate Inventory.

Run from the repository root with ``python -m scripts.build_ai_hub_inventory``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.ai_hub_inventory import build_inventory, write_inventory
from src.path_config import load_project_paths


def exit_code_for_status(status: str) -> int:
    """Return success only for the two explicitly accepted inventory states."""

    return 0 if status in {"PASS", "PASS_WITH_WARNINGS"} else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=None)
    parser.add_argument("--metadata-xlsx", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, default=None, help="Smoke-test limit; disables full-count checks.")
    parser.add_argument("--skip-metadata", action="store_true", help="Diagnostic use only; full runs should join metadata.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_candidates is not None and args.max_candidates <= 0:
        raise SystemExit("--max-candidates must be positive")
    paths = load_project_paths(args.paths_config)
    output_dir = args.output_dir or paths.work_root / "manifests" / "ai_hub"
    output_dir = output_dir.expanduser().resolve(strict=False)
    for raw_label, raw_root in (
        ("AI-Hub", paths.ai_hub_raw_root),
        ("ETRI", paths.etri_raw_root),
    ):
        if output_dir == raw_root or raw_root in output_dir.parents:
            raise SystemExit(f"Refusing to write inventory inside {raw_label} Raw root.")
    result = build_inventory(
        raw_root=paths.ai_hub_raw_root,
        metadata_path=args.metadata_xlsx,
        max_candidates=args.max_candidates,
        use_metadata=not args.skip_metadata,
    )
    inventory_path, summary_path = write_inventory(result, output_dir)
    print(json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"inventory: {inventory_path}")
    print(f"summary: {summary_path}")
    return exit_code_for_status(result.summary["status"])


if __name__ == "__main__":
    sys.exit(main())
