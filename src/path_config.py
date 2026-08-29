"""Load and validate machine-local project paths safely."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


REQUIRED_PATH_KEYS = (
    "project_root",
    "ai_hub_raw_root",
    "etri_raw_root",
    "work_root",
)


class PathConfigError(ValueError):
    """Raised when a local path configuration is invalid or unsafe."""


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical paths loaded from ``configs/paths.local.yaml``."""

    project_root: Path
    ai_hub_raw_root: Path
    etri_raw_root: Path
    work_root: Path
    paths: Mapping[str, Path]

    def __getitem__(self, key: str) -> Path:
        return self.paths[key]


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "paths.local.yaml"


def _is_same_or_descendant(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _validate_raw_work_separation(raw_root: Path, work_root: Path, label: str) -> None:
    if _is_same_or_descendant(work_root, raw_root):
        raise PathConfigError(
            f"Unsafe path configuration: work_root must not equal or be inside "
            f"{label}. ({work_root} vs {raw_root})"
        )
    if _is_same_or_descendant(raw_root, work_root):
        raise PathConfigError(
            f"Unsafe path configuration: {label} must not equal or be inside "
            f"work_root. ({raw_root} vs {work_root})"
        )


def _read_yaml(config_path: Path) -> Mapping[str, Any]:
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            document = yaml.safe_load(config_file)
    except FileNotFoundError as exc:
        raise PathConfigError(f"Path config does not exist: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise PathConfigError(f"Invalid YAML in path config: {config_path}") from exc
    except OSError as exc:
        raise PathConfigError(f"Cannot read path config: {config_path}") from exc

    if not isinstance(document, Mapping):
        raise PathConfigError("Path config must contain a YAML mapping at its root.")
    configured_paths = document.get("paths")
    if not isinstance(configured_paths, Mapping):
        raise PathConfigError("Path config must contain a 'paths' mapping.")
    return configured_paths


def load_project_paths(config_path: str | Path | None = None) -> ProjectPaths:
    """Load paths, canonicalize them, and reject unsafe Raw/Working layouts.

    Required roots must already exist and be directories. Optional configured paths
    are canonicalized without creating or requiring their final targets.
    """

    source = Path(config_path) if config_path is not None else _default_config_path()
    source = source.expanduser().resolve(strict=False)
    configured_paths = _read_yaml(source)

    missing = [key for key in REQUIRED_PATH_KEYS if key not in configured_paths]
    if missing:
        raise PathConfigError(f"Missing required path key(s): {', '.join(missing)}")

    canonical: dict[str, Path] = {}
    for key, value in configured_paths.items():
        if not isinstance(key, str) or not isinstance(value, str) or not value.strip():
            raise PathConfigError(f"Path value for {key!r} must be a non-empty string.")
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            raise PathConfigError(f"Path value for '{key}' must be absolute: {value}")
        canonical[key] = candidate.resolve(strict=False)

    for key in REQUIRED_PATH_KEYS:
        root = canonical[key]
        if not root.exists():
            raise PathConfigError(f"Required root does not exist for '{key}': {root}")
        if not root.is_dir():
            raise PathConfigError(f"Required root is not a directory for '{key}': {root}")
        canonical[key] = root.resolve(strict=True)

    work_root = canonical["work_root"]
    _validate_raw_work_separation(canonical["ai_hub_raw_root"], work_root, "ai_hub_raw_root")
    _validate_raw_work_separation(canonical["etri_raw_root"], work_root, "etri_raw_root")

    return ProjectPaths(
        project_root=canonical["project_root"],
        ai_hub_raw_root=canonical["ai_hub_raw_root"],
        etri_raw_root=canonical["etri_raw_root"],
        work_root=work_root,
        paths=canonical,
    )
