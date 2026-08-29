from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.path_config import PathConfigError, load_project_paths


def _write_config(config_path: Path, **paths: Path) -> None:
    lines = ["paths:"]
    lines.extend(f'  {key}: "{value}"' for key, value in paths.items())
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class PathConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.roots = {
            "project_root": self.root / "project",
            "ai_hub_raw_root": self.root / "ai_hub_raw",
            "etri_raw_root": self.root / "etri_raw",
            "work_root": self.root / "work",
        }
        for path in self.roots.values():
            path.mkdir()
        self.config = self.root / "paths.local.yaml"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_loads_yaml_and_canonicalizes_required_roots(self) -> None:
        _write_config(self.config, **self.roots)
        loaded = load_project_paths(self.config)
        self.assertEqual(loaded.project_root, self.roots["project_root"].resolve(strict=True))
        self.assertEqual(loaded.ai_hub_raw_root, self.roots["ai_hub_raw_root"].resolve(strict=True))
        self.assertEqual(loaded.etri_raw_root, self.roots["etri_raw_root"].resolve(strict=True))
        self.assertEqual(loaded.work_root, self.roots["work_root"].resolve(strict=True))

    def test_rejects_invalid_yaml(self) -> None:
        self.config.write_text("paths: [unterminated\n", encoding="utf-8")
        with self.assertRaisesRegex(PathConfigError, "Invalid YAML"):
            load_project_paths(self.config)

    def test_rejects_missing_required_key(self) -> None:
        self.roots.pop("work_root")
        _write_config(self.config, **self.roots)
        with self.assertRaisesRegex(PathConfigError, "work_root"):
            load_project_paths(self.config)

    def test_rejects_nonexistent_required_roots(self) -> None:
        for missing_key in ("project_root", "ai_hub_raw_root", "etri_raw_root", "work_root"):
            with self.subTest(missing_key=missing_key):
                paths = dict(self.roots)
                paths[missing_key] = self.root / f"missing-{missing_key}"
                _write_config(self.config, **paths)
                with self.assertRaisesRegex(PathConfigError, "does not exist"):
                    load_project_paths(self.config)

    def test_rejects_work_root_equal_to_raw_roots(self) -> None:
        for raw_key in ("ai_hub_raw_root", "etri_raw_root"):
            with self.subTest(raw_key=raw_key):
                paths = dict(self.roots)
                paths["work_root"] = paths[raw_key]
                _write_config(self.config, **paths)
                with self.assertRaisesRegex(PathConfigError, "must not equal or be inside"):
                    load_project_paths(self.config)

    def test_rejects_work_root_inside_raw_roots(self) -> None:
        for raw_key in ("ai_hub_raw_root", "etri_raw_root"):
            with self.subTest(raw_key=raw_key):
                paths = dict(self.roots)
                nested_work = paths[raw_key] / "derived"
                nested_work.mkdir(exist_ok=True)
                paths["work_root"] = nested_work
                _write_config(self.config, **paths)
                with self.assertRaisesRegex(PathConfigError, "must not equal or be inside"):
                    load_project_paths(self.config)

    def test_rejects_raw_roots_inside_work_root(self) -> None:
        for raw_key in ("ai_hub_raw_root", "etri_raw_root"):
            with self.subTest(raw_key=raw_key):
                paths = dict(self.roots)
                nested_raw = paths["work_root"] / f"nested-{raw_key}"
                nested_raw.mkdir(exist_ok=True)
                paths[raw_key] = nested_raw
                _write_config(self.config, **paths)
                with self.assertRaisesRegex(PathConfigError, "must not equal or be inside"):
                    load_project_paths(self.config)

    def test_rejects_symlink_alias_of_raw_root(self) -> None:
        work_alias = self.root / "work-alias"
        work_alias.symlink_to(self.roots["ai_hub_raw_root"], target_is_directory=True)
        self.roots["work_root"] = work_alias
        _write_config(self.config, **self.roots)
        with self.assertRaisesRegex(PathConfigError, "must not equal or be inside"):
            load_project_paths(self.config)


if __name__ == "__main__":
    unittest.main()
