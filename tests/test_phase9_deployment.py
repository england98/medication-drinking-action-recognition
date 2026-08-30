from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scripts.run_phase9_deployment import parse_args
from src.etri_embedding import deterministic_cache_name, source_clip_key
from src.etri_stage_b import (EtriEmbeddingDataset, StageBError, build_model,
    load_stage_b_checkpoint, load_stage_b_config, model_config_for, save_stage_b_checkpoint)
from src.phase9_deployment import (full_pilot_diagnostics, load_phase9_selection,
                                   phase9_checkpoint_provenance, validate_selected_cache_identity)
from src.stage_a import CLASS_NAMES, CLASS_TO_INDEX
from tests.test_etri_stage_b import payload_for, rows_for_folds


CONFIG = Path("configs/phase7_ablation.yaml")
SELECTION = Path("configs/phase8_selected_model.yaml")


class Phase9DeploymentTests(unittest.TestCase):
    def _cache(self, root: Path, rows: list[dict]) -> Path:
        clips = root / "clips"; clips.mkdir()
        for row in rows:
            torch.save(payload_for(row), clips / deterministic_cache_name(source_clip_key(row)))
        return root

    def test_full_pilot_loader_has_no_fold_filter_and_preserves_cv_mode(self) -> None:
        rows, config = rows_for_folds(), load_stage_b_config(CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            root = self._cache(Path(directory), rows)
            full = EtriEmbeddingDataset(rows, root, "encoder_b", config, None, "full")
            diagnostic = full_pilot_diagnostics(rows, full)
            self.assertEqual(len(full), len(rows)); self.assertFalse(diagnostic["fold_filter_applied"])
            self.assertEqual(diagnostic["fold_counts"], {str(fold): 2 for fold in range(5)})
            self.assertEqual(diagnostic["cache_encoder_key"], "encoder_b")
            train = EtriEmbeddingDataset(rows, root, "encoder_b", config, 0, "train")
            val = EtriEmbeddingDataset(rows, root, "encoder_b", config, 0, "val")
            self.assertEqual((len(train), len(val)), (8, 2))
            with self.assertRaisesRegex(StageBError, "must not apply"):
                EtriEmbeddingDataset(rows, root, "encoder_b", config, 0, "full")

    def test_full_pilot_rejects_unselected_invalid_duplicate_and_missing_cache(self) -> None:
        rows, config = rows_for_folds(), load_stage_b_config(CONFIG)
        for field in ("pilot_selected", "valid"):
            bad = copy.deepcopy(rows); bad[0][field] = False
            with tempfile.TemporaryDirectory() as directory:
                root = self._cache(Path(directory), bad)
                with self.assertRaises(StageBError): EtriEmbeddingDataset(bad, root, "encoder_b", config, None, "full")
        duplicate = copy.deepcopy(rows) + [copy.deepcopy(rows[0])]
        with tempfile.TemporaryDirectory() as directory:
            root = self._cache(Path(directory), rows)
            with self.assertRaisesRegex(StageBError, "Duplicate"):
                EtriEmbeddingDataset(duplicate, root, "encoder_b", config, None, "full")
        with tempfile.TemporaryDirectory() as directory:
            root = self._cache(Path(directory), rows); dataset = EtriEmbeddingDataset(rows, root, "encoder_b", config, None, "full")
            (root / "clips" / deterministic_cache_name(source_clip_key(rows[0]))).unlink()
            with self.assertRaisesRegex(StageBError, "missing"): dataset[0]

    def test_selection_locks_experiment_d_and_phase7_policy(self) -> None:
        config = load_stage_b_config(CONFIG); selected = load_phase9_selection(SELECTION, config)
        self.assertEqual(selected["selected_experiment"], "D")
        self.assertEqual(selected["encoder"]["cache_key"], "encoder_b")
        self.assertTrue(selected["encoder"]["frozen"])
        self.assertEqual(selected["stage_b"]["model_config"], model_config_for("gru", config))
        self.assertEqual((config.sequence_length, config.embedding_dimension, config.epochs), (64, 1024, 15))
        self.assertEqual((config.learning_rate, config.optimizer, config.batch_size, config.seed), (.0001, "adamw", 16, 42))
        self.assertEqual(CLASS_TO_INDEX, {"복약": 0, "음수": 1, "기타": 2})

    def test_selection_mutations_fail_fast(self) -> None:
        config = load_stage_b_config(CONFIG)
        source = SELECTION.read_text(encoding="utf-8")
        mutations = (source.replace("selected_experiment: D", "selected_experiment: C", 1),
                     source.replace("cache_key: encoder_b", "cache_key: encoder_a", 1),
                     source.replace("embedding_dimension: 1024", "embedding_dimension: 1000", 1),
                     source.replace("복약: 0", "복약: 2", 1))
        with tempfile.TemporaryDirectory() as directory:
            for index, value in enumerate(mutations):
                path = Path(directory) / f"bad-{index}.yaml"; path.write_text(value, encoding="utf-8")
                with self.subTest(index=index), self.assertRaises(StageBError): load_phase9_selection(path, config)

    def test_selected_encoder_cache_identity_must_match(self) -> None:
        selection = load_phase9_selection(SELECTION, load_stage_b_config(CONFIG))
        identifier = selection["encoder"]["identifier"]
        validate_selected_cache_identity(selection, {"encoder_b": {"checkpoint_identifier": identifier}})
        with self.assertRaisesRegex(StageBError, "cache identity"):
            validate_selected_cache_identity(selection, {"encoder_b": {"checkpoint_identifier": "wrong"}})

    def test_phase9_checkpoint_round_trip_and_provenance(self) -> None:
        config = load_stage_b_config(CONFIG); selection = load_phase9_selection(SELECTION, config)
        diagnostics = {"total_samples": 10, "participant_count": 5,
            "class_counts": {name: 1 for name in CLASS_NAMES}, "fold_counts": {str(i): 2 for i in range(5)}}
        summary = {"roi_config_sha256": "r", "stage_a_config_sha256": "a", "sampling": "fixed_uniform"}
        provenance = phase9_checkpoint_provenance(selection_path=SELECTION, selection=selection,
            config_path=CONFIG, config=config, manifest=Path("manifest.jsonl"), cache_root=Path("cache"),
            summary=summary, diagnostics=diagnostics, project_root=Path("."), completed_epochs=15,
            smoke=False, mlflow_run_id="run", git={"git_commit_hash": "abc", "git_dirty": False})
        required = {"phase", "role", "selected_experiment", "training_scope", "encoder_frozen",
            "model_config", "manifest_sha256", "sample_counts", "selection_sha256", "epoch_policy"}
        self.assertFalse(required - set(provenance)); self.assertEqual(provenance["phase"], 9)
        values = torch.randn(2, 64, 1024); model = build_model("gru", config).eval()
        expected = model(values).detach()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deployment_check.pt"; save_stage_b_checkpoint(path, model, provenance)
            loaded, restored = load_stage_b_checkpoint(path)
            self.assertTrue(torch.equal(expected, loaded(values).detach()))
            self.assertEqual((restored["role"], restored["selected_experiment"]), ("deployment_check", "D"))

    def test_encoder_cache_only_forward_backward_shape(self) -> None:
        config = load_stage_b_config(CONFIG); model = build_model("gru", config)
        parameters = dict(model.named_parameters())
        self.assertTrue(parameters); self.assertFalse(any("encoder" in name for name in parameters))
        output = model(torch.randn(2, 64, 1024)); self.assertEqual(tuple(output.shape), (2, 3))
        output.sum().backward()

    def test_cli_modes_are_explicit(self) -> None:
        self.assertTrue(parse_args(["--dry-run"]).dry_run)
        self.assertTrue(parse_args(["--train", "--smoke", "--no-mlflow"]).smoke)
        self.assertEqual(parse_args(["--verify-checkpoint", "x.pt"]).verify_checkpoint, Path("x.pt"))
        with self.assertRaises(SystemExit): parse_args([])
        with self.assertRaises(SystemExit): parse_args(["--dry-run", "--smoke"])


if __name__ == "__main__": unittest.main()
