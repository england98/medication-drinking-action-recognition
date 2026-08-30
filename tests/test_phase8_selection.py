from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from src.etri_stage_b import EXPERIMENTS, validate_fold_contract
from src.phase8_selection import (Phase8SelectionError, _assert_close, _validate_oof_rows,
    build_selection_artifact, effect_analysis, rank_experiments, validate_cross_experiment_fairness,
    validate_intended_folds, write_selection_artifact)


def manifest_rows() -> list[dict]:
    return [{"dataset": "etri", "participant": f"P20{fold}", "action": "A003", "take": "G001",
        "height": "H070", "fold": fold, "target_class": "복약", "pilot_selected": True, "valid": True}
        for fold in range(5)]


def result(experiment: str, macro: float, macro_std: float = .1,
           medication: float = .2, drinking: float = .3, other: float = .8) -> dict:
    provenance = {"T": 64, "D": 1024, "class_mapping": {"복약": 0, "음수": 1, "기타": 2},
        "manifest_sha256": "a" * 64, "roi_config_sha256": "b" * 64,
        "stage_a_config_sha256": "c" * 64, "sampling": "fixed_uniform",
        "normalization_identity": "torchvision_imagenet_default", "seed_policy": "base_seed_plus_fold",
        "base_seed": 42, "loss": "cross_entropy", "optimizer": "adamw", "learning_rate": .0001,
        "weight_decay": .0001, "batch_size": 16, "configured_epochs": 15,
        "best_monitor": "val_macro_f1", "git_commit_hash": "60c", "git_dirty": False,
        "git_diff_sha256": None, "encoder_cache_identifier":
            "torchvision:weights" if experiment in "AC" else "stage-a-checkpoint",
        "model_config": {"type": EXPERIMENTS[experiment]["stage_b"]}}
    return {"experiment": experiment, "definition": dict(EXPERIMENTS[experiment]), "provenance": provenance,
        "aggregate": {"macro_f1_mean": macro, "macro_f1_std": macro_std,
            "class_recall": {"복약": {"mean": medication, "std": .01},
                "음수": {"mean": drinking, "std": .02}, "기타": {"mean": other, "std": .03}}},
        "evidence_hashes": {"folds": {}, "summary": "d" * 64, "oof": "e" * 64}}


def four_results(values=(.4, .45, .5, .55)) -> dict:
    return {experiment: result(experiment, value) for experiment, value in zip("ABCD", values)}


class Phase8SelectionTests(unittest.TestCase):
    def test_complete_fold_contract_and_missing_duplicate_rejection(self) -> None:
        validate_intended_folds([0, 1, 2, 3, 4])
        for folds in ([0, 1, 2, 3], [0, 1, 2, 3, 3]):
            with self.subTest(folds=folds), self.assertRaises(Phase8SelectionError): validate_intended_folds(folds)

    def test_primary_metric_ranking_selects_highest(self) -> None:
        ranking, tie = rank_experiments(four_results())
        self.assertEqual(ranking, ["D", "C", "B", "A"]); self.assertFalse(tie)

    def test_exact_tie_uses_medication_then_drinking_then_std(self) -> None:
        values = four_results((.5, .5, .4, .3))
        values["A"]["aggregate"]["class_recall"]["복약"]["mean"] = .4
        values["B"]["aggregate"]["class_recall"]["복약"]["mean"] = .3
        self.assertEqual(rank_experiments(values)[0][0], "A")
        values["B"]["aggregate"]["class_recall"]["복약"]["mean"] = .4
        values["B"]["aggregate"]["class_recall"]["음수"]["mean"] = .5
        self.assertEqual(rank_experiments(values)[0][0], "B")
        values["A"]["aggregate"]["class_recall"]["음수"]["mean"] = .5
        values["A"]["aggregate"]["macro_f1_std"] = .05
        self.assertEqual(rank_experiments(values)[0][0], "A")

    def test_exact_tie_uses_model_simplicity_after_metrics(self) -> None:
        values = four_results((.5, .3, .5, .2))
        values["A"]["aggregate"] = copy.deepcopy(values["C"]["aggregate"])
        ranking, tie = rank_experiments(values)
        self.assertEqual(ranking[0], "A"); self.assertTrue(tie)

    def test_oof_duplicate_missing_and_manifest_mismatch_rejection(self) -> None:
        manifest = manifest_rows(); rows = []
        for source in manifest:
            rows.append({"clip_key": f"{source['participant']}:A003:G001:H070", "participant": source["participant"],
                "fold": str(source["fold"]), "true_label": "0", "pred_label": "0",
                "prob_medication": "1", "prob_drinking": "0", "prob_other": "0", "experiment": "A"})
        _validate_oof_rows(rows, manifest, "A")
        with self.assertRaisesRegex(Phase8SelectionError, "Duplicate"): _validate_oof_rows(rows + [rows[0]], manifest, "A")
        with self.assertRaisesRegex(Phase8SelectionError, "missing/extra"): _validate_oof_rows(rows[:-1], manifest, "A")
        changed = copy.deepcopy(rows); changed[0]["participant"] = "wrong"
        with self.assertRaisesRegex(Phase8SelectionError, "mismatch"): _validate_oof_rows(changed, manifest, "A")

    def test_participant_leakage_rejection_is_preserved(self) -> None:
        rows = manifest_rows(); rows.append({**rows[0], "take": "G002", "fold": 1})
        with self.assertRaisesRegex(Exception, "multiple folds"): validate_fold_contract(rows)

    def test_cross_experiment_fairness_rejects_manifest_difference(self) -> None:
        values = four_results(); validate_cross_experiment_fairness(values)
        values["D"]["provenance"]["manifest_sha256"] = "x" * 64
        with self.assertRaisesRegex(Phase8SelectionError, "common provenance"): validate_cross_experiment_fairness(values)

    def test_malformed_metric_input_is_rejected(self) -> None:
        _assert_close({"macro_f1": .5}, {"macro_f1": .5}, "metric")
        with self.assertRaisesRegex(Phase8SelectionError, "Metric mismatch"):
            _assert_close({"macro_f1": .4}, {"macro_f1": .5}, "metric")

    def test_effect_analysis_values(self) -> None:
        effects = effect_analysis(four_results((.4, .45, .5, .55)))
        self.assertAlmostEqual(effects["B_minus_A"]["macro_f1_mean"], .05)
        self.assertAlmostEqual(effects["D_minus_C"]["macro_f1_mean"], .05)
        self.assertAlmostEqual(effects["D_minus_A"]["macro_f1_mean"], .15)

    def test_selection_artifact_schema_and_phase9_boundary(self) -> None:
        values = four_results(); artifact = build_selection_artifact(values, manifest_rows(),
            Path("configs/phase7_ablation.yaml"), "checkpoints/phase7_ablation", "checkpoints/stage_a/best.pt")
        selected = artifact["phase8"]
        self.assertEqual(selected["selected_experiment"], "D")
        self.assertEqual(selected["encoder"]["type"], "ai_hub_fine_tuned")
        self.assertEqual(selected["stage_b"]["type"], "gru")
        self.assertFalse(selected["selection"]["secondary_tie_break_used"])
        self.assertTrue(selected["phase9"]["stage_b_reinitialize"])
        self.assertFalse(selected["phase9"]["deployment_checkpoint_created"])

    def test_selection_artifact_write_is_idempotent(self) -> None:
        artifact = {"phase8": {"status": "complete", "selected_experiment": "D"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.yaml"
            self.assertEqual(write_selection_artifact(path, artifact), "CREATED")
            first = path.read_bytes(); self.assertEqual(write_selection_artifact(path, artifact), "UNCHANGED")
            self.assertEqual(first, path.read_bytes()); self.assertEqual(yaml.safe_load(first), artifact)
            with self.assertRaisesRegex(Phase8SelectionError, "differs"):
                write_selection_artifact(path, {"phase8": {"status": "blocked"}})


if __name__ == "__main__": unittest.main()
