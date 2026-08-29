from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scripts.run_phase7_ablation import parse_args
from src.etri_embedding import (encoder_a_training_provenance, make_cache_payload,
    make_encoder_provenance, uniform_frame_indices)
from src.etri_stage_b import (EXPERIMENTS, EtriEmbeddingDataset, GruClassifier,
    MeanPoolingClassifier, StageBError, _validate_selected_embedding, aggregate_fold_results,
    build_model, load_stage_b_checkpoint, load_stage_b_config, make_oof_rows,
    model_config_for, save_stage_b_checkpoint, stage_b_metrics, validate_experiment_fairness,
    validate_fold_contract, validate_oof)
from src.stage_a import CLASS_NAMES, CLASS_TO_INDEX


CONFIG_PATH = Path("configs/phase7_ablation.yaml")


def rows_for_folds() -> list[dict]:
    rows = []
    for fold in range(5):
        for take in range(2):
            rows.append({"dataset": "etri", "participant": f"P20{fold}", "action": "A003",
                "take": f"G{take:03d}", "height": "H070", "fold": fold,
                "target_class": CLASS_NAMES[(fold + take) % 3], "pilot_selected": True,
                "valid": True, "roi_status": "pending", "rgb_frame_count": 120})
    return rows


def payload_for(row: dict, phase7_config=None) -> dict:
    from src.etri_embedding import load_embedding_config
    phase6 = load_embedding_config(Path("configs/etri_embedding.yaml")); training = encoder_a_training_provenance()
    a = make_encoder_provenance(row, "imagenet_only", "a", phase6, "roi", "stage-a", 224, training)
    b = make_encoder_provenance(row, "ai_hub_fine_tuned", "b", phase6, "roi", "stage-a", 224,
        {"source": "ai_hub_stage_a", "fine_tuning_mode": "last_n_blocks",
         "frozen_batchnorm_policy": "freeze_running_stats"})
    return make_cache_payload(uniform_frame_indices(120), torch.randn(64, 1024), torch.randn(64, 1024),
        a, b, {"success": 0, "partial": 64, "fallback": 0}, 120, "manifest")


class StageBTests(unittest.TestCase):
    def test_label_mapping_and_probability_order_are_canonical(self) -> None:
        self.assertEqual(CLASS_NAMES, ("복약", "음수", "기타"))
        self.assertEqual(CLASS_TO_INDEX, {"복약": 0, "음수": 1, "기타": 2})
        metrics = stage_b_metrics([0, 1, 2], torch.eye(3))
        self.assertEqual(metrics["label_order"], list(CLASS_NAMES))
        self.assertEqual(metrics["probability_columns"], ["prob_복약", "prob_음수", "prob_기타"])
        self.assertEqual(metrics["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_config_and_experiment_fairness(self) -> None:
        config = load_stage_b_config(CONFIG_PATH); validate_experiment_fairness(config)
        self.assertEqual((config.sequence_length, config.embedding_dimension), (64, 1024))
        self.assertEqual(EXPERIMENTS["A"], {"encoder_key": "encoder_a", "encoder": "imagenet_only", "stage_b": "mean"})
        self.assertEqual(EXPERIMENTS["D"], {"encoder_key": "encoder_b", "encoder": "ai_hub_fine_tuned", "stage_b": "gru"})

    def test_fold_contract_and_leakage_rejection(self) -> None:
        rows = rows_for_folds()
        for fold in range(5): validate_fold_contract(rows, fold)
        bad = copy.deepcopy(rows); bad[1]["participant"] = bad[0]["participant"]; bad[1]["fold"] = 1
        with self.assertRaisesRegex(StageBError, "multiple folds"): validate_fold_contract(bad)
        duplicate = copy.deepcopy(rows); duplicate.append(copy.deepcopy(rows[0]))
        with self.assertRaisesRegex(StageBError, "Duplicate clip"): validate_fold_contract(duplicate)
        invalid = copy.deepcopy(rows); invalid[0]["valid"] = False
        with self.assertRaisesRegex(StageBError, "non-selected/invalid"): validate_fold_contract(invalid)
        unselected = copy.deepcopy(rows); unselected[0]["pilot_selected"] = False
        with self.assertRaisesRegex(StageBError, "non-selected/invalid"): validate_fold_contract(unselected)

    def test_dataset_loads_selected_encoder_and_batches(self) -> None:
        from src.etri_embedding import deterministic_cache_name, source_clip_key
        rows, config = rows_for_folds(), load_stage_b_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory); clips = cache_root / "clips"; clips.mkdir()
            for row in rows: torch.save(payload_for(row), clips / deterministic_cache_name(source_clip_key(row)))
            dataset = EtriEmbeddingDataset(rows, cache_root, "encoder_a", config, 0, "val")
            sample = dataset[0]
            self.assertEqual(tuple(sample["embedding"].shape), (64, 1024))
            self.assertEqual(sample["target"], CLASS_TO_INDEX[sample["target_class"]])
            self.assertEqual(sample["encoder_key"], "encoder_a")
            batch = next(iter(DataLoader(dataset, batch_size=2)))
            self.assertEqual(tuple(batch["embedding"].shape), (2, 64, 1024))
            missing = clips / deterministic_cache_name(source_clip_key(rows[0])); missing.unlink()
            with self.assertRaisesRegex(StageBError, "missing"): dataset[0]

    def test_embedding_contract_mutations_are_rejected(self) -> None:
        row, config = rows_for_folds()[0], load_stage_b_config(CONFIG_PATH); original = payload_for(row)
        mutations = {
            "rank": lambda value: value["encoders"]["encoder_a"].__setitem__("embedding", torch.zeros(1024)),
            "T": lambda value: value["encoders"]["encoder_a"].__setitem__("embedding", torch.zeros(63, 1024)),
            "D": lambda value: value["encoders"]["encoder_a"].__setitem__("embedding", torch.zeros(64, 1000)),
            "NaN": lambda value: value["encoders"]["encoder_a"]["embedding"].__setitem__((0, 0), float("nan")),
            "Inf": lambda value: value["encoders"]["encoder_a"]["embedding"].__setitem__((0, 0), float("inf")),
            "encoder": lambda value: value["encoders"]["encoder_a"]["provenance"].__setitem__("encoder_type", "ai_hub_fine_tuned"),
            "clip": lambda value: value["encoders"]["encoder_a"]["provenance"].__setitem__("source_clip_key", "wrong"),
            "participant": lambda value: value["encoders"]["encoder_a"]["provenance"].__setitem__("participant", "wrong"),
            "fold": lambda value: value["encoders"]["encoder_a"]["provenance"].__setitem__("fold", 4),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                payload = copy.deepcopy(original); mutation(payload)
                with self.assertRaises(StageBError): _validate_selected_embedding(payload, row, "encoder_a", config)

    def test_mean_model_shapes_backward_and_invalid_input(self) -> None:
        for batch in (1, 4):
            model = MeanPoolingClassifier(); values = torch.randn(batch, 64, 1024)
            output = model(values); self.assertEqual(tuple(output.shape), (batch, 3)); output.sum().backward()
        with self.assertRaises(StageBError): MeanPoolingClassifier()(torch.randn(2, 63, 1024))

    def test_gru_model_shapes_backward_and_serialized_config(self) -> None:
        config = load_stage_b_config(CONFIG_PATH); model = GruClassifier(1024, 3, config.gru)
        for batch in (1, 3):
            output = model(torch.randn(batch, 64, 1024)); self.assertEqual(tuple(output.shape), (batch, 3))
            output.sum().backward(); model.zero_grad(set_to_none=True)
        self.assertEqual(model_config_for("gru", config)["hidden_size"], 128)
        with self.assertRaises(StageBError): model(torch.randn(2, 64, 1000))

    def test_metrics_macro_precision_recall_and_confusion(self) -> None:
        logits = torch.tensor([[4., 0., 0.], [0., 4., 0.], [0., 4., 0.]])
        metrics = stage_b_metrics([0, 1, 2], logits)
        self.assertAlmostEqual(metrics["macro_f1"], (1 + 2 / 3 + 0) / 3)
        self.assertEqual(metrics["per_class"]["복약"]["recall"], 1.0)
        self.assertEqual(metrics["per_class"]["음수"]["precision"], 0.5)
        self.assertEqual(metrics["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [0, 1, 0]])

    def test_oof_schema_exact_once_and_manifest_consistency(self) -> None:
        rows = rows_for_folds(); metadata = [{"clip_key": f"{r['participant']}:{r['action']}:{r['take']}:{r['height']}",
            "participant": r["participant"], "fold": r["fold"], "roi_status": "partial"} for r in rows]
        targets = [CLASS_TO_INDEX[r["target_class"]] for r in rows]; logits = torch.eye(3)[targets]
        oof = make_oof_rows(metadata, targets, logits, "A"); validate_oof(oof, rows, "A")
        self.assertTrue({"prob_medication", "prob_drinking", "prob_other"}.issubset(oof[0]))
        with self.assertRaisesRegex(StageBError, "Duplicate"): validate_oof(oof + [oof[0]], rows, "A")
        with self.assertRaisesRegex(StageBError, "missing"): validate_oof(oof[:-1], rows, "A")
        wrong = copy.deepcopy(oof); wrong[0]["participant"] = "wrong"
        with self.assertRaisesRegex(StageBError, "mismatch"): validate_oof(wrong, rows, "A")

    def test_fold_aggregation_mean_std_recall_and_oof_confusion(self) -> None:
        fold_metrics = []
        for fold in range(5):
            fold_metrics.append({"fold": fold, "macro_f1": fold / 4,
                "per_class": {name: {"recall": (fold + index) / 6} for index, name in enumerate(CLASS_NAMES)},
                "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})
        oof = [{"true_label": index, "pred_label": index} for index in range(3)]
        result = aggregate_fold_results(fold_metrics, oof)
        self.assertEqual(result["macro_f1_mean"], .5)
        self.assertAlmostEqual(result["macro_f1_std"], float(torch.tensor([0., .25, .5, .75, 1.]).std(correction=0)))
        self.assertEqual(result["class_recall"]["복약"]["mean"], 1 / 3)
        self.assertEqual(result["oof_confusion_matrix"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_checkpoint_round_trip_reproduces_prediction(self) -> None:
        config = load_stage_b_config(CONFIG_PATH); values = torch.randn(2, 64, 1024)
        for stage_b in ("mean", "gru"):
            with self.subTest(stage_b=stage_b), tempfile.TemporaryDirectory() as directory:
                model = build_model(stage_b, config).eval(); expected = model(values).detach()
                provenance = {"phase": 7, "experiment": "A" if stage_b == "mean" else "C",
                    "fold": 0, "stage_b": stage_b, "model_config": model_config_for(stage_b, config),
                    "class_mapping": CLASS_TO_INDEX, "config_path": str(CONFIG_PATH)}
                path = Path(directory) / "best.pt"; save_stage_b_checkpoint(path, model, provenance)
                loaded, restored = load_stage_b_checkpoint(path)
                self.assertEqual(restored["phase"], 7)
                self.assertTrue(torch.equal(expected, loaded(values).detach()))

    def test_cli_rejects_invalid_combinations(self) -> None:
        with self.assertRaises(SystemExit): parse_args(["--experiment", "Z"])
        with self.assertRaises(SystemExit): parse_args(["--experiment", "A", "--fold", "5"])
        with self.assertRaises(SystemExit): parse_args(["--validate-only", "--experiment", "A"])
        with self.assertRaises(SystemExit): parse_args(["--experiment", "A", "--epochs", "1"])
        valid = parse_args(["--experiment", "A", "--fold", "0", "--smoke", "--epochs", "1"])
        self.assertTrue(valid.smoke); self.assertEqual(valid.epochs, 1)


if __name__ == "__main__": unittest.main()
