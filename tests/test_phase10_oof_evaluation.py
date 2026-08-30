from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.run_phase10_raw_video_oof import checkpoint_paths
from src.phase10_oof_evaluation import (RawVideoOofError, aggregate_metrics,
    compare_cached_raw, validate_oof_universe, validate_predictions)
from src.stage_a import CLASS_NAMES


def manifest_rows() -> list[dict]:
    rows = []
    class_counts = {"복약": 59, "음수": 60, "기타": 120}
    number = 0
    for label, count in class_counts.items():
        for _ in range(count):
            fold = number % 5; participant = f"P{fold}{number % 6}"
            rows.append({"participant": participant, "action": f"A{number:03}", "take": "G001",
                "height": "H070", "fold": fold, "target_class": label,
                "pilot_selected": True, "valid": True})
            number += 1
    # Make participant identity depend only on fold, preserving disjointness.
    for index, row in enumerate(rows): row["participant"] = f"P{row['fold']}{index % 6}"
    return rows


def prediction_rows(rows: list[dict]) -> list[dict]:
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}; result = []
    for row in rows:
        index = mapping[row["target_class"]]
        result.append({"clip_key": f"{row['participant']}:{row['action']}:{row['take']}:{row['height']}",
            "participant": row["participant"], "fold": row["fold"], "ground_truth_index": index,
            "ground_truth_label": row["target_class"], "prediction_index": index,
            "prob_medication": .8 if index == 0 else .1, "prob_drinking": .8 if index == 1 else .1,
            "prob_other": .8 if index == 2 else .1, "roi_success": 0, "roi_partial": 64,
            "roi_fallback": 0, "inference_success": True})
    return result


class Phase10OofEvaluationTests(unittest.TestCase):
    def test_fold_checkpoint_mapping(self) -> None:
        paths = checkpoint_paths(Path("/work"))
        self.assertEqual(set(paths), set(range(5)))
        for fold, path in paths.items(): self.assertIn(f"phase7_D_fold{fold}", str(path))

    def test_universe_exact_once_and_participant_leakage(self) -> None:
        rows = manifest_rows(); summary = validate_oof_universe(rows)
        self.assertEqual(summary["total_samples"], 239)
        self.assertTrue(all(not value["overlap"] for value in summary["leakage"].values()))
        bad = [dict(row) for row in rows]; bad[1]["participant"] = bad[0]["participant"]
        with self.assertRaisesRegex(RawVideoOofError, "leakage"):
            validate_oof_universe(bad)

    def test_duplicate_missing_wrong_fold_and_probability_are_rejected(self) -> None:
        rows = manifest_rows(); predictions = prediction_rows(rows)
        self.assertEqual(validate_predictions(predictions, rows), {"duplicates": 0, "missing": 0, "failed": 0})
        for changed in (predictions[:-1], predictions + [dict(predictions[0])]):
            with self.assertRaises(RawVideoOofError): validate_predictions(changed, rows)
        wrong = [dict(row) for row in predictions]; wrong[0]["fold"] = (wrong[0]["fold"] + 1) % 5
        with self.assertRaisesRegex(RawVideoOofError, "Wrong fold"):
            validate_predictions(wrong, rows)
        wrong = [dict(row) for row in predictions]; wrong[0]["prob_other"] += .2
        with self.assertRaisesRegex(RawVideoOofError, "Probability"):
            validate_predictions(wrong, rows)

    def test_metrics_class_and_confusion_order(self) -> None:
        metrics = aggregate_metrics(prediction_rows(manifest_rows()))
        self.assertEqual(metrics["overall"]["label_order"], list(CLASS_NAMES))
        self.assertEqual(metrics["overall"]["macro_f1"], 1.0)
        self.assertEqual(metrics["overall"]["confusion_matrix"], [[59, 0, 0], [0, 60, 0], [0, 0, 120]])
        self.assertEqual(set(metrics["fold_metrics"]), {"0", "1", "2", "3", "4"})

    def test_cached_raw_alignment_and_probability_difference(self) -> None:
        raw = prediction_rows(manifest_rows())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cached.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                fields = ("clip_key", "participant", "fold", "true_label", "pred_label",
                          "prob_medication", "prob_drinking", "prob_other")
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
                for row in raw:
                    writer.writerow({"clip_key": row["clip_key"], "participant": row["participant"],
                        "fold": row["fold"], "true_label": row["ground_truth_index"],
                        "pred_label": row["prediction_index"], "prob_medication": row["prob_medication"],
                        "prob_drinking": row["prob_drinking"], "prob_other": row["prob_other"]})
            summary, comparison = compare_cached_raw(raw, path)
        self.assertTrue(summary["sample_set_match"]); self.assertEqual(len(comparison), 239)
        self.assertEqual(summary["prediction_agreement_rate"], 1.0)
        self.assertEqual(summary["probability_max_absolute_difference"], 0.0)
        self.assertEqual(summary["final_consistency_verdict"], "EXACT_MATCH")


if __name__ == "__main__": unittest.main()
