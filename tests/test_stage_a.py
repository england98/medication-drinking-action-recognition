from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

from src.roi_preflight import Detection, load_roi_config
from src.stage_a import (CLASS_TO_INDEX, StageAFrameDataset, StageAMobileNetV3,
    build_transform, classification_metrics, evaluate_predictions, load_checkpoint,
    load_stage_a_config, save_checkpoint, trainable_parameter_summary)
from src.stage_a_training import is_better_checkpoint
from scripts.run_stage_a import _provenance, best_final_metrics, make_run_id, resolve_effective_config
CONFIG = load_roi_config(Path("configs/roi_preflight.yaml"))


class Detector:
    def __init__(self, detection=Detection()): self.detection = detection
    def detect(self, image): return self.detection
    def close(self): pass


def manifest_rows(root_key="ai_hub_raw_root"):
    rows = []
    for split, actor, video, target in (("train", "a", "v1", "복약"), ("val", "b", "v2", "음수")):
        rows.append({"dataset": "ai_hub", "root_key": root_key, "relative_path": video + ".json",
            "frame_relative_paths": [f"{video}_{i}.jpg" for i in range(3)], "video_id": video,
            "actor": actor, "viewpoint": 3, "target_class": target, "split": split,
            "pilot_selected": True, "valid": True})
    return rows


class StageATests(unittest.TestCase):
    def _dataset(self, rows):
        return StageAFrameDataset(rows, "train", {"ai_hub_raw_root": Path("/tmp")}, CONFIG,
                                  Detector(), lambda image: torch.from_numpy(image))

    def test_manifest_negative_invariants_fail(self):
        overlap = manifest_rows(); overlap[1]["actor"] = overlap[0]["actor"]
        with self.assertRaisesRegex(RuntimeError, "overlap"): self._dataset(overlap)
        viewpoint = manifest_rows(); viewpoint[0]["viewpoint"] = 1
        with self.assertRaisesRegex(RuntimeError, "viewpoint"): self._dataset(viewpoint)
        invalid = manifest_rows(); invalid[0]["valid"] = False
        with self.assertRaisesRegex(RuntimeError, "must be valid"): self._dataset(invalid)
        frames = manifest_rows(); frames[0]["frame_relative_paths"] = frames[0]["frame_relative_paths"][:2]
        with self.assertRaisesRegex(RuntimeError, "three frames"): self._dataset(frames)

    def test_effective_cli_override_and_run_id(self):
        config = load_stage_a_config(Path("configs/stage_a.yaml"))
        effective = resolve_effective_config(config, 8, 10, torch.device("cpu"))
        self.assertEqual((effective["effective_batch_size"], effective["effective_epochs"]), (8, 10))
        self.assertEqual(effective["cli_overrides"], {"batch_size": 8, "epochs": 10})
        now = datetime(2026, 8, 30, tzinfo=timezone.utc)
        self.assertEqual(make_run_id(effective, now), make_run_id(effective, now))
        with self.assertRaisesRegex(RuntimeError, "positive"): resolve_effective_config(config, -1, None, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"; roi = Path(directory) / "roi.yaml"
            manifest.write_text("{}\n", encoding="utf-8"); roi.write_text("phase4: {}\n", encoding="utf-8")
            summary = {"total_parameter_count": 1, "trainable_parameter_count": 1,
                       "trainable_ratio": 1.0, "trainable_modules": ["classifier"]}
            provenance = _provenance(config, effective, manifest, Path.cwd(), roi, "run", "run", summary, None)
            self.assertEqual(provenance["model_config"]["effective_batch_size"], 8)
            self.assertEqual(provenance["effective_epochs"], 10)
            self.assertEqual(provenance["cli_overrides"], {"batch_size": 8, "epochs": 10})

    def test_num_workers_other_than_zero_fails(self):
        source = Path("configs/stage_a.yaml").read_text(encoding="utf-8").replace("num_workers: 0", "num_workers: 1")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage_a.yaml"; path.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "num_workers=0"): load_stage_a_config(path)

    def test_validation_preprocessing_is_deterministic(self):
        config = load_stage_a_config(Path("configs/stage_a.yaml")); transform = build_transform(config, False)
        image = np.random.default_rng(42).integers(0, 255, (80, 100, 3), dtype=np.uint8)
        self.assertTrue(torch.equal(transform(image), transform(image)))

    def test_dataset_filter_metadata_roi_and_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for row in manifest_rows():
                for path in row["frame_relative_paths"]: cv2.imwrite(str(root / path), np.zeros((30, 40, 3), np.uint8))
            transform = lambda image: torch.from_numpy(image.copy()).permute(2, 0, 1).float()
            train = StageAFrameDataset(manifest_rows(), "train", {"ai_hub_raw_root": root}, CONFIG, Detector(), transform)
            self.assertEqual(len(train), 3); sample = train[0]
            self.assertEqual(sample["target"], CLASS_TO_INDEX["복약"]); self.assertEqual(sample["roi_status"], "fallback")
            partial = StageAFrameDataset(manifest_rows(), "val", {"ai_hub_raw_root": root}, CONFIG,
                Detector(Detection(face_points=((.2, .2), (.6, .6)))), transform)[0]
            self.assertEqual(partial["roi_status"], "partial")

    def test_model_shapes_encoder_compatibility_and_backward(self):
        first = StageAMobileNetV3(pretrained=False); second = StageAMobileNetV3(pretrained=False)
        values = torch.randn(2, 3, 64, 64); logits = first(values); embedding_a = first.extract_features(values)
        embedding_b = second.extract_features(values)
        self.assertEqual(tuple(logits.shape), (2, 3)); self.assertEqual(tuple(embedding_a.shape), (2, 1024))
        self.assertEqual(embedding_a.shape, embedding_b.shape); logits.sum().backward()

    def test_frozen_bn_stats_and_trainable_scope(self):
        model = StageAMobileNetV3(pretrained=False); model.configure_fine_tuning("last_n_blocks", 2)
        frozen_bn = next(module for block in list(model.features.children())[:-2]
                         for module in block.modules() if isinstance(module, torch.nn.BatchNorm2d))
        trainable_bn = next(module for block in list(model.features.children())[-2:]
                            for module in block.modules() if isinstance(module, torch.nn.BatchNorm2d))
        before = frozen_bn.running_mean.clone(); model.train()
        self.assertFalse(frozen_bn.training); self.assertTrue(trainable_bn.training)
        model(torch.randn(4, 3, 64, 64))
        self.assertTrue(torch.equal(before, frozen_bn.running_mean))
        summary = trainable_parameter_summary(model, "last_n_blocks", 2)
        self.assertEqual(summary["trainable_modules"], ["features.11", "features.12", "embedding_head", "classifier"])
        self.assertLess(summary["trainable_parameter_count"], summary["total_parameter_count"])

    def test_evaluation_frame_and_video_mean_logits(self):
        targets = [0, 0, 0, 1, 1, 1]
        logits = torch.tensor([[4., 0., 0.], [0., 3., 0.], [4., 0., 0.],
                               [0., 4., 0.], [2., 0., 0.], [0., 4., 0.]])
        result = evaluate_predictions(targets, logits, ["a"] * 3 + ["b"] * 3)
        self.assertEqual(result["video"]["macro_f1"], 2 / 3)
        self.assertEqual(len(result["frame"]["confusion_matrix"]), 3)
        self.assertIn("precision", classification_metrics(targets, logits)["per_class"]["복약"])

    def test_checkpoint_round_trip_provenance_and_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"; model = StageAMobileNetV3(pretrained=False)
            provenance = {"backbone": "mobilenet_v3_small", "class_mapping": CLASS_TO_INDEX,
                          "embedding_dimension": 1024, "best_epoch": 1}
            save_checkpoint(path, model, provenance); loaded, restored = load_checkpoint(path)
            self.assertEqual(restored["best_epoch"], 1)
            loaded.eval()
            self.assertEqual(tuple(loaded(torch.randn(1, 3, 64, 64)).shape), (1, 3))
            bad = {**provenance, "backbone": "incompatible"}; save_checkpoint(path, model, bad)
            with self.assertRaisesRegex(RuntimeError, "incompatible"): load_checkpoint(path)

    def test_best_monitor_tie_break_and_best_final_metrics(self):
        self.assertTrue(is_better_checkpoint(.7, .5, .6, .4))
        self.assertTrue(is_better_checkpoint(.7, .3, .7, .4))
        self.assertFalse(is_better_checkpoint(.7, .4, .7, .4))
        def result(frame, video):
            classes = {name: {"precision": .1, "recall": .2, "f1": .15} for name in CLASS_TO_INDEX}
            return {"metrics": {"frame": {"macro_f1": frame, "per_class": classes},
                                "video": {"macro_f1": video, "per_class": classes}}}
        metrics = best_final_metrics(2, result(.6, .7), 4, result(.5, .55), .7)
        self.assertEqual(metrics["best_epoch"], 2.0)
        self.assertEqual(metrics["best_val_video_macro_f1"], .7)
        self.assertEqual(metrics["final_val_video_macro_f1"], .55)
        self.assertIn("best_val_frame_0_recall", metrics)


if __name__ == "__main__": unittest.main()
