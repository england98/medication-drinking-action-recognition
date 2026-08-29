from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.roi_preflight import (
    Detection, RepresentativeSample, RoiPreflightConfig, RoiPreflightError,
    aggregate_report, calculate_bbox, calculate_contextual_bbox, extract_roi, load_sample_image,
    read_manifest, run_preflight, select_representatives, write_report,
)


CONFIG = RoiPreflightConfig(
    selection_seed=42, representative_per_stratum=1, ai_hub_frame_index=1,
    etri_frame_positions=(0.5,), padding_ratio=0.25, minimum_width_px=20,
    minimum_height_px=20, fallback_mode="full_frame", allow_partial=True,
    hand_only_min_width_ratio=0.55, hand_only_min_height_ratio=0.70,
    hand_only_vertical_anchor=0.75, face_only_min_width_ratio=0.55,
    face_only_min_height_ratio=0.75, face_only_vertical_anchor=0.22,
    pose_only_min_width_ratio=0.60, pose_only_min_height_ratio=0.80,
    pose_only_vertical_anchor=0.42,
    num_faces=1, num_hands=2, num_poses=1, minimum_face_detection_confidence=0.5,
    minimum_face_presence_confidence=0.5, minimum_hand_detection_confidence=0.5,
    minimum_hand_presence_confidence=0.5, face_model_relative_path="cache/face.task",
    minimum_pose_detection_confidence=0.5, minimum_pose_presence_confidence=0.5,
    hand_model_relative_path="cache/hand.task", face_model_url="https://example/face",
    pose_model_relative_path="cache/pose.task", hand_model_url="https://example/hand",
    pose_model_url="https://example/pose", output_relative_path="roi_preflight",
)


def row(dataset: str, index: int, target: str, hard: bool = False) -> dict:
    value = {
        "dataset": dataset, "root_key": f"{dataset}_raw_root", "relative_path": f"{index}.mp4",
        "target_class": target, "is_hard_negative": hard, "valid": True,
        "pilot_selected": True,
    }
    if dataset == "ai_hub":
        value.update(video_id=str(index), viewpoint=3,
                     frame_relative_paths=[f"{index}_0.jpg", f"{index}_1.jpg", f"{index}_2.jpg"])
    else:
        value.update(batch="B", participant="P201", action=f"A{index:03d}", take="G001", height="H070")
    return value


class FakeDetector:
    def __init__(self, detection: Detection): self.detection, self.closed = detection, False
    def detect(self, image_bgr: np.ndarray) -> Detection: return self.detection
    def close(self) -> None: self.closed = True


class RoiPreflightTests(unittest.TestCase):
    def test_bbox_padding_clamp_minimum_and_invalid(self) -> None:
        self.assertEqual(calculate_bbox(((0.0, 0.0), (0.1, 0.1)), 100, 80, 0.5, 20, 20), (0, 0, 20, 20))
        self.assertEqual(calculate_bbox(((0.9, 0.9), (1.1, 1.1)), 100, 80, 0.25, 10, 10), (69, 56, 100, 80))
        self.assertIsNone(calculate_bbox(((0.5, 0.5),), 100, 80, 0.2, 10, 10))

    def test_success_partial_and_fallback_never_empty(self) -> None:
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        face = ((0.3, 0.3), (0.6, 0.6)); hand = ((0.55, 0.55), (0.8, 0.8))
        success = extract_roi(image, Detection(face, hand), CONFIG)
        partial = extract_roi(image, Detection(face, ()), CONFIG)
        fallback = extract_roi(image, Detection(), CONFIG)
        self.assertEqual((success.roi_status, partial.roi_status, fallback.roi_status), ("success", "partial", "fallback"))
        self.assertTrue(all(result.crop.size for result in (success, partial, fallback)))
        self.assertEqual(fallback.bbox, (0, 0, 120, 100))

    def test_pose_only_is_partial_and_uses_contextual_extent(self) -> None:
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        pose = tuple((0.4 + i * 0.01, 0.2 + i * 0.02) for i in range(10))
        result = extract_roi(image, Detection(pose_points=pose), CONFIG)
        self.assertEqual(result.roi_status, "partial")
        self.assertEqual(result.reason, "pose_only")
        self.assertGreaterEqual(result.bbox[2] - result.bbox[0], 72)
        self.assertGreaterEqual(result.bbox[3] - result.bbox[1], 80)

    def test_partial_context_expands_hand_upward_and_face_downward(self) -> None:
        hand = calculate_contextual_bbox(((0.48, 0.62), (0.55, 0.70)), 1000, 800, 0.3, 64, 64, 0.55, 0.70, 0.75)
        face = calculate_contextual_bbox(((0.42, 0.12), (0.58, 0.30)), 1000, 800, 0.3, 64, 64, 0.55, 0.75, 0.22)
        self.assertIsNotNone(hand); self.assertIsNotNone(face)
        assert hand is not None and face is not None
        self.assertGreaterEqual(hand[2] - hand[0], 550); self.assertGreaterEqual(hand[3] - hand[1], 560)
        self.assertLess(hand[1], int(0.62 * 800))
        self.assertGreaterEqual(face[2] - face[0], 550); self.assertGreaterEqual(face[3] - face[1], 600)
        self.assertGreater(face[3], int(0.30 * 800))

    def test_selection_is_deterministic_filtered_and_stratified(self) -> None:
        ai, etri = [], []
        specs = (("복약", False), ("음수", False), ("기타", True), ("기타", False))
        for index, (target, hard) in enumerate(specs, 1):
            ai.append(row("ai_hub", index, target, hard)); etri.append(row("etri", index, target, hard))
        excluded = row("ai_hub", 99, "복약"); excluded["pilot_selected"] = False; ai.append(excluded)
        first = select_representatives(ai, etri, CONFIG)
        self.assertEqual(first, select_representatives(ai, etri, CONFIG))
        self.assertEqual(len(first), 8)
        self.assertNotIn("99", {sample.source_sample_key.split(":")[0] for sample in first})
        self.assertEqual({(s.dataset, s.target_class, s.is_hard_negative) for s in first},
                         {(d, t, h) for d in ("ai_hub", "etri") for t, h in specs})

    def test_image_video_loading_and_missing_file_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); image = np.full((32, 40, 3), 127, np.uint8)
            cv2.imwrite(str(root / "frame.jpg"), image)
            ai = RepresentativeSample("ai_hub", "v:1", "ai_hub_raw_root", "meta.json", "복약", False, "frame.jpg")
            self.assertEqual(load_sample_image(ai, {"ai_hub_raw_root": root}).shape[:2], (32, 40))
            video_path = root / "clip.avi"; writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (40, 32))
            for value in (0, 80, 160): writer.write(np.full((32, 40, 3), value, np.uint8))
            writer.release()
            etri = RepresentativeSample("etri", "p@0.5", "etri_raw_root", "clip.avi", "음수", False, frame_position=0.5)
            self.assertEqual(load_sample_image(etri, {"etri_raw_root": root}).shape[:2], (32, 40))
            missing = RepresentativeSample("ai_hub", "bad", "ai_hub_raw_root", "x", "복약", False, "missing.jpg")
            with self.assertRaises(RoiPreflightError): load_sample_image(missing, {"ai_hub_raw_root": root})

    def test_report_aggregation_and_visual_outputs(self) -> None:
        records = [
            {"dataset": "ai_hub", "target_class": "복약", "is_hard_negative": False, "roi_status": "success", "reason": "ok"},
            {"dataset": "etri", "target_class": "기타", "is_hard_negative": True, "roi_status": "fallback", "reason": "none"},
        ]
        summary = aggregate_report(records)
        self.assertEqual(summary["overall"]["fallback_rate"], 0.5)
        self.assertEqual(summary["by_dataset"]["etri"]["fallback"], 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cv2.imwrite(str(root / "frame.jpg"), np.zeros((40, 40, 3), np.uint8))
            sample = RepresentativeSample("ai_hub", "v:1", "ai_hub_raw_root", "meta", "복약", False, "frame.jpg")
            detector = FakeDetector(Detection(((0.2, 0.2), (0.7, 0.7)), ((0.5, 0.5), (0.8, 0.8))))
            sample_records, sample_summary = run_preflight([sample], {"ai_hub_raw_root": root}, detector, CONFIG, root / "out")
            paths = write_report(sample_records, sample_summary, root / "out")
            self.assertTrue(detector.closed); self.assertTrue(all(path.is_file() for path in paths))
            self.assertTrue((root / "out" / "ai_hub").is_dir())

    def test_invalid_manifest_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            invalid = row("ai_hub", 1, "복약"); invalid["valid"] = False
            path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
            with self.assertRaises(RoiPreflightError): read_manifest(path, "ai_hub")
            with self.assertRaises(RoiPreflightError): read_manifest(path.parent / "missing", "ai_hub")


if __name__ == "__main__":
    unittest.main()
