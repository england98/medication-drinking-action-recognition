from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from src.etri_embedding import load_embedding_config, uniform_frame_indices
from src.phase10_inference import (Phase10InferenceError, VideoInferencePipeline,
                                   resolve_device, validate_video_path)
from src.roi_preflight import Detection, load_roi_config
from src.stage_a import CLASS_NAMES


class FakeDetector:
    def __init__(self) -> None:
        self.index = 0; self.closed = False

    def detect(self, image: np.ndarray) -> Detection:
        values = (Detection(((.2, .2), (.5, .5)), ((.5, .5), (.8, .8))),
                  Detection(face_points=((.2, .2), (.5, .5))), Detection())
        value = values[self.index % len(values)]; self.index += 1; return value

    def close(self) -> None:
        self.closed = True


class FakeEncoder(nn.Module):
    embedding_dim = 1024

    def __init__(self) -> None:
        super().__init__(); self.anchor = nn.Parameter(torch.zeros(1))

    def extract_features(self, values: torch.Tensor) -> torch.Tensor:
        return torch.ones(len(values), 1024, device=values.device) + self.anchor


class FakeStageB(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if tuple(values.shape) != (1, 64, 1024): raise RuntimeError("bad shape")
        return torch.tensor([[1.0, 2.0, 0.0]], device=values.device)


class Phase10InferenceTests(unittest.TestCase):
    def test_uniform_sampling_is_deterministic(self) -> None:
        self.assertEqual(uniform_frame_indices(101), uniform_frame_indices(101))
        self.assertEqual((uniform_frame_indices(101)[0], uniform_frame_indices(101)[-1]), (0, 100))

    def test_invalid_video_and_device_errors_are_clear(self) -> None:
        with self.assertRaisesRegex(Phase10InferenceError, "does not exist"):
            validate_video_path(Path("/definitely/missing/video.mp4"))
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("auto"), torch.device("cpu"))
            with self.assertRaisesRegex(Phase10InferenceError, "unavailable"):
                resolve_device("cuda")

    def test_single_video_contract_probability_mapping_and_roi_no_drop(self) -> None:
        config = load_embedding_config(Path("configs/etri_embedding.yaml"))
        detector = FakeDetector()
        pipeline = VideoInferencePipeline(encoder=FakeEncoder(), stage_b=FakeStageB(),
            detector=detector, roi_config=load_roi_config(Path("configs/roi_preflight.yaml")),
            transform=lambda image: torch.zeros(3, 224, 224), embedding_config=config,
            device=torch.device("cpu"), encoder_batch_size=8,
            metadata={"selected_encoder": "ai_hub_fine_tuned", "selected_stage_b": "gru",
                "encoder_checkpoint": "encoder.pt", "stage_b_checkpoint": "stage-b.pt",
                "T": 64, "D": 1024, "image_size": 224})
        frames = [np.zeros((40, 50, 3), np.uint8) for _ in range(64)]
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "fixture.mp4"; video.touch()
            with patch("src.phase10_inference.video_frame_count", return_value=70), \
                 patch("src.phase10_inference.decode_sampled_frames", return_value=(frames, 70)):
                result = pipeline.predict(video)
        self.assertEqual(result["predicted_class"], "음수")
        self.assertEqual(result["predicted_index"], 1)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=6)
        self.assertEqual(set(result["probabilities"]), set(CLASS_NAMES))
        self.assertEqual(result["tensor_shapes"], {"encoder_input": [64, 3, 224, 224],
            "encoder_output": [64, 1024], "stage_b_input": [1, 64, 1024],
            "stage_b_output": [1, 3]})
        self.assertEqual(sum(result["roi_counts"].values()), 64)
        self.assertTrue(all(result["roi_counts"][key] > 0 for key in ("success", "partial", "fallback")))
        self.assertFalse(pipeline.encoder.training); self.assertFalse(pipeline.stage_b.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in pipeline.encoder.parameters()))
        pipeline.close(); self.assertTrue(detector.closed)

    def test_stage_b_shape_mismatch_fails(self) -> None:
        class WrongStageB(FakeStageB):
            def forward(self, values: torch.Tensor) -> torch.Tensor:
                return torch.zeros(1, 2)
        config = load_embedding_config(Path("configs/etri_embedding.yaml"))
        pipeline = VideoInferencePipeline(encoder=FakeEncoder(), stage_b=WrongStageB(),
            detector=FakeDetector(), roi_config=load_roi_config(Path("configs/roi_preflight.yaml")),
            transform=lambda image: torch.zeros(3, 224, 224), embedding_config=config,
            device=torch.device("cpu"), encoder_batch_size=8,
            metadata={"selected_encoder": "x", "selected_stage_b": "x", "encoder_checkpoint": "x",
                "stage_b_checkpoint": "x", "T": 64, "D": 1024, "image_size": 224})
        frames = [np.zeros((40, 50, 3), np.uint8) for _ in range(64)]
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "fixture.mp4"; video.touch()
            with patch("src.phase10_inference.video_frame_count", return_value=64), \
                 patch("src.phase10_inference.decode_sampled_frames", return_value=(frames, 64)), \
                 self.assertRaisesRegex(Phase10InferenceError, "Stage B output contract"):
                pipeline.predict(video)


if __name__ == "__main__": unittest.main()
