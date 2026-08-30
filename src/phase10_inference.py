"""Phase 10 production-like video-to-label inference pipeline."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

from src.etri_embedding import (EtriEmbeddingConfig, EtriEmbeddingError,
    decode_sampled_frames, extract_frozen_embeddings, freeze_encoder,
    preprocess_shared_frames, uniform_frame_indices, video_frame_count)
from src.etri_stage_b import StageBConfig, StageBError, load_stage_b_checkpoint, model_config_for
from src.roi_preflight import LandmarkDetector, RoiPreflightConfig
from src.stage_a import (CLASS_NAMES, CLASS_TO_INDEX, StageAConfig, StageAError,
                         StageAMobileNetV3, load_checkpoint)


class Phase10InferenceError(RuntimeError):
    """Raised when input, provenance, or a tensor contract is invalid."""


def file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise Phase10InferenceError(f"Cannot hash required artifact {path}: {exc}") from exc


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise Phase10InferenceError(f"Invalid device {requested!r}: {exc}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise Phase10InferenceError("CUDA was explicitly requested but is unavailable; use --device auto or cpu")
    return device


def validate_video_path(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise Phase10InferenceError(f"Input video does not exist or is not a file: {resolved}")
    return resolved


def validate_checkpoint_provenance(
    *, stage_b_provenance: Mapping[str, Any], stage_a_provenance: Mapping[str, Any],
    selection: Mapping[str, Any], embedding_config: EtriEmbeddingConfig,
    stage_b_config: StageBConfig, stage_a_config: StageAConfig,
    stage_a_checkpoint: Path, roi_config_path: Path, stage_a_config_path: Path,
) -> None:
    expected_encoder_id = selection.get("encoder", {}).get("identifier")
    expected_hash = str(expected_encoder_id).rsplit(":", 1)[-1] if expected_encoder_id else None
    checks = (
        (stage_b_provenance.get("phase") == 9, "Stage B checkpoint is not Phase 9"),
        (stage_b_provenance.get("role") == "deployment_check", "Stage B role is not deployment_check"),
        (stage_b_provenance.get("selected_experiment") == "D", "Selected experiment is not D"),
        (stage_b_provenance.get("stage_b") == "gru", "Selected Stage B is not GRU"),
        (stage_b_provenance.get("model_config") == model_config_for("gru", stage_b_config),
         "Stage B architecture/config mismatch"),
        (stage_b_provenance.get("encoder") == "ai_hub_fine_tuned", "Encoder type mismatch"),
        (stage_b_provenance.get("encoder_key") == "encoder_b", "Encoder cache key mismatch"),
        (stage_b_provenance.get("encoder_frozen") is True, "Encoder frozen provenance mismatch"),
        (stage_b_provenance.get("encoder_cache_identifier") == expected_encoder_id,
         "Stage B/selection encoder identifier mismatch"),
        (file_sha256(stage_a_checkpoint) == expected_hash, "Selected Stage A checkpoint hash mismatch"),
        (stage_b_provenance.get("T") == embedding_config.sequence_length == stage_b_config.sequence_length,
         "Sequence length T mismatch"),
        (stage_b_provenance.get("D") == stage_b_config.embedding_dimension == StageAMobileNetV3.embedding_dim,
         "Embedding dimension D mismatch"),
        (stage_b_provenance.get("num_classes") == 3 == stage_b_config.num_classes, "Class count mismatch"),
        (stage_b_provenance.get("class_mapping") == CLASS_TO_INDEX, "Class mapping mismatch"),
        (stage_b_provenance.get("sampling") == embedding_config.sampling_method == "fixed_uniform",
         "Sampling policy mismatch"),
        (stage_b_provenance.get("normalization_identity") == "torchvision_imagenet_default",
         "Normalization identity mismatch"),
        (stage_b_provenance.get("roi_config_sha256") == file_sha256(roi_config_path),
         "ROI config hash mismatch"),
        (stage_b_provenance.get("stage_a_config_sha256") == file_sha256(stage_a_config_path),
         "Stage A preprocessing config hash mismatch"),
        (stage_a_provenance.get("class_mapping") == CLASS_TO_INDEX, "Stage A class mapping mismatch"),
        (stage_a_provenance.get("embedding_dimension") == StageAMobileNetV3.embedding_dim,
         "Stage A embedding dimension mismatch"),
        (stage_a_config.image_size > 0, "Invalid Stage A image size"),
    )
    for passed, message in checks:
        if not passed:
            raise Phase10InferenceError(message)


class VideoInferencePipeline:
    """Connect the frozen Phase 9 artifacts to the shared Phase 6 preprocessing path."""

    def __init__(self, *, encoder: StageAMobileNetV3, stage_b: torch.nn.Module,
                 detector: LandmarkDetector, roi_config: RoiPreflightConfig,
                 transform: Any, embedding_config: EtriEmbeddingConfig,
                 device: torch.device, encoder_batch_size: int,
                 metadata: Mapping[str, Any]) -> None:
        self.encoder = freeze_encoder(encoder.to(device))
        self.stage_b = stage_b.to(device).eval()
        self.detector, self.roi_config, self.transform = detector, roi_config, transform
        self.embedding_config, self.device = embedding_config, device
        self.encoder_batch_size, self.metadata = encoder_batch_size, dict(metadata)
        if encoder_batch_size <= 0:
            raise Phase10InferenceError("Encoder batch size must be positive")

    def close(self) -> None:
        self.detector.close()

    def predict(self, video_path: Path) -> dict[str, Any]:
        path = validate_video_path(video_path)
        try:
            frame_count = video_frame_count(path)
            indices = uniform_frame_indices(frame_count, self.embedding_config.sequence_length)
            frames, decoded_count = decode_sampled_frames(path, indices)
            if decoded_count != frame_count:
                raise Phase10InferenceError("Video frame count changed during decoding")
            frame_tensor, roi_counts = preprocess_shared_frames(
                frames, self.detector, self.roi_config, self.transform)
            expected_frames = (self.embedding_config.sequence_length, 3,
                               self.metadata["image_size"], self.metadata["image_size"])
            if tuple(frame_tensor.shape) != expected_frames:
                raise Phase10InferenceError(
                    f"Encoder input must be [T,3,H,W]={expected_frames}, got {tuple(frame_tensor.shape)}")
            embeddings = extract_frozen_embeddings(
                self.encoder, frame_tensor, self.device, self.encoder_batch_size)
            sequence = embeddings.unsqueeze(0).to(self.device)
            if tuple(sequence.shape) != (1, self.metadata["T"], self.metadata["D"]):
                raise Phase10InferenceError(f"Stage B input contract failed: {tuple(sequence.shape)}")
            with torch.inference_mode():
                logits = self.stage_b(sequence)
                probabilities = torch.softmax(logits, dim=1)
            if tuple(logits.shape) != (1, 3) or not torch.isfinite(logits).all():
                raise Phase10InferenceError(f"Stage B output contract failed: {tuple(logits.shape)}")
            if not torch.isclose(probabilities.sum(), torch.tensor(1.0, device=self.device), atol=1e-6):
                raise Phase10InferenceError("Malformed inference result: probabilities do not sum to 1")
            index = int(probabilities.argmax(1).item())
            if index < 0 or index >= len(CLASS_NAMES):
                raise Phase10InferenceError(f"Predicted class index is invalid: {index}")
            counts = {status: int(roi_counts.get(status, 0))
                      for status in ("success", "partial", "fallback")}
            if sum(counts.values()) != self.metadata["T"]:
                raise Phase10InferenceError("ROI processing dropped sampled frames")
            summary = "fallback" if counts["fallback"] else "partial" if counts["partial"] else "success"
            return {"status": "PASS", "input_video": str(path), "video_identifier": path.name,
                "original_frame_count": frame_count, "sampled_frame_count": len(indices),
                "sampled_frame_indices": list(indices), "device": str(self.device),
                "selected_encoder": self.metadata["selected_encoder"],
                "selected_stage_b": self.metadata["selected_stage_b"],
                "encoder_checkpoint": self.metadata["encoder_checkpoint"],
                "stage_b_checkpoint": self.metadata["stage_b_checkpoint"],
                "T": self.metadata["T"], "D": self.metadata["D"],
                "tensor_shapes": {"encoder_input": list(frame_tensor.shape),
                    "encoder_output": list(embeddings.shape), "stage_b_input": list(sequence.shape),
                    "stage_b_output": list(logits.shape)},
                "roi_counts": counts, "roi_status": summary,
                "logits": [float(value) for value in logits[0].cpu()],
                "probabilities": {name: float(probabilities[0, CLASS_TO_INDEX[name]].cpu())
                                  for name in CLASS_NAMES},
                "predicted_index": index, "predicted_class": CLASS_NAMES[index],
                "confidence": float(probabilities[0, index].cpu())}
        except (EtriEmbeddingError, StageAError, StageBError, RuntimeError, ValueError) as exc:
            if isinstance(exc, Phase10InferenceError):
                raise
            raise Phase10InferenceError(f"Inference failed for {path}: {exc}") from exc
