"""Phase 6 ETRI frozen-encoder embedding contract and smoke-cache helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
import yaml

from src.pilot_manifest import PilotManifestError, load_pilot_config, read_jsonl, validate_etri_manifest
from src.roi_preflight import LandmarkDetector, RoiPreflightConfig, extract_roi
from src.stage_a import IMAGENET_MEAN, IMAGENET_STD, StageAMobileNetV3


REQUIRED_PROVENANCE_FIELDS = {
    "source_clip_key", "participant", "fold", "target_class", "encoder_type",
    "encoder_checkpoint_identifier", "preprocessing_config", "normalization",
    "roi_config_version", "sampling_method", "T", "D", "inference_batch_size",
    "encoder_training",
}


class EtriEmbeddingError(RuntimeError):
    """Raised when the Phase 6 embedding contract is violated."""


@dataclass(frozen=True)
class EtriEmbeddingConfig:
    sequence_length: int
    sampling_method: str
    inference_batch_size: int
    device: str
    stage_a_config_path: str
    roi_config_path: str
    pilot_config_path: str
    encoder_b_checkpoint_relative_path: str
    full_cache_relative_path: str


def load_embedding_config(path: Path) -> EtriEmbeddingConfig:
    try:
        root = yaml.safe_load(path.read_text(encoding="utf-8"))["phase6"]
        config = EtriEmbeddingConfig(
            sequence_length=int(root["sequence_length"]), sampling_method=str(root["sampling_method"]),
            inference_batch_size=int(root["inference_batch_size"]), device=str(root["device"]),
            stage_a_config_path=str(root["stage_a_config"]), roi_config_path=str(root["roi_config"]),
            pilot_config_path=str(root["pilot_config"]),
            encoder_b_checkpoint_relative_path=str(root["encoder_b_checkpoint_relative_path"]),
            full_cache_relative_path=str(root["full_cache_relative_path"]),
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise EtriEmbeddingError(f"Invalid Phase 6 config {path}: {exc}") from exc
    if config.sequence_length != 64 or config.sampling_method != "fixed_uniform":
        raise EtriEmbeddingError("Phase 6 preflight requires T=64 fixed_uniform sampling")
    if config.inference_batch_size <= 0: raise EtriEmbeddingError("inference_batch_size must be positive")
    for relative in (config.stage_a_config_path, config.roi_config_path, config.pilot_config_path,
                     config.encoder_b_checkpoint_relative_path):
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise EtriEmbeddingError(f"Phase 6 configured path must be safe and relative: {relative}")
    output_relative = Path(config.full_cache_relative_path)
    if output_relative.is_absolute() or ".." in output_relative.parts:
        raise EtriEmbeddingError("full_cache_relative_path must be safe and relative")
    return config


def uniform_frame_indices(frame_count: int, sequence_length: int = 64) -> tuple[int, ...]:
    if frame_count <= 0 or sequence_length <= 0: raise EtriEmbeddingError("Frame count and T must be positive")
    indices = np.rint(np.linspace(0, frame_count - 1, num=sequence_length)).astype(np.int64)
    result = tuple(int(value) for value in indices)
    if len(result) != sequence_length or result[0] != 0 or result[-1] != frame_count - 1:
        raise EtriEmbeddingError("Uniform sampling contract failed")
    return result


def source_clip_key(row: Mapping[str, Any]) -> str:
    return f"{row['participant']}:{row['action']}:{row['take']}:{row['height']}"


def read_selected_etri_manifest(path: Path, pilot_config_path: Path) -> tuple[list[dict[str, Any]], str]:
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        rows = read_jsonl(path)
        validate_etri_manifest(rows, load_pilot_config(pilot_config_path))
    except PilotManifestError as exc:
        raise EtriEmbeddingError(f"Frozen ETRI Pilot Manifest validation failed: {exc}") from exc
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    if before != after: raise EtriEmbeddingError("Frozen ETRI Pilot Manifest changed while being read")
    if any(row.get("dataset") != "etri" for row in rows):
        raise EtriEmbeddingError("Frozen ETRI Pilot Manifest contains a non-ETRI row")
    if any(row.get("pilot_selected") is True and (row.get("valid") is not True or row.get("roi_status") != "pending")
           for row in rows):
        raise EtriEmbeddingError("Selected ETRI row violates valid/pending frozen state")
    selected = [row for row in rows if row.get("dataset") == "etri" and row.get("pilot_selected") is True
                and row.get("valid") is True]
    if not selected: raise EtriEmbeddingError("No selected valid ETRI Pilot clips")
    return selected, before


def select_smoke_clip(rows: Sequence[Mapping[str, Any]], requested_key: str | None = None) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: source_clip_key(row))
    if requested_key is None: return dict(ordered[0])
    matches = [row for row in ordered if source_clip_key(row) == requested_key]
    if len(matches) != 1: raise EtriEmbeddingError(f"Smoke clip key did not resolve exactly once: {requested_key}")
    return dict(matches[0])


def resolve_video_path(row: Mapping[str, Any], roots: Mapping[str, Path]) -> Path:
    root = roots.get(str(row.get("root_key"))); relative = Path(str(row.get("relative_path", "")))
    if root is None or relative.is_absolute() or ".." in relative.parts: raise EtriEmbeddingError("Unsafe ETRI video path")
    path = (root / relative).resolve(strict=False)
    try: path.relative_to(root.resolve(strict=True))
    except ValueError as exc: raise EtriEmbeddingError(f"ETRI video escapes configured root: {path}") from exc
    if not path.is_file(): raise EtriEmbeddingError(f"ETRI video does not exist: {path}")
    return path


def decode_sampled_frames(path: Path, indices: Sequence[int]) -> tuple[list[np.ndarray], int]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened(): raise EtriEmbeddingError(f"Cannot open ETRI video: {path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0: raise EtriEmbeddingError(f"Invalid ETRI frame count: {path}")
        frames: list[np.ndarray] = []
        for index in indices:
            if index < 0 or index >= frame_count: raise EtriEmbeddingError(f"Sampled index outside video: {index}")
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, int(index)):
                raise EtriEmbeddingError(f"Cannot seek ETRI frame {index}: {path}")
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0: raise EtriEmbeddingError(f"Cannot decode ETRI frame {index}: {path}")
            frames.append(frame)
        return frames, frame_count
    finally: capture.release()


def video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened(): raise EtriEmbeddingError(f"Cannot open ETRI video: {path}")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0: raise EtriEmbeddingError(f"Invalid ETRI frame count: {path}")
        return count
    finally: capture.release()


def validate_manifest_frame_count(row: Mapping[str, Any], actual_frame_count: int) -> None:
    manifest_count = row.get("rgb_frame_count")
    if not isinstance(manifest_count, int) or manifest_count <= 0:
        raise EtriEmbeddingError(f"Manifest rgb_frame_count is invalid: {manifest_count}")
    if actual_frame_count != manifest_count:
        raise EtriEmbeddingError(
            f"Video/manifest frame count mismatch for {source_clip_key(row)}: "
            f"cv2={actual_frame_count}, manifest={manifest_count}"
        )


def preprocess_shared_frames(frames: Sequence[np.ndarray], detector: LandmarkDetector,
                             roi_config: RoiPreflightConfig, transform: Any) -> tuple[torch.Tensor, dict[str, int]]:
    tensors, statuses = [], Counter()
    for frame in frames:
        result = extract_roi(frame, detector.detect(frame), roi_config)
        statuses[result.roi_status] += 1
        tensors.append(transform(cv2.cvtColor(result.crop, cv2.COLOR_BGR2RGB)))
    return torch.stack(tensors), {status: statuses[status] for status in ("success", "partial", "fallback")}


def freeze_encoder(model: StageAMobileNetV3) -> StageAMobileNetV3:
    for parameter in model.parameters(): parameter.requires_grad = False
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()) or model.training:
        raise EtriEmbeddingError("Encoder freeze contract failed")
    return model


def extract_frozen_embeddings(model: StageAMobileNetV3, frames: torch.Tensor, device: torch.device,
                              batch_size: int) -> torch.Tensor:
    freeze_encoder(model); outputs = []
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            outputs.append(model.extract_features(frames[start:start + batch_size].to(device)).cpu())
    embeddings = torch.cat(outputs)
    validate_embedding(embeddings, len(frames), StageAMobileNetV3.embedding_dim)
    return embeddings


def extract_encoder_pair(encoder_a: StageAMobileNetV3, encoder_b: StageAMobileNetV3,
                         shared_frames: torch.Tensor, device: torch.device,
                         batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply both frozen encoders to the exact same preprocessed tensor object."""
    embedding_a = extract_frozen_embeddings(encoder_a, shared_frames, device, batch_size)
    embedding_b = extract_frozen_embeddings(encoder_b, shared_frames, device, batch_size)
    return embedding_a, embedding_b


def validate_embedding(embedding: torch.Tensor, sequence_length: int, dimension: int) -> None:
    if tuple(embedding.shape) != (sequence_length, dimension):
        raise EtriEmbeddingError(f"Embedding shape mismatch: {tuple(embedding.shape)}")
    if not torch.isfinite(embedding).all(): raise EtriEmbeddingError("Embedding contains NaN or Inf")


def make_encoder_provenance(row: Mapping[str, Any], encoder_type: str, checkpoint_identifier: str,
                            config: EtriEmbeddingConfig, roi_config_hash: str,
                            stage_a_config_hash: str, image_size: int,
                            encoder_training: Mapping[str, Any]) -> dict[str, Any]:
    value = {"source_clip_key": source_clip_key(row), "participant": str(row["participant"]),
        "fold": int(row["fold"]), "target_class": str(row["target_class"]), "encoder_type": encoder_type,
        "encoder_checkpoint_identifier": checkpoint_identifier,
        "preprocessing_config": {"stage_a_config_sha256": stage_a_config_hash, "image_size": image_size},
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "roi_config_version": roi_config_hash, "sampling_method": config.sampling_method,
        "T": config.sequence_length, "D": StageAMobileNetV3.embedding_dim,
        "inference_batch_size": config.inference_batch_size,
        "encoder_training": dict(encoder_training)}
    missing = REQUIRED_PROVENANCE_FIELDS - set(value)
    if missing: raise EtriEmbeddingError(f"Cache provenance missing fields: {sorted(missing)}")
    return value


def encoder_b_training_provenance(checkpoint_provenance: Mapping[str, Any]) -> dict[str, Any]:
    model_config = checkpoint_provenance.get("model_config")
    if not isinstance(model_config, Mapping): raise EtriEmbeddingError("Encoder B checkpoint lacks model_config")
    fine_tuning_mode = model_config.get("fine_tuning_mode")
    frozen_policy = checkpoint_provenance.get("frozen_batchnorm_policy")
    if not isinstance(fine_tuning_mode, str) or not isinstance(frozen_policy, str):
        raise EtriEmbeddingError("Encoder B checkpoint lacks fine-tuning/BatchNorm provenance")
    return {"source": "ai_hub_stage_a", "fine_tuning_mode": fine_tuning_mode,
            "frozen_batchnorm_policy": frozen_policy,
            "stage_a_run_id": checkpoint_provenance.get("run_id"),
            "stage_a_best_epoch": checkpoint_provenance.get("best_epoch")}


def encoder_a_training_provenance() -> dict[str, Any]:
    return {"source": "torchvision_imagenet_pretrained", "fine_tuning_mode": "none",
            "frozen_batchnorm_policy": "not_applicable"}


def make_cache_payload(indices: Sequence[int], embedding_a: torch.Tensor, embedding_b: torch.Tensor,
                       provenance_a: Mapping[str, Any], provenance_b: Mapping[str, Any],
                       roi_counts: Mapping[str, int], original_frame_count: int,
                       manifest_sha256: str) -> dict[str, Any]:
    validate_embedding(embedding_a, len(indices), 1024); validate_embedding(embedding_b, len(indices), 1024)
    for provenance in (provenance_a, provenance_b):
        missing = REQUIRED_PROVENANCE_FIELDS - set(provenance)
        if missing: raise EtriEmbeddingError(f"Cache provenance missing fields: {sorted(missing)}")
    if provenance_a["source_clip_key"] != provenance_b["source_clip_key"]:
        raise EtriEmbeddingError("Encoder A/B clip keys differ")
    if sum(int(roi_counts.get(status, 0)) for status in ("success", "partial", "fallback")) != len(indices):
        raise EtriEmbeddingError("ROI status counts do not equal sampled frame count")
    shared_indices = list(indices)
    return {"format_version": 1, "source_clip_key": provenance_a["source_clip_key"],
        "participant": provenance_a["participant"], "fold": provenance_a["fold"],
        "target_class": provenance_a["target_class"],
        "original_video_frame_count": original_frame_count, "sampled_frame_indices": shared_indices,
        "roi_counts": dict(roi_counts), "manifest_sha256": manifest_sha256,
        "encoders": {"encoder_a": {"embedding": embedding_a, "sampled_frame_indices": shared_indices,
                                     "provenance": dict(provenance_a)},
                     "encoder_b": {"embedding": embedding_b, "sampled_frame_indices": shared_indices,
                                     "provenance": dict(provenance_b)}}}


def save_and_reload_cache(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary); temporary.replace(path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("encoder_a", "encoder_b"):
        original = payload["encoders"][key]["embedding"]; restored = loaded["encoders"][key]["embedding"]
        if original.shape != restored.shape or not torch.equal(original, restored):
            raise EtriEmbeddingError(f"Cache reload mismatch for {key}")
        if loaded["encoders"][key]["sampled_frame_indices"] != loaded["sampled_frame_indices"]:
            raise EtriEmbeddingError(f"Cache reload sampled indices mismatch for {key}")
    if loaded["sampled_frame_indices"] != list(payload["sampled_frame_indices"]):
        raise EtriEmbeddingError("Cache reload sampled indices mismatch")
    return loaded


def process_clip(
    row: Mapping[str, Any], video_path: Path, config: EtriEmbeddingConfig,
    roi_config: RoiPreflightConfig, transform: Any, detector: LandmarkDetector,
    encoder_a: StageAMobileNetV3, encoder_b: StageAMobileNetV3, device: torch.device,
    encoder_a_identifier: str, encoder_b_identifier: str,
    encoder_b_training: Mapping[str, Any], roi_config_hash: str,
    stage_a_config_hash: str, image_size: int, manifest_sha256: str,
) -> dict[str, Any]:
    actual_count = video_frame_count(video_path)
    validate_manifest_frame_count(row, actual_count)
    indices = uniform_frame_indices(actual_count, config.sequence_length)
    frames, decoded_count = decode_sampled_frames(video_path, indices)
    if decoded_count != actual_count: raise EtriEmbeddingError("Video frame count changed during processing")
    shared_tensor, roi_counts = preprocess_shared_frames(frames, detector, roi_config, transform)
    if len(shared_tensor) != config.sequence_length: raise EtriEmbeddingError("ROI preprocessing dropped frames")
    embedding_a, embedding_b = extract_encoder_pair(
        encoder_a, encoder_b, shared_tensor, device, config.inference_batch_size,
    )
    provenance_a = make_encoder_provenance(
        row, "imagenet_only", encoder_a_identifier, config, roi_config_hash,
        stage_a_config_hash, image_size, encoder_a_training_provenance(),
    )
    provenance_b = make_encoder_provenance(
        row, "ai_hub_fine_tuned", encoder_b_identifier, config, roi_config_hash,
        stage_a_config_hash, image_size, encoder_b_training,
    )
    return make_cache_payload(indices, embedding_a, embedding_b, provenance_a, provenance_b,
                              roi_counts, actual_count, manifest_sha256)


def deterministic_cache_name(clip_key: str) -> str:
    parts = clip_key.split(":")
    if len(parts) != 4 or any(re.fullmatch(r"[A-Za-z0-9]+", part) is None for part in parts):
        raise EtriEmbeddingError(f"Unsafe source clip key for cache naming: {clip_key}")
    return "__".join(parts) + ".pt"


def load_valid_cache(path: Path, row: Mapping[str, Any], manifest_sha256: str,
                     encoder_a_identifier: str, encoder_b_identifier: str,
                     config: EtriEmbeddingConfig, roi_config_hash: str,
                     stage_a_config_hash: str) -> dict[str, Any]:
    try: payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc: raise EtriEmbeddingError(f"Cannot load existing cache {path}: {exc}") from exc
    validate_cache_payload(payload, row, manifest_sha256, encoder_a_identifier, encoder_b_identifier,
                           config, roi_config_hash, stage_a_config_hash)
    return payload


def validate_cache_payload(payload: Mapping[str, Any], row: Mapping[str, Any], manifest_sha256: str,
                           encoder_a_identifier: str, encoder_b_identifier: str,
                           config: EtriEmbeddingConfig, roi_config_hash: str,
                           stage_a_config_hash: str) -> None:
    key = source_clip_key(row)
    if payload.get("format_version") != 1 or payload.get("source_clip_key") != key:
        raise EtriEmbeddingError(f"Cache identity mismatch for {key}")
    if payload.get("manifest_sha256") != manifest_sha256:
        raise EtriEmbeddingError(f"Cache manifest hash mismatch for {key}")
    for field in ("participant", "fold", "target_class"):
        if payload.get(field) != row.get(field): raise EtriEmbeddingError(f"Cache {field} mismatch for {key}")
    indices = payload.get("sampled_frame_indices")
    if not isinstance(indices, list) or len(indices) != config.sequence_length:
        raise EtriEmbeddingError(f"Cache sampled indices invalid for {key}")
    manifest_frame_count = row.get("rgb_frame_count")
    if payload.get("original_video_frame_count") != manifest_frame_count:
        raise EtriEmbeddingError(f"Cache original frame count mismatch for {key}")
    if indices != list(uniform_frame_indices(int(manifest_frame_count), config.sequence_length)):
        raise EtriEmbeddingError(f"Cache uniform sampling grid mismatch for {key}")
    if sum(int(payload.get("roi_counts", {}).get(status, 0)) for status in ("success", "partial", "fallback")) != config.sequence_length:
        raise EtriEmbeddingError(f"Cache ROI counts invalid for {key}")
    expected_ids = {"encoder_a": encoder_a_identifier, "encoder_b": encoder_b_identifier}
    for encoder_key, identifier in expected_ids.items():
        record = payload.get("encoders", {}).get(encoder_key, {})
        embedding, provenance = record.get("embedding"), record.get("provenance", {})
        if not isinstance(embedding, torch.Tensor): raise EtriEmbeddingError(f"Cache embedding missing for {key}/{encoder_key}")
        validate_embedding(embedding, config.sequence_length, 1024)
        if record.get("sampled_frame_indices") != indices: raise EtriEmbeddingError(f"Cache index parity failed for {key}")
        if provenance.get("source_clip_key") != key or provenance.get("encoder_checkpoint_identifier") != identifier:
            raise EtriEmbeddingError(f"Cache encoder provenance mismatch for {key}/{encoder_key}")
        missing = REQUIRED_PROVENANCE_FIELDS - set(provenance)
        if missing: raise EtriEmbeddingError(f"Cache provenance incomplete for {key}/{encoder_key}: {sorted(missing)}")
        for field in ("participant", "fold", "target_class"):
            if provenance.get(field) != row.get(field):
                raise EtriEmbeddingError(f"Cache encoder {field} mismatch for {key}/{encoder_key}")
        if provenance.get("inference_batch_size") != config.inference_batch_size:
            raise EtriEmbeddingError(f"Cache inference batch size mismatch for {key}/{encoder_key}")
        if provenance.get("roi_config_version") != roi_config_hash:
            raise EtriEmbeddingError(f"Cache ROI config mismatch for {key}/{encoder_key}")
        if provenance.get("preprocessing_config", {}).get("stage_a_config_sha256") != stage_a_config_hash:
            raise EtriEmbeddingError(f"Cache Stage A config mismatch for {key}/{encoder_key}")
        training = provenance.get("encoder_training")
        if not isinstance(training, Mapping) or not isinstance(training.get("fine_tuning_mode"), str):
            raise EtriEmbeddingError(f"Cache encoder training provenance missing for {key}/{encoder_key}")
        if encoder_key == "encoder_b" and not isinstance(training.get("frozen_batchnorm_policy"), str):
            raise EtriEmbeddingError(f"Cache Encoder B BatchNorm provenance missing for {key}")


def cache_statistics(payload: Mapping[str, Any]) -> dict[str, Any]:
    a = payload["encoders"]["encoder_a"]["embedding"]
    b = payload["encoders"]["encoder_b"]["embedding"]
    return {"source_clip_key": payload["source_clip_key"], "participant": payload["participant"],
        "fold": payload["fold"], "target_class": payload["target_class"], "roi_counts": payload["roi_counts"],
        "encoder_a_shape": list(a.shape), "encoder_a_nan": int(torch.isnan(a).sum()),
        "encoder_a_inf": int(torch.isinf(a).sum()), "encoder_b_shape": list(b.shape),
        "encoder_b_nan": int(torch.isnan(b).sum()), "encoder_b_inf": int(torch.isinf(b).sum()),
        "clip_key_parity": payload["encoders"]["encoder_a"]["provenance"]["source_clip_key"]
            == payload["encoders"]["encoder_b"]["provenance"]["source_clip_key"],
        "frame_index_parity": payload["encoders"]["encoder_a"]["sampled_frame_indices"]
            == payload["encoders"]["encoder_b"]["sampled_frame_indices"]}
