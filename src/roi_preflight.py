"""Phase 4 ROI Preflight selection, MediaPipe ROI, visuals, and reports."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import cv2
import mediapipe as mp
import numpy as np
import yaml


ROI_STATUSES = {"pending", "success", "partial", "fallback"}
REPORT_FIELDS = (
    "dataset", "source_sample_key", "target_class", "is_hard_negative",
    "roi_status", "bbox", "reason", "visual_output_path",
)


class RoiPreflightError(RuntimeError):
    """Raised for structural errors that make the Preflight unreliable."""


@dataclass(frozen=True)
class RoiPreflightConfig:
    selection_seed: int
    representative_per_stratum: int
    ai_hub_frame_index: int
    etri_frame_positions: tuple[float, ...]
    padding_ratio: float
    minimum_width_px: int
    minimum_height_px: int
    hand_only_min_width_ratio: float
    hand_only_min_height_ratio: float
    hand_only_vertical_anchor: float
    face_only_min_width_ratio: float
    face_only_min_height_ratio: float
    face_only_vertical_anchor: float
    pose_only_min_width_ratio: float
    pose_only_min_height_ratio: float
    pose_only_vertical_anchor: float
    fallback_mode: str
    allow_partial: bool
    num_faces: int
    num_hands: int
    num_poses: int
    minimum_face_detection_confidence: float
    minimum_face_presence_confidence: float
    minimum_hand_detection_confidence: float
    minimum_hand_presence_confidence: float
    minimum_pose_detection_confidence: float
    minimum_pose_presence_confidence: float
    face_model_relative_path: str
    hand_model_relative_path: str
    pose_model_relative_path: str
    face_model_url: str
    hand_model_url: str
    pose_model_url: str
    output_relative_path: str


@dataclass(frozen=True)
class RepresentativeSample:
    dataset: str
    source_sample_key: str
    root_key: str
    relative_path: str
    target_class: str
    is_hard_negative: bool
    frame_relative_path: str | None = None
    frame_position: float | None = None


@dataclass(frozen=True)
class Detection:
    face_points: tuple[tuple[float, float], ...] = ()
    hand_points: tuple[tuple[float, float], ...] = ()
    pose_points: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class RoiResult:
    roi_status: str
    bbox: tuple[int, int, int, int]
    reason: str
    crop: np.ndarray


class LandmarkDetector(Protocol):
    def detect(self, image_bgr: np.ndarray) -> Detection: ...
    def close(self) -> None: ...


def load_roi_config(path: Path) -> RoiPreflightConfig:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))["phase4"]
        roi, media = document["roi"], document["mediapipe"]
        config = RoiPreflightConfig(
            selection_seed=int(document["selection_seed"]),
            representative_per_stratum=int(document["representative_per_stratum"]),
            ai_hub_frame_index=int(document["ai_hub_frame_index"]),
            etri_frame_positions=tuple(float(x) for x in document["etri_frame_positions"]),
            padding_ratio=float(roi["padding_ratio"]), minimum_width_px=int(roi["minimum_width_px"]),
            minimum_height_px=int(roi["minimum_height_px"]), fallback_mode=str(roi["fallback_mode"]),
            hand_only_min_width_ratio=float(roi["hand_only_min_width_ratio"]),
            hand_only_min_height_ratio=float(roi["hand_only_min_height_ratio"]),
            hand_only_vertical_anchor=float(roi["hand_only_vertical_anchor"]),
            face_only_min_width_ratio=float(roi["face_only_min_width_ratio"]),
            face_only_min_height_ratio=float(roi["face_only_min_height_ratio"]),
            face_only_vertical_anchor=float(roi["face_only_vertical_anchor"]),
            pose_only_min_width_ratio=float(roi["pose_only_min_width_ratio"]),
            pose_only_min_height_ratio=float(roi["pose_only_min_height_ratio"]),
            pose_only_vertical_anchor=float(roi["pose_only_vertical_anchor"]),
            allow_partial=bool(roi["allow_partial"]), num_faces=int(media["num_faces"]),
            num_hands=int(media["num_hands"]),
            num_poses=int(media["num_poses"]),
            minimum_face_detection_confidence=float(media["minimum_face_detection_confidence"]),
            minimum_face_presence_confidence=float(media["minimum_face_presence_confidence"]),
            minimum_hand_detection_confidence=float(media["minimum_hand_detection_confidence"]),
            minimum_hand_presence_confidence=float(media["minimum_hand_presence_confidence"]),
            minimum_pose_detection_confidence=float(media["minimum_pose_detection_confidence"]),
            minimum_pose_presence_confidence=float(media["minimum_pose_presence_confidence"]),
            face_model_relative_path=str(media["face_model_relative_path"]),
            hand_model_relative_path=str(media["hand_model_relative_path"]),
            pose_model_relative_path=str(media["pose_model_relative_path"]),
            face_model_url=str(media["face_model_url"]), hand_model_url=str(media["hand_model_url"]),
            pose_model_url=str(media["pose_model_url"]),
            output_relative_path=str(document["output_relative_path"]),
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise RoiPreflightError(f"Invalid ROI Preflight config {path}: {exc}") from exc
    if config.representative_per_stratum <= 0 or config.ai_hub_frame_index < 0:
        raise RoiPreflightError("Representative counts/index must be positive")
    if not config.etri_frame_positions or any(not 0 <= x <= 1 for x in config.etri_frame_positions):
        raise RoiPreflightError("ETRI frame positions must be non-empty values in [0, 1]")
    if config.padding_ratio < 0 or config.minimum_width_px <= 0 or config.minimum_height_px <= 0:
        raise RoiPreflightError("ROI padding/minimum dimensions are invalid")
    ratios = (config.hand_only_min_width_ratio, config.hand_only_min_height_ratio,
              config.face_only_min_width_ratio, config.face_only_min_height_ratio,
              config.pose_only_min_width_ratio, config.pose_only_min_height_ratio)
    anchors = (config.hand_only_vertical_anchor, config.face_only_vertical_anchor,
               config.pose_only_vertical_anchor)
    if any(not 0 < value <= 1 for value in (*ratios, *anchors)):
        raise RoiPreflightError("Partial contextual ratios/anchors must be in (0, 1]")
    if config.fallback_mode != "full_frame":
        raise RoiPreflightError("Only the Baseline-compatible full_frame fallback is supported")
    for relative in (config.face_model_relative_path, config.hand_model_relative_path,
                     config.pose_model_relative_path, config.output_relative_path):
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RoiPreflightError(f"Working path must be a safe relative path: {relative}")
    return config


def read_manifest(path: Path, expected_dataset: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RoiPreflightError(f"Fixed Pilot Manifest does not exist: {path}")
    rows: list[dict[str, Any]] = []
    required = {"dataset", "root_key", "relative_path", "target_class", "is_hard_negative", "valid", "pilot_selected"}
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip(): continue
            row = json.loads(line)
            if not isinstance(row, dict): raise RoiPreflightError(f"Manifest row {number} is not an object")
            missing = required - set(row)
            if missing: raise RoiPreflightError(f"Manifest row {number} missing fields: {sorted(missing)}")
            if row["dataset"] != expected_dataset: raise RoiPreflightError(f"Unexpected dataset at row {number}: {row['dataset']}")
            if row.get("pilot_selected") and not row.get("valid"): raise RoiPreflightError(f"Selected invalid row at {number}")
            rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise RoiPreflightError(f"Cannot read manifest {path}: {exc}") from exc
    if not rows: raise RoiPreflightError(f"Manifest is empty: {path}")
    return rows


def _stratum(row: Mapping[str, Any]) -> str:
    if row["target_class"] != "기타": return str(row["target_class"])
    return "hard_negative" if row.get("is_hard_negative") else "general_other"


def _shuffle(rows: Iterable[Mapping[str, Any]], seed: int, namespace: str) -> list[Mapping[str, Any]]:
    result = sorted(rows, key=lambda row: str(row["relative_path"]))
    digest = hashlib.sha256(f"{seed}:{namespace}".encode()).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(result)
    return result


def select_representatives(ai_rows: Sequence[Mapping[str, Any]], etri_rows: Sequence[Mapping[str, Any]], config: RoiPreflightConfig) -> list[RepresentativeSample]:
    samples: list[RepresentativeSample] = []
    for dataset, rows in (("ai_hub", ai_rows), ("etri", etri_rows)):
        eligible = [row for row in rows if row.get("pilot_selected") is True and row.get("valid") is True]
        if dataset == "ai_hub" and any(row.get("viewpoint") != 3 for row in eligible):
            raise RoiPreflightError("AI-Hub selected Preflight input contains non-viewpoint_3 row")
        if dataset == "etri" and any(row.get("batch") != "B" for row in eligible):
            raise RoiPreflightError("ETRI selected Preflight input contains non-Batch-B row")
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in eligible: grouped[_stratum(row)].append(row)
        required_strata = {"복약", "음수", "hard_negative", "general_other"}
        missing = required_strata - set(grouped)
        if missing: raise RoiPreflightError(f"{dataset} lacks representative strata: {sorted(missing)}")
        for stratum in sorted(required_strata):
            chosen = _shuffle(grouped[stratum], config.selection_seed, f"{dataset}:{stratum}")[:config.representative_per_stratum]
            if len(chosen) < config.representative_per_stratum:
                raise RoiPreflightError(f"Insufficient {dataset}/{stratum} candidates")
            for row in chosen:
                if dataset == "ai_hub":
                    frames = row.get("frame_relative_paths")
                    if not isinstance(frames, list) or config.ai_hub_frame_index >= len(frames):
                        raise RoiPreflightError(f"Invalid AI-Hub frame_relative_paths: {row['relative_path']}")
                    samples.append(RepresentativeSample(
                        dataset=dataset, source_sample_key=f"{row['video_id']}:{config.ai_hub_frame_index}",
                        root_key=str(row["root_key"]), relative_path=str(row["relative_path"]),
                        frame_relative_path=str(frames[config.ai_hub_frame_index]), target_class=str(row["target_class"]),
                        is_hard_negative=bool(row["is_hard_negative"]),
                    ))
                else:
                    for position in config.etri_frame_positions:
                        samples.append(RepresentativeSample(
                            dataset=dataset, source_sample_key=f"{row['participant']}:{row['action']}:{row['take']}:{row['height']}@{position:.2f}",
                            root_key=str(row["root_key"]), relative_path=str(row["relative_path"]),
                            target_class=str(row["target_class"]), is_hard_negative=bool(row["is_hard_negative"]),
                            frame_position=position,
                        ))
    return samples


def resolve_input(sample: RepresentativeSample, roots: Mapping[str, Path]) -> Path:
    root = roots.get(sample.root_key)
    if root is None: raise RoiPreflightError(f"Unknown root_key: {sample.root_key}")
    relative = sample.frame_relative_path if sample.dataset == "ai_hub" else sample.relative_path
    if relative is None or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise RoiPreflightError(f"Unsafe input relative path: {relative}")
    path = (root / relative).resolve(strict=False)
    try: path.relative_to(root.resolve(strict=True))
    except ValueError as exc: raise RoiPreflightError(f"Input path escapes root: {path}") from exc
    if not path.is_file(): raise RoiPreflightError(f"Input file does not exist: {path}")
    return path


def load_sample_image(sample: RepresentativeSample, roots: Mapping[str, Path]) -> np.ndarray:
    path = resolve_input(sample, roots)
    if sample.dataset == "ai_hub":
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0: raise RoiPreflightError(f"Cannot read AI-Hub image: {path}")
        return image
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened(): raise RoiPreflightError(f"Cannot open ETRI video: {path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0: raise RoiPreflightError(f"Invalid ETRI frame count: {path}")
        index = min(frame_count - 1, max(0, round((frame_count - 1) * float(sample.frame_position))))
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
            raise RoiPreflightError(f"Cannot seek ETRI frame {index}: {path}")
        ok, image = capture.read()
        if not ok or image is None or image.size == 0: raise RoiPreflightError(f"Cannot read ETRI frame {index}: {path}")
        return image
    finally: capture.release()


def calculate_bbox(points: Sequence[tuple[float, float]], width: int, height: int,
                   padding_ratio: float, minimum_width: int, minimum_height: int) -> tuple[int, int, int, int] | None:
    finite = [(x, y) for x, y in points if np.isfinite(x) and np.isfinite(y)]
    if not finite or width <= 0 or height <= 0: return None
    xs = [x * width for x, _ in finite]; ys = [y * height for _, y in finite]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    box_width, box_height = x2 - x1, y2 - y1
    if box_width <= 0 or box_height <= 0: return None
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    target_width = max(float(minimum_width), box_width * (1 + 2 * padding_ratio))
    target_height = max(float(minimum_height), box_height * (1 + 2 * padding_ratio))
    target_width = min(target_width, float(width)); target_height = min(target_height, float(height))
    left = int(np.floor(center_x - target_width / 2)); top = int(np.floor(center_y - target_height / 2))
    right = int(np.ceil(left + target_width)); bottom = int(np.ceil(top + target_height))
    if left < 0: right -= left; left = 0
    if top < 0: bottom -= top; top = 0
    if right > width: left -= right - width; right = width
    if bottom > height: top -= bottom - height; bottom = height
    left, top = max(0, left), max(0, top)
    if right - left < min(minimum_width, width) or bottom - top < min(minimum_height, height): return None
    return left, top, right, bottom


def calculate_contextual_bbox(
    points: Sequence[tuple[float, float]], width: int, height: int,
    padding_ratio: float, minimum_width: int, minimum_height: int,
    minimum_width_ratio: float, minimum_height_ratio: float, vertical_anchor: float,
) -> tuple[int, int, int, int] | None:
    """Expand a single-landmark-type ROI using image-relative context extents.

    ``vertical_anchor`` is the landmark center's intended vertical position inside
    the contextual crop. A high value expands upward from a hand; a low value
    expands downward from a face.
    """
    finite = [(x, y) for x, y in points if np.isfinite(x) and np.isfinite(y)]
    if len(finite) < 2 or width <= 0 or height <= 0:
        return None
    xs = [x * width for x, _ in finite]; ys = [y * height for _, y in finite]
    raw_width, raw_height = max(xs) - min(xs), max(ys) - min(ys)
    if raw_width <= 0 or raw_height <= 0:
        return None
    target_width = min(float(width), max(float(minimum_width), width * minimum_width_ratio,
                                         raw_width * (1 + 2 * padding_ratio)))
    target_height = min(float(height), max(float(minimum_height), height * minimum_height_ratio,
                                           raw_height * (1 + 2 * padding_ratio)))
    center_x, center_y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    left = int(np.floor(center_x - target_width / 2))
    top = int(np.floor(center_y - vertical_anchor * target_height))
    right = int(np.ceil(left + target_width)); bottom = int(np.ceil(top + target_height))
    if left < 0: right -= left; left = 0
    if top < 0: bottom -= top; top = 0
    if right > width: left -= right - width; right = width
    if bottom > height: top -= bottom - height; bottom = height
    left, top = max(0, left), max(0, top)
    required_width = min(width, int(np.ceil(width * minimum_width_ratio)))
    required_height = min(height, int(np.ceil(height * minimum_height_ratio)))
    if right - left < required_width or bottom - top < required_height:
        return None
    return left, top, right, bottom


def extract_roi(image: np.ndarray, detection: Detection, config: RoiPreflightConfig) -> RoiResult:
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0: raise RoiPreflightError("ROI input must be a non-empty BGR image")
    has_face, has_hand, has_pose = bool(detection.face_points), bool(detection.hand_points), bool(detection.pose_points)
    status = "success" if has_face and has_hand else "partial" if (has_face or has_hand or has_pose) and config.allow_partial else "fallback"
    reason = ("face_and_hand_landmarks" if status == "success" else "face_only" if has_face
              else "hand_only" if has_hand else "pose_only" if has_pose else "no_landmarks")
    if status == "success":
        bbox = calculate_bbox((*detection.face_points, *detection.hand_points), image.shape[1], image.shape[0],
                              config.padding_ratio, config.minimum_width_px, config.minimum_height_px)
    elif status == "partial" and has_hand:
        bbox = calculate_contextual_bbox(
            detection.hand_points, image.shape[1], image.shape[0], config.padding_ratio,
            config.minimum_width_px, config.minimum_height_px, config.hand_only_min_width_ratio,
            config.hand_only_min_height_ratio, config.hand_only_vertical_anchor,
        )
    elif status == "partial" and has_face:
        bbox = calculate_contextual_bbox(
            detection.face_points, image.shape[1], image.shape[0], config.padding_ratio,
            config.minimum_width_px, config.minimum_height_px, config.face_only_min_width_ratio,
            config.face_only_min_height_ratio, config.face_only_vertical_anchor,
        )
    elif status == "partial" and has_pose:
        bbox = calculate_contextual_bbox(
            detection.pose_points, image.shape[1], image.shape[0], config.padding_ratio,
            config.minimum_width_px, config.minimum_height_px, config.pose_only_min_width_ratio,
            config.pose_only_min_height_ratio, config.pose_only_vertical_anchor,
        )
    else:
        bbox = None
    if status != "fallback" and bbox is None:
        status, reason = "fallback", "invalid_landmark_bbox"
    if status == "fallback": bbox = (0, 0, image.shape[1], image.shape[0])
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2].copy()
    if crop.size == 0: raise RoiPreflightError("ROI crop is unexpectedly empty")
    return RoiResult(status, bbox, reason, crop)


class MediaPipeLandmarkDetector:
    def __init__(self, face_model: Path, hand_model: Path, pose_model: Path, config: RoiPreflightConfig):
        if not face_model.is_file() or not hand_model.is_file() or not pose_model.is_file():
            raise RoiPreflightError("MediaPipe model assets are missing; run CLI with --download-models")
        base = mp.tasks.BaseOptions
        vision = mp.tasks.vision
        try:
            self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=base(model_asset_path=str(face_model)), num_faces=config.num_faces,
                min_face_detection_confidence=config.minimum_face_detection_confidence,
                min_face_presence_confidence=config.minimum_face_presence_confidence,
            ))
            self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=base(model_asset_path=str(hand_model)), num_hands=config.num_hands,
                min_hand_detection_confidence=config.minimum_hand_detection_confidence,
                min_hand_presence_confidence=config.minimum_hand_presence_confidence,
            ))
            self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=base(model_asset_path=str(pose_model)), num_poses=config.num_poses,
                min_pose_detection_confidence=config.minimum_pose_detection_confidence,
                min_pose_presence_confidence=config.minimum_pose_presence_confidence,
            ))
        except Exception as exc:
            raise RoiPreflightError(f"Cannot initialize MediaPipe detector: {exc}") from exc

    def detect(self, image_bgr: np.ndarray) -> Detection:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        try: face_result, hand_result = self.face.detect(image), self.hand.detect(image)
        except Exception as exc: raise RoiPreflightError(f"MediaPipe detection failed: {exc}") from exc
        face = tuple((float(p.x), float(p.y)) for group in face_result.face_landmarks for p in group)
        hands = tuple((float(p.x), float(p.y)) for group in hand_result.hand_landmarks for p in group)
        pose: tuple[tuple[float, float], ...] = ()
        if not face and not hands:
            try: pose_result = self.pose.detect(image)
            except Exception as exc: raise RoiPreflightError(f"MediaPipe pose detection failed: {exc}") from exc
            # Indices 0..24 cover head, shoulders, arms/hands, and hips; legs are
            # intentionally excluded because Pose serves only contextual RGB ROI.
            pose = tuple((float(p.x), float(p.y)) for group in pose_result.pose_landmarks for p in group[:25])
        return Detection(face, hands, pose)

    def close(self) -> None:
        self.face.close(); self.hand.close(); self.pose.close()


def download_model(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as target:
            shutil.copyfileobj(response, target)
        if temporary.stat().st_size == 0: raise RoiPreflightError(f"Downloaded empty model: {url}")
        temporary.replace(destination)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise RoiPreflightError(f"Cannot download MediaPipe model {url}: {exc}") from exc


def aggregate_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def stats(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        counts = Counter(str(row["roi_status"]) for row in group)
        total = len(group)
        return {"total": total, "success": counts["success"], "partial": counts["partial"],
                "fallback": counts["fallback"], "fallback_rate": counts["fallback"] / total if total else 0.0}
    by_dataset = {key: stats([row for row in records if row["dataset"] == key]) for key in sorted({str(row["dataset"]) for row in records})}
    by_class = {key: stats([row for row in records if row["target_class"] == key]) for key in sorted({str(row["target_class"]) for row in records})}
    by_hard = {str(key).lower(): stats([row for row in records if bool(row["is_hard_negative"]) is key]) for key in (False, True)}
    return {"overall": stats(records), "by_dataset": by_dataset, "by_target_class": by_class,
            "by_hard_negative": by_hard, "reason_counts": dict(sorted(Counter(str(row["reason"]) for row in records).items()))}


def run_preflight(samples: Sequence[RepresentativeSample], roots: Mapping[str, Path], detector: LandmarkDetector,
                  config: RoiPreflightConfig, output_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        for index, sample in enumerate(samples):
            image = load_sample_image(sample, roots)
            result = extract_roi(image, detector.detect(image), config)
            directory = output_root / sample.dataset
            directory.mkdir(parents=True, exist_ok=True)
            stem = f"{index:04d}_{sample.source_sample_key.replace(':', '_').replace('@', '_')}"
            preview_path, overlay_path, crop_path = directory / f"{stem}_preview.jpg", directory / f"{stem}_overlay.jpg", directory / f"{stem}_crop.jpg"
            overlay = image.copy(); x1, y1, x2, y2 = result.bbox
            cv2.rectangle(overlay, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), (0, 255, 0) if result.roi_status == "success" else (0, 165, 255) if result.roi_status == "partial" else (0, 0, 255), 3)
            for path, value in ((preview_path, image), (overlay_path, overlay), (crop_path, result.crop)):
                if not cv2.imwrite(str(path), value): raise RoiPreflightError(f"Cannot write visual output: {path}")
            records.append({"dataset": sample.dataset, "source_sample_key": sample.source_sample_key,
                            "target_class": sample.target_class, "is_hard_negative": sample.is_hard_negative,
                            "roi_status": result.roi_status, "bbox": list(result.bbox), "reason": result.reason,
                            "visual_output_path": {"preview": str(preview_path.relative_to(output_root)), "overlay": str(overlay_path.relative_to(output_root)), "crop": str(crop_path.relative_to(output_root))}})
    finally: detector.close()
    return records, aggregate_report(records)


def write_report(records: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], output_root: Path) -> tuple[Path, Path]:
    reports = output_root / "reports"; reports.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = reports / "roi_preflight_report.json", reports / "roi_preflight_samples.csv"
    json_path.write_text(json.dumps({"summary": summary, "samples": list(records)}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=REPORT_FIELDS); writer.writeheader()
        for row in records: writer.writerow({**row, "bbox": json.dumps(row["bbox"]), "visual_output_path": json.dumps(row["visual_output_path"], ensure_ascii=False)})
    return json_path, csv_path
