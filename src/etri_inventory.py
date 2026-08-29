"""ETRI Batch B RGB Full Candidate Inventory construction and validation."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import cv2


DATASET = "etri"
ROOT_KEY = "etri_raw_root"
BATCH = "B"
RGB_ROOT_RELATIVE = Path("RGB Videos")
EXPECTED_PARTICIPANTS = 30
EXPECTED_CLIPS = 6_589
EXPECTED_TARGET_COUNTS = {"A003": 119, "A004": 120}
EXPECTED_TARGET_PARTICIPANT_COVERAGE = {"A003": 30, "A004": 30}
MIN_RGB_BYTES = 300 * 1024
PARTICIPANTS = tuple(f"P{number}" for number in range(201, 231))
MULTI_PERSON_ACTIONS = frozenset({"A045", "A046", "A047", "A048"})
HARD_NEGATIVE_ACTIONS = frozenset(
    {
        "A001",
        "A010",
        "A012",
        "A013",
        "A014",
        "A015",
        "A016",
        "A017",
        "A021",
        "A035",
        "A038",
        "A040",
        "A043",
    }
)
FILENAME_RE = re.compile(
    r"^(?P<action>A\d{3})_(?P<participant>P\d{3})_"
    r"(?P<take>G\d{3})_(?P<height>H(?:070|120))\.mp4$",
    re.IGNORECASE,
)
FATAL_REASON_CODES = frozenset(
    {
        "invalid_filename",
        "invalid_action",
        "invalid_participant",
        "participant_token_mismatch",
        "invalid_take",
        "invalid_height",
        "missing_mp4",
        "duplicate_clip",
    }
)
INVENTORY_FIELDS = (
    "dataset",
    "root_key",
    "relative_path",
    "batch",
    "participant",
    "action",
    "target_class",
    "fold",
    "height",
    "take",
    "is_multi_person",
    "is_hard_negative",
    "valid",
    "exclusion_reason",
    "pilot_selected",
    "pilot_selection_reason",
    "pilot_seed",
    "roi_status",
)


class EtriInventoryError(RuntimeError):
    """Raised when ETRI inventory construction cannot proceed safely."""


@dataclass
class EtriInventoryRecord:
    dataset: str = DATASET
    root_key: str = ROOT_KEY
    relative_path: str = ""
    batch: str = BATCH
    participant: str | None = None
    action: str | None = None
    target_class: str | None = None
    fold: str = "unassigned"
    height: str | None = None
    take: str | None = None
    is_multi_person: bool = False
    is_hard_negative: bool = False
    valid: bool = True
    exclusion_reason: str = ""
    pilot_selected: bool = False
    pilot_selection_reason: str = ""
    pilot_seed: int | None = None
    roi_status: str = "pending"
    rgb_size_bytes: int | None = None
    rgb_width: int | None = None
    rgb_height: int | None = None
    rgb_frame_count: int | None = None
    rgb_fps: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EtriBuildResult:
    records: list[EtriInventoryRecord]
    summary: dict[str, Any]


def map_etri_class(action: str) -> str:
    if action == "A003":
        return "복약"
    if action == "A004":
        return "음수"
    if re.fullmatch(r"A\d{3}", action) and 1 <= int(action[1:]) <= 55:
        return "기타"
    raise ValueError(f"Invalid ETRI action: {action!r}")


def _add_reason(reasons: list[str], code: str, detail: str = "") -> None:
    reason = f"{code}: {detail}" if detail else code
    if reason not in reasons:
        reasons.append(reason)


def _safe_relative(path: Path, raw_root: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(raw_root).as_posix()
    except ValueError as exc:
        raise EtriInventoryError(f"Path escapes ETRI Raw root: {path}") from exc


def discover_batch_b_mp4(
    raw_root: Path, max_candidates: int | None = None
) -> Iterator[tuple[str, Path]]:
    """Yield Batch B RGB candidates without scanning Batch A."""

    rgb_root = raw_root / RGB_ROOT_RELATIVE
    if not rgb_root.is_dir():
        raise EtriInventoryError(f"ETRI RGB root does not exist: {rgb_root}")
    emitted = 0
    for participant in PARTICIPANTS:
        participant_root = rgb_root / participant
        if not participant_root.is_dir():
            raise EtriInventoryError(f"ETRI Batch B participant root does not exist: {participant_root}")
        for path in sorted(participant_root.iterdir(), key=lambda item: item.name):
            if not path.is_file() or path.suffix.lower() != ".mp4":
                continue
            yield participant, path
            emitted += 1
            if max_candidates is not None and emitted >= max_candidates:
                return


def _read_rgb_metadata(path: Path, reasons: list[str]) -> dict[str, int | float | None]:
    metadata: dict[str, int | float | None] = {
        "width": None,
        "height": None,
        "frame_count": None,
        "fps": None,
    }
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            _add_reason(reasons, "rgb_open_failed")
            return metadata
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        metadata.update(width=width, height=height, frame_count=frame_count, fps=fps)
        if width <= 0 or height <= 0 or frame_count <= 0 or fps <= 0:
            _add_reason(
                reasons,
                "rgb_invalid_metadata",
                f"width={width}, height={height}, frames={frame_count}, fps={fps}",
            )
    finally:
        capture.release()
    return metadata


def build_candidate_record(
    mp4_path: Path,
    raw_root: Path,
    participant_directory: str,
    validate_rgb_metadata: bool = True,
) -> EtriInventoryRecord:
    """Parse and validate one Batch B MP4 without modifying Raw data."""

    raw_root = raw_root.resolve(strict=True)
    reasons: list[str] = []
    record = EtriInventoryRecord(
        relative_path=_safe_relative(mp4_path, raw_root),
        participant=participant_directory,
    )
    if not mp4_path.is_file():
        _add_reason(reasons, "missing_mp4")
        record.valid = False
        record.exclusion_reason = "; ".join(reasons)
        return record

    name_match = FILENAME_RE.fullmatch(mp4_path.name)
    if not name_match:
        _add_reason(reasons, "invalid_filename", mp4_path.name)
    else:
        action = name_match["action"].upper()
        participant = name_match["participant"].upper()
        take = name_match["take"].upper()
        height = name_match["height"].upper()
        record.action = action
        record.participant = participant
        record.take = take
        record.height = height
        if participant not in PARTICIPANTS:
            _add_reason(reasons, "invalid_participant", participant)
        if participant != participant_directory:
            _add_reason(
                reasons,
                "participant_token_mismatch",
                f"directory={participant_directory}, filename={participant}",
            )
        if not 1 <= int(action[1:]) <= 55:
            _add_reason(reasons, "invalid_action", action)
        take_number = int(take[1:])
        expected_take_range = range(1, 13) if participant_directory in PARTICIPANTS[:6] else range(1, 3)
        if take_number not in expected_take_range:
            _add_reason(reasons, "invalid_take", f"{take} for {participant_directory}")
        if height not in {"H070", "H120"}:
            _add_reason(reasons, "invalid_height", height)
        try:
            record.target_class = map_etri_class(action)
        except ValueError as exc:
            _add_reason(reasons, "invalid_action", str(exc))
        record.is_multi_person = action in MULTI_PERSON_ACTIONS
        record.is_hard_negative = action in HARD_NEGATIVE_ACTIONS
        if record.is_multi_person:
            record.pilot_selection_reason = "excluded_multi_person_action"

    record.rgb_size_bytes = mp4_path.stat().st_size
    if record.rgb_size_bytes < MIN_RGB_BYTES:
        _add_reason(
            reasons,
            "rgb_file_too_small",
            f"{record.rgb_size_bytes} < {MIN_RGB_BYTES} bytes",
        )
    if validate_rgb_metadata:
        metadata = _read_rgb_metadata(mp4_path, reasons)
        record.rgb_width = metadata["width"]  # type: ignore[assignment]
        record.rgb_height = metadata["height"]  # type: ignore[assignment]
        record.rgb_frame_count = metadata["frame_count"]  # type: ignore[assignment]
        record.rgb_fps = metadata["fps"]  # type: ignore[assignment]

    record.valid = not reasons
    record.exclusion_reason = "; ".join(reasons)
    return record


def _mark_duplicate_clips(records: Sequence[EtriInventoryRecord]) -> int:
    by_key: dict[tuple[str, str, str, str], list[EtriInventoryRecord]] = defaultdict(list)
    for record in records:
        if all((record.participant, record.action, record.take, record.height)):
            key = (record.participant, record.action, record.take, record.height)  # type: ignore[arg-type]
            by_key[key].append(record)
    duplicate_count = 0
    for key, matches in by_key.items():
        if len(matches) < 2:
            continue
        duplicate_count += len(matches)
        for record in matches:
            record.valid = False
            reason = f"duplicate_clip: {'/'.join(key)}"
            record.exclusion_reason = "; ".join(filter(None, [record.exclusion_reason, reason]))
    return duplicate_count


def classify_etri_status(
    records: Sequence[EtriInventoryRecord],
    count_mismatches: Mapping[str, Any],
    duplicate_clips: int,
) -> dict[str, Any]:
    invalid_records = [record for record in records if not record.valid]
    unisolated = [
        record
        for record in invalid_records
        if not record.exclusion_reason.strip() or record.pilot_selected
    ]
    fatal = []
    for record in invalid_records:
        codes = {
            reason.split(":", 1)[0].strip()
            for reason in record.exclusion_reason.split("; ")
            if reason.strip()
        }
        if codes & FATAL_REASON_CODES:
            fatal.append(record)
    build_failed = bool(count_mismatches or duplicate_clips or unisolated or fatal)
    if build_failed:
        status, build_status = "FAIL", "FAIL"
        data_quality_status = "ERROR" if invalid_records else "PASS"
    elif invalid_records:
        status, build_status, data_quality_status = "PASS_WITH_WARNINGS", "PASS", "WARNING"
    else:
        status, build_status, data_quality_status = "PASS", "PASS", "PASS"
    return {
        "status": status,
        "build_status": build_status,
        "data_quality_status": data_quality_status,
        "invalid_candidates": len(invalid_records),
        "usable_candidates": len(records) - len(invalid_records),
        "unisolated_invalid_candidates": len(unisolated),
        "fatal_invalid_candidates": len(fatal),
    }


def validate_pilot_selection(records: Sequence[EtriInventoryRecord]) -> None:
    violations = [
        record.relative_path
        for record in records
        if record.pilot_selected and (not record.valid or record.is_multi_person)
    ]
    if violations:
        raise EtriInventoryError(
            f"Pilot selection contains {len(violations)} invalid or multi-person candidate(s): "
            + ", ".join(violations[:3])
        )


def build_inventory(
    raw_root: Path,
    max_candidates: int | None = None,
    validate_rgb_metadata: bool = True,
) -> EtriBuildResult:
    raw_root = raw_root.resolve(strict=True)
    records = [
        build_candidate_record(path, raw_root, participant, validate_rgb_metadata)
        for participant, path in discover_batch_b_mp4(raw_root, max_candidates)
    ]
    duplicate_count = _mark_duplicate_clips(records)
    full_run = max_candidates is None
    participant_set = {record.participant for record in records if record.participant in PARTICIPANTS}
    action_counts = Counter(record.action for record in records if record.action)
    coverage = {
        action: len({record.participant for record in records if record.action == action})
        for action in EXPECTED_TARGET_PARTICIPANT_COVERAGE
    }
    observed_counts = {
        "participants": len(participant_set),
        "rgb_clips": len(records),
        "A003": action_counts["A003"],
        "A004": action_counts["A004"],
        "A003_participant_coverage": coverage["A003"],
        "A004_participant_coverage": coverage["A004"],
    }
    expected_counts = {
        "participants": EXPECTED_PARTICIPANTS,
        "rgb_clips": EXPECTED_CLIPS,
        "A003": EXPECTED_TARGET_COUNTS["A003"],
        "A004": EXPECTED_TARGET_COUNTS["A004"],
        "A003_participant_coverage": EXPECTED_TARGET_PARTICIPANT_COVERAGE["A003"],
        "A004_participant_coverage": EXPECTED_TARGET_PARTICIPANT_COVERAGE["A004"],
    }
    count_mismatches = {
        key: {"expected": expected_counts[key], "observed": value}
        for key, value in observed_counts.items()
        if full_run and value != expected_counts[key]
    }
    reason_counts: Counter[str] = Counter()
    for record in records:
        for reason in record.exclusion_reason.split("; ") if record.exclusion_reason else []:
            reason_counts[reason.split(":", 1)[0]] += 1
    status_fields = classify_etri_status(records, count_mismatches, duplicate_count)
    summary = {
        **status_fields,
        "full_run": full_run,
        "observed_counts": observed_counts,
        "expected_counts": expected_counts if full_run else None,
        "count_mismatches": count_mismatches,
        "duplicate_clips": duplicate_count,
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "multi_person_candidates": sum(record.is_multi_person for record in records),
        "hard_negative_candidates": sum(record.is_hard_negative for record in records),
    }
    return EtriBuildResult(records=records, summary=summary)


def write_inventory(result: EtriBuildResult, output_dir: Path) -> tuple[Path, Path]:
    output_dir = output_dir.resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "etri_batch_b_full_candidate_inventory.jsonl"
    summary_path = output_dir / "etri_batch_b_inventory_summary.json"
    inventory_tmp = inventory_path.with_suffix(inventory_path.suffix + ".tmp")
    summary_tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with inventory_tmp.open("w", encoding="utf-8") as output:
        for record in result.records:
            output.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    summary_tmp.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    inventory_tmp.replace(inventory_path)
    summary_tmp.replace(summary_path)
    return inventory_path, summary_path
