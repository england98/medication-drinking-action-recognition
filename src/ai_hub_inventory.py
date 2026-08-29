"""AI-Hub Full Candidate Inventory construction and validation."""

from __future__ import annotations

import json
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence
from xml.etree import ElementTree


DATASET = "ai_hub"
ROOT_KEY = "ai_hub_raw_root"
EXPECTED_JSON_COUNT = 18_420
EXPECTED_JPG_COUNT = 55_260
EXPECTED_VIEWPOINT_3_COUNT = 10_988
INVENTORY_FIELDS = (
    "dataset",
    "root_key",
    "relative_path",
    "frame_relative_paths",
    "video_id",
    "source_split",
    "actor",
    "viewpoint",
    "original_class",
    "target_class",
    "target_role",
    "split",
    "is_hard_negative",
    "valid",
    "exclusion_reason",
    "pilot_selected",
    "pilot_selection_reason",
    "pilot_seed",
    "roi_status",
)

SOURCE_LAYOUT = {
    "Training": Path("1.Training") / "라벨링데이터",
    "Validation": Path("2.Validation") / "라벨링데이터",
}
DEFAULT_METADATA_RELATIVE_PATH = (
    Path("1.Training")
    / "메타데이터_230525_add"
    / "metadata"
    / "(2021-1-35-57) metadata.csv.xlsx"
)

JSON_NAME_RE = re.compile(
    r"^C(?P<category>\d{3})_A(?P<age>[1-4])_(?P<gender>[MF])"
    r"(?P<video_id>\d{5})_V(?P<viewpoint>[13])\.json$"
)
CATEGORY_DIR_RE = re.compile(r"^[TV]LCategory(?P<category>\d{3})$")
VIEWPOINT_DIR_RE = re.compile(r"^viewpoint_(?P<viewpoint>[13])$")
METADATA_COLUMNS = (
    "Filename",
    "video_id",
    "category_id",
    "actor_id",
    "viewpoint",
    "gender",
    "height",
    "age",
    "family_number",
    "job",
    "region",
    "place",
    "interact_person",
    "interact_ICT",
    "explan",
    "video_date",
    "video_time",
    "video_length",
    "record_device",
)
EXPECTED_MAPPED_CLASS_NAMES = {
    1: "Eat_food",
    2: "Drink_bever",
    3: "Take_pills",
    5: "Drink_alcohol",
}
FATAL_EXCLUSION_REASON_CODES = {
    "metadata_not_matched",
    "metadata_video_id_mismatch",
    "metadata_category_id_mismatch",
    "metadata_actor_mismatch",
    "metadata_viewpoint_mismatch",
    "duplicate_candidate_video_id",
    "invalid_class_mapping",
    "invalid_class_mapping_name",
}


class InventoryBuildError(RuntimeError):
    """Raised when inventory construction cannot proceed safely."""


@dataclass
class InventoryRecord:
    dataset: str = DATASET
    root_key: str = ROOT_KEY
    relative_path: str = ""
    frame_relative_paths: list[str] = field(default_factory=list)
    video_id: str | None = None
    source_split: str | None = None
    actor: int | None = None
    viewpoint: int | None = None
    original_class: str | None = None
    target_class: str | None = None
    target_role: str | None = None
    split: str = "unassigned"
    is_hard_negative: bool = False
    valid: bool = True
    exclusion_reason: str = ""
    pilot_selected: bool = False
    pilot_selection_reason: str = ""
    pilot_seed: int | None = None
    roi_status: str = "pending"
    category_id: int | None = None
    metadata_matched: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BuildResult:
    records: list[InventoryRecord]
    summary: dict[str, Any]


def classify_inventory_status(
    records: Sequence[InventoryRecord],
    count_mismatches: Mapping[str, Any],
    metadata_unmatched: int | None,
    duplicate_candidates: int,
) -> dict[str, Any]:
    """Separate inventory completeness failures from isolated Raw-data warnings."""

    invalid_records = [record for record in records if not record.valid]
    unisolated_invalid = [
        record
        for record in invalid_records
        if not record.exclusion_reason.strip() or record.pilot_selected
    ]
    fatal_invalid = []
    for record in invalid_records:
        reason_codes = {
            reason.split(":", 1)[0].strip()
            for reason in record.exclusion_reason.split("; ")
            if reason.strip()
        }
        if reason_codes & FATAL_EXCLUSION_REASON_CODES:
            fatal_invalid.append(record)

    build_failed = bool(
        count_mismatches
        or metadata_unmatched not in {0, None}
        or duplicate_candidates
        or unisolated_invalid
        or fatal_invalid
    )
    if build_failed:
        status = "FAIL"
        build_status = "FAIL"
        data_quality_status = "ERROR" if invalid_records else "PASS"
    elif invalid_records:
        status = "PASS_WITH_WARNINGS"
        build_status = "PASS"
        data_quality_status = "WARNING"
    else:
        status = "PASS"
        build_status = "PASS"
        data_quality_status = "PASS"

    return {
        "status": status,
        "build_status": build_status,
        "data_quality_status": data_quality_status,
        "invalid_candidates": len(invalid_records),
        "usable_candidates": len(records) - len(invalid_records),
        "unisolated_invalid_candidates": len(unisolated_invalid),
        "fatal_invalid_candidates": len(fatal_invalid),
    }


def validate_pilot_selection(records: Sequence[InventoryRecord]) -> None:
    """Fail when Phase 3 selects any candidate already marked invalid."""

    violations = [record.relative_path for record in records if record.pilot_selected and not record.valid]
    if violations:
        preview = ", ".join(violations[:3])
        raise InventoryBuildError(
            f"Pilot selection contains {len(violations)} invalid candidate(s): {preview}"
        )


def map_ai_hub_class(category_id: int, original_class: str | None) -> tuple[str, str, bool]:
    """Map the folder-number category taxonomy to the Pilot taxonomy."""

    if not isinstance(category_id, int) or isinstance(category_id, bool) or category_id <= 0:
        raise ValueError(f"Invalid AI-Hub category ID: {category_id!r}")
    if not isinstance(original_class, str) or not original_class.strip():
        raise ValueError("AI-Hub original class must be a non-empty string.")
    if category_id == 3:
        return "복약", "direct_target", False
    if category_id in {2, 5}:
        return "음수", "auxiliary_positive", False
    return "기타", "other", category_id == 1


def discover_json_files(raw_root: Path, max_candidates: int | None = None) -> Iterator[tuple[str, Path]]:
    """Yield local Training/Validation JSON files; metadata never creates candidates."""

    emitted = 0
    for source_split, relative_root in SOURCE_LAYOUT.items():
        label_root = raw_root / relative_root
        if not label_root.is_dir():
            raise InventoryBuildError(f"AI-Hub label root does not exist: {label_root}")
        for directory, directory_names, filenames in label_root.walk():
            directory_names.sort()
            for filename in sorted(filenames):
                if not filename.endswith(".json"):
                    continue
                yield source_split, directory / filename
                emitted += 1
                if max_candidates is not None and emitted >= max_candidates:
                    return


def _safe_relative(path: Path, raw_root: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        relative = resolved.relative_to(raw_root)
    except ValueError as exc:
        raise InventoryBuildError(f"Path escapes AI-Hub Raw root: {path}") from exc
    return relative.as_posix()


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _add_reason(reasons: list[str], code: str, detail: str = "") -> None:
    reason = f"{code}: {detail}" if detail else code
    if reason not in reasons:
        reasons.append(reason)


def _path_facts(json_path: Path, raw_root: Path, source_split: str) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    relative = _safe_relative(json_path, raw_root)
    parts = PurePosixPath(relative).parts
    facts: dict[str, Any] = {
        "relative_path": relative,
        "source_split": source_split,
        "video_id": json_path.parent.name if json_path.parent.name.isdigit() else None,
        "category_id": None,
        "viewpoint": None,
        "age": None,
        "gender": None,
    }

    name_match = JSON_NAME_RE.fullmatch(json_path.name)
    if not name_match:
        _add_reason(reasons, "invalid_json_filename", json_path.name)
    else:
        facts.update(
            video_id=name_match["video_id"],
            category_id=int(name_match["category"]),
            viewpoint=int(name_match["viewpoint"]),
            age=int(name_match["age"]),
            gender={"M": "male", "F": "female"}[name_match["gender"]],
        )

    if len(parts) < 7:
        _add_reason(reasons, "invalid_directory_layout", relative)
        return facts, reasons

    viewpoint_match = VIEWPOINT_DIR_RE.fullmatch(parts[2])
    category_match = CATEGORY_DIR_RE.fullmatch(parts[3])
    directory_facts = {
        "viewpoint": int(viewpoint_match["viewpoint"]) if viewpoint_match else None,
        "category_id": int(category_match["category"]) if category_match else None,
        "age": int(parts[4][3:]) if re.fullmatch(r"age[1-4]", parts[4]) else None,
        "gender": parts[5] if parts[5] in {"male", "female"} else None,
        "video_id": parts[6] if re.fullmatch(r"\d{5}", parts[6]) else None,
    }
    for key, directory_value in directory_facts.items():
        filename_value = facts.get(key)
        if directory_value is None:
            _add_reason(reasons, f"invalid_{key}_directory", parts[{"viewpoint": 2, "category_id": 3, "age": 4, "gender": 5, "video_id": 6}[key]])
        elif filename_value is not None and filename_value != directory_value:
            _add_reason(reasons, f"filename_directory_{key}_mismatch", f"{filename_value} != {directory_value}")
    return facts, reasons


def build_candidate_record(json_path: Path, raw_root: Path, source_split: str) -> InventoryRecord:
    """Parse and validate one local JSON candidate without modifying Raw data."""

    raw_root = raw_root.resolve(strict=True)
    facts, reasons = _path_facts(json_path, raw_root, source_split)
    record = InventoryRecord(
        relative_path=facts["relative_path"],
        video_id=facts["video_id"],
        source_split=source_split,
        viewpoint=facts["viewpoint"],
        category_id=facts["category_id"],
    )
    try:
        document = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _add_reason(reasons, "json_parse_error", str(exc))
        sibling_frames = sorted(json_path.parent.glob(f"{json_path.stem}_image_*.jpg"))
        record.frame_relative_paths = [_safe_relative(path, raw_root) for path in sibling_frames]
        record.valid = False
        record.exclusion_reason = "; ".join(reasons)
        return record

    if not isinstance(document, Mapping):
        _add_reason(reasons, "json_root_not_object")
        document = {}
    categories = document.get("categories")
    video_meta = document.get("video.meta")
    images = document.get("images")
    if not isinstance(categories, Mapping):
        _add_reason(reasons, "missing_categories_object")
        categories = {}
    if not isinstance(video_meta, Mapping):
        _add_reason(reasons, "missing_video_meta_object")
        video_meta = {}
    if not isinstance(images, list):
        _add_reason(reasons, "missing_images_list")
        images = []

    json_category = _as_int(categories.get("id"))
    meta_category = _as_int(video_meta.get("category_id"))
    actor = _as_int(video_meta.get("actor"))
    viewpoint = _as_int(video_meta.get("viewpoint"))
    age = _as_int(video_meta.get("age"))
    gender = video_meta.get("gender")
    video_id_value = video_meta.get("video_id")
    video_id = str(video_id_value).zfill(5) if video_id_value is not None else None
    original_class = categories.get("name")

    required_values = {
        "category_id": json_category,
        "video_meta_category_id": meta_category,
        "actor": actor,
        "viewpoint": viewpoint,
        "video_id": video_id,
        "age": age,
        "gender": gender if gender in {"male", "female"} else None,
    }
    for key, value in required_values.items():
        if value is None:
            _add_reason(reasons, f"missing_or_invalid_{key}")
    if not isinstance(original_class, str) or not original_class.strip():
        _add_reason(reasons, "missing_or_invalid_original_class")

    expected = {
        "category_id": facts["category_id"],
        "viewpoint": facts["viewpoint"],
        "video_id": facts["video_id"],
        "age": facts["age"],
        "gender": facts["gender"],
    }
    actual = {
        "category_id": json_category,
        "viewpoint": viewpoint,
        "video_id": video_id,
        "age": age,
        "gender": gender if gender in {"male", "female"} else None,
    }
    for key in expected:
        if expected[key] is not None and actual[key] is not None and expected[key] != actual[key]:
            _add_reason(reasons, f"path_json_{key}_mismatch", f"{expected[key]} != {actual[key]}")
    if json_category is not None and meta_category is not None and json_category != meta_category:
        _add_reason(reasons, "json_category_fields_mismatch", f"{json_category} != {meta_category}")

    frame_paths: list[str] = []
    if len(images) != 3:
        _add_reason(reasons, "unexpected_frame_count", f"expected 3, got {len(images)}")
    for image in images:
        if not isinstance(image, Mapping) or not isinstance(image.get("filename"), str):
            _add_reason(reasons, "invalid_image_entry")
            continue
        filename = image["filename"]
        if Path(filename).name != filename:
            _add_reason(reasons, "unsafe_image_filename", filename)
            continue
        expected_prefix = json_path.stem + "_image_"
        if not filename.startswith(expected_prefix) or not filename.lower().endswith(".jpg"):
            _add_reason(reasons, "json_image_filename_mismatch", filename)
        frame_path = json_path.parent / filename
        frame_paths.append(_safe_relative(frame_path, raw_root))
        if not frame_path.is_file():
            _add_reason(reasons, "missing_jpg", filename)
    if len(frame_paths) != len(set(frame_paths)):
        _add_reason(reasons, "duplicate_frame_reference")

    record.frame_relative_paths = frame_paths
    record.video_id = video_id or record.video_id
    record.actor = actor
    record.viewpoint = viewpoint if viewpoint is not None else record.viewpoint
    record.original_class = original_class.strip() if isinstance(original_class, str) else None
    record.category_id = json_category if json_category is not None else record.category_id
    try:
        record.target_class, record.target_role, record.is_hard_negative = map_ai_hub_class(
            record.category_id, record.original_class
        )
    except ValueError as exc:
        _add_reason(reasons, "invalid_class_mapping", str(exc))
    expected_class_name = EXPECTED_MAPPED_CLASS_NAMES.get(record.category_id)
    if expected_class_name is not None and record.original_class != expected_class_name:
        _add_reason(
            reasons,
            "invalid_class_mapping_name",
            f"category {record.category_id} expects {expected_class_name}, got {record.original_class}",
        )
    record.valid = not reasons
    record.exclusion_reason = "; ".join(reasons)
    return record


def _column_index(cell_reference: str) -> int:
    letters = "".join(character for character in cell_reference if character.isalpha())
    result = 0
    for character in letters.upper():
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def _xlsx_rows(path: Path, sheet_name: str) -> Iterator[list[Any]]:
    """Read scalar XLSX rows using only the Python standard library."""

    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
    office_rel = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise InventoryBuildError(f"Cannot open metadata XLSX: {path}") from exc
    with archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("m:si", ns):
                shared_strings.append("".join(node.text or "" for node in item.iterfind(".//m:t", ns)))
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relationship_targets = {
            item.attrib["Id"]: item.attrib["Target"] for item in relationships.findall("r:Relationship", rel_ns)
        }
        target = None
        for sheet in workbook.findall("m:sheets/m:sheet", ns):
            if sheet.attrib.get("name") == sheet_name:
                target = relationship_targets.get(sheet.attrib[office_rel])
                break
        if target is None:
            raise InventoryBuildError(f"Metadata sheet not found: {sheet_name}")
        sheet_path = target.lstrip("/") if target.startswith("/xl/") else "xl/" + target.lstrip("/")
        sheet_root = ElementTree.fromstring(archive.read(sheet_path))
        for row in sheet_root.findall(".//m:sheetData/m:row", ns):
            values: list[Any] = []
            for cell in row.findall("m:c", ns):
                index = _column_index(cell.attrib.get("r", "A1"))
                while len(values) <= index:
                    values.append(None)
                cell_type = cell.attrib.get("t")
                value_node = cell.find("m:v", ns)
                if cell_type == "inlineStr":
                    value = "".join(node.text or "" for node in cell.iterfind(".//m:t", ns))
                elif value_node is None:
                    value = None
                elif cell_type == "s":
                    value = shared_strings[int(value_node.text or "0")]
                elif cell_type == "b":
                    value = value_node.text == "1"
                else:
                    text = value_node.text or ""
                    try:
                        number = float(text)
                        value = int(number) if number.is_integer() else number
                    except ValueError:
                        value = text
                values[index] = value
            yield values


def load_metadata_index(metadata_path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    """Index corrected metadata rows without treating XLSX-only rows as candidates."""

    rows = _xlsx_rows(metadata_path, "video_meta")
    try:
        headers = [str(value).strip() if value is not None else "" for value in next(rows)]
    except StopIteration as exc:
        raise InventoryBuildError("Metadata sheet 'video_meta' is empty.") from exc
    missing = [column for column in METADATA_COLUMNS if column not in headers]
    if missing:
        raise InventoryBuildError(f"Metadata is missing required column(s): {', '.join(missing)}")
    index: dict[str, dict[str, Any]] = {}
    duplicate_keys: set[str] = set()
    row_count = 0
    for values in rows:
        row = {header: values[position] if position < len(values) else None for position, header in enumerate(headers)}
        if not any(value is not None for value in row.values()):
            continue
        row_count += 1
        filename = row.get("Filename")
        video_id = row.get("video_id")
        keys = set()
        if filename is not None:
            filename_key = Path(str(filename)).stem
            keys.add(filename_key)
        if video_id is not None:
            keys.add(str(video_id).split(".")[0].zfill(5))
        for key in keys:
            if key in index and index[key] != row:
                duplicate_keys.add(key)
            index[key] = row
    if duplicate_keys:
        raise InventoryBuildError(f"Duplicate metadata join key(s): {len(duplicate_keys)}")
    return index, row_count


def join_metadata(records: Sequence[InventoryRecord], metadata_index: Mapping[str, dict[str, Any]]) -> int:
    """Left-join metadata onto local candidates and return the match count."""

    matches = 0
    for record in records:
        filename_key = Path(record.relative_path).stem
        metadata = metadata_index.get(filename_key)
        if metadata is None and record.video_id is not None:
            metadata = metadata_index.get(record.video_id)
        if metadata is None:
            record.valid = False
            record.exclusion_reason = "; ".join(
                filter(None, [record.exclusion_reason, "metadata_not_matched"])
            )
            continue
        record.metadata_matched = True
        record.metadata = dict(metadata)
        matches += 1
        mismatches = []
        comparisons = {
            "video_id": (record.video_id, str(metadata.get("video_id")).split(".")[0].zfill(5)),
            "category_id": (record.category_id, _as_int(metadata.get("category_id"))),
            "actor": (record.actor, _as_int(metadata.get("actor_id"))),
            "viewpoint": (record.viewpoint, _as_int(metadata.get("viewpoint"))),
        }
        for key, (local_value, metadata_value) in comparisons.items():
            if local_value is not None and metadata_value is not None and local_value != metadata_value:
                mismatches.append(f"metadata_{key}_mismatch: {local_value} != {metadata_value}")
        if mismatches:
            record.valid = False
            record.exclusion_reason = "; ".join(filter(None, [record.exclusion_reason, *mismatches]))
    return matches


def _mark_duplicate_candidates(records: Sequence[InventoryRecord]) -> int:
    by_video_id: dict[str, list[InventoryRecord]] = defaultdict(list)
    for record in records:
        if record.video_id:
            by_video_id[record.video_id].append(record)
    duplicate_count = 0
    for video_id, matches in by_video_id.items():
        if len(matches) < 2:
            continue
        duplicate_count += len(matches)
        for record in matches:
            record.valid = False
            reason = f"duplicate_candidate_video_id: {video_id}"
            record.exclusion_reason = "; ".join(filter(None, [record.exclusion_reason, reason]))
    return duplicate_count


def _count_local_jpg_files(raw_root: Path) -> int:
    count = 0
    for relative_root in SOURCE_LAYOUT.values():
        label_root = raw_root / relative_root
        for _directory, _directory_names, filenames in label_root.walk():
            count += sum(filename.lower().endswith(".jpg") for filename in filenames)
    return count


def build_inventory(
    raw_root: Path,
    metadata_path: Path | None = None,
    max_candidates: int | None = None,
    use_metadata: bool = True,
) -> BuildResult:
    """Build the in-memory inventory and its validation summary."""

    raw_root = raw_root.resolve(strict=True)
    records = [
        build_candidate_record(path, raw_root, source_split)
        for source_split, path in discover_json_files(raw_root, max_candidates=max_candidates)
    ]
    duplicate_count = _mark_duplicate_candidates(records)
    metadata_rows = 0
    metadata_matches = 0
    if use_metadata:
        resolved_metadata = (metadata_path or raw_root / DEFAULT_METADATA_RELATIVE_PATH).resolve(strict=True)
        metadata_index, metadata_rows = load_metadata_index(resolved_metadata)
        metadata_matches = join_metadata(records, metadata_index)

    reason_counts: Counter[str] = Counter()
    for record in records:
        for reason in record.exclusion_reason.split("; ") if record.exclusion_reason else []:
            reason_counts[reason.split(":", 1)[0]] += 1
    full_run = max_candidates is None
    observed = {
        "json_candidates": len(records),
        "jpg_references": sum(len(record.frame_relative_paths) for record in records),
        "jpg_files": _count_local_jpg_files(raw_root) if full_run else None,
        "viewpoint_3_candidates": sum(record.viewpoint == 3 for record in records),
    }
    expected = {
        "json_candidates": EXPECTED_JSON_COUNT,
        "jpg_references": EXPECTED_JPG_COUNT,
        "jpg_files": EXPECTED_JPG_COUNT,
        "viewpoint_3_candidates": EXPECTED_VIEWPOINT_3_COUNT,
    }
    count_mismatches = {
        key: {"expected": expected[key], "observed": value}
        for key, value in observed.items()
        if full_run and value != expected[key]
    }
    invalid_count = sum(not record.valid for record in records)
    metadata_unmatched = len(records) - metadata_matches if use_metadata else None
    status_fields = classify_inventory_status(
        records=records,
        count_mismatches=count_mismatches,
        metadata_unmatched=metadata_unmatched,
        duplicate_candidates=duplicate_count,
    )
    summary = {
        **status_fields,
        "full_run": full_run,
        "observed_counts": observed,
        "expected_counts": expected if full_run else None,
        "count_mismatches": count_mismatches,
        "valid_candidates": len(records) - invalid_count,
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "duplicate_candidates": duplicate_count,
        "metadata_enabled": use_metadata,
        "metadata_rows": metadata_rows,
        "metadata_matches": metadata_matches,
        "metadata_unmatched_local_candidates": metadata_unmatched,
        "metadata_only_rows": metadata_rows - metadata_matches if use_metadata and full_run else None,
    }
    return BuildResult(records=records, summary=summary)


def write_inventory(result: BuildResult, output_dir: Path) -> tuple[Path, Path]:
    """Atomically write JSONL inventory and summary outside Raw data."""

    output_dir = output_dir.resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "ai_hub_full_candidate_inventory.jsonl"
    summary_path = output_dir / "ai_hub_inventory_summary.json"
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
