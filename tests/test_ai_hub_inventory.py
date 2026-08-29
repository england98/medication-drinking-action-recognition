from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.build_ai_hub_inventory import exit_code_for_status
from src.ai_hub_inventory import (
    INVENTORY_FIELDS,
    InventoryBuildError,
    InventoryRecord,
    build_candidate_record,
    build_inventory,
    classify_inventory_status,
    load_metadata_index,
    map_ai_hub_class,
    validate_pilot_selection,
    write_inventory,
)


def _candidate_dir(root: Path, split: str = "Training", category: int = 3) -> Path:
    split_dir = "1.Training" if split == "Training" else "2.Validation"
    category_prefix = "TL" if split == "Training" else "VL"
    path = (
        root
        / split_dir
        / "라벨링데이터"
        / "viewpoint_3"
        / f"{category_prefix}Category{category:03d}"
        / "age1"
        / "male"
        / "12345"
    )
    path.mkdir(parents=True)
    return path


def _write_candidate(root: Path, category: int = 3, class_name: str = "Take_pills") -> Path:
    directory = _candidate_dir(root, category=category)
    stem = f"C{category:03d}_A1_M12345_V3"
    images = []
    for frame_index in (10, 20, 30):
        filename = f"{stem}_image_{frame_index}.jpg"
        (directory / filename).write_bytes(b"jpg")
        images.append({"id": filename.removesuffix(".jpg"), "filename": filename})
    document = {
        "categories": {"id": category, "name": class_name},
        "images": images,
        "video.meta": {
            "video_id": "12345",
            "category_id": category,
            "actor": 7,
            "viewpoint": 3,
            "age": 1,
            "gender": "male",
        },
    }
    json_path = directory / f"{stem}.json"
    json_path.write_text(json.dumps(document), encoding="utf-8")
    return json_path


def _write_minimal_metadata_xlsx(path: Path, data_rows: list[list[object]]) -> None:
    headers = [
        "Filename", "video_id", "category_id", "actor_id", "viewpoint", "gender",
        "height", "age", "family_number", "job", "region", "place", "interact_person",
        "interact_ICT", "explan", "video_date", "video_time", "video_length", "record_device",
    ]

    def cell(reference: str, value: object) -> str:
        if isinstance(value, (int, float)):
            return f'<c r="{reference}"><v>{value}</v></c>'
        escaped = str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f'<c r="{reference}" t="inlineStr"><is><t>{escaped}</t></is></c>'

    def column_name(index: int) -> str:
        result = ""
        while index:
            index, remainder = divmod(index - 1, 26)
            result = chr(65 + remainder) + result
        return result

    rows = [headers, *data_rows]
    row_xml = []
    for row_number, values in enumerate(rows, start=1):
        cells = "".join(cell(f"{column_name(index)}{row_number}", value) for index, value in enumerate(values, start=1))
        row_xml.append(f'<row r="{row_number}">{cells}</row>')
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(row_xml)}</sheetData></worksheet>'
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="video_meta" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Target="worksheets/sheet1.xml" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/>'
        '</Relationships>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)


class AiHubInventoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        for split_dir in ("1.Training", "2.Validation"):
            (self.root / split_dir / "라벨링데이터").mkdir(parents=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_valid_candidate_has_required_schema_and_initial_state(self) -> None:
        json_path = _write_candidate(self.root)
        record = build_candidate_record(json_path, self.root, "Training")
        self.assertTrue(record.valid)
        self.assertTrue(set(INVENTORY_FIELDS).issubset(record.to_dict()))
        self.assertEqual(record.split, "unassigned")
        self.assertFalse(record.pilot_selected)
        self.assertEqual(record.pilot_selection_reason, "")
        self.assertIsNone(record.pilot_seed)
        self.assertEqual(record.roi_status, "pending")
        self.assertFalse(Path(record.relative_path).is_absolute())
        self.assertTrue(all(not Path(path).is_absolute() for path in record.frame_relative_paths))

    def test_class_mapping(self) -> None:
        self.assertEqual(map_ai_hub_class(3, "Take_pills"), ("복약", "direct_target", False))
        self.assertEqual(map_ai_hub_class(2, "Drink_bever"), ("음수", "auxiliary_positive", False))
        self.assertEqual(map_ai_hub_class(5, "Drink_alcohol"), ("음수", "auxiliary_positive", False))
        self.assertEqual(map_ai_hub_class(1, "Eat_food"), ("기타", "other", True))
        self.assertEqual(map_ai_hub_class(9, "Wash_hand"), ("기타", "other", False))

    def test_invalid_json_is_retained(self) -> None:
        json_path = _write_candidate(self.root)
        json_path.write_text("{broken", encoding="utf-8")
        record = build_candidate_record(json_path, self.root, "Training")
        self.assertFalse(record.valid)
        self.assertIn("json_parse_error", record.exclusion_reason)
        self.assertEqual(len(record.frame_relative_paths), 3)

    def test_missing_jpg_and_non_three_frame_structure_are_invalid(self) -> None:
        json_path = _write_candidate(self.root)
        document = json.loads(json_path.read_text(encoding="utf-8"))
        document["images"].pop()
        (json_path.parent / document["images"][0]["filename"]).unlink()
        json_path.write_text(json.dumps(document), encoding="utf-8")
        record = build_candidate_record(json_path, self.root, "Training")
        self.assertFalse(record.valid)
        self.assertIn("unexpected_frame_count", record.exclusion_reason)
        self.assertIn("missing_jpg", record.exclusion_reason)

    def test_path_json_mismatch_is_invalid(self) -> None:
        json_path = _write_candidate(self.root)
        document = json.loads(json_path.read_text(encoding="utf-8"))
        document["video.meta"]["viewpoint"] = 1
        json_path.write_text(json.dumps(document), encoding="utf-8")
        record = build_candidate_record(json_path, self.root, "Training")
        self.assertFalse(record.valid)
        self.assertIn("path_json_viewpoint_mismatch", record.exclusion_reason)

    def test_metadata_xlsx_intersection_join_and_output(self) -> None:
        _write_candidate(self.root)
        metadata_path = self.root / "metadata.xlsx"
        local_row = [
            "C003_A1_M12345_V3.mp4", 12345, 3, 7, 3, "male", 170, 1, 1, "job",
            "Seoul", "in", "alone", "unuse", "takes pills", "2021-01-01", "10:00:00", 100, "Gopro",
        ]
        metadata_only_row = [
            "C003_A1_M99999_V3.mp4", 99999, 3, 8, 3, "male", 170, 1, 1, "job",
            "Seoul", "in", "alone", "unuse", "not local", "2021-01-01", "10:00:00", 100, "Gopro",
        ]
        _write_minimal_metadata_xlsx(metadata_path, [local_row, metadata_only_row])
        result = build_inventory(self.root, metadata_path=metadata_path)
        self.assertEqual(len(result.records), 1)
        self.assertTrue(result.records[0].metadata_matched)
        self.assertEqual(result.summary["metadata_rows"], 2)
        self.assertEqual(result.summary["metadata_matches"], 1)
        self.assertEqual(result.summary["metadata_only_rows"], 1)
        output_dir = self.root / "output"
        inventory_path, summary_path = write_inventory(result, output_dir)
        self.assertTrue(inventory_path.is_file())
        self.assertTrue(summary_path.is_file())

    def test_unmatched_metadata_marks_local_candidate_invalid(self) -> None:
        _write_candidate(self.root)
        metadata_path = self.root / "metadata.xlsx"
        _write_minimal_metadata_xlsx(metadata_path, [])
        result = build_inventory(self.root, metadata_path=metadata_path, max_candidates=1)
        self.assertFalse(result.records[0].valid)
        self.assertIn("metadata_not_matched", result.records[0].exclusion_reason)

    def test_duplicate_video_id_candidates_are_retained_and_invalid(self) -> None:
        first = _write_candidate(self.root, category=3, class_name="Take_pills")
        second_dir = _candidate_dir(self.root, category=2)
        second = second_dir / first.name.replace("C003", "C002")
        document = json.loads(first.read_text(encoding="utf-8"))
        document["categories"] = {"id": 2, "name": "Drink_bever"}
        document["video.meta"]["category_id"] = 2
        for image in document["images"]:
            image["filename"] = image["filename"].replace("C003", "C002")
            (second_dir / image["filename"]).write_bytes(b"jpg")
        second.write_text(json.dumps(document), encoding="utf-8")
        result = build_inventory(self.root, max_candidates=None, use_metadata=False)
        self.assertEqual(len(result.records), 2)
        self.assertTrue(all(not record.valid for record in result.records))
        self.assertEqual(result.summary["duplicate_candidates"], 2)

    def test_missing_source_root_fails_explicitly(self) -> None:
        (self.root / "2.Validation" / "라벨링데이터").rmdir()
        with self.assertRaises(InventoryBuildError):
            build_inventory(self.root, use_metadata=False)

    def test_status_pass_when_counts_and_candidates_are_valid(self) -> None:
        records = [InventoryRecord(relative_path="valid.json")]
        status = classify_inventory_status(records, {}, 0, 0)
        self.assertEqual(status["status"], "PASS")
        self.assertEqual(status["build_status"], "PASS")
        self.assertEqual(status["data_quality_status"], "PASS")
        self.assertEqual(status["usable_candidates"], 1)

    def test_status_pass_with_warnings_for_isolated_raw_quality_issue(self) -> None:
        records = [
            InventoryRecord(relative_path="valid.json"),
            InventoryRecord(
                relative_path="raw-issue.json",
                valid=False,
                exclusion_reason="duplicate_frame_reference",
                pilot_selected=False,
            ),
        ]
        status = classify_inventory_status(records, {}, 0, 0)
        self.assertEqual(status["status"], "PASS_WITH_WARNINGS")
        self.assertEqual(status["build_status"], "PASS")
        self.assertEqual(status["data_quality_status"], "WARNING")
        self.assertEqual(status["invalid_candidates"], 1)
        self.assertEqual(status["usable_candidates"], 1)

    def test_status_fail_for_count_mismatch(self) -> None:
        records = [InventoryRecord(relative_path="valid.json")]
        status = classify_inventory_status(records, {"json_candidates": {}}, 0, 0)
        self.assertEqual(status["status"], "FAIL")

    def test_status_fail_for_metadata_unmatched(self) -> None:
        records = [InventoryRecord(relative_path="valid.json")]
        status = classify_inventory_status(records, {}, 1, 0)
        self.assertEqual(status["status"], "FAIL")

    def test_status_fail_for_duplicate_candidate(self) -> None:
        records = [
            InventoryRecord(
                relative_path="duplicate.json",
                valid=False,
                exclusion_reason="duplicate_candidate_video_id: 12345",
            )
        ]
        status = classify_inventory_status(records, {}, 0, 1)
        self.assertEqual(status["status"], "FAIL")

    def test_phase_3_validation_rejects_selected_invalid_candidate(self) -> None:
        records = [
            InventoryRecord(
                relative_path="invalid.json",
                valid=False,
                exclusion_reason="duplicate_frame_reference",
                pilot_selected=True,
            )
        ]
        with self.assertRaises(InventoryBuildError):
            validate_pilot_selection(records)

    def test_cli_exit_codes(self) -> None:
        self.assertEqual(exit_code_for_status("PASS"), 0)
        self.assertEqual(exit_code_for_status("PASS_WITH_WARNINGS"), 0)
        self.assertEqual(exit_code_for_status("FAIL"), 1)
        self.assertEqual(exit_code_for_status("UNKNOWN"), 1)


if __name__ == "__main__":
    unittest.main()
