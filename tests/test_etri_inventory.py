from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_etri_inventory import exit_code_for_status
from src.etri_inventory import (
    HARD_NEGATIVE_ACTIONS,
    INVENTORY_FIELDS,
    MULTI_PERSON_ACTIONS,
    EtriInventoryError,
    EtriInventoryRecord,
    build_candidate_record,
    build_inventory,
    classify_etri_status,
    map_etri_class,
    validate_pilot_selection,
)


class EtriInventoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.rgb_root = self.root / "RGB Videos"
        for number in range(201, 231):
            (self.rgb_root / f"P{number}").mkdir(parents=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_clip(self, filename: str, directory: str = "P201", size: int = 400_000) -> Path:
        path = self.rgb_root / directory / filename
        path.write_bytes(b"0" * size)
        return path

    def test_valid_structure_schema_and_initial_state(self) -> None:
        path = self._write_clip("A003_P201_G001_H070.mp4")
        record = build_candidate_record(path, self.root, "P201", validate_rgb_metadata=False)
        self.assertTrue(record.valid)
        self.assertTrue(set(INVENTORY_FIELDS).issubset(record.to_dict()))
        self.assertEqual(record.target_class, "복약")
        self.assertEqual(record.fold, "unassigned")
        self.assertFalse(record.pilot_selected)
        self.assertEqual(record.roi_status, "pending")
        self.assertFalse(Path(record.relative_path).is_absolute())

    def test_class_hard_negative_and_multi_person_mapping(self) -> None:
        self.assertEqual(map_etri_class("A003"), "복약")
        self.assertEqual(map_etri_class("A004"), "음수")
        self.assertEqual(map_etri_class("A040"), "기타")
        self.assertEqual(
            HARD_NEGATIVE_ACTIONS,
            frozenset({"A001", "A010", "A012", "A013", "A014", "A015", "A016", "A017", "A021", "A035", "A038", "A040", "A043"}),
        )
        self.assertEqual(MULTI_PERSON_ACTIONS, frozenset({"A045", "A046", "A047", "A048"}))

        hard_path = self._write_clip("A040_P201_G001_H070.mp4")
        hard = build_candidate_record(hard_path, self.root, "P201", validate_rgb_metadata=False)
        self.assertTrue(hard.is_hard_negative)

        multi_path = self._write_clip("A045_P201_G001_H070.mp4")
        multi = build_candidate_record(multi_path, self.root, "P201", validate_rgb_metadata=False)
        self.assertTrue(multi.valid)
        self.assertTrue(multi.is_multi_person)
        self.assertFalse(multi.pilot_selected)
        self.assertEqual(multi.pilot_selection_reason, "excluded_multi_person_action")

    def test_filename_participant_take_and_height_validation(self) -> None:
        invalid_name = self._write_clip("bad.mp4")
        record = build_candidate_record(invalid_name, self.root, "P201", validate_rgb_metadata=False)
        self.assertFalse(record.valid)
        self.assertIn("invalid_filename", record.exclusion_reason)

        mismatch = self._write_clip("A003_P202_G001_H070.mp4", directory="P201")
        record = build_candidate_record(mismatch, self.root, "P201", validate_rgb_metadata=False)
        self.assertIn("participant_token_mismatch", record.exclusion_reason)

        bad_take = self._write_clip("A003_P207_G003_H070.mp4", directory="P207")
        record = build_candidate_record(bad_take, self.root, "P207", validate_rgb_metadata=False)
        self.assertIn("invalid_take", record.exclusion_reason)

    def test_small_rgb_is_isolated_warning(self) -> None:
        path = self._write_clip("A053_P205_G011_H120.mp4", directory="P205", size=232_133)
        record = build_candidate_record(path, self.root, "P205", validate_rgb_metadata=False)
        self.assertFalse(record.valid)
        self.assertIn("rgb_file_too_small", record.exclusion_reason)
        status = classify_etri_status([record], {}, 0)
        self.assertEqual(status["status"], "PASS_WITH_WARNINGS")
        self.assertEqual(status["build_status"], "PASS")
        self.assertEqual(status["data_quality_status"], "WARNING")

    def test_partial_build_does_not_apply_full_count_expectations(self) -> None:
        self._write_clip("A003_P201_G001_H070.mp4")
        result = build_inventory(self.root, max_candidates=1, validate_rgb_metadata=False)
        self.assertFalse(result.summary["full_run"])
        self.assertEqual(result.summary["count_mismatches"], {})
        self.assertEqual(result.summary["status"], "PASS")

    def test_systemic_status_failures(self) -> None:
        valid = EtriInventoryRecord(relative_path="valid.mp4")
        self.assertEqual(classify_etri_status([valid], {"rgb_clips": {}}, 0)["status"], "FAIL")
        duplicate = EtriInventoryRecord(
            relative_path="duplicate.mp4",
            valid=False,
            exclusion_reason="duplicate_clip: P201/A003/G001/H070",
        )
        self.assertEqual(classify_etri_status([duplicate], {}, 2)["status"], "FAIL")
        unisolated = EtriInventoryRecord(relative_path="bad.mp4", valid=False)
        self.assertEqual(classify_etri_status([unisolated], {}, 0)["status"], "FAIL")

    def test_pilot_selection_rejects_invalid_and_multi_person(self) -> None:
        invalid = EtriInventoryRecord(
            relative_path="invalid.mp4",
            valid=False,
            exclusion_reason="rgb_file_too_small",
            pilot_selected=True,
        )
        with self.assertRaises(EtriInventoryError):
            validate_pilot_selection([invalid])
        multi = EtriInventoryRecord(
            relative_path="multi.mp4",
            valid=True,
            is_multi_person=True,
            pilot_selected=True,
        )
        with self.assertRaises(EtriInventoryError):
            validate_pilot_selection([multi])

    def test_cli_exit_codes(self) -> None:
        self.assertEqual(exit_code_for_status("PASS"), 0)
        self.assertEqual(exit_code_for_status("PASS_WITH_WARNINGS"), 0)
        self.assertEqual(exit_code_for_status("FAIL"), 1)
        self.assertEqual(exit_code_for_status("UNKNOWN"), 1)


if __name__ == "__main__":
    unittest.main()
