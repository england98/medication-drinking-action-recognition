from __future__ import annotations

import unittest

from src.pilot_manifest import (
    AI_HUB_REQUIRED_FIELDS, ETRI_REQUIRED_FIELDS, PilotManifestConfig,
    build_ai_hub_manifest, build_etri_manifest,
)


CONFIG = PilotManifestConfig(
    split_seed=42, pilot_seed=42, ai_hub_train_ratio=0.75,
    ai_hub_class_targets={"복약": 4, "음수": 4, "기타": 4}, etri_n_folds=5,
    etri_target_cap=2, etri_other_cap=4, etri_hard_negative_cap=2,
    etri_general_other_cap=2,
)


def ai_row(actor: int, number: int, cls: str, original: str, valid: bool = True) -> dict:
    role = "direct_target" if cls == "복약" else "auxiliary_positive" if cls == "음수" else "other"
    return {
        "dataset": "ai_hub", "root_key": "ai_hub_raw_root", "relative_path": f"{number}.json",
        "frame_relative_paths": [f"{number}_{i}.jpg" for i in range(3)], "video_id": str(number),
        "source_split": "Training", "actor": actor, "viewpoint": 3, "original_class": original,
        "target_class": cls, "target_role": role, "split": "unassigned",
        "is_hard_negative": original == "Eat_food", "valid": valid,
        "exclusion_reason": "broken" if not valid else "", "pilot_selected": False,
        "pilot_selection_reason": "", "pilot_seed": None, "roi_status": "pending",
    }


def etri_row(participant: str, action: str, take: int, height: str = "H070", valid: bool = True) -> dict:
    cls = "복약" if action == "A003" else "음수" if action == "A004" else "기타"
    return {
        "dataset": "etri", "root_key": "etri_raw_root", "relative_path": f"{participant}/{action}_{take}.mp4",
        "batch": "B", "participant": participant, "action": action, "target_class": cls,
        "fold": "unassigned", "height": height, "take": f"G{take:03d}",
        "is_multi_person": action in {"A045", "A046", "A047", "A048"},
        "is_hard_negative": action in {"A040", "A016"}, "valid": valid,
        "exclusion_reason": "broken" if not valid else "", "pilot_selected": False,
        "pilot_selection_reason": "", "pilot_seed": None, "roi_status": "pending",
    }


class PilotManifestTests(unittest.TestCase):
    def test_ai_hub_is_deterministic_disjoint_complete_and_retains_rows(self) -> None:
        rows = []
        number = 0
        for actor in range(1, 9):
            for cls, original in (("복약", "Take_pills"), ("음수", "Drink_bever"),
                                  ("음수", "Drink_alcohol"), ("기타", "Eat_food"),
                                  ("기타", "Wash_hand")):
                number += 1; rows.append(ai_row(actor, number, cls, original))
        rows.append(ai_row(1, 999, "복약", "Take_pills", valid=False))
        first, _ = build_ai_hub_manifest(rows, CONFIG)
        second, _ = build_ai_hub_manifest(rows, CONFIG)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(rows))
        train = {r["actor"] for r in first if r["split"] == "train"}
        val = {r["actor"] for r in first if r["split"] == "val"}
        self.assertFalse(train & val)
        self.assertFalse(next(r for r in first if r["video_id"] == "999")["pilot_selected"])
        self.assertTrue(AI_HUB_REQUIRED_FIELDS.issubset(first[0]))
        self.assertTrue(all(r["roi_status"] == "pending" for r in first))

    def test_etri_is_deterministic_disjoint_caps_and_exclusions(self) -> None:
        rows = []
        for p in range(201, 211):
            participant = f"P{p}"
            for action in ("A003", "A004"):
                for take, height in ((1, "H070"), (2, "H120"), (3, "H070")):
                    rows.append(etri_row(participant, action, take, height))
            for index, action in enumerate(("A040", "A016", "A021", "A022", "A023", "A024"), 1):
                rows.append(etri_row(participant, action, index))
            rows.append(etri_row(participant, "A045", 20))
        rows.append(etri_row("P201", "A003", 99, valid=False))
        first, _ = build_etri_manifest(rows, CONFIG)
        second, _ = build_etri_manifest(rows, CONFIG)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(rows))
        self.assertTrue(ETRI_REQUIRED_FIELDS.issubset(first[0]))
        for participant in {r["participant"] for r in first}:
            self.assertEqual(len({r["fold"] for r in first if r["participant"] == participant}), 1)
            for action in ("A003", "A004"):
                self.assertLessEqual(sum(r["pilot_selected"] for r in first if r["participant"] == participant and r["action"] == action), 2)
        self.assertFalse(any(r["pilot_selected"] for r in first if r["action"] == "A045" or not r["valid"]))
        self.assertTrue(all(r["roi_status"] == "pending" for r in first))


if __name__ == "__main__":
    unittest.main()
