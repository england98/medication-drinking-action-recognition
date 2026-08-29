"""Deterministic Phase 3 Fixed Pilot Manifest construction and validation."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


AI_HUB_REQUIRED_FIELDS = {
    "dataset", "root_key", "relative_path", "frame_relative_paths", "video_id",
    "source_split", "actor", "viewpoint", "original_class", "target_class",
    "target_role", "split", "is_hard_negative", "valid", "exclusion_reason",
    "pilot_selected", "pilot_selection_reason", "pilot_seed", "roi_status",
}
ETRI_REQUIRED_FIELDS = {
    "dataset", "root_key", "relative_path", "batch", "participant", "action",
    "target_class", "fold", "height", "take", "is_multi_person",
    "is_hard_negative", "valid", "exclusion_reason", "pilot_selected",
    "pilot_selection_reason", "pilot_seed", "roi_status",
}
MULTI_PERSON_ACTIONS = {"A045", "A046", "A047", "A048"}


class PilotManifestError(RuntimeError):
    """Raised when Phase 3 policy or consistency validation fails."""


@dataclass(frozen=True)
class PilotManifestConfig:
    split_seed: int
    pilot_seed: int
    ai_hub_train_ratio: float
    ai_hub_class_targets: Mapping[str, int]
    etri_n_folds: int
    etri_target_cap: int
    etri_other_cap: int
    etri_hard_negative_cap: int
    etri_general_other_cap: int


def load_pilot_config(path: Path) -> PilotManifestConfig:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        root = document["phase3"]
        ai = root["ai_hub"]
        etri = root["etri"]
        config = PilotManifestConfig(
            split_seed=int(root["split_seed"]), pilot_seed=int(root["pilot_seed"]),
            ai_hub_train_ratio=float(ai["train_ratio"]),
            ai_hub_class_targets={str(k): int(v) for k, v in ai["class_target_counts"].items()},
            etri_n_folds=int(etri["n_folds"]),
            etri_target_cap=int(etri["target_cap_per_participant"]),
            etri_other_cap=int(etri["other_cap_per_participant"]),
            etri_hard_negative_cap=int(etri["hard_negative_per_participant"]),
            etri_general_other_cap=int(etri["general_other_per_participant"]),
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise PilotManifestError(f"Invalid Phase 3 config {path}: {exc}") from exc
    if not 0 < config.ai_hub_train_ratio < 1 or config.etri_n_folds != 5:
        raise PilotManifestError("AI-Hub train_ratio must be in (0, 1) and ETRI n_folds must be 5")
    if any(value < 0 for value in (*config.ai_hub_class_targets.values(), config.etri_target_cap,
                                  config.etri_other_cap, config.etri_hard_negative_cap,
                                  config.etri_general_other_cap)):
        raise PilotManifestError("Pilot caps and targets must be non-negative")
    if config.etri_hard_negative_cap + config.etri_general_other_cap > config.etri_other_cap:
        raise PilotManifestError("ETRI hard/general other caps exceed total other cap")
    return config


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("row is not an object")
                    records.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise PilotManifestError(f"Cannot read inventory {path}: {exc}") from exc
    if not records:
        raise PilotManifestError(f"Inventory is empty: {path}")
    return records


def _stable_shuffle(values: Iterable[Any], seed: int, namespace: str) -> list[Any]:
    result = sorted(values, key=lambda item: str(item))
    digest = hashlib.sha256(f"{seed}:{namespace}".encode()).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(result)
    return result


def _reset(rows: Sequence[dict[str, Any]], seed: int) -> None:
    for row in rows:
        row["pilot_selected"] = False
        row["pilot_seed"] = seed
        row["roi_status"] = "pending"
        if not row.get("valid"):
            row["pilot_selection_reason"] = "invalid"
        else:
            row["pilot_selection_reason"] = "not_selected_quota"


def _round_robin(candidates: Sequence[dict[str, Any]], group_key: str, count: int,
                 seed: int, namespace: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[str(row[group_key])].append(row)
    group_order = _stable_shuffle(groups, seed, namespace + ":groups")
    for group in groups:
        groups[group] = _stable_shuffle(groups[group], seed, namespace + ":" + group)
    chosen: list[dict[str, Any]] = []
    while len(chosen) < count and group_order:
        next_order = []
        for group in group_order:
            if groups[group] and len(chosen) < count:
                chosen.append(groups[group].pop())
            if groups[group]:
                next_order.append(group)
        group_order = next_order
    return chosen


def build_ai_hub_manifest(inventory: Sequence[Mapping[str, Any]], config: PilotManifestConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [dict(row) for row in inventory]
    _reset(rows, config.pilot_seed)
    actors = _stable_shuffle({row.get("actor") for row in rows if row.get("actor") is not None},
                             config.split_seed, "ai_hub_actors")
    val_count = max(1, round(len(actors) * (1 - config.ai_hub_train_ratio)))
    val_actors = set(actors[:val_count])
    for row in rows:
        row["split"] = "val" if row.get("actor") in val_actors else "train"

    eligible = [row for row in rows if row.get("valid") is True and row.get("viewpoint") == 3]
    split_sizes = Counter(row["split"] for row in eligible)
    for target_class, total_target in config.ai_hub_class_targets.items():
        class_rows = [row for row in eligible if row.get("target_class") == target_class]
        # Preserve both splits and approximate their eligible-data proportion.
        val_target = round(total_target * split_sizes["val"] / max(1, sum(split_sizes.values())))
        targets = {"val": val_target, "train": total_target - val_target}
        for split, target in targets.items():
            pool = [row for row in class_rows if row["split"] == split]
            if target_class == "기타":
                hard = [row for row in pool if row.get("is_hard_negative")]
                selected = _round_robin(hard, "actor", min(target, max(1, min(len(hard), target // 4))),
                                        config.pilot_seed, f"ai:{split}:hard")
                remaining = [row for row in pool if row not in selected]
                selected += _round_robin(remaining, "actor", target - len(selected), config.pilot_seed,
                                         f"ai:{split}:other")
            else:
                selected = _round_robin(pool, "actor", target, config.pilot_seed,
                                        f"ai:{split}:{target_class}")
            for row in selected:
                row["pilot_selected"] = True
                row["pilot_selection_reason"] = "hard_negative" if row.get("is_hard_negative") else "actor_diversity_class_quota"
    validate_ai_hub_manifest(rows)
    return rows, summarize_ai_hub(rows)


def _choose_target(pool: Sequence[dict[str, Any]], cap: int, seed: int, namespace: str) -> list[dict[str, Any]]:
    ordered = _stable_shuffle(pool, seed, namespace)
    chosen: list[dict[str, Any]] = []
    for height in ("H070", "H120"):
        match = next((row for row in ordered if row.get("height") == height and row not in chosen), None)
        if match is not None and len(chosen) < cap:
            chosen.append(match)
    for row in ordered:
        if len(chosen) >= cap:
            break
        if row not in chosen:
            chosen.append(row)
    return chosen


def build_etri_manifest(inventory: Sequence[Mapping[str, Any]], config: PilotManifestConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [dict(row) for row in inventory]
    _reset(rows, config.pilot_seed)
    participants = _stable_shuffle({row.get("participant") for row in rows if row.get("participant")},
                                   config.split_seed, "etri_participants")
    participant_fold = {participant: index % config.etri_n_folds for index, participant in enumerate(participants)}
    for row in rows:
        row["fold"] = participant_fold.get(row.get("participant"), "unassigned")
        if row.get("is_multi_person") or row.get("action") in MULTI_PERSON_ACTIONS:
            row["pilot_selection_reason"] = "excluded_multi_person"

    by_participant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_participant[str(row.get("participant"))].append(row)
    for participant, participant_rows in sorted(by_participant.items()):
        eligible = [row for row in participant_rows if row.get("valid") is True and row.get("batch") == "B"
                    and not row.get("is_multi_person") and row.get("action") not in MULTI_PERSON_ACTIONS]
        for action in ("A003", "A004"):
            selected = _choose_target([row for row in eligible if row.get("action") == action],
                                      config.etri_target_cap, config.pilot_seed, f"etri:{participant}:{action}")
            for row in selected:
                row["pilot_selected"] = True
                row["pilot_selection_reason"] = "target_height_diversity" if len({x.get('height') for x in selected}) > 1 else "target_participant_cap"
        other = [row for row in eligible if row.get("target_class") == "기타"]
        hard = _round_robin([row for row in other if row.get("is_hard_negative")], "action",
                            config.etri_hard_negative_cap, config.pilot_seed, f"etri:{participant}:hard")
        general = _round_robin([row for row in other if not row.get("is_hard_negative")], "action",
                               config.etri_general_other_cap, config.pilot_seed, f"etri:{participant}:general")
        selected_other = hard + general
        if len(selected_other) < config.etri_other_cap:
            remainder = [row for row in other if row not in selected_other]
            selected_other += _round_robin(remainder, "action", config.etri_other_cap - len(selected_other),
                                           config.pilot_seed, f"etri:{participant}:other_fill")
        for row in selected_other:
            row["pilot_selected"] = True
            row["pilot_selection_reason"] = "hard_negative" if row.get("is_hard_negative") else "general_other"
    validate_etri_manifest(rows, config)
    return rows, summarize_etri(rows)


def _require_fields(rows: Sequence[Mapping[str, Any]], fields: set[str], dataset: str) -> None:
    if not rows:
        raise PilotManifestError(f"{dataset} manifest is empty")
    for index, row in enumerate(rows):
        missing = fields - set(row)
        if missing:
            raise PilotManifestError(f"{dataset} row {index} missing required fields: {sorted(missing)}")


def validate_ai_hub_manifest(rows: Sequence[Mapping[str, Any]]) -> None:
    _require_fields(rows, AI_HUB_REQUIRED_FIELDS, "AI-Hub")
    actors_by_split: dict[str, set[Any]] = defaultdict(set)
    videos: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        actors_by_split[str(row["split"])].add(row.get("actor"))
        videos[row.get("video_id")].append(row)
        if row.get("roi_status") != "pending": raise PilotManifestError("AI-Hub roi_status must be pending")
        if row.get("pilot_selected") and (not row.get("valid") or row.get("viewpoint") != 3): raise PilotManifestError("AI-Hub selected invalid/non-viewpoint_3 candidate")
        if row.get("pilot_selected") and len(row.get("frame_relative_paths", [])) != 3: raise PilotManifestError("AI-Hub selected video does not have three frames")
        if row.get("pilot_selected") and row.get("original_class") in {"Drink_bever", "Drink_alcohol"} and row.get("target_role") != "auxiliary_positive": raise PilotManifestError("AI-Hub auxiliary role mismatch")
        if row.get("pilot_selected") and row.get("original_class") == "Take_pills" and row.get("target_role") != "direct_target": raise PilotManifestError("AI-Hub direct target role mismatch")
    if actors_by_split["train"] & actors_by_split["val"]: raise PilotManifestError("AI-Hub actor leakage")
    selected_auxiliary = {row.get("original_class") for row in rows if row.get("pilot_selected") and row.get("target_class") == "음수"}
    if not {"Drink_bever", "Drink_alcohol"}.issubset(selected_auxiliary):
        raise PilotManifestError("AI-Hub Pilot must include both Drink_bever and Drink_alcohol")
    if not any(row.get("pilot_selected") and row.get("original_class") == "Eat_food" and row.get("is_hard_negative") for row in rows): raise PilotManifestError("AI-Hub selected Pilot lacks Eat_food hard negative")
    for same_video in videos.values():
        if len({row["split"] for row in same_video}) != 1 or len({row["pilot_selected"] for row in same_video}) != 1: raise PilotManifestError("AI-Hub video split/selection inconsistency")


def validate_etri_manifest(rows: Sequence[Mapping[str, Any]], config: PilotManifestConfig) -> None:
    _require_fields(rows, ETRI_REQUIRED_FIELDS, "ETRI")
    participant_folds: dict[Any, set[Any]] = defaultdict(set)
    counts: Counter[tuple[Any, Any]] = Counter()
    for row in rows:
        participant_folds[row.get("participant")].add(row.get("fold"))
        if row.get("fold") not in range(config.etri_n_folds): raise PilotManifestError("ETRI fold outside configured range")
        if row.get("roi_status") != "pending": raise PilotManifestError("ETRI roi_status must be pending")
        if row.get("pilot_selected") and (not row.get("valid") or row.get("batch") != "B" or row.get("is_multi_person") or row.get("action") in MULTI_PERSON_ACTIONS): raise PilotManifestError("ETRI selected excluded candidate")
        if row.get("pilot_selected"): counts[(row.get("participant"), row.get("action"))] += 1
    if any(len(folds) != 1 for folds in participant_folds.values()): raise PilotManifestError("ETRI participant leakage")
    for (participant, action), count in counts.items():
        if action in {"A003", "A004"} and count > config.etri_target_cap: raise PilotManifestError(f"ETRI target cap exceeded: {participant}/{action}")
    selected_other = Counter(row.get("participant") for row in rows if row.get("pilot_selected") and row.get("target_class") == "기타")
    if any(count > config.etri_other_cap for count in selected_other.values()):
        raise PilotManifestError("ETRI participant other cap exceeded")


def summarize_ai_hub(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row.get("pilot_selected")]
    summary = {
        "status": "PASS", "candidate_rows": len(rows),
        "valid_viewpoint_3_candidates": sum(row.get("valid") is True and row.get("viewpoint") == 3 for row in rows),
        "actors": len({row.get("actor") for row in rows}),
        "actors_by_split": {split: len({row.get("actor") for row in rows if row["split"] == split}) for split in ("train", "val")},
        "selected_class_counts": dict(sorted(Counter(row["target_class"] for row in selected).items())),
        "selected_original_class_counts": dict(sorted(Counter(row["original_class"] for row in selected).items())),
        "selected_by_split_and_class": {f"{s}|{c}": n for (s, c), n in sorted(Counter((row["split"], row["target_class"]) for row in selected).items())},
        "selected_videos": len(selected), "selected_frames": sum(len(row["frame_relative_paths"]) for row in selected),
        "selected_hard_negatives": sum(bool(row.get("is_hard_negative")) for row in selected),
    }
    return summary


def summarize_etri(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row.get("pilot_selected")]
    participants = sorted({row.get("participant") for row in rows})
    return {
        "status": "PASS", "candidate_rows": len(rows), "valid_candidates": sum(row.get("valid") is True for row in rows),
        "participants": len(participants),
        "participants_by_fold": {str(fold): len({row.get("participant") for row in rows if row["fold"] == fold}) for fold in range(5)},
        "selected_class_counts": dict(sorted(Counter(row["target_class"] for row in selected).items())),
        "selected_by_fold_and_class": {f"{f}|{c}": n for (f, c), n in sorted(Counter((row["fold"], row["target_class"]) for row in selected).items())},
        "selected_clips": len(selected), "selected_hard_negatives": sum(bool(row.get("is_hard_negative")) for row in selected),
        "selected_general_other": sum(row.get("target_class") == "기타" and not row.get("is_hard_negative") for row in selected),
        "selected_by_participant": {participant: dict(sorted(Counter(row["target_class"] for row in selected if row.get("participant") == participant).items())) for participant in participants},
    }


def write_manifest(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], output_dir: Path,
                   stem: str) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / f"{stem}.jsonl"
    selected_csv = output_dir / f"{stem.replace('_manifest', '_selected')}.csv"
    summary_path = output_dir / f"{stem}_summary.json"
    with manifest.with_suffix(".jsonl.tmp").open("w", encoding="utf-8") as target:
        for row in rows: target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest.with_suffix(".jsonl.tmp").replace(manifest)
    selected = [row for row in rows if row.get("pilot_selected")]
    with selected_csv.with_suffix(".csv.tmp").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(selected)
    selected_csv.with_suffix(".csv.tmp").replace(selected_csv)
    sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    final_summary = {**summary, "manifest_file": manifest.name, "manifest_sha256": sha256}
    summary_path.write_text(json.dumps(final_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest, selected_csv, summary_path
