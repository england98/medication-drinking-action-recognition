"""Validation, metrics, and artifact helpers for Phase 10 raw-video OOF evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

from src.etri_embedding import source_clip_key
from src.stage_a import CLASS_NAMES, CLASS_TO_INDEX


class RawVideoOofError(RuntimeError):
    """Raised when an OOF evaluation contract is violated."""


PREDICTION_FIELDS = (
    "dataset", "clip_key", "participant", "fold", "root_key", "relative_path", "action",
    "ground_truth_index", "ground_truth_label", "prediction_index", "prediction_label",
    "prob_medication", "prob_drinking", "prob_other", "confidence", "correct",
    "roi_success", "roi_partial", "roi_fallback", "n_sampled_frames", "encoder_checkpoint",
    "encoder_checkpoint_sha256", "stage_b_checkpoint", "stage_b_checkpoint_sha256", "T", "D",
    "roi_config_sha256", "preprocessing_config_sha256", "manifest_sha256", "git_commit",
    "evaluation_timestamp", "inference_success", "error",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_oof_universe(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 239:
        raise RawVideoOofError(f"Expected 239 selected-valid samples, got {len(rows)}")
    keys = [source_clip_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise RawVideoOofError("Fixed Pilot Manifest has duplicate selected clip keys")
    if any(row.get("pilot_selected") is not True or row.get("valid") is not True for row in rows):
        raise RawVideoOofError("OOF universe includes unselected or invalid rows")
    if {int(row["fold"]) for row in rows} != set(range(5)):
        raise RawVideoOofError("OOF universe must contain folds 0..4")
    classes = Counter(str(row["target_class"]) for row in rows)
    expected = {"복약": 59, "음수": 60, "기타": 120}
    if dict(classes) != expected:
        raise RawVideoOofError(f"OOF class counts differ from frozen expectation: {dict(classes)}")
    leakage: dict[str, Any] = {}
    for fold in range(5):
        evaluation = {str(row["participant"]) for row in rows if int(row["fold"]) == fold}
        training = {str(row["participant"]) for row in rows if int(row["fold"]) != fold}
        overlap = sorted(evaluation & training)
        if overlap:
            raise RawVideoOofError(f"Participant leakage for fold {fold}: {overlap}")
        leakage[str(fold)] = {"evaluation_participants": sorted(evaluation),
                              "training_participant_count": len(training), "overlap": overlap}
    return {"total_samples": len(rows), "participant_count": len({row["participant"] for row in rows}),
            "class_counts": expected, "leakage": leakage}


def validate_predictions(predictions: Sequence[Mapping[str, Any]],
                         rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    expected = {source_clip_key(row): row for row in rows}
    keys = [str(record.get("clip_key")) for record in predictions]
    duplicates = len(keys) - len(set(keys)); missing = set(expected) - set(keys)
    extras = set(keys) - set(expected)
    failures = [record for record in predictions if record.get("inference_success") is not True]
    if duplicates or missing or extras or failures or len(predictions) != len(rows):
        raise RawVideoOofError(f"OOF completeness failed: predictions={len(predictions)}, "
            f"duplicates={duplicates}, missing={len(missing)}, extras={len(extras)}, failures={len(failures)}")
    for record in predictions:
        row = expected[str(record["clip_key"])]
        if (int(record["fold"]) != int(row["fold"])
                or int(record["ground_truth_index"]) != CLASS_TO_INDEX[str(row["target_class"])]):
            raise RawVideoOofError(f"Wrong fold checkpoint or target mapping: {record['clip_key']}")
        probability_sum = sum(float(record[name]) for name in
                              ("prob_medication", "prob_drinking", "prob_other"))
        if not np.isclose(probability_sum, 1.0, atol=1e-6):
            raise RawVideoOofError(f"Probability sum mismatch: {record['clip_key']}")
    return {"duplicates": 0, "missing": 0, "failed": 0}


def compute_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    truth = np.asarray([int(record["ground_truth_index"]) for record in records], dtype=np.int64)
    prediction = np.asarray([int(record["prediction_index"]) for record in records], dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=range(3), zero_division=0)
    return {"sample_count": len(records), "participant_count": len({r["participant"] for r in records}),
        "accuracy": float(accuracy_score(truth, prediction)), "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()),
        "per_class": {name: {"precision": float(precision[index]), "recall": float(recall[index]),
            "f1": float(f1[index]), "support": int(support[index])} for index, name in enumerate(CLASS_NAMES)},
        "confusion_matrix": confusion_matrix(truth, prediction, labels=range(3)).tolist(),
        "label_order": list(CLASS_NAMES)}


def aggregate_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    overall = compute_metrics(records); folds = {}
    for fold in range(5):
        selected = [record for record in records if int(record["fold"]) == fold]
        folds[str(fold)] = {"fold": fold, **compute_metrics(selected)}
    macro = np.asarray([folds[str(f)]["macro_f1"] for f in range(5)])
    recall = {name: np.asarray([folds[str(f)]["per_class"][name]["recall"] for f in range(5)])
              for name in CLASS_NAMES}
    return {"overall": overall, "fold_metrics": folds,
        "fold_macro_f1_mean": float(macro.mean()), "fold_macro_f1_std": float(macro.std(ddof=0)),
        "fold_class_recall": {name: {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
                              for name, values in recall.items()}, "std_definition": "population_ddof_0"}


def roi_statistics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        counts = {name: sum(int(row[f"roi_{name}"]) for row in selected)
                  for name in ("success", "partial", "fallback")}
        total = sum(counts.values())
        return {"total_frames": total, **counts,
                "ratios": {name: counts[name] / total if total else 0.0 for name in counts}}
    return {"overall": summarize(records),
        "by_class": {name: summarize([r for r in records if r["ground_truth_label"] == name])
                     for name in CLASS_NAMES},
        "by_fold": {str(fold): summarize([r for r in records if int(r["fold"]) == fold])
                    for fold in range(5)}}


def compare_cached_raw(raw: Sequence[Mapping[str, Any]], cached_csv: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with cached_csv.open(encoding="utf-8", newline="") as handle:
        cached = list(csv.DictReader(handle))
    raw_by_key = {str(row["clip_key"]): row for row in raw}; cached_by_key = {row["clip_key"]: row for row in cached}
    sample_match = set(raw_by_key) == set(cached_by_key) and len(cached) == len(cached_by_key)
    if not sample_match:
        raise RawVideoOofError("Raw/cached OOF sample universes do not match exactly")
    compared = []
    for key in sorted(raw_by_key):
        r, c = raw_by_key[key], cached_by_key[key]
        cached_probs = [float(c[name]) for name in ("prob_medication", "prob_drinking", "prob_other")]
        raw_probs = [float(r[name]) for name in ("prob_medication", "prob_drinking", "prob_other")]
        differences = [raw_probs[i] - cached_probs[i] for i in range(3)]
        compared.append({"clip_key": key, "fold": int(r["fold"]),
            "cached_prediction_index": int(c["pred_label"]), "raw_prediction_index": int(r["prediction_index"]),
            "prediction_match": int(c["pred_label"]) == int(r["prediction_index"]),
            "cached_prob_medication": cached_probs[0], "cached_prob_drinking": cached_probs[1],
            "cached_prob_other": cached_probs[2], "raw_prob_medication": raw_probs[0],
            "raw_prob_drinking": raw_probs[1], "raw_prob_other": raw_probs[2],
            "diff_prob_medication": differences[0], "diff_prob_drinking": differences[1],
            "diff_prob_other": differences[2]})
    cached_records = [{"ground_truth_index": int(row["true_label"]), "prediction_index": int(row["pred_label"]),
        "participant": row["participant"], "fold": int(row["fold"])} for row in cached]
    raw_metrics, cached_metrics = aggregate_metrics(raw), aggregate_metrics(cached_records)
    flat_diffs = np.asarray([[row[f"diff_prob_{name}"] for name in ("medication", "drinking", "other")]
                             for row in compared])
    agreements = sum(bool(row["prediction_match"]) for row in compared)
    per_class_difference = {name: {
        metric: raw_metrics["overall"]["per_class"][name][metric]
                - cached_metrics["overall"]["per_class"][name][metric]
        for metric in ("precision", "recall", "f1")} for name in CLASS_NAMES}
    fold_difference = {str(fold): raw_metrics["fold_metrics"][str(fold)]["macro_f1"]
        - cached_metrics["fold_metrics"][str(fold)]["macro_f1"] for fold in range(5)}
    summary = {"sample_count_raw": len(raw), "sample_count_cached": len(cached),
        "sample_set_match": True, "prediction_agreement_count": agreements,
        "prediction_agreement_rate": agreements / len(compared),
        "raw_macro_f1": raw_metrics["overall"]["macro_f1"],
        "cached_macro_f1": cached_metrics["overall"]["macro_f1"],
        "macro_f1_difference": raw_metrics["overall"]["macro_f1"] - cached_metrics["overall"]["macro_f1"],
        "per_class_metric_difference": per_class_difference, "fold_macro_f1_difference": fold_difference,
        "confusion_matrix_equal": raw_metrics["overall"]["confusion_matrix"] == cached_metrics["overall"]["confusion_matrix"],
        "probability_max_absolute_difference": float(np.abs(flat_diffs).max()),
        "probability_mean_absolute_difference": float(np.abs(flat_diffs).mean()),
        "final_consistency_verdict": "EXACT_MATCH" if agreements == len(compared) and np.abs(flat_diffs).max() <= 1e-6
                                     else "DIFFERENCE_OBSERVED"}
    return summary, compared


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def write_confusion_csv(path: Path, matrix: Sequence[Sequence[int]]) -> None:
    rows = [{"ground_truth": name, **{CLASS_NAMES[index]: int(value) for index, value in enumerate(matrix[row])}}
            for row, name in enumerate(CLASS_NAMES)]
    write_csv(path, rows, ("ground_truth", *CLASS_NAMES))


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2); handle.write("\n")
