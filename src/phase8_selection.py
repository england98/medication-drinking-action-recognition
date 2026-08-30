"""Phase 8 independent evidence validation and deterministic structure selection."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.etri_embedding import load_embedding_config, read_selected_etri_manifest, source_clip_key
from src.etri_stage_b import (EXPECTED_FOLDS, EXPERIMENTS, StageBError, aggregate_fold_results,
    load_stage_b_config, validate_experiment_fairness, validate_fold_contract)
from src.stage_a import CLASS_NAMES, CLASS_TO_INDEX


PHASE7_COMMIT = "60c72bc125b0cc2fe94bd9500858192e3edaf521"
PRIMARY_METRIC = "macro_f1_5fold_mean"
TOLERANCE = 1e-12


class Phase8SelectionError(RuntimeError):
    """Raised when Phase 7 evidence cannot support a Phase 8 decision."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try: value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise Phase8SelectionError(f"Cannot read JSON evidence: {path}") from exc
    if not isinstance(value, dict): raise Phase8SelectionError(f"JSON evidence must be an object: {path}")
    return value


def _read_oof(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8", newline="") as source: rows = list(csv.DictReader(source))
    except OSError as exc: raise Phase8SelectionError(f"Cannot read OOF evidence: {path}") from exc
    required = {"clip_key", "participant", "fold", "true_label", "pred_label",
                "prob_medication", "prob_drinking", "prob_other", "experiment"}
    if not rows or required - set(rows[0]): raise Phase8SelectionError(f"Malformed OOF evidence: {path}")
    return rows


def fold_assignment_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    values = sorted((str(row["participant"]), int(row["fold"])) for row in rows)
    unique = []
    for value in values:
        if value not in unique: unique.append(value)
    return hashlib.sha256(json.dumps(unique, separators=(",", ":")).encode()).hexdigest()


def _validate_oof_rows(rows: Sequence[Mapping[str, Any]], manifest_rows: Sequence[Mapping[str, Any]],
                       experiment: str, expected_fold: int | None = None) -> None:
    manifest = {source_clip_key(row): row for row in manifest_rows}; seen = set()
    for row in rows:
        key = str(row.get("clip_key")); source = manifest.get(key)
        if key in seen: raise Phase8SelectionError(f"Duplicate OOF prediction: {experiment}/{key}")
        if source is None: raise Phase8SelectionError(f"OOF contains excluded/unknown clip: {experiment}/{key}")
        seen.add(key)
        try:
            fold, truth, prediction = int(row["fold"]), int(row["true_label"]), int(row["pred_label"])
            probabilities = np.asarray([float(row["prob_medication"]), float(row["prob_drinking"]),
                                        float(row["prob_other"])])
        except (KeyError, TypeError, ValueError) as exc: raise Phase8SelectionError(f"Malformed OOF row: {experiment}/{key}") from exc
        if expected_fold is not None and fold != expected_fold: raise Phase8SelectionError(f"OOF fold mismatch: {experiment}/{key}")
        if (row.get("experiment") != experiment or row.get("participant") != source.get("participant")
                or fold != source.get("fold") or truth != CLASS_TO_INDEX.get(str(source.get("target_class")))):
            raise Phase8SelectionError(f"OOF manifest/provenance mismatch: {experiment}/{key}")
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
            raise Phase8SelectionError(f"OOF probability invalid: {experiment}/{key}")
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-6) or prediction != int(probabilities.argmax()):
            raise Phase8SelectionError(f"OOF probability/prediction mismatch: {experiment}/{key}")
    expected = {key for key, row in manifest.items() if expected_fold is None or row["fold"] == expected_fold}
    if seen != expected: raise Phase8SelectionError(f"OOF missing/extra prediction set: {experiment}/fold={expected_fold}")


def _metrics_from_oof(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    truth = np.asarray([int(row["true_label"]) for row in rows]); predictions = np.asarray([int(row["pred_label"]) for row in rows])
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=range(3), zero_division=0)
    return {"macro_f1": float(f1.mean()), "per_class": {name: {"precision": float(precision[index]),
        "recall": float(recall[index]), "f1": float(f1[index]), "support": int(support[index])}
        for index, name in enumerate(CLASS_NAMES)},
        "confusion_matrix": confusion_matrix(truth, predictions, labels=range(3)).tolist()}


def _assert_close(actual: Any, expected: Any, context: str) -> None:
    if isinstance(expected, Mapping):
        for key, value in expected.items():
            if key not in actual: raise Phase8SelectionError(f"Evidence missing {context}.{key}")
            _assert_close(actual[key], value, f"{context}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected): raise Phase8SelectionError(f"Evidence length mismatch: {context}")
        for index, value in enumerate(expected): _assert_close(actual[index], value, f"{context}[{index}]")
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not np.isclose(float(actual), float(expected), atol=TOLERANCE, rtol=0):
            raise Phase8SelectionError(f"Metric mismatch at {context}: {actual} != {expected}")
    elif actual != expected: raise Phase8SelectionError(f"Evidence mismatch at {context}: {actual} != {expected}")


def _common_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    return {key: provenance.get(key) for key in ("T", "D", "class_mapping", "manifest_sha256",
        "roi_config_sha256", "stage_a_config_sha256", "sampling", "normalization_identity",
        "seed_policy", "base_seed", "loss", "optimizer", "learning_rate", "weight_decay",
        "batch_size", "configured_epochs", "best_monitor", "git_commit_hash", "git_dirty",
        "git_diff_sha256")}


def validate_intended_folds(folds: Sequence[int]) -> None:
    if len(folds) != 5 or set(folds) != EXPECTED_FOLDS:
        raise Phase8SelectionError(f"Expected exactly one result for folds 0..4, found {list(folds)}")


def validate_and_aggregate_experiment(experiment: str, output_root: Path,
                                      manifest_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if experiment not in EXPERIMENTS: raise Phase8SelectionError(f"Unknown experiment: {experiment}")
    fold_metrics, fold_oof, provenances, evidence_hashes = [], [], [], {}
    for fold in sorted(EXPECTED_FOLDS):
        directory = output_root / f"phase7_{experiment}_fold{fold}"
        required = {name: directory / name for name in ("best.pt", "effective_config.json", "fold_metrics.json", "oof_predictions.csv")}
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing: raise Phase8SelectionError(f"Missing Phase 7 fold evidence: {missing}")
        payload = torch.load(required["best.pt"], map_location="cpu", weights_only=False); provenance = payload.get("provenance", {})
        definition = EXPERIMENTS[experiment]
        if (provenance.get("phase") != 7 or provenance.get("experiment") != experiment
                or provenance.get("fold") != fold or provenance.get("smoke") is not False
                or provenance.get("encoder") != definition["encoder"]
                or provenance.get("encoder_key") != definition["encoder_key"]
                or provenance.get("stage_b") != definition["stage_b"]):
            raise Phase8SelectionError(f"Fold checkpoint identity mismatch: {experiment}/fold={fold}")
        if provenance.get("mlflow_run_id") in (None, ""): raise Phase8SelectionError(f"Missing MLflow run provenance: {experiment}/fold={fold}")
        provenances.append(dict(provenance)); metrics = _read_json(required["fold_metrics.json"])
        if metrics.get("fold") != fold: raise Phase8SelectionError(f"Duplicate/wrong fold metric: {experiment}/fold={fold}")
        oof = _read_oof(required["oof_predictions.csv"]); _validate_oof_rows(oof, manifest_rows, experiment, fold)
        recalculated = _metrics_from_oof(oof); _assert_close(metrics, recalculated, f"{experiment}.fold{fold}")
        fold_metrics.append(metrics); fold_oof.extend(oof)
        evidence_hashes[str(fold)] = {name: _sha256(path) for name, path in required.items()}
    validate_intended_folds([int(provenance["fold"]) for provenance in provenances])
    if len({provenance["mlflow_run_id"] for provenance in provenances}) != 5: raise Phase8SelectionError(f"Duplicate MLflow run: {experiment}")
    common = {_json_canonical(_common_provenance(provenance)) for provenance in provenances}
    if len(common) != 1: raise Phase8SelectionError(f"Within-experiment fairness mismatch: {experiment}")
    aggregate_dir = output_root / f"phase7_{experiment}_aggregate"
    stored_summary = _read_json(aggregate_dir / "summary.json"); aggregate_oof = _read_oof(aggregate_dir / "oof_predictions.csv")
    _validate_oof_rows(aggregate_oof, manifest_rows, experiment)
    if _oof_identity(aggregate_oof) != _oof_identity(fold_oof): raise Phase8SelectionError(f"Fold/aggregate OOF mismatch: {experiment}")
    recalculated = aggregate_fold_results(fold_metrics, [{"true_label": int(row["true_label"]),
        "pred_label": int(row["pred_label"])} for row in fold_oof])
    _assert_close(stored_summary, recalculated, f"{experiment}.aggregate")
    return {"experiment": experiment, "definition": dict(EXPERIMENTS[experiment]),
        "fold_metrics": fold_metrics, "aggregate": recalculated, "provenance": provenances[0],
        "evidence_hashes": {"folds": evidence_hashes, "summary": _sha256(aggregate_dir / "summary.json"),
                            "oof": _sha256(aggregate_dir / "oof_predictions.csv")}}


def _json_canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _oof_identity(rows: Sequence[Mapping[str, Any]]) -> str:
    return _json_canonical(sorted((dict(row) for row in rows), key=lambda row: str(row["clip_key"])))


def validate_cross_experiment_fairness(results: Mapping[str, Mapping[str, Any]]) -> None:
    if set(results) != set(EXPERIMENTS): raise Phase8SelectionError("Fairness requires experiments A/B/C/D")
    common = {_json_canonical(_common_provenance(result["provenance"])) for result in results.values()}
    if len(common) != 1: raise Phase8SelectionError("Cross-experiment common provenance differs")
    for experiment, result in results.items():
        definition, provenance = EXPERIMENTS[experiment], result["provenance"]
        expected_identifier_type = "torchvision:" if definition["encoder_key"] == "encoder_a" else "stage-a-"
        if not str(provenance.get("encoder_cache_identifier", "")).startswith(expected_identifier_type):
            raise Phase8SelectionError(f"Encoder cache provenance mismatch: {experiment}")


def _simplicity_score(stage_b: str) -> int:
    return 1 if stage_b == "mean" else 0


def rank_experiments(results: Mapping[str, Mapping[str, Any]]) -> tuple[list[str], bool]:
    def key(experiment: str) -> tuple[Any, ...]:
        aggregate = results[experiment]["aggregate"]
        return (aggregate["macro_f1_mean"], aggregate["class_recall"]["복약"]["mean"],
            aggregate["class_recall"]["음수"]["mean"], -aggregate["macro_f1_std"],
            _simplicity_score(results[experiment]["definition"]["stage_b"]),
            tuple(-ord(character) for character in experiment))
    ordered = sorted(results, key=key, reverse=True)
    primary_values = [results[experiment]["aggregate"]["macro_f1_mean"] for experiment in ordered]
    tie_break_used = len(primary_values) > 1 and primary_values[0] == primary_values[1]
    return ordered, tie_break_used


def effect_analysis(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    pairs = {"B_minus_A": ("B", "A"), "D_minus_C": ("D", "C"),
             "C_minus_A": ("C", "A"), "D_minus_B": ("D", "B"), "D_minus_A": ("D", "A")}
    effects = {}
    for name, (left, right) in pairs.items():
        left_aggregate, right_aggregate = results[left]["aggregate"], results[right]["aggregate"]
        effects[name] = {"macro_f1_mean": left_aggregate["macro_f1_mean"] - right_aggregate["macro_f1_mean"],
            "class_recall_mean": {class_name: left_aggregate["class_recall"][class_name]["mean"]
                - right_aggregate["class_recall"][class_name]["mean"] for class_name in CLASS_NAMES}}
    return effects


def build_selection_artifact(results: Mapping[str, Mapping[str, Any]], manifest_rows: Sequence[Mapping[str, Any]],
                             phase7_config_path: Path, phase7_output_relative: str,
                             encoder_b_checkpoint_relative: str) -> dict[str, Any]:
    validate_cross_experiment_fairness(results); ranking, tie_break_used = rank_experiments(results)
    selected_id = ranking[0]; selected = results[selected_id]; aggregate, provenance = selected["aggregate"], selected["provenance"]
    ranking_rows = [{"rank": rank, "experiment": experiment,
        "macro_f1_mean": results[experiment]["aggregate"]["macro_f1_mean"],
        "macro_f1_std": results[experiment]["aggregate"]["macro_f1_std"]}
        for rank, experiment in enumerate(ranking, 1)]
    return {"phase8": {"status": "complete", "selected_experiment": selected_id,
        "selected_rank": 1,
        "encoder": {"type": selected["definition"]["encoder"], "cache_key": selected["definition"]["encoder_key"],
            "identifier": provenance["encoder_cache_identifier"],
            "checkpoint_relative_path": encoder_b_checkpoint_relative if selected["definition"]["encoder_key"] == "encoder_b" else None,
            "frozen": True},
        "stage_b": {"type": selected["definition"]["stage_b"], "model_config": provenance["model_config"]},
        "selection": {"primary_metric": PRIMARY_METRIC, "primary_value": aggregate["macro_f1_mean"],
            "macro_f1_std": aggregate["macro_f1_std"],
            "class_recall": {name: {"mean": aggregate["class_recall"][name]["mean"],
                "std": aggregate["class_recall"][name]["std"]} for name in CLASS_NAMES},
            "ranking": ranking_rows, "secondary_tie_break_used": tie_break_used,
            "reason": "Highest 5-fold mean Macro-F1; no numerical tie, so secondary criteria were not used."},
        "provenance": {"phase7_config": str(phase7_config_path), "phase7_config_sha256": _sha256(phase7_config_path),
            "phase7_output_relative_path": phase7_output_relative, "phase7_implementation_commit": PHASE7_COMMIT,
            "manifest_sha256": provenance["manifest_sha256"], "fold_assignment_sha256": fold_assignment_sha256(manifest_rows),
            "fold_ids": sorted(EXPECTED_FOLDS), "selected_valid_clips": len(manifest_rows),
            "roi_config_sha256": provenance["roi_config_sha256"], "stage_a_config_sha256": provenance["stage_a_config_sha256"],
            "sampling": provenance["sampling"], "T": provenance["T"], "embedding_dimension": provenance["D"],
            "normalization": provenance["normalization_identity"], "class_mapping": CLASS_TO_INDEX,
            "source_evidence": {
                experiment: results[experiment]["evidence_hashes"]
                for experiment in EXPERIMENTS
            }},
        "effects": effect_analysis(results),
        "phase9": {"status": "ready_to_start", "retrain_scope": "etri_pilot_all_selected_valid",
            "encoder_frozen": True, "stage_b_reinitialize": True,
            "cv_fold_checkpoint_reuse": False, "deployment_checkpoint_created": False}}}


def run_selection(project_root: Path, work_root: Path, manifest: Path,
                  phase7_config_path: Path) -> dict[str, Any]:
    config = load_stage_b_config(phase7_config_path); validate_experiment_fairness(config)
    phase6 = load_embedding_config(Path(config.phase6_config_path))
    manifest_rows, manifest_hash = read_selected_etri_manifest(manifest, Path(phase6.pilot_config_path))
    if manifest_hash != config.manifest_sha256: raise Phase8SelectionError("Phase 7/manifest hash mismatch")
    validate_fold_contract(manifest_rows)
    if any(row.get("action") in {"A045", "A046", "A047", "A048"} for row in manifest_rows):
        raise Phase8SelectionError("Excluded multi-person action is present")
    output_root = work_root / config.output_relative_path
    results = {experiment: validate_and_aggregate_experiment(experiment, output_root, manifest_rows)
               for experiment in EXPERIMENTS}
    return build_selection_artifact(results, manifest_rows, phase7_config_path,
        config.output_relative_path, phase6.encoder_b_checkpoint_relative_path)


def write_selection_artifact(path: Path, artifact: Mapping[str, Any]) -> str:
    content = yaml.safe_dump(dict(artifact), allow_unicode=True, sort_keys=False)
    if path.exists():
        if path.read_text(encoding="utf-8") != content: raise Phase8SelectionError(f"Existing selection artifact differs: {path}")
        return "UNCHANGED"
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8"); temporary.replace(path); return "CREATED"
