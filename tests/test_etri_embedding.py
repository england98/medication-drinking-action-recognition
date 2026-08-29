from __future__ import annotations

import copy
import tempfile
import unittest
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.run_etri_embedding_cache import (_safe_output_root, build_summary,
    validate_manifest_hash_unchanged, validate_summary_gate)
from src.etri_embedding import (REQUIRED_PROVENANCE_FIELDS, EtriEmbeddingError,
    deterministic_cache_name,
    encoder_a_training_provenance, encoder_b_training_provenance, extract_encoder_pair,
    extract_frozen_embeddings, freeze_encoder,
    load_embedding_config, make_cache_payload, make_encoder_provenance, preprocess_shared_frames,
    read_selected_etri_manifest, resolve_video_path, save_and_reload_cache, source_clip_key,
    uniform_frame_indices, validate_cache_payload, validate_embedding, validate_manifest_frame_count)
from src.roi_preflight import Detection, load_roi_config
from src.stage_a import StageAMobileNetV3


def row() -> dict:
    return {"participant": "P201", "action": "A003", "take": "G001", "height": "H070",
            "fold": 2, "target_class": "복약"}


def manifest_row(selected: bool = True) -> dict:
    return {"dataset": "etri", "root_key": "etri_raw_root", "relative_path": "clip.mp4", "batch": "B",
        "participant": "P201", "action": "A003", "target_class": "복약", "fold": 2,
        "height": "H070", "take": "G001", "is_multi_person": False, "is_hard_negative": False,
        "valid": True, "exclusion_reason": "", "pilot_selected": selected,
        "pilot_selection_reason": "target", "pilot_seed": 42, "roi_status": "pending", "rgb_frame_count": 64}


def synthetic_summary_inputs(count: int = 239) -> tuple[list[dict], list[dict]]:
    rows, records = [], []
    for index in range(count):
        value = manifest_row()
        value["participant"] = f"P{index:03d}"
        value["fold"] = index % 5 + 1
        value["target_class"] = ("복약", "음수", "기타")[index % 3]
        rows.append(value)
        records.append({"source_clip_key": source_clip_key(value), "participant": value["participant"],
            "fold": value["fold"], "target_class": value["target_class"],
            "roi_counts": {"success": 0, "partial": 64, "fallback": 0},
            "encoder_a_shape": [64, 1024], "encoder_a_nan": 0, "encoder_a_inf": 0,
            "encoder_b_shape": [64, 1024], "encoder_b_nan": 0, "encoder_b_inf": 0,
            "clip_key_parity": True, "frame_index_parity": True})
    return rows, records


def valid_cache_payload() -> tuple[dict, dict, object]:
    config = load_embedding_config(Path("configs/etri_embedding.yaml"))
    value = manifest_row(); value["rgb_frame_count"] = 120
    a = make_encoder_provenance(value, "imagenet_only", "a", config, "roi", "stage-a", 224,
                                encoder_a_training_provenance())
    b_training = {"source": "ai_hub_stage_a", "fine_tuning_mode": "last_n_blocks",
                  "frozen_batchnorm_policy": "freeze_running_stats"}
    b = make_encoder_provenance(value, "ai_hub_fine_tuned", "b", config, "roi", "stage-a", 224, b_training)
    payload = make_cache_payload(uniform_frame_indices(120), torch.zeros(64, 1024), torch.zeros(64, 1024),
        a, b, {"success": 0, "partial": 64, "fallback": 0}, 120, "manifest")
    return payload, value, config


class RecordingEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.anchor = torch.nn.Parameter(torch.zeros(1)); self.seen = []
    def extract_features(self, images):
        self.seen.append(images.detach().clone()); return images.mean((1, 2, 3), keepdim=False)[:, None].repeat(1, 1024)


class EtriEmbeddingTests(unittest.TestCase):
    def test_deterministic_cache_name_rejects_ambiguous_or_malformed_keys(self) -> None:
        self.assertEqual(deterministic_cache_name("P201:A003:G006:H070"), "P201__A003__G006__H070.pt")
        with self.assertRaises(EtriEmbeddingError): deterministic_cache_name("P_201:A003:G006:H070")
        for malformed in ("P201:A003:G006", "P201::G006:H070", "P201:A003:G006:H070:extra"):
            with self.subTest(malformed=malformed), self.assertRaises(EtriEmbeddingError):
                deterministic_cache_name(malformed)

    def test_uniform_sampling_contract(self) -> None:
        indices = uniform_frame_indices(101, 64)
        self.assertEqual((len(indices), indices[0], indices[-1]), (64, 0, 100))
        self.assertTrue(all(left <= right for left, right in zip(indices, indices[1:])))
        self.assertEqual(len(uniform_frame_indices(10, 64)), 64)
        with self.assertRaises(EtriEmbeddingError): uniform_frame_indices(0, 64)

    def test_embedding_shape_and_finite_contract(self) -> None:
        validate_embedding(torch.zeros(64, 1024), 64, 1024)
        with self.assertRaisesRegex(EtriEmbeddingError, "shape"): validate_embedding(torch.zeros(63, 1024), 64, 1024)
        bad = torch.zeros(64, 1024); bad[0, 0] = float("nan")
        with self.assertRaisesRegex(EtriEmbeddingError, "NaN or Inf"): validate_embedding(bad, 64, 1024)

    def test_encoder_freeze_contract(self) -> None:
        model = freeze_encoder(StageAMobileNetV3(pretrained=False))
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

    def test_provenance_required_fields(self) -> None:
        config = load_embedding_config(Path("configs/etri_embedding.yaml"))
        provenance = make_encoder_provenance(row(), "imagenet_only", "weights", config, "roi", "stage-a", 224,
                                             encoder_a_training_provenance())
        self.assertFalse(REQUIRED_PROVENANCE_FIELDS - set(provenance))
        self.assertEqual(provenance["source_clip_key"], "P201:A003:G001:H070")
        self.assertEqual((provenance["T"], provenance["D"]), (64, 1024))
        self.assertEqual(provenance["inference_batch_size"], config.inference_batch_size)
        self.assertEqual(provenance["encoder_training"]["fine_tuning_mode"], "none")
        checkpoint = {"model_config": {"fine_tuning_mode": "last_n_blocks"},
                      "frozen_batchnorm_policy": "freeze_running_stats", "run_id": "run", "best_epoch": 12}
        training = encoder_b_training_provenance(checkpoint)
        self.assertEqual(training["fine_tuning_mode"], "last_n_blocks")
        self.assertEqual(training["frozen_batchnorm_policy"], "freeze_running_stats")

    def test_shared_clip_and_indices_cache_contract(self) -> None:
        config = load_embedding_config(Path("configs/etri_embedding.yaml"))
        training = encoder_a_training_provenance()
        a = make_encoder_provenance(row(), "imagenet_only", "a", config, "roi", "stage-a", 224, training)
        b = make_encoder_provenance(row(), "ai_hub_fine_tuned", "b", config, "roi", "stage-a", 224, training)
        indices = uniform_frame_indices(120, 64); embedding_a = torch.randn(64, 1024); embedding_b = torch.randn(64, 1024)
        payload = make_cache_payload(indices, embedding_a, embedding_b, a, b,
                                     {"success": 1, "partial": 63, "fallback": 0}, 120, "manifest")
        self.assertEqual(payload["sampled_frame_indices"], list(indices))
        self.assertEqual(payload["encoders"]["encoder_a"]["sampled_frame_indices"],
                         payload["encoders"]["encoder_b"]["sampled_frame_indices"])
        self.assertEqual(payload["encoders"]["encoder_a"]["provenance"]["source_clip_key"],
                         payload["encoders"]["encoder_b"]["provenance"]["source_clip_key"])
        with tempfile.TemporaryDirectory() as directory:
            loaded = save_and_reload_cache(Path(directory) / "smoke.pt", payload)
            self.assertTrue(torch.equal(loaded["encoders"]["encoder_a"]["embedding"], embedding_a))
            self.assertEqual(loaded["encoders"]["encoder_b"]["provenance"]["inference_batch_size"], 8)

    def test_mismatched_encoder_clip_key_fails(self) -> None:
        config = load_embedding_config(Path("configs/etri_embedding.yaml"))
        training = encoder_a_training_provenance()
        a = make_encoder_provenance(row(), "imagenet_only", "a", config, "roi", "stage-a", 224, training)
        other = row(); other["participant"] = "P202"
        b = make_encoder_provenance(other, "ai_hub_fine_tuned", "b", config, "roi", "stage-a", 224, training)
        with self.assertRaisesRegex(EtriEmbeddingError, "clip keys differ"):
            make_cache_payload(range(64), torch.zeros(64, 1024), torch.zeros(64, 1024), a, b,
                               {"success": 0, "partial": 0, "fallback": 64}, 64, "manifest")

    def test_cache_resume_validation_rejects_incomplete_provenance(self) -> None:
        config = load_embedding_config(Path("configs/etri_embedding.yaml")); value = manifest_row(); value["rgb_frame_count"] = 120
        a_training = encoder_a_training_provenance()
        b_training = {"source": "ai_hub_stage_a", "fine_tuning_mode": "last_n_blocks",
                      "frozen_batchnorm_policy": "freeze_running_stats"}
        a = make_encoder_provenance(value, "imagenet_only", "a", config, "roi", "stage-a", 224, a_training)
        b = make_encoder_provenance(value, "ai_hub_fine_tuned", "b", config, "roi", "stage-a", 224, b_training)
        payload = make_cache_payload(uniform_frame_indices(120), torch.zeros(64, 1024), torch.zeros(64, 1024),
            a, b, {"success": 0, "partial": 64, "fallback": 0}, 120, "manifest")
        validate_cache_payload(payload, value, "manifest", "a", "b", config, "roi", "stage-a")
        del payload["encoders"]["encoder_b"]["provenance"]["inference_batch_size"]
        with self.assertRaisesRegex(EtriEmbeddingError, "incomplete"):
            validate_cache_payload(payload, value, "manifest", "a", "b", config, "roi", "stage-a")

    def test_cache_resume_rejects_critical_identity_mutations(self) -> None:
        payload, value, config = valid_cache_payload()
        mutations = {
            "encoder_checkpoint_identifier": lambda item: item["encoders"]["encoder_b"]["provenance"].__setitem__(
                "encoder_checkpoint_identifier", "wrong"),
            "manifest_sha256": lambda item: item.__setitem__("manifest_sha256", "wrong"),
            "sampled_frame_indices": lambda item: (
                item["sampled_frame_indices"].__setitem__(1, item["sampled_frame_indices"][1] + 1),
                item["encoders"]["encoder_a"]["sampled_frame_indices"].__setitem__(
                    1, item["encoders"]["encoder_a"]["sampled_frame_indices"][1] + 1),
                item["encoders"]["encoder_b"]["sampled_frame_indices"].__setitem__(
                    1, item["encoders"]["encoder_b"]["sampled_frame_indices"][1] + 1)),
            "participant": lambda item: item.__setitem__("participant", "P999"),
            "fold": lambda item: item.__setitem__("fold", 5),
            "target_class": lambda item: item.__setitem__("target_class", "기타"),
        }
        for name, mutate in mutations.items():
            with self.subTest(mutation=name):
                changed = copy.deepcopy(payload); mutate(changed)
                with self.assertRaises(EtriEmbeddingError):
                    validate_cache_payload(changed, value, "manifest", "a", "b", config, "roi", "stage-a")

    def test_shared_tensor_fairness(self) -> None:
        frames = torch.randn(6, 3, 4, 4); a, b = RecordingEncoder(), RecordingEncoder()
        embedding_a, embedding_b = extract_encoder_pair(a, b, frames, torch.device("cpu"), 6)
        self.assertTrue(torch.equal(a.seen[0], b.seen[0])); self.assertTrue(torch.equal(embedding_a, embedding_b))

    def test_roi_no_drop_for_all_statuses(self) -> None:
        class Detector:
            def __init__(self): self.index = 0
            def detect(self, image):
                choices = (Detection(((.2, .2), (.5, .5)), ((.5, .5), (.8, .8))),
                           Detection(face_points=((.2, .2), (.5, .5))), Detection())
                value = choices[self.index % 3]; self.index += 1; return value
            def close(self): pass
        frames = [np.zeros((40, 50, 3), np.uint8) for _ in range(64)]
        tensor, counts = preprocess_shared_frames(frames, Detector(), load_roi_config(Path("configs/roi_preflight.yaml")),
                                                   lambda image: torch.zeros(3, 16, 16))
        self.assertEqual(len(tensor), 64); self.assertEqual(sum(counts.values()), 64)
        self.assertTrue(all(counts[status] > 0 for status in ("success", "partial", "fallback")))

    def test_manifest_integrity_and_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"; rows = [manifest_row(True), manifest_row(False)]
            rows[1]["participant"] = "P202"; rows[1]["fold"] = 3; rows[1]["relative_path"] = "other.mp4"
            content = "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in rows)
            path.write_text(content, encoding="utf-8"); before = hashlib.sha256(path.read_bytes()).hexdigest()
            selected, digest = read_selected_etri_manifest(path, Path("configs/pilot_manifest.yaml"))
            self.assertEqual(len(selected), 1); self.assertTrue(selected[0]["pilot_selected"] and selected[0]["valid"])
            self.assertEqual(selected[0]["dataset"], "etri"); self.assertEqual(selected[0]["roi_status"], "pending")
            self.assertEqual(before, digest); self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bad = manifest_row(); bad["relative_path"] = "../escape.mp4"
            with self.assertRaisesRegex(EtriEmbeddingError, "Unsafe"):
                resolve_video_path(bad, {"etri_raw_root": Path(directory)})

    def test_frozen_inference_is_deterministic_and_no_grad(self) -> None:
        model = RecordingEncoder(); frames = torch.randn(5, 3, 4, 4)
        first = extract_frozen_embeddings(model, frames, torch.device("cpu"), 5)
        second = extract_frozen_embeddings(model, frames, torch.device("cpu"), 5)
        self.assertTrue(torch.equal(first, second)); self.assertFalse(first.requires_grad)
        self.assertFalse(model.training); self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_manifest_frame_count_cross_check(self) -> None:
        validate_manifest_frame_count(manifest_row(), 64)
        with self.assertRaisesRegex(EtriEmbeddingError, "frame count mismatch"):
            validate_manifest_frame_count(manifest_row(), 63)

    def test_limited_summary_gate(self) -> None:
        config = load_embedding_config(Path("configs/etri_embedding.yaml")); rows = [manifest_row()]
        record = {"source_clip_key": source_clip_key(rows[0]), "participant": "P201", "fold": 2,
            "target_class": "복약", "roi_counts": {"success": 0, "partial": 64, "fallback": 0},
            "encoder_a_shape": [64, 1024], "encoder_a_nan": 0, "encoder_a_inf": 0,
            "encoder_b_shape": [64, 1024], "encoder_b_nan": 0, "encoder_b_inf": 0,
            "clip_key_parity": True, "frame_index_parity": True}
        summary = build_summary(rows, [record], 1, [], 0, "manifest", "encoder-b", config, "roi", "stage-a", True)
        self.assertEqual(summary["status"], "PREFLIGHT_PASS"); validate_summary_gate(summary, True)

    def test_full_run_summary_gate_passes_exact_239_records(self) -> None:
        config = load_embedding_config(Path("configs/etri_embedding.yaml")); rows, records = synthetic_summary_inputs()
        summary = build_summary(rows, records, 227, [], 12, "manifest", "encoder-b", config, "roi", "stage-a", False)
        self.assertEqual(summary["status"], "PASS")
        validate_summary_gate(summary, False)

    def test_full_run_summary_gate_rejects_238_records_parity_and_failures(self) -> None:
        config = load_embedding_config(Path("configs/etri_embedding.yaml")); rows, records = synthetic_summary_inputs()
        cases = [
            (records[:-1], []),
            ([dict(record, frame_index_parity=False) if index == 0 else record
              for index, record in enumerate(records)], []),
            (records, [{"source_clip_key": None, "error": "synthetic failure"}]),
        ]
        for changed_records, failures in cases:
            with self.subTest(records=len(changed_records), failures=bool(failures)):
                summary = build_summary(rows, changed_records, len(changed_records), failures, 0,
                    "manifest", "encoder-b", config, "roi", "stage-a", False)
                self.assertEqual(summary["status"], "FAIL")
                with self.assertRaises(EtriEmbeddingError): validate_summary_gate(summary, False)

    def test_limited_summary_cannot_report_full_pass(self) -> None:
        config = load_embedding_config(Path("configs/etri_embedding.yaml")); rows, records = synthetic_summary_inputs()
        summary = build_summary(rows, records[:3], 3, [], 0, "manifest", "encoder-b", config, "roi", "stage-a", True)
        self.assertEqual(summary["status"], "PREFLIGHT_PASS")
        self.assertNotEqual(summary["status"], "PASS")
        validate_summary_gate(summary, True)

    def test_output_root_must_remain_inside_work_root(self) -> None:
        config = load_embedding_config(Path("configs/etri_embedding.yaml"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); work_root = root / "work"; work_root.mkdir()
            safe = _safe_output_root(work_root / "embeddings", work_root, config, "abcdef123456")
            self.assertEqual(safe, work_root / "embeddings" / "manifest-abcdef12")
            with self.assertRaisesRegex(EtriEmbeddingError, "inside work_root"):
                _safe_output_root(root / "outside", work_root, config, "abcdef123456")

    def test_manifest_mutation_cannot_reach_full_pass(self) -> None:
        config = load_embedding_config(Path("configs/etri_embedding.yaml")); rows, records = synthetic_summary_inputs()
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"; manifest.write_text("initial\n", encoding="utf-8")
            expected = hashlib.sha256(manifest.read_bytes()).hexdigest()
            manifest.write_text("mutated\n", encoding="utf-8")
            with self.assertRaisesRegex(EtriEmbeddingError, "changed during run") as raised:
                validate_manifest_hash_unchanged(manifest, expected)
            failures = [{"source_clip_key": None, "error": str(raised.exception)}]
            summary = build_summary(rows, records, 239, failures, 0, expected,
                                    "encoder-b", config, "roi", "stage-a", False)
            self.assertEqual(summary["status"], "FAIL")
            with self.assertRaises(EtriEmbeddingError): validate_summary_gate(summary, False)


if __name__ == "__main__": unittest.main()
