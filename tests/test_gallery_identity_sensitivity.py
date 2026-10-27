from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "11_analyze_gallery_identity_sensitivity.py"
spec = importlib.util.spec_from_file_location("gallery_identity_sensitivity_test", SCRIPT_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load experiment 11.3-C module")
sensitivity = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sensitivity
spec.loader.exec_module(sensitivity)


def _record(path: str, sample: str, label: str) -> dict[str, str]:
    return {
        "relative_path": path,
        "sample_id": sample,
        "sample_id_source": "filename_header_match",
        "measurement_stem": Path(path).stem,
        "reference_label": label,
        "excitation_wavelength_nm": "532",
    }


def _key(label: str) -> str:
    return sensitivity.exp10._reference_label_identity(_record("x.txt", "S", label))[2]


class GalleryIdentitySensitivityTests(unittest.TestCase):
    def test_full_gallery_baseline_validation_matches_published_metrics(self):
        results = [
            {"top1_label": "A", "top1_label_group_key": _key("A"), "correct_label_rank": 1, "hit_at_1": 1, "hit_at_3": 1, "hit_at_5": 1},
            {"top1_label": "B", "top1_label_group_key": _key("B"), "correct_label_rank": 2, "hit_at_1": 0, "hit_at_3": 1, "hit_at_5": 1},
        ]
        records = [
            {"reference_label_top1_reference_label": "A", "reference_label_matching_rank": "1", "reference_label_hit_at_1": "1", "reference_label_hit_at_3": "1", "reference_label_hit_at_5": "1"},
            {"reference_label_top1_reference_label": "B", "reference_label_matching_rank": "2", "reference_label_hit_at_1": "0", "reference_label_hit_at_3": "1", "reference_label_hit_at_5": "1"},
        ]
        metrics = sensitivity.aggregate_metrics(results, ["S1", "S2"])
        summary = {"metrics": {
            "record_level_118_spectra": {f"reference_label_top_{k}": metrics[f"record_top_{k}"] for k in sensitivity.TOP_K_VALUES},
            "sample_id_equal_weight_60_sample_ids": {f"reference_label_top_{k}": metrics[f"sample_equal_top_{k}"] for k in sensitivity.TOP_K_VALUES},
        }}
        sensitivity._validate_full_baseline(results, metrics, records, summary)

    def test_deleting_identity_removes_all_its_reference_files(self):
        records = (
            _record("s1-a.txt", "S1", "A"),
            _record("s1-b.txt", "S1", "A"),
            _record("s2.txt", "S2", "A"),
            _record("s3.txt", "S3", "A"),
        )
        matrix = np.arange(8, dtype=float).reshape(2, 4)
        identities = sensitivity.build_gallery_identity_map(records, minimum_ids_per_label=3)
        remaining_records, remaining_matrix = sensitivity.remove_gallery_identity(records, matrix, identities["s1"])
        self.assertEqual([row["relative_path"] for row in remaining_records], ["s2.txt", "s3.txt"])
        self.assertTrue(np.array_equal(remaining_matrix, matrix[:, [2, 3]]))

    def test_labels_must_have_three_ids_before_and_two_after_deletion(self):
        too_small = (_record("a.txt", "S1", "A"), _record("b.txt", "S2", "A"))
        with self.assertRaisesRegex(ValueError, "fewer than 3"):
            sensitivity.build_gallery_identity_map(too_small, minimum_ids_per_label=3)
        valid = too_small + (_record("c.txt", "S3", "A"),)
        identities = sensitivity.build_gallery_identity_map(valid, minimum_ids_per_label=3)
        remaining, matrix = sensitivity.remove_gallery_identity(valid, np.ones((1, 3)), identities["s1"])
        after = sensitivity.build_gallery_identity_map(remaining, minimum_ids_per_label=2)
        self.assertEqual(len(after), 2)
        self.assertEqual(matrix.shape, (1, 2))

    def test_method_a_deletion_ranking_is_deterministic_on_ties(self):
        records = (_record("z.txt", "Z1", "Zulu"), _record("a.txt", "A1", "Alpha"))
        matrix = np.asarray([[0.5, 0.5]])
        first = sensitivity.evaluate_method_a_matrix(matrix, records, [_key("Alpha")], expected_label_count=2)
        second = sensitivity.evaluate_method_a_matrix(matrix, records, [_key("Alpha")], expected_label_count=2)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["top1_label"], "Alpha")

    def test_topk_and_sample_equal_aggregation(self):
        results = [
            {"hit_at_1": 1, "hit_at_3": 1, "hit_at_5": 1},
            {"hit_at_1": 0, "hit_at_3": 1, "hit_at_5": 1},
            {"hit_at_1": 1, "hit_at_3": 1, "hit_at_5": 1},
        ]
        metrics = sensitivity.aggregate_metrics(results, ["S1", "S1", "S2"])
        self.assertAlmostEqual(metrics["record_top_1"], 2 / 3)
        self.assertAlmostEqual(metrics["sample_equal_top_1"], 0.75)
        self.assertEqual(metrics["sample_equal_top_3"], 1.0)

    def test_query_sample_stability_aggregates_multiple_measurements(self):
        manifest = [
            {"sample_id": "S1", "reference_label_group_display": "A", "reference_label_group_key": _key("A")},
            {"sample_id": "S1", "reference_label_group_display": "A", "reference_label_group_key": _key("A")},
        ]
        trackers = [
            {"ranks": [1, 2, 1], "top1_changed_scenarios": {2}, "hit_changed_scenarios": {1: {2}, 3: set(), 5: set()}},
            {"ranks": [1, 1, 3], "top1_changed_scenarios": {2, 3}, "hit_changed_scenarios": {1: {3}, 3: set(), 5: set()}},
        ]
        row = sensitivity.build_sample_stability(manifest, trackers, 3)[0]
        self.assertEqual(row["query_spectrum_count"], 2)
        self.assertEqual(row["measurement_scenario_evaluation_count"], 6)
        self.assertEqual(row["top1_prediction_changed_measurement_scenario_count"], 3)
        self.assertEqual(row["scenarios_with_any_top1_prediction_change"], 2)
        self.assertEqual(row["correct_label_rank_median"], 1.0)

    def test_all_scenarios_are_reproducible_and_leave_two_ids_per_label(self):
        records = tuple(
            [_record(f"a-{index}.txt", f"A{index}", "A") for index in range(3)]
            + [_record(f"b-{index}.txt", f"B{index}", "B") for index in range(3)]
        )
        similarities = np.asarray([[0.9, 0.8, 0.7, 0.3, 0.2, 0.1]])
        manifest = [{"sample_id": "Q1", "reference_label_group_key": _key("A")}]
        identities = sensitivity.build_gallery_identity_map(records, minimum_ids_per_label=3)
        baseline_results = sensitivity.evaluate_method_a_matrix(similarities, records, [_key("A")], expected_label_count=2)
        metrics = sensitivity.aggregate_metrics(baseline_results, ["Q1"])
        first = sensitivity.run_deletion_scenarios(
            similarities, records, manifest, baseline_results, metrics, identities,
            expected_label_count=2, minimum_remaining_ids_per_label=2,
        )
        second = sensitivity.run_deletion_scenarios(
            similarities, records, manifest, baseline_results, metrics, identities,
            expected_label_count=2, minimum_remaining_ids_per_label=2,
        )
        self.assertEqual(first[0], second[0])
        self.assertEqual(len(first[0]), 6)
        self.assertTrue(all(row["label_sample_id_count_after"] == 2 for row in first[0]))

    def test_similarity_matrix_contains_only_supplied_gallery_and_development(self):
        gallery = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        development = np.asarray([[1.0, 0.0]])
        result = sensitivity.compute_similarity_matrix(gallery, development)
        self.assertEqual(result.shape, (1, 2))
        self.assertTrue(np.allclose(result, [[1.0, 0.0]]))

    def test_source_hash_error_and_missing_input_stop(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "does not exist"):
                sensitivity.diagnostics.load_and_validate_baseline_results(root)
            names = ("evaluation_group_manifest.csv", "reference_label_sample_distribution.csv", "evaluation_group_audit.json", "EVALUATION_PROTOCOL.md")
            for name in names:
                (root / name).write_bytes(name.encode())
            cache = root / "cache.npz"
            index = root / "index.csv"
            cache.write_bytes(b"cache")
            index.write_bytes(b"index")

            def item(path: Path):
                return {"size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

            lock_data = {
                "evaluation_snapshot": {name: item(root / name) for name in names},
                "source_data": {"experiment_9_3_npz": item(cache), "experiment_9_3_index": item(index)},
                "frozen_evaluation": {},
            }
            lock_data["source_data"]["experiment_9_3_npz"]["sha256"] = "0" * 64
            lock = root / "lock.json"
            lock.write_text(json.dumps(lock_data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                sensitivity.baseline.validate_source_lock(lock, root, cache, index)

    def test_outputs_are_utf8_sig_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            output = Path(temporary) / "output"
            paths = sensitivity.write_outputs(output, [], [], [], [], {"ok": True}, "# report\n")
            self.assertTrue(all(path.exists() for path in paths))
            self.assertTrue(paths[0].read_bytes().startswith(b"\xef\xbb\xbf"))
            with self.assertRaises(FileExistsError):
                sensitivity.write_outputs(output, [], [], [], [], {"ok": True}, "# report\n")


if __name__ == "__main__":
    unittest.main()
