from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "11_compare_reference_label_aggregation.py"
spec = importlib.util.spec_from_file_location("reference_label_aggregation_test", SCRIPT_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load experiment 11.3-B module")
comparison = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = comparison
spec.loader.exec_module(comparison)


def _record(path: str, sample: str, label: str, source: str = "filename_header_match") -> dict[str, str]:
    return {
        "relative_path": path,
        "sample_id": sample,
        "sample_id_source": source,
        "measurement_stem": Path(path).stem,
        "reference_label": label,
        "excitation_wavelength_nm": "532",
    }


def _library(records: tuple[dict[str, str], ...], matrix: np.ndarray):
    return comparison.exp10.ReferenceLibrary(
        grid=np.arange(matrix.shape[1], dtype=float),
        all_success_matrix=matrix,
        all_success_records=records,
        reference_matrix=matrix,
        reference_records=records,
        path_to_success_row={row["relative_path"]: index for index, row in enumerate(records)},
    )


def _key(label: str) -> str:
    return comparison.exp10._reference_label_identity(_record("x.txt", "S", label))[2]


class ReferenceLabelAggregationTests(unittest.TestCase):
    def test_method_a_matches_experiment_10_label_ranking(self):
        records = (
            _record("a2.txt", "A1", "A"),
            _record("a1.txt", "A2", "A"),
            _record("b.txt", "B1", "B"),
        )
        matrix = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        library = _library(records, matrix)
        query = comparison.exp10.QuerySpectrum(vector=np.asarray([1.0, 0.0]), mode="test", relative_path="q.txt")
        similarities = comparison.exp10.compute_cosine_similarities(matrix, query.vector)
        groups = comparison.validate_gallery_for_aggregation(records)
        ours = comparison.rank_reference_labels(similarities, records, groups, comparison.METHOD_A)
        existing = comparison.exp10.search_topk(
            library, query, 2, result_level="reference_label", exclude_self=False, exclude_same_sample=False
        )
        self.assertEqual([row["reference_label_group_key"] for row in ours], [row["reference_label_group_key"] for row in existing])
        self.assertTrue(np.allclose([row["score"] for row in ours], [row["cosine_similarity"] for row in existing]))
        self.assertEqual(ours[0]["record"]["relative_path"], existing[0]["record"]["relative_path"])

    def test_method_b_averages_measurements_within_sample_id(self):
        records = (
            _record("s1-a.txt", "S1", "A"),
            _record("s1-b.txt", "S1", "A"),
            _record("s2.txt", "S2", "A"),
            _record("b.txt", "B1", "B"),
        )
        similarities = np.asarray([0.9, 0.1, 0.6, 0.2])
        ranking = comparison.rank_reference_labels(
            similarities, records, comparison.validate_gallery_for_aggregation(records), comparison.METHOD_B
        )
        a = next(row for row in ranking if row["reference_label_group_key"] == _key("A"))
        self.assertAlmostEqual(a["score"], ((0.9 + 0.1) / 2 + 0.6) / 2)

    def test_method_b_gives_different_sample_ids_equal_weight(self):
        records = tuple(
            [_record(f"s1-{index}.txt", "S1", "A") for index in range(4)]
            + [_record("s2.txt", "S2", "A"), _record("b.txt", "B1", "B")]
        )
        similarities = np.asarray([1.0, 1.0, 1.0, 1.0, 0.0, 0.2])
        ranking = comparison.rank_reference_labels(
            similarities, records, comparison.validate_gallery_for_aggregation(records), comparison.METHOD_B
        )
        a = next(row for row in ranking if row["reference_label_group_key"] == _key("A"))
        self.assertAlmostEqual(a["score"], 0.5)
        self.assertNotAlmostEqual(a["score"], 0.8)

    def test_duplicate_measurement_count_does_not_increase_identity_weight(self):
        base_records = (
            _record("s1.txt", "S1", "A"),
            _record("s2.txt", "S2", "A"),
            _record("b.txt", "B1", "B"),
        )
        duplicate_records = (
            _record("s1.txt", "S1", "A"),
            _record("s1-copy.txt", "S1", "A"),
            _record("s2.txt", "S2", "A"),
            _record("b.txt", "B1", "B"),
        )
        base = comparison.rank_reference_labels(
            np.asarray([1.0, 0.0, 0.2]), base_records,
            comparison.validate_gallery_for_aggregation(base_records), comparison.METHOD_B
        )
        duplicate = comparison.rank_reference_labels(
            np.asarray([1.0, 1.0, 0.0, 0.2]), duplicate_records,
            comparison.validate_gallery_for_aggregation(duplicate_records), comparison.METHOD_B
        )
        base_a = next(row for row in base if row["reference_label_group_key"] == _key("A"))
        duplicate_a = next(row for row in duplicate if row["reference_label_group_key"] == _key("A"))
        self.assertEqual(base_a["score"], duplicate_a["score"])

    def test_label_ties_use_deterministic_label_then_path_order(self):
        records = (_record("z.txt", "Z1", "Zulu"), _record("a.txt", "A1", "Alpha"))
        groups = comparison.validate_gallery_for_aggregation(records)
        first = comparison.rank_reference_labels(np.asarray([0.5, 0.5]), records, groups, comparison.METHOD_B)
        second = comparison.rank_reference_labels(np.asarray([0.5, 0.5]), records, groups, comparison.METHOD_B)
        self.assertEqual([row["reference_label_group_display"] for row in first], ["Alpha", "Zulu"])
        self.assertEqual([row["reference_label_group_key"] for row in first], [row["reference_label_group_key"] for row in second])

    def test_both_methods_consume_same_similarity_vector_and_gallery(self):
        records = (_record("a.txt", "A1", "A"), _record("b.txt", "B1", "B"))
        values = np.asarray([0.7, 0.4])
        groups = comparison.validate_gallery_for_aggregation(records)
        a = comparison.rank_reference_labels(values, records, groups, comparison.METHOD_A)
        b = comparison.rank_reference_labels(values, records, groups, comparison.METHOD_B)
        self.assertEqual({row["reference_label_group_key"] for row in a}, {row["reference_label_group_key"] for row in b})
        self.assertEqual(sum(row["label_reference_file_count"] for row in a), 2)
        self.assertEqual(sum(row["label_reference_file_count"] for row in b), 2)

    def test_missing_label_duplicate_path_and_cross_label_sample_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing or unusable"):
            comparison.validate_gallery_for_aggregation((_record("a.txt", "S1", "Unknown"),))
        with self.assertRaisesRegex(ValueError, "duplicate relative_path"):
            comparison.validate_gallery_for_aggregation((_record("a.txt", "S1", "A"), _record("a.txt", "S2", "A")))
        with self.assertRaisesRegex(ValueError, "multiple labels"):
            comparison.validate_gallery_for_aggregation((_record("a.txt", "S1", "A"), _record("b.txt", "S1", "B")))

    def test_missing_or_conflicted_sample_id_and_nonfinite_similarity_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Sample_ID"):
            comparison.validate_gallery_for_aggregation((_record("a.txt", "Unknown", "A", "missing"),))
        records = (_record("a.txt", "S1", "A"), _record("b.txt", "S2", "B"))
        groups = comparison.validate_gallery_for_aggregation(records)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            comparison.rank_reference_labels(np.asarray([np.nan, 0.2]), records, groups, comparison.METHOD_B)

    def test_final_test_is_not_passed_to_similarity(self):
        records = tuple(
            _record(f"g-{index:02d}.txt", f"G{index:02d}", f"L{index:02d}")
            for index in range(60)
        )
        matrix = np.eye(60, dtype=float)
        library = _library(records, matrix)
        manifest = [{
            "relative_path": "dev.txt", "cache_row_index": "2", "sample_id": "DEV",
            "reference_label_group_display": "L00", "reference_label_group_key": _key("L00"),
            "excitation_wavelength_nm": "532",
        }]
        baseline_rows = [{
            "query_relative_path": "dev.txt", "query_cache_row_index": "2",
            "query_sample_id": "DEV", "query_reference_label_group_key": _key("L00"),
            "reference_label_top1_reference_label": "L00", "reference_label_matching_rank": "1",
            "reference_label_top1_cosine_similarity": "1.0",
            "reference_label_hit_at_1": "1", "reference_label_hit_at_3": "1", "reference_label_hit_at_5": "1",
        }]
        original = comparison.exp10.compute_cosine_similarities
        observed: list[tuple[int, int]] = []

        def checked(matrix_arg, query_arg):
            observed.append((matrix_arg.shape[0], query_arg.shape[0]))
            return original(matrix_arg, query_arg)

        with mock.patch.object(comparison.exp10, "compute_cosine_similarities", side_effect=checked):
            rows = comparison.compare_queries(library, manifest, np.eye(60, dtype=float)[[0]], baseline_rows)
        self.assertEqual(observed, [(60, 60)])
        self.assertNotIn("final.txt", [row["relative_path"] for row in library.reference_records])
        self.assertEqual(len(rows), 1)

    def test_source_hash_error_stops_before_comparison(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
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
            lock_data["source_data"]["experiment_9_3_index"]["sha256"] = "0" * 64
            lock = root / "lock.json"
            lock.write_text(json.dumps(lock_data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                comparison.baseline.validate_source_lock(lock, root, cache, index)

    def test_baseline_identity_mapping_error_stops(self):
        records = (_record("g-a.txt", "GA", "A"), _record("g-b.txt", "GB", "B"))
        matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        library = _library(records, matrix)
        manifest = [{
            "relative_path": "dev.txt", "cache_row_index": "2", "sample_id": "DEV",
            "reference_label_group_display": "A", "reference_label_group_key": _key("A"),
            "excitation_wavelength_nm": "532",
        }]
        wrong = [{"query_relative_path": "other.txt"}]
        with self.assertRaisesRegex(ValueError, "identities differ"):
            comparison.compare_queries(library, manifest, np.asarray([[1.0, 0.0]]), wrong)

    def test_outputs_are_utf8_sig_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            output = Path(temporary) / "output"
            paths = comparison.write_outputs(output, [], [], [], [], [], {"ok": True}, "# report\n")
            self.assertTrue(all(path.exists() for path in paths))
            self.assertTrue(paths[0].read_bytes().startswith(b"\xef\xbb\xbf"))
            with self.assertRaises(FileExistsError):
                comparison.write_outputs(output, [], [], [], [], [], {"ok": True}, "# report\n")


if __name__ == "__main__":
    unittest.main()
