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
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "11_analyze_development_errors.py"
spec = importlib.util.spec_from_file_location("development_error_analysis_test", SCRIPT_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load experiment 11.3-A module")
analysis = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = analysis
spec.loader.exec_module(analysis)


def _record(path: str, sample: str, label: str) -> dict[str, str]:
    return {
        "relative_path": path,
        "sample_id": sample,
        "sample_id_source": "filename_header_match",
        "measurement_stem": Path(path).stem,
        "reference_label": label,
        "excitation_wavelength_nm": "532",
    }


def _hit(rank: int, score: float, label: str, path: str, sample: str) -> dict[str, object]:
    record = _record(path, sample, label)
    display, basis, key = analysis.exp10._reference_label_identity(record)
    return {
        "rank": rank,
        "cosine_similarity": score,
        "record": record,
        "reference_label_group_display": display,
        "reference_label_grouping_basis": basis,
        "reference_label_group_key": key,
    }


def _target_key(label: str) -> str:
    return analysis.exp10._reference_label_identity(
        _record("query.txt", "Q", label)
    )[2]


def _diagnostic_row(
    path: str,
    sample: str,
    label: str,
    prediction: str,
    rank: int,
    margin: float,
) -> dict[str, object]:
    return {
        "query_relative_path": path,
        "query_sample_id": sample,
        "query_reference_label": label,
        "query_reference_label_group_key": _target_key(label),
        "query_excitation_wavelength_nm": "532",
        "top1_prediction_label": prediction,
        "top1_prediction_label_group_key": _target_key(prediction),
        "reference_label_hit_at_1": int(rank <= 1),
        "reference_label_hit_at_3": int(rank <= 3),
        "reference_label_hit_at_5": int(rank <= 5),
        "correct_label_rank": rank,
        "correct_label_score": 0.8,
        "top1_score": 0.9,
        "is_top1_error": int(rank != 1),
        "correct_minus_highest_wrong_score": margin,
    }


class DevelopmentErrorAnalysisTests(unittest.TestCase):
    def test_correct_label_rank_and_topk_hits(self):
        hits = [
            _hit(1, 0.95, "Wrong A", "a.txt", "A"),
            _hit(2, 0.90, "Wrong B", "b.txt", "B"),
            _hit(3, 0.85, "Target", "target.txt", "T"),
            _hit(4, 0.80, "Wrong C", "c.txt", "C"),
        ]
        result = analysis.extract_full_label_diagnostic(hits, _target_key("Target"))
        self.assertEqual(result["correct_label_rank"], 3)
        self.assertEqual(result["reference_label_hit_at_1"], 0)
        self.assertEqual(result["reference_label_hit_at_3"], 1)
        self.assertEqual(result["reference_label_hit_at_5"], 1)

    def test_score_margin_sign_definition(self):
        positive = analysis.extract_full_label_diagnostic(
            [_hit(1, 0.9, "Target", "t.txt", "T"), _hit(2, 0.8, "Wrong", "w.txt", "W")],
            _target_key("Target"),
        )
        negative = analysis.extract_full_label_diagnostic(
            [_hit(1, 0.9, "Wrong", "w.txt", "W"), _hit(2, 0.8, "Target", "t.txt", "T")],
            _target_key("Target"),
        )
        self.assertAlmostEqual(positive["correct_minus_highest_wrong_score"], 0.1)
        self.assertEqual(positive["score_margin_relation"], "positive")
        self.assertAlmostEqual(negative["correct_minus_highest_wrong_score"], -0.1)
        self.assertEqual(negative["score_margin_relation"], "negative")

    def test_tied_scores_are_zero_margin_and_keep_stable_input_order(self):
        hits = [
            _hit(1, 0.9, "Wrong", "a.txt", "A"),
            _hit(2, 0.9, "Target", "b.txt", "B"),
        ]
        result = analysis.extract_full_label_diagnostic(hits, _target_key("Target"))
        self.assertEqual(result["score_margin_relation"], "zero_tie")
        self.assertTrue(result["top_score_tie"])
        self.assertEqual(result["top1_prediction_relative_path"], "a.txt")

    def test_full_ranking_recovers_truth_absent_from_truncated_top5(self):
        hits = [
            _hit(index, 1.0 - index * 0.05, f"Wrong {index}", f"w{index}.txt", f"W{index}")
            for index in range(1, 6)
        ]
        hits.append(_hit(6, 0.65, "Target", "target.txt", "T"))
        result = analysis.extract_full_label_diagnostic(hits, _target_key("Target"))
        self.assertEqual(result["correct_label_rank"], 6)
        self.assertEqual(result["reference_label_hit_at_5"], 0)
        self.assertEqual(result["correct_label_score"], 0.65)

    def test_missing_target_from_full_ranking_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly once"):
            analysis.extract_full_label_diagnostic(
                [_hit(1, 0.9, "Wrong", "w.txt", "W")], _target_key("Target")
            )

    def test_sample_aggregation_detects_partial_hits_and_inconsistent_predictions(self):
        rows = [
            _diagnostic_row("a.txt", "S1", "Target", "Target", 1, 0.1),
            _diagnostic_row("b.txt", "S1", "Target", "Wrong", 2, -0.1),
        ]
        result = analysis.aggregate_sample_diagnostics(rows)[0]
        self.assertEqual(result["query_spectrum_count"], 2)
        self.assertEqual(result["top1_hit_fraction"], 0.5)
        self.assertTrue(result["partial_top1_hit_across_measurements"])
        self.assertFalse(result["top1_prediction_consistent"])
        self.assertEqual(result["correct_label_rank_median"], 1.5)

    def test_gallery_label_size_counts_files_and_distinct_sample_ids(self):
        records = [
            _record("a1.txt", "S1", "A"),
            _record("a2.txt", "S1", "A"),
            _record("a3.txt", "S2", "A"),
            _record("b1.txt", "S3", "B"),
        ]
        sizes = analysis.gallery_label_sizes(records)
        self.assertEqual(sizes[_target_key("A")]["file_count"], 3)
        self.assertEqual(sizes[_target_key("A")]["sample_id_count"], 2)

    def test_missing_excitation_wavelength_is_explicit_unknown(self):
        for value in (None, "", "Unknown", "N/A"):
            self.assertEqual(analysis.normalize_excitation_wavelength(value), "Unknown")
        self.assertEqual(analysis.normalize_excitation_wavelength("532"), "532")

    def test_final_test_vector_is_not_passed_to_similarity(self):
        gallery_records = (
            _record("g-a.txt", "GA", "A"),
            _record("g-b.txt", "GB", "B"),
        )
        gallery_matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        library = analysis.exp10.ReferenceLibrary(
            grid=np.asarray([0.0, 1.0]),
            all_success_matrix=gallery_matrix,
            all_success_records=gallery_records,
            reference_matrix=gallery_matrix,
            reference_records=gallery_records,
            path_to_success_row={"g-a.txt": 0, "g-b.txt": 1},
        )
        manifest = [{
            "relative_path": "dev.txt",
            "cache_row_index": "2",
            "sample_id": "DEV",
            "sample_id_source": "filename_header_match",
            "measurement_stem": "dev",
            "reference_label": "A",
            "reference_label_group_display": "A",
            "reference_label_group_key": _target_key("A"),
            "excitation_wavelength_nm": "532",
            "excitation_wavelength_source": "header",
        }]
        baseline_row = [{
            "query_relative_path": "dev.txt",
            "query_cache_row_index": "2",
            "query_sample_id": "DEV",
            "query_measurement_stem": "dev",
            "query_reference_label_group_key": _target_key("A"),
            "reference_label_hit_at_1": "1",
            "reference_label_hit_at_3": "1",
            "reference_label_hit_at_5": "1",
            "reference_label_matching_rank": "1",
            "reference_label_top1_reference_label": "A",
            "reference_label_top1_cosine_similarity": "1.0",
        }]
        original = analysis.exp10.search_topk
        observed: list[tuple[int, str]] = []

        def checked_search(library_arg, query_arg, *args, **kwargs):
            observed.append((len(library_arg.reference_records), query_arg.relative_path))
            self.assertNotIn("final.txt", [row["relative_path"] for row in library_arg.reference_records])
            return original(library_arg, query_arg, *args, **kwargs)

        with mock.patch.object(analysis.exp10, "search_topk", side_effect=checked_search):
            rows = analysis.recompute_development_diagnostics(
                library, manifest, np.asarray([[1.0, 0.0]]), baseline_row
            )
        self.assertEqual(observed, [(2, "dev.txt")])
        self.assertEqual(len(rows), 1)

    def test_baseline_identity_mismatch_is_rejected(self):
        library_records = (_record("g-a.txt", "GA", "A"), _record("g-b.txt", "GB", "B"))
        matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        library = analysis.exp10.ReferenceLibrary(
            grid=np.asarray([0.0, 1.0]), all_success_matrix=matrix,
            all_success_records=library_records, reference_matrix=matrix,
            reference_records=library_records, path_to_success_row={"g-a.txt": 0, "g-b.txt": 1},
        )
        manifest = [{
            "relative_path": "dev.txt", "cache_row_index": "2", "sample_id": "DEV",
            "sample_id_source": "filename_header_match", "measurement_stem": "dev",
            "reference_label": "A", "reference_label_group_display": "A",
            "reference_label_group_key": _target_key("A"), "excitation_wavelength_nm": "532",
        }]
        existing = [{
            "query_relative_path": "dev.txt", "query_cache_row_index": "2",
            "query_sample_id": "WRONG", "query_measurement_stem": "dev",
            "query_reference_label_group_key": _target_key("A"),
        }]
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            analysis.recompute_development_diagnostics(
                library, manifest, np.asarray([[1.0, 0.0]]), existing
            )

    def test_missing_baseline_file_fails_explicitly(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            with self.assertRaisesRegex(ValueError, "does not exist"):
                analysis.load_and_validate_baseline_results(Path(temporary))

    def test_source_hash_mismatch_fails_explicitly(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            snapshot_names = (
                "evaluation_group_manifest.csv",
                "reference_label_sample_distribution.csv",
                "evaluation_group_audit.json",
                "EVALUATION_PROTOCOL.md",
            )
            for name in snapshot_names:
                (root / name).write_bytes(name.encode("utf-8"))
            cache = root / "cache.npz"
            index = root / "index.csv"
            cache.write_bytes(b"cache")
            index.write_bytes(b"index")

            def locked(path: Path) -> dict[str, object]:
                return {
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }

            lock_data = {
                "evaluation_snapshot": {
                    name: locked(root / name) for name in snapshot_names
                },
                "source_data": {
                    "experiment_9_3_npz": locked(cache),
                    "experiment_9_3_index": locked(index),
                },
                "frozen_evaluation": {},
            }
            lock_data["source_data"]["experiment_9_3_npz"]["sha256"] = "0" * 64
            lock = root / "lock.json"
            lock.write_text(json.dumps(lock_data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                analysis.baseline.validate_source_lock(
                    lock, root, cache, index
                )

    def test_outputs_use_utf8_sig_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            output = Path(temporary) / "output"
            paths = analysis.write_outputs(
                output, [], [], [], [], {"ok": True}, "# report\n"
            )
            self.assertTrue(all(path.exists() for path in paths))
            self.assertTrue(paths[0].read_bytes().startswith(b"\xef\xbb\xbf"))
            with self.assertRaises(FileExistsError):
                analysis.write_outputs(
                    output, [], [], [], [], {"ok": True}, "# report\n"
                )


if __name__ == "__main__":
    unittest.main()
