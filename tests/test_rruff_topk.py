from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "10_search_rruff_topk.py"
SPEC = importlib.util.spec_from_file_location("rruff_topk", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
topk = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = topk
SPEC.loader.exec_module(topk)


def _record(
    relative_path: str,
    sample_id: str,
    *,
    sample_source: str = "header_and_filename_agree",
    stem: str | None = None,
    reference: str = "True",
    label: str = "Reference",
) -> dict[str, str]:
    return {
        "relative_path": relative_path,
        "file_name": Path(relative_path).name,
        "measurement_stem": stem or Path(relative_path).stem,
        "sample_id": sample_id,
        "sample_id_source": sample_source,
        "filename_sample_id": sample_id,
        "header_rruff_id": sample_id,
        "reference_label": label,
        "reference_label_status": "candidate_reference_label_not_independently_validated",
        "reference_candidate": reference,
        "excitation_wavelength_nm": "532",
        "excitation_wavelength_source": "header_and_filename_agree",
        "filename_suspected_excitation_nm": "532",
        "header_raman_wavelength": "532",
        "requires_review": "False",
        "review_reasons": "",
        "audit_metadata_warnings": "",
        "has_missing_metadata": "False",
        "has_conflict": "False",
        "metadata_status": "complete_no_flag",
        "metadata_flags": "",
        "processing_status": "success",
        "failure_code": "",
        "cache_row_index": "0",
    }


def _library(matrix: np.ndarray, records: list[dict[str, str]]) -> topk.ReferenceLibrary:
    grid = np.arange(matrix.shape[1], dtype=float)
    return topk.ReferenceLibrary(
        grid=grid,
        all_success_matrix=matrix,
        all_success_records=tuple(records),
        reference_matrix=matrix,
        reference_records=tuple(records),
        path_to_success_row={row["relative_path"]: i for i, row in enumerate(records)},
    )


def _query(vector: np.ndarray, path: str = "query.txt", sample_id: str = "RQ"):
    return topk.QuerySpectrum(
        vector=np.asarray(vector, dtype=float),
        mode="test",
        relative_path=path,
        sample_id=sample_id,
        sample_id_source="header_and_filename_agree",
        measurement_stem="query_stem",
    )


def _write_index(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted(topk.REQUIRED_INDEX_COLUMNS)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class RruffTopKTests(unittest.TestCase):
    def test_identical_orthogonal_and_negative_cosine(self):
        matrix = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        similarities = topk.compute_cosine_similarities(matrix, [1.0, 0.0])
        np.testing.assert_allclose(similarities, [1.0, 0.0, -1.0])

    def test_zero_norm_and_nan_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "zero or invalid norm"):
            topk.compute_cosine_similarities(np.asarray([[1.0, 0.0]]), [0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            topk.compute_cosine_similarities(
                np.asarray([[1.0, np.nan]]), [1.0, 0.0]
            )
        with self.assertRaisesRegex(ValueError, "zero-norm"):
            topk.compute_cosine_similarities(
                np.asarray([[0.0, 0.0]]), [1.0, 0.0]
            )

    def test_cosine_roundoff_boundary_is_clipped_but_large_error_rejected(self):
        corrected = topk.correct_cosine_boundaries(
            np.asarray([1.0 + 5e-13, -1.0 - 5e-13, 0.25])
        )
        np.testing.assert_array_equal(corrected, [1.0, -1.0, 0.25])
        with self.assertRaisesRegex(ValueError, "materially outside"):
            topk.correct_cosine_boundaries(np.asarray([1.0 + 1e-6]))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            topk.correct_cosine_boundaries(np.asarray([np.nan]))

    def test_tied_similarity_uses_relative_path_order(self):
        records = [_record("z.txt", "R2"), _record("a.txt", "R1")]
        library = _library(np.asarray([[1.0, 0.0], [1.0, 0.0]]), records)
        hits = topk.search_topk(
            library,
            _query([1.0, 0.0]),
            10,
            result_level="file",
            exclude_self=False,
            exclude_same_sample=False,
        )
        self.assertEqual(
            [hit["record"]["relative_path"] for hit in hits],
            ["a.txt", "z.txt"],
        )

    def test_reference_label_groups_multiple_files_by_maximum_score(self):
        records = [
            _record("a_lower.txt", "R1", label="Quartz"),
            _record("a_best.txt", "R2", label=" quartz  "),
            _record("b.txt", "R3", label="Calcite"),
        ]
        library = _library(
            np.asarray([[0.8, 0.2], [1.0, 0.0], [0.6, 0.4]]), records
        )
        hits = topk.search_topk(
            library,
            _query([1.0, 0.0]),
            5,
            result_level="reference_label",
            exclude_self=False,
            exclude_same_sample=False,
        )
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0]["reference_label_group_display"], "quartz")
        self.assertEqual(hits[0]["record"]["relative_path"], "a_best.txt")
        self.assertEqual(hits[0]["label_reference_file_count"], 2)
        self.assertEqual(hits[0]["label_distinguishable_sample_id_count"], 2)
        self.assertEqual(hits[0]["label_score_definition"], "maximum_valid_reference_cosine")

    def test_label_aggregation_uses_full_library_before_topk_truncation(self):
        records = [
            _record("a1.txt", "R1", label="Label A"),
            _record("a2.txt", "R2", label="Label A"),
            _record("b1.txt", "R3", label="Label B"),
        ]
        library = _library(
            np.asarray([[1.0, 0.0], [0.99, 0.01], [0.8, 0.2]]), records
        )
        file_top2 = topk.search_topk(
            library,
            _query([1.0, 0.0]),
            2,
            result_level="file",
            exclude_self=False,
            exclude_same_sample=False,
        )
        label_top2 = topk.search_topk(
            library,
            _query([1.0, 0.0]),
            2,
            result_level="reference_label",
            exclude_self=False,
            exclude_same_sample=False,
        )
        self.assertEqual(
            [hit["record"]["reference_label"] for hit in file_top2],
            ["Label A", "Label A"],
        )
        self.assertEqual(
            [hit["reference_label_group_display"] for hit in label_top2],
            ["Label A", "Label B"],
        )

    def test_label_ties_are_stable_and_missing_labels_remain_separate(self):
        records = [
            _record("z.txt", "R1", label="Zulu"),
            _record("a.txt", "R2", label="Alpha"),
            _record("missing_a.txt", "R3", label="Unknown"),
            _record("missing_b.txt", "R4", label=""),
        ]
        library = _library(np.tile([1.0, 0.0], (4, 1)), records)
        hits = topk.search_topk(
            library,
            _query([1.0, 0.0]),
            10,
            result_level="reference_label",
            exclude_self=False,
            exclude_same_sample=False,
        )
        self.assertEqual(len(hits), 4)
        self.assertEqual(hits[0]["reference_label_group_display"], "Alpha")
        missing_hits = [
            hit
            for hit in hits
            if hit["reference_label_grouping_basis"]
            == "relative_path_fallback_for_missing_reference_label"
        ]
        self.assertEqual(len(missing_hits), 2)
        self.assertNotEqual(
            missing_hits[0]["reference_label_group_key"],
            missing_hits[1]["reference_label_group_key"],
        )

    def test_self_exclusion(self):
        records = [_record("query.txt", "RQ"), _record("other.txt", "R2")]
        library = _library(np.asarray([[1.0, 0.0], [0.9, 0.1]]), records)
        hits = topk.search_topk(
            library,
            _query([1.0, 0.0]),
            5,
            result_level="file",
            exclude_self=True,
            exclude_same_sample=False,
        )
        self.assertEqual([hit["record"]["relative_path"] for hit in hits], ["other.txt"])

    def test_known_sample_id_exclusion_and_deduplication(self):
        records = [
            _record("same_a.txt", "RQ"),
            _record("same_b.txt", "RQ"),
            _record("other.txt", "R2"),
        ]
        library = _library(
            np.asarray([[1.0, 0.0], [0.99, 0.01], [0.8, 0.2]]),
            records,
        )
        excluded = topk.search_topk(
            library,
            _query([1.0, 0.0]),
            10,
            result_level="file",
            exclude_self=False,
            exclude_same_sample=True,
        )
        self.assertEqual(
            [hit["record"]["relative_path"] for hit in excluded], ["other.txt"]
        )
        deduplicated = topk.search_topk(
            library,
            _query([1.0, 0.0], sample_id="R9"),
            10,
            result_level="sample_id",
            exclude_self=False,
            exclude_same_sample=False,
        )
        self.assertEqual(len(deduplicated), 2)
        self.assertEqual(deduplicated[0]["record"]["relative_path"], "same_a.txt")

    def test_label_summary_applies_same_sample_exclusion_before_counts(self):
        records = [
            _record("same_a.txt", "RQ", label="Shared Label"),
            _record("same_b.txt", "RQ", label="Shared Label"),
            _record("other.txt", "R2", label="Shared Label"),
        ]
        library = _library(
            np.asarray([[1.0, 0.0], [0.99, 0.01], [0.8, 0.2]]), records
        )
        hits = topk.search_topk(
            library,
            _query([1.0, 0.0]),
            5,
            result_level="reference_label",
            exclude_self=False,
            exclude_same_sample=True,
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["record"]["relative_path"], "other.txt")
        self.assertEqual(hits[0]["label_reference_file_count"], 1)
        self.assertEqual(hits[0]["label_distinguishable_sample_id_count"], 1)

    def test_unknown_ids_fall_back_to_measurement_stem_not_one_unknown_group(self):
        records = [
            _record(
                "unknown_a.txt",
                "Unknown",
                sample_source="missing",
                stem="stem_a",
            ),
            _record(
                "unknown_b.txt",
                "Unknown",
                sample_source="missing",
                stem="stem_b",
            ),
            _record(
                "unknown_a_repeat.txt",
                "Unknown",
                sample_source="filename_header_conflict_unresolved",
                stem="stem_a",
            ),
        ]
        library = _library(
            np.asarray([[1.0, 0.0], [0.9, 0.1], [0.95, 0.05]]), records
        )
        hits = topk.search_topk(
            library,
            _query([1.0, 0.0]),
            10,
            result_level="sample_id",
            exclude_self=False,
            exclude_same_sample=False,
        )
        self.assertEqual(len(hits), 2)
        self.assertEqual(
            {hit["dedupe_identity_basis"] for hit in hits},
            {"measurement_stem_fallback_for_unknown_sample_id"},
        )

    def test_label_sample_counts_use_unknown_identity_fallback(self):
        records = [
            _record("a.txt", "Unknown", sample_source="missing", stem="stem_a", label="L"),
            _record("a_repeat.txt", "Unknown", sample_source="missing", stem="stem_a", label="L"),
            _record("b.txt", "Unknown", sample_source="missing", stem="stem_b", label="L"),
        ]
        library = _library(
            np.asarray([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]), records
        )
        hits = topk.search_topk(
            library,
            _query([1.0, 0.0]),
            5,
            result_level="reference_label",
            exclude_self=False,
            exclude_same_sample=False,
        )
        self.assertEqual(hits[0]["label_reference_file_count"], 3)
        self.assertEqual(hits[0]["label_distinguishable_sample_id_count"], 2)

    def test_scenario_output_preserves_three_level_evidence_mapping(self):
        records = [
            _record("a1.txt", "R1", label="A"),
            _record("a2.txt", "R1", label="A"),
            _record("b.txt", "R2", label="B"),
        ]
        library = _library(
            np.asarray([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]), records
        )
        rows, summary = topk._scenario_search(
            library,
            _query([1.0, 0.0], path="external.txt", sample_id="RX"),
            5,
            "three_levels",
            False,
            False,
        )
        levels = {row["result_level"] for row in rows}
        self.assertEqual(levels, {"file", "sample_id", "reference_label"})
        label_rows = [row for row in rows if row["result_level"] == "reference_label"]
        self.assertEqual(label_rows[0]["candidate_relative_path"], "a1.txt")
        self.assertEqual(label_rows[0]["label_reference_file_count"], 2)
        self.assertTrue(summary["reference_label_top5_sort_valid"])

    def test_k_larger_than_candidates_returns_all_candidates(self):
        records = [_record("one.txt", "R1"), _record("two.txt", "R2")]
        library = _library(np.eye(2), records)
        hits = topk.search_topk(
            library,
            _query([1.0, 0.0]),
            99,
            result_level="file",
            exclude_self=False,
            exclude_same_sample=False,
        )
        self.assertEqual(len(hits), 2)

    def test_cache_index_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            cache_path = root / "cache.npz"
            index_path = root / "index.csv"
            grid = topk.exp09.make_trial_grid()
            np.savez_compressed(
                cache_path,
                wavenumber_grid_cm_1=grid,
                normalized_spectra=np.ones((1, len(grid))),
                relative_paths=np.asarray(["cache.txt"]),
                measurement_stems=np.asarray(["stem"]),
            )
            row = _record("index.txt", "R1")
            row["measurement_stem"] = "stem"
            _write_index(index_path, [row])
            with self.assertRaisesRegex(ValueError, "relative_path identity"):
                topk.load_reference_library(cache_path, index_path)

    def test_external_raw_preprocessing_matches_experiment_9_3(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            raw_path = root / "query.txt"
            x = np.arange(290.0, 861.0, 1.0)
            y = 100.0 + 700.0 * np.exp(-0.5 * ((x - 575.0) / 15.0) ** 2)
            raw_path.write_text(
                "\n".join(f"{xv},{yv}" for xv, yv in zip(x, y)),
                encoding="utf-8",
            )
            query = topk.query_from_external_raw(raw_path)
            content = raw_path.read_bytes()
            _raw, _corrected, normalized, _range = topk.exp09._run_v1_preprocessing(
                content
            )
            expected = topk.exp09.interpolation.linear_interpolate_no_extrapolation(
                np.asarray(normalized.spectral_axis),
                np.asarray(normalized.spectral_data),
                topk.exp09.make_trial_grid(),
            )
        np.testing.assert_allclose(query.vector, expected)

    def test_external_raw_without_full_band_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            raw_path = Path(temporary) / "short.txt"
            x = np.arange(400.0, 901.0, 1.0)
            y = 100.0 + 500.0 * np.exp(-0.5 * ((x - 575.0) / 15.0) ** 2)
            raw_path.write_text(
                "\n".join(f"{xv},{yv}" for xv, yv in zip(x, y)),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "without extrapolation"):
                topk.query_from_external_raw(raw_path)

    def test_invalid_external_raw_reports_preprocessing_stage(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            raw_path = Path(temporary) / "invalid.txt"
            raw_path.write_text("not,numeric\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                r"failed at parse \(upload_parse_error\)",
            ):
                topk.query_from_external_raw(raw_path)

    def test_outputs_refuse_overwrite(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            output_dir = Path(temporary) / "output"
            topk.write_outputs(output_dir, [], {"status": "test"})
            with self.assertRaises(FileExistsError):
                topk.write_outputs(output_dir, [], {"status": "test"})


if __name__ == "__main__":
    unittest.main()
