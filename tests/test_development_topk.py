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
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "11_evaluate_development_topk.py"
spec = importlib.util.spec_from_file_location("development_topk_test_module", SCRIPT_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load experiment 11.2 module")
evaluation = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluation
spec.loader.exec_module(evaluation)


def _index_record(
    path: str,
    sample_id: str,
    label: str,
    row_index: int,
    stem: str | None = None,
) -> dict[str, str]:
    record = {field: "" for field in evaluation.exp10.REQUIRED_INDEX_COLUMNS}
    record.update(
        {
            "relative_path": path,
            "file_name": Path(path).name,
            "measurement_stem": stem or Path(path).stem,
            "sample_id": sample_id,
            "sample_id_source": "filename_header_match",
            "reference_label": label,
            "reference_label_status": "available",
            "reference_candidate": "True",
            "excitation_wavelength_nm": "532",
            "requires_review": "False",
            "has_missing_metadata": "False",
            "has_conflict": "False",
            "processing_status": "success",
            "cache_row_index": str(row_index),
        }
    )
    return record


def _manifest_record(
    index: dict[str, str], group: str, label_key: str | None = None
) -> dict[str, str]:
    display, _basis, calculated_key = evaluation.exp10._reference_label_identity(index)
    return {
        "relative_path": index["relative_path"],
        "cache_row_index": index["cache_row_index"],
        "measurement_stem": index["measurement_stem"],
        "sample_id": index["sample_id"],
        "sample_id_source": index["sample_id_source"],
        "reference_label": index["reference_label"],
        "reference_label_group_display": display,
        "reference_label_group_key": label_key or calculated_key,
        "excitation_wavelength_nm": index["excitation_wavelength_nm"],
        "evaluation_group": group,
        "known_sample_id_usable": "True",
        "sample_id_metadata_conflict": "False",
        "strict_closed_set_eligible": "True",
    }


def _library() -> object:
    records = (
        _index_record("a1.txt", "A-G1", "Label A", 0),
        _index_record("a2.txt", "A-G2", "Label A", 1),
        _index_record("b.txt", "B-G1", "Label B", 2),
        _index_record("c.txt", "C-G1", "Label C", 3),
    )
    matrix = np.asarray(
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    return evaluation.exp10.ReferenceLibrary(
        grid=np.asarray([0.0, 1.0, 2.0]),
        all_success_matrix=matrix,
        all_success_records=records,
        reference_matrix=matrix,
        reference_records=records,
        path_to_success_row={record["relative_path"]: index for index, record in enumerate(records)},
    )


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class DevelopmentTopKTests(unittest.TestCase):
    def test_real_frozen_snapshot_has_exact_required_counts(self):
        lock = evaluation.validate_source_lock(
            evaluation.DEFAULT_LOCK,
            evaluation.DEFAULT_SNAPSHOT_DIR,
            evaluation.DEFAULT_CACHE,
            evaluation.DEFAULT_INDEX,
        )
        rows, preflight = evaluation.validate_frozen_manifest(
            evaluation.DEFAULT_MANIFEST, lock
        )
        self.assertEqual(preflight["gallery_record_count"], 724)
        self.assertEqual(preflight["development_record_count"], 118)
        self.assertEqual(preflight["final_test_record_count"], 129)
        self.assertEqual(preflight["development_sample_id_count"], 60)
        self.assertEqual(len(rows), 5529)

    def test_three_level_hit_definitions_use_frozen_label_key(self):
        library = _library()
        query_index = _index_record("query.txt", "A-Q", "Label A", 4)
        manifest = [_manifest_record(query_index, evaluation.exp11.GROUP_DEVELOPMENT)]
        rows, rankings = evaluation.evaluate_development_queries(
            library, manifest, np.asarray([[1.0, 0.0, 0.0]])
        )
        self.assertEqual(len(rows), 1)
        for level in evaluation.LEVELS:
            self.assertEqual(rows[0][f"{level}_hit_at_1"], 1)
            self.assertEqual(rows[0][f"{level}_matching_rank"], 1)
            self.assertIn(level, rankings["query.txt"])

    def test_file_sample_and_label_levels_apply_distinct_ranking_units(self):
        records = (
            _index_record("wrong-1.txt", "W-G1", "Wrong", 0),
            _index_record("wrong-2.txt", "W-G1", "Wrong", 1),
            _index_record("target.txt", "A-G1", "Target", 2),
        )
        matrix = np.asarray(
            [[1.0, 0.0], [0.99, 0.01], [0.8, 0.2]], dtype=np.float64
        )
        library = evaluation.exp10.ReferenceLibrary(
            grid=np.asarray([0.0, 1.0]),
            all_success_matrix=matrix,
            all_success_records=records,
            reference_matrix=matrix,
            reference_records=records,
            path_to_success_row={record["relative_path"]: index for index, record in enumerate(records)},
        )
        query_index = _index_record("query.txt", "A-Q", "Target", 3)
        rows, _rankings = evaluation.evaluate_development_queries(
            library,
            [_manifest_record(query_index, evaluation.exp11.GROUP_DEVELOPMENT)],
            np.asarray([[1.0, 0.0]]),
        )
        self.assertEqual(rows[0]["file_matching_rank"], 3)
        self.assertEqual(rows[0]["sample_id_matching_rank"], 2)
        self.assertEqual(rows[0]["reference_label_matching_rank"], 2)

    def test_same_sample_multiple_measurements_are_averaged_before_equal_weighting(self):
        base = {
            "query_reference_label": "A",
            "query_reference_label_group_key": "reference_label:a",
        }
        rows = []
        for sample, hits in (("S1", (1, 0)), ("S2", (1,))):
            for number, hit in enumerate(hits):
                row = dict(base)
                row.update({"query_sample_id": sample, "query_relative_path": f"{sample}-{number}"})
                for level in evaluation.LEVELS:
                    for k in evaluation.TOP_K_VALUES:
                        row[f"{level}_hit_at_{k}"] = hit
                rows.append(row)
        aggregated = evaluation.aggregate_by_sample_id(rows)
        by_sample = {row["query_sample_id"]: row for row in aggregated}
        self.assertEqual(by_sample["S1"]["file_mean_hit_at_1"], 0.5)
        self.assertEqual(by_sample["S2"]["file_mean_hit_at_1"], 1.0)
        self.assertEqual(np.mean([row["file_mean_hit_at_1"] for row in aggregated]), 0.75)

    def test_tied_cosine_results_are_stably_sorted_by_relative_path(self):
        library = _library()
        query = evaluation.exp10.QuerySpectrum(
            vector=np.asarray([0.0, 0.0, 1.0]),
            mode="test",
            relative_path="query.txt",
        )
        first = evaluation.exp10.search_topk(
            library, query, 4, result_level="file", exclude_self=False, exclude_same_sample=False
        )
        second = evaluation.exp10.search_topk(
            library, query, 4, result_level="file", exclude_self=False, exclude_same_sample=False
        )
        paths = [hit["record"]["relative_path"] for hit in first]
        self.assertEqual(paths, sorted(paths, key=lambda value: (value.casefold(), value)))
        self.assertEqual(paths, [hit["record"]["relative_path"] for hit in second])

    def test_zero_norm_and_nonfinite_queries_are_rejected(self):
        library = _library()
        with self.assertRaisesRegex(ValueError, "zero or invalid norm"):
            evaluation.exp10.compute_cosine_similarities(
                library.reference_matrix, np.zeros(3)
            )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            evaluation.exp10.compute_cosine_similarities(
                library.reference_matrix, np.asarray([np.nan, 0.0, 1.0])
            )

    def test_prepare_data_selects_gallery_and_development_but_not_final(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            cache = root / "cache.npz"
            index_path = root / "index.csv"
            records = [
                _index_record("gallery.txt", "G1", "Label A", 0),
                _index_record("development.txt", "D1", "Label A", 1),
                _index_record("final.txt", "F1", "Label A", 2),
            ]
            grid = evaluation.exp10.exp09.make_trial_grid()
            matrix = np.vstack(
                [np.linspace(1.0 + index, 2.0 + index, len(grid)) for index in range(3)]
            )
            np.savez_compressed(
                cache,
                wavenumber_grid_cm_1=grid,
                normalized_spectra=matrix,
                relative_paths=np.asarray([row["relative_path"] for row in records]),
                measurement_stems=np.asarray([row["measurement_stem"] for row in records]),
            )
            _write_csv(index_path, records)
            manifest = [
                _manifest_record(records[0], evaluation.exp11.GROUP_GALLERY),
                _manifest_record(records[1], evaluation.exp11.GROUP_DEVELOPMENT),
                _manifest_record(records[2], evaluation.exp11.GROUP_FINAL),
            ]
            gallery, development, development_matrix = evaluation.prepare_frozen_data(
                cache, index_path, manifest
            )
            self.assertEqual([row["relative_path"] for row in gallery.reference_records], ["gallery.txt"])
            self.assertEqual([row["relative_path"] for row in development], ["development.txt"])
            self.assertTrue(np.array_equal(development_matrix, matrix[[1]]))
            self.assertNotIn("final.txt", [row["relative_path"] for row in gallery.reference_records])

    def test_manifest_index_identity_mismatch_is_rejected(self):
        index = _index_record("a.txt", "S1", "Label A", 0)
        manifest = _manifest_record(index, evaluation.exp11.GROUP_GALLERY)
        manifest["sample_id"] = "CHANGED"
        with self.assertRaisesRegex(ValueError, "sample_id"):
            evaluation._reconcile_manifest_record(manifest, index)

    def test_source_lock_detects_hash_change_before_evaluation(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            snapshot_names = (
                "evaluation_group_manifest.csv",
                "reference_label_sample_distribution.csv",
                "evaluation_group_audit.json",
                "EVALUATION_PROTOCOL.md",
            )
            for name in snapshot_names:
                (snapshot / name).write_bytes(name.encode("utf-8"))
            cache = root / "cache.npz"
            index = root / "index.csv"
            cache.write_bytes(b"cache")
            index.write_bytes(b"index")

            def spec(path: Path) -> dict[str, object]:
                return {
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }

            lock = {
                "evaluation_snapshot": {name: spec(snapshot / name) for name in snapshot_names},
                "source_data": {
                    "experiment_9_3_npz": spec(cache),
                    "experiment_9_3_index": spec(index),
                },
                "frozen_evaluation": {},
            }
            lock_path = root / "lock.json"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            evaluation.validate_source_lock(lock_path, snapshot, cache, index)
            cache.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                evaluation.validate_source_lock(lock_path, snapshot, cache, index)

    def test_frozen_manifest_rejects_changed_final_identity(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            records = [
                _manifest_record(_index_record("g.txt", "G1", "A", 0), evaluation.exp11.GROUP_GALLERY),
                _manifest_record(_index_record("d.txt", "D1", "A", 1), evaluation.exp11.GROUP_DEVELOPMENT),
                _manifest_record(_index_record("f.txt", "F1", "A", 2), evaluation.exp11.GROUP_FINAL),
            ]
            manifest = root / "manifest.csv"
            _write_csv(manifest, records)
            lock = {
                "frozen_evaluation": {
                    "manifest_sha256": evaluation.sha256_file(manifest),
                    "all_group_identity_sha256": evaluation.exp11._identity_hash(records),
                    "final_test_identity_sha256": "0" * 64,
                    "expected_counts": {
                        "reference_gallery_records": 1,
                        "reference_gallery_known_sample_ids": 1,
                        "development_query_records": 1,
                        "development_query_known_sample_ids": 1,
                        "final_test_query_records": 1,
                        "final_test_query_known_sample_ids": 1,
                        "strict_closed_set_reference_labels": 1,
                    },
                }
            }
            with self.assertRaisesRegex(ValueError, "final-test identity"):
                evaluation.validate_frozen_manifest(manifest, lock)

    def test_failed_preflight_does_not_create_output_directory(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            output = root / "output"
            with self.assertRaisesRegex(ValueError, "does not exist"):
                evaluation.run_evaluation(
                    cache_path=root / "missing.npz",
                    index_path=root / "missing.csv",
                    snapshot_dir=root / "snapshot",
                    lock_path=root / "missing-lock.json",
                    output_dir=output,
                )
            self.assertFalse(output.exists())

    def test_outputs_refuse_overwrite(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            output = Path(temporary) / "output"
            paths = evaluation.write_outputs(output, [], [], [], {"ok": True})
            self.assertTrue(all(path.exists() for path in paths))
            with self.assertRaises(FileExistsError):
                evaluation.write_outputs(output, [], [], [], {"ok": True})


if __name__ == "__main__":
    unittest.main()
