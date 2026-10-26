from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import ramanspy as rp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "09_batch_preprocess_rruff.py"
SPEC = importlib.util.spec_from_file_location("batch_preprocessing", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


def _manifest_row(relative_path: str, *, candidate: str = "True", raw: str = "True"):
    processed = "False" if raw == "True" else "True"
    return {
        "relative_path": relative_path,
        "file_name": Path(relative_path).name,
        "measurement_stem": Path(relative_path).stem + "_measurement",
        "file_type": "RAW" if raw == "True" else "Processed",
        "is_raw": raw,
        "is_processed": processed,
        "preprocessing_candidate": candidate,
        "reference_candidate": "True",
        "requires_review": "False",
        "review_reasons": "",
        "reference_label": "SyntheticReference",
        "reference_label_status": "candidate_reference_label_not_independently_validated",
        "filename_sample_id": "R000001",
        "header_rruff_id": "R000001",
        "filename_suspected_excitation_nm": "532",
        "header_raman_wavelength": "532",
        "audit_metadata_warnings": "",
        "has_missing_metadata": "False",
        "has_conflict": "False",
        "metadata_status": "complete_no_flag",
        "metadata_flags": "",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _spectrum_bytes(offset: float = 0.0) -> bytes:
    x = np.arange(290.0, 861.0, 1.0)
    peak = 800.0 * np.exp(-0.5 * ((x - (575.0 + offset)) / 15.0) ** 2)
    baseline = 100.0 + 0.02 * (x - 290.0)
    lines = ["##NAMES=Synthetic", "##RRUFFID=R000001"]
    lines.extend(f"{xv:.6f},{yv:.12f}" for xv, yv in zip(x, peak + baseline))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _out_of_band_spectrum_bytes() -> bytes:
    x = np.arange(400.0, 901.0, 1.0)
    y = 500.0 * np.exp(-0.5 * ((x - 575.0) / 15.0) ** 2) + 50.0
    lines = [f"{xv:.6f},{yv:.12f}" for xv, yv in zip(x, y)]
    return ("\n".join(lines) + "\n").encode("utf-8")


class BatchPreprocessingTests(unittest.TestCase):
    def test_trial_grid_has_exact_1101_points(self):
        grid = batch.make_trial_grid()
        self.assertEqual(grid.shape, (1101,))
        self.assertEqual(grid[0], 300.0)
        self.assertEqual(grid[-1], 850.0)
        np.testing.assert_array_equal(np.diff(grid), np.full(1100, 0.5))

    def test_manifest_admission_is_reused_and_processed_is_excluded(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            rows = [
                _manifest_row("admitted.txt"),
                _manifest_row("rejected.txt", candidate="False"),
                _manifest_row("processed.txt", candidate="False", raw="False"),
            ]
            _write_csv(manifest, rows)
            selected, total = batch.load_candidate_records(manifest, "full")
        self.assertEqual(total, 3)
        self.assertEqual([row["relative_path"] for row in selected], ["admitted.txt"])

    def test_inconsistent_processed_candidate_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            manifest = Path(temporary) / "manifest.csv"
            _write_csv(
                manifest,
                [_manifest_row("processed.txt", candidate="True", raw="False")],
            )
            with self.assertRaisesRegex(ValueError, "inconsistent admitted record"):
                batch.load_candidate_records(manifest, "full")

    def test_trial_paths_must_be_unique_admitted_candidates(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            trial = root / "trial.csv"
            _write_csv(manifest, [_manifest_row("one.txt")])
            _write_csv(trial, [{"relative_path": "missing.txt"}])
            with self.assertRaisesRegex(ValueError, "absent from admitted"):
                batch.load_candidate_records(manifest, "trial", trial)

    def test_preprocessing_matches_direct_v1_calls_and_interpolation(self):
        content = _spectrum_bytes()
        raw, corrected, normalized, dynamic_range = batch._run_v1_preprocessing(content)
        direct_raw = batch.parse_uploaded_spectrum(content)
        direct_denoised = batch.denoise_spectrum(
            direct_raw,
            window_length=21,
            polyorder=3,
        )
        direct_corrected = batch.baseline_correct_spectrum(
            direct_denoised,
            lam=1e6,
            p=0.01,
        )
        direct_normalized = batch.normalize_spectrum(direct_corrected, method="max")
        np.testing.assert_allclose(corrected.spectral_data, direct_corrected.spectral_data)
        np.testing.assert_allclose(normalized.spectral_data, direct_normalized.spectral_data)
        self.assertGreater(dynamic_range, 0)
        self.assertEqual(len(raw.spectral_axis), len(direct_raw.spectral_axis))

    def test_missing_file_is_retained_as_failure(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            record, values = batch.process_candidate(
                root,
                _manifest_row("missing.txt"),
                1,
                batch.make_trial_grid(),
            )
        self.assertIsNone(values)
        self.assertEqual(record["processing_status"], "failed")
        self.assertEqual(record["failure_code"], "file_missing")

    def test_invalid_dynamic_range_is_recorded(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            source = root / "constant.txt"
            source.write_bytes(_spectrum_bytes())
            axis = np.arange(290.0, 861.0, 1.0)
            zero_spectrum = rp.Spectrum(np.zeros_like(axis), axis)
            with mock.patch.object(
                batch,
                "baseline_correct_spectrum",
                return_value=zero_spectrum,
            ):
                record, values = batch.process_candidate(
                    root,
                    _manifest_row("constant.txt"),
                    1,
                    batch.make_trial_grid(),
                )
        self.assertIsNone(values)
        self.assertEqual(record["failure_code"], "invalid_corrected_dynamic_range")

    def test_incomplete_target_band_is_recorded_without_extrapolation(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            (root / "short_band.txt").write_bytes(_out_of_band_spectrum_bytes())
            record, values = batch.process_candidate(
                root,
                _manifest_row("short_band.txt"),
                1,
                batch.make_trial_grid(),
            )
        self.assertIsNone(values)
        self.assertEqual(record["failure_stage"], "interpolation_validation")
        self.assertEqual(record["failure_code"], "target_band_incomplete")

    def test_mixed_success_and_missing_file_preserve_all_index_rows(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            (root / "good.txt").write_bytes(_spectrum_bytes())
            rows = [_manifest_row("good.txt"), _manifest_row("missing.txt")]
            records, _grid, matrix = batch.process_records(
                root, rows, progress_every=0
            )
        self.assertEqual(len(records), 2)
        self.assertEqual(matrix.shape, (1, 1101))
        self.assertEqual(records[0]["processing_status"], "success")
        self.assertEqual(records[0]["cache_row_index"], 0)
        self.assertEqual(records[1]["processing_status"], "failed")
        self.assertEqual(records[1]["failure_code"], "file_missing")
        self.assertEqual(records[1]["cache_row_index"], "")

    def test_cache_round_trip_preserves_metadata_alignment(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            (root / "a.txt").write_bytes(_spectrum_bytes())
            (root / "b.txt").write_bytes(_spectrum_bytes(offset=3.0))
            rows = [_manifest_row("a.txt"), _manifest_row("b.txt")]
            rows[1]["measurement_stem"] = "second_measurement"
            records, grid, matrix = batch.process_records(root, rows, progress_every=0)
            summary = batch.build_summary(
                root / "manifest.csv",
                None,
                root,
                root / "output",
                "full",
                2,
                records,
                grid,
                matrix,
            )
            paths = batch.write_outputs(root / "output", records, grid, matrix, summary)
            validation = batch.validate_cache(paths[0], records)
            with np.load(paths[0], allow_pickle=False) as cache:
                np.testing.assert_array_equal(cache["relative_paths"], ["a.txt", "b.txt"])
                np.testing.assert_allclose(cache["normalized_spectra"], matrix)
        self.assertEqual(validation["matrix_shape"], [2, 1101])
        self.assertTrue(validation["identity_mapping_verified"])
        self.assertEqual(records[0]["cache_row_index"], 0)
        self.assertEqual(records[1]["cache_row_index"], 1)

    def test_repeated_processing_is_numerically_reproducible(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            (root / "one.txt").write_bytes(_spectrum_bytes())
            rows = [_manifest_row("one.txt")]
            records_a, grid_a, matrix_a = batch.process_records(
                root, rows, progress_every=0
            )
            records_b, grid_b, matrix_b = batch.process_records(
                root, rows, progress_every=0
            )
        self.assertEqual(records_a, records_b)
        np.testing.assert_array_equal(grid_a, grid_b)
        np.testing.assert_array_equal(matrix_a, matrix_b)

    def test_outputs_refuse_overwrite(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            output_dir = Path(temporary) / "output"
            grid = batch.make_trial_grid()
            matrix = np.empty((0, len(grid)))
            summary = {"status": "test"}
            batch.write_outputs(output_dir, [], grid, matrix, summary)
            with self.assertRaises(FileExistsError):
                batch.write_outputs(output_dir, [], grid, matrix, summary)


if __name__ == "__main__":
    unittest.main()
