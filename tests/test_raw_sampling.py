"""Tests for real RAW Raman-shift sampling inspection."""

import csv
import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_DIR / "scripts" / "09_inspect_raw_sampling.py"
SPEC = importlib.util.spec_from_file_location("raw_sampling", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load RAW sampling inspector")
raw_sampling = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = raw_sampling
SPEC.loader.exec_module(raw_sampling)


def make_trial_row(relative_path: str, **overrides: str) -> dict[str, str]:
    row = {
        "selection_rank": "1",
        "relative_path": relative_path,
        "file_name": Path(relative_path).name,
        "sample_id": "R123456",
        "filename_sample_id": "R123456",
        "header_rruff_id": "R123456",
        "reference_label": "Testite",
        "header_raman_wavelength": "532",
        "wavenumber_min_cm-1": "100",
        "wavenumber_max_cm-1": "120",
        "point_count": "21",
    }
    row.update(overrides)
    return row


def make_audit_row(relative_path: str, **overrides: str) -> dict[str, str]:
    row = {
        "relative_path": relative_path,
        "file_name": Path(relative_path).name,
        "file_type": "RAW",
        "header_names": "Testite",
        "header_rruff_id": "R123456",
        "header_raman_wavelength": "532",
        "point_count": "21",
        "wavenumber_min": "100",
        "wavenumber_max": "120",
        "metadata_warnings": "",
    }
    row.update(overrides)
    return row


def spectrum_bytes(
    wavenumbers: list[float],
    intensities: list[float] | None = None,
) -> bytes:
    if intensities is None:
        intensities = [1000.0 + index for index in range(len(wavenumbers))]
    rows = ["##NAMES=Testite", "##RRUFFID=R123456"]
    rows.extend(
        f"{wavenumber},{intensity}"
        for wavenumber, intensity in zip(wavenumbers, intensities)
    )
    return ("\n".join(rows) + "\n").encode("utf-8")


def write_csv(path: Path, fields: set[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=sorted(fields))
        writer.writeheader()
        writer.writerows(rows)


class RawSamplingTests(unittest.TestCase):
    def test_even_spacing_statistics(self):
        axis = np.arange(100.0, 121.0)
        result, gaps = raw_sampling.inspect_arrays(
            axis,
            np.ones_like(axis),
            "1",
            "even.txt",
            "R123456",
        )

        self.assertTrue(result["wavenumber_strictly_increasing"])
        self.assertEqual(result["delta_min"], 1.0)
        self.assertEqual(result["delta_median"], 1.0)
        self.assertEqual(result["delta_p95"], 1.0)
        self.assertEqual(result["delta_max"], 1.0)
        self.assertEqual(gaps, [])

    def test_nonuniform_sampling_statistics(self):
        axis = np.array([100.0, 100.5, 101.5, 103.0, 105.0])
        result, _ = raw_sampling.inspect_arrays(
            axis,
            np.ones_like(axis),
            "1",
            "nonuniform.txt",
            "R123456",
        )

        self.assertEqual(result["delta_min"], 0.5)
        self.assertEqual(result["delta_median"], 1.25)
        self.assertEqual(result["delta_max"], 2.0)

    def test_single_large_internal_gap_is_detailed(self):
        axis = np.array([300.0, 301.0, 302.0, 312.0, 313.0, 314.0])
        result, gaps = raw_sampling.inspect_arrays(
            axis,
            np.ones_like(axis),
            "2",
            "gap.txt",
            "R123456",
        )

        self.assertEqual(result["gap_count_gt_3x_median"], 1)
        self.assertEqual(result["gap_count_gt_5x_median"], 1)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["left_wavenumber"], 302.0)
        self.assertEqual(gaps[0]["right_wavenumber"], 312.0)

    def test_duplicate_wavenumber_is_flagged(self):
        axis = np.array([100.0, 101.0, 101.0, 102.0])
        result, _ = raw_sampling.inspect_arrays(
            axis,
            np.ones_like(axis),
            "1",
            "duplicate.txt",
            "R123456",
        )

        self.assertTrue(result["has_duplicate_wavenumber"])
        self.assertFalse(result["wavenumber_strictly_increasing"])

    def test_nonfinite_values_are_flagged(self):
        axis = np.array([100.0, 101.0, 102.0])
        intensities = np.array([1.0, np.nan, 3.0])
        result, _ = raw_sampling.inspect_arrays(
            axis,
            intensities,
            "1",
            "nonfinite.txt",
            "R123456",
        )

        self.assertTrue(result["axis_values_finite"])
        self.assertFalse(result["intensity_values_finite"])
        self.assertFalse(result["all_values_finite"])

    def test_missing_file_is_retained_as_failure(self):
        with TemporaryDirectory(dir=PROJECT_DIR / "tests") as temporary_directory:
            root = Path(temporary_directory)
            joined = [
                (make_trial_row("missing.txt"), make_audit_row("missing.txt"))
            ]

            records, gaps = raw_sampling.inspect_trial_files(root, joined)

            self.assertEqual(len(records), 1)
            self.assertFalse(records[0]["read_success"])
            self.assertEqual(records[0]["error_code"], "file_not_found")
            self.assertEqual(gaps, [])

    def test_path_traversal_is_rejected(self):
        with TemporaryDirectory(dir=PROJECT_DIR / "tests") as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaises(raw_sampling.UnsafeRelativePathError):
                raw_sampling.resolve_safe_raw_path(root, "../outside.txt")

    def test_actual_values_are_compared_with_audit(self):
        record = {
            "actual_point_count": 21,
            "actual_wavenumber_min": 100.0,
            "actual_wavenumber_max": 120.0,
        }
        audit = make_audit_row(
            "sample.txt",
            point_count="22",
            wavenumber_min="99",
            wavenumber_max="121",
        )

        raw_sampling._compare_with_audit(record, audit)

        self.assertEqual(record["audit_consistency_status"], "mismatch")
        self.assertIn("point_count_mismatch", record["audit_mismatch_reasons"])
        self.assertIn("wavenumber_min_mismatch", record["audit_mismatch_reasons"])
        self.assertIn("wavenumber_max_mismatch", record["audit_mismatch_reasons"])

    def test_one_file_failure_does_not_stop_other_files(self):
        with TemporaryDirectory(dir=PROJECT_DIR / "tests") as temporary_directory:
            root = Path(temporary_directory)
            valid_path = root / "valid.txt"
            valid_path.write_bytes(
                spectrum_bytes([float(value) for value in range(100, 121)])
            )
            joined = [
                (make_trial_row("missing.txt"), make_audit_row("missing.txt")),
                (make_trial_row("valid.txt"), make_audit_row("valid.txt")),
            ]

            records, _ = raw_sampling.inspect_trial_files(root, joined)

            self.assertEqual(len(records), 2)
            self.assertFalse(records[0]["read_success"])
            self.assertTrue(records[1]["read_success"])
            self.assertEqual(records[1]["audit_consistency_status"], "match")

    def test_outputs_refuse_overwrite(self):
        with TemporaryDirectory(dir=PROJECT_DIR / "tests") as temporary_directory:
            root = Path(temporary_directory)
            raw_root = root / "raw"
            raw_root.mkdir()
            relative_path = "sample.txt"
            (raw_root / relative_path).write_bytes(
                spectrum_bytes([float(value) for value in range(100, 121)])
            )
            trial_path = root / "trial.csv"
            audit_path = root / "audit.csv"
            output_dir = root / "output"
            write_csv(
                trial_path,
                raw_sampling.REQUIRED_TRIAL_COLUMNS,
                [make_trial_row(relative_path)],
            )
            write_csv(
                audit_path,
                raw_sampling.REQUIRED_AUDIT_COLUMNS,
                [make_audit_row(relative_path)],
            )

            result = raw_sampling.run_inspection(
                raw_root,
                trial_path,
                audit_path,
                output_dir,
            )

            self.assertEqual(result[2]["read_success_count"], 1)
            for output_path in result[3]:
                self.assertTrue(output_path.read_bytes().startswith(b"\xef\xbb\xbf"))
            with self.assertRaises(FileExistsError):
                raw_sampling.run_inspection(
                    raw_root,
                    trial_path,
                    audit_path,
                    output_dir,
                )


if __name__ == "__main__":
    unittest.main()
