"""Tests for real and synthetic linear-interpolation validation."""

import csv
import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_DIR / "scripts" / "09_validate_interpolation.py"
SPEC = importlib.util.spec_from_file_location(
    "interpolation_validation", SCRIPT_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load interpolation validation module")
interpolation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = interpolation
SPEC.loader.exec_module(interpolation)


def make_trial_row(relative_path: str) -> dict[str, str]:
    return {
        "selection_rank": "1",
        "relative_path": relative_path,
        "file_name": Path(relative_path).name,
        "sample_id": "R123456",
        "reference_label": "Testite",
        "header_raman_wavelength": "532",
    }


def make_quality_row(relative_path: str, **overrides: str) -> dict[str, str]:
    row = {
        "selection_rank": "1",
        "relative_path": relative_path,
        "file_name": Path(relative_path).name,
        "sample_id": "R123456",
        "reference_label": "Testite",
        "header_raman_wavelength": "532",
        "metadata_requires_review": "False",
        "read_success": "True",
        "all_values_finite": "True",
        "wavenumber_strictly_increasing": "True",
        "band_300_850_endpoint_covered": "True",
        "delta_min": "1",
        "delta_median": "1",
        "delta_p95": "1",
        "delta_max": "1",
    }
    row.update(overrides)
    return row


def spectrum_bytes(axis: np.ndarray, intensity: np.ndarray) -> bytes:
    rows = ["##NAMES=Testite", "##RRUFFID=R123456"]
    rows.extend(f"{x},{y}" for x, y in zip(axis, intensity))
    return ("\n".join(rows) + "\n").encode("utf-8")


def write_csv(path: Path, fields: set[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=sorted(fields))
        writer.writeheader()
        writer.writerows(rows)


class InterpolationValidationTests(unittest.TestCase):
    def test_equal_and_nonuniform_sampling_round_trip(self):
        equal_x = np.arange(299.0, 852.0, 1.0)
        unequal_x = np.sort(
            np.concatenate(
                [np.arange(299.0, 500.0, 0.7), np.arange(500.0, 852.0, 1.3)]
            )
        )
        for axis in (equal_x, unequal_x):
            with self.subTest(points=len(axis)):
                intensity = 2 * axis + 3
                result = interpolation.validate_round_trip(axis, intensity, 1.0)
                self.assertAlmostEqual(result["rmse"], 0.0, places=10)

    def test_target_grid_has_exact_endpoints(self):
        for step, expected_count in ((0.5, 1101), (1.0, 551), (2.0, 276)):
            grid = interpolation.make_target_grid(300.0, 850.0, step)
            self.assertEqual(grid[0], 300.0)
            self.assertEqual(grid[-1], 850.0)
            self.assertEqual(len(grid), expected_count)

    def test_extrapolation_is_rejected(self):
        source_x = np.array([300.0, 301.0, 302.0])
        source_y = np.array([1.0, 2.0, 3.0])
        with self.assertRaisesRegex(ValueError, "Extrapolation"):
            interpolation.linear_interpolate_no_extrapolation(
                source_x,
                source_y,
                np.array([299.0, 300.0]),
            )

    def test_duplicate_and_descending_axes_are_rejected(self):
        for axis in (
            np.array([299.0, 300.0, 300.0, 851.0]),
            np.array([299.0, 301.0, 300.0, 851.0]),
        ):
            with self.subTest(axis=axis.tolist()):
                with self.assertRaisesRegex(ValueError, "strictly increasing"):
                    interpolation.validate_spectrum(axis, np.ones_like(axis))

    def test_nonfinite_data_are_rejected(self):
        axis = np.array([299.0, 300.0, 851.0])
        intensity = np.array([1.0, np.nan, 2.0])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            interpolation.validate_spectrum(axis, intensity)

    def test_constant_signal_has_zero_error_and_undefined_normalized_rmse(self):
        axis = np.arange(299.0, 852.0, 1.0)
        result = interpolation.validate_round_trip(
            axis,
            np.full_like(axis, 5.0),
            2.0,
        )
        self.assertEqual(result["rmse"], 0.0)
        self.assertIsNone(result["normalized_rmse"])
        self.assertEqual(
            result["normalized_rmse_status"],
            "zero_dynamic_range_not_defined",
        )

    def test_synthetic_gaussian_truth_error_and_fwhm_note(self):
        records = interpolation.run_synthetic_gaussian_validation()
        self.assertEqual(len(records), 72)
        self.assertTrue(
            all(record["rmse_vs_continuous_truth"] >= 0 for record in records)
        )
        self.assertTrue(
            any(
                record["peak_metric_status"] == "reported"
                for record in records
            )
        )
        self.assertTrue(
            all("mathematical" in record["fwhm_definition_note"] for record in records)
        )

    def test_single_file_failure_does_not_remove_other_records(self):
        with TemporaryDirectory(dir=PROJECT_DIR / "tests") as temporary_directory:
            root = Path(temporary_directory)
            axis = np.arange(299.0, 852.0, 1.0)
            (root / "valid.txt").write_bytes(
                spectrum_bytes(axis, np.sin(axis / 20) + 2)
            )
            joined = [
                (make_trial_row("missing.txt"), make_quality_row("missing.txt")),
                (make_trial_row("valid.txt"), make_quality_row("valid.txt")),
            ]
            records = interpolation.validate_real_spectra(root, joined)
            self.assertEqual(len(records), 6)
            self.assertEqual(sum(record["status"] == "failed" for record in records), 3)
            self.assertEqual(sum(record["status"] == "success" for record in records), 3)

    def test_fixed_synthetic_inputs_are_reproducible(self):
        first = interpolation.run_synthetic_gaussian_validation()
        second = interpolation.run_synthetic_gaussian_validation()
        self.assertEqual(first, second)

    def test_outputs_refuse_overwrite(self):
        with TemporaryDirectory(dir=PROJECT_DIR / "tests") as temporary_directory:
            root = Path(temporary_directory)
            raw_root = root / "raw"
            raw_root.mkdir()
            axis = np.arange(299.0, 852.0, 1.0)
            (raw_root / "sample.txt").write_bytes(
                spectrum_bytes(axis, np.cos(axis / 25) + 3)
            )
            trial_path = root / "trial.csv"
            quality_path = root / "quality.csv"
            output_dir = root / "output"
            write_csv(
                trial_path,
                interpolation.REQUIRED_TRIAL_COLUMNS,
                [make_trial_row("sample.txt")],
            )
            write_csv(
                quality_path,
                interpolation.REQUIRED_QUALITY_COLUMNS,
                [make_quality_row("sample.txt")],
            )

            result = interpolation.run_validation(
                raw_root,
                trial_path,
                quality_path,
                output_dir,
            )

            self.assertEqual(len(result[0]), 3)
            for output_path in result[3]:
                self.assertTrue(output_path.read_bytes().startswith(b"\xef\xbb\xbf"))
            with self.assertRaises(FileExistsError):
                interpolation.run_validation(
                    raw_root,
                    trial_path,
                    quality_path,
                    output_dir,
                )


if __name__ == "__main__":
    unittest.main()
