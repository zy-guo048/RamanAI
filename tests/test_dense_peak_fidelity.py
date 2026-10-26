"""Tests for dense continuous-truth Gaussian peak-fidelity validation."""

import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_DIR / "scripts" / "09_validate_dense_peak_fidelity.py"
SPEC = importlib.util.spec_from_file_location("dense_peak_fidelity", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load dense peak-fidelity module")
dense = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dense
SPEC.loader.exec_module(dense)


class DensePeakFidelityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = dense.build_dense_scenarios()
        cls.summary = dense.build_summary(cls.records)

    def test_fixed_scenario_matrix_contains_72_rows(self):
        self.assertEqual(len(self.records), 72)
        self.assertEqual(
            {record["original_sampling_step_cm-1"] for record in self.records},
            {0.5, 1.4},
        )
        self.assertEqual(
            {record["target_grid_step_cm-1"] for record in self.records},
            {0.5, 1.0, 2.0},
        )

    def test_required_example_has_zero_node_error_but_nonzero_dense_error(self):
        record = next(
            record
            for record in self.records
            if record["original_sampling_step_cm-1"] == 0.5
            and record["target_grid_step_cm-1"] == 2.0
            and record["true_gaussian_fwhm_cm-1"] == 10.0
            and record["center_offset_fraction_of_original_step"] == 0.25
        )
        self.assertAlmostEqual(record["target_node_nrmse"], 0.0, places=15)
        self.assertGreater(record["dense_full_band_nrmse"], 0.0)
        self.assertGreater(record["dense_local_3fwhm_nrmse"], 0.0)

    def test_actual_linear_maximum_is_separate_from_quadratic_height(self):
        reported = [
            record
            for record in self.records
            if record["quadratic_peak_metric_status"] == "reported"
        ]
        self.assertTrue(reported)
        self.assertTrue(
            any(
                abs(
                    float(record["target_grid_actual_maximum_intensity"])
                    - float(record["quadratic_estimated_peak_height"])
                )
                > 1e-12
                for record in reported
            )
        )
        self.assertTrue(
            all(
                "not the actual maximum"
                in record["quadratic_height_method_note"]
                for record in reported
            )
        )

    def test_underresolved_scenarios_keep_errors_without_peak_claims(self):
        underresolved = [
            record
            for record in self.records
            if record["combined_resolvability_status"]
            != "peak_parameters_reportable"
        ]
        self.assertTrue(underresolved)
        self.assertTrue(
            all(record["dense_full_band_nrmse"] >= 0 for record in underresolved)
        )
        self.assertTrue(
            all(
                record["quadratic_peak_metric_status"]
                == "not_reported_underresolved"
                for record in underresolved
            )
        )

    def test_dense_grid_is_fixed_and_inside_target_band(self):
        record = self.records[0]
        self.assertEqual(record["dense_evaluation_step_cm-1"], 0.02)
        self.assertEqual(record["dense_grid_point_count"], 27501)

    def test_paired_summary_uses_all_24_base_scenarios(self):
        self.assertEqual(self.summary["paired_base_scenario_count"], 24)
        comparison = self.summary["all_scenario_paired_comparisons"]
        for metric in dense.ALL_SCENARIO_METRICS:
            for result in comparison[metric].values():
                self.assertEqual(result["paired_scenario_count"], 24)

    def test_peak_parameter_comparison_uses_common_reportable_subset(self):
        common = self.summary["common_reportable_peak_scenarios"]
        self.assertEqual(common["paired_scenario_count"], 6)
        for metric in dense.COMMON_PEAK_METRICS:
            for result in common["paired_comparisons"][metric].values():
                self.assertEqual(result["paired_scenario_count"], 6)

    def test_repeated_runs_are_deterministic(self):
        self.assertEqual(self.records, dense.build_dense_scenarios())
        self.assertEqual(self.summary, dense.build_summary(self.records))

    def test_outputs_refuse_overwrite(self):
        with TemporaryDirectory(dir=PROJECT_DIR / "tests") as temporary_directory:
            output_dir = Path(temporary_directory) / "output"
            result = dense.run_validation(output_dir)
            self.assertEqual(len(result[0]), 72)
            for output_path in result[2]:
                self.assertTrue(output_path.read_bytes().startswith(b"\xef\xbb\xbf"))
            with self.assertRaises(FileExistsError):
                dense.run_validation(output_dir)


if __name__ == "__main__":
    unittest.main()
