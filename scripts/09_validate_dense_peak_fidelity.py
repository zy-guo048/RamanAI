"""Evaluate Gaussian peak fidelity on a fixed dense truth coordinate."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import importlib.util
import json
import math
from pathlib import Path
from statistics import fmean, median
import sys
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "09_validate_interpolation.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "exp09_dense_peak_fidelity"
DENSE_STEP = 0.02


def _load_base_module():
    spec = importlib.util.spec_from_file_location(
        "interpolation_validation_base",
        BASE_SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load experiment 9.2-A validation module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base_module()


DETAIL_FIELDS = [
    "scenario_id",
    "paired_scenario_id",
    "original_sampling_step_cm-1",
    "target_grid_step_cm-1",
    "dense_evaluation_step_cm-1",
    "true_peak_center_cm-1",
    "center_offset_fraction_of_original_step",
    "true_peak_height",
    "true_gaussian_fwhm_cm-1",
    "original_samples_per_fwhm",
    "target_samples_per_fwhm",
    "original_sampling_status",
    "target_grid_status",
    "combined_resolvability_status",
    "target_grid_point_count",
    "dense_grid_point_count",
    "target_node_nrmse",
    "dense_full_band_nrmse",
    "dense_local_3fwhm_nrmse",
    "reconstructed_intensity_at_true_center",
    "true_center_intensity_bias",
    "target_grid_actual_maximum_intensity",
    "target_grid_actual_maximum_intensity_bias",
    "quadratic_peak_metric_status",
    "quadratic_estimated_peak_center_cm-1",
    "quadratic_peak_center_bias_cm-1",
    "quadratic_estimated_peak_height",
    "quadratic_estimated_peak_height_bias",
    "linear_crossing_estimated_fwhm_cm-1",
    "linear_crossing_fwhm_bias_cm-1",
    "quadratic_height_method_note",
    "fwhm_definition_note",
    "dense_grid_note",
]

ALL_SCENARIO_METRICS = (
    "target_node_nrmse",
    "dense_full_band_nrmse",
    "dense_local_3fwhm_nrmse",
    "absolute_true_center_intensity_bias",
    "absolute_target_grid_actual_maximum_intensity_bias",
)

COMMON_PEAK_METRICS = (
    "absolute_quadratic_peak_center_bias_cm-1",
    "absolute_quadratic_estimated_peak_height_bias",
    "absolute_linear_crossing_fwhm_bias_cm-1",
)


def _resolvability_status(
    original_samples_per_fwhm: float,
    target_samples_per_fwhm: float,
) -> tuple[str, str, str, bool]:
    original_ok = original_samples_per_fwhm >= 3
    target_ok = target_samples_per_fwhm >= 3
    original_status = (
        "sufficient_by_3_samples_per_fwhm_rule"
        if original_ok
        else "underresolved_by_3_samples_per_fwhm_rule"
    )
    target_status = (
        "sufficient_by_3_samples_per_fwhm_rule"
        if target_ok
        else "underresolved_by_3_samples_per_fwhm_rule"
    )
    if original_ok and target_ok:
        combined = "peak_parameters_reportable"
    elif not original_ok and not target_ok:
        combined = "original_and_target_underresolved"
    elif not original_ok:
        combined = "original_sampling_underresolved"
    else:
        combined = "target_grid_underresolved"
    return original_status, target_status, combined, original_ok and target_ok


def build_dense_scenarios(
    dense_step: float = DENSE_STEP,
) -> list[dict[str, object]]:
    """Run the fixed 72 scenarios against a shared dense truth coordinate."""
    dense_x = base.make_target_grid(
        base.BAND_START,
        base.BAND_END,
        dense_step,
    )
    records: list[dict[str, object]] = []
    scenario_id = 0
    paired_scenario_id = 0
    for original_step in base.SYNTHETIC_ORIGINAL_STEPS:
        original_x = np.arange(
            290.0,
            860.0 + original_step / 2,
            original_step,
        )
        for true_fwhm in base.SYNTHETIC_FWHM_VALUES:
            for offset_fraction in base.SYNTHETIC_OFFSET_FRACTIONS:
                paired_scenario_id += 1
                true_center = 575.0 + offset_fraction * original_step
                original_y = base.gaussian(
                    original_x,
                    true_center,
                    true_fwhm,
                )
                dense_truth = base.gaussian(
                    dense_x,
                    true_center,
                    true_fwhm,
                )
                local_mask = (
                    (dense_x >= true_center - 3 * true_fwhm)
                    & (dense_x <= true_center + 3 * true_fwhm)
                )
                for target_step in base.TARGET_STEPS:
                    scenario_id += 1
                    target_x = base.make_target_grid(
                        base.BAND_START,
                        base.BAND_END,
                        target_step,
                    )
                    target_y = base.linear_interpolate_no_extrapolation(
                        original_x,
                        original_y,
                        target_x,
                    )
                    target_truth = base.gaussian(
                        target_x,
                        true_center,
                        true_fwhm,
                    )
                    node_errors = base.calculate_errors(target_truth, target_y)
                    dense_reconstruction = base.linear_interpolate_no_extrapolation(
                        target_x,
                        target_y,
                        dense_x,
                    )
                    dense_full_errors = base.calculate_errors(
                        dense_truth,
                        dense_reconstruction,
                    )
                    dense_local_errors = base.calculate_errors(
                        dense_truth[local_mask],
                        dense_reconstruction[local_mask],
                    )
                    reconstructed_center_intensity = float(
                        np.interp(true_center, target_x, target_y)
                    )
                    actual_maximum = float(np.max(target_y))
                    original_samples_per_fwhm = true_fwhm / original_step
                    target_samples_per_fwhm = true_fwhm / target_step
                    (
                        original_status,
                        target_status,
                        combined_status,
                        reportable,
                    ) = _resolvability_status(
                        original_samples_per_fwhm,
                        target_samples_per_fwhm,
                    )
                    record: dict[str, object] = {
                        "scenario_id": scenario_id,
                        "paired_scenario_id": paired_scenario_id,
                        "original_sampling_step_cm-1": original_step,
                        "target_grid_step_cm-1": target_step,
                        "dense_evaluation_step_cm-1": dense_step,
                        "true_peak_center_cm-1": true_center,
                        "center_offset_fraction_of_original_step": offset_fraction,
                        "true_peak_height": 1.0,
                        "true_gaussian_fwhm_cm-1": true_fwhm,
                        "original_samples_per_fwhm": original_samples_per_fwhm,
                        "target_samples_per_fwhm": target_samples_per_fwhm,
                        "original_sampling_status": original_status,
                        "target_grid_status": target_status,
                        "combined_resolvability_status": combined_status,
                        "target_grid_point_count": len(target_x),
                        "dense_grid_point_count": len(dense_x),
                        "target_node_nrmse": node_errors["normalized_rmse"],
                        "dense_full_band_nrmse": dense_full_errors[
                            "normalized_rmse"
                        ],
                        "dense_local_3fwhm_nrmse": dense_local_errors[
                            "normalized_rmse"
                        ],
                        "reconstructed_intensity_at_true_center": (
                            reconstructed_center_intensity
                        ),
                        "true_center_intensity_bias": (
                            reconstructed_center_intensity - 1.0
                        ),
                        "target_grid_actual_maximum_intensity": actual_maximum,
                        "target_grid_actual_maximum_intensity_bias": (
                            actual_maximum - 1.0
                        ),
                        "quadratic_peak_metric_status": (
                            "not_reported_underresolved"
                        ),
                        "quadratic_estimated_peak_center_cm-1": "",
                        "quadratic_peak_center_bias_cm-1": "",
                        "quadratic_estimated_peak_height": "",
                        "quadratic_estimated_peak_height_bias": "",
                        "linear_crossing_estimated_fwhm_cm-1": "",
                        "linear_crossing_fwhm_bias_cm-1": "",
                        "quadratic_height_method_note": (
                            "Quadratic peak height is the vertex of a local "
                            "three-node quadratic fit; it is not the actual maximum "
                            "of the piecewise-linear reconstructed curve."
                        ),
                        "fwhm_definition_note": (
                            "Synthetic Gaussian FWHM is mathematical half-maximum "
                            "width, not V1 half-prominence width."
                        ),
                        "dense_grid_note": (
                            "The dense coordinate is a numerical truth-evaluation "
                            "grid and does not increase instrument resolution."
                        ),
                    }
                    if reportable:
                        try:
                            center, fitted_height, estimated_fwhm = (
                                base.estimate_peak_metrics(target_x, target_y)
                            )
                            record.update(
                                {
                                    "quadratic_peak_metric_status": "reported",
                                    "quadratic_estimated_peak_center_cm-1": center,
                                    "quadratic_peak_center_bias_cm-1": (
                                        center - true_center
                                    ),
                                    "quadratic_estimated_peak_height": fitted_height,
                                    "quadratic_estimated_peak_height_bias": (
                                        fitted_height - 1.0
                                    ),
                                    "linear_crossing_estimated_fwhm_cm-1": (
                                        estimated_fwhm
                                    ),
                                    "linear_crossing_fwhm_bias_cm-1": (
                                        estimated_fwhm - true_fwhm
                                    ),
                                }
                            )
                        except ValueError as exc:
                            record["quadratic_peak_metric_status"] = (
                                f"not_reported_measurement_failure:{exc}"
                            )
                    records.append(record)
    return records


def _percentile(sorted_values: list[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return (
        sorted_values[lower] * (1 - weight)
        + sorted_values[upper] * weight
    )


def _distribution(values: Iterable[object]) -> dict[str, object]:
    numeric = sorted(float(value) for value in values if value not in ("", None))
    if not numeric:
        return {
            "count": 0,
            "minimum": None,
            "q1": None,
            "median": None,
            "p95": None,
            "maximum": None,
            "mean": None,
        }
    return {
        "count": len(numeric),
        "minimum": numeric[0],
        "q1": _percentile(numeric, 0.25),
        "median": median(numeric),
        "p95": _percentile(numeric, 0.95),
        "maximum": numeric[-1],
        "mean": fmean(numeric),
    }


def _metric_value(record: dict[str, object], metric: str) -> float:
    if metric == "absolute_true_center_intensity_bias":
        return abs(float(record["true_center_intensity_bias"]))
    if metric == "absolute_target_grid_actual_maximum_intensity_bias":
        return abs(float(record["target_grid_actual_maximum_intensity_bias"]))
    if metric == "absolute_quadratic_peak_center_bias_cm-1":
        return abs(float(record["quadratic_peak_center_bias_cm-1"]))
    if metric == "absolute_quadratic_estimated_peak_height_bias":
        return abs(float(record["quadratic_estimated_peak_height_bias"]))
    if metric == "absolute_linear_crossing_fwhm_bias_cm-1":
        return abs(float(record["linear_crossing_fwhm_bias_cm-1"]))
    return float(record[metric])


def _group_by_pair(
    records: list[dict[str, object]],
) -> dict[int, dict[float, dict[str, object]]]:
    paired: dict[int, dict[float, dict[str, object]]] = defaultdict(dict)
    for record in records:
        paired[int(record["paired_scenario_id"])][
            float(record["target_grid_step_cm-1"])
        ] = record
    return dict(paired)


def _paired_comparisons(
    paired: dict[int, dict[float, dict[str, object]]],
    pair_ids: list[int],
    metrics: tuple[str, ...],
) -> dict[str, object]:
    comparisons: dict[str, object] = {}
    for metric in metrics:
        metric_result: dict[str, object] = {}
        for left_step, right_step in ((0.5, 1.0), (0.5, 2.0), (1.0, 2.0)):
            left_wins = 0
            equal = 0
            right_wins = 0
            differences: list[float] = []
            for pair_id in pair_ids:
                left_value = _metric_value(
                    paired[pair_id][left_step],
                    metric,
                )
                right_value = _metric_value(
                    paired[pair_id][right_step],
                    metric,
                )
                differences.append(left_value - right_value)
                if math.isclose(left_value, right_value, rel_tol=1e-12, abs_tol=1e-15):
                    equal += 1
                elif left_value < right_value:
                    left_wins += 1
                else:
                    right_wins += 1
            metric_result[f"{left_step:g}_vs_{right_step:g}"] = {
                "paired_scenario_count": len(pair_ids),
                "lower_error_left_step_count": left_wins,
                "equal_error_count": equal,
                "lower_error_right_step_count": right_wins,
                "left_minus_right_distribution": _distribution(differences),
            }
        comparisons[metric] = metric_result
    return comparisons


def build_summary(records: list[dict[str, object]]) -> dict[str, object]:
    """Build overall and paired summaries with a common peak-metric subset."""
    paired = _group_by_pair(records)
    all_pair_ids = sorted(paired)
    expected_steps = set(base.TARGET_STEPS)
    if any(set(paired[pair_id]) != expected_steps for pair_id in all_pair_ids):
        raise ValueError("Each paired scenario must contain all three target steps")

    by_target_step: dict[str, object] = {}
    for target_step in base.TARGET_STEPS:
        step_records = [
            record
            for record in records
            if float(record["target_grid_step_cm-1"]) == target_step
        ]
        by_target_step[f"{target_step:g}"] = {
            "scenario_count": len(step_records),
            "underresolved_count": sum(
                record["combined_resolvability_status"]
                != "peak_parameters_reportable"
                for record in step_records
            ),
            **{
                metric: _distribution(
                    _metric_value(record, metric) for record in step_records
                )
                for metric in ALL_SCENARIO_METRICS
            },
        }

    common_peak_pair_ids = [
        pair_id
        for pair_id in all_pair_ids
        if all(
            paired[pair_id][step]["quadratic_peak_metric_status"] == "reported"
            for step in base.TARGET_STEPS
        )
    ]
    common_peak_by_step: dict[str, object] = {}
    for target_step in base.TARGET_STEPS:
        common_records = [
            paired[pair_id][target_step] for pair_id in common_peak_pair_ids
        ]
        common_peak_by_step[f"{target_step:g}"] = {
            metric: _distribution(
                _metric_value(record, metric) for record in common_records
            )
            for metric in COMMON_PEAK_METRICS
        }

    example = next(
        record
        for record in records
        if float(record["original_sampling_step_cm-1"]) == 0.5
        and float(record["target_grid_step_cm-1"]) == 2.0
        and float(record["true_gaussian_fwhm_cm-1"]) == 10.0
        and float(record["center_offset_fraction_of_original_step"]) == 0.25
    )
    return {
        "scenario_count": len(records),
        "paired_base_scenario_count": len(all_pair_ids),
        "dense_evaluation_step_cm-1": DENSE_STEP,
        "by_target_grid_step": by_target_step,
        "all_scenario_paired_comparisons": _paired_comparisons(
            paired,
            all_pair_ids,
            ALL_SCENARIO_METRICS,
        ),
        "common_reportable_peak_scenarios": {
            "paired_scenario_count": len(common_peak_pair_ids),
            "paired_scenario_ids": common_peak_pair_ids,
            "by_target_grid_step": common_peak_by_step,
            "paired_comparisons": _paired_comparisons(
                paired,
                common_peak_pair_ids,
                COMMON_PEAK_METRICS,
            ),
            "fairness_note": (
                "Peak-parameter comparisons use only scenarios reportable for "
                "all three target steps."
            ),
        },
        "required_example": {
            key: example[key]
            for key in (
                "scenario_id",
                "original_sampling_step_cm-1",
                "target_grid_step_cm-1",
                "true_gaussian_fwhm_cm-1",
                "center_offset_fraction_of_original_step",
                "target_node_nrmse",
                "dense_full_band_nrmse",
                "dense_local_3fwhm_nrmse",
                "true_center_intensity_bias",
                "target_grid_actual_maximum_intensity_bias",
            )
        },
        "dense_grid_note": (
            "The 0.02 cm-1 dense coordinate is a numerical truth-evaluation grid "
            "and does not imply increased instrument resolution."
        ),
        "quadratic_height_note": (
            "Quadratic fitted peak height is reported separately from the actual "
            "maximum of the piecewise-linear reconstructed curve."
        ),
        "underresolution_note": (
            "Underresolved scenarios retain true curve errors but are not claimed "
            "to have successfully recovered peak parameters."
        ),
        "fwhm_note": (
            "Synthetic Gaussian FWHM is mathematical half-maximum width, not V1 "
            "half-prominence width."
        ),
        "output_encoding": "utf-8-sig",
    }


def _output_paths(output_dir: Path) -> tuple[Path, Path]:
    return (
        output_dir / "dense_gaussian_peak_fidelity_scenarios.csv",
        output_dir / "dense_gaussian_peak_fidelity_summary.json",
    )


def write_outputs(
    output_dir: Path,
    records: list[dict[str, object]],
    summary: dict[str, object],
) -> tuple[Path, Path]:
    """Write deterministic UTF-8-SIG reports without overwriting."""
    paths = _output_paths(output_dir)
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Dense peak-fidelity output already exists: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path, summary_path = paths
    with details_path.open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    with summary_path.open("x", encoding="utf-8-sig", newline="") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    return paths


def run_validation(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[list[dict[str, object]], dict[str, object], tuple[Path, Path]]:
    """Run the fixed dense-fidelity scenarios and write their reports."""
    records = build_dense_scenarios()
    summary = build_summary(records)
    paths = write_outputs(output_dir, records, summary)
    return records, summary, paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the fixed experiment 9.2-A Gaussian scenarios on a dense "
            "continuous-truth coordinate."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()
    try:
        records, summary, paths = run_validation(arguments.output_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Scenario records: {len(records)}")
    print(
        "Common reportable peak scenarios: "
        f"{summary['common_reportable_peak_scenarios']['paired_scenario_count']}"
    )
    print(f"Scenario CSV: {paths[0]}")
    print(f"Summary JSON: {paths[1]}")


if __name__ == "__main__":
    main()
