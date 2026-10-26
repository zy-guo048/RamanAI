"""Validate linear interpolation numerically on real and synthetic spectra."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean, median
import sys
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raman_app.upload import parse_uploaded_spectrum


BAND_START = 300.0
BAND_END = 850.0
TARGET_STEPS = (0.5, 1.0, 2.0)
SYNTHETIC_ORIGINAL_STEPS = (0.5, 1.4)
SYNTHETIC_FWHM_VALUES = (1.0, 2.0, 5.0, 10.0)
SYNTHETIC_OFFSET_FRACTIONS = (0.0, 0.25, 0.5)

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "exp09_interpolation"
)

REQUIRED_TRIAL_COLUMNS = {
    "selection_rank",
    "relative_path",
    "file_name",
    "sample_id",
    "reference_label",
    "header_raman_wavelength",
}

REQUIRED_QUALITY_COLUMNS = {
    "selection_rank",
    "relative_path",
    "file_name",
    "sample_id",
    "reference_label",
    "header_raman_wavelength",
    "metadata_requires_review",
    "read_success",
    "all_values_finite",
    "wavenumber_strictly_increasing",
    "band_300_850_endpoint_covered",
    "delta_min",
    "delta_median",
    "delta_p95",
    "delta_max",
}

REAL_FIELDS = [
    "selection_rank",
    "relative_path",
    "file_name",
    "sample_id",
    "reference_label",
    "reference_label_status",
    "header_raman_wavelength",
    "metadata_requires_review",
    "target_grid_step_cm-1",
    "target_grid_start_cm-1",
    "target_grid_end_cm-1",
    "target_grid_point_count",
    "status",
    "error_code",
    "error_message",
    "original_band_point_count",
    "original_delta_min_cm-1",
    "original_delta_median_cm-1",
    "original_delta_p95_cm-1",
    "original_delta_max_cm-1",
    "original_intensity_dynamic_range",
    "mae",
    "rmse",
    "maximum_absolute_error",
    "normalized_rmse",
    "normalized_rmse_status",
    "round_trip_metric_note",
]

SYNTHETIC_FIELDS = [
    "scenario_id",
    "original_sampling_step_cm-1",
    "target_grid_step_cm-1",
    "true_peak_center_cm-1",
    "center_offset_fraction_of_original_step",
    "true_peak_height",
    "true_gaussian_fwhm_cm-1",
    "original_samples_per_fwhm",
    "target_samples_per_fwhm",
    "resolvability_status",
    "target_grid_point_count",
    "mae_vs_continuous_truth",
    "rmse_vs_continuous_truth",
    "maximum_absolute_error_vs_continuous_truth",
    "normalized_rmse_vs_continuous_truth",
    "peak_metric_status",
    "estimated_peak_center_cm-1",
    "peak_center_bias_cm-1",
    "estimated_peak_height",
    "peak_height_bias",
    "estimated_fwhm_cm-1",
    "fwhm_bias_cm-1",
    "fwhm_definition_note",
]


class UnsafeRelativePathError(ValueError):
    """Raised when a trial path escapes the supplied raw root."""


def _parse_bool(value: str | None) -> bool | None:
    normalized = (value or "").strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def load_csv(
    path: Path,
    required_columns: set[str],
    source_name: str,
) -> list[dict[str, str]]:
    """Load an input CSV and validate required traceability columns."""
    if not path.is_file():
        raise ValueError(f"{source_name} CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"{source_name} CSV has no header row")
        missing = sorted(required_columns - set(reader.fieldnames))
        if missing:
            raise ValueError(
                f"{source_name} CSV is missing required columns: "
                + ", ".join(missing)
            )
        return [dict(row) for row in reader]


def _unique_by_path(
    rows: list[dict[str, str]],
    source_name: str,
) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    for row in rows:
        relative_path = row.get("relative_path", "")
        if not relative_path:
            raise ValueError(f"{source_name} CSV contains an empty relative_path")
        if relative_path in indexed:
            duplicates.append(relative_path)
        else:
            indexed[relative_path] = row
    if duplicates:
        raise ValueError(
            f"{source_name} CSV contains duplicate relative_path values: "
            + ", ".join(sorted(set(duplicates), key=str.casefold)[:3])
        )
    return indexed


def join_inputs(
    trial_rows: list[dict[str, str]],
    quality_rows: list[dict[str, str]],
) -> list[tuple[dict[str, str], dict[str, str]]]:
    """Join the trial list and sampling report one-to-one by exact path."""
    trial_by_path = _unique_by_path(trial_rows, "Trial")
    quality_by_path = _unique_by_path(quality_rows, "Sampling quality")
    if set(trial_by_path) != set(quality_by_path):
        trial_only = sorted(
            set(trial_by_path) - set(quality_by_path), key=str.casefold
        )
        quality_only = sorted(
            set(quality_by_path) - set(trial_by_path), key=str.casefold
        )
        raise ValueError(
            "Input record sets do not match by relative_path: "
            f"trial_only={len(trial_only)} {trial_only[:3]}, "
            f"quality_only={len(quality_only)} {quality_only[:3]}"
        )
    joined: list[tuple[dict[str, str], dict[str, str]]] = []
    common_fields = (
        "selection_rank",
        "file_name",
        "sample_id",
        "reference_label",
        "header_raman_wavelength",
    )
    for trial_row in sorted(
        trial_rows,
        key=lambda row: int(row.get("selection_rank", "0")),
    ):
        quality_row = quality_by_path[trial_row["relative_path"]]
        mismatches = [
            field
            for field in common_fields
            if trial_row.get(field, "") != quality_row.get(field, "")
        ]
        if mismatches:
            raise ValueError(
                "Trial and sampling-quality records disagree for relative_path "
                f"{trial_row['relative_path']}: {', '.join(mismatches)}"
            )
        joined.append((trial_row, quality_row))
    return joined


def resolve_safe_raw_path(raw_root: Path, relative_path: str) -> Path:
    """Resolve a relative RAW path without allowing traversal outside root."""
    root = raw_root.resolve(strict=True)
    supplied = Path(relative_path)
    if supplied.is_absolute():
        raise UnsafeRelativePathError(
            f"Absolute trial path is not allowed: {relative_path}"
        )
    target = (root / supplied).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise UnsafeRelativePathError(
            f"Trial path escapes raw_root: {relative_path}"
        ) from exc
    return target


def make_target_grid(
    start: float,
    end: float,
    step: float,
) -> np.ndarray:
    """Create an inclusive deterministic grid with exact requested endpoints."""
    if not all(math.isfinite(value) for value in (start, end, step)):
        raise ValueError("Grid parameters must be finite")
    if end <= start or step <= 0:
        raise ValueError("Grid bounds and step must be positive and increasing")
    interval_count_float = (end - start) / step
    interval_count = round(interval_count_float)
    if not math.isclose(
        interval_count_float,
        interval_count,
        rel_tol=0,
        abs_tol=1e-10,
    ):
        raise ValueError("Grid step does not divide the requested range exactly")
    grid = start + np.arange(interval_count + 1, dtype=float) * step
    grid[0] = start
    grid[-1] = end
    return grid


def validate_spectrum(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    start: float = BAND_START,
    end: float = BAND_END,
) -> None:
    """Validate arrays and endpoint coverage without changing the data."""
    if wavenumbers.ndim != 1 or intensities.ndim != 1:
        raise ValueError("Spectrum arrays must be one-dimensional")
    if len(wavenumbers) != len(intensities) or len(wavenumbers) < 2:
        raise ValueError("Spectrum arrays must have equal length of at least two")
    if not np.all(np.isfinite(wavenumbers)) or not np.all(
        np.isfinite(intensities)
    ):
        raise ValueError("Spectrum contains non-finite values")
    if not np.all(np.diff(wavenumbers) > 0):
        raise ValueError("Wavenumbers must be strictly increasing without duplicates")
    if wavenumbers[0] > start or wavenumbers[-1] < end:
        raise ValueError(
            f"Spectrum does not cover the target band {start:g}-{end:g} cm-1"
        )


def linear_interpolate_no_extrapolation(
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate only when every target point is inside source_x."""
    if target_x.ndim != 1 or len(target_x) == 0:
        raise ValueError("Target grid must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(target_x)) or not np.all(np.diff(target_x) > 0):
        raise ValueError("Target grid must be finite and strictly increasing")
    validate_spectrum(source_x, source_y, source_x[0], source_x[-1])
    tolerance = 1e-12
    if target_x[0] < source_x[0] - tolerance or target_x[-1] > source_x[-1] + tolerance:
        raise ValueError("Extrapolation is not allowed")
    return np.interp(target_x, source_x, source_y)


def calculate_errors(
    observed: np.ndarray,
    reconstructed: np.ndarray,
) -> dict[str, object]:
    """Calculate absolute errors and RMSE normalized by observed range."""
    if observed.shape != reconstructed.shape or observed.size == 0:
        raise ValueError("Error arrays must have the same non-zero shape")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(reconstructed)):
        raise ValueError("Error arrays must contain only finite values")
    residual = reconstructed - observed
    absolute = np.abs(residual)
    rmse = float(np.sqrt(np.mean(residual**2)))
    dynamic_range = float(np.max(observed) - np.min(observed))
    if dynamic_range == 0:
        normalized_rmse: float | None = None
        normalized_status = "zero_dynamic_range_not_defined"
    else:
        normalized_rmse = rmse / dynamic_range
        normalized_status = "defined"
    return {
        "mae": float(np.mean(absolute)),
        "rmse": rmse,
        "maximum_absolute_error": float(np.max(absolute)),
        "dynamic_range": dynamic_range,
        "normalized_rmse": normalized_rmse,
        "normalized_rmse_status": normalized_status,
    }


def validate_round_trip(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    target_step: float,
) -> dict[str, object]:
    """Perform original→grid→original interpolation inside 300–850 cm-1."""
    validate_spectrum(wavenumbers, intensities)
    target_grid = make_target_grid(BAND_START, BAND_END, target_step)
    grid_intensity = linear_interpolate_no_extrapolation(
        wavenumbers,
        intensities,
        target_grid,
    )
    band_mask = (wavenumbers >= BAND_START) & (wavenumbers <= BAND_END)
    original_x = wavenumbers[band_mask]
    original_y = intensities[band_mask]
    if len(original_x) < 2:
        raise ValueError("Fewer than two original samples lie inside target band")
    reconstructed = linear_interpolate_no_extrapolation(
        target_grid,
        grid_intensity,
        original_x,
    )
    errors = calculate_errors(original_y, reconstructed)
    deltas = np.diff(original_x)
    return {
        "target_grid": target_grid,
        "original_band_point_count": len(original_x),
        "original_delta_min_cm-1": float(np.min(deltas)),
        "original_delta_median_cm-1": float(np.median(deltas)),
        "original_delta_p95_cm-1": float(np.percentile(deltas, 95)),
        "original_delta_max_cm-1": float(np.max(deltas)),
        **errors,
    }


def _real_base_record(
    trial_row: dict[str, str],
    quality_row: dict[str, str],
    step: float,
) -> dict[str, object]:
    grid = make_target_grid(BAND_START, BAND_END, step)
    return {
        "selection_rank": trial_row.get("selection_rank", ""),
        "relative_path": trial_row.get("relative_path", ""),
        "file_name": trial_row.get("file_name", ""),
        "sample_id": trial_row.get("sample_id", ""),
        "reference_label": trial_row.get("reference_label", ""),
        "reference_label_status": "candidate_label_not_independently_validated",
        "header_raman_wavelength": trial_row.get(
            "header_raman_wavelength", ""
        ),
        "metadata_requires_review": quality_row.get(
            "metadata_requires_review", ""
        ),
        "target_grid_step_cm-1": step,
        "target_grid_start_cm-1": float(grid[0]),
        "target_grid_end_cm-1": float(grid[-1]),
        "target_grid_point_count": len(grid),
        "status": "failed",
        "error_code": "",
        "error_message": "",
        "original_band_point_count": "",
        "original_delta_min_cm-1": quality_row.get("delta_min", ""),
        "original_delta_median_cm-1": quality_row.get("delta_median", ""),
        "original_delta_p95_cm-1": quality_row.get("delta_p95", ""),
        "original_delta_max_cm-1": quality_row.get("delta_max", ""),
        "original_intensity_dynamic_range": "",
        "mae": "",
        "rmse": "",
        "maximum_absolute_error": "",
        "normalized_rmse": "",
        "normalized_rmse_status": "not_calculated",
        "round_trip_metric_note": (
            "Original-to-grid-to-original error is a numerical reconstruction "
            "metric, not independent measured truth and not proof of peak-shape "
            "fidelity."
        ),
    }


def validate_real_spectra(
    raw_root: Path,
    joined_rows: list[tuple[dict[str, str], dict[str, str]]],
) -> list[dict[str, object]]:
    """Validate every real file for each target grid while isolating failures."""
    records: list[dict[str, object]] = []
    for trial_row, quality_row in joined_rows:
        file_error: tuple[str, str] | None = None
        wavenumbers: np.ndarray | None = None
        intensities: np.ndarray | None = None
        try:
            if _parse_bool(quality_row.get("read_success")) is not True:
                raise ValueError("Sampling-quality report marks file read as failed")
            if _parse_bool(quality_row.get("all_values_finite")) is not True:
                raise ValueError("Sampling-quality report marks non-finite values")
            if (
                _parse_bool(quality_row.get("wavenumber_strictly_increasing"))
                is not True
            ):
                raise ValueError(
                    "Sampling-quality report marks axis as not strictly increasing"
                )
            if (
                _parse_bool(
                    quality_row.get("band_300_850_endpoint_covered")
                )
                is not True
            ):
                raise ValueError("Sampling-quality report marks target band uncovered")
            path = resolve_safe_raw_path(raw_root, trial_row["relative_path"])
            spectrum = parse_uploaded_spectrum(path.read_bytes())
            wavenumbers = np.asarray(spectrum.spectral_axis, dtype=float)
            intensities = np.asarray(spectrum.spectral_data, dtype=float)
            validate_spectrum(wavenumbers, intensities)
        except UnsafeRelativePathError as exc:
            file_error = ("unsafe_relative_path", str(exc))
        except FileNotFoundError as exc:
            file_error = ("file_not_found", str(exc))
        except (OSError, TypeError, ValueError) as exc:
            file_error = ("file_validation_failed", str(exc))

        for step in TARGET_STEPS:
            record = _real_base_record(trial_row, quality_row, step)
            if file_error is not None:
                record["error_code"], record["error_message"] = file_error
                records.append(record)
                continue
            try:
                result = validate_round_trip(wavenumbers, intensities, step)
                record.update(
                    {
                        "status": "success",
                        "original_band_point_count": result[
                            "original_band_point_count"
                        ],
                        "original_delta_min_cm-1": result[
                            "original_delta_min_cm-1"
                        ],
                        "original_delta_median_cm-1": result[
                            "original_delta_median_cm-1"
                        ],
                        "original_delta_p95_cm-1": result[
                            "original_delta_p95_cm-1"
                        ],
                        "original_delta_max_cm-1": result[
                            "original_delta_max_cm-1"
                        ],
                        "original_intensity_dynamic_range": result[
                            "dynamic_range"
                        ],
                        "mae": result["mae"],
                        "rmse": result["rmse"],
                        "maximum_absolute_error": result[
                            "maximum_absolute_error"
                        ],
                        "normalized_rmse": (
                            ""
                            if result["normalized_rmse"] is None
                            else result["normalized_rmse"]
                        ),
                        "normalized_rmse_status": result[
                            "normalized_rmse_status"
                        ],
                    }
                )
            except ValueError as exc:
                record["error_code"] = "round_trip_validation_failed"
                record["error_message"] = str(exc)
            records.append(record)
    return records


def gaussian(
    x: np.ndarray,
    center: float,
    fwhm: float,
    height: float = 1.0,
) -> np.ndarray:
    """Evaluate a continuous Gaussian parameterized by mathematical FWHM."""
    if fwhm <= 0 or height <= 0:
        raise ValueError("Gaussian FWHM and height must be positive")
    sigma = fwhm / (2 * math.sqrt(2 * math.log(2)))
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _linear_crossing(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    level: float,
) -> float:
    if y2 == y1:
        return (x1 + x2) / 2
    return x1 + (level - y1) * (x2 - x1) / (y2 - y1)


def estimate_peak_metrics(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float]:
    """Estimate center, height, and FWHM from a sufficiently resolved peak."""
    peak_index = int(np.argmax(y))
    if peak_index == 0 or peak_index == len(y) - 1:
        raise ValueError("Peak maximum lies on grid boundary")
    local_x = x[peak_index - 1 : peak_index + 2]
    local_y = y[peak_index - 1 : peak_index + 2]
    coefficients = np.polyfit(local_x, local_y, 2)
    if coefficients[0] >= 0:
        center = float(x[peak_index])
        height = float(y[peak_index])
    else:
        center = float(-coefficients[1] / (2 * coefficients[0]))
        height = float(np.polyval(coefficients, center))
    half_height = height / 2

    left_candidates = np.flatnonzero(y[: peak_index + 1] <= half_height)
    right_candidates = np.flatnonzero(y[peak_index:] <= half_height)
    if len(left_candidates) == 0 or len(right_candidates) == 0:
        raise ValueError("Half-height crossings are not bracketed")
    left_low = int(left_candidates[-1])
    left_high = left_low + 1
    right_high = peak_index + int(right_candidates[0])
    right_low = right_high - 1
    left_crossing = _linear_crossing(
        float(x[left_low]),
        float(y[left_low]),
        float(x[left_high]),
        float(y[left_high]),
        half_height,
    )
    right_crossing = _linear_crossing(
        float(x[right_low]),
        float(y[right_low]),
        float(x[right_high]),
        float(y[right_high]),
        half_height,
    )
    return center, height, right_crossing - left_crossing


def run_synthetic_gaussian_validation() -> list[dict[str, object]]:
    """Compare interpolated Gaussian values with known continuous truth."""
    records: list[dict[str, object]] = []
    scenario_number = 0
    for original_step in SYNTHETIC_ORIGINAL_STEPS:
        original_x = np.arange(290.0, 860.0 + original_step / 2, original_step)
        for true_fwhm in SYNTHETIC_FWHM_VALUES:
            for offset_fraction in SYNTHETIC_OFFSET_FRACTIONS:
                true_center = 575.0 + offset_fraction * original_step
                original_y = gaussian(original_x, true_center, true_fwhm)
                for target_step in TARGET_STEPS:
                    scenario_number += 1
                    target_x = make_target_grid(BAND_START, BAND_END, target_step)
                    interpolated = linear_interpolate_no_extrapolation(
                        original_x,
                        original_y,
                        target_x,
                    )
                    truth = gaussian(target_x, true_center, true_fwhm)
                    errors = calculate_errors(truth, interpolated)
                    original_samples_per_fwhm = true_fwhm / original_step
                    target_samples_per_fwhm = true_fwhm / target_step
                    reliable = (
                        original_samples_per_fwhm >= 3
                        and target_samples_per_fwhm >= 3
                    )
                    record: dict[str, object] = {
                        "scenario_id": scenario_number,
                        "original_sampling_step_cm-1": original_step,
                        "target_grid_step_cm-1": target_step,
                        "true_peak_center_cm-1": true_center,
                        "center_offset_fraction_of_original_step": offset_fraction,
                        "true_peak_height": 1.0,
                        "true_gaussian_fwhm_cm-1": true_fwhm,
                        "original_samples_per_fwhm": original_samples_per_fwhm,
                        "target_samples_per_fwhm": target_samples_per_fwhm,
                        "resolvability_status": (
                            "sufficient_for_reported_peak_metrics"
                            if reliable
                            else "underresolved_peak_metrics_not_reported"
                        ),
                        "target_grid_point_count": len(target_x),
                        "mae_vs_continuous_truth": errors["mae"],
                        "rmse_vs_continuous_truth": errors["rmse"],
                        "maximum_absolute_error_vs_continuous_truth": errors[
                            "maximum_absolute_error"
                        ],
                        "normalized_rmse_vs_continuous_truth": errors[
                            "normalized_rmse"
                        ],
                        "peak_metric_status": "not_reported_underresolved",
                        "estimated_peak_center_cm-1": "",
                        "peak_center_bias_cm-1": "",
                        "estimated_peak_height": "",
                        "peak_height_bias": "",
                        "estimated_fwhm_cm-1": "",
                        "fwhm_bias_cm-1": "",
                        "fwhm_definition_note": (
                            "Synthetic Gaussian FWHM is the mathematical "
                            "half-maximum width and is not V1 half-prominence width."
                        ),
                    }
                    if reliable:
                        try:
                            center, height, estimated_fwhm = estimate_peak_metrics(
                                target_x,
                                interpolated,
                            )
                            record.update(
                                {
                                    "peak_metric_status": "reported",
                                    "estimated_peak_center_cm-1": center,
                                    "peak_center_bias_cm-1": center - true_center,
                                    "estimated_peak_height": height,
                                    "peak_height_bias": height - 1.0,
                                    "estimated_fwhm_cm-1": estimated_fwhm,
                                    "fwhm_bias_cm-1": estimated_fwhm - true_fwhm,
                                }
                            )
                        except ValueError as exc:
                            record["peak_metric_status"] = (
                                f"not_reported_measurement_failure:{exc}"
                            )
                    records.append(record)
    return records


def _distribution(values: Iterable[object]) -> dict[str, object]:
    numeric = [float(value) for value in values if value not in ("", None)]
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
    array = np.sort(np.asarray(numeric, dtype=float))
    return {
        "count": len(numeric),
        "minimum": float(array[0]),
        "q1": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(array[-1]),
        "mean": fmean(numeric),
    }


def build_summary(
    raw_root: Path,
    trial_path: Path,
    quality_path: Path,
    real_records: list[dict[str, object]],
    synthetic_records: list[dict[str, object]],
) -> dict[str, object]:
    """Summarize real round-trip and synthetic truth errors separately."""
    real_by_step: dict[str, object] = {}
    for step in TARGET_STEPS:
        step_records = [
            record
            for record in real_records
            if float(record["target_grid_step_cm-1"]) == step
        ]
        successful = [record for record in step_records if record["status"] == "success"]
        ranked = sorted(
            [record for record in successful if record["normalized_rmse"] != ""],
            key=lambda record: float(record["normalized_rmse"]),
            reverse=True,
        )
        real_by_step[f"{step:g}"] = {
            "record_count": len(step_records),
            "success_count": len(successful),
            "failure_count": len(step_records) - len(successful),
            "zero_dynamic_range_count": sum(
                record["normalized_rmse_status"]
                == "zero_dynamic_range_not_defined"
                for record in successful
            ),
            "mae": _distribution(record["mae"] for record in successful),
            "rmse": _distribution(record["rmse"] for record in successful),
            "maximum_absolute_error": _distribution(
                record["maximum_absolute_error"] for record in successful
            ),
            "normalized_rmse": _distribution(
                record["normalized_rmse"] for record in successful
            ),
            "highest_normalized_rmse_records": [
                {
                    "relative_path": record["relative_path"],
                    "sample_id": record["sample_id"],
                    "normalized_rmse": record["normalized_rmse"],
                    "rmse": record["rmse"],
                }
                for record in ranked[:5]
            ],
        }

    synthetic_by_step: dict[str, object] = {}
    for step in TARGET_STEPS:
        step_records = [
            record
            for record in synthetic_records
            if float(record["target_grid_step_cm-1"]) == step
        ]
        reliable = [
            record
            for record in step_records
            if record["peak_metric_status"] == "reported"
        ]
        synthetic_by_step[f"{step:g}"] = {
            "scenario_count": len(step_records),
            "underresolved_scenario_count": sum(
                record["resolvability_status"]
                == "underresolved_peak_metrics_not_reported"
                for record in step_records
            ),
            "rmse_vs_continuous_truth": _distribution(
                record["rmse_vs_continuous_truth"] for record in step_records
            ),
            "normalized_rmse_vs_continuous_truth": _distribution(
                record["normalized_rmse_vs_continuous_truth"]
                for record in step_records
            ),
            "maximum_absolute_error_vs_continuous_truth": _distribution(
                record["maximum_absolute_error_vs_continuous_truth"]
                for record in step_records
            ),
            "reliably_reported_peak_metrics": len(reliable),
            "absolute_peak_center_bias_cm-1": _distribution(
                abs(float(record["peak_center_bias_cm-1"]))
                for record in reliable
            ),
            "absolute_peak_height_bias": _distribution(
                abs(float(record["peak_height_bias"])) for record in reliable
            ),
            "absolute_fwhm_bias_cm-1": _distribution(
                abs(float(record["fwhm_bias_cm-1"])) for record in reliable
            ),
        }
    return {
        "raw_root": str(raw_root.resolve()),
        "source_trial_csv": str(trial_path.resolve()),
        "source_sampling_quality_csv": str(quality_path.resolve()),
        "analysis_band_cm-1": [BAND_START, BAND_END],
        "target_grid_steps_cm-1": list(TARGET_STEPS),
        "real_spectrum_round_trip": real_by_step,
        "synthetic_gaussian_truth": synthetic_by_step,
        "real_metric_limitation": (
            "Original-to-grid-to-original error is a numerical reconstruction "
            "metric. It is not independent measured truth and cannot alone prove "
            "peak-shape fidelity."
        ),
        "synthetic_metric_note": (
            "Synthetic values are compared directly with a known continuous "
            "Gaussian truth. Underresolved narrow peaks are not credited as "
            "successfully recovered."
        ),
        "fwhm_note": (
            "Gaussian FWHM here is mathematical half-maximum width and must not "
            "be confused with V1 half-prominence width."
        ),
        "grid_decision_status": (
            "These results compare candidate steps only and do not select a final "
            "unified Raman-shift grid."
        ),
        "output_encoding": "utf-8-sig",
    }


def _output_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    return (
        output_dir / "real_spectrum_interpolation_errors.csv",
        output_dir / "synthetic_gaussian_interpolation_results.csv",
        output_dir / "interpolation_validation_summary.json",
    )


def write_outputs(
    output_dir: Path,
    real_records: list[dict[str, object]],
    synthetic_records: list[dict[str, object]],
    summary: dict[str, object],
) -> tuple[Path, Path, Path]:
    """Write deterministic UTF-8-SIG outputs without overwriting history."""
    paths = _output_paths(output_dir)
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Interpolation output already exists: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    real_path, synthetic_path, summary_path = paths
    with real_path.open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=REAL_FIELDS)
        writer.writeheader()
        writer.writerows(real_records)
    with synthetic_path.open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=SYNTHETIC_FIELDS)
        writer.writeheader()
        writer.writerows(synthetic_records)
    with summary_path.open("x", encoding="utf-8-sig", newline="") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    return paths


def run_validation(
    raw_root: Path,
    trial_path: Path,
    quality_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    tuple[Path, Path, Path],
]:
    """Run real and synthetic interpolation validation reproducibly."""
    if not raw_root.is_dir():
        raise ValueError(f"raw_root is not a directory: {raw_root}")
    trial_rows = load_csv(trial_path, REQUIRED_TRIAL_COLUMNS, "Trial")
    quality_rows = load_csv(
        quality_path,
        REQUIRED_QUALITY_COLUMNS,
        "Sampling quality",
    )
    joined_rows = join_inputs(trial_rows, quality_rows)
    real_records = validate_real_spectra(raw_root, joined_rows)
    synthetic_records = run_synthetic_gaussian_validation()
    summary = build_summary(
        raw_root,
        trial_path,
        quality_path,
        real_records,
        synthetic_records,
    )
    output_paths = write_outputs(
        output_dir,
        real_records,
        synthetic_records,
        summary,
    )
    return real_records, synthetic_records, summary, output_paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate linear interpolation on real RAW spectra and synthetic "
            "Gaussian truth without preprocessing or extrapolation."
        )
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--sampling-quality", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()
    try:
        real, synthetic, summary, paths = run_validation(
            arguments.raw_root,
            arguments.trial,
            arguments.sampling_quality,
            arguments.output_dir,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Real-spectrum records: {len(real)}")
    print(f"Synthetic Gaussian records: {len(synthetic)}")
    print(f"Real errors CSV: {paths[0]}")
    print(f"Synthetic results CSV: {paths[1]}")
    print(f"Summary JSON: {paths[2]}")
    print(f"Grid decision: {summary['grid_decision_status']}")


if __name__ == "__main__":
    main()
