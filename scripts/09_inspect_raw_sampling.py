"""Inspect actual sampling quality for a trial list of RRUFF RAW spectra."""

from __future__ import annotations

import argparse
import csv
from io import StringIO
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


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "exp09_sampling"
)

REFERENCE_BANDS = (
    ("300_850", 300.0, 850.0),
    ("300_1150", 300.0, 1150.0),
    ("232_1466", 232.0, 1466.0),
)

REQUIRED_TRIAL_COLUMNS = {
    "selection_rank",
    "relative_path",
    "file_name",
    "sample_id",
    "filename_sample_id",
    "header_rruff_id",
    "reference_label",
    "header_raman_wavelength",
    "wavenumber_min_cm-1",
    "wavenumber_max_cm-1",
    "point_count",
}

REQUIRED_AUDIT_COLUMNS = {
    "relative_path",
    "file_name",
    "file_type",
    "header_names",
    "header_rruff_id",
    "header_raman_wavelength",
    "point_count",
    "wavenumber_min",
    "wavenumber_max",
    "metadata_warnings",
}

QUALITY_FIELDS = [
    "selection_rank",
    "relative_path",
    "file_name",
    "sample_id",
    "filename_sample_id",
    "header_rruff_id",
    "reference_label",
    "reference_label_status",
    "header_raman_wavelength",
    "metadata_requires_review",
    "audit_metadata_warnings",
    "resolved_path",
    "file_size_bytes",
    "read_success",
    "v1_parser_success",
    "error_code",
    "error_message",
    "actual_point_count",
    "audit_point_count",
    "point_count_matches_audit",
    "actual_wavenumber_min",
    "audit_wavenumber_min",
    "wavenumber_min_matches_audit",
    "actual_wavenumber_max",
    "audit_wavenumber_max",
    "wavenumber_max_matches_audit",
    "audit_consistency_status",
    "audit_mismatch_reasons",
    "axis_values_finite",
    "intensity_values_finite",
    "all_values_finite",
    "wavenumber_strictly_increasing",
    "has_duplicate_wavenumber",
    "has_descending_wavenumber",
    "delta_min",
    "delta_median",
    "delta_p95",
    "delta_max",
    "delta_max_median_ratio",
    "gap_count_gt_3x_median",
    "gap_count_gt_5x_median",
]
for _band_key, _band_start, _band_end in REFERENCE_BANDS:
    QUALITY_FIELDS.extend(
        [
            f"band_{_band_key}_endpoint_covered",
            f"band_{_band_key}_sample_point_count",
            f"band_{_band_key}_internal_max_delta",
            f"band_{_band_key}_diagnostic_status",
        ]
    )

GAP_FIELDS = [
    "selection_rank",
    "relative_path",
    "sample_id",
    "gap_left_point_index",
    "gap_right_point_index",
    "left_wavenumber",
    "right_wavenumber",
    "delta",
    "spectrum_delta_median",
    "ratio_to_median",
    "exceeds_3x_median",
    "exceeds_5x_median",
]


class UnsafeRelativePathError(ValueError):
    """Raised when a trial relative path escapes the supplied RAW root."""


def _parse_bool(value: str | None) -> bool | None:
    normalized = (value or "").strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _split_flags(value: str | None) -> list[str]:
    return [part.strip() for part in (value or "").split(";") if part.strip()]


def _is_missing(value: str | None) -> bool:
    if value is None:
        return True
    return value.strip().casefold() in {"", "unknown", "none", "n/a"}


def load_csv(
    path: Path,
    required_columns: set[str],
    source_name: str,
) -> list[dict[str, str]]:
    """Load a traceability CSV with the fields required by this experiment."""
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
        examples = sorted(set(duplicates), key=str.casefold)[:3]
        raise ValueError(
            f"{source_name} CSV contains duplicate relative_path values: "
            + ", ".join(examples)
        )
    return indexed


def _float_equal(left: str, right: str, tolerance: float = 1e-6) -> bool:
    try:
        left_value = float(left)
        right_value = float(right)
    except ValueError:
        return False
    return (
        math.isfinite(left_value)
        and math.isfinite(right_value)
        and math.isclose(
            left_value,
            right_value,
            rel_tol=1e-9,
            abs_tol=tolerance,
        )
    )


def join_trial_with_audit(
    trial_rows: list[dict[str, str]],
    audit_rows: list[dict[str, str]],
) -> list[tuple[dict[str, str], dict[str, str]]]:
    """Match every trial record to exactly one complete-audit record."""
    trial_by_path = _unique_by_path(trial_rows, "Trial")
    audit_by_path = _unique_by_path(audit_rows, "Audit")
    missing_paths = sorted(
        set(trial_by_path) - set(audit_by_path),
        key=str.casefold,
    )
    if missing_paths:
        raise ValueError(
            "Trial records are missing from the audit CSV: "
            + ", ".join(missing_paths[:3])
        )

    joined: list[tuple[dict[str, str], dict[str, str]]] = []
    for trial_row in sorted(
        trial_rows,
        key=lambda row: int(row.get("selection_rank", "0")),
    ):
        relative_path = trial_row["relative_path"]
        audit_row = audit_by_path[relative_path]
        mismatches: list[str] = []
        direct_pairs = (
            ("file_name", "file_name"),
            ("reference_label", "header_names"),
            ("header_rruff_id", "header_rruff_id"),
            ("header_raman_wavelength", "header_raman_wavelength"),
            ("point_count", "point_count"),
        )
        for trial_field, audit_field in direct_pairs:
            if trial_row.get(trial_field, "") != audit_row.get(audit_field, ""):
                mismatches.append(f"{trial_field}/{audit_field}")
        if not _float_equal(
            trial_row.get("wavenumber_min_cm-1", ""),
            audit_row.get("wavenumber_min", ""),
        ):
            mismatches.append("wavenumber_min_cm-1/wavenumber_min")
        if not _float_equal(
            trial_row.get("wavenumber_max_cm-1", ""),
            audit_row.get("wavenumber_max", ""),
        ):
            mismatches.append("wavenumber_max_cm-1/wavenumber_max")
        if audit_row.get("file_type", "").strip().casefold() != "raw":
            mismatches.append("audit_file_type_not_raw")
        if mismatches:
            raise ValueError(
                "Trial and audit records disagree for relative_path "
                f"{relative_path}: {', '.join(mismatches)}"
            )
        joined.append((trial_row, audit_row))
    return joined


def resolve_safe_raw_path(raw_root: Path, relative_path: str) -> Path:
    """Resolve a trial path and ensure it remains inside raw_root."""
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


def _load_numeric_spectrum(content: bytes) -> tuple[np.ndarray, np.ndarray]:
    if not content:
        raise ValueError("empty_file")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("decode_error") from exc
    try:
        data = np.loadtxt(StringIO(text), delimiter=",", comments="#")
    except (TypeError, ValueError) as exc:
        raise ValueError("numeric_parse_error") from exc
    if data.ndim == 1 and data.size == 2:
        data = data.reshape(1, 2)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("data_column_count_not_two")
    return np.asarray(data[:, 0], dtype=float), np.asarray(data[:, 1], dtype=float)


def _percentile(values: np.ndarray, fraction: float) -> float:
    sorted_values = np.sort(np.asarray(values, dtype=float))
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
    )


def inspect_arrays(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    selection_rank: str,
    relative_path: str,
    sample_id: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Inspect numeric arrays without interpolation or signal processing."""
    if wavenumbers.ndim != 1 or intensities.ndim != 1:
        raise ValueError("spectrum_arrays_must_be_one_dimensional")
    if len(wavenumbers) != len(intensities):
        raise ValueError("spectrum_array_lengths_differ")
    if len(wavenumbers) == 0:
        raise ValueError("empty_spectrum")

    axis_finite = bool(np.all(np.isfinite(wavenumbers)))
    intensity_finite = bool(np.all(np.isfinite(intensities)))
    result: dict[str, object] = {
        "actual_point_count": len(wavenumbers),
        "actual_wavenumber_min": (
            float(np.min(wavenumbers)) if axis_finite else ""
        ),
        "actual_wavenumber_max": (
            float(np.max(wavenumbers)) if axis_finite else ""
        ),
        "axis_values_finite": axis_finite,
        "intensity_values_finite": intensity_finite,
        "all_values_finite": axis_finite and intensity_finite,
        "wavenumber_strictly_increasing": False,
        "has_duplicate_wavenumber": False,
        "has_descending_wavenumber": False,
        "delta_min": "",
        "delta_median": "",
        "delta_p95": "",
        "delta_max": "",
        "delta_max_median_ratio": "",
        "gap_count_gt_3x_median": 0,
        "gap_count_gt_5x_median": 0,
    }
    for band_key, _band_start, _band_end in REFERENCE_BANDS:
        result[f"band_{band_key}_endpoint_covered"] = False
        result[f"band_{band_key}_sample_point_count"] = 0
        result[f"band_{band_key}_internal_max_delta"] = ""
        result[f"band_{band_key}_diagnostic_status"] = "axis_not_usable"

    if not axis_finite:
        return result, []

    deltas = np.diff(wavenumbers)
    has_duplicate = bool(np.any(deltas == 0))
    has_descending = bool(np.any(deltas < 0))
    strictly_increasing = bool(np.all(deltas > 0)) if len(deltas) else True
    result["wavenumber_strictly_increasing"] = strictly_increasing
    result["has_duplicate_wavenumber"] = has_duplicate
    result["has_descending_wavenumber"] = has_descending

    gap_rows: list[dict[str, object]] = []
    if len(deltas):
        delta_median = float(np.median(deltas))
        delta_max = float(np.max(deltas))
        result.update(
            {
                "delta_min": float(np.min(deltas)),
                "delta_median": delta_median,
                "delta_p95": _percentile(deltas, 0.95),
                "delta_max": delta_max,
                "delta_max_median_ratio": (
                    delta_max / delta_median if delta_median > 0 else ""
                ),
            }
        )
        if delta_median > 0:
            exceeds_3 = deltas > 3 * delta_median
            exceeds_5 = deltas > 5 * delta_median
            result["gap_count_gt_3x_median"] = int(np.count_nonzero(exceeds_3))
            result["gap_count_gt_5x_median"] = int(np.count_nonzero(exceeds_5))
            for index in np.flatnonzero(exceeds_3):
                delta = float(deltas[index])
                gap_rows.append(
                    {
                        "selection_rank": selection_rank,
                        "relative_path": relative_path,
                        "sample_id": sample_id,
                        "gap_left_point_index": int(index),
                        "gap_right_point_index": int(index + 1),
                        "left_wavenumber": float(wavenumbers[index]),
                        "right_wavenumber": float(wavenumbers[index + 1]),
                        "delta": delta,
                        "spectrum_delta_median": delta_median,
                        "ratio_to_median": delta / delta_median,
                        "exceeds_3x_median": True,
                        "exceeds_5x_median": bool(exceeds_5[index]),
                    }
                )

    for band_key, band_start, band_end in REFERENCE_BANDS:
        in_band = (wavenumbers >= band_start) & (wavenumbers <= band_end)
        band_values = wavenumbers[in_band]
        endpoint_covered = bool(
            strictly_increasing
            and wavenumbers[0] <= band_start
            and wavenumbers[-1] >= band_end
        )
        if len(band_values) >= 2:
            internal_max_delta: float | str = float(np.max(np.diff(band_values)))
        else:
            internal_max_delta = ""
        if not strictly_increasing:
            status = "axis_not_strictly_increasing"
        elif endpoint_covered:
            status = "endpoint_covered_internal_sampling_observed"
        else:
            status = "target_band_not_fully_covered_no_extrapolation"
        result[f"band_{band_key}_endpoint_covered"] = endpoint_covered
        result[f"band_{band_key}_sample_point_count"] = int(len(band_values))
        result[f"band_{band_key}_internal_max_delta"] = internal_max_delta
        result[f"band_{band_key}_diagnostic_status"] = status
    return result, gap_rows


def _base_record(
    trial_row: dict[str, str],
    audit_row: dict[str, str],
) -> dict[str, object]:
    warnings = _split_flags(audit_row.get("metadata_warnings"))
    return {
        "selection_rank": trial_row.get("selection_rank", ""),
        "relative_path": trial_row.get("relative_path", ""),
        "file_name": trial_row.get("file_name", ""),
        "sample_id": trial_row.get("sample_id", ""),
        "filename_sample_id": trial_row.get("filename_sample_id", ""),
        "header_rruff_id": trial_row.get("header_rruff_id", ""),
        "reference_label": trial_row.get("reference_label", ""),
        "reference_label_status": "candidate_label_not_independently_validated",
        "header_raman_wavelength": trial_row.get(
            "header_raman_wavelength", ""
        ),
        "metadata_requires_review": bool(warnings),
        "audit_metadata_warnings": ";".join(warnings),
        "resolved_path": "",
        "file_size_bytes": "",
        "read_success": False,
        "v1_parser_success": False,
        "error_code": "",
        "error_message": "",
        "actual_point_count": "",
        "audit_point_count": audit_row.get("point_count", ""),
        "point_count_matches_audit": False,
        "actual_wavenumber_min": "",
        "audit_wavenumber_min": audit_row.get("wavenumber_min", ""),
        "wavenumber_min_matches_audit": False,
        "actual_wavenumber_max": "",
        "audit_wavenumber_max": audit_row.get("wavenumber_max", ""),
        "wavenumber_max_matches_audit": False,
        "audit_consistency_status": "not_checked_read_failure",
        "audit_mismatch_reasons": "actual_data_unavailable",
        "axis_values_finite": False,
        "intensity_values_finite": False,
        "all_values_finite": False,
        "wavenumber_strictly_increasing": False,
        "has_duplicate_wavenumber": False,
        "has_descending_wavenumber": False,
        "delta_min": "",
        "delta_median": "",
        "delta_p95": "",
        "delta_max": "",
        "delta_max_median_ratio": "",
        "gap_count_gt_3x_median": 0,
        "gap_count_gt_5x_median": 0,
    }


def _compare_with_audit(
    record: dict[str, object],
    audit_row: dict[str, str],
) -> None:
    mismatch_reasons: list[str] = []
    point_match = int(record["actual_point_count"]) == int(
        audit_row["point_count"]
    )
    min_match = _float_equal(
        str(record["actual_wavenumber_min"]),
        audit_row["wavenumber_min"],
    )
    max_match = _float_equal(
        str(record["actual_wavenumber_max"]),
        audit_row["wavenumber_max"],
    )
    if not point_match:
        mismatch_reasons.append("point_count_mismatch")
    if not min_match:
        mismatch_reasons.append("wavenumber_min_mismatch")
    if not max_match:
        mismatch_reasons.append("wavenumber_max_mismatch")
    record["point_count_matches_audit"] = point_match
    record["wavenumber_min_matches_audit"] = min_match
    record["wavenumber_max_matches_audit"] = max_match
    record["audit_consistency_status"] = (
        "match" if not mismatch_reasons else "mismatch"
    )
    record["audit_mismatch_reasons"] = ";".join(mismatch_reasons)


def inspect_trial_files(
    raw_root: Path,
    joined_rows: list[tuple[dict[str, str], dict[str, str]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Read only listed files, retaining failures while continuing the batch."""
    quality_records: list[dict[str, object]] = []
    gap_records: list[dict[str, object]] = []
    for trial_row, audit_row in joined_rows:
        record = _base_record(trial_row, audit_row)
        try:
            path = resolve_safe_raw_path(raw_root, trial_row["relative_path"])
            record["resolved_path"] = str(path)
            content = path.read_bytes()
            record["file_size_bytes"] = len(content)
            wavenumbers, intensities = _load_numeric_spectrum(content)
            try:
                parse_uploaded_spectrum(content)
                record["v1_parser_success"] = True
            except (TypeError, ValueError) as exc:
                record["v1_parser_success"] = False
                record["error_code"] = "v1_validation_failed"
                record["error_message"] = str(exc)
            diagnostics, gaps = inspect_arrays(
                wavenumbers,
                intensities,
                trial_row.get("selection_rank", ""),
                trial_row["relative_path"],
                trial_row.get("sample_id", ""),
            )
            record.update(diagnostics)
            record["read_success"] = True
            _compare_with_audit(record, audit_row)
            gap_records.extend(gaps)
        except UnsafeRelativePathError as exc:
            record["error_code"] = "unsafe_relative_path"
            record["error_message"] = str(exc)
        except FileNotFoundError as exc:
            record["error_code"] = "file_not_found"
            record["error_message"] = str(exc)
        except PermissionError as exc:
            record["error_code"] = "file_permission_error"
            record["error_message"] = str(exc)
        except (OSError, ValueError) as exc:
            record["error_code"] = "file_read_or_parse_error"
            record["error_message"] = str(exc)
        quality_records.append(record)
    return quality_records, gap_records


def _numeric_distribution(values: Iterable[object]) -> dict[str, object]:
    numeric = [float(value) for value in values if value not in ("", None)]
    if not numeric:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "p95": None,
            "maximum": None,
            "mean": None,
        }
    array = np.asarray(numeric, dtype=float)
    return {
        "count": len(numeric),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p95": _percentile(array, 0.95),
        "maximum": float(np.max(array)),
        "mean": fmean(numeric),
    }


def build_summary(
    raw_root: Path,
    trial_path: Path,
    audit_path: Path,
    quality_records: list[dict[str, object]],
    gap_records: list[dict[str, object]],
) -> dict[str, object]:
    """Summarize actual reads, audit reconciliation, deltas, and target bands."""
    successful = [record for record in quality_records if record["read_success"]]
    band_summary: dict[str, object] = {}
    for band_key, band_start, band_end in REFERENCE_BANDS:
        covered = [
            record
            for record in successful
            if record[f"band_{band_key}_endpoint_covered"]
        ]
        band_summary[band_key] = {
            "band_start_cm-1": band_start,
            "band_end_cm-1": band_end,
            "endpoint_covered_count": len(covered),
            "not_fully_covered_count": len(successful) - len(covered),
            "sample_point_count_distribution": _numeric_distribution(
                record[f"band_{band_key}_sample_point_count"]
                for record in successful
            ),
            "internal_max_delta_distribution": _numeric_distribution(
                record[f"band_{band_key}_internal_max_delta"]
                for record in successful
            ),
        }
    return {
        "raw_root": str(raw_root.resolve()),
        "source_trial_csv": str(trial_path.resolve()),
        "source_audit_csv": str(audit_path.resolve()),
        "trial_record_count": len(quality_records),
        "read_success_count": len(successful),
        "read_failure_count": len(quality_records) - len(successful),
        "v1_parser_success_count": sum(
            bool(record["v1_parser_success"]) for record in quality_records
        ),
        "all_values_finite_count": sum(
            bool(record["all_values_finite"]) for record in successful
        ),
        "strictly_increasing_count": sum(
            bool(record["wavenumber_strictly_increasing"])
            for record in successful
        ),
        "duplicate_wavenumber_count": sum(
            bool(record["has_duplicate_wavenumber"])
            for record in successful
        ),
        "audit_match_count": sum(
            record["audit_consistency_status"] == "match"
            for record in quality_records
        ),
        "audit_mismatch_count": sum(
            record["audit_consistency_status"] == "mismatch"
            for record in quality_records
        ),
        "metadata_requires_review_count": sum(
            bool(record["metadata_requires_review"])
            for record in quality_records
        ),
        "reference_candidate_count": sum(
            not bool(record["metadata_requires_review"])
            and not _is_missing(str(record["reference_label"]))
            and not _is_missing(str(record["filename_sample_id"]))
            and not _is_missing(str(record["header_rruff_id"]))
            for record in quality_records
        ),
        "delta_statistics_across_spectra": {
            "delta_min": _numeric_distribution(
                record["delta_min"] for record in successful
            ),
            "delta_median": _numeric_distribution(
                record["delta_median"] for record in successful
            ),
            "delta_p95": _numeric_distribution(
                record["delta_p95"] for record in successful
            ),
            "delta_max": _numeric_distribution(
                record["delta_max"] for record in successful
            ),
            "delta_max_median_ratio": _numeric_distribution(
                record["delta_max_median_ratio"] for record in successful
            ),
        },
        "gap_diagnostics": {
            "intervals_over_3x_median": len(gap_records),
            "intervals_over_5x_median": sum(
                bool(record["exceeds_5x_median"]) for record in gap_records
            ),
            "spectra_with_intervals_over_3x_median": len(
                {
                    str(record["relative_path"])
                    for record in gap_records
                }
            ),
            "threshold_note": (
                "The 3x and 5x median-delta multiples are diagnostic markers, "
                "not validated scientific quality acceptance thresholds."
            ),
        },
        "reference_band_sampling": band_summary,
        "coverage_note": (
            "Uncovered target bands are reported without interpolation, "
            "extrapolation, zero filling, or deletion."
        ),
        "reference_label_note": (
            "reference_label is traceability metadata and is not asserted as "
            "independently validated mineral ground truth."
        ),
        "unified_grid_status": "No unified Raman-shift grid is selected here.",
        "output_encoding": "utf-8-sig",
    }


def _output_paths(trial_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    base_name = trial_path.stem
    return (
        output_dir / f"{base_name}_sampling_quality.csv",
        output_dir / f"{base_name}_sampling_gap_details.csv",
        output_dir / f"{base_name}_sampling_summary.json",
    )


def write_outputs(
    trial_path: Path,
    output_dir: Path,
    quality_records: list[dict[str, object]],
    gap_records: list[dict[str, object]],
    summary: dict[str, object],
) -> tuple[Path, Path, Path]:
    """Write new UTF-8-SIG diagnostics and refuse to overwrite prior results."""
    paths = _output_paths(trial_path, output_dir)
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Sampling output already exists: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    quality_path, gaps_path, summary_path = paths
    with quality_path.open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=QUALITY_FIELDS)
        writer.writeheader()
        writer.writerows(quality_records)
    with gaps_path.open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=GAP_FIELDS)
        writer.writeheader()
        writer.writerows(gap_records)
    with summary_path.open("x", encoding="utf-8-sig", newline="") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    return paths


def run_inspection(
    raw_root: Path,
    trial_path: Path,
    audit_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    tuple[Path, Path, Path],
]:
    """Inspect only trial-listed files and create traceable sampling reports."""
    if not raw_root.is_dir():
        raise ValueError(f"raw_root is not a directory: {raw_root}")
    trial_rows = load_csv(trial_path, REQUIRED_TRIAL_COLUMNS, "Trial")
    audit_rows = load_csv(audit_path, REQUIRED_AUDIT_COLUMNS, "Audit")
    joined_rows = join_trial_with_audit(trial_rows, audit_rows)
    quality_records, gap_records = inspect_trial_files(raw_root, joined_rows)
    summary = build_summary(
        raw_root,
        trial_path,
        audit_path,
        quality_records,
        gap_records,
    )
    output_paths = write_outputs(
        trial_path,
        output_dir,
        quality_records,
        gap_records,
        summary,
    )
    return quality_records, gap_records, summary, output_paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect real Raman-shift sampling for only the trial-listed RAW "
            "spectra without preprocessing or interpolation."
        )
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()
    try:
        quality, gaps, summary, output_paths = run_inspection(
            arguments.raw_root,
            arguments.trial,
            arguments.audit,
            arguments.output_dir,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Trial records: {len(quality)}")
    print(f"Read success: {summary['read_success_count']}")
    print(f"Gap detail rows: {len(gaps)}")
    print(f"Sampling quality CSV: {output_paths[0]}")
    print(f"Gap details CSV: {output_paths[1]}")
    print(f"Summary JSON: {output_paths[2]}")


if __name__ == "__main__":
    main()
