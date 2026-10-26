"""Offline preprocessing of admitted RRUFF RAW spectra for retrieval experiments.

This experiment deliberately reuses the RamanAI V1 parser and preprocessing
functions.  It adds only a deterministic common-grid cache after the V1-style
processing sequence; it does not redefine dataset admission rules.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raman_app.upload import parse_uploaded_spectrum
from scripts.raman_utils import (
    baseline_correct_spectrum,
    denoise_spectrum,
    normalize_spectrum,
)


INTERPOLATION_SCRIPT = PROJECT_ROOT / "scripts" / "09_validate_interpolation.py"
GRID_START_CM_1 = 300.0
GRID_END_CM_1 = 850.0
GRID_STEP_CM_1 = 0.5
EXPECTED_GRID_POINT_COUNT = 1101

SG_WINDOW_LENGTH = 21
SG_POLYORDER = 3
ASLS_LAM = 1e6
ASLS_P = 0.01
NORMALIZATION_METHOD = "max"
NORMALIZATION_STAGE = "after_asls_on_original_axis_before_interpolation"
PROCESSING_ORDER = (
    "parse_uploaded_spectrum",
    "denoise_spectrum",
    "baseline_correct_spectrum",
    "normalize_spectrum",
    "linear_interpolate_no_extrapolation",
)

REQUIRED_MANIFEST_COLUMNS = {
    "relative_path",
    "file_name",
    "measurement_stem",
    "file_type",
    "is_raw",
    "is_processed",
    "preprocessing_candidate",
    "reference_candidate",
    "requires_review",
    "review_reasons",
    "reference_label",
    "reference_label_status",
    "filename_sample_id",
    "header_rruff_id",
    "filename_suspected_excitation_nm",
    "header_raman_wavelength",
    "audit_metadata_warnings",
    "has_missing_metadata",
    "has_conflict",
    "metadata_status",
    "metadata_flags",
}
REQUIRED_TRIAL_COLUMNS = {"relative_path"}

INDEX_FIELDS = [
    "input_order",
    "relative_path",
    "file_name",
    "measurement_stem",
    "sample_id",
    "sample_id_source",
    "filename_sample_id",
    "header_rruff_id",
    "reference_label",
    "reference_label_status",
    "reference_candidate",
    "excitation_wavelength_nm",
    "excitation_wavelength_source",
    "filename_suspected_excitation_nm",
    "header_raman_wavelength",
    "requires_review",
    "review_reasons",
    "audit_metadata_warnings",
    "has_missing_metadata",
    "has_conflict",
    "metadata_status",
    "metadata_flags",
    "processing_status",
    "failure_stage",
    "failure_code",
    "failure_message",
    "resolved_path",
    "file_size_bytes",
    "original_point_count",
    "original_wavenumber_min_cm-1",
    "original_wavenumber_max_cm-1",
    "corrected_dynamic_range",
    "normalized_original_axis_max",
    "grid_point_count",
    "grid_wavenumber_min_cm-1",
    "grid_wavenumber_max_cm-1",
    "grid_intensity_min",
    "grid_intensity_max",
    "cache_row_index",
    "cache_array_name",
    "sg_window_length_points",
    "sg_polyorder",
    "asls_lam",
    "asls_p",
    "normalization_method",
    "normalization_stage",
    "interpolation_method",
    "target_grid_start_cm-1",
    "target_grid_end_cm-1",
    "target_grid_step_cm-1",
]


def _load_interpolation_module():
    spec = importlib.util.spec_from_file_location(
        "raman_exp09_interpolation",
        INTERPOLATION_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load experiment 9.2-A interpolation module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


interpolation = _load_interpolation_module()


class UnsafeRelativePathError(ValueError):
    """Raised when a manifest path escapes the supplied RAW root."""


class ProcessingFailure(RuntimeError):
    """A stage-specific, recordable preprocessing failure."""

    def __init__(self, stage: str, code: str, message: str):
        super().__init__(message)
        self.stage = stage
        self.code = code


def _parse_bool(value: str | None, field_name: str) -> bool:
    normalized = (value or "").strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid boolean value for {field_name}: {value!r}")


def _is_missing(value: str | None) -> bool:
    return (value or "").strip().casefold() in {"", "unknown", "none", "n/a"}


def _load_csv(
    path: Path,
    required_columns: set[str],
    source_name: str,
) -> list[dict[str, str]]:
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


def _unique_by_relative_path(
    rows: Iterable[dict[str, str]],
    source_name: str,
) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    for row in rows:
        relative_path = row.get("relative_path", "").strip()
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


def load_candidate_records(
    manifest_path: Path,
    mode: str,
    trial_path: Path | None = None,
) -> tuple[list[dict[str, str]], int]:
    """Select only manifest-admitted RAW records, optionally by trial paths."""
    if mode not in {"trial", "full"}:
        raise ValueError("mode must be 'trial' or 'full'")
    manifest_rows = _load_csv(
        manifest_path,
        REQUIRED_MANIFEST_COLUMNS,
        "Candidate manifest",
    )
    manifest_by_path = _unique_by_relative_path(manifest_rows, "Candidate manifest")
    candidate_by_path: dict[str, dict[str, str]] = {}
    for relative_path, row in manifest_by_path.items():
        is_candidate = _parse_bool(
            row.get("preprocessing_candidate"),
            "preprocessing_candidate",
        )
        is_raw = _parse_bool(row.get("is_raw"), "is_raw")
        is_processed = _parse_bool(row.get("is_processed"), "is_processed")
        file_type = row.get("file_type", "").strip().casefold()
        if is_candidate:
            if not is_raw or is_processed or file_type != "raw":
                raise ValueError(
                    "Candidate manifest contains an inconsistent admitted record: "
                    + relative_path
                )
            candidate_by_path[relative_path] = row

    if mode == "full":
        selected = [
            candidate_by_path[path]
            for path in sorted(candidate_by_path, key=str.casefold)
        ]
        return selected, len(manifest_rows)

    if trial_path is None:
        raise ValueError("--trial is required in trial mode")
    trial_rows = _load_csv(trial_path, REQUIRED_TRIAL_COLUMNS, "Trial")
    _unique_by_relative_path(trial_rows, "Trial")
    selected: list[dict[str, str]] = []
    missing_or_ineligible: list[str] = []
    for trial_row in trial_rows:
        relative_path = trial_row["relative_path"].strip()
        candidate = candidate_by_path.get(relative_path)
        if candidate is None:
            missing_or_ineligible.append(relative_path)
        else:
            selected.append(candidate)
    if missing_or_ineligible:
        raise ValueError(
            "Trial records are absent from admitted RAW candidates: "
            + ", ".join(missing_or_ineligible[:3])
        )
    return selected, len(manifest_rows)


def make_trial_grid() -> np.ndarray:
    """Return the fixed engineering trial grid of exactly 1,101 points."""
    grid = interpolation.make_target_grid(
        GRID_START_CM_1,
        GRID_END_CM_1,
        GRID_STEP_CM_1,
    )
    if len(grid) != EXPECTED_GRID_POINT_COUNT:
        raise RuntimeError(
            f"Unexpected grid size: {len(grid)} != {EXPECTED_GRID_POINT_COUNT}"
        )
    return grid


def resolve_safe_path(raw_root: Path, relative_path: str) -> Path:
    """Resolve one manifest path without allowing traversal outside raw_root."""
    candidate_relative = Path(relative_path)
    if candidate_relative.is_absolute():
        raise UnsafeRelativePathError("absolute_path_not_allowed")
    root_resolved = raw_root.resolve()
    candidate = (root_resolved / candidate_relative).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise UnsafeRelativePathError("path_escapes_raw_root") from exc
    return candidate


def _derive_sample_id(row: dict[str, str]) -> tuple[str, str]:
    filename_value = row.get("filename_sample_id", "").strip()
    header_value = row.get("header_rruff_id", "").strip()
    filename_known = not _is_missing(filename_value)
    header_known = not _is_missing(header_value)
    if filename_known and header_known:
        if filename_value.casefold() == header_value.casefold():
            return header_value, "header_and_filename_agree"
        return "Unknown", "filename_header_conflict_unresolved"
    if header_known:
        return header_value, "header_rruff_id"
    if filename_known:
        return filename_value, "filename_sample_id"
    return "Unknown", "missing"


def _derive_excitation(row: dict[str, str]) -> tuple[str, str]:
    filename_value = row.get("filename_suspected_excitation_nm", "").strip()
    header_value = row.get("header_raman_wavelength", "").strip()
    filename_known = not _is_missing(filename_value)
    header_known = not _is_missing(header_value)
    if header_known and filename_known:
        try:
            values_agree = math.isclose(
                float(header_value),
                float(filename_value),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        except ValueError:
            values_agree = header_value.casefold() == filename_value.casefold()
        if values_agree:
            return header_value, "header_and_filename_agree"
        return "Unknown", "filename_header_conflict_unresolved"
    if header_known:
        return header_value, "header_raman_wavelength"
    if filename_known:
        return filename_value, "filename_suspected_excitation_nm"
    return "Unknown", "missing"


def _base_index_record(
    manifest_row: dict[str, str],
    input_order: int,
) -> dict[str, object]:
    sample_id, sample_source = _derive_sample_id(manifest_row)
    excitation, excitation_source = _derive_excitation(manifest_row)
    return {
        "input_order": input_order,
        "relative_path": manifest_row.get("relative_path", ""),
        "file_name": manifest_row.get("file_name", ""),
        "measurement_stem": manifest_row.get("measurement_stem", ""),
        "sample_id": sample_id,
        "sample_id_source": sample_source,
        "filename_sample_id": manifest_row.get("filename_sample_id", ""),
        "header_rruff_id": manifest_row.get("header_rruff_id", ""),
        "reference_label": manifest_row.get("reference_label", ""),
        "reference_label_status": manifest_row.get("reference_label_status", ""),
        "reference_candidate": manifest_row.get("reference_candidate", ""),
        "excitation_wavelength_nm": excitation,
        "excitation_wavelength_source": excitation_source,
        "filename_suspected_excitation_nm": manifest_row.get(
            "filename_suspected_excitation_nm", ""
        ),
        "header_raman_wavelength": manifest_row.get(
            "header_raman_wavelength", ""
        ),
        "requires_review": manifest_row.get("requires_review", ""),
        "review_reasons": manifest_row.get("review_reasons", ""),
        "audit_metadata_warnings": manifest_row.get(
            "audit_metadata_warnings", ""
        ),
        "has_missing_metadata": manifest_row.get("has_missing_metadata", ""),
        "has_conflict": manifest_row.get("has_conflict", ""),
        "metadata_status": manifest_row.get("metadata_status", ""),
        "metadata_flags": manifest_row.get("metadata_flags", ""),
        "processing_status": "failed",
        "failure_stage": "not_started",
        "failure_code": "not_started",
        "failure_message": "",
        "resolved_path": "",
        "file_size_bytes": "",
        "original_point_count": "",
        "original_wavenumber_min_cm-1": "",
        "original_wavenumber_max_cm-1": "",
        "corrected_dynamic_range": "",
        "normalized_original_axis_max": "",
        "grid_point_count": "",
        "grid_wavenumber_min_cm-1": GRID_START_CM_1,
        "grid_wavenumber_max_cm-1": GRID_END_CM_1,
        "grid_intensity_min": "",
        "grid_intensity_max": "",
        "cache_row_index": "",
        "cache_array_name": "",
        "sg_window_length_points": SG_WINDOW_LENGTH,
        "sg_polyorder": SG_POLYORDER,
        "asls_lam": ASLS_LAM,
        "asls_p": ASLS_P,
        "normalization_method": NORMALIZATION_METHOD,
        "normalization_stage": NORMALIZATION_STAGE,
        "interpolation_method": "numpy_linear_no_extrapolation",
        "target_grid_start_cm-1": GRID_START_CM_1,
        "target_grid_end_cm-1": GRID_END_CM_1,
        "target_grid_step_cm-1": GRID_STEP_CM_1,
    }


def _run_v1_preprocessing(content: bytes):
    try:
        raw = parse_uploaded_spectrum(content)
    except Exception as exc:
        raise ProcessingFailure("parse", "upload_parse_error", str(exc)) from exc
    try:
        denoised = denoise_spectrum(
            raw,
            window_length=SG_WINDOW_LENGTH,
            polyorder=SG_POLYORDER,
        )
    except Exception as exc:
        raise ProcessingFailure("sg", "sg_processing_error", str(exc)) from exc
    try:
        corrected = baseline_correct_spectrum(
            denoised,
            lam=ASLS_LAM,
            p=ASLS_P,
        )
    except Exception as exc:
        raise ProcessingFailure(
            "asls", "asls_processing_error", str(exc)
        ) from exc

    corrected_y = np.asarray(corrected.spectral_data, dtype=float)
    if not np.all(np.isfinite(corrected_y)):
        raise ProcessingFailure(
            "post_asls_validation",
            "nonfinite_corrected_values",
            "AsLS output contains non-finite intensities",
        )
    dynamic_range = float(np.max(corrected_y) - np.min(corrected_y))
    if not math.isfinite(dynamic_range) or dynamic_range <= 0:
        raise ProcessingFailure(
            "post_asls_validation",
            "invalid_corrected_dynamic_range",
            "AsLS output has a non-positive or non-finite dynamic range",
        )
    try:
        normalized = normalize_spectrum(corrected, method=NORMALIZATION_METHOD)
    except Exception as exc:
        raise ProcessingFailure(
            "normalization", "normalization_error", str(exc)
        ) from exc
    return raw, corrected, normalized, dynamic_range


def process_candidate(
    raw_root: Path,
    manifest_row: dict[str, str],
    input_order: int,
    target_grid: np.ndarray,
) -> tuple[dict[str, object], np.ndarray | None]:
    """Process one admitted RAW while retaining failures as index records."""
    record = _base_index_record(manifest_row, input_order)
    try:
        try:
            path = resolve_safe_path(raw_root, manifest_row["relative_path"])
        except UnsafeRelativePathError as exc:
            raise ProcessingFailure("path", "unsafe_relative_path", str(exc)) from exc
        record["resolved_path"] = str(path)
        if not path.is_file():
            raise ProcessingFailure("read", "file_missing", "Source file does not exist")
        try:
            record["file_size_bytes"] = path.stat().st_size
            content = path.read_bytes()
        except OSError as exc:
            raise ProcessingFailure(
                "read", f"file_read_error:{type(exc).__name__}", str(exc)
            ) from exc

        raw, _corrected, normalized, dynamic_range = _run_v1_preprocessing(content)
        raw_x = np.asarray(raw.spectral_axis, dtype=float)
        normalized_y = np.asarray(normalized.spectral_data, dtype=float)
        record.update(
            {
                "original_point_count": len(raw_x),
                "original_wavenumber_min_cm-1": float(raw_x[0]),
                "original_wavenumber_max_cm-1": float(raw_x[-1]),
                "corrected_dynamic_range": dynamic_range,
                "normalized_original_axis_max": float(np.max(normalized_y)),
            }
        )
        try:
            interpolation.validate_spectrum(
                raw_x,
                normalized_y,
                GRID_START_CM_1,
                GRID_END_CM_1,
            )
        except ValueError as exc:
            code = (
                "target_band_incomplete"
                if "does not cover the target band" in str(exc)
                else "pre_interpolation_validation_error"
            )
            raise ProcessingFailure("interpolation_validation", code, str(exc)) from exc
        try:
            grid_y = interpolation.linear_interpolate_no_extrapolation(
                raw_x,
                normalized_y,
                target_grid,
            )
        except ValueError as exc:
            raise ProcessingFailure(
                "interpolation", "linear_interpolation_error", str(exc)
            ) from exc
        grid_y = np.asarray(grid_y, dtype=np.float64)
        if grid_y.shape != target_grid.shape or not np.all(np.isfinite(grid_y)):
            raise ProcessingFailure(
                "post_interpolation_validation",
                "invalid_grid_output",
                "Interpolated output has the wrong shape or non-finite values",
            )
        record.update(
            {
                "processing_status": "success",
                "failure_stage": "",
                "failure_code": "",
                "failure_message": "",
                "grid_point_count": len(target_grid),
                "grid_intensity_min": float(np.min(grid_y)),
                "grid_intensity_max": float(np.max(grid_y)),
                "cache_array_name": "normalized_spectra",
            }
        )
        return record, grid_y
    except ProcessingFailure as exc:
        record.update(
            {
                "processing_status": "failed",
                "failure_stage": exc.stage,
                "failure_code": exc.code,
                "failure_message": str(exc),
            }
        )
        return record, None
    except Exception as exc:
        record.update(
            {
                "processing_status": "failed",
                "failure_stage": "unexpected_internal_error",
                "failure_code": f"unexpected:{type(exc).__name__}",
                "failure_message": str(exc),
            }
        )
        return record, None


def process_records(
    raw_root: Path,
    selected_rows: list[dict[str, str]],
    progress_every: int = 100,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    """Process selected records sequentially with per-file failure isolation."""
    target_grid = make_trial_grid()
    index_records: list[dict[str, object]] = []
    spectra: list[np.ndarray] = []
    for input_order, manifest_row in enumerate(selected_rows, start=1):
        record, grid_y = process_candidate(
            raw_root,
            manifest_row,
            input_order,
            target_grid,
        )
        if grid_y is not None:
            record["cache_row_index"] = len(spectra)
            spectra.append(grid_y)
        index_records.append(record)
        if progress_every > 0 and (
            input_order % progress_every == 0 or input_order == len(selected_rows)
        ):
            success_count = len(spectra)
            print(
                f"Processed {input_order}/{len(selected_rows)} "
                f"(success={success_count}, failed={input_order - success_count})",
                flush=True,
            )
    matrix = (
        np.vstack(spectra).astype(np.float64, copy=False)
        if spectra
        else np.empty((0, len(target_grid)), dtype=np.float64)
    )
    return index_records, target_grid, matrix


def build_summary(
    manifest_path: Path,
    trial_path: Path | None,
    raw_root: Path,
    output_dir: Path,
    mode: str,
    manifest_record_count: int,
    index_records: list[dict[str, object]],
    target_grid: np.ndarray,
    matrix: np.ndarray,
) -> dict[str, object]:
    successes = [r for r in index_records if r["processing_status"] == "success"]
    failures = [r for r in index_records if r["processing_status"] == "failed"]
    failure_counts = Counter(str(r["failure_code"]) for r in failures)
    review_count = sum(
        _parse_bool(str(record["requires_review"]), "requires_review")
        for record in index_records
    )
    return {
        "experiment": "9.3",
        "mode": mode,
        "candidate_manifest": str(manifest_path),
        "trial_manifest": str(trial_path) if trial_path is not None else None,
        "raw_root": str(raw_root),
        "output_directory": str(output_dir),
        "candidate_manifest_record_count": manifest_record_count,
        "selected_admitted_raw_count": len(index_records),
        "processing_success_count": len(successes),
        "processing_failure_count": len(failures),
        "failure_reason_counts": dict(sorted(failure_counts.items())),
        "metadata_requires_review_count": review_count,
        "reference_label_note": (
            "reference_label is retained from the metadata index and is not an "
            "independently validated mineral ground truth."
        ),
        "engineering_grid_note": (
            "The 300-850 cm-1, 0.5 cm-1 grid is an experiment 9.3 engineering "
            "prototype and is not a final scientific grid decision."
        ),
        "processing_order": list(PROCESSING_ORDER),
        "processing_order_note": (
            "SG, AsLS, and max normalization are applied on the original valid "
            "sampling axis. The normalized spectrum is then linearly interpolated "
            "without extrapolation."
        ),
        "v1_equivalence_note": (
            "The parser and SG/AsLS/max-normalization functions and parameters are "
            "the V1 implementations. Common-grid interpolation is an added offline "
            "experiment step and is not part of the V1 upload pipeline."
        ),
        "sg_physical_width_note": (
            "The SG window length is 21 sample points; its physical width in cm-1 "
            "therefore varies with each original spectrum's sampling interval."
        ),
        "parameters": {
            "sg": {"window_length_points": SG_WINDOW_LENGTH, "polyorder": SG_POLYORDER},
            "asls": {"lam": ASLS_LAM, "p": ASLS_P},
            "normalization": {
                "method": NORMALIZATION_METHOD,
                "stage": NORMALIZATION_STAGE,
            },
            "interpolation": {
                "method": "linear_no_extrapolation",
                "start_cm-1": GRID_START_CM_1,
                "end_cm-1": GRID_END_CM_1,
                "step_cm-1": GRID_STEP_CM_1,
                "point_count": len(target_grid),
            },
        },
        "called_apis": {
            "parser": "raman_app.upload.parse_uploaded_spectrum",
            "sg": "scripts.raman_utils.denoise_spectrum",
            "asls": "scripts.raman_utils.baseline_correct_spectrum",
            "normalization": "scripts.raman_utils.normalize_spectrum",
            "interpolation": (
                "scripts/09_validate_interpolation.py:"
                "linear_interpolate_no_extrapolation"
            ),
        },
        "cache": {
            "format": "numpy_npz_compressed",
            "allow_pickle_required": False,
            "normalized_spectra_shape": list(matrix.shape),
            "normalized_spectra_dtype": str(matrix.dtype),
            "wavenumber_grid_shape": list(target_grid.shape),
            "relative_paths_count": len(successes),
            "index_mapping": "index CSV cache_row_index -> NPZ normalized_spectra row",
        },
    }


def _output_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    return (
        output_dir / "rruff_preprocessed_300_850_step0p5.npz",
        output_dir / "rruff_preprocessing_index.csv",
        output_dir / "rruff_preprocessing_summary.json",
    )


def write_outputs(
    output_dir: Path,
    index_records: list[dict[str, object]],
    target_grid: np.ndarray,
    matrix: np.ndarray,
    summary: dict[str, object],
) -> tuple[Path, Path, Path]:
    """Write a machine cache and traceability reports without overwriting."""
    paths = _output_paths(output_dir)
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Batch preprocessing output already exists: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path, index_path, summary_path = paths
    successful_records = [
        record for record in index_records if record["processing_status"] == "success"
    ]
    relative_paths = np.asarray(
        [str(record["relative_path"]) for record in successful_records],
        dtype=str,
    )
    measurement_stems = np.asarray(
        [str(record["measurement_stem"]) for record in successful_records],
        dtype=str,
    )
    np.savez_compressed(
        cache_path,
        wavenumber_grid_cm_1=target_grid.astype(np.float64, copy=False),
        normalized_spectra=matrix.astype(np.float64, copy=False),
        relative_paths=relative_paths,
        measurement_stems=measurement_stems,
    )
    with index_path.open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(index_records)
    summary["cache_validation"] = validate_cache(cache_path, index_records)
    with summary_path.open("x", encoding="utf-8-sig", newline="") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    return paths


def validate_cache(
    cache_path: Path,
    index_records: list[dict[str, object]],
) -> dict[str, object]:
    """Reload the NPZ without pickle and verify index-to-array identity mapping."""
    successful = [
        record for record in index_records if record["processing_status"] == "success"
    ]
    with np.load(cache_path, allow_pickle=False) as cache:
        required = {
            "wavenumber_grid_cm_1",
            "normalized_spectra",
            "relative_paths",
            "measurement_stems",
        }
        missing = sorted(required - set(cache.files))
        if missing:
            raise ValueError("Cache is missing arrays: " + ", ".join(missing))
        grid = cache["wavenumber_grid_cm_1"]
        matrix = cache["normalized_spectra"]
        paths = cache["relative_paths"]
        stems = cache["measurement_stems"]
        if grid.shape != (EXPECTED_GRID_POINT_COUNT,):
            raise ValueError("Cache grid shape is invalid")
        if matrix.shape != (len(successful), EXPECTED_GRID_POINT_COUNT):
            raise ValueError("Cache matrix shape does not match successful index rows")
        expected_paths = np.asarray(
            [str(record["relative_path"]) for record in successful], dtype=str
        )
        expected_stems = np.asarray(
            [str(record["measurement_stem"]) for record in successful], dtype=str
        )
        if not np.array_equal(paths, expected_paths):
            raise ValueError("Cache relative_paths do not match index order")
        if not np.array_equal(stems, expected_stems):
            raise ValueError("Cache measurement_stems do not match index order")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Cache contains non-finite spectra")
        if not np.array_equal(grid, make_trial_grid()):
            raise ValueError("Cache grid values do not match experiment configuration")
        return {
            "grid_shape": list(grid.shape),
            "matrix_shape": list(matrix.shape),
            "relative_path_count": len(paths),
            "finite": True,
            "identity_mapping_verified": True,
        }


def run_batch(
    raw_root: Path,
    manifest_path: Path,
    output_dir: Path,
    mode: str,
    trial_path: Path | None = None,
    progress_every: int = 100,
) -> tuple[list[dict[str, object]], dict[str, object], tuple[Path, Path, Path]]:
    selected_rows, manifest_count = load_candidate_records(
        manifest_path,
        mode,
        trial_path,
    )
    index_records, target_grid, matrix = process_records(
        raw_root,
        selected_rows,
        progress_every=progress_every,
    )
    summary = build_summary(
        manifest_path,
        trial_path,
        raw_root,
        output_dir,
        mode,
        manifest_count,
        index_records,
        target_grid,
        matrix,
    )
    paths = write_outputs(output_dir, index_records, target_grid, matrix, summary)
    return index_records, summary, paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess admitted RRUFF RAW spectra on their original axes and "
            "cache normalized spectra on the experiment 9.3 engineering grid."
        )
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("trial", "full"), required=True)
    parser.add_argument("--trial", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()
    if arguments.mode == "trial" and arguments.trial is None:
        parser.error("--trial is required when --mode trial")
    if arguments.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    try:
        _records, summary, paths = run_batch(
            raw_root=arguments.raw_root,
            manifest_path=arguments.manifest,
            output_dir=arguments.output_dir,
            mode=arguments.mode,
            trial_path=arguments.trial,
            progress_every=arguments.progress_every,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Selected admitted RAW: {summary['selected_admitted_raw_count']}")
    print(f"Success: {summary['processing_success_count']}")
    print(f"Failed: {summary['processing_failure_count']}")
    print(f"Cache: {paths[0]}")
    print(f"Index: {paths[1]}")
    print(f"Summary: {paths[2]}")


if __name__ == "__main__":
    main()
