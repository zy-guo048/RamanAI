"""Select traceable RAW candidates from existing RRUFF audit indexes."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Iterable


FILE_TYPE_RAW = "RAW"
FILE_TYPE_PROCESSED = "Processed"
FILE_TYPE_UNKNOWN = "Unknown"

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "exp08_rruff_candidates"
)

REQUIRED_AUDIT_COLUMNS = {
    "file_name",
    "relative_path",
    "file_type",
    "filename_type",
    "header_file_type",
    "mineral",
    "sample_id",
    "suspected_excitation_nm",
    "header_names",
    "header_rruff_id",
    "header_raman_wavelength",
    "format_parse_success",
    "data_column_count",
    "nonfinite_value_count",
    "has_duplicate_wavenumber",
    "wavenumber_strictly_increasing",
    "v1_upload_compatible",
    "parse_errors",
    "v1_incompatibility_reasons",
    "metadata_warnings",
}

REQUIRED_INDEX_COLUMNS = {
    "relative_path",
    "file_name",
    "file_type",
    "filename_type",
    "header_file_type",
    "measurement_stem",
    "filename_mineral",
    "reference_label",
    "filename_sample_id",
    "header_rruff_id",
    "filename_suspected_excitation_nm",
    "header_raman_wavelength",
    "format_parse_success",
    "v1_upload_compatible",
    "audit_metadata_warnings",
    "has_missing_metadata",
    "has_conflict",
    "needs_manual_review",
    "metadata_status",
    "metadata_flags",
}

OUTPUT_FIELDS = [
    "relative_path",
    "file_name",
    "measurement_stem",
    "file_type",
    "filename_type",
    "header_file_type",
    "is_raw",
    "is_processed",
    "sample_unit_role",
    "spectral_quality_review_reasons",
    "preprocessing_candidate",
    "preprocessing_rejection_reasons",
    "reference_candidate",
    "reference_rejection_reasons",
    "requires_review",
    "review_reasons",
    "reference_label",
    "reference_label_status",
    "filename_mineral",
    "filename_sample_id",
    "header_rruff_id",
    "filename_suspected_excitation_nm",
    "header_raman_wavelength",
    "format_parse_success",
    "data_column_count",
    "nonfinite_value_count",
    "wavenumber_strictly_increasing",
    "v1_upload_compatible",
    "parse_errors",
    "v1_incompatibility_reasons",
    "audit_metadata_warnings",
    "has_missing_metadata",
    "has_conflict",
    "index_requires_manual_review",
    "metadata_status",
    "metadata_flags",
]


def _canonical_file_type(value: str | None) -> str:
    normalized = (value or "").strip().casefold()
    if normalized == "raw":
        return FILE_TYPE_RAW
    if normalized == "processed":
        return FILE_TYPE_PROCESSED
    return FILE_TYPE_UNKNOWN


def _parse_bool(value: str | None) -> bool | None:
    normalized = (value or "").strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _is_missing(value: str | None) -> bool:
    if value is None:
        return True
    return value.strip().casefold() in {"", "unknown", "none", "n/a"}


def _split_reasons(value: str | None) -> list[str]:
    return [part.strip() for part in (value or "").split(";") if part.strip()]


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return sorted(set(values), key=lambda value: (value.casefold(), value))


def load_csv(
    path: Path,
    required_columns: set[str],
    source_name: str,
) -> list[dict[str, str]]:
    """Load a UTF-8-compatible CSV and validate its required columns."""
    if not path.is_file():
        raise ValueError(f"{source_name} CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"{source_name} CSV has no header row")
        missing_columns = sorted(required_columns - set(reader.fieldnames))
        if missing_columns:
            raise ValueError(
                f"{source_name} CSV is missing required columns: "
                + ", ".join(missing_columns)
            )
        return [dict(row) for row in reader]


def _rows_by_relative_path(
    rows: list[dict[str, str]],
    source_name: str,
) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    duplicate_paths: list[str] = []
    for row in rows:
        relative_path = row.get("relative_path", "")
        if not relative_path:
            raise ValueError(f"{source_name} CSV contains an empty relative_path")
        if relative_path in indexed:
            duplicate_paths.append(relative_path)
        else:
            indexed[relative_path] = row
    if duplicate_paths:
        examples = ", ".join(_ordered_unique(duplicate_paths)[:3])
        raise ValueError(
            f"{source_name} CSV contains duplicate relative_path values: {examples}"
        )
    return indexed


def join_inputs(
    audit_rows: list[dict[str, str]],
    index_rows: list[dict[str, str]],
) -> list[tuple[dict[str, str], dict[str, str]]]:
    """Join one-to-one records by exact relative_path and validate identity."""
    audit_by_path = _rows_by_relative_path(audit_rows, "Audit")
    index_by_path = _rows_by_relative_path(index_rows, "Index")
    audit_paths = set(audit_by_path)
    index_paths = set(index_by_path)
    if audit_paths != index_paths:
        audit_only = sorted(
            audit_paths - index_paths,
            key=lambda value: (value.casefold(), value),
        )
        index_only = sorted(
            index_paths - audit_paths,
            key=lambda value: (value.casefold(), value),
        )
        raise ValueError(
            "Input record sets do not match by relative_path: "
            f"audit_only={len(audit_only)} {audit_only[:3]}, "
            f"index_only={len(index_only)} {index_only[:3]}"
        )

    joined: list[tuple[dict[str, str], dict[str, str]]] = []
    corresponding_fields = (
        ("file_name", "file_name"),
        ("file_type", "file_type"),
        ("filename_type", "filename_type"),
        ("header_file_type", "header_file_type"),
        ("mineral", "filename_mineral"),
        ("sample_id", "filename_sample_id"),
        ("header_names", "reference_label"),
        ("header_rruff_id", "header_rruff_id"),
        (
            "suspected_excitation_nm",
            "filename_suspected_excitation_nm",
        ),
        ("header_raman_wavelength", "header_raman_wavelength"),
        ("format_parse_success", "format_parse_success"),
        ("v1_upload_compatible", "v1_upload_compatible"),
    )
    for relative_path in sorted(
        audit_paths,
        key=lambda value: (value.casefold(), value),
    ):
        audit_row = audit_by_path[relative_path]
        index_row = index_by_path[relative_path]
        mismatched_fields = [
            f"{audit_field}/{index_field}"
            for audit_field, index_field in corresponding_fields
            if audit_row.get(audit_field, "")
            != index_row.get(index_field, "")
        ]
        audit_warnings = set(
            _split_reasons(audit_row.get("metadata_warnings"))
        )
        index_audit_warnings = set(
            _split_reasons(index_row.get("audit_metadata_warnings"))
        )
        if audit_warnings != index_audit_warnings:
            mismatched_fields.append(
                "metadata_warnings/audit_metadata_warnings"
            )
        index_metadata_flags = set(
            _split_reasons(index_row.get("metadata_flags"))
        )
        if not audit_warnings.issubset(index_metadata_flags):
            mismatched_fields.append("metadata_warnings/metadata_flags")
        if (
            any("conflict" in warning for warning in audit_warnings)
            and _parse_bool(index_row.get("has_conflict")) is not True
        ):
            mismatched_fields.append("metadata_warnings/has_conflict")
        if mismatched_fields:
            raise ValueError(
                "Joined records disagree for relative_path "
                f"{relative_path}: {', '.join(mismatched_fields)}"
            )
        joined.append((audit_row, index_row))
    return joined


def _spectral_quality_reasons(audit_row: dict[str, str]) -> list[str]:
    reasons: list[str] = []
    if _parse_bool(audit_row.get("format_parse_success")) is not True:
        reasons.append("format_parse_failed")
    if audit_row.get("data_column_count", "").strip() != "2":
        reasons.append("data_column_count_not_two")
    nonfinite_text = audit_row.get("nonfinite_value_count", "").strip()
    if not nonfinite_text:
        reasons.append("nonfinite_value_count_missing")
    else:
        try:
            nonfinite_count = int(nonfinite_text)
        except ValueError:
            nonfinite_count = None
            reasons.append("nonfinite_value_count_invalid")
        if nonfinite_count is not None and nonfinite_count < 0:
            reasons.append("nonfinite_value_count_negative")
        elif nonfinite_count is not None and nonfinite_count > 0:
            reasons.append("nonfinite_values")
    duplicate_flag = _parse_bool(audit_row.get("has_duplicate_wavenumber"))
    if duplicate_flag is True:
        reasons.append("duplicate_wavenumber")
    elif duplicate_flag is None:
        reasons.append("duplicate_wavenumber_flag_invalid")
    if _parse_bool(audit_row.get("wavenumber_strictly_increasing")) is not True:
        reasons.append("wavenumber_not_strictly_increasing")
    if _parse_bool(audit_row.get("v1_upload_compatible")) is not True:
        reasons.append("v1_static_compatibility_failed")
    return _ordered_unique(reasons)


def _preprocessing_decision(
    audit_row: dict[str, str],
    is_raw: bool,
) -> tuple[bool, list[str]]:
    if not is_raw:
        return False, ["not_raw_file"]

    reasons = _spectral_quality_reasons(audit_row)
    return not reasons, reasons


def _reference_decision(
    index_row: dict[str, str],
    is_raw: bool,
    preprocessing_candidate: bool,
) -> tuple[bool, list[str]]:
    if not is_raw:
        return False, ["not_raw_file"]

    reasons: list[str] = []
    if not preprocessing_candidate:
        reasons.append("not_preprocessing_candidate")
    if _is_missing(index_row.get("reference_label")):
        reasons.append("reference_label_missing")
    if _is_missing(index_row.get("filename_sample_id")):
        reasons.append("filename_sample_id_missing")
    if _is_missing(index_row.get("header_rruff_id")):
        reasons.append("header_sample_id_missing")
    if _parse_bool(index_row.get("has_conflict")) is not False:
        reasons.append("metadata_conflict_or_invalid_flag")
    if _parse_bool(index_row.get("has_missing_metadata")) is not False:
        reasons.append("metadata_missing_or_invalid_flag")
    if _parse_bool(index_row.get("needs_manual_review")) is not False:
        reasons.append("metadata_requires_review")
    if _split_reasons(index_row.get("audit_metadata_warnings")):
        reasons.append("audit_metadata_warning_present")
    if _split_reasons(index_row.get("metadata_flags")):
        reasons.append("metadata_flag_present")
    return not reasons, _ordered_unique(reasons)


def build_candidate_manifest(
    joined_rows: list[tuple[dict[str, str], dict[str, str]]],
) -> list[dict[str, object]]:
    """Build a complete manifest without deleting ineligible records."""
    records: list[dict[str, object]] = []
    for audit_row, index_row in joined_rows:
        file_type = _canonical_file_type(audit_row.get("file_type"))
        is_raw = file_type == FILE_TYPE_RAW
        is_processed = file_type == FILE_TYPE_PROCESSED
        spectral_quality_reasons = _spectral_quality_reasons(audit_row)
        preprocessing_candidate, preprocessing_reasons = (
            _preprocessing_decision(audit_row, is_raw)
        )
        reference_candidate, reference_reasons = _reference_decision(
            index_row,
            is_raw,
            preprocessing_candidate,
        )

        metadata_flags = _split_reasons(index_row.get("metadata_flags"))
        review_reasons = list(metadata_flags)
        if is_raw:
            review_reasons.extend(preprocessing_reasons)
            review_reasons.extend(reference_reasons)
        elif is_processed:
            review_reasons.extend(spectral_quality_reasons)
            if _parse_bool(index_row.get("needs_manual_review")) is not False:
                review_reasons.append("metadata_requires_review")
        if file_type == FILE_TYPE_UNKNOWN:
            review_reasons.append("file_type_unknown")
        review_reasons = [
            reason
            for reason in review_reasons
            if reason != "not_raw_file"
        ]
        review_reasons = _ordered_unique(review_reasons)

        if is_raw:
            sample_unit_role = "raw_primary_measurement"
        elif is_processed:
            sample_unit_role = "processed_derivative_not_independent_sample"
        else:
            sample_unit_role = "unclassified_not_independent_sample"

        reference_label = index_row.get("reference_label", "")
        reference_label_status = (
            "candidate_reference_label_not_independently_validated"
            if not _is_missing(reference_label)
            else "reference_label_missing"
        )
        records.append(
            {
                "relative_path": audit_row.get("relative_path", ""),
                "file_name": audit_row.get("file_name", ""),
                "measurement_stem": index_row.get("measurement_stem", ""),
                "file_type": audit_row.get("file_type", ""),
                "filename_type": audit_row.get("filename_type", ""),
                "header_file_type": audit_row.get("header_file_type", ""),
                "is_raw": is_raw,
                "is_processed": is_processed,
                "sample_unit_role": sample_unit_role,
                "spectral_quality_review_reasons": ";".join(
                    spectral_quality_reasons
                ),
                "preprocessing_candidate": preprocessing_candidate,
                "preprocessing_rejection_reasons": ";".join(
                    preprocessing_reasons
                ),
                "reference_candidate": reference_candidate,
                "reference_rejection_reasons": ";".join(reference_reasons),
                "requires_review": bool(review_reasons),
                "review_reasons": ";".join(review_reasons),
                "reference_label": reference_label,
                "reference_label_status": reference_label_status,
                "filename_mineral": index_row.get("filename_mineral", ""),
                "filename_sample_id": index_row.get("filename_sample_id", ""),
                "header_rruff_id": index_row.get("header_rruff_id", ""),
                "filename_suspected_excitation_nm": index_row.get(
                    "filename_suspected_excitation_nm", ""
                ),
                "header_raman_wavelength": index_row.get(
                    "header_raman_wavelength", ""
                ),
                "format_parse_success": audit_row.get(
                    "format_parse_success", ""
                ),
                "data_column_count": audit_row.get("data_column_count", ""),
                "nonfinite_value_count": audit_row.get(
                    "nonfinite_value_count", ""
                ),
                "wavenumber_strictly_increasing": audit_row.get(
                    "wavenumber_strictly_increasing", ""
                ),
                "v1_upload_compatible": audit_row.get(
                    "v1_upload_compatible", ""
                ),
                "parse_errors": audit_row.get("parse_errors", ""),
                "v1_incompatibility_reasons": audit_row.get(
                    "v1_incompatibility_reasons", ""
                ),
                "audit_metadata_warnings": audit_row.get(
                    "metadata_warnings", ""
                ),
                "has_missing_metadata": index_row.get(
                    "has_missing_metadata", ""
                ),
                "has_conflict": index_row.get("has_conflict", ""),
                "index_requires_manual_review": index_row.get(
                    "needs_manual_review", ""
                ),
                "metadata_status": index_row.get("metadata_status", ""),
                "metadata_flags": index_row.get("metadata_flags", ""),
            }
        )

    return records


def build_summary(
    audit_path: Path,
    index_path: Path,
    records: list[dict[str, object]],
) -> dict[str, object]:
    """Build deterministic reconciliation and candidate statistics."""
    type_counts = Counter(
        _canonical_file_type(str(record["file_type"])) for record in records
    )
    raw_records = [record for record in records if bool(record["is_raw"])]
    processed_records = [
        record for record in records if bool(record["is_processed"])
    ]
    preprocessing_candidates = [
        record
        for record in raw_records
        if bool(record["preprocessing_candidate"])
    ]
    return {
        "source_audit_csv": str(audit_path.resolve()),
        "source_file_index_csv": str(index_path.resolve()),
        "joined_record_count": len(records),
        "join_validation": {
            "audit_paths_unique": True,
            "index_paths_unique": True,
            "path_sets_equal": True,
            "corresponding_fields_equal": True,
        },
        "file_type_counts": {
            FILE_TYPE_RAW: type_counts.get(FILE_TYPE_RAW, 0),
            FILE_TYPE_PROCESSED: type_counts.get(FILE_TYPE_PROCESSED, 0),
            FILE_TYPE_UNKNOWN: type_counts.get(FILE_TYPE_UNKNOWN, 0),
        },
        "raw_statistics": {
            "raw_records": len(raw_records),
            "format_parse_success": sum(
                _parse_bool(str(record["format_parse_success"])) is True
                for record in raw_records
            ),
            "preprocessing_candidates": len(preprocessing_candidates),
            "reference_candidates": sum(
                bool(record["reference_candidate"]) for record in raw_records
            ),
            "raw_without_audit_metadata_warnings": sum(
                not _split_reasons(str(record["audit_metadata_warnings"]))
                for record in raw_records
            ),
            "preprocessing_candidates_without_audit_metadata_warnings": sum(
                not _split_reasons(str(record["audit_metadata_warnings"]))
                for record in preprocessing_candidates
            ),
            "requires_review": sum(
                bool(record["requires_review"]) for record in raw_records
            ),
        },
        "processed_statistics": {
            "processed_records": len(processed_records),
            "independent_sample_candidates": 0,
            "spectral_quality_exceptions": sum(
                bool(record["spectral_quality_review_reasons"])
                for record in processed_records
            ),
            "requires_review": sum(
                bool(record["requires_review"])
                for record in processed_records
            ),
        },
        "overall_requires_review": sum(
            bool(record["requires_review"]) for record in records
        ),
        "preprocessing_candidate_rule": (
            "RAW only; audit format_parse_success=True, data_column_count=2, "
            "nonfinite_value_count=0, wavenumber_strictly_increasing=True, "
            "and v1_upload_compatible=True."
        ),
        "reference_candidate_rule": (
            "RAW preprocessing candidate with a reference_label, both filename "
            "and header Sample_ID sources, and no missing, conflict, audit "
            "warning, metadata flag, or manual-review marker."
        ),
        "reference_label_caveat": (
            "reference_label is copied from the RRUFF NAMES header. Candidate "
            "status does not mean that the label was independently validated."
        ),
        "processed_data_caveat": (
            "Processed records are retained for traceability and pairing but "
            "are not counted as independent samples."
        ),
        "output_encoding": "utf-8-sig",
    }


def _output_paths(audit_path: Path, output_dir: Path) -> tuple[Path, Path]:
    base_name = audit_path.stem
    return (
        output_dir / f"{base_name}_candidate_manifest.csv",
        output_dir / f"{base_name}_candidate_summary.json",
    )


def write_outputs(
    audit_path: Path,
    output_dir: Path,
    records: list[dict[str, object]],
    summary: dict[str, object],
) -> tuple[Path, Path]:
    """Write new deterministic UTF-8-SIG outputs without overwriting."""
    manifest_path, summary_path = _output_paths(audit_path, output_dir)
    existing = [path for path in (manifest_path, summary_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "Candidate output already exists: "
            + ", ".join(str(path) for path in existing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    with summary_path.open("x", encoding="utf-8-sig", newline="") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    return manifest_path, summary_path


def run_selection(
    audit_path: Path,
    index_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[list[dict[str, object]], dict[str, object], tuple[Path, Path]]:
    """Create a complete candidate manifest from the two approved CSV inputs."""
    audit_rows = load_csv(audit_path, REQUIRED_AUDIT_COLUMNS, "Audit")
    index_rows = load_csv(index_path, REQUIRED_INDEX_COLUMNS, "Index")
    joined_rows = join_inputs(audit_rows, index_rows)
    records = build_candidate_manifest(joined_rows)
    summary = build_summary(audit_path, index_path, records)
    output_paths = write_outputs(audit_path, output_dir, records, summary)
    return records, summary, output_paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a complete RAW preprocessing and reference candidate "
            "manifest from existing RRUFF audit and metadata-index CSV files."
        )
    )
    parser.add_argument(
        "--audit",
        type=Path,
        required=True,
        help="Complete experiment 8.1 audit CSV",
    )
    parser.add_argument(
        "--index",
        type=Path,
        required=True,
        help="Experiment 8.2-A per-file metadata index CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for new candidate manifest outputs",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()
    try:
        records, summary, output_paths = run_selection(
            audit_path=arguments.audit,
            index_path=arguments.index,
            output_dir=arguments.output_dir,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Joined records: {len(records)}")
    print(f"File types: {summary['file_type_counts']}")
    print(f"RAW statistics: {summary['raw_statistics']}")
    print(f"Candidate manifest CSV: {output_paths[0]}")
    print(f"Candidate summary JSON: {output_paths[1]}")


if __name__ == "__main__":
    main()
