"""Build a deterministic RRUFF metadata index from an audit CSV."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path, PurePosixPath
import re
from typing import Iterable
import unicodedata


FILE_TYPE_RAW = "RAW"
FILE_TYPE_PROCESSED = "Processed"
FILE_TYPE_UNKNOWN = "Unknown"

PAIR_STATUS_BOTH = "Both"
PAIR_STATUS_RAW_ONLY = "RAW-only"
PAIR_STATUS_PROCESSED_ONLY = "Processed-only"
PAIR_STATUS_UNCLASSIFIED = "Unclassified"

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "exp08_metadata_index"
)

REQUIRED_INPUT_COLUMNS = {
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
    "v1_upload_compatible",
    "metadata_warnings",
}

FILE_INDEX_FIELDS = [
    "source_row_number",
    "relative_path",
    "file_name",
    "file_type",
    "filename_type",
    "header_file_type",
    "measurement_stem",
    "filename_suffix_type",
    "filename_hash",
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
]

PAIR_INDEX_FIELDS = [
    "measurement_stem",
    "pair_status",
    "total_file_count",
    "raw_count",
    "processed_count",
    "unclassified_count",
    "raw_relative_paths",
    "processed_relative_paths",
    "unclassified_relative_paths",
    "filename_minerals",
    "reference_labels",
    "filename_sample_ids",
    "header_rruff_ids",
    "filename_suspected_excitation_nm",
    "header_raman_wavelengths",
    "has_metadata_issue",
    "metadata_flags",
]

_MEASUREMENT_SUFFIX_PATTERN = re.compile(
    r"^(?P<stem>.+)_(?P<file_type>RAW|Processed)__(?P<hash>[^/\\]+)$",
    re.IGNORECASE,
)
_SAMPLE_ID_PATTERN = re.compile(r"[A-Z]\d{6}", re.IGNORECASE)


def _is_missing(value: str | None) -> bool:
    if value is None:
        return True
    return value.strip().casefold() in {"", "unknown", "none", "n/a"}


def _canonical_file_type(value: str | None) -> str:
    normalized = (value or "").strip().casefold()
    if normalized == "raw":
        return FILE_TYPE_RAW
    if normalized == "processed":
        return FILE_TYPE_PROCESSED
    return FILE_TYPE_UNKNOWN


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.replace("_", " ").split())


def _parse_wavelength(value: str) -> Decimal | None:
    match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*(?:nm)?\s*",
        value,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    try:
        return Decimal(match.group(1))
    except InvalidOperation:
        return None


def derive_measurement_stem(
    relative_path: str,
) -> tuple[str, str, str, bool]:
    """Remove the terminal data type and hash from an audited relative path."""
    normalized_path = relative_path.replace("\\", "/")
    path = PurePosixPath(normalized_path)
    suffixless_name = path.name[:-4] if path.name.casefold().endswith(".txt") else path.name
    match = _MEASUREMENT_SUFFIX_PATTERN.fullmatch(suffixless_name)
    if match is None:
        fallback = str(path.with_name(suffixless_name))
        return fallback, FILE_TYPE_UNKNOWN, "", False

    suffix_type = _canonical_file_type(match.group("file_type"))
    stem_path = path.with_name(match.group("stem"))
    return str(stem_path), suffix_type, match.group("hash"), True


def _add_source_comparison_flags(
    flags: set[str],
    filename_value: str,
    header_value: str,
    field_name: str,
) -> None:
    filename_missing = _is_missing(filename_value)
    header_missing = _is_missing(header_value)
    if filename_missing:
        flags.add(f"{field_name}_filename_missing")
    if header_missing:
        flags.add(f"{field_name}_header_missing")
    if filename_missing or header_missing:
        return

    if field_name == "sample_id":
        if not _SAMPLE_ID_PATTERN.fullmatch(header_value.strip()):
            flags.add("sample_id_comparison_requires_review")
        elif filename_value.strip().casefold() != header_value.strip().casefold():
            flags.add("sample_id_conflict")
    elif field_name == "mineral":
        if _normalize_text(filename_value) != _normalize_text(header_value):
            flags.add("mineral_name_mismatch_requires_review")
    elif field_name == "excitation_wavelength":
        filename_numeric = _parse_wavelength(filename_value)
        header_numeric = _parse_wavelength(header_value)
        if filename_numeric is None or header_numeric is None:
            flags.add("excitation_wavelength_comparison_requires_review")
        elif filename_numeric != header_numeric:
            flags.add("excitation_wavelength_conflict")


def derive_metadata_flags(
    row: dict[str, str],
    suffix_type: str,
    stem_parsed: bool,
) -> list[str]:
    """Return deterministic quality flags while preserving both data sources."""
    flags = {
        warning.strip()
        for warning in row.get("metadata_warnings", "").split(";")
        if warning.strip()
    }
    _add_source_comparison_flags(
        flags,
        row.get("sample_id", ""),
        row.get("header_rruff_id", ""),
        "sample_id",
    )
    _add_source_comparison_flags(
        flags,
        row.get("mineral", ""),
        row.get("header_names", ""),
        "mineral",
    )
    _add_source_comparison_flags(
        flags,
        row.get("suspected_excitation_nm", ""),
        row.get("header_raman_wavelength", ""),
        "excitation_wavelength",
    )

    audited_type = _canonical_file_type(row.get("file_type"))
    if audited_type == FILE_TYPE_UNKNOWN:
        flags.add("file_type_missing_or_unknown")
    if not stem_parsed:
        flags.add("measurement_stem_unparsed_requires_review")
    elif audited_type != FILE_TYPE_UNKNOWN and audited_type != suffix_type:
        flags.add("file_type_suffix_conflict")

    return sorted(flags, key=lambda value: (value.casefold(), value))


def _metadata_status(flags: Iterable[str]) -> tuple[bool, bool, bool, str]:
    flag_list = list(flags)
    has_missing = any("missing" in flag for flag in flag_list)
    has_conflict = any("conflict" in flag for flag in flag_list)
    needs_review = bool(flag_list)
    if not flag_list:
        return False, False, False, "complete_no_flag"

    categories: list[str] = []
    if has_missing:
        categories.append("missing")
    if has_conflict:
        categories.append("conflict")
    if any("requires_review" in flag for flag in flag_list):
        categories.append("requires_review")
    if not categories:
        categories.append("flagged")
    return has_missing, has_conflict, needs_review, ";".join(categories)


def load_audit_csv(input_path: Path) -> list[dict[str, str]]:
    """Load the existing audit CSV without reading any source spectrum."""
    if not input_path.is_file():
        raise ValueError(f"Audit CSV does not exist: {input_path}")

    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("Audit CSV has no header row")
        missing_columns = sorted(REQUIRED_INPUT_COLUMNS - set(reader.fieldnames))
        if missing_columns:
            raise ValueError(
                "Audit CSV is missing required columns: "
                + ", ".join(missing_columns)
            )
        return [dict(row) for row in reader]


def build_file_index(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Create deterministic per-file index records from audited metadata."""
    records: list[dict[str, object]] = []
    for source_row_number, row in enumerate(rows, start=2):
        relative_path = row.get("relative_path") or row.get("file_name", "")
        stem, suffix_type, filename_hash, stem_parsed = derive_measurement_stem(
            relative_path
        )
        flags = derive_metadata_flags(row, suffix_type, stem_parsed)
        has_missing, has_conflict, needs_review, status = _metadata_status(flags)
        records.append(
            {
                "source_row_number": source_row_number,
                "relative_path": relative_path,
                "file_name": row.get("file_name", ""),
                "file_type": row.get("file_type", ""),
                "filename_type": row.get("filename_type", ""),
                "header_file_type": row.get("header_file_type", ""),
                "measurement_stem": stem,
                "filename_suffix_type": suffix_type,
                "filename_hash": filename_hash,
                "filename_mineral": row.get("mineral", ""),
                "reference_label": row.get("header_names", ""),
                "filename_sample_id": row.get("sample_id", ""),
                "header_rruff_id": row.get("header_rruff_id", ""),
                "filename_suspected_excitation_nm": row.get(
                    "suspected_excitation_nm", ""
                ),
                "header_raman_wavelength": row.get(
                    "header_raman_wavelength", ""
                ),
                "format_parse_success": row.get("format_parse_success", ""),
                "v1_upload_compatible": row.get("v1_upload_compatible", ""),
                "audit_metadata_warnings": row.get("metadata_warnings", ""),
                "has_missing_metadata": has_missing,
                "has_conflict": has_conflict,
                "needs_manual_review": needs_review,
                "metadata_status": status,
                "metadata_flags": ";".join(flags),
            }
        )

    return sorted(
        records,
        key=lambda record: (
            str(record["relative_path"]).casefold(),
            str(record["relative_path"]),
            int(record["source_row_number"]),
        ),
    )


def _joined_unique(records: list[dict[str, object]], field: str) -> str:
    values = {
        str(record[field]).strip()
        for record in records
        if not _is_missing(str(record[field]))
    }
    return " | ".join(sorted(values, key=lambda value: (value.casefold(), value)))


def build_pair_index(
    file_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Group candidate measurements into Both, RAW-only, or Processed-only."""
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in file_records:
        grouped[str(record["measurement_stem"])].append(record)

    pair_records: list[dict[str, object]] = []
    for measurement_stem in sorted(
        grouped,
        key=lambda value: (value.casefold(), value),
    ):
        records = grouped[measurement_stem]
        raw_records = [
            record
            for record in records
            if _canonical_file_type(str(record["file_type"])) == FILE_TYPE_RAW
        ]
        processed_records = [
            record
            for record in records
            if _canonical_file_type(str(record["file_type"]))
            == FILE_TYPE_PROCESSED
        ]
        unclassified_records = [
            record
            for record in records
            if _canonical_file_type(str(record["file_type"]))
            == FILE_TYPE_UNKNOWN
        ]

        if raw_records and processed_records:
            pair_status = PAIR_STATUS_BOTH
        elif raw_records:
            pair_status = PAIR_STATUS_RAW_ONLY
        elif processed_records:
            pair_status = PAIR_STATUS_PROCESSED_ONLY
        else:
            pair_status = PAIR_STATUS_UNCLASSIFIED

        combined_flags = sorted(
            {
                flag
                for record in records
                for flag in str(record["metadata_flags"]).split(";")
                if flag
            },
            key=lambda value: (value.casefold(), value),
        )
        pair_records.append(
            {
                "measurement_stem": measurement_stem,
                "pair_status": pair_status,
                "total_file_count": len(records),
                "raw_count": len(raw_records),
                "processed_count": len(processed_records),
                "unclassified_count": len(unclassified_records),
                "raw_relative_paths": _joined_unique(
                    raw_records, "relative_path"
                ),
                "processed_relative_paths": _joined_unique(
                    processed_records, "relative_path"
                ),
                "unclassified_relative_paths": _joined_unique(
                    unclassified_records, "relative_path"
                ),
                "filename_minerals": _joined_unique(
                    records, "filename_mineral"
                ),
                "reference_labels": _joined_unique(records, "reference_label"),
                "filename_sample_ids": _joined_unique(
                    records, "filename_sample_id"
                ),
                "header_rruff_ids": _joined_unique(records, "header_rruff_id"),
                "filename_suspected_excitation_nm": _joined_unique(
                    records, "filename_suspected_excitation_nm"
                ),
                "header_raman_wavelengths": _joined_unique(
                    records, "header_raman_wavelength"
                ),
                "has_metadata_issue": bool(combined_flags),
                "metadata_flags": ";".join(combined_flags),
            }
        )

    return pair_records


def build_summary(
    input_path: Path,
    file_records: list[dict[str, object]],
    pair_records: list[dict[str, object]],
) -> dict[str, object]:
    """Build a deterministic summary for the generated metadata index."""
    pair_counts = Counter(str(record["pair_status"]) for record in pair_records)
    file_type_counts = Counter(
        _canonical_file_type(str(record["file_type"]))
        for record in file_records
    )
    return {
        "source_audit_csv": str(input_path.resolve()),
        "input_record_count": len(file_records),
        "measurement_stem_count": len(pair_records),
        "pair_status_counts": {
            status: pair_counts.get(status, 0)
            for status in (
                PAIR_STATUS_BOTH,
                PAIR_STATUS_RAW_ONLY,
                PAIR_STATUS_PROCESSED_ONLY,
                PAIR_STATUS_UNCLASSIFIED,
            )
        },
        "file_type_counts": {
            file_type: file_type_counts.get(file_type, 0)
            for file_type in (
                FILE_TYPE_RAW,
                FILE_TYPE_PROCESSED,
                FILE_TYPE_UNKNOWN,
            )
        },
        "metadata_quality": {
            "records_with_missing_metadata": sum(
                bool(record["has_missing_metadata"]) for record in file_records
            ),
            "records_with_conflicts": sum(
                bool(record["has_conflict"]) for record in file_records
            ),
            "records_requiring_manual_review": sum(
                bool(record["needs_manual_review"]) for record in file_records
            ),
            "records_without_flags": sum(
                not bool(record["needs_manual_review"])
                for record in file_records
            ),
        },
        "measurement_stem_rule": (
            "Remove the terminal _RAW__<hash> or _Processed__<hash> from "
            "the audited relative path without changing source metadata."
        ),
        "reference_label_note": (
            "reference_label preserves the audit header_names value and is "
            "not treated as a confirmed mineral class."
        ),
        "output_encoding": "utf-8-sig",
    }


def _output_paths(input_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    base_name = input_path.stem
    return (
        output_dir / f"{base_name}_file_index.csv",
        output_dir / f"{base_name}_measurement_pairs.csv",
        output_dir / f"{base_name}_metadata_index_summary.json",
    )


def _write_csv(
    path: Path,
    fieldnames: list[str],
    records: list[dict[str, object]],
) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_outputs(
    input_path: Path,
    output_dir: Path,
    file_records: list[dict[str, object]],
    pair_records: list[dict[str, object]],
    summary: dict[str, object],
) -> tuple[Path, Path, Path]:
    """Write three new UTF-8-SIG outputs and refuse to overwrite any file."""
    paths = _output_paths(input_path, output_dir)
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Metadata index output already exists: "
            + ", ".join(str(path) for path in existing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    file_index_path, pair_index_path, summary_path = paths
    _write_csv(file_index_path, FILE_INDEX_FIELDS, file_records)
    _write_csv(pair_index_path, PAIR_INDEX_FIELDS, pair_records)
    with summary_path.open("x", encoding="utf-8-sig", newline="") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    return paths


def run_index(
    input_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    tuple[Path, Path, Path],
]:
    """Build and write metadata indexes solely from an existing audit CSV."""
    rows = load_audit_csv(input_path)
    file_records = build_file_index(rows)
    pair_records = build_pair_index(file_records)
    summary = build_summary(input_path, file_records, pair_records)
    output_paths = write_outputs(
        input_path,
        output_dir,
        file_records,
        pair_records,
        summary,
    )
    return file_records, pair_records, summary, output_paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic metadata and RAW/Processed pairing index "
            "from an existing RRUFF audit CSV."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Full RRUFF audit CSV produced by the experiment 8.1 auditor",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for new metadata index outputs",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()
    try:
        file_records, pair_records, summary, output_paths = run_index(
            input_path=arguments.input,
            output_dir=arguments.output_dir,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Input records: {len(file_records)}")
    print(f"Measurement stems: {len(pair_records)}")
    print(f"Pair status counts: {summary['pair_status_counts']}")
    print(f"Per-file index CSV: {output_paths[0]}")
    print(f"Measurement pair CSV: {output_paths[1]}")
    print(f"Summary JSON: {output_paths[2]}")


if __name__ == "__main__":
    main()
