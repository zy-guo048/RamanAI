"""Read-only preview audit for a sampled RRUFF Raman dataset."""

from __future__ import annotations

import argparse
import codecs
import csv
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import random
import re
from statistics import median
from typing import Iterable
import unicodedata


FILE_TYPE_RAW = "RAW"
FILE_TYPE_PROCESSED = "Processed"
FILE_TYPE_UNKNOWN = "Unknown"
SUPPORTED_V1_ENCODINGS = {"ascii", "utf-8", "utf-8-sig"}
V1_MAX_UPLOAD_SIZE_BYTES = 5 * 1024 * 1024
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "exp08_audit_preview"
)

_FILENAME_METADATA_PATTERN = re.compile(
    r"^(?P<mineral>.+?)__"
    r"(?P<sample_id>[A-Z]\d{6})__"
    r"Raman__"
    r"(?P<wavelength>\d+(?:\.\d+)?)"
    r"_+Raman_Data_(?:RAW|Processed)__",
    re.IGNORECASE,
)

_BOM_DECODERS = (
    (codecs.BOM_UTF8, "utf-8-sig", "utf-8-sig"),
    (codecs.BOM_UTF32_LE, "utf-32", "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32", "utf-32-be"),
    (codecs.BOM_UTF16_LE, "utf-16", "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16", "utf-16-be"),
)


@dataclass
class AuditRecord:
    """Per-file audit data written to the preview CSV."""

    file_name: str
    relative_path: str
    file_size_bytes: int
    file_type: str
    filename_type: str
    header_file_type: str
    mineral: str
    sample_id: str
    suspected_excitation_nm: str
    encoding: str
    has_bom: bool
    delimiter: str
    header_line_count: int
    header_field_count: int
    header_names: str
    header_rruff_id: str
    header_raman_wavelength: str
    end_marker_present: bool
    data_row_count: int
    data_column_count: str
    point_count: int
    format_parse_success: bool
    v1_upload_compatible: bool
    wavenumber_min: float | None
    wavenumber_max: float | None
    wavenumber_strictly_increasing: bool | None
    has_duplicate_wavenumber: bool | None
    has_descending_wavenumber: bool | None
    nonfinite_value_count: int
    parse_errors: str
    v1_incompatibility_reasons: str
    metadata_warnings: str


def classify_filename(file_name: str) -> str:
    """Classify a filename as RAW, Processed, or Unknown."""
    lowered = file_name.casefold()
    if "raman_data_raw" in lowered:
        return FILE_TYPE_RAW
    if "raman_data_processed" in lowered:
        return FILE_TYPE_PROCESSED
    return FILE_TYPE_UNKNOWN


def classify_header_file_type(value: str) -> str:
    """Classify the RRUFF FILETYPE header without guessing."""
    normalized = value.strip().casefold()
    if normalized == "raman raw":
        return FILE_TYPE_RAW
    if normalized == "raman processed":
        return FILE_TYPE_PROCESSED
    return FILE_TYPE_UNKNOWN


def extract_filename_metadata(file_name: str) -> dict[str, str]:
    """Extract reliable metadata encoded in a standard RRUFF filename."""
    match = _FILENAME_METADATA_PATTERN.match(file_name)
    if match is None:
        return {
            "mineral": FILE_TYPE_UNKNOWN,
            "sample_id": FILE_TYPE_UNKNOWN,
            "suspected_excitation_nm": FILE_TYPE_UNKNOWN,
        }

    wavelength = match.group("wavelength")
    wavelength_value = float(wavelength)
    if not 100 <= wavelength_value <= 2000:
        wavelength = FILE_TYPE_UNKNOWN

    return {
        "mineral": match.group("mineral"),
        "sample_id": match.group("sample_id"),
        "suspected_excitation_nm": wavelength,
    }


def discover_txt_files(root: Path) -> list[Path]:
    """Return a stable inventory of TXT files without reading their content."""
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".txt"
        ),
        key=lambda path: str(path.relative_to(root)).casefold(),
    )


def select_sample(
    paths: Iterable[Path],
    limit: int,
    seed: int,
) -> list[Path]:
    """Select a reproducible, type-balanced random sample."""
    if limit < 1:
        raise ValueError("limit must be at least 1")

    ordered_paths = sorted(paths, key=lambda path: str(path).casefold())
    if not ordered_paths:
        return []

    sample_size = min(limit, len(ordered_paths))
    randomizer = random.Random(seed)
    groups = {
        FILE_TYPE_RAW: [],
        FILE_TYPE_PROCESSED: [],
        FILE_TYPE_UNKNOWN: [],
    }
    for path in ordered_paths:
        groups[classify_filename(path.name)].append(path)

    active_groups = [
        file_type
        for file_type in (
            FILE_TYPE_RAW,
            FILE_TYPE_PROCESSED,
            FILE_TYPE_UNKNOWN,
        )
        if groups[file_type]
    ]
    for file_type in active_groups:
        randomizer.shuffle(groups[file_type])

    if sample_size < len(active_groups):
        combined = list(ordered_paths)
        randomizer.shuffle(combined)
        return combined[:sample_size]

    base_count, remainder = divmod(sample_size, len(active_groups))
    selected: list[Path] = []
    leftovers: list[Path] = []
    for index, file_type in enumerate(active_groups):
        requested = base_count + (1 if index < remainder else 0)
        group = groups[file_type]
        take_count = min(requested, len(group))
        selected.extend(group[:take_count])
        leftovers.extend(group[take_count:])

    if len(selected) < sample_size:
        randomizer.shuffle(leftovers)
        selected.extend(leftovers[: sample_size - len(selected)])

    randomizer.shuffle(selected)
    return selected


def decode_text(content: bytes) -> tuple[str, str, bool]:
    """Decode text and report the detected practical encoding."""
    if not content:
        return "", "empty", False

    for bom, decoder, label in _BOM_DECODERS:
        if content.startswith(bom):
            return content.decode(decoder), label, True

    try:
        return content.decode("ascii"), "ascii", False
    except UnicodeDecodeError:
        pass

    try:
        return content.decode("utf-8"), "utf-8", False
    except UnicodeDecodeError:
        pass

    try:
        return content.decode("cp1252"), "cp1252", False
    except UnicodeDecodeError:
        return content.decode("latin-1"), "latin-1", False


def _encoding_hint(content: bytes) -> tuple[str, bool]:
    for bom, _decoder, label in _BOM_DECODERS:
        if content.startswith(bom):
            return label, True
    return FILE_TYPE_UNKNOWN, False


def _failure_record(
    path: Path,
    relative_path: str,
    file_size: int,
    filename_type: str,
    filename_metadata: dict[str, str],
    encoding: str,
    has_bom: bool,
    parse_error: str,
    v1_reason: str,
    metadata_warning: str = "",
) -> AuditRecord:
    return AuditRecord(
        file_name=path.name,
        relative_path=relative_path,
        file_size_bytes=file_size,
        file_type=filename_type,
        filename_type=filename_type,
        header_file_type=FILE_TYPE_UNKNOWN,
        mineral=filename_metadata["mineral"],
        sample_id=filename_metadata["sample_id"],
        suspected_excitation_nm=filename_metadata[
            "suspected_excitation_nm"
        ],
        encoding=encoding,
        has_bom=has_bom,
        delimiter="unknown",
        header_line_count=0,
        header_field_count=0,
        header_names=FILE_TYPE_UNKNOWN,
        header_rruff_id=FILE_TYPE_UNKNOWN,
        header_raman_wavelength=FILE_TYPE_UNKNOWN,
        end_marker_present=False,
        data_row_count=0,
        data_column_count=FILE_TYPE_UNKNOWN,
        point_count=0,
        format_parse_success=False,
        v1_upload_compatible=False,
        wavenumber_min=None,
        wavenumber_max=None,
        wavenumber_strictly_increasing=None,
        has_duplicate_wavenumber=None,
        has_descending_wavenumber=None,
        nonfinite_value_count=0,
        parse_errors=parse_error,
        v1_incompatibility_reasons=f"format_parse_failed;{v1_reason}",
        metadata_warnings=metadata_warning,
    )


def _is_unknown(value: str) -> bool:
    return not value.strip() or value == FILE_TYPE_UNKNOWN


def _normalize_mineral_name(value: str) -> str:
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


def compare_filename_and_header_metadata(
    filename_metadata: dict[str, str],
    headers: dict[str, str],
) -> list[str]:
    """Compare reliable filename metadata with headers without rewriting it."""
    warnings: list[str] = []

    filename_sample_id = filename_metadata["sample_id"]
    header_sample_id = headers.get("RRUFFID", FILE_TYPE_UNKNOWN)
    if _is_unknown(filename_sample_id) and _is_unknown(header_sample_id):
        warnings.append("sample_id_missing_both")
    elif _is_unknown(filename_sample_id):
        warnings.append("sample_id_filename_missing_or_unparsed")
    elif _is_unknown(header_sample_id):
        warnings.append("sample_id_header_missing")
    elif not re.fullmatch(r"[A-Z]\d{6}", header_sample_id, re.IGNORECASE):
        warnings.append("sample_id_comparison_requires_review")
    elif filename_sample_id.casefold() != header_sample_id.casefold():
        warnings.append("sample_id_conflict")

    filename_mineral = filename_metadata["mineral"]
    header_mineral = headers.get("NAMES", FILE_TYPE_UNKNOWN)
    if _is_unknown(filename_mineral) and _is_unknown(header_mineral):
        warnings.append("mineral_missing_both")
    elif _is_unknown(filename_mineral):
        warnings.append("mineral_filename_missing_or_unparsed")
    elif _is_unknown(header_mineral):
        warnings.append("mineral_header_missing")
    elif _normalize_mineral_name(filename_mineral) != _normalize_mineral_name(
        header_mineral
    ):
        warnings.append("mineral_name_mismatch_requires_review")

    filename_wavelength = filename_metadata["suspected_excitation_nm"]
    header_wavelength = headers.get(
        "RAMAN WAVELENGTH",
        FILE_TYPE_UNKNOWN,
    )
    if _is_unknown(filename_wavelength) and _is_unknown(header_wavelength):
        warnings.append("excitation_wavelength_missing_both")
    elif _is_unknown(filename_wavelength):
        warnings.append("excitation_wavelength_filename_missing_or_unparsed")
    elif _is_unknown(header_wavelength):
        warnings.append("excitation_wavelength_header_missing")
    else:
        filename_numeric = _parse_wavelength(filename_wavelength)
        header_numeric = _parse_wavelength(header_wavelength)
        if filename_numeric is None or header_numeric is None:
            warnings.append("excitation_wavelength_comparison_requires_review")
        elif filename_numeric != header_numeric:
            warnings.append("excitation_wavelength_conflict")

    return warnings


def _split_data_line(line: str, delimiter: str) -> list[str]:
    if delimiter == "comma":
        return [part.strip() for part in line.split(",")]
    if delimiter == "tab":
        return [part.strip() for part in line.split("\t")]
    if delimiter == "semicolon":
        return [part.strip() for part in line.split(";")]
    if delimiter == "whitespace":
        return re.split(r"\s+", line.strip())
    return [line.strip()]


def detect_delimiter(line: str) -> str:
    """Detect a practical numeric delimiter from the first data row."""
    for delimiter, character in (
        ("comma", ","),
        ("tab", "\t"),
        ("semicolon", ";"),
    ):
        if len(line.split(character)) >= 2:
            return delimiter
    if len(re.split(r"\s+", line.strip())) >= 2:
        return "whitespace"
    return "unknown"


def _parse_header_and_data(
    text: str,
) -> tuple[dict[str, str], int, list[tuple[int, str]]]:
    headers: dict[str, str] = {}
    header_line_count = 0
    data_lines: list[tuple[int, str]] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            header_line_count += 1
            if stripped.startswith("##"):
                key, separator, value = stripped[2:].partition("=")
                if separator:
                    headers[key.strip().upper()] = value.strip()
            continue
        data_lines.append((line_number, stripped))

    return headers, header_line_count, data_lines


def audit_file(path: Path, root: Path | None = None) -> AuditRecord:
    """Audit one file without modifying it or applying Raman algorithms."""
    filename_type = classify_filename(path.name)
    filename_metadata = extract_filename_metadata(path.name)
    relative_path = path.name
    if root is not None:
        try:
            relative_path = str(path.relative_to(root))
        except ValueError:
            relative_path = path.name

    try:
        file_size = path.stat().st_size
        content = path.read_bytes()
    except OSError as exc:
        return _failure_record(
            path=path,
            relative_path=relative_path,
            file_size=0,
            filename_type=filename_type,
            filename_metadata=filename_metadata,
            encoding=FILE_TYPE_UNKNOWN,
            has_bom=False,
            parse_error=f"read_error:{type(exc).__name__}",
            v1_reason="file_read_failed",
        )

    try:
        text, encoding, has_bom = decode_text(content)
    except UnicodeDecodeError:
        encoding, has_bom = _encoding_hint(content)
        return _failure_record(
            path=path,
            relative_path=relative_path,
            file_size=file_size,
            filename_type=filename_type,
            filename_metadata=filename_metadata,
            encoding=encoding,
            has_bom=has_bom,
            parse_error="decode_error:UnicodeDecodeError",
            v1_reason="decode_error",
            metadata_warning="header_metadata_unavailable_due_to_decode_error",
        )

    headers, header_line_count, data_lines = _parse_header_and_data(text)
    header_file_type = classify_header_file_type(headers.get("FILETYPE", ""))

    metadata_warnings: list[str] = []
    if (
        filename_type != FILE_TYPE_UNKNOWN
        and header_file_type != FILE_TYPE_UNKNOWN
        and filename_type != header_file_type
    ):
        file_type = FILE_TYPE_UNKNOWN
        metadata_warnings.append("filename_header_type_conflict")
    elif filename_type != FILE_TYPE_UNKNOWN:
        file_type = filename_type
    else:
        file_type = header_file_type
    metadata_warnings.extend(
        compare_filename_and_header_metadata(filename_metadata, headers)
    )

    delimiter = detect_delimiter(data_lines[0][1]) if data_lines else "unknown"
    parse_errors: list[str] = []
    numeric_rows: list[list[float]] = []
    column_counts: set[int] = set()

    if not content:
        parse_errors.append("empty_file")
    elif not data_lines:
        parse_errors.append("no_data_rows")
    elif delimiter == "unknown":
        parse_errors.append("delimiter_not_detected")

    for line_number, line in data_lines:
        parts = _split_data_line(line, delimiter)
        column_counts.add(len(parts))
        try:
            numeric_rows.append([float(part) for part in parts])
        except ValueError:
            parse_errors.append(f"non_numeric_line:{line_number}")

    if len(column_counts) > 1:
        parse_errors.append("inconsistent_column_count")

    if not column_counts:
        data_column_count = FILE_TYPE_UNKNOWN
    elif len(column_counts) == 1:
        data_column_count = str(next(iter(column_counts)))
    else:
        data_column_count = "mixed:" + ",".join(
            str(count) for count in sorted(column_counts)
        )

    format_parse_success = bool(data_lines) and not parse_errors
    nonfinite_value_count = sum(
        1
        for row in numeric_rows
        for value in row
        if not math.isfinite(value)
    )

    wavenumbers = [row[0] for row in numeric_rows if row]
    finite_wavenumbers = [
        value for value in wavenumbers if math.isfinite(value)
    ]
    wavenumber_min = min(finite_wavenumbers) if finite_wavenumbers else None
    wavenumber_max = max(finite_wavenumbers) if finite_wavenumbers else None

    if not wavenumbers:
        strictly_increasing = None
        has_duplicate = None
        has_descending = None
    else:
        has_duplicate = len(finite_wavenumbers) != len(set(finite_wavenumbers))
        has_descending = any(
            math.isfinite(left)
            and math.isfinite(right)
            and right < left
            for left, right in zip(wavenumbers, wavenumbers[1:])
        )
        strictly_increasing = (
            len(finite_wavenumbers) == len(wavenumbers)
            and all(
                right > left
                for left, right in zip(wavenumbers, wavenumbers[1:])
            )
        )

    v1_reasons: list[str] = []
    if not format_parse_success:
        v1_reasons.append("format_parse_failed")
    if file_size > V1_MAX_UPLOAD_SIZE_BYTES:
        v1_reasons.append("file_exceeds_5_mb")
    if encoding not in SUPPORTED_V1_ENCODINGS:
        v1_reasons.append("encoding_not_utf8_compatible")
    if delimiter != "comma":
        v1_reasons.append("delimiter_not_comma")
    if data_column_count != "2":
        v1_reasons.append("column_count_not_two")
    if nonfinite_value_count:
        v1_reasons.append("nonfinite_values")
    if len(numeric_rows) < 21:
        v1_reasons.append("fewer_than_21_points")
    if strictly_increasing is not True:
        v1_reasons.append("wavenumber_not_strictly_increasing")

    return AuditRecord(
        file_name=path.name,
        relative_path=relative_path,
        file_size_bytes=file_size,
        file_type=file_type,
        filename_type=filename_type,
        header_file_type=header_file_type,
        mineral=filename_metadata["mineral"],
        sample_id=filename_metadata["sample_id"],
        suspected_excitation_nm=filename_metadata["suspected_excitation_nm"],
        encoding=encoding,
        has_bom=has_bom,
        delimiter=delimiter,
        header_line_count=header_line_count,
        header_field_count=len(headers),
        header_names=headers.get("NAMES", FILE_TYPE_UNKNOWN),
        header_rruff_id=headers.get("RRUFFID", FILE_TYPE_UNKNOWN),
        header_raman_wavelength=headers.get(
            "RAMAN WAVELENGTH", FILE_TYPE_UNKNOWN
        ),
        end_marker_present="END" in headers,
        data_row_count=len(data_lines),
        data_column_count=data_column_count,
        point_count=len(numeric_rows),
        format_parse_success=format_parse_success,
        v1_upload_compatible=not v1_reasons,
        wavenumber_min=wavenumber_min,
        wavenumber_max=wavenumber_max,
        wavenumber_strictly_increasing=strictly_increasing,
        has_duplicate_wavenumber=has_duplicate,
        has_descending_wavenumber=has_descending,
        nonfinite_value_count=nonfinite_value_count,
        parse_errors=";".join(parse_errors),
        v1_incompatibility_reasons=";".join(v1_reasons),
        metadata_warnings=";".join(metadata_warnings),
    )


def _counter_dict(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def build_summary(
    records: list[AuditRecord],
    root: Path,
    discovered_count: int,
    requested_limit: int,
    seed: int,
) -> dict[str, object]:
    """Build a JSON-serializable summary without generalizing the sample."""
    parsed_point_counts = [
        record.point_count for record in records if record.format_parse_success
    ]
    parse_failures = [
        record for record in records if not record.format_parse_success
    ]
    audited_count = len(records)
    is_full_dataset = audited_count == discovered_count
    return {
        "audit_scope": (
            "full_dataset"
            if is_full_dataset
            else "sample_only_not_full_dataset"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root.resolve()),
        "discovered_txt_files": discovered_count,
        "requested_limit": requested_limit,
        "audited_files": audited_count,
        "sampled_files": audited_count,
        "seed": seed,
        "file_type_counts": _counter_dict(
            record.file_type for record in records
        ),
        "format_parse": {
            "success": len(records) - len(parse_failures),
            "failure": len(parse_failures),
        },
        "v1_upload_compatibility": {
            "compatible": sum(
                record.v1_upload_compatible for record in records
            ),
            "not_compatible": sum(
                not record.v1_upload_compatible for record in records
            ),
        },
        "encoding_counts": _counter_dict(
            record.encoding for record in records
        ),
        "delimiter_counts": _counter_dict(
            record.delimiter for record in records
        ),
        "data_column_count_counts": _counter_dict(
            record.data_column_count for record in records
        ),
        "structure_checks": {
            "files_over_v1_size_limit": sum(
                record.file_size_bytes > V1_MAX_UPLOAD_SIZE_BYTES
                for record in records
            ),
            "files_missing_end_marker": sum(
                not record.end_marker_present for record in records
            ),
            "files_with_duplicate_wavenumber": sum(
                record.has_duplicate_wavenumber is True for record in records
            ),
            "files_with_descending_wavenumber": sum(
                record.has_descending_wavenumber is True for record in records
            ),
            "files_with_nonfinite_values": sum(
                record.nonfinite_value_count > 0 for record in records
            ),
        },
        "filename_metadata_unknown_counts": {
            "mineral": sum(
                record.mineral == FILE_TYPE_UNKNOWN for record in records
            ),
            "sample_id": sum(
                record.sample_id == FILE_TYPE_UNKNOWN for record in records
            ),
            "suspected_excitation_nm": sum(
                record.suspected_excitation_nm == FILE_TYPE_UNKNOWN
                for record in records
            ),
        },
        "parse_failure_reasons": _counter_dict(
            record.parse_errors or "none" for record in parse_failures
        ),
        "v1_incompatibility_reasons": _counter_dict(
            reason
            for record in records
            for reason in record.v1_incompatibility_reasons.split(";")
            if reason
        ),
        "metadata_warning_counts": _counter_dict(
            warning
            for record in records
            for warning in record.metadata_warnings.split(";")
            if warning
        ),
        "point_count_summary_for_parsed_files": {
            "minimum": min(parsed_point_counts) if parsed_point_counts else None,
            "median": median(parsed_point_counts) if parsed_point_counts else None,
            "maximum": max(parsed_point_counts) if parsed_point_counts else None,
        },
        "interpretation_warning": (
            "Statistics describe the complete discovered TXT inventory for "
            "this run."
            if is_full_dataset
            else "Statistics describe only the reproducible sample and must "
            "not be generalized to the complete dataset."
        ),
    }


def _create_output_paths(
    output_dir: Path,
    limit: int,
    seed: int,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_name = f"rruff_audit_limit{limit}_seed{seed}_{timestamp}"
    csv_path = output_dir / f"{base_name}.csv"
    json_path = output_dir / f"{base_name}_summary.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("Audit output path already exists")
    return csv_path, json_path


def write_outputs(
    records: list[AuditRecord],
    summary: dict[str, object],
    output_dir: Path,
    limit: int,
    seed: int,
) -> tuple[Path, Path]:
    """Write new UTF-8-SIG CSV and JSON files without overwriting."""
    csv_path, json_path = _create_output_paths(output_dir, limit, seed)
    fieldnames = list(asdict(records[0]).keys()) if records else []

    with csv_path.open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    summary_with_outputs = dict(summary)
    summary_with_outputs["per_file_csv"] = str(csv_path.resolve())
    summary_with_outputs["summary_json"] = str(json_path.resolve())
    with json_path.open("x", encoding="utf-8") as output:
        json.dump(summary_with_outputs, output, ensure_ascii=False, indent=2)
        output.write("\n")

    return csv_path, json_path


def run_audit(
    root: Path,
    limit: int,
    seed: int,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[list[AuditRecord], dict[str, object], Path, Path]:
    """Run the sampled read-only audit and create new reports."""
    if not root.is_dir():
        raise ValueError(f"Dataset root is not a directory: {root}")

    paths = discover_txt_files(root)
    if not paths:
        raise ValueError(f"No .txt files found under: {root}")

    sampled_paths = select_sample(paths, limit=limit, seed=seed)
    records = [audit_file(path, root=root) for path in sampled_paths]
    summary = build_summary(
        records,
        root=root,
        discovered_count=len(paths),
        requested_limit=limit,
        seed=seed,
    )
    csv_path, json_path = write_outputs(
        records,
        summary,
        output_dir=output_dir,
        limit=limit,
        seed=seed,
    )
    return records, summary, csv_path, json_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a reproducible sample of RRUFF TXT files without applying "
            "denoising, baseline correction, normalization, or peak detection."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="RRUFF dataset directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of files to audit (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for new CSV and JSON reports",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()
    if arguments.limit < 1:
        parser.error("--limit must be at least 1")

    try:
        records, summary, csv_path, json_path = run_audit(
            root=arguments.root,
            limit=arguments.limit,
            seed=arguments.seed,
            output_dir=arguments.output_dir,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Audited files: {len(records)}")
    print(f"File types: {summary['file_type_counts']}")
    print(f"Format parse: {summary['format_parse']}")
    print(f"V1 compatibility: {summary['v1_upload_compatibility']}")
    print(f"Per-file CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")


if __name__ == "__main__":
    main()
