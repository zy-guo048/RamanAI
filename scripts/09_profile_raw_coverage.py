"""Profile audited RAW coverage and select a reproducible trial set."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import random
from statistics import fmean, median
from typing import Iterable


FILE_TYPE_RAW = "RAW"
FILE_TYPE_PROCESSED = "Processed"
FILE_TYPE_UNKNOWN = "Unknown"
UNKNOWN_GROUP = "Unknown"

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "exp09_coverage"
)

REFERENCE_BANDS = (
    (300.0, 850.0),
    (300.0, 1150.0),
    (232.0, 1466.0),
)

REQUIRED_AUDIT_COLUMNS = {
    "file_name",
    "relative_path",
    "file_type",
    "filename_type",
    "header_file_type",
    "header_names",
    "header_rruff_id",
    "header_raman_wavelength",
    "point_count",
    "format_parse_success",
    "data_column_count",
    "nonfinite_value_count",
    "v1_upload_compatible",
    "wavenumber_min",
    "wavenumber_max",
    "wavenumber_strictly_increasing",
    "parse_errors",
    "v1_incompatibility_reasons",
    "metadata_warnings",
}

REQUIRED_MANIFEST_COLUMNS = {
    "relative_path",
    "file_name",
    "measurement_stem",
    "file_type",
    "filename_type",
    "header_file_type",
    "is_raw",
    "is_processed",
    "preprocessing_candidate",
    "reference_label",
    "filename_sample_id",
    "header_rruff_id",
    "header_raman_wavelength",
    "format_parse_success",
    "data_column_count",
    "nonfinite_value_count",
    "wavenumber_strictly_increasing",
    "v1_upload_compatible",
    "parse_errors",
    "v1_incompatibility_reasons",
    "audit_metadata_warnings",
}

COVERAGE_FIELDS = [
    "section",
    "group",
    "metric",
    "candidate_count",
    "percentage_of_candidates",
    "minimum",
    "q1",
    "median",
    "q3",
    "maximum",
    "mean",
    "band_start_cm-1",
    "band_end_cm-1",
    "fully_covered_count",
    "coverage_percent",
    "notes",
]

SAMPLE_FIELDS = [
    "selection_rank",
    "relative_path",
    "file_name",
    "measurement_stem",
    "sample_id",
    "sample_id_source",
    "filename_sample_id",
    "header_rruff_id",
    "reference_label",
    "excitation_wavelength_group",
    "header_raman_wavelength",
    "wavenumber_min_cm-1",
    "wavenumber_max_cm-1",
    "wavenumber_span_cm-1",
    "point_count",
    "span_bin",
    "point_count_bin",
]


def _parse_bool(value: str | None) -> bool | None:
    normalized = (value or "").strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _canonical_file_type(value: str | None) -> str:
    normalized = (value or "").strip().casefold()
    if normalized == "raw":
        return FILE_TYPE_RAW
    if normalized == "processed":
        return FILE_TYPE_PROCESSED
    return FILE_TYPE_UNKNOWN


def _ordered(values: Iterable[str]) -> list[str]:
    return sorted(values, key=lambda value: (value.casefold(), value))


def load_csv(
    path: Path,
    required_columns: set[str],
    source_name: str,
) -> list[dict[str, str]]:
    """Load an existing CSV and validate the columns needed for profiling."""
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


def _index_unique_paths(
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
            + ", ".join(_ordered(set(duplicates))[:3])
        )
    return indexed


def join_inputs(
    audit_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
) -> list[tuple[dict[str, str], dict[str, str]]]:
    """Join the complete audit and manifest one-to-one by exact path."""
    audit_by_path = _index_unique_paths(audit_rows, "Audit")
    manifest_by_path = _index_unique_paths(manifest_rows, "Manifest")
    audit_paths = set(audit_by_path)
    manifest_paths = set(manifest_by_path)
    if audit_paths != manifest_paths:
        audit_only = _ordered(audit_paths - manifest_paths)
        manifest_only = _ordered(manifest_paths - audit_paths)
        raise ValueError(
            "Input record sets do not match by relative_path: "
            f"audit_only={len(audit_only)} {audit_only[:3]}, "
            f"manifest_only={len(manifest_only)} {manifest_only[:3]}"
        )

    corresponding_fields = (
        ("file_name", "file_name"),
        ("file_type", "file_type"),
        ("filename_type", "filename_type"),
        ("header_file_type", "header_file_type"),
        ("header_names", "reference_label"),
        ("header_rruff_id", "header_rruff_id"),
        ("header_raman_wavelength", "header_raman_wavelength"),
        ("format_parse_success", "format_parse_success"),
        ("data_column_count", "data_column_count"),
        ("nonfinite_value_count", "nonfinite_value_count"),
        (
            "wavenumber_strictly_increasing",
            "wavenumber_strictly_increasing",
        ),
        ("v1_upload_compatible", "v1_upload_compatible"),
        ("parse_errors", "parse_errors"),
        ("v1_incompatibility_reasons", "v1_incompatibility_reasons"),
        ("metadata_warnings", "audit_metadata_warnings"),
    )
    joined: list[tuple[dict[str, str], dict[str, str]]] = []
    for relative_path in _ordered(audit_paths):
        audit_row = audit_by_path[relative_path]
        manifest_row = manifest_by_path[relative_path]
        mismatches = [
            f"{audit_field}/{manifest_field}"
            for audit_field, manifest_field in corresponding_fields
            if audit_row.get(audit_field, "")
            != manifest_row.get(manifest_field, "")
        ]
        file_type = _canonical_file_type(audit_row.get("file_type"))
        if _parse_bool(manifest_row.get("is_raw")) is not (
            file_type == FILE_TYPE_RAW
        ):
            mismatches.append("file_type/is_raw")
        if _parse_bool(manifest_row.get("is_processed")) is not (
            file_type == FILE_TYPE_PROCESSED
        ):
            mismatches.append("file_type/is_processed")
        if (
            _parse_bool(manifest_row.get("preprocessing_candidate")) is True
            and file_type != FILE_TYPE_RAW
        ):
            mismatches.append("file_type/preprocessing_candidate")
        if mismatches:
            raise ValueError(
                "Joined records disagree for relative_path "
                f"{relative_path}: {', '.join(mismatches)}"
            )
        joined.append((audit_row, manifest_row))
    return joined


def normalize_excitation(value: str | None) -> str:
    """Normalize simple numeric wavelength headers; keep others Unknown."""
    normalized = (value or "").strip().casefold()
    if normalized.endswith("nm"):
        normalized = normalized[:-2].strip()
    try:
        numeric = Decimal(normalized)
    except InvalidOperation:
        return UNKNOWN_GROUP
    if not numeric.is_finite() or numeric <= 0:
        return UNKNOWN_GROUP
    return format(numeric.normalize(), "f")


def _parse_finite_float(value: str, field: str, relative_path: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {field} for {relative_path}: {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Invalid {field} for {relative_path}: {value!r}")
    return parsed


def _parse_positive_int(value: str, field: str, relative_path: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {field} for {relative_path}: {value!r}"
        ) from exc
    if parsed <= 0:
        raise ValueError(f"Invalid {field} for {relative_path}: {value!r}")
    return parsed


def build_candidate_records(
    joined_rows: list[tuple[dict[str, str], dict[str, str]]],
) -> list[dict[str, object]]:
    """Filter admitted RAW rows and validate their audited coverage values."""
    records: list[dict[str, object]] = []
    for audit_row, manifest_row in joined_rows:
        if _parse_bool(manifest_row.get("preprocessing_candidate")) is not True:
            continue
        if _canonical_file_type(audit_row.get("file_type")) != FILE_TYPE_RAW:
            continue

        relative_path = audit_row["relative_path"]
        wavenumber_min = _parse_finite_float(
            audit_row.get("wavenumber_min", ""),
            "wavenumber_min",
            relative_path,
        )
        wavenumber_max = _parse_finite_float(
            audit_row.get("wavenumber_max", ""),
            "wavenumber_max",
            relative_path,
        )
        if wavenumber_max <= wavenumber_min:
            raise ValueError(
                "Invalid wavenumber range for "
                f"{relative_path}: min={wavenumber_min}, max={wavenumber_max}"
            )
        point_count = _parse_positive_int(
            audit_row.get("point_count", ""),
            "point_count",
            relative_path,
        )
        filename_sample_id = manifest_row.get("filename_sample_id", "")
        header_rruff_id = manifest_row.get("header_rruff_id", "")
        if header_rruff_id and header_rruff_id.casefold() != "unknown":
            sample_id = header_rruff_id
            sample_id_source = "header_rruff_id"
        elif filename_sample_id and filename_sample_id.casefold() != "unknown":
            sample_id = filename_sample_id
            sample_id_source = "filename_sample_id"
        else:
            sample_id = UNKNOWN_GROUP
            sample_id_source = "unavailable"

        records.append(
            {
                "relative_path": relative_path,
                "file_name": audit_row.get("file_name", ""),
                "measurement_stem": manifest_row.get("measurement_stem", ""),
                "sample_id": sample_id,
                "sample_id_source": sample_id_source,
                "filename_sample_id": filename_sample_id,
                "header_rruff_id": header_rruff_id,
                "reference_label": manifest_row.get("reference_label", ""),
                "excitation_wavelength_group": normalize_excitation(
                    manifest_row.get("header_raman_wavelength")
                ),
                "header_raman_wavelength": manifest_row.get(
                    "header_raman_wavelength", ""
                ),
                "wavenumber_min_cm-1": wavenumber_min,
                "wavenumber_max_cm-1": wavenumber_max,
                "wavenumber_span_cm-1": wavenumber_max - wavenumber_min,
                "point_count": point_count,
            }
        )
    return sorted(
        records,
        key=lambda record: (
            str(record["relative_path"]).casefold(),
            str(record["relative_path"]),
        ),
    )


def fully_covers_band(
    wavenumber_min: float,
    wavenumber_max: float,
    band_start: float,
    band_end: float,
) -> bool:
    """Check endpoint coverage only; this does not prove internal continuity."""
    if not all(
        math.isfinite(value)
        for value in (wavenumber_min, wavenumber_max, band_start, band_end)
    ):
        raise ValueError("Coverage bounds must be finite")
    if wavenumber_max <= wavenumber_min or band_end <= band_start:
        raise ValueError("Coverage ranges must have increasing bounds")
    return wavenumber_min <= band_start and wavenumber_max >= band_end


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile from an empty sequence")
    position = (len(sorted_values) - 1) * fraction
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    return (
        sorted_values[lower_index] * (1 - weight)
        + sorted_values[upper_index] * weight
    )


def distribution(values: Iterable[float | int]) -> dict[str, float | int]:
    """Return deterministic descriptive statistics for a non-empty sequence."""
    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        raise ValueError("Cannot summarize an empty sequence")
    return {
        "count": len(sorted_values),
        "minimum": sorted_values[0],
        "q1": _percentile(sorted_values, 0.25),
        "median": median(sorted_values),
        "q3": _percentile(sorted_values, 0.75),
        "maximum": sorted_values[-1],
        "mean": fmean(sorted_values),
    }


def _distribution_values(
    records: list[dict[str, object]],
) -> dict[str, dict[str, float | int]]:
    return {
        "wavenumber_min_cm-1": distribution(
            float(record["wavenumber_min_cm-1"]) for record in records
        ),
        "wavenumber_max_cm-1": distribution(
            float(record["wavenumber_max_cm-1"]) for record in records
        ),
        "wavenumber_span_cm-1": distribution(
            float(record["wavenumber_span_cm-1"]) for record in records
        ),
        "point_count": distribution(
            int(record["point_count"]) for record in records
        ),
    }


def build_coverage_analysis(
    records: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Build distribution, excitation-group, and endpoint coverage results."""
    if not records:
        raise ValueError("No preprocessing-candidate RAW records were found")
    total = len(records)
    overall = _distribution_values(records)
    csv_rows: list[dict[str, object]] = []
    for metric, stats in overall.items():
        csv_stats = {key: value for key, value in stats.items() if key != "count"}
        csv_rows.append(
            {
                "section": "overall_distribution",
                "group": "All",
                "metric": metric,
                "candidate_count": total,
                "percentage_of_candidates": 100.0,
                **csv_stats,
                "notes": "",
            }
        )

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["excitation_wavelength_group"])].append(record)
    excitation_summary: dict[str, object] = {}
    for group in _ordered(grouped):
        group_records = grouped[group]
        group_distributions = _distribution_values(group_records)
        percentage = len(group_records) / total * 100
        excitation_summary[group] = {
            "count": len(group_records),
            "percentage": percentage,
            "distributions": group_distributions,
        }
        for metric, stats in group_distributions.items():
            csv_stats = {
                key: value for key, value in stats.items() if key != "count"
            }
            csv_rows.append(
                {
                    "section": "excitation_group_distribution",
                    "group": group,
                    "metric": metric,
                    "candidate_count": len(group_records),
                    "percentage_of_candidates": percentage,
                    **csv_stats,
                    "notes": "",
                }
            )

    coverage_limitation = (
        "Coverage uses only audited minimum and maximum Raman shifts; endpoint "
        "coverage does not prove that the interval contains no internal gaps."
    )
    band_summary: list[dict[str, object]] = []
    for band_start, band_end in REFERENCE_BANDS:
        covered_count = sum(
            fully_covers_band(
                float(record["wavenumber_min_cm-1"]),
                float(record["wavenumber_max_cm-1"]),
                band_start,
                band_end,
            )
            for record in records
        )
        coverage_percent = covered_count / total * 100
        band_record = {
            "band_start_cm-1": band_start,
            "band_end_cm-1": band_end,
            "fully_covered_count": covered_count,
            "coverage_percent": coverage_percent,
        }
        band_summary.append(band_record)
        csv_rows.append(
            {
                "section": "reference_band_coverage",
                "group": f"{band_start:g}-{band_end:g}",
                "metric": "endpoint_full_coverage",
                "candidate_count": total,
                "percentage_of_candidates": 100.0,
                **band_record,
                "notes": coverage_limitation,
            }
        )

    summary = {
        "candidate_raw_count": total,
        "overall_distributions": overall,
        "excitation_groups": excitation_summary,
        "reference_band_coverage": band_summary,
        "coverage_limitation": coverage_limitation,
        "grid_decision_status": (
            "The three reference bands are comparison candidates only; no "
            "final unified Raman-shift grid is selected in this experiment."
        ),
    }
    return csv_rows, summary


def _bin_label(value: float, lower_cut: float, upper_cut: float) -> str:
    if value <= lower_cut:
        return "low"
    if value <= upper_cut:
        return "middle"
    return "high"


def select_diverse_trial_set(
    records: list[dict[str, object]],
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Greedily cover wavelength, span, point-count, and Sample_ID diversity."""
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")
    if not records:
        return [], {
            "requested_sample_size": sample_size,
            "selected_sample_size": 0,
            "shortfall": sample_size,
            "compromise_note": "No eligible RAW candidates were available.",
        }

    spans = sorted(float(record["wavenumber_span_cm-1"]) for record in records)
    points = sorted(float(record["point_count"]) for record in records)
    span_cuts = (_percentile(spans, 1 / 3), _percentile(spans, 2 / 3))
    point_cuts = (_percentile(points, 1 / 3), _percentile(points, 2 / 3))

    enriched: list[dict[str, object]] = []
    randomizer = random.Random(seed)
    for record in sorted(
        records,
        key=lambda item: (
            str(item["relative_path"]).casefold(),
            str(item["relative_path"]),
        ),
    ):
        copied = dict(record)
        copied["span_bin"] = _bin_label(
            float(record["wavenumber_span_cm-1"]),
            *span_cuts,
        )
        copied["point_count_bin"] = _bin_label(
            float(record["point_count"]),
            *point_cuts,
        )
        copied["_tie_breaker"] = randomizer.random()
        enriched.append(copied)

    target_size = min(sample_size, len(enriched))
    selected: list[dict[str, object]] = []
    remaining = list(enriched)
    excitation_counts: Counter[str] = Counter()
    selected_span_bins: set[str] = set()
    selected_point_bins: set[str] = set()
    selected_combinations: set[tuple[str, str]] = set()
    selected_sample_ids: set[str] = set()

    while len(selected) < target_size:
        def score(record: dict[str, object]) -> tuple[object, ...]:
            excitation = str(record["excitation_wavelength_group"])
            span_bin = str(record["span_bin"])
            point_bin = str(record["point_count_bin"])
            sample_id = str(record["sample_id"])
            return (
                excitation_counts[excitation] == 0,
                (span_bin, point_bin) not in selected_combinations,
                span_bin not in selected_span_bins,
                point_bin not in selected_point_bins,
                sample_id not in selected_sample_ids,
                -excitation_counts[excitation],
                float(record["_tie_breaker"]),
            )

        chosen = max(remaining, key=score)
        remaining.remove(chosen)
        selected.append(chosen)
        excitation = str(chosen["excitation_wavelength_group"])
        span_bin = str(chosen["span_bin"])
        point_bin = str(chosen["point_count_bin"])
        excitation_counts[excitation] += 1
        selected_span_bins.add(span_bin)
        selected_point_bins.add(point_bin)
        selected_combinations.add((span_bin, point_bin))
        selected_sample_ids.add(str(chosen["sample_id"]))

    output_records: list[dict[str, object]] = []
    for rank, record in enumerate(selected, start=1):
        output_record = {
            field: record.get(field, "")
            for field in SAMPLE_FIELDS
            if field != "selection_rank"
        }
        output_record["selection_rank"] = rank
        output_records.append(output_record)

    available_excitation_groups = {
        str(record["excitation_wavelength_group"]) for record in records
    }
    represented_excitation_groups = {
        str(record["excitation_wavelength_group"]) for record in selected
    }
    compromise_notes: list[str] = []
    if len(records) < sample_size:
        compromise_notes.append(
            f"Only {len(records)} eligible records were available for the "
            f"requested sample size of {sample_size}."
        )
    if represented_excitation_groups != available_excitation_groups:
        compromise_notes.append(
            "The requested sample size could not represent every excitation "
            "wavelength group."
        )
    compromise_notes.append(
        "Selection is diversity-oriented rather than statistically balanced; "
        "group sizes in the trial set do not reproduce population proportions."
    )
    selection_summary = {
        "seed": seed,
        "requested_sample_size": sample_size,
        "selected_sample_size": len(output_records),
        "shortfall": sample_size - len(output_records),
        "method": (
            "Deterministic seeded greedy selection prioritizing previously "
            "unrepresented excitation groups, span/point-count bin combinations, "
            "individual bins, and Sample_ID values."
        ),
        "span_bin_cutoffs_cm-1": {
            "lower_third": span_cuts[0],
            "upper_third": span_cuts[1],
        },
        "point_count_bin_cutoffs": {
            "lower_third": point_cuts[0],
            "upper_third": point_cuts[1],
        },
        "available_excitation_groups": _ordered(available_excitation_groups),
        "represented_excitation_groups": _ordered(
            represented_excitation_groups
        ),
        "represented_span_bins": _ordered(selected_span_bins),
        "represented_point_count_bins": _ordered(selected_point_bins),
        "distinct_sample_ids": len(selected_sample_ids),
        "compromise_note": " ".join(compromise_notes),
    }
    return output_records, selection_summary


def _output_paths(
    audit_path: Path,
    output_dir: Path,
    sample_size: int,
    seed: int,
) -> tuple[Path, Path, Path]:
    base_name = audit_path.stem
    return (
        output_dir / f"{base_name}_raw_coverage_statistics.csv",
        output_dir
        / f"{base_name}_raw_trial_sample{sample_size}_seed{seed}.csv",
        output_dir / f"{base_name}_raw_coverage_summary_seed{seed}.json",
    )


def write_outputs(
    audit_path: Path,
    output_dir: Path,
    sample_size: int,
    seed: int,
    coverage_rows: list[dict[str, object]],
    sample_records: list[dict[str, object]],
    summary: dict[str, object],
) -> tuple[Path, Path, Path]:
    """Write deterministic UTF-8-SIG reports and refuse to overwrite."""
    paths = _output_paths(audit_path, output_dir, sample_size, seed)
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Coverage output already exists: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_path, sample_path, summary_path = paths
    with coverage_path.open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=COVERAGE_FIELDS)
        writer.writeheader()
        writer.writerows(coverage_rows)
    with sample_path.open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(sample_records)
    with summary_path.open("x", encoding="utf-8-sig", newline="") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    return paths


def run_profile(
    audit_path: Path,
    manifest_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
    sample_size: int = 30,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    tuple[Path, Path, Path],
]:
    """Build coverage statistics and a trial list without reading spectra."""
    audit_rows = load_csv(audit_path, REQUIRED_AUDIT_COLUMNS, "Audit")
    manifest_rows = load_csv(
        manifest_path,
        REQUIRED_MANIFEST_COLUMNS,
        "Manifest",
    )
    joined_rows = join_inputs(audit_rows, manifest_rows)
    candidates = build_candidate_records(joined_rows)
    coverage_rows, coverage_summary = build_coverage_analysis(candidates)
    sample_records, selection_summary = select_diverse_trial_set(
        candidates,
        sample_size,
        seed,
    )
    summary = {
        "source_audit_csv": str(audit_path.resolve()),
        "source_candidate_manifest_csv": str(manifest_path.resolve()),
        "join_validation": {
            "audit_paths_unique": True,
            "manifest_paths_unique": True,
            "path_sets_equal": True,
            "corresponding_fields_equal": True,
        },
        **coverage_summary,
        "trial_selection": selection_summary,
        "output_encoding": "utf-8-sig",
    }
    output_paths = write_outputs(
        audit_path,
        output_dir,
        sample_size,
        seed,
        coverage_rows,
        sample_records,
        summary,
    )
    return coverage_rows, sample_records, summary, output_paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Profile Raman-shift endpoint coverage for admitted RAW spectra and "
            "select a reproducible diversity-oriented trial set."
        )
    )
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=30)
    return parser


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()
    try:
        _, sample_records, summary, output_paths = run_profile(
            audit_path=arguments.audit,
            manifest_path=arguments.manifest,
            output_dir=arguments.output_dir,
            seed=arguments.seed,
            sample_size=arguments.sample_size,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Candidate RAW spectra: {summary['candidate_raw_count']}")
    print(f"Reference band coverage: {summary['reference_band_coverage']}")
    print(f"Selected trial spectra: {len(sample_records)}")
    print(f"Coverage statistics CSV: {output_paths[0]}")
    print(f"Trial sample CSV: {output_paths[1]}")
    print(f"Summary JSON: {output_paths[2]}")


if __name__ == "__main__":
    main()
