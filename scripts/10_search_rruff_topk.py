"""Offline cosine Top-K retrieval for the experiment 9.3 RRUFF cache."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Iterable
import unicodedata

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP09_BATCH_SCRIPT = PROJECT_ROOT / "scripts" / "09_batch_preprocess_rruff.py"
DEFAULT_CACHE = (
    PROJECT_ROOT
    / "results"
    / "exp09_batch_preprocessing_full"
    / "rruff_preprocessed_300_850_step0p5.npz"
)
DEFAULT_INDEX = (
    PROJECT_ROOT
    / "results"
    / "exp09_batch_preprocessing_full"
    / "rruff_preprocessing_index.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "exp10_topk_baseline"

REQUIRED_INDEX_COLUMNS = {
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
    "failure_code",
    "cache_row_index",
}

RESULT_FIELDS = [
    "scenario_id",
    "query_mode",
    "query_relative_path",
    "query_sample_id",
    "query_sample_id_source",
    "query_measurement_stem",
    "query_reference_label",
    "query_excitation_wavelength_nm",
    "query_known_sample_id",
    "result_level",
    "rank",
    "candidate_relative_path",
    "candidate_file_name",
    "candidate_sample_id",
    "candidate_sample_id_source",
    "candidate_measurement_stem",
    "candidate_reference_label",
    "candidate_reference_label_status",
    "candidate_excitation_wavelength_nm",
    "candidate_excitation_wavelength_source",
    "cosine_similarity",
    "dedupe_identity_basis",
    "dedupe_identity_key",
    "reference_label_group_display",
    "reference_label_grouping_basis",
    "reference_label_group_key",
    "label_score_definition",
    "label_reference_file_count",
    "label_distinguishable_sample_id_count",
    "exclude_self",
    "exclude_same_known_sample_id",
    "exclusion_rule",
    "candidate_requires_review",
    "candidate_review_reasons",
    "candidate_audit_metadata_warnings",
    "candidate_has_missing_metadata",
    "candidate_has_conflict",
    "candidate_metadata_status",
    "candidate_metadata_flags",
]


def _load_exp09_batch_module():
    spec = importlib.util.spec_from_file_location(
        "raman_exp09_batch_preprocessing",
        EXP09_BATCH_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load experiment 9.3 preprocessing module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp09 = _load_exp09_batch_module()


@dataclass(frozen=True)
class QuerySpectrum:
    vector: np.ndarray
    mode: str
    relative_path: str
    sample_id: str = "Unknown"
    sample_id_source: str = "missing"
    measurement_stem: str = "Unknown"
    reference_label: str = "Unknown"
    excitation_wavelength_nm: str = "Unknown"


@dataclass(frozen=True)
class ReferenceLibrary:
    grid: np.ndarray
    all_success_matrix: np.ndarray
    all_success_records: tuple[dict[str, str], ...]
    reference_matrix: np.ndarray
    reference_records: tuple[dict[str, str], ...]
    path_to_success_row: dict[str, int]


def _parse_bool(value: str | None, field_name: str) -> bool:
    normalized = (value or "").strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid boolean value for {field_name}: {value!r}")


def _is_missing(value: str | None) -> bool:
    return (value or "").strip().casefold() in {"", "unknown", "none", "n/a"}


def _load_index(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Index CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("Index CSV has no header row")
        missing = sorted(REQUIRED_INDEX_COLUMNS - set(reader.fieldnames))
        if missing:
            raise ValueError(
                "Index CSV is missing required columns: " + ", ".join(missing)
            )
        rows = [dict(row) for row in reader]
    paths = [row.get("relative_path", "").strip() for row in rows]
    if any(not path_value for path_value in paths):
        raise ValueError("Index CSV contains an empty relative_path")
    duplicates = [path for path, count in Counter(paths).items() if count > 1]
    if duplicates:
        raise ValueError(
            "Index CSV contains duplicate relative_path values: "
            + ", ".join(sorted(duplicates, key=str.casefold)[:3])
        )
    return rows


def _parse_cache_row_index(row: dict[str, str]) -> int:
    raw_value = row.get("cache_row_index", "").strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Successful index row has invalid cache_row_index: {raw_value!r}"
        ) from exc
    if value < 0:
        raise ValueError("cache_row_index must be non-negative")
    return value


def _validate_grid(grid: np.ndarray) -> None:
    expected = exp09.make_trial_grid()
    if grid.ndim != 1 or not np.array_equal(grid, expected):
        raise ValueError(
            "Cache wavenumber grid does not match experiment 9.3 "
            "300-850 cm-1, 0.5 cm-1 configuration"
        )


def load_reference_library(cache_path: Path, index_path: Path) -> ReferenceLibrary:
    """Load and fully reconcile the experiment 9.3 cache and index."""
    if not cache_path.is_file():
        raise ValueError(f"NPZ cache does not exist: {cache_path}")
    index_rows = _load_index(index_path)
    with np.load(cache_path, allow_pickle=False) as cache:
        required_arrays = {
            "wavenumber_grid_cm_1",
            "normalized_spectra",
            "relative_paths",
            "measurement_stems",
        }
        missing_arrays = sorted(required_arrays - set(cache.files))
        if missing_arrays:
            raise ValueError(
                "NPZ cache is missing arrays: " + ", ".join(missing_arrays)
            )
        grid = np.asarray(cache["wavenumber_grid_cm_1"], dtype=np.float64).copy()
        matrix = np.asarray(cache["normalized_spectra"], dtype=np.float64).copy()
        cache_paths = np.asarray(cache["relative_paths"], dtype=str).copy()
        cache_stems = np.asarray(cache["measurement_stems"], dtype=str).copy()

    _validate_grid(grid)
    if matrix.ndim != 2 or matrix.shape[1] != len(grid):
        raise ValueError("Cache matrix dimensions do not match the wavenumber grid")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Cache matrix contains non-finite values")
    if cache_paths.ndim != 1 or cache_stems.ndim != 1:
        raise ValueError("Cache identity arrays must be one-dimensional")
    if len(cache_paths) != matrix.shape[0] or len(cache_stems) != matrix.shape[0]:
        raise ValueError("Cache identity array lengths do not match matrix rows")
    if len(set(cache_paths.tolist())) != len(cache_paths):
        raise ValueError("NPZ cache contains duplicate relative_paths")

    success_rows = [
        row
        for row in index_rows
        if row.get("processing_status", "").strip().casefold() == "success"
    ]
    indexed_success: list[tuple[int, dict[str, str]]] = [
        (_parse_cache_row_index(row), row) for row in success_rows
    ]
    indexed_success.sort(key=lambda item: item[0])
    cache_indices = [item[0] for item in indexed_success]
    if cache_indices != list(range(len(indexed_success))):
        raise ValueError("Successful index cache_row_index values are not contiguous")
    if len(indexed_success) != matrix.shape[0]:
        raise ValueError("Successful index row count does not match cache matrix rows")
    ordered_success_rows = tuple(item[1] for item in indexed_success)
    expected_paths = np.asarray(
        [row["relative_path"] for row in ordered_success_rows], dtype=str
    )
    expected_stems = np.asarray(
        [row["measurement_stem"] for row in ordered_success_rows], dtype=str
    )
    if not np.array_equal(cache_paths, expected_paths):
        raise ValueError("Index and NPZ relative_path identity order do not match")
    if not np.array_equal(cache_stems, expected_stems):
        raise ValueError("Index and NPZ measurement_stem identity order do not match")

    row_norms = np.linalg.norm(matrix, axis=1)
    invalid_norms = ~np.isfinite(row_norms) | (row_norms <= 0)
    if np.any(invalid_norms):
        bad_path = cache_paths[int(np.flatnonzero(invalid_norms)[0])]
        raise ValueError(f"Cache contains a zero-norm or invalid spectrum: {bad_path}")

    reference_indices: list[int] = []
    reference_rows: list[dict[str, str]] = []
    for cache_row, row in enumerate(ordered_success_rows):
        if _parse_bool(row.get("reference_candidate"), "reference_candidate"):
            reference_indices.append(cache_row)
            reference_rows.append(row)
    reference_matrix = matrix[np.asarray(reference_indices, dtype=int)]
    if not reference_rows:
        raise ValueError("No successful reference candidates are available")
    path_to_success_row = {
        row["relative_path"]: cache_row
        for cache_row, row in enumerate(ordered_success_rows)
    }
    return ReferenceLibrary(
        grid=grid,
        all_success_matrix=matrix,
        all_success_records=ordered_success_rows,
        reference_matrix=reference_matrix,
        reference_records=tuple(reference_rows),
        path_to_success_row=path_to_success_row,
    )


def _validate_query_vector(vector: np.ndarray, expected_length: int) -> np.ndarray:
    query = np.asarray(vector, dtype=np.float64)
    if query.ndim != 1 or len(query) != expected_length:
        raise ValueError("Query vector must be one-dimensional and match the grid")
    if not np.all(np.isfinite(query)):
        raise ValueError("Query vector contains non-finite values")
    norm = float(np.linalg.norm(query))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Query vector has zero or invalid norm")
    return query


def compute_cosine_similarities(
    matrix: np.ndarray,
    query_vector: np.ndarray,
) -> np.ndarray:
    """Compute one query-to-library cosine vector, never a pairwise matrix."""
    library_matrix = np.asarray(matrix, dtype=np.float64)
    if library_matrix.ndim != 2:
        raise ValueError("Reference matrix must be two-dimensional")
    if not np.all(np.isfinite(library_matrix)):
        raise ValueError("Reference matrix contains non-finite values")
    query = _validate_query_vector(query_vector, library_matrix.shape[1])
    row_norms = np.linalg.norm(library_matrix, axis=1)
    if np.any(~np.isfinite(row_norms)) or np.any(row_norms <= 0):
        raise ValueError("Reference matrix contains a zero-norm or invalid vector")
    similarities = (library_matrix @ query) / (row_norms * np.linalg.norm(query))
    if not np.all(np.isfinite(similarities)):
        raise ValueError("Cosine similarity calculation produced non-finite values")
    return correct_cosine_boundaries(similarities)


def correct_cosine_boundaries(
    similarities: np.ndarray,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Clip only floating-point roundoff just outside the cosine interval."""
    values = np.asarray(similarities, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Cosine similarities must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("Cosine similarities contain non-finite values")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("Cosine boundary tolerance must be finite and non-negative")
    if np.any(values > 1.0 + tolerance) or np.any(values < -1.0 - tolerance):
        raise ValueError("Cosine similarity is materially outside the [-1, 1] range")
    return np.clip(values, -1.0, 1.0)


def _known_sample_id(record: dict[str, str] | QuerySpectrum) -> str | None:
    sample_id = (
        record.sample_id if isinstance(record, QuerySpectrum) else record.get("sample_id", "")
    ).strip()
    source = (
        record.sample_id_source
        if isinstance(record, QuerySpectrum)
        else record.get("sample_id_source", "")
    ).strip().casefold()
    if _is_missing(sample_id) or "conflict" in source or source == "missing":
        return None
    return sample_id


def _dedupe_identity(record: dict[str, str]) -> tuple[str, str]:
    sample_id = _known_sample_id(record)
    if sample_id is not None:
        return "known_sample_id", "sample_id:" + sample_id.casefold()
    measurement_stem = record.get("measurement_stem", "").strip()
    if not _is_missing(measurement_stem):
        return (
            "measurement_stem_fallback_for_unknown_sample_id",
            "measurement_stem:" + measurement_stem.casefold(),
        )
    return (
        "relative_path_fallback_for_unknown_sample_id",
        "relative_path:" + record["relative_path"].casefold(),
    )


def _reference_label_identity(record: dict[str, str]) -> tuple[str, str, str]:
    """Return display label, safe grouping basis, and deterministic group key."""
    raw_label = record.get("reference_label", "")
    if _is_missing(raw_label):
        relative_path = record["relative_path"]
        return (
            "Unknown",
            "relative_path_fallback_for_missing_reference_label",
            "missing_label_relative_path:" + relative_path.casefold(),
        )
    normalized = unicodedata.normalize("NFKC", raw_label)
    normalized = " ".join(normalized.strip().split())
    if not normalized:
        relative_path = record["relative_path"]
        return (
            "Unknown",
            "relative_path_fallback_for_missing_reference_label",
            "missing_label_relative_path:" + relative_path.casefold(),
        )
    return (
        normalized,
        "safe_unicode_whitespace_case_normalization",
        "reference_label:" + normalized.casefold(),
    )


def query_from_cache(library: ReferenceLibrary, relative_path: str) -> QuerySpectrum:
    cache_row = library.path_to_success_row.get(relative_path)
    if cache_row is None:
        raise ValueError(
            "Cached query relative_path is absent or did not process successfully: "
            + relative_path
        )
    record = library.all_success_records[cache_row]
    return QuerySpectrum(
        vector=library.all_success_matrix[cache_row].copy(),
        mode="cached_relative_path",
        relative_path=record["relative_path"],
        sample_id=record.get("sample_id", "Unknown"),
        sample_id_source=record.get("sample_id_source", "missing"),
        measurement_stem=record.get("measurement_stem", "Unknown"),
        reference_label=record.get("reference_label", "Unknown"),
        excitation_wavelength_nm=record.get("excitation_wavelength_nm", "Unknown"),
    )


def query_from_external_raw(
    raw_path: Path,
    *,
    sample_id: str = "Unknown",
    measurement_stem: str = "Unknown",
    reference_label: str = "Unknown",
    excitation_wavelength_nm: str = "Unknown",
) -> QuerySpectrum:
    """Apply the exact experiment 9.3 order to one external RAW TXT."""
    if raw_path.suffix.casefold() != ".txt":
        raise ValueError("External RAW query must use a .txt file")
    if not raw_path.is_file():
        raise ValueError(f"External RAW query does not exist: {raw_path}")
    try:
        content = raw_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Unable to read external RAW query: {exc}") from exc
    try:
        _raw, _corrected, normalized, _dynamic_range = (
            exp09._run_v1_preprocessing(content)
        )
    except exp09.ProcessingFailure as exc:
        raise ValueError(
            "External RAW preprocessing failed at "
            f"{exc.stage} ({exc.code}): {exc}"
        ) from exc
    source_x = np.asarray(normalized.spectral_axis, dtype=np.float64)
    source_y = np.asarray(normalized.spectral_data, dtype=np.float64)
    target_grid = exp09.make_trial_grid()
    try:
        exp09.interpolation.validate_spectrum(
            source_x,
            source_y,
            exp09.GRID_START_CM_1,
            exp09.GRID_END_CM_1,
        )
        vector = exp09.interpolation.linear_interpolate_no_extrapolation(
            source_x,
            source_y,
            target_grid,
        )
    except ValueError as exc:
        raise ValueError(
            "External RAW query cannot cover the common grid without "
            f"extrapolation: {exc}"
        ) from exc
    sample_source = "user_supplied" if not _is_missing(sample_id) else "missing"
    return QuerySpectrum(
        vector=np.asarray(vector, dtype=np.float64),
        mode="external_raw_txt",
        relative_path=f"external:{raw_path.name}",
        sample_id=sample_id,
        sample_id_source=sample_source,
        measurement_stem=measurement_stem,
        reference_label=reference_label,
        excitation_wavelength_nm=excitation_wavelength_nm,
    )


def search_topk(
    library: ReferenceLibrary,
    query: QuerySpectrum,
    k: int,
    *,
    result_level: str,
    exclude_self: bool,
    exclude_same_sample: bool,
) -> list[dict[str, object]]:
    """Return deterministic file, sample-ID, or reference-label Top-K hits."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("K must be a positive integer")
    if result_level not in {"file", "sample_id", "reference_label"}:
        raise ValueError(
            "result_level must be 'file', 'sample_id', or 'reference_label'"
        )
    similarities = compute_cosine_similarities(
        library.reference_matrix,
        query.vector,
    )
    query_sample_id = _known_sample_id(query)
    candidates: list[tuple[float, str, dict[str, str], str, str]] = []
    for row_number, record in enumerate(library.reference_records):
        relative_path = record["relative_path"]
        if exclude_self and relative_path == query.relative_path:
            continue
        candidate_sample_id = _known_sample_id(record)
        if (
            exclude_same_sample
            and query_sample_id is not None
            and candidate_sample_id is not None
            and candidate_sample_id.casefold() == query_sample_id.casefold()
        ):
            continue
        basis, identity_key = _dedupe_identity(record)
        candidates.append(
            (
                float(similarities[row_number]),
                relative_path,
                record,
                basis,
                identity_key,
            )
        )
    candidates.sort(key=lambda item: (-item[0], item[1].casefold(), item[1]))

    if result_level == "reference_label":
        grouped: dict[str, dict[str, object]] = {}
        for candidate in candidates:
            similarity, _path, record, _identity_basis, identity_key = candidate
            label_display, label_basis, label_key = _reference_label_identity(record)
            group = grouped.get(label_key)
            if group is None:
                group = {
                    "display": label_display,
                    "basis": label_basis,
                    "key": label_key,
                    "best": candidate,
                    "file_count": 0,
                    "sample_identity_keys": set(),
                }
                grouped[label_key] = group
            group["file_count"] = int(group["file_count"]) + 1
            sample_keys = group["sample_identity_keys"]
            assert isinstance(sample_keys, set)
            sample_keys.add(identity_key)
            best = group["best"]
            assert isinstance(best, tuple)
            if (
                similarity > best[0]
                or (
                    math.isclose(similarity, best[0], rel_tol=0.0, abs_tol=0.0)
                    and (record["relative_path"].casefold(), record["relative_path"])
                    < (best[1].casefold(), best[1])
                )
            ):
                group["best"] = candidate

        label_groups = list(grouped.values())
        label_groups.sort(
            key=lambda group: (
                -float(group["best"][0]),
                str(group["display"]).casefold(),
                str(group["key"]),
                str(group["best"][1]).casefold(),
                str(group["best"][1]),
            )
        )
        label_hits: list[dict[str, object]] = []
        for rank, group in enumerate(label_groups[:k], start=1):
            best = group["best"]
            assert isinstance(best, tuple)
            sample_keys = group["sample_identity_keys"]
            assert isinstance(sample_keys, set)
            label_hits.append(
                {
                    "rank": rank,
                    "cosine_similarity": float(best[0]),
                    "record": best[2],
                    "dedupe_identity_basis": "reference_label_group",
                    "dedupe_identity_key": str(group["key"]),
                    "reference_label_group_display": str(group["display"]),
                    "reference_label_grouping_basis": str(group["basis"]),
                    "reference_label_group_key": str(group["key"]),
                    "label_score_definition": "maximum_valid_reference_cosine",
                    "label_reference_file_count": int(group["file_count"]),
                    "label_distinguishable_sample_id_count": len(sample_keys),
                }
            )
        return label_hits

    if result_level == "sample_id":
        deduplicated: list[tuple[float, str, dict[str, str], str, str]] = []
        seen: set[str] = set()
        for candidate in candidates:
            identity_key = candidate[4]
            if identity_key in seen:
                continue
            seen.add(identity_key)
            deduplicated.append(candidate)
        candidates = deduplicated

    hits: list[dict[str, object]] = []
    for rank, (similarity, _path, record, basis, identity_key) in enumerate(
        candidates[:k], start=1
    ):
        hits.append(
            {
                "rank": rank,
                "cosine_similarity": similarity,
                "record": record,
                "dedupe_identity_basis": basis,
                "dedupe_identity_key": identity_key,
                "reference_label_group_display": "",
                "reference_label_grouping_basis": "",
                "reference_label_group_key": "",
                "label_score_definition": "",
                "label_reference_file_count": "",
                "label_distinguishable_sample_id_count": "",
            }
        )
    return hits


def _exclusion_rule(exclude_self: bool, exclude_same_sample: bool) -> str:
    rules: list[str] = []
    if exclude_self:
        rules.append("exclude_exact_query_relative_path")
    if exclude_same_sample:
        rules.append("exclude_same_known_sample_id_only")
    return ";".join(rules) if rules else "none"


def format_result_rows(
    scenario_id: str,
    query: QuerySpectrum,
    hits: list[dict[str, object]],
    *,
    result_level: str,
    exclude_self: bool,
    exclude_same_sample: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for hit in hits:
        record = hit["record"]
        assert isinstance(record, dict)
        rows.append(
            {
                "scenario_id": scenario_id,
                "query_mode": query.mode,
                "query_relative_path": query.relative_path,
                "query_sample_id": query.sample_id,
                "query_sample_id_source": query.sample_id_source,
                "query_measurement_stem": query.measurement_stem,
                "query_reference_label": query.reference_label,
                "query_excitation_wavelength_nm": query.excitation_wavelength_nm,
                "query_known_sample_id": _known_sample_id(query) is not None,
                "result_level": result_level,
                "rank": hit["rank"],
                "candidate_relative_path": record.get("relative_path", ""),
                "candidate_file_name": record.get("file_name", ""),
                "candidate_sample_id": record.get("sample_id", ""),
                "candidate_sample_id_source": record.get("sample_id_source", ""),
                "candidate_measurement_stem": record.get("measurement_stem", ""),
                "candidate_reference_label": record.get("reference_label", ""),
                "candidate_reference_label_status": record.get(
                    "reference_label_status", ""
                ),
                "candidate_excitation_wavelength_nm": record.get(
                    "excitation_wavelength_nm", ""
                ),
                "candidate_excitation_wavelength_source": record.get(
                    "excitation_wavelength_source", ""
                ),
                "cosine_similarity": hit["cosine_similarity"],
                "dedupe_identity_basis": hit["dedupe_identity_basis"],
                "dedupe_identity_key": hit["dedupe_identity_key"],
                "reference_label_group_display": hit.get(
                    "reference_label_group_display", ""
                ),
                "reference_label_grouping_basis": hit.get(
                    "reference_label_grouping_basis", ""
                ),
                "reference_label_group_key": hit.get(
                    "reference_label_group_key", ""
                ),
                "label_score_definition": hit.get("label_score_definition", ""),
                "label_reference_file_count": hit.get(
                    "label_reference_file_count", ""
                ),
                "label_distinguishable_sample_id_count": hit.get(
                    "label_distinguishable_sample_id_count", ""
                ),
                "exclude_self": exclude_self,
                "exclude_same_known_sample_id": exclude_same_sample,
                "exclusion_rule": _exclusion_rule(
                    exclude_self, exclude_same_sample
                ),
                "candidate_requires_review": record.get("requires_review", ""),
                "candidate_review_reasons": record.get("review_reasons", ""),
                "candidate_audit_metadata_warnings": record.get(
                    "audit_metadata_warnings", ""
                ),
                "candidate_has_missing_metadata": record.get(
                    "has_missing_metadata", ""
                ),
                "candidate_has_conflict": record.get("has_conflict", ""),
                "candidate_metadata_status": record.get("metadata_status", ""),
                "candidate_metadata_flags": record.get("metadata_flags", ""),
            }
        )
    return rows


def _paths(hits: list[dict[str, object]], count: int) -> list[str]:
    return [str(hit["record"]["relative_path"]) for hit in hits[:count]]


def _label_displays(hits: list[dict[str, object]], count: int) -> list[str]:
    return [str(hit["reference_label_group_display"]) for hit in hits[:count]]


def _ranking_is_valid(hits: list[dict[str, object]]) -> bool:
    keys: list[tuple[object, ...]] = []
    for hit in hits:
        if hit.get("reference_label_group_key"):
            keys.append(
                (
                    -float(hit["cosine_similarity"]),
                    str(hit["reference_label_group_display"]).casefold(),
                    str(hit["reference_label_group_key"]),
                    str(hit["record"]["relative_path"]).casefold(),
                    str(hit["record"]["relative_path"]),
                )
            )
        else:
            keys.append(
                (
                    -float(hit["cosine_similarity"]),
                    str(hit["record"]["relative_path"]).casefold(),
                    str(hit["record"]["relative_path"]),
                )
            )
    return keys == sorted(keys)


def select_diagnostic_queries(
    library: ReferenceLibrary,
    count: int = 3,
) -> tuple[list[dict[str, str]], str]:
    """Choose deterministic queries with distinct labels and wavelengths."""
    if count < 1:
        raise ValueError("Diagnostic query count must be positive")
    records = sorted(
        library.reference_records,
        key=lambda record: (record["relative_path"].casefold(), record["relative_path"]),
    )
    known_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        known = _known_sample_id(record)
        if known is not None:
            known_groups[known.casefold()].append(record)
    duplicate_anchor = next(
        (
            group[0]
            for _sample_id, group in sorted(known_groups.items())
            if len(group) > 1
        ),
        None,
    )
    selected: list[dict[str, str]] = []
    used_labels: set[str] = set()
    used_wavelengths: set[str] = set()

    def add_if_distinct(record: dict[str, str]) -> bool:
        label = record.get("reference_label", "").strip().casefold()
        wavelength = record.get("excitation_wavelength_nm", "").strip().casefold()
        if _is_missing(label) or _is_missing(wavelength):
            return False
        if label in used_labels or wavelength in used_wavelengths:
            return False
        selected.append(record)
        used_labels.add(label)
        used_wavelengths.add(wavelength)
        return True

    if duplicate_anchor is not None:
        add_if_distinct(duplicate_anchor)
    for record in records:
        if len(selected) >= count:
            break
        if record in selected:
            continue
        add_if_distinct(record)
    if len(selected) < count:
        raise ValueError(
            "Reference library cannot provide the requested number of queries "
            "with distinct reference labels and excitation wavelengths"
        )
    method = (
        "deterministic_relative_path_order_with_duplicate_known_sample_anchor_"
        "and_distinct_reference_label_and_excitation"
    )
    return selected, method


def _scenario_search(
    library: ReferenceLibrary,
    query: QuerySpectrum,
    k: int,
    scenario_id: str,
    exclude_self: bool,
    exclude_same_sample: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    file_hits = search_topk(
        library,
        query,
        k,
        result_level="file",
        exclude_self=exclude_self,
        exclude_same_sample=exclude_same_sample,
    )
    sample_hits = search_topk(
        library,
        query,
        k,
        result_level="sample_id",
        exclude_self=exclude_self,
        exclude_same_sample=exclude_same_sample,
    )
    label_hits = search_topk(
        library,
        query,
        k,
        result_level="reference_label",
        exclude_self=exclude_self,
        exclude_same_sample=exclude_same_sample,
    )
    rows = format_result_rows(
        scenario_id,
        query,
        file_hits,
        result_level="file",
        exclude_self=exclude_self,
        exclude_same_sample=exclude_same_sample,
    )
    rows.extend(
        format_result_rows(
            scenario_id,
            query,
            sample_hits,
            result_level="sample_id",
            exclude_self=exclude_self,
            exclude_same_sample=exclude_same_sample,
        )
    )
    rows.extend(
        format_result_rows(
            scenario_id,
            query,
            label_hits,
            result_level="reference_label",
            exclude_self=exclude_self,
            exclude_same_sample=exclude_same_sample,
        )
    )
    return rows, {
        "scenario_id": scenario_id,
        "exclude_self": exclude_self,
        "exclude_same_known_sample_id": exclude_same_sample,
        "query_known_sample_id": _known_sample_id(query),
        "file_result_count": len(file_hits),
        "sample_id_result_count": len(sample_hits),
        "reference_label_result_count": len(label_hits),
        "file_top5_relative_paths": _paths(file_hits, 5),
        "file_top10_relative_paths": _paths(file_hits, 10),
        "sample_id_top5_relative_paths": _paths(sample_hits, 5),
        "sample_id_top10_relative_paths": _paths(sample_hits, 10),
        "reference_label_top5": _label_displays(label_hits, 5),
        "reference_label_top10": _label_displays(label_hits, 10),
        "reference_label_top5_best_evidence_paths": _paths(label_hits, 5),
        "reference_label_top10_best_evidence_paths": _paths(label_hits, 10),
        "file_top5_sort_valid": _ranking_is_valid(file_hits[:5]),
        "file_top10_sort_valid": _ranking_is_valid(file_hits[:10]),
        "sample_id_top5_sort_valid": _ranking_is_valid(sample_hits[:5]),
        "sample_id_top10_sort_valid": _ranking_is_valid(sample_hits[:10]),
        "reference_label_top5_sort_valid": _ranking_is_valid(label_hits[:5]),
        "reference_label_top10_sort_valid": _ranking_is_valid(label_hits[:10]),
        "file_and_sample_id_results_differ": _paths(file_hits, k)
        != _paths(sample_hits, k),
        "file_and_reference_label_results_differ": _paths(file_hits, k)
        != _paths(label_hits, k),
    }


def run_diagnostics(
    library: ReferenceLibrary,
    k: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    diagnostic_k = max(k, 10)
    selected_records, selection_method = select_diagnostic_queries(library, 3)
    all_rows: list[dict[str, object]] = []
    query_summaries: list[dict[str, object]] = []
    for query_number, record in enumerate(selected_records, start=1):
        query = query_from_cache(library, record["relative_path"])
        self_similarity = float(
            compute_cosine_similarities(query.vector.reshape(1, -1), query.vector)[0]
        )
        scenario_summaries: list[dict[str, object]] = []
        for suffix, exclude_self, exclude_same_sample in (
            ("include_self", False, False),
            ("exclude_self", True, False),
            ("exclude_self_and_same_sample", True, True),
        ):
            scenario_id = f"query_{query_number}_{suffix}"
            rows, scenario_summary = _scenario_search(
                library,
                query,
                diagnostic_k,
                scenario_id,
                exclude_self,
                exclude_same_sample,
            )
            all_rows.extend(rows)
            scenario_summaries.append(scenario_summary)
        excluded_self_rows = [
            row
            for row in all_rows
            if row["scenario_id"] == f"query_{query_number}_exclude_self"
        ]
        excluded_same_rows = [
            row
            for row in all_rows
            if row["scenario_id"]
            == f"query_{query_number}_exclude_self_and_same_sample"
        ]
        known_id = _known_sample_id(query)
        query_summaries.append(
            {
                "query_number": query_number,
                "relative_path": query.relative_path,
                "sample_id": query.sample_id,
                "sample_id_source": query.sample_id_source,
                "reference_label": query.reference_label,
                "excitation_wavelength_nm": query.excitation_wavelength_nm,
                "self_cosine_similarity": self_similarity,
                "self_similarity_approximately_one": math.isclose(
                    self_similarity, 1.0, rel_tol=1e-12, abs_tol=1e-12
                ),
                "self_absent_after_exact_path_exclusion": all(
                    row["candidate_relative_path"] != query.relative_path
                    for row in excluded_self_rows
                ),
                "same_known_sample_id_exclusion_status": (
                    "verified_absent"
                    if known_id is not None
                    and all(
                        str(row["candidate_sample_id"]).casefold()
                        != known_id.casefold()
                        or "conflict"
                        in str(row["candidate_sample_id_source"]).casefold()
                        for row in excluded_same_rows
                    )
                    else "not_applicable_unknown_query_sample_id"
                    if known_id is None
                    else "failed"
                ),
                "scenarios": scenario_summaries,
            }
        )
    return all_rows, {
        "mode": "diagnostic",
        "requested_k": k,
        "effective_diagnostic_k": diagnostic_k,
        "query_selection_method": selection_method,
        "query_count": len(query_summaries),
        "queries": query_summaries,
        "all_self_similarity_checks_passed": all(
            bool(query["self_similarity_approximately_one"])
            for query in query_summaries
        ),
        "all_exact_self_exclusion_checks_passed": all(
            bool(query["self_absent_after_exact_path_exclusion"])
            for query in query_summaries
        ),
        "file_vs_sample_id_difference_observed": any(
            bool(scenario["file_and_sample_id_results_differ"])
            for query in query_summaries
            for scenario in query["scenarios"]
        ),
        "file_vs_reference_label_difference_observed": any(
            bool(scenario["file_and_reference_label_results_differ"])
            for query in query_summaries
            for scenario in query["scenarios"]
        ),
    }


def _library_summary(library: ReferenceLibrary) -> dict[str, object]:
    known_sample_count = len(
        {
            known.casefold()
            for record in library.reference_records
            if (known := _known_sample_id(record)) is not None
        }
    )
    fallback_counts = Counter(
        _dedupe_identity(record)[0] for record in library.reference_records
    )
    return {
        "successful_cached_spectrum_count": len(library.all_success_records),
        "reference_library_count": len(library.reference_records),
        "known_sample_id_count": known_sample_count,
        "dedupe_identity_basis_counts": dict(sorted(fallback_counts.items())),
        "grid": {
            "start_cm-1": float(library.grid[0]),
            "end_cm-1": float(library.grid[-1]),
            "step_cm-1": float(library.grid[1] - library.grid[0]),
            "point_count": len(library.grid),
        },
    }


def build_summary(
    cache_path: Path,
    index_path: Path,
    library: ReferenceLibrary,
    run_summary: dict[str, object],
) -> dict[str, object]:
    return {
        "experiment": "10.1",
        "cache_path": str(cache_path),
        "index_path": str(index_path),
        "cache_loaded_with_allow_pickle": False,
        "similarity_metric": "raw_cosine_similarity",
        "cosine_boundary_rule": (
            "Values at most 1e-12 outside [-1, 1] are clipped as floating-point "
            "roundoff; non-finite or materially out-of-range values are rejected."
        ),
        "similarity_interpretation": (
            "Cosine similarity is a spectral-shape baseline score, not confidence, "
            "probability, classification accuracy, or validated mineral identity."
        ),
        "reference_label_note": (
            "reference_label is a candidate metadata label and is not independent "
            "mineral ground truth; some labels may not be mineral names."
        ),
        "reference_label_scoring": {
            "score": "maximum cosine similarity among all valid references in label",
            "aggregation_scope": (
                "all candidates remaining after query exclusions, before any Top-K "
                "file truncation"
            ),
            "normalization": (
                "NFKC Unicode normalization, surrounding/internal whitespace "
                "normalization, and case-insensitive grouping only"
            ),
            "missing_label_rule": (
                "missing labels remain separate using relative_path fallback"
            ),
            "interpretation": (
                "label score is neither probability, confidence, nor classification "
                "accuracy"
            ),
        },
        "independence_note": (
            "Different files or Sample_ID values are not automatically proven "
            "independent physical specimens. RAW/Processed derivatives are not "
            "treated as independent retrieval evidence."
        ),
        "preprocessing_order": list(exp09.PROCESSING_ORDER),
        "normalization_stage": exp09.NORMALIZATION_STAGE,
        "library": _library_summary(library),
        "run": run_summary,
    }


def _output_paths(output_dir: Path) -> tuple[Path, Path]:
    return (
        output_dir / "rruff_topk_results.csv",
        output_dir / "rruff_topk_summary.json",
    )


def write_outputs(
    output_dir: Path,
    result_rows: list[dict[str, object]],
    summary: dict[str, object],
) -> tuple[Path, Path]:
    paths = _output_paths(output_dir)
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Top-K output already exists: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path, summary_path = paths
    with results_path.open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(result_rows)
    with summary_path.open("x", encoding="utf-8-sig", newline="") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    return paths


def run_cli_operation(arguments: argparse.Namespace) -> tuple[
    list[dict[str, object]], dict[str, object], tuple[Path, Path]
]:
    library = load_reference_library(arguments.cache, arguments.index)
    if arguments.diagnostic:
        result_rows, run_summary = run_diagnostics(library, arguments.k)
    else:
        if arguments.query_relative_path is not None:
            query = query_from_cache(library, arguments.query_relative_path)
        else:
            query = query_from_external_raw(
                arguments.query_raw,
                sample_id=arguments.query_sample_id,
                measurement_stem=arguments.query_measurement_stem,
                reference_label=arguments.query_reference_label,
                excitation_wavelength_nm=arguments.query_excitation_wavelength,
            )
        result_rows, scenario_summary = _scenario_search(
            library,
            query,
            arguments.k,
            "single_query",
            exclude_self=not arguments.include_self,
            exclude_same_sample=arguments.exclude_same_sample,
        )
        run_summary = {
            "mode": query.mode,
            "requested_k": arguments.k,
            "query": {
                "relative_path": query.relative_path,
                "sample_id": query.sample_id,
                "sample_id_source": query.sample_id_source,
                "measurement_stem": query.measurement_stem,
                "reference_label": query.reference_label,
                "excitation_wavelength_nm": query.excitation_wavelength_nm,
            },
            "scenario": scenario_summary,
        }
    summary = build_summary(arguments.cache, arguments.index, library, run_summary)
    paths = write_outputs(arguments.output_dir, result_rows, summary)
    return result_rows, summary, paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic cosine Top-K retrieval on the RRUFF cache."
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=10)
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--diagnostic", action="store_true")
    query_group.add_argument("--query-relative-path")
    query_group.add_argument("--query-raw", type=Path)
    parser.add_argument("--include-self", action="store_true")
    parser.add_argument("--exclude-same-sample", action="store_true")
    parser.add_argument("--query-sample-id", default="Unknown")
    parser.add_argument("--query-measurement-stem", default="Unknown")
    parser.add_argument("--query-reference-label", default="Unknown")
    parser.add_argument("--query-excitation-wavelength", default="Unknown")
    return parser


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()
    if arguments.k < 1:
        parser.error("--k must be a positive integer")
    try:
        rows, summary, paths = run_cli_operation(arguments)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        "Reference library: "
        f"{summary['library']['reference_library_count']} spectra"
    )
    print(f"Result rows: {len(rows)}")
    print(f"Results CSV: {paths[0]}")
    print(f"Summary JSON: {paths[1]}")


if __name__ == "__main__":
    main()
