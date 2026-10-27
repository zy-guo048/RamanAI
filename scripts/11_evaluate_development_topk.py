"""Evaluate the frozen experiment 11 development split with experiment 10 Top-K."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP10_SCRIPT = PROJECT_ROOT / "scripts" / "10_search_rruff_topk.py"
EXP11_SCRIPT = PROJECT_ROOT / "scripts" / "11_build_evaluation_protocol.py"
DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "docs" / "evaluation" / "exp11_v1"
DEFAULT_LOCK = DEFAULT_SNAPSHOT_DIR / "DATA_SOURCE_LOCK.json"
DEFAULT_MANIFEST = DEFAULT_SNAPSHOT_DIR / "evaluation_group_manifest.csv"
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "exp11_development_baseline"
TOP_K_VALUES = (1, 3, 5)
LEVELS = ("file", "sample_id", "reference_label")

RECORD_FIELDS = [
    "query_relative_path",
    "query_cache_row_index",
    "query_sample_id",
    "query_measurement_stem",
    "query_reference_label",
    "query_reference_label_group_key",
    "query_excitation_wavelength_nm",
    "query_measurement_count",
]
for _level in LEVELS:
    RECORD_FIELDS.extend(
        [
            f"{_level}_hit_at_1",
            f"{_level}_hit_at_3",
            f"{_level}_hit_at_5",
            f"{_level}_matching_rank",
            f"{_level}_top1_reference_label",
            f"{_level}_top1_relative_path",
            f"{_level}_top1_cosine_similarity",
            f"{_level}_top5_reference_labels",
        ]
    )

SAMPLE_FIELDS = [
    "query_sample_id",
    "query_reference_label",
    "query_reference_label_group_key",
    "query_spectrum_count",
]
for _level in LEVELS:
    SAMPLE_FIELDS.extend(
        [
            f"{_level}_mean_hit_at_1",
            f"{_level}_mean_hit_at_3",
            f"{_level}_mean_hit_at_5",
        ]
    )

LABEL_ERROR_FIELDS = [
    "row_type",
    "reference_label",
    "reference_label_group_key",
    "development_sample_id_count",
    "development_spectrum_count",
    "gallery_file_count",
    "gallery_sample_id_count",
    "result_level",
    "k",
    "sample_equal_hit_rate",
    "query_relative_path",
    "query_sample_id",
    "correct_label_rank",
    "top_candidate_reference_label",
    "top_candidate_relative_path",
    "top_candidate_cosine_similarity",
    "top5_reference_labels",
]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp10 = _load_module("raman_exp10_topk_for_exp11", EXP10_SCRIPT)
exp11 = _load_module("raman_exp11_protocol_for_evaluation", EXP11_SCRIPT)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Required CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def _parse_bool(value: str | None, field: str) -> bool:
    normalized = (value or "").strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid boolean value for {field}: {value!r}")


def _verify_locked_file(path: Path, specification: dict[str, object], name: str) -> None:
    if not path.is_file():
        raise ValueError(f"Locked {name} file does not exist: {path}")
    expected_size = int(specification["size_bytes"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"Locked {name} size mismatch: expected {expected_size}, got {actual_size}"
        )
    expected_hash = str(specification["sha256"])
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Locked {name} SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )


def validate_source_lock(
    lock_path: Path,
    snapshot_dir: Path,
    cache_path: Path,
    index_path: Path,
) -> dict[str, object]:
    """Validate every frozen snapshot and experiment 9.3 source before evaluation."""
    if not lock_path.is_file():
        raise ValueError(f"Data source lock does not exist: {lock_path}")
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read data source lock: {exc}") from exc
    snapshots = lock.get("evaluation_snapshot")
    sources = lock.get("source_data")
    frozen = lock.get("frozen_evaluation")
    if not isinstance(snapshots, dict) or not isinstance(sources, dict) or not isinstance(frozen, dict):
        raise ValueError("Data source lock is missing required sections")
    for filename in (
        "evaluation_group_manifest.csv",
        "reference_label_sample_distribution.csv",
        "evaluation_group_audit.json",
        "EVALUATION_PROTOCOL.md",
    ):
        specification = snapshots.get(filename)
        if not isinstance(specification, dict):
            raise ValueError(f"Data source lock is missing snapshot entry: {filename}")
        _verify_locked_file(snapshot_dir / filename, specification, filename)
    for key, path in (
        ("experiment_9_3_npz", cache_path),
        ("experiment_9_3_index", index_path),
    ):
        specification = sources.get(key)
        if not isinstance(specification, dict):
            raise ValueError(f"Data source lock is missing source entry: {key}")
        _verify_locked_file(path, specification, key)
    return lock


def _booleanized_for_leakage(rows: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    converted: list[dict[str, object]] = []
    boolean_fields = (
        "known_sample_id_usable",
        "sample_id_metadata_conflict",
        "strict_closed_set_eligible",
    )
    for source in rows:
        row: dict[str, object] = dict(source)
        for field in boolean_fields:
            row[field] = _parse_bool(source.get(field), field)
        converted.append(row)
    return converted


def validate_frozen_manifest(
    manifest_path: Path,
    lock: dict[str, object],
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Validate frozen identities, exact split sizes, and all separation rules."""
    rows = _read_csv(manifest_path)
    if not rows:
        raise ValueError("Frozen evaluation manifest is empty")
    paths = [row.get("relative_path", "").strip() for row in rows]
    duplicates = [path for path, count in Counter(paths).items() if not path or count > 1]
    if duplicates:
        raise ValueError("Frozen manifest has empty or duplicate relative_path values")

    frozen = lock["frozen_evaluation"]
    assert isinstance(frozen, dict)
    if sha256_file(manifest_path) != str(frozen["manifest_sha256"]):
        raise ValueError("Frozen Manifest SHA-256 does not match the data source lock")
    if exp11._identity_hash(rows) != str(frozen["all_group_identity_sha256"]):
        raise ValueError("Frozen all-group identity hash mismatch")
    final_hash = exp11._identity_hash(rows, {exp11.GROUP_FINAL})
    if final_hash != str(frozen["final_test_identity_sha256"]):
        raise ValueError("Frozen final-test identity SHA-256 mismatch")

    leakage = exp11.audit_leakage(_booleanized_for_leakage(rows))
    exp11.require_leakage_audit_passed(leakage)
    expected = frozen.get("expected_counts")
    if not isinstance(expected, dict):
        raise ValueError("Data source lock is missing expected split counts")

    groups = {
        name: [row for row in rows if row.get("evaluation_group") == name]
        for name in (exp11.GROUP_GALLERY, exp11.GROUP_DEVELOPMENT, exp11.GROUP_FINAL)
    }
    count_requirements = {
        exp11.GROUP_GALLERY: (
            int(expected["reference_gallery_records"]),
            int(expected["reference_gallery_known_sample_ids"]),
        ),
        exp11.GROUP_DEVELOPMENT: (
            int(expected["development_query_records"]),
            int(expected["development_query_known_sample_ids"]),
        ),
        exp11.GROUP_FINAL: (
            int(expected["final_test_query_records"]),
            int(expected["final_test_query_known_sample_ids"]),
        ),
    }
    group_sets: dict[str, dict[str, set[str]]] = {}
    for name, group_rows in groups.items():
        sample_ids = {row["sample_id"].casefold() for row in group_rows}
        expected_records, expected_samples = count_requirements[name]
        if len(group_rows) != expected_records or len(sample_ids) != expected_samples:
            raise ValueError(
                f"Frozen {name} size mismatch: {len(group_rows)} records, "
                f"{len(sample_ids)} Sample_IDs"
            )
        group_sets[name] = {
            "sample_ids": sample_ids,
            "paths": {row["relative_path"] for row in group_rows},
            "stems": {row["measurement_stem"].casefold() for row in group_rows},
        }
    names = list(group_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            for identity_type in ("sample_ids", "paths", "stems"):
                if group_sets[left][identity_type] & group_sets[right][identity_type]:
                    raise ValueError(
                        f"Frozen groups {left} and {right} overlap by {identity_type}"
                    )
    gallery_labels = {
        row["reference_label_group_key"] for row in groups[exp11.GROUP_GALLERY]
    }
    query_labels = {
        row["reference_label_group_key"]
        for name in (exp11.GROUP_DEVELOPMENT, exp11.GROUP_FINAL)
        for row in groups[name]
    }
    if query_labels - gallery_labels:
        raise ValueError("One or more formal query labels are absent from gallery")
    if len(gallery_labels) != int(expected["strict_closed_set_reference_labels"]):
        raise ValueError("Strict closed-set reference-label count mismatch")
    return rows, {
        "leakage": leakage,
        "final_test_identity_sha256": final_hash,
        "gallery_record_count": len(groups[exp11.GROUP_GALLERY]),
        "development_record_count": len(groups[exp11.GROUP_DEVELOPMENT]),
        "final_test_record_count": len(groups[exp11.GROUP_FINAL]),
        "gallery_sample_id_count": len(group_sets[exp11.GROUP_GALLERY]["sample_ids"]),
        "development_sample_id_count": len(group_sets[exp11.GROUP_DEVELOPMENT]["sample_ids"]),
        "final_test_sample_id_count": len(group_sets[exp11.GROUP_FINAL]["sample_ids"]),
        "strict_label_count": len(gallery_labels),
    }


def _reconcile_manifest_record(
    manifest: dict[str, str], index: dict[str, str]
) -> None:
    for field in (
        "relative_path",
        "cache_row_index",
        "measurement_stem",
        "sample_id",
        "sample_id_source",
        "reference_label",
        "excitation_wavelength_nm",
    ):
        if manifest.get(field, "") != index.get(field, ""):
            raise ValueError(
                f"Manifest/index mismatch for {manifest.get('relative_path', '')}: {field}"
            )
    if index.get("processing_status", "").casefold() != "success":
        raise ValueError("Formal evaluation record is not a successful cached spectrum")
    if not _parse_bool(index.get("reference_candidate"), "reference_candidate"):
        raise ValueError("Formal evaluation record is not a reference candidate")
    _display, _basis, label_key = exp10._reference_label_identity(index)
    if label_key != manifest.get("reference_label_group_key"):
        raise ValueError(
            "Manifest/index frozen reference_label_group_key mismatch for "
            + manifest["relative_path"]
        )


def prepare_frozen_data(
    cache_path: Path,
    index_path: Path,
    manifest_rows: list[dict[str, str]],
):
    """Create only the frozen gallery and development matrix views."""
    full_library = exp10.load_reference_library(cache_path, index_path)
    success_by_path = {
        row["relative_path"]: row for row in full_library.all_success_records
    }
    formal_rows = [
        row
        for row in manifest_rows
        if row["evaluation_group"]
        in {exp11.GROUP_GALLERY, exp11.GROUP_DEVELOPMENT, exp11.GROUP_FINAL}
    ]
    for manifest in formal_rows:
        index = success_by_path.get(manifest["relative_path"])
        if index is None:
            raise ValueError(
                "Frozen formal record is absent from successful cache index: "
                + manifest["relative_path"]
            )
        _reconcile_manifest_record(manifest, index)
        if int(manifest["cache_row_index"]) != full_library.path_to_success_row[
            manifest["relative_path"]
        ]:
            raise ValueError("Frozen cache_row_index does not match the NPZ row identity")

    gallery_manifest = [
        row for row in manifest_rows if row["evaluation_group"] == exp11.GROUP_GALLERY
    ]
    development_manifest = [
        row
        for row in manifest_rows
        if row["evaluation_group"] == exp11.GROUP_DEVELOPMENT
    ]
    gallery_indices = np.asarray(
        [full_library.path_to_success_row[row["relative_path"]] for row in gallery_manifest],
        dtype=int,
    )
    development_indices = np.asarray(
        [full_library.path_to_success_row[row["relative_path"]] for row in development_manifest],
        dtype=int,
    )
    gallery_records = tuple(success_by_path[row["relative_path"]] for row in gallery_manifest)
    gallery_matrix = full_library.all_success_matrix[gallery_indices].copy()
    development_matrix = full_library.all_success_matrix[development_indices].copy()
    gallery_library = exp10.ReferenceLibrary(
        grid=full_library.grid.copy(),
        all_success_matrix=gallery_matrix,
        all_success_records=gallery_records,
        reference_matrix=gallery_matrix,
        reference_records=gallery_records,
        path_to_success_row={
            record["relative_path"]: index
            for index, record in enumerate(gallery_records)
        },
    )
    return gallery_library, development_manifest, development_matrix


def _hit_label(hit: dict[str, object]) -> tuple[str, str]:
    if hit.get("reference_label_group_key"):
        return (
            str(hit["reference_label_group_key"]),
            str(hit["reference_label_group_display"]),
        )
    record = hit["record"]
    assert isinstance(record, dict)
    display, _basis, key = exp10._reference_label_identity(record)
    return key, display


def _rank_summary(
    hits: list[dict[str, object]], target_key: str
) -> dict[str, object]:
    ranked_keys = [_hit_label(hit)[0] for hit in hits]
    matching_rank = next(
        (index + 1 for index, key in enumerate(ranked_keys) if key == target_key),
        None,
    )
    top = hits[0]
    top_record = top["record"]
    assert isinstance(top_record, dict)
    _top_key, top_display = _hit_label(top)
    return {
        "hit_at_1": int(target_key in ranked_keys[:1]),
        "hit_at_3": int(target_key in ranked_keys[:3]),
        "hit_at_5": int(target_key in ranked_keys[:5]),
        "matching_rank": matching_rank if matching_rank is not None else "",
        "top1_reference_label": top_display,
        "top1_relative_path": top_record["relative_path"],
        "top1_cosine_similarity": float(top["cosine_similarity"]),
        "top5_reference_labels": " | ".join(
            _hit_label(hit)[1] for hit in hits[:5]
        ),
    }


def evaluate_development_queries(
    gallery_library,
    development_manifest: list[dict[str, str]],
    development_matrix: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, dict[str, list[dict[str, object]]]]]:
    """Evaluate all development spectra against gallery at all three levels."""
    if len(development_manifest) != len(development_matrix):
        raise ValueError("Development metadata and matrix row counts differ")
    measurement_counts = Counter(row["sample_id"] for row in development_manifest)
    rows: list[dict[str, object]] = []
    rankings: dict[str, dict[str, list[dict[str, object]]]] = {}
    level_k = {
        "file": len(gallery_library.reference_records),
        "sample_id": len(
            {row["sample_id"].casefold() for row in gallery_library.reference_records}
        ),
        "reference_label": len(
            {
                exp10._reference_label_identity(row)[2]
                for row in gallery_library.reference_records
            }
        ),
    }
    for manifest, vector in zip(development_manifest, development_matrix, strict=True):
        query = exp10.QuerySpectrum(
            vector=np.asarray(vector, dtype=np.float64),
            mode="frozen_development_query",
            relative_path=manifest["relative_path"],
            sample_id=manifest["sample_id"],
            sample_id_source=manifest["sample_id_source"],
            measurement_stem=manifest["measurement_stem"],
            reference_label=manifest["reference_label"],
            excitation_wavelength_nm=manifest["excitation_wavelength_nm"],
        )
        row: dict[str, object] = {
            "query_relative_path": manifest["relative_path"],
            "query_cache_row_index": manifest["cache_row_index"],
            "query_sample_id": manifest["sample_id"],
            "query_measurement_stem": manifest["measurement_stem"],
            "query_reference_label": manifest["reference_label_group_display"],
            "query_reference_label_group_key": manifest["reference_label_group_key"],
            "query_excitation_wavelength_nm": manifest["excitation_wavelength_nm"],
            "query_measurement_count": measurement_counts[manifest["sample_id"]],
        }
        query_rankings: dict[str, list[dict[str, object]]] = {}
        for level in LEVELS:
            hits = exp10.search_topk(
                gallery_library,
                query,
                level_k[level],
                result_level=level,
                exclude_self=False,
                exclude_same_sample=False,
            )
            if not hits:
                raise ValueError(f"No {level} candidates are available")
            query_rankings[level] = hits
            for key, value in _rank_summary(
                hits, manifest["reference_label_group_key"]
            ).items():
                row[f"{level}_{key}"] = value
        rows.append(row)
        rankings[manifest["relative_path"]] = query_rankings
    return rows, rankings


def aggregate_by_sample_id(
    record_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in record_rows:
        grouped[str(row["query_sample_id"])].append(row)
    result: list[dict[str, object]] = []
    for sample_id in sorted(grouped, key=lambda value: (value.casefold(), value)):
        rows = grouped[sample_id]
        label_keys = {str(row["query_reference_label_group_key"]) for row in rows}
        if len(label_keys) != 1:
            raise ValueError(f"Development Sample_ID spans labels: {sample_id}")
        aggregate: dict[str, object] = {
            "query_sample_id": sample_id,
            "query_reference_label": rows[0]["query_reference_label"],
            "query_reference_label_group_key": rows[0][
                "query_reference_label_group_key"
            ],
            "query_spectrum_count": len(rows),
        }
        for level in LEVELS:
            for k in TOP_K_VALUES:
                aggregate[f"{level}_mean_hit_at_{k}"] = float(
                    np.mean([float(row[f"{level}_hit_at_{k}"]) for row in rows])
                )
        result.append(aggregate)
    return result


def _gallery_label_counts(gallery_records: Iterable[dict[str, str]]) -> dict[str, tuple[int, int]]:
    files: Counter[str] = Counter()
    samples: dict[str, set[str]] = defaultdict(set)
    for record in gallery_records:
        _display, _basis, key = exp10._reference_label_identity(record)
        files[key] += 1
        samples[key].add(record["sample_id"].casefold())
    return {key: (files[key], len(samples[key])) for key in files}


def build_label_and_error_rows(
    record_rows: list[dict[str, object]],
    sample_rows: list[dict[str, object]],
    rankings: dict[str, dict[str, list[dict[str, object]]]],
    gallery_records: Iterable[dict[str, str]],
) -> list[dict[str, object]]:
    sample_by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    records_by_label: Counter[str] = Counter()
    display_by_label: dict[str, str] = {}
    for row in sample_rows:
        key = str(row["query_reference_label_group_key"])
        sample_by_label[key].append(row)
        display_by_label[key] = str(row["query_reference_label"])
    for row in record_rows:
        records_by_label[str(row["query_reference_label_group_key"])] += 1
    gallery_counts = _gallery_label_counts(gallery_records)
    output: list[dict[str, object]] = []
    for label_key in sorted(sample_by_label):
        samples = sample_by_label[label_key]
        file_count, sample_count = gallery_counts[label_key]
        for level in LEVELS:
            for k in TOP_K_VALUES:
                output.append(
                    {
                        "row_type": "reference_label_summary",
                        "reference_label": display_by_label[label_key],
                        "reference_label_group_key": label_key,
                        "development_sample_id_count": len(samples),
                        "development_spectrum_count": records_by_label[label_key],
                        "gallery_file_count": file_count,
                        "gallery_sample_id_count": sample_count,
                        "result_level": level,
                        "k": k,
                        "sample_equal_hit_rate": float(
                            np.mean(
                                [float(row[f"{level}_mean_hit_at_{k}"]) for row in samples]
                            )
                        ),
                        "query_relative_path": "",
                        "query_sample_id": "",
                        "correct_label_rank": "",
                        "top_candidate_reference_label": "",
                        "top_candidate_relative_path": "",
                        "top_candidate_cosine_similarity": "",
                        "top5_reference_labels": "",
                    }
                )
    for row in record_rows:
        path = str(row["query_relative_path"])
        for level in LEVELS:
            for k in TOP_K_VALUES:
                if int(row[f"{level}_hit_at_{k}"]) == 1:
                    continue
                hits = rankings[path][level]
                top = hits[0]
                top_record = top["record"]
                assert isinstance(top_record, dict)
                _top_key, top_display = _hit_label(top)
                file_count, sample_count = gallery_counts[
                    str(row["query_reference_label_group_key"])
                ]
                output.append(
                    {
                        "row_type": "error_case",
                        "reference_label": row["query_reference_label"],
                        "reference_label_group_key": row[
                            "query_reference_label_group_key"
                        ],
                        "development_sample_id_count": 1,
                        "development_spectrum_count": row["query_measurement_count"],
                        "gallery_file_count": file_count,
                        "gallery_sample_id_count": sample_count,
                        "result_level": level,
                        "k": k,
                        "sample_equal_hit_rate": "",
                        "query_relative_path": path,
                        "query_sample_id": row["query_sample_id"],
                        "correct_label_rank": row[f"{level}_matching_rank"],
                        "top_candidate_reference_label": top_display,
                        "top_candidate_relative_path": top_record["relative_path"],
                        "top_candidate_cosine_similarity": float(
                            top["cosine_similarity"]
                        ),
                        "top5_reference_labels": row[f"{level}_top5_reference_labels"],
                    }
                )
    return output


def _average_rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for index in order[cursor:end]:
            ranks[index] = average
        cursor = end
    return ranks


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    return float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1])


def build_metrics_summary(
    record_rows: list[dict[str, object]],
    sample_rows: list[dict[str, object]],
    label_error_rows: list[dict[str, object]],
) -> dict[str, object]:
    label_rows = [
        row for row in label_error_rows if row["row_type"] == "reference_label_summary"
    ]
    record_metrics: dict[str, float] = {}
    sample_metrics: dict[str, float] = {}
    macro_metrics: dict[str, float] = {}
    size_analysis: dict[str, object] = {}
    for level in LEVELS:
        for k in TOP_K_VALUES:
            metric = f"{level}_top_{k}"
            record_metrics[metric] = float(
                np.mean([float(row[f"{level}_hit_at_{k}"]) for row in record_rows])
            )
            sample_metrics[metric] = float(
                np.mean(
                    [float(row[f"{level}_mean_hit_at_{k}"]) for row in sample_rows]
                )
            )
            selected = [
                row
                for row in label_rows
                if row["result_level"] == level and int(row["k"]) == k
            ]
            macro_metrics[metric] = float(
                np.mean([float(row["sample_equal_hit_rate"]) for row in selected])
            )
            if not math.isclose(
                sample_metrics[metric], macro_metrics[metric], rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    "Sample_ID equal-weight and label macro metrics differ despite "
                    "one development Sample_ID per label"
                )
            file_counts = [float(row["gallery_file_count"]) for row in selected]
            sample_counts = [float(row["gallery_sample_id_count"]) for row in selected]
            outcomes = [float(row["sample_equal_hit_rate"]) for row in selected]
            size_analysis[metric] = {
                "gallery_file_count_pearson": _correlation(file_counts, outcomes),
                "gallery_file_count_spearman": _correlation(
                    _average_rank(file_counts), _average_rank(outcomes)
                ),
                "gallery_sample_id_count_pearson": _correlation(sample_counts, outcomes),
                "gallery_sample_id_count_spearman": _correlation(
                    _average_rank(sample_counts), _average_rank(outcomes)
                ),
            }
    return {
        "record_level_118_spectra": record_metrics,
        "sample_id_equal_weight_60_sample_ids": sample_metrics,
        "reference_label_macro_60_labels": macro_metrics,
        "sample_equal_equals_label_macro": True,
        "gallery_size_association": size_analysis,
        "gallery_size_caution": (
            "These descriptive correlations do not establish that gallery size "
            "causes retrieval success or failure."
        ),
    }


def _git_version() -> dict[str, object]:
    def run(*arguments: str) -> str:
        try:
            return subprocess.run(
                ["git", *arguments],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    return {
        "head": run("rev-parse", "HEAD"),
        "describe": run("describe", "--always", "--dirty"),
        "evaluation_script_sha256": sha256_file(Path(__file__)),
        "experiment_10_script_sha256": sha256_file(EXP10_SCRIPT),
    }


def output_paths(output_dir: Path) -> tuple[Path, Path, Path, Path]:
    return (
        output_dir / "development_query_record_results.csv",
        output_dir / "development_sample_id_results.csv",
        output_dir / "development_label_stats_and_errors.csv",
        output_dir / "development_baseline_summary.json",
    )


def write_outputs(
    output_dir: Path,
    record_rows: list[dict[str, object]],
    sample_rows: list[dict[str, object]],
    label_error_rows: list[dict[str, object]],
    summary: dict[str, object],
) -> tuple[Path, Path, Path, Path]:
    paths = output_paths(output_dir)
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Development baseline output already exists: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, fields, rows in (
        (paths[0], RECORD_FIELDS, record_rows),
        (paths[1], SAMPLE_FIELDS, sample_rows),
        (paths[2], LABEL_ERROR_FIELDS, label_error_rows),
    ):
        with path.open("x", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with paths[3].open("x", encoding="utf-8-sig", newline="") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    return paths


def run_evaluation(
    *,
    cache_path: Path,
    index_path: Path,
    snapshot_dir: Path,
    lock_path: Path,
    output_dir: Path,
) -> tuple[dict[str, object], tuple[Path, Path, Path, Path]]:
    """Run preflight first, then evaluate development only and write results."""
    lock = validate_source_lock(lock_path, snapshot_dir, cache_path, index_path)
    manifest_path = snapshot_dir / "evaluation_group_manifest.csv"
    manifest_rows, preflight = validate_frozen_manifest(manifest_path, lock)
    gallery, development_manifest, development_matrix = prepare_frozen_data(
        cache_path, index_path, manifest_rows
    )
    record_rows, rankings = evaluate_development_queries(
        gallery, development_manifest, development_matrix
    )
    sample_rows = aggregate_by_sample_id(record_rows)
    label_error_rows = build_label_and_error_rows(
        record_rows, sample_rows, rankings, gallery.reference_records
    )
    metrics = build_metrics_summary(record_rows, sample_rows, label_error_rows)
    summary: dict[str, object] = {
        "experiment": "11.2",
        "evaluation_scope": "frozen_development_query_only",
        "similarity": "experiment_10_raw_cosine_similarity_with_stable_sorting",
        "result_levels": list(LEVELS),
        "top_k_values": list(TOP_K_VALUES),
        "hit_definition": (
            "file/sample levels: at least one candidate evidence label-group key "
            "matches query within K; label level: query label-group key appears within K"
        ),
        "aggregation": (
            "average record hit indicators within query Sample_ID, then equal-weight "
            "Sample_IDs; macro-average labels"
        ),
        "denominators": {
            "query_spectra": len(record_rows),
            "query_sample_ids": len(sample_rows),
            "query_reference_labels": len(
                {row["query_reference_label_group_key"] for row in sample_rows}
            ),
            "gallery_spectra": len(gallery.reference_records),
            "gallery_sample_ids": len(
                {row["sample_id"].casefold() for row in gallery.reference_records}
            ),
        },
        "metrics": metrics,
        "safety_checks": {
            **preflight,
            "source_lock_validated": True,
            "manifest_reconciled_to_npz_and_index": True,
            "final_test_queries_executed": 0,
            "final_test_rows_selected_for_similarity": 0,
            "gallery_only_similarity_candidate_count": len(gallery.reference_records),
        },
        "locked_inputs": {
            "lock_path": str(lock_path),
            "lock_sha256": sha256_file(lock_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "cache_path": str(cache_path),
            "cache_sha256": sha256_file(cache_path),
            "index_path": str(index_path),
            "index_sha256": sha256_file(index_path),
        },
        "code_version": _git_version(),
        "scientific_interpretation": (
            "Rates are retrieval hits relative to frozen database candidate reference "
            "labels, not mineral-identification probability, confidence, or independently "
            "validated classification accuracy."
        ),
    }
    paths = write_outputs(output_dir, record_rows, sample_rows, label_error_rows, summary)
    return summary, paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen experiment 11 development Top-K baseline."
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()
    try:
        summary, paths = run_evaluation(
            cache_path=arguments.cache,
            index_path=arguments.index,
            snapshot_dir=arguments.snapshot_dir,
            lock_path=arguments.lock,
            output_dir=arguments.output_dir,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print("Development query spectra: " + str(summary["denominators"]["query_spectra"]))
    print("Development query Sample_IDs: " + str(summary["denominators"]["query_sample_ids"]))
    for level in LEVELS:
        metrics = summary["metrics"]["sample_id_equal_weight_60_sample_ids"]
        print(
            f"{level}: "
            + ", ".join(
                f"Top-{k}={metrics[f'{level}_top_{k}']:.6f}" for k in TOP_K_VALUES
            )
        )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
