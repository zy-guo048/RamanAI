"""Reproducible error profiling for the frozen experiment 11 development split."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP11_EVALUATOR = PROJECT_ROOT / "scripts" / "11_evaluate_development_topk.py"
DEFAULT_BASELINE_DIR = PROJECT_ROOT / "results" / "exp11_development_baseline"
DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "docs" / "evaluation" / "exp11_v1"
DEFAULT_LOCK = DEFAULT_SNAPSHOT_DIR / "DATA_SOURCE_LOCK.json"
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "exp11_development_error_analysis"
TOP_K_VALUES = (1, 3, 5)

BASELINE_FILES = {
    "records": "development_query_record_results.csv",
    "samples": "development_sample_id_results.csv",
    "labels": "development_label_stats_and_errors.csv",
    "summary": "development_baseline_summary.json",
}

ERROR_CASE_FIELDS = [
    "query_relative_path",
    "query_cache_row_index",
    "query_sample_id",
    "query_measurement_stem",
    "query_reference_label",
    "query_reference_label_group_key",
    "query_excitation_wavelength_nm",
    "query_excitation_wavelength_source",
    "query_spectrum_count_for_sample_id",
    "metadata_status",
    "metadata_requires_review",
    "metadata_review_reasons",
    "metadata_flags",
    "correct_label_rank",
    "correct_label_score",
    "reference_label_hit_at_1",
    "reference_label_hit_at_3",
    "reference_label_hit_at_5",
    "is_top1_error",
    "top1_prediction_label",
    "top1_prediction_label_group_key",
    "top1_prediction_relative_path",
    "top1_prediction_sample_id",
    "top1_score",
    "highest_wrong_label",
    "highest_wrong_label_group_key",
    "highest_wrong_relative_path",
    "highest_wrong_sample_id",
    "highest_wrong_score",
    "correct_minus_highest_wrong_score",
    "score_margin_relation",
    "top_score_tie",
    "full_label_candidate_count",
]

SAMPLE_DIAGNOSTIC_FIELDS = [
    "query_sample_id",
    "query_reference_label",
    "query_reference_label_group_key",
    "query_spectrum_count",
    "excitation_wavelengths_nm",
    "top1_prediction_labels",
    "top1_prediction_consistent",
    "top1_hit_fraction",
    "top3_hit_fraction",
    "top5_hit_fraction",
    "correct_label_rank_min",
    "correct_label_rank_median",
    "correct_label_rank_max",
    "score_margin_min",
    "score_margin_median",
    "score_margin_max",
    "partial_top1_hit_across_measurements",
    "query_relative_paths",
]

LABEL_SUMMARY_FIELDS = [
    "reference_label",
    "reference_label_group_key",
    "development_query_spectrum_count",
    "development_query_sample_id_count",
    "gallery_file_count",
    "gallery_sample_id_count",
    "gallery_sample_id_size_group",
    "record_top1_hit_rate",
    "record_top3_hit_rate",
    "record_top5_hit_rate",
    "sample_equal_top1_hit_rate",
    "sample_equal_top3_hit_rate",
    "sample_equal_top5_hit_rate",
    "top1_error_record_count",
    "top1_error_sample_id_count",
    "common_wrong_labels",
    "most_common_wrong_label_record_count",
    "repeated_same_error_observed",
    "correct_label_score_min",
    "correct_label_score_median",
    "correct_label_score_max",
    "score_margin_min",
    "score_margin_median",
    "score_margin_max",
]

CONFUSION_FIELDS = [
    "true_reference_label",
    "true_reference_label_group_key",
    "top1_wrong_reference_label",
    "top1_wrong_reference_label_group_key",
    "error_spectrum_count",
    "distinct_query_sample_id_count",
    "query_sample_ids",
    "excitation_wavelengths_nm",
    "mean_top1_wrong_score",
    "mean_correct_label_score",
    "mean_correct_minus_wrong_score",
]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


baseline = _load_module("raman_exp11_development_for_diagnostics", EXP11_EVALUATOR)
exp10 = baseline.exp10
exp11 = baseline.exp11


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Required input file does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"Required input file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_columns(
    rows: list[dict[str, str]], required: set[str], name: str
) -> None:
    if not rows:
        raise ValueError(f"{name} is empty")
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"{name} is missing columns: " + ", ".join(missing))


def _unique_by_path(rows: list[dict[str, str]], name: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        path = row.get("query_relative_path", "").strip()
        if not path:
            raise ValueError(f"{name} contains an empty query_relative_path")
        if path in result:
            raise ValueError(f"{name} contains duplicate query_relative_path: {path}")
        result[path] = row
    return result


def normalize_excitation_wavelength(value: str | None) -> str:
    """Preserve supplied metadata and map only explicit missing values to Unknown."""
    cleaned = (value or "").strip()
    if cleaned.casefold() in {"", "unknown", "none", "n/a"}:
        return "Unknown"
    return cleaned


def load_and_validate_baseline_results(
    baseline_dir: Path,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, object],
    dict[str, str],
]:
    """Load actual 11.2 files and validate identities and published denominators."""
    paths = {key: baseline_dir / filename for key, filename in BASELINE_FILES.items()}
    record_rows = _read_csv(paths["records"])
    sample_rows = _read_csv(paths["samples"])
    label_rows = _read_csv(paths["labels"])
    summary = _read_json(paths["summary"])
    _require_columns(
        record_rows,
        {
            "query_relative_path",
            "query_cache_row_index",
            "query_sample_id",
            "query_measurement_stem",
            "query_reference_label",
            "query_reference_label_group_key",
            "query_excitation_wavelength_nm",
            "reference_label_hit_at_1",
            "reference_label_hit_at_3",
            "reference_label_hit_at_5",
            "reference_label_matching_rank",
            "reference_label_top1_reference_label",
            "reference_label_top1_relative_path",
            "reference_label_top1_cosine_similarity",
        },
        "Development record results",
    )
    _require_columns(
        sample_rows,
        {
            "query_sample_id",
            "query_reference_label_group_key",
            "query_spectrum_count",
            "reference_label_mean_hit_at_1",
            "reference_label_mean_hit_at_3",
            "reference_label_mean_hit_at_5",
        },
        "Development Sample_ID results",
    )
    _require_columns(
        label_rows,
        {"row_type", "reference_label_group_key", "gallery_file_count", "gallery_sample_id_count"},
        "Development label results",
    )
    _unique_by_path(record_rows, "Development record results")
    denominators = summary.get("denominators")
    safety = summary.get("safety_checks")
    if not isinstance(denominators, dict) or not isinstance(safety, dict):
        raise ValueError("Baseline summary is missing denominators or safety checks")
    expected = {
        "query_spectra": 118,
        "query_sample_ids": 60,
        "query_reference_labels": 60,
        "gallery_spectra": 724,
        "gallery_sample_ids": 342,
    }
    for key, value in expected.items():
        if int(denominators.get(key, -1)) != value:
            raise ValueError(f"Baseline denominator mismatch for {key}")
    if len(record_rows) != 118 or len(sample_rows) != 60:
        raise ValueError("Baseline result row counts do not match the frozen protocol")
    if int(safety.get("final_test_queries_executed", -1)) != 0:
        raise ValueError("Baseline summary indicates final-test queries were executed")
    if int(safety.get("final_test_rows_selected_for_similarity", -1)) != 0:
        raise ValueError("Baseline summary indicates final-test rows entered similarity")
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    return record_rows, sample_rows, label_rows, summary, hashes


def _hit_label(hit: dict[str, object]) -> tuple[str, str]:
    return baseline._hit_label(hit)


def extract_full_label_diagnostic(
    hits: list[dict[str, object]], target_key: str
) -> dict[str, object]:
    """Extract full-rank truth, wrong-label evidence, and signed score margin."""
    if not hits:
        raise ValueError("Full reference-label ranking is empty")
    target_matches = [hit for hit in hits if _hit_label(hit)[0] == target_key]
    if len(target_matches) != 1:
        raise ValueError(
            "Full reference-label ranking must contain the frozen query label exactly once"
        )
    correct = target_matches[0]
    wrong = next((hit for hit in hits if _hit_label(hit)[0] != target_key), None)
    if wrong is None:
        raise ValueError("At least one wrong reference label is required for diagnosis")
    correct_record = correct["record"]
    wrong_record = wrong["record"]
    top_record = hits[0]["record"]
    assert isinstance(correct_record, dict)
    assert isinstance(wrong_record, dict)
    assert isinstance(top_record, dict)
    correct_score = float(correct["cosine_similarity"])
    wrong_score = float(wrong["cosine_similarity"])
    margin = correct_score - wrong_score
    tolerance = 1e-15
    relation = "zero_tie" if abs(margin) <= tolerance else "positive" if margin > 0 else "negative"
    top_key, top_display = _hit_label(hits[0])
    wrong_key, wrong_display = _hit_label(wrong)
    return {
        "correct_label_rank": int(correct["rank"]),
        "correct_label_score": correct_score,
        "reference_label_hit_at_1": int(int(correct["rank"]) <= 1),
        "reference_label_hit_at_3": int(int(correct["rank"]) <= 3),
        "reference_label_hit_at_5": int(int(correct["rank"]) <= 5),
        "is_top1_error": int(int(correct["rank"]) != 1),
        "top1_prediction_label": top_display,
        "top1_prediction_label_group_key": top_key,
        "top1_prediction_relative_path": top_record["relative_path"],
        "top1_prediction_sample_id": top_record.get("sample_id", "Unknown"),
        "top1_score": float(hits[0]["cosine_similarity"]),
        "highest_wrong_label": wrong_display,
        "highest_wrong_label_group_key": wrong_key,
        "highest_wrong_relative_path": wrong_record["relative_path"],
        "highest_wrong_sample_id": wrong_record.get("sample_id", "Unknown"),
        "highest_wrong_score": wrong_score,
        "correct_minus_highest_wrong_score": margin,
        "score_margin_relation": relation,
        "top_score_tie": abs(float(hits[0]["cosine_similarity"]) - float(hits[1]["cosine_similarity"])) <= tolerance,
        "full_label_candidate_count": len(hits),
    }


def _compare_baseline_record(
    existing: dict[str, str], diagnostic: dict[str, object]
) -> None:
    for k in TOP_K_VALUES:
        expected = int(existing[f"reference_label_hit_at_{k}"])
        actual = int(diagnostic[f"reference_label_hit_at_{k}"])
        if expected != actual:
            raise ValueError(
                f"Recomputed Top-{k} differs for {existing['query_relative_path']}"
            )
    if int(existing["reference_label_matching_rank"]) != int(
        diagnostic["correct_label_rank"]
    ):
        raise ValueError(
            "Recomputed correct-label rank differs for " + existing["query_relative_path"]
        )
    if existing["reference_label_top1_reference_label"] != str(
        diagnostic["top1_prediction_label"]
    ):
        raise ValueError(
            "Recomputed Top-1 label differs for " + existing["query_relative_path"]
        )
    if not math.isclose(
        float(existing["reference_label_top1_cosine_similarity"]),
        float(diagnostic["top1_score"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Recomputed Top-1 score differs for " + existing["query_relative_path"]
        )


def recompute_development_diagnostics(
    gallery_library,
    development_manifest: list[dict[str, str]],
    development_matrix: np.ndarray,
    baseline_record_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Compute complete 60-label rankings for Development only."""
    existing_by_path = _unique_by_path(
        baseline_record_rows, "Development record results"
    )
    development_paths = {row["relative_path"] for row in development_manifest}
    if set(existing_by_path) != development_paths:
        raise ValueError("Baseline and frozen Development relative_path sets differ")
    sample_counts = Counter(row["sample_id"] for row in development_manifest)
    label_count = len(
        {
            exp10._reference_label_identity(record)[2]
            for record in gallery_library.reference_records
        }
    )
    results: list[dict[str, object]] = []
    for manifest, vector in zip(development_manifest, development_matrix, strict=True):
        existing = existing_by_path[manifest["relative_path"]]
        for field in (
            "query_cache_row_index",
            "query_sample_id",
            "query_measurement_stem",
            "query_reference_label_group_key",
        ):
            manifest_field = field.removeprefix("query_")
            if existing[field] != manifest[manifest_field]:
                raise ValueError(
                    f"Baseline/frozen identity mismatch for {manifest['relative_path']}: {field}"
                )
        query = exp10.QuerySpectrum(
            vector=np.asarray(vector, dtype=np.float64),
            mode="frozen_development_error_diagnostic",
            relative_path=manifest["relative_path"],
            sample_id=manifest["sample_id"],
            sample_id_source=manifest["sample_id_source"],
            measurement_stem=manifest["measurement_stem"],
            reference_label=manifest["reference_label"],
            excitation_wavelength_nm=manifest["excitation_wavelength_nm"],
        )
        hits = exp10.search_topk(
            gallery_library,
            query,
            label_count,
            result_level="reference_label",
            exclude_self=False,
            exclude_same_sample=False,
        )
        diagnostic = extract_full_label_diagnostic(
            hits, manifest["reference_label_group_key"]
        )
        _compare_baseline_record(existing, diagnostic)
        result: dict[str, object] = {
            "query_relative_path": manifest["relative_path"],
            "query_cache_row_index": manifest["cache_row_index"],
            "query_sample_id": manifest["sample_id"],
            "query_measurement_stem": manifest["measurement_stem"],
            "query_reference_label": manifest["reference_label_group_display"],
            "query_reference_label_group_key": manifest[
                "reference_label_group_key"
            ],
            "query_excitation_wavelength_nm": normalize_excitation_wavelength(
                manifest.get("excitation_wavelength_nm")
            ),
            "query_excitation_wavelength_source": manifest.get(
                "excitation_wavelength_source", ""
            ),
            "query_spectrum_count_for_sample_id": sample_counts[
                manifest["sample_id"]
            ],
            "metadata_status": manifest.get("metadata_status", ""),
            "metadata_requires_review": manifest.get(
                "metadata_requires_review", ""
            ),
            "metadata_review_reasons": manifest.get("review_reasons", ""),
            "metadata_flags": manifest.get("metadata_flags", ""),
            **diagnostic,
        }
        results.append(result)
    return results


def aggregate_sample_diagnostics(
    error_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in error_rows:
        groups[str(row["query_sample_id"])].append(row)
    output: list[dict[str, object]] = []
    for sample_id in sorted(groups, key=lambda value: (value.casefold(), value)):
        rows = groups[sample_id]
        keys = {str(row["query_reference_label_group_key"]) for row in rows}
        if len(keys) != 1:
            raise ValueError(f"Development Sample_ID spans reference labels: {sample_id}")
        predictions = sorted(
            {str(row["top1_prediction_label"]) for row in rows}, key=str.casefold
        )
        ranks = [int(row["correct_label_rank"]) for row in rows]
        margins = [float(row["correct_minus_highest_wrong_score"]) for row in rows]
        top1_fraction = float(
            np.mean([int(row["reference_label_hit_at_1"]) for row in rows])
        )
        output.append(
            {
                "query_sample_id": sample_id,
                "query_reference_label": rows[0]["query_reference_label"],
                "query_reference_label_group_key": rows[0][
                    "query_reference_label_group_key"
                ],
                "query_spectrum_count": len(rows),
                "excitation_wavelengths_nm": " | ".join(
                    sorted(
                        {str(row["query_excitation_wavelength_nm"]) for row in rows},
                        key=str.casefold,
                    )
                ),
                "top1_prediction_labels": " | ".join(predictions),
                "top1_prediction_consistent": len(predictions) == 1,
                "top1_hit_fraction": top1_fraction,
                "top3_hit_fraction": float(
                    np.mean([int(row["reference_label_hit_at_3"]) for row in rows])
                ),
                "top5_hit_fraction": float(
                    np.mean([int(row["reference_label_hit_at_5"]) for row in rows])
                ),
                "correct_label_rank_min": min(ranks),
                "correct_label_rank_median": statistics.median(ranks),
                "correct_label_rank_max": max(ranks),
                "score_margin_min": min(margins),
                "score_margin_median": statistics.median(margins),
                "score_margin_max": max(margins),
                "partial_top1_hit_across_measurements": 0.0 < top1_fraction < 1.0,
                "query_relative_paths": " | ".join(
                    str(row["query_relative_path"]) for row in rows
                ),
            }
        )
    return output


def gallery_label_sizes(
    gallery_records: Iterable[dict[str, str]],
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in gallery_records:
        _display, _basis, key = exp10._reference_label_identity(record)
        grouped[key].append(record)
    result: dict[str, dict[str, object]] = {}
    for key, rows in grouped.items():
        display, _basis, _key = exp10._reference_label_identity(rows[0])
        result[key] = {
            "display": display,
            "file_count": len(rows),
            "sample_id_count": len({row["sample_id"].casefold() for row in rows}),
        }
    return result


def _size_group(sample_count: int) -> str:
    if sample_count <= 4:
        return "3-4"
    if sample_count <= 7:
        return "5-7"
    return ">=8"


def aggregate_label_diagnostics(
    error_rows: list[dict[str, object]],
    sample_rows: list[dict[str, object]],
    gallery_sizes: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    records_by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    samples_by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in error_rows:
        records_by_label[str(row["query_reference_label_group_key"])].append(row)
    for row in sample_rows:
        samples_by_label[str(row["query_reference_label_group_key"])].append(row)
    output: list[dict[str, object]] = []
    for key in sorted(records_by_label):
        records = records_by_label[key]
        samples = samples_by_label[key]
        size = gallery_sizes.get(key)
        if size is None:
            raise ValueError(f"Development label is absent from Gallery: {key}")
        wrong_counts = Counter(
            str(row["top1_prediction_label"])
            for row in records
            if int(row["is_top1_error"]) == 1
        )
        common_wrong = " | ".join(
            f"{label}:{count}"
            for label, count in sorted(
                wrong_counts.items(), key=lambda item: (-item[1], item[0].casefold(), item[0])
            )
        )
        correct_scores = [float(row["correct_label_score"]) for row in records]
        margins = [float(row["correct_minus_highest_wrong_score"]) for row in records]
        output.append(
            {
                "reference_label": records[0]["query_reference_label"],
                "reference_label_group_key": key,
                "development_query_spectrum_count": len(records),
                "development_query_sample_id_count": len(samples),
                "gallery_file_count": size["file_count"],
                "gallery_sample_id_count": size["sample_id_count"],
                "gallery_sample_id_size_group": _size_group(int(size["sample_id_count"])),
                "record_top1_hit_rate": float(np.mean([int(row["reference_label_hit_at_1"]) for row in records])),
                "record_top3_hit_rate": float(np.mean([int(row["reference_label_hit_at_3"]) for row in records])),
                "record_top5_hit_rate": float(np.mean([int(row["reference_label_hit_at_5"]) for row in records])),
                "sample_equal_top1_hit_rate": float(np.mean([float(row["top1_hit_fraction"]) for row in samples])),
                "sample_equal_top3_hit_rate": float(np.mean([float(row["top3_hit_fraction"]) for row in samples])),
                "sample_equal_top5_hit_rate": float(np.mean([float(row["top5_hit_fraction"]) for row in samples])),
                "top1_error_record_count": sum(int(row["is_top1_error"]) for row in records),
                "top1_error_sample_id_count": len({str(row["query_sample_id"]) for row in records if int(row["is_top1_error"]) == 1}),
                "common_wrong_labels": common_wrong,
                "most_common_wrong_label_record_count": max(wrong_counts.values(), default=0),
                "repeated_same_error_observed": max(wrong_counts.values(), default=0) >= 2,
                "correct_label_score_min": min(correct_scores),
                "correct_label_score_median": statistics.median(correct_scores),
                "correct_label_score_max": max(correct_scores),
                "score_margin_min": min(margins),
                "score_margin_median": statistics.median(margins),
                "score_margin_max": max(margins),
            }
        )
    return output


def build_confusion_pairs(
    error_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in error_rows:
        if int(row["is_top1_error"]) == 1:
            grouped[
                (
                    str(row["query_reference_label_group_key"]),
                    str(row["top1_prediction_label_group_key"]),
                )
            ].append(row)
    output: list[dict[str, object]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        output.append(
            {
                "true_reference_label": rows[0]["query_reference_label"],
                "true_reference_label_group_key": key[0],
                "top1_wrong_reference_label": rows[0]["top1_prediction_label"],
                "top1_wrong_reference_label_group_key": key[1],
                "error_spectrum_count": len(rows),
                "distinct_query_sample_id_count": len({str(row["query_sample_id"]) for row in rows}),
                "query_sample_ids": " | ".join(sorted({str(row["query_sample_id"]) for row in rows}, key=str.casefold)),
                "excitation_wavelengths_nm": " | ".join(sorted({str(row["query_excitation_wavelength_nm"]) for row in rows}, key=str.casefold)),
                "mean_top1_wrong_score": float(np.mean([float(row["top1_score"]) for row in rows])),
                "mean_correct_label_score": float(np.mean([float(row["correct_label_score"]) for row in rows])),
                "mean_correct_minus_wrong_score": float(np.mean([float(row["correct_minus_highest_wrong_score"]) for row in rows])),
            }
        )
    output.sort(
        key=lambda row: (
            -int(row["error_spectrum_count"]),
            str(row["true_reference_label"]).casefold(),
            str(row["top1_wrong_reference_label"]).casefold(),
        )
    )
    return output


def _metric_summary(
    error_rows: list[dict[str, object]], sample_rows: list[dict[str, object]]
) -> dict[str, object]:
    return {
        "record_level": {
            f"reference_label_top_{k}": float(np.mean([int(row[f"reference_label_hit_at_{k}"]) for row in error_rows]))
            for k in TOP_K_VALUES
        },
        "sample_id_equal_weight": {
            f"reference_label_top_{k}": float(np.mean([float(row[f"top{k}_hit_fraction"]) for row in sample_rows]))
            for k in TOP_K_VALUES
        },
    }


def compare_metrics_to_baseline(
    metrics: dict[str, object], baseline_summary: dict[str, object]
) -> dict[str, object]:
    published_metrics = baseline_summary.get("metrics")
    if not isinstance(published_metrics, dict):
        raise ValueError("Baseline summary is missing metrics")
    published_record = published_metrics.get("record_level_118_spectra")
    published_sample = published_metrics.get("sample_id_equal_weight_60_sample_ids")
    if not isinstance(published_record, dict) or not isinstance(published_sample, dict):
        raise ValueError("Baseline summary metric sections are missing")
    comparisons: dict[str, object] = {}
    for k in TOP_K_VALUES:
        key = f"reference_label_top_{k}"
        recalculated_record = float(metrics["record_level"][key])
        recalculated_sample = float(metrics["sample_id_equal_weight"][key])
        published_record_value = float(published_record[key])
        published_sample_value = float(published_sample[key])
        if not math.isclose(recalculated_record, published_record_value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Recalculated record-level {key} differs from baseline")
        if not math.isclose(recalculated_sample, published_sample_value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Recalculated Sample_ID-level {key} differs from baseline")
        comparisons[key] = {
            "record_level_match": True,
            "sample_id_equal_weight_match": True,
            "record_level_value": recalculated_record,
            "sample_id_equal_weight_value": recalculated_sample,
        }
    return comparisons


def build_wavelength_summary(
    error_rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in error_rows:
        groups[str(row["query_excitation_wavelength_nm"])].append(row)
    output: dict[str, dict[str, object]] = {}
    for wavelength in sorted(groups, key=str.casefold):
        rows = groups[wavelength]
        output[wavelength] = {
            "spectrum_count": len(rows),
            "distinct_sample_id_count": len({str(row["query_sample_id"]) for row in rows}),
            **{
                f"record_top_{k}_hit_rate": float(np.mean([int(row[f"reference_label_hit_at_{k}"]) for row in rows]))
                for k in TOP_K_VALUES
            },
        }
    return output


def build_size_group_summary(
    label_rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in label_rows:
        groups[str(row["gallery_sample_id_size_group"])].append(row)
    output: dict[str, dict[str, object]] = {}
    for group_name in ("3-4", "5-7", ">=8"):
        rows = groups.get(group_name, [])
        output[group_name] = {
            "label_count": len(rows),
            "development_spectrum_count": sum(int(row["development_query_spectrum_count"]) for row in rows),
            "development_sample_id_count": sum(int(row["development_query_sample_id_count"]) for row in rows),
            **{
                f"sample_equal_top_{k}_hit_rate": (
                    float(np.mean([float(row[f"sample_equal_top{k}_hit_rate"]) for row in rows]))
                    if rows
                    else None
                )
                for k in TOP_K_VALUES
            },
            "stability_note": (
                "small_group_descriptive_only" if len(rows) < 10 else "descriptive_only_no_causal_interpretation"
            ),
        }
    return output


def build_report(summary: dict[str, object], error_rows: list[dict[str, object]], confusion_rows: list[dict[str, object]]) -> str:
    metrics = summary["metrics"]
    record = metrics["record_level"]
    sample = metrics["sample_id_equal_weight"]
    top_failures = sorted(
        [row for row in error_rows if int(row["is_top1_error"]) == 1],
        key=lambda row: (float(row["correct_minus_highest_wrong_score"]), str(row["query_relative_path"])),
    )[:10]
    failure_lines = "\n".join(
        f"- {row['query_reference_label']} / {row['query_sample_id']}: "
        f"Top-1={row['top1_prediction_label']}，正确标签排名={row['correct_label_rank']}，"
        f"分数差={float(row['correct_minus_highest_wrong_score']):.6f}。"
        for row in top_failures
    ) or "- 无 Top-1 错误。"
    confusion_lines = "\n".join(
        f"- {row['true_reference_label']} → {row['top1_wrong_reference_label']}: "
        f"{row['error_spectrum_count']} 条光谱，{row['distinct_query_sample_id_count']} 个 Sample_ID。"
        for row in confusion_rows[:10]
    ) or "- 无 Top-1 混淆。"
    return f"""# RamanAI 实验 11.3-A：Development 错误画像

## 范围与隔离

本报告只分析冻结的 118 条 Development 查询和 724 条 Gallery 光谱，共 60 个查询
Sample_ID、342 个 Gallery Sample_ID、60 个严格闭集参考标签。未运行 final-test 检索，
final-test 进入相似度计算的记录数为 0。

实验 9.3 NPZ 是单体压缩缓存。现有验证加载器为了核对全缓存的行身份，会将整个数值矩阵
解压到内存，因此这里是逻辑分组隔离，而不是物理文件级隔离。真正传入余弦相似度计算的
候选仅有 724 条 Gallery，查询仅有 118 条 Development；final-test 行未被选入。

## 基线复现

逐光谱参考标签级命中率：Top-1 {record['reference_label_top_1']:.4%}、
Top-3 {record['reference_label_top_3']:.4%}、Top-5 {record['reference_label_top_5']:.4%}。

按 Sample_ID 内先平均、再对 60 个 Sample_ID 等权：Top-1
{sample['reference_label_top_1']:.4%}、Top-3 {sample['reference_label_top_3']:.4%}、
Top-5 {sample['reference_label_top_5']:.4%}。这些数值与实验 11.2 完全一致。

## 典型 Top-1 错误

{failure_lines}

## 高频混淆方向

{confusion_lines}

## 科学解释限制

`reference_label` 是数据库候选参考标签，不是独立验证的矿物真值。余弦相似度和分数差都
不是概率或置信度。Gallery 规模、激发波长与命中结果的分组统计仅用于描述，不证明因果。
本轮没有修改算法、阈值、冻结分组或检索排序规则。
"""


def _git_version() -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    return {"head": run("rev-parse", "HEAD"), "describe": run("describe", "--always", "--dirty")}


def output_paths(output_dir: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    return (
        output_dir / "development_error_cases.csv",
        output_dir / "development_sample_id_diagnostics.csv",
        output_dir / "development_label_error_summary.csv",
        output_dir / "development_confusion_pairs.csv",
        output_dir / "development_error_analysis_summary.json",
        output_dir / "development_error_analysis_report.md",
    )


def write_outputs(
    output_dir: Path,
    error_rows: list[dict[str, object]],
    sample_rows: list[dict[str, object]],
    label_rows: list[dict[str, object]],
    confusion_rows: list[dict[str, object]],
    summary: dict[str, object],
    report: str,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    paths = output_paths(output_dir)
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Development error-analysis output already exists: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, fields, rows in (
        (paths[0], ERROR_CASE_FIELDS, error_rows),
        (paths[1], SAMPLE_DIAGNOSTIC_FIELDS, sample_rows),
        (paths[2], LABEL_SUMMARY_FIELDS, label_rows),
        (paths[3], CONFUSION_FIELDS, confusion_rows),
    ):
        with path.open("x", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with paths[4].open("x", encoding="utf-8-sig", newline="") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    paths[5].write_text(report, encoding="utf-8-sig", newline="\n")
    return paths


def run_analysis(
    *,
    baseline_dir: Path,
    cache_path: Path,
    index_path: Path,
    snapshot_dir: Path,
    lock_path: Path,
    output_dir: Path,
) -> tuple[dict[str, object], tuple[Path, Path, Path, Path, Path, Path]]:
    baseline_records, _baseline_samples, _baseline_labels, baseline_summary, baseline_hashes = load_and_validate_baseline_results(baseline_dir)
    lock = baseline.validate_source_lock(lock_path, snapshot_dir, cache_path, index_path)
    manifest_rows, preflight = baseline.validate_frozen_manifest(
        snapshot_dir / "evaluation_group_manifest.csv", lock
    )
    gallery, development_manifest, development_matrix = baseline.prepare_frozen_data(
        cache_path, index_path, manifest_rows
    )
    error_rows = recompute_development_diagnostics(
        gallery, development_manifest, development_matrix, baseline_records
    )
    sample_rows = aggregate_sample_diagnostics(error_rows)
    sizes = gallery_label_sizes(gallery.reference_records)
    label_rows = aggregate_label_diagnostics(error_rows, sample_rows, sizes)
    confusion_rows = build_confusion_pairs(error_rows)
    metrics = _metric_summary(error_rows, sample_rows)
    metric_comparison = compare_metrics_to_baseline(metrics, baseline_summary)
    summary: dict[str, object] = {
        "experiment": "11.3-A",
        "scope": "frozen_development_error_diagnostics_only",
        "denominators": {
            "development_query_spectra": len(error_rows),
            "development_query_sample_ids": len(sample_rows),
            "gallery_spectra": len(gallery.reference_records),
            "gallery_sample_ids": len({row["sample_id"].casefold() for row in gallery.reference_records}),
            "strict_closed_set_reference_labels": len(label_rows),
        },
        "metrics": metrics,
        "baseline_reproduction": metric_comparison,
        "error_counts": {
            "top1_error_spectra": sum(int(row["is_top1_error"]) for row in error_rows),
            "top3_error_spectra": sum(1 - int(row["reference_label_hit_at_3"]) for row in error_rows),
            "top5_error_spectra": sum(1 - int(row["reference_label_hit_at_5"]) for row in error_rows),
            "sample_ids_with_any_top1_error": len({str(row["query_sample_id"]) for row in error_rows if int(row["is_top1_error"]) == 1}),
            "sample_ids_with_partial_top1_results": sum(bool(row["partial_top1_hit_across_measurements"]) for row in sample_rows),
            "sample_ids_with_inconsistent_top1_predictions": sum(not bool(row["top1_prediction_consistent"]) for row in sample_rows),
            "confusion_pair_count": len(confusion_rows),
        },
        "gallery_sample_id_size_groups": build_size_group_summary(label_rows),
        "excitation_wavelength_groups": build_wavelength_summary(error_rows),
        "safety_checks": {
            **preflight,
            "source_lock_validated": True,
            "baseline_files_validated": True,
            "baseline_identity_reconciled": True,
            "baseline_metrics_reproduced": True,
            "final_test_queries_executed": 0,
            "final_test_rows_selected_for_similarity": 0,
            "similarity_gallery_rows": len(gallery.reference_records),
            "similarity_development_query_rows": len(error_rows),
        },
        "cache_loading_limit": (
            "The experiment 9.3 NPZ is a monolithic compressed cache. The existing "
            "validated loader decompresses the full matrix into memory for identity "
            "checks; isolation is logical, not physical-file-level. Only frozen Gallery "
            "and Development slices are passed to cosine similarity, and no final-test "
            "row is selected for similarity."
        ),
        "input_hashes": {
            "baseline_files": baseline_hashes,
            "data_source_lock": sha256_file(lock_path),
            "frozen_manifest": sha256_file(snapshot_dir / "evaluation_group_manifest.csv"),
            "cache": sha256_file(cache_path),
            "index": sha256_file(index_path),
        },
        "code_version": {
            **_git_version(),
            "analysis_script_sha256": sha256_file(Path(__file__)),
            "experiment_11_2_script_sha256": sha256_file(EXP11_EVALUATOR),
        },
        "scientific_interpretation": (
            "reference_label is a candidate database label, not independently validated "
            "ground truth. Cosine similarity and score margins are not probabilities or confidence."
        ),
    }
    report = build_report(summary, error_rows, confusion_rows)
    paths = write_outputs(
        output_dir, error_rows, sample_rows, label_rows, confusion_rows, summary, report
    )
    return summary, paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze errors in the frozen experiment 11 Development baseline."
    )
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
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
        summary, paths = run_analysis(
            baseline_dir=arguments.baseline_dir,
            cache_path=arguments.cache,
            index_path=arguments.index,
            snapshot_dir=arguments.snapshot_dir,
            lock_path=arguments.lock,
            output_dir=arguments.output_dir,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        "Development spectra/Sample_IDs: "
        f"{summary['denominators']['development_query_spectra']}/"
        f"{summary['denominators']['development_query_sample_ids']}"
    )
    for k in TOP_K_VALUES:
        record = summary["metrics"]["record_level"][f"reference_label_top_{k}"]
        sample = summary["metrics"]["sample_id_equal_weight"][f"reference_label_top_{k}"]
        print(f"Reference-label Top-{k}: record={record:.6f}, Sample_ID={sample:.6f}")
    print("Final-test similarity rows: 0")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
