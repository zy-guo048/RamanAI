"""Frozen RamanAI reference-label evaluation core for experiments 12 and 13.

Experiment 12 may evaluate synthetic data and the frozen Development split only.
The real final-test branch is guarded and must not be activated until a separately
approved executable lock and one-time approval record exist.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import uuid
from typing import Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP11_EVALUATOR_PATH = PROJECT_ROOT / "scripts" / "11_evaluate_development_topk.py"
DEFAULT_METHOD_LOCK = (
    PROJECT_ROOT / "docs" / "evaluation" / "exp12_v1" / "RETRIEVAL_METHOD_LOCK.json"
)
DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "docs" / "evaluation" / "exp11_v1"
DEFAULT_DATA_LOCK = DEFAULT_SNAPSHOT_DIR / "DATA_SOURCE_LOCK.json"
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
DEFAULT_BASELINE_DIR = PROJECT_ROOT / "results" / "exp11_development_baseline"

METHOD_ID = "A_max_spectrum_cosine"
DEVELOPMENT_GROUP = "development_query"
FINAL_TEST_GROUP = "final_test_query"
GALLERY_GROUP = "reference_gallery"
TOP_K_VALUES = (1, 3, 5)
EXPECTED_DEVELOPMENT_METRICS = {
    1: 0.8805555555555556,
    3: 0.95,
    5: 0.9666666666666667,
}
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42
BOOTSTRAP_UNIT_COUNT = 60
BOOTSTRAP_QUANTILES = (0.025, 0.975)
BOOTSTRAP_QUANTILE_METHOD = "linear"
FORMAL_ATTEMPT_AUDIT_DIRNAME = "final_test_attempt_audits"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load required module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp11_evaluator = _load_module(
    "raman_exp11_evaluator_for_final_core", EXP11_EVALUATOR_PATH
)
exp10 = exp11_evaluator.exp10
exp11_protocol = exp11_evaluator.exp11


RECORD_FIELDS = (
    "query_relative_path",
    "query_cache_row_index",
    "query_sample_id",
    "query_measurement_stem",
    "query_reference_label",
    "query_reference_label_group_key",
    "query_excitation_wavelength_nm",
    "query_measurement_count",
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "correct_label_rank",
    "correct_label_score",
    "top1_reference_label",
    "top1_reference_label_group_key",
    "top1_relative_path",
    "top1_cosine_similarity",
    "top3_reference_labels",
    "top5_reference_labels",
    "highest_wrong_reference_label",
    "highest_wrong_score",
    "correct_minus_highest_wrong_score",
)
RANKING_FIELDS = (
    "query_relative_path",
    "query_sample_id",
    "query_reference_label_group_key",
    "rank",
    "candidate_reference_label",
    "candidate_reference_label_group_key",
    "label_score",
    "best_support_relative_path",
    "is_correct_label",
)
SAMPLE_FIELDS = (
    "query_sample_id",
    "query_reference_label",
    "query_reference_label_group_key",
    "query_spectrum_count",
    "mean_hit_at_1",
    "mean_hit_at_3",
    "mean_hit_at_5",
)
LABEL_FIELDS = (
    "query_reference_label",
    "query_reference_label_group_key",
    "query_sample_id_count",
    "query_spectrum_count",
    "macro_component_hit_at_1",
    "macro_component_hit_at_3",
    "macro_component_hit_at_5",
)
CONFUSION_FIELDS = (
    "true_reference_label",
    "true_reference_label_group_key",
    "predicted_reference_label",
    "predicted_reference_label_group_key",
    "query_record_count",
    "distinct_query_sample_id_count",
)


@dataclass(frozen=True)
class BootstrapResult:
    """Deterministic Sample_ID-level percentile bootstrap result."""

    sample_id_order: tuple[str, ...]
    sample_id_order_sha256: str
    draw_indices_sha256: str
    point_estimates: dict[int, float]
    intervals: dict[int, tuple[float, float]]
    replicate_count: int
    random_seed: int
    quantile_method: str


@dataclass(frozen=True)
class EvaluationResult:
    """Reference-label evaluation results for one explicitly selected query group."""

    query_group: str
    record_rows: tuple[dict[str, object], ...]
    ranking_rows: tuple[dict[str, object], ...]
    sample_rows: tuple[dict[str, object], ...]
    label_rows: tuple[dict[str, object], ...]
    confusion_rows: tuple[dict[str, object], ...]
    sample_equal_metrics: dict[int, float]
    label_macro_metrics: dict[int, float]
    bootstrap: BootstrapResult


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one file."""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"Unable to hash required file {Path(path).name}: {exc}") from exc
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return value


def _require_equal(actual: object, expected: object, name: str) -> None:
    if actual != expected:
        raise ValueError(f"Method lock mismatch for {name}: {actual!r}")


def load_and_validate_method_lock(path: Path = DEFAULT_METHOD_LOCK) -> dict[str, object]:
    """Validate the experiment 12 method-lock draft without approving it."""
    lock = _read_json(path, "method lock")
    _require_equal(
        lock.get("schema"), "ramanai_final_evaluation_method_lock_draft_v1", "schema"
    )
    method = lock.get("method")
    frozen = lock.get("frozen_evaluation_data")
    metrics = lock.get("metrics")
    bootstrap = lock.get("bootstrap")
    confirmed = lock.get("confirmed_inputs")
    approval = lock.get("approval")
    for name, value in (
        ("method", method),
        ("frozen_evaluation_data", frozen),
        ("metrics", metrics),
        ("bootstrap", bootstrap),
        ("confirmed_inputs", confirmed),
        ("approval", approval),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"Method lock is missing object: {name}")

    assert isinstance(method, dict)
    _require_equal(method.get("method_id"), METHOD_ID, "method_id")
    _require_equal(method.get("method_selection_closed"), True, "method_selection_closed")
    _require_equal(method.get("method_b_allowed"), False, "method_b_allowed")
    label_scoring = method.get("label_scoring")
    if not isinstance(label_scoring, dict):
        raise ValueError("Method lock is missing label_scoring")
    _require_equal(label_scoring.get("score"), "maximum_cosine_similarity", "label score")
    _require_equal(
        label_scoring.get("scope"),
        "all_eligible_frozen_gallery_spectra_before_any_file_top_k_truncation",
        "label scoring scope",
    )

    assert isinstance(frozen, dict)
    expected_counts = {
        "gallery": (724, 342, 60),
        "development": (118, 60, 60),
        "final_test": (129, 60, 60),
    }
    for group, expected in expected_counts.items():
        specification = frozen.get(group)
        if not isinstance(specification, dict):
            raise ValueError(f"Method lock is missing frozen group: {group}")
        actual = (
            specification.get("record_count"),
            specification.get("known_sample_id_count"),
            specification.get("reference_label_count"),
        )
        _require_equal(actual, expected, f"{group} counts")
    final_spec = frozen["final_test"]
    assert isinstance(final_spec, dict)
    _require_equal(
        final_spec.get("experiment_12_similarity_calculation_allowed"),
        False,
        "experiment 12 final-test similarity policy",
    )
    _require_equal(
        final_spec.get("experiment_12_prediction_allowed"),
        False,
        "experiment 12 final-test prediction policy",
    )

    assert isinstance(metrics, dict)
    _require_equal(metrics.get("top_k_values"), [1, 3, 5], "top_k_values")
    primary = metrics.get("primary")
    if not isinstance(primary, dict):
        raise ValueError("Method lock is missing primary metric")
    _require_equal(
        (primary.get("result_level"), primary.get("k"), primary.get("aggregation")),
        ("reference_label", 1, "query_sample_id_equal_weight"),
        "primary metric",
    )

    assert isinstance(bootstrap, dict)
    expected_bootstrap = {
        "resampling_unit": "query_sample_id",
        "unit_count": BOOTSTRAP_UNIT_COUNT,
        "draws_per_replicate": BOOTSTRAP_UNIT_COUNT,
        "with_replacement": True,
        "replicate_count": BOOTSTRAP_REPLICATES,
        "random_seed": BOOTSTRAP_SEED,
        "lower_percentile": 2.5,
        "upper_percentile": 97.5,
        "quantile_function": "numpy.quantile",
        "quantile_method": BOOTSTRAP_QUANTILE_METHOD,
        "shared_draw_indices_across_top_k": True,
        "split_repeated_measurements": False,
    }
    for key, expected in expected_bootstrap.items():
        _require_equal(bootstrap.get(key), expected, f"bootstrap.{key}")
    return lock


def _resolve_locked_path(relative_path: str, project_root: Path) -> Path:
    candidate = (Path(project_root) / relative_path).resolve()
    root = Path(project_root).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Locked path escapes project root: {relative_path}") from exc
    return candidate


def _verify_file_specification(
    specification: dict[str, object], project_root: Path, name: str
) -> dict[str, object]:
    relative_path = str(specification.get("path", ""))
    if not relative_path:
        raise ValueError(f"Locked {name} path is missing")
    path = _resolve_locked_path(relative_path, project_root)
    if not path.is_file():
        raise ValueError(f"Locked {name} does not exist: {path.name}")
    actual_size = path.stat().st_size
    actual_hash = sha256_file(path)
    if actual_size != int(specification.get("size_bytes", -1)):
        raise ValueError(f"Locked {name} size mismatch")
    if actual_hash != str(specification.get("sha256", "")):
        raise ValueError(f"Locked {name} SHA-256 mismatch")
    return {
        "path": relative_path,
        "size_bytes": actual_size,
        "sha256": actual_hash,
        "verified": True,
    }


def verify_locked_inputs(
    method_lock: dict[str, object], project_root: Path = PROJECT_ROOT
) -> dict[str, dict[str, object]]:
    """Verify every file pinned by the method lock, including the exp11 lock."""
    confirmed = method_lock.get("confirmed_inputs")
    if not isinstance(confirmed, dict):
        raise ValueError("Method lock has no confirmed_inputs section")
    required = (
        "evaluation_manifest",
        "evaluation_protocol",
        "evaluation_group_audit",
        "reference_label_sample_distribution",
        "experiment_11_data_source_lock",
        "experiment_9_3_npz",
        "experiment_9_3_index",
    )
    verified: dict[str, dict[str, object]] = {}
    for key in required:
        specification = confirmed.get(key)
        if not isinstance(specification, dict):
            raise ValueError(f"Method lock is missing confirmed input: {key}")
        verified[key] = _verify_file_specification(specification, project_root, key)
    return verified


def validate_frozen_inputs(
    *,
    method_lock_path: Path = DEFAULT_METHOD_LOCK,
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
    data_lock_path: Path = DEFAULT_DATA_LOCK,
    cache_path: Path = DEFAULT_CACHE,
    index_path: Path = DEFAULT_INDEX,
    project_root: Path = PROJECT_ROOT,
) -> tuple[dict[str, object], list[dict[str, str]], dict[str, object]]:
    """Validate locks, hashes, frozen identities, counts, and cross-group isolation."""
    method_lock = load_and_validate_method_lock(method_lock_path)
    verified_files = verify_locked_inputs(method_lock, project_root)
    data_lock = exp11_evaluator.validate_source_lock(
        data_lock_path, snapshot_dir, cache_path, index_path
    )
    manifest_path = snapshot_dir / "evaluation_group_manifest.csv"
    manifest_rows, manifest_audit = exp11_evaluator.validate_frozen_manifest(
        manifest_path, data_lock
    )
    confirmed = method_lock["confirmed_inputs"]
    assert isinstance(confirmed, dict)
    if manifest_audit["final_test_identity_sha256"] != method_lock[
        "frozen_evaluation_data"
    ]["final_test"]["identity_sha256"]:
        raise ValueError("Method-lock final-test identity SHA-256 mismatch")
    if exp11_protocol._identity_hash(manifest_rows) != str(
        confirmed["all_group_identity_sha256"]
    ):
        raise ValueError("Method-lock all-group identity SHA-256 mismatch")
    audit = {
        "method_lock_schema_valid": True,
        "verified_files": verified_files,
        "manifest": manifest_audit,
        "logical_isolation_only": True,
        "physical_npz_isolation": False,
        "final_test_metadata_read": True,
        "final_test_vector_selection_count": 0,
        "final_test_similarity_row_count": 0,
        "final_test_prediction_count": 0,
        "final_test_score_count": 0,
    }
    return method_lock, manifest_rows, audit


def prepare_development_data(
    cache_path: Path,
    index_path: Path,
    manifest_rows: list[dict[str, str]],
):
    """Load the monolithic cache but select only Gallery and Development rows."""
    gallery, development_manifest, development_matrix = (
        exp11_evaluator.prepare_frozen_data(cache_path, index_path, manifest_rows)
    )
    if len(gallery.reference_records) != 724:
        raise ValueError("Frozen Gallery must contain exactly 724 spectra")
    if len(development_manifest) != 118 or development_matrix.shape != (118, 1101):
        raise ValueError("Frozen Development selection must have shape (118, 1101)")
    if any(row["evaluation_group"] != DEVELOPMENT_GROUP for row in development_manifest):
        raise ValueError("Non-Development row entered the Development query selection")
    final_paths = {
        row["relative_path"]
        for row in manifest_rows
        if row["evaluation_group"] == FINAL_TEST_GROUP
    }
    if final_paths & {row["relative_path"] for row in development_manifest}:
        raise ValueError("A final-test row entered the Development query selection")
    return gallery, development_manifest, development_matrix


def _label_hit_identity(hit: dict[str, object]) -> tuple[str, str]:
    return (
        str(hit["reference_label_group_key"]),
        str(hit["reference_label_group_display"]),
    )


def evaluate_reference_label_queries(
    gallery_library,
    query_manifest: Sequence[dict[str, str]],
    query_matrix: np.ndarray,
    *,
    query_group: str,
    authorization: dict[str, object] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate one explicitly selected query group with experiment 10 method A."""
    if query_group == FINAL_TEST_GROUP:
        if not authorization or authorization.get("authorized") is not True:
            raise PermissionError(
                "Final-test evaluation is prohibited before approval"
            )
    elif query_group != DEVELOPMENT_GROUP:
        raise ValueError(f"Unsupported experiment 12 query group: {query_group}")
    matrix = np.asarray(query_matrix, dtype=np.float64)
    if matrix.ndim != 2 or len(query_manifest) != len(matrix):
        raise ValueError("Query metadata and matrix dimensions do not match")
    if matrix.shape[1] != len(gallery_library.grid):
        raise ValueError("Query vectors do not match the frozen grid")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Query matrix contains non-finite values")
    measurement_counts = Counter(row["sample_id"] for row in query_manifest)
    label_count = len(
        {
            exp10._reference_label_identity(record)[2]
            for record in gallery_library.reference_records
        }
    )
    if label_count != 60:
        raise ValueError(f"Frozen Gallery must contain 60 labels, got {label_count}")

    record_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    for manifest, vector in zip(query_manifest, matrix, strict=True):
        if manifest.get("evaluation_group") != query_group:
            raise ValueError("Query Manifest row belongs to the wrong evaluation group")
        query = exp10.QuerySpectrum(
            vector=np.asarray(vector, dtype=np.float64),
            mode="frozen_development_query_exp12_preflight",
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
        if len(hits) != label_count:
            raise ValueError("Method A did not return the complete 60-label ranking")
        target_key = manifest["reference_label_group_key"]
        correct_hit = next(
            (hit for hit in hits if str(hit["reference_label_group_key"]) == target_key),
            None,
        )
        if correct_hit is None:
            raise ValueError("Correct frozen reference label is absent from Gallery ranking")
        correct_rank = int(correct_hit["rank"])
        correct_score = float(correct_hit["cosine_similarity"])
        wrong_hits = [
            hit for hit in hits if str(hit["reference_label_group_key"]) != target_key
        ]
        highest_wrong = wrong_hits[0]
        top = hits[0]
        top_record = top["record"]
        assert isinstance(top_record, dict)
        record_rows.append(
            {
                "query_relative_path": manifest["relative_path"],
                "query_cache_row_index": int(manifest["cache_row_index"]),
                "query_sample_id": manifest["sample_id"],
                "query_measurement_stem": manifest["measurement_stem"],
                "query_reference_label": manifest["reference_label_group_display"],
                "query_reference_label_group_key": target_key,
                "query_excitation_wavelength_nm": manifest["excitation_wavelength_nm"],
                "query_measurement_count": measurement_counts[manifest["sample_id"]],
                "hit_at_1": int(correct_rank <= 1),
                "hit_at_3": int(correct_rank <= 3),
                "hit_at_5": int(correct_rank <= 5),
                "correct_label_rank": correct_rank,
                "correct_label_score": correct_score,
                "top1_reference_label": str(top["reference_label_group_display"]),
                "top1_reference_label_group_key": str(top["reference_label_group_key"]),
                "top1_relative_path": top_record["relative_path"],
                "top1_cosine_similarity": float(top["cosine_similarity"]),
                "top3_reference_labels": " | ".join(
                    str(hit["reference_label_group_display"]) for hit in hits[:3]
                ),
                "top5_reference_labels": " | ".join(
                    str(hit["reference_label_group_display"]) for hit in hits[:5]
                ),
                "highest_wrong_reference_label": str(
                    highest_wrong["reference_label_group_display"]
                ),
                "highest_wrong_score": float(highest_wrong["cosine_similarity"]),
                "correct_minus_highest_wrong_score": correct_score
                - float(highest_wrong["cosine_similarity"]),
            }
        )
        for hit in hits:
            support = hit["record"]
            assert isinstance(support, dict)
            hit_key, hit_display = _label_hit_identity(hit)
            ranking_rows.append(
                {
                    "query_relative_path": manifest["relative_path"],
                    "query_sample_id": manifest["sample_id"],
                    "query_reference_label_group_key": target_key,
                    "rank": int(hit["rank"]),
                    "candidate_reference_label": hit_display,
                    "candidate_reference_label_group_key": hit_key,
                    "label_score": float(hit["cosine_similarity"]),
                    "best_support_relative_path": support["relative_path"],
                    "is_correct_label": int(hit_key == target_key),
                }
            )
    return record_rows, ranking_rows


def aggregate_query_sample_ids(
    record_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Average repeated query measurements, then expose one row per Sample_ID."""
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in record_rows:
        sample_id = str(row["query_sample_id"])
        if not sample_id or sample_id.casefold() in {"unknown", "missing", "none", "nan"}:
            raise ValueError("Formal query has no usable Sample_ID")
        grouped[sample_id].append(row)
    output: list[dict[str, object]] = []
    for sample_id in sorted(grouped, key=lambda value: (value.casefold(), value)):
        rows = grouped[sample_id]
        label_keys = {str(row["query_reference_label_group_key"]) for row in rows}
        if len(label_keys) != 1:
            raise ValueError(f"Query Sample_ID spans multiple labels: {sample_id}")
        output.append(
            {
                "query_sample_id": sample_id,
                "query_reference_label": str(rows[0]["query_reference_label"]),
                "query_reference_label_group_key": str(
                    rows[0]["query_reference_label_group_key"]
                ),
                "query_spectrum_count": len(rows),
                **{
                    f"mean_hit_at_{k}": float(
                        np.mean([float(row[f"hit_at_{k}"]) for row in rows])
                    )
                    for k in TOP_K_VALUES
                },
            }
        )
    return output


def build_label_macro_rows(
    sample_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in sample_rows:
        grouped[str(row["query_reference_label_group_key"])].append(row)
    output: list[dict[str, object]] = []
    for key in sorted(grouped, key=lambda value: (value.casefold(), value)):
        rows = grouped[key]
        output.append(
            {
                "query_reference_label": str(rows[0]["query_reference_label"]),
                "query_reference_label_group_key": key,
                "query_sample_id_count": len(rows),
                "query_spectrum_count": sum(int(row["query_spectrum_count"]) for row in rows),
                **{
                    f"macro_component_hit_at_{k}": float(
                        np.mean([float(row[f"mean_hit_at_{k}"]) for row in rows])
                    )
                    for k in TOP_K_VALUES
                },
            }
        )
    return output


def compute_sample_equal_metrics(
    sample_rows: Sequence[dict[str, object]],
) -> dict[int, float]:
    if not sample_rows:
        raise ValueError("No query Sample_ID rows are available")
    return {
        k: float(np.mean([float(row[f"mean_hit_at_{k}"]) for row in sample_rows]))
        for k in TOP_K_VALUES
    }


def compute_label_macro_metrics(
    label_rows: Sequence[dict[str, object]],
) -> dict[int, float]:
    if not label_rows:
        raise ValueError("No query label rows are available")
    return {
        k: float(
            np.mean([float(row[f"macro_component_hit_at_{k}"]) for row in label_rows])
        )
        for k in TOP_K_VALUES
    }


def build_confusion_pairs(
    record_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in record_rows:
        if int(row["hit_at_1"]) == 1:
            continue
        key = (
            str(row["query_reference_label"]),
            str(row["query_reference_label_group_key"]),
            str(row["top1_reference_label"]),
            str(row["top1_reference_label_group_key"]),
        )
        grouped[key].append(row)
    output: list[dict[str, object]] = []
    for key in sorted(
        grouped,
        key=lambda value: tuple(part.casefold() for part in value) + value,
    ):
        rows = grouped[key]
        output.append(
            {
                "true_reference_label": key[0],
                "true_reference_label_group_key": key[1],
                "predicted_reference_label": key[2],
                "predicted_reference_label_group_key": key[3],
                "query_record_count": len(rows),
                "distinct_query_sample_id_count": len(
                    {str(row["query_sample_id"]).casefold() for row in rows}
                ),
            }
        )
    return output


def deterministic_sample_id_bootstrap(
    sample_rows: Sequence[dict[str, object]],
    *,
    expected_unit_count: int = BOOTSTRAP_UNIT_COUNT,
    replicate_count: int = BOOTSTRAP_REPLICATES,
    random_seed: int = BOOTSTRAP_SEED,
) -> BootstrapResult:
    """Bootstrap pre-aggregated Sample_ID hit fractions with shared draws for all K."""
    if replicate_count < 1:
        raise ValueError("Bootstrap replicate_count must be positive")
    ordered = sorted(
        sample_rows,
        key=lambda row: (
            str(row["query_sample_id"]).casefold(),
            str(row["query_sample_id"]),
        ),
    )
    if len(ordered) != expected_unit_count:
        raise ValueError(
            f"Bootstrap requires {expected_unit_count} Sample_ID units, got {len(ordered)}"
        )
    sample_ids = tuple(str(row["query_sample_id"]) for row in ordered)
    if len({value.casefold() for value in sample_ids}) != len(sample_ids):
        raise ValueError("Bootstrap Sample_ID order contains duplicates")
    values = np.asarray(
        [
            [float(row[f"mean_hit_at_{k}"]) for k in TOP_K_VALUES]
            for row in ordered
        ],
        dtype=np.float64,
    )
    if values.shape != (expected_unit_count, len(TOP_K_VALUES)):
        raise ValueError("Bootstrap hit matrix has an unexpected shape")
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Bootstrap hit fractions must be finite values in [0, 1]")
    rng = np.random.default_rng(random_seed)
    draw_indices = rng.integers(
        0,
        expected_unit_count,
        size=(replicate_count, expected_unit_count),
        endpoint=False,
    )
    replicate_metrics = values[draw_indices].mean(axis=1)
    quantiles = np.quantile(
        replicate_metrics,
        BOOTSTRAP_QUANTILES,
        axis=0,
        method=BOOTSTRAP_QUANTILE_METHOD,
    )
    order_hash = hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()
    draw_hash = hashlib.sha256(draw_indices.tobytes(order="C")).hexdigest()
    return BootstrapResult(
        sample_id_order=sample_ids,
        sample_id_order_sha256=order_hash,
        draw_indices_sha256=draw_hash,
        point_estimates={
            k: float(values[:, index].mean()) for index, k in enumerate(TOP_K_VALUES)
        },
        intervals={
            k: (float(quantiles[0, index]), float(quantiles[1, index]))
            for index, k in enumerate(TOP_K_VALUES)
        },
        replicate_count=replicate_count,
        random_seed=random_seed,
        quantile_method=BOOTSTRAP_QUANTILE_METHOD,
    )


def run_development_evaluation(
    gallery_library,
    development_manifest: Sequence[dict[str, str]],
    development_matrix: np.ndarray,
) -> EvaluationResult:
    record_rows, ranking_rows = evaluate_reference_label_queries(
        gallery_library,
        development_manifest,
        development_matrix,
        query_group=DEVELOPMENT_GROUP,
    )
    sample_rows = aggregate_query_sample_ids(record_rows)
    label_rows = build_label_macro_rows(sample_rows)
    sample_metrics = compute_sample_equal_metrics(sample_rows)
    label_metrics = compute_label_macro_metrics(label_rows)
    for k in TOP_K_VALUES:
        if not math.isclose(sample_metrics[k], label_metrics[k], rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"Sample_ID-equal and label-macro Top-{k} differ")
    bootstrap = deterministic_sample_id_bootstrap(sample_rows)
    return EvaluationResult(
        query_group=DEVELOPMENT_GROUP,
        record_rows=tuple(record_rows),
        ranking_rows=tuple(ranking_rows),
        sample_rows=tuple(sample_rows),
        label_rows=tuple(label_rows),
        confusion_rows=tuple(build_confusion_pairs(record_rows)),
        sample_equal_metrics=sample_metrics,
        label_macro_metrics=label_metrics,
        bootstrap=bootstrap,
    )


def prepare_final_test_data(
    cache_path: Path,
    index_path: Path,
    manifest_rows: list[dict[str, str]],
    *,
    authorization: dict[str, object] | None,
):
    """Select Gallery and final-test vectors only after the approval gate passes."""
    if not authorization or authorization.get("authorized") is not True:
        raise PermissionError(
            "Final-test authorization must pass before loading and selecting vectors"
        )
    full_library = exp10.load_reference_library(cache_path, index_path)
    success_by_path = {
        row["relative_path"]: row for row in full_library.all_success_records
    }
    formal_rows = [
        row
        for row in manifest_rows
        if row["evaluation_group"]
        in {GALLERY_GROUP, DEVELOPMENT_GROUP, FINAL_TEST_GROUP}
    ]
    for manifest in formal_rows:
        index = success_by_path.get(manifest["relative_path"])
        if index is None:
            raise ValueError(
                "Frozen formal record is absent from successful cache index: "
                + manifest["relative_path"]
            )
        exp11_evaluator._reconcile_manifest_record(manifest, index)
        if int(manifest["cache_row_index"]) != full_library.path_to_success_row[
            manifest["relative_path"]
        ]:
            raise ValueError("Frozen cache_row_index does not match the NPZ row identity")

    gallery_manifest = [
        row for row in manifest_rows if row["evaluation_group"] == GALLERY_GROUP
    ]
    gallery_indices = np.asarray(
        [full_library.path_to_success_row[row["relative_path"]] for row in gallery_manifest],
        dtype=int,
    )
    gallery_records = tuple(
        success_by_path[row["relative_path"]] for row in gallery_manifest
    )
    gallery_matrix = full_library.all_success_matrix[gallery_indices].copy()
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
    final_manifest, final_matrix = select_query_rows_for_formal_evaluation(
        manifest_rows,
        full_library.all_success_matrix,
        query_group=FINAL_TEST_GROUP,
        authorization=authorization,
    )
    if len(gallery_records) != 724 or gallery_matrix.shape != (724, 1101):
        raise ValueError("Frozen Gallery selection must have shape (724, 1101)")
    if len(final_manifest) != 129 or final_matrix.shape != (129, 1101):
        raise ValueError("Frozen final-test selection must have shape (129, 1101)")
    return gallery_library, final_manifest, final_matrix


def run_final_test_evaluation(
    gallery_library,
    final_manifest: Sequence[dict[str, str]],
    final_matrix: np.ndarray,
    *,
    authorization: dict[str, object] | None,
) -> EvaluationResult:
    """Run the frozen evaluator after a separately validated authorization."""
    if not authorization or authorization.get("authorized") is not True:
        raise PermissionError("Final-test evaluation authorization is missing")
    record_rows, ranking_rows = evaluate_reference_label_queries(
        gallery_library,
        final_manifest,
        final_matrix,
        query_group=FINAL_TEST_GROUP,
        authorization=authorization,
    )
    sample_rows = aggregate_query_sample_ids(record_rows)
    label_rows = build_label_macro_rows(sample_rows)
    sample_metrics = compute_sample_equal_metrics(sample_rows)
    label_metrics = compute_label_macro_metrics(label_rows)
    for k in TOP_K_VALUES:
        if not math.isclose(sample_metrics[k], label_metrics[k], rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"Sample_ID-equal and label-macro Top-{k} differ")
    bootstrap = deterministic_sample_id_bootstrap(sample_rows)
    return EvaluationResult(
        query_group=FINAL_TEST_GROUP,
        record_rows=tuple(record_rows),
        ranking_rows=tuple(ranking_rows),
        sample_rows=tuple(sample_rows),
        label_rows=tuple(label_rows),
        confusion_rows=tuple(build_confusion_pairs(record_rows)),
        sample_equal_metrics=sample_metrics,
        label_macro_metrics=label_metrics,
        bootstrap=bootstrap,
    )


def verify_development_baseline(
    result: EvaluationResult,
    baseline_dir: Path = DEFAULT_BASELINE_DIR,
    *,
    tolerance: float = 1e-12,
) -> dict[str, object]:
    """Compare Development metrics and query rankings with experiment 11.2 outputs."""
    if result.query_group != DEVELOPMENT_GROUP:
        raise ValueError("Only Development results may be compared with the baseline")
    for k, expected in EXPECTED_DEVELOPMENT_METRICS.items():
        actual = result.sample_equal_metrics[k]
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(
                f"Development Top-{k} baseline mismatch: expected {expected}, got {actual}"
            )
    baseline_path = Path(baseline_dir) / "development_query_record_results.csv"
    baseline_rows = exp11_evaluator._read_csv(baseline_path)
    baseline_by_path = {row["query_relative_path"]: row for row in baseline_rows}
    if len(baseline_by_path) != 118 or len(result.record_rows) != 118:
        raise ValueError("Development record denominator differs from experiment 11.2")
    comparison_fields = {
        "hit_at_1": "reference_label_hit_at_1",
        "hit_at_3": "reference_label_hit_at_3",
        "hit_at_5": "reference_label_hit_at_5",
        "correct_label_rank": "reference_label_matching_rank",
        "top1_reference_label": "reference_label_top1_reference_label",
        "top1_relative_path": "reference_label_top1_relative_path",
        "top5_reference_labels": "reference_label_top5_reference_labels",
    }
    for row in result.record_rows:
        path = str(row["query_relative_path"])
        baseline = baseline_by_path.get(path)
        if baseline is None:
            raise ValueError(f"Development query is absent from baseline: {path}")
        for actual_field, baseline_field in comparison_fields.items():
            if str(row[actual_field]) != str(baseline[baseline_field]):
                raise ValueError(
                    f"Development ranking mismatch for {path}: {actual_field}"
                )
        baseline_score = float(baseline["reference_label_top1_cosine_similarity"])
        if not math.isclose(
            float(row["top1_cosine_similarity"]),
            baseline_score,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError(f"Development Top-1 score mismatch for {path}")
    return {
        "baseline_record_count": 118,
        "baseline_sample_id_count": 60,
        "query_identity_match": True,
        "complete_correct_label_rank_match": True,
        "top1_evidence_match": True,
        "top5_label_order_match": True,
        "metric_absolute_tolerance": tolerance,
        "sample_equal_metrics": {str(k): result.sample_equal_metrics[k] for k in TOP_K_VALUES},
    }


def _approval_hash_payload(path: Path) -> str:
    return sha256_file(path)


def validate_executable_source_lock(
    executable_lock_path: Path,
    method_lock_path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    require_approved: bool,
) -> dict[str, object]:
    """Validate the executable lock and every source hash recorded in it."""
    executable_lock = _read_json(executable_lock_path, "executable source lock")
    if executable_lock.get("schema") != "ramanai_final_evaluation_executable_lock_v1":
        raise ValueError("Executable source lock schema is invalid")
    expected_status = "frozen_candidate_pending_human_approval"
    if executable_lock.get("lock_status") != expected_status:
        error = PermissionError if require_approved else ValueError
        raise error(f"Executable source lock status must be {expected_status}")
    method_spec = executable_lock.get("method_lock")
    if not isinstance(method_spec, dict):
        raise ValueError("Executable source lock has no method_lock specification")
    if method_spec.get("sha256") != sha256_file(method_lock_path):
        raise ValueError("Executable source lock method-lock SHA-256 mismatch")
    sources = executable_lock.get("executable_sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Executable source lock has no executable_sources")
    verified: list[dict[str, object]] = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Executable source entry must be an object")
        verified.append(
            _verify_file_specification(source, project_root, "executable source")
        )
    return {"lock": executable_lock, "verified_sources": verified}


def require_final_test_approval(
    *,
    method_lock_path: Path,
    executable_lock_path: Path,
    approval_path: Path,
    output_dir: Path,
    run_id: str,
    explicit_final_mode: bool,
) -> dict[str, object]:
    """Validate the future one-time gate before any final-test row may be selected."""
    if not explicit_final_mode:
        raise PermissionError("Explicit formal final-test mode was not requested")
    if Path(output_dir).exists():
        raise FileExistsError("Formal final-test output directory already exists")
    method_lock = load_and_validate_method_lock(method_lock_path)
    approval_state = method_lock.get("approval")
    assert isinstance(approval_state, dict)
    candidate_status = "frozen_candidate_pending_external_human_approval"
    if approval_state.get("method_configuration") != candidate_status:
        raise PermissionError("Method configuration is not a frozen approval candidate")
    if approval_state.get("bootstrap_configuration") != candidate_status:
        raise PermissionError("Bootstrap configuration is not a frozen approval candidate")
    if approval_state.get("executable_source_lock") != candidate_status:
        raise PermissionError("Executable source lock is not a frozen approval candidate")
    if (
        approval_state.get("final_test_execution")
        != "requires_valid_experiment_13_single_run_approval_record"
    ):
        raise PermissionError("Formal final-test execution gate is not sealed")
    executable_validation = validate_executable_source_lock(
        executable_lock_path,
        method_lock_path,
        project_root=PROJECT_ROOT,
        require_approved=True,
    )
    runtime = validate_runtime_environment(executable_validation["lock"])
    approval = _read_json(approval_path, "one-time final-test approval")
    if approval.get("approval_status") != "approved_for_single_final_test_attempt":
        raise PermissionError("One-time final-test approval is absent or invalid")
    required_matches = {
        "method_lock_sha256": _approval_hash_payload(method_lock_path),
        "executable_lock_sha256": _approval_hash_payload(executable_lock_path),
        "final_test_identity_sha256": method_lock["frozen_evaluation_data"][
            "final_test"
        ]["identity_sha256"],
        "npz_sha256": method_lock["confirmed_inputs"]["experiment_9_3_npz"][
            "sha256"
        ],
        "index_sha256": method_lock["confirmed_inputs"]["experiment_9_3_index"][
            "sha256"
        ],
        "evaluation_manifest_sha256": method_lock["confirmed_inputs"][
            "evaluation_manifest"
        ]["sha256"],
        "data_source_lock_sha256": method_lock["confirmed_inputs"][
            "experiment_11_data_source_lock"
        ]["sha256"],
        "run_id": run_id,
    }
    for key, expected in required_matches.items():
        if approval.get(key) != expected:
            raise PermissionError(f"One-time approval does not match {key}")
    canonical_output = str(Path(output_dir).resolve())
    if str(approval.get("output_dir", "")) != canonical_output:
        raise PermissionError("One-time approval is bound to a different output directory")
    if not approval.get("approved_by") or not approval.get("approved_at"):
        raise PermissionError("One-time approval lacks approver or approval time")
    return {
        "authorized": True,
        "approval_sha256": sha256_file(approval_path),
        "output_dir": canonical_output,
        "run_id": run_id,
        "runtime_environment": runtime,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_run_id(run_id: str) -> str:
    """Require a canonical UUID so an approval cannot ambiguously name an attempt."""
    try:
        parsed = uuid.UUID(str(run_id))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("Formal run_id must be a canonical UUID") from exc
    canonical = str(parsed)
    if str(run_id) != canonical:
        raise ValueError("Formal run_id must be a canonical lowercase UUID")
    return canonical


def begin_formal_run_attempt(output_dir: Path, run_id: str) -> Path:
    """Create an immutable-identity audit record before formal authorization."""
    canonical_run_id = _validate_run_id(run_id)
    output_dir = Path(output_dir).resolve()
    audit_dir = output_dir.parent / FORMAL_ATTEMPT_AUDIT_DIRNAME
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"{canonical_run_id}.json"
    now = _utc_now()
    record = {
        "schema": "ramanai_formal_final_test_attempt_v1",
        "run_id": canonical_run_id,
        "status": "started_not_authorized",
        "stage": "authorization",
        "started_at": now,
        "updated_at": now,
        "output_dir": str(output_dir),
        "authorization_verified": False,
        "final_test_query_row_count": 0,
        "gallery_reference_row_count": 0,
        "similarity_pair_count": 0,
        "prediction_count": 0,
        "score_computation_started": False,
        "scores_computed": False,
        "events": [{"stage": "authorization", "status": "started", "at": now}],
    }
    try:
        _write_json(audit_path, record)
    except FileExistsError as exc:
        raise PermissionError(
            "This run_id has already started and its approval cannot be reused"
        ) from exc
    return audit_path


def update_formal_run_attempt(
    audit_path: Path,
    *,
    stage: str,
    status: str,
    updates: dict[str, object] | None = None,
    exception: BaseException | None = None,
) -> dict[str, object]:
    """Atomically append one lifecycle event to an existing attempt audit."""
    audit_path = Path(audit_path)
    record = _read_json(audit_path, "formal final-test attempt audit")
    event: dict[str, object] = {"stage": stage, "status": status, "at": _utc_now()}
    if exception is not None:
        event["exception_type"] = type(exception).__name__
        event["exception_message"] = str(exception)
        record["exception_type"] = type(exception).__name__
        record["exception_message"] = str(exception)
    events = record.get("events")
    if not isinstance(events, list):
        raise ValueError("Formal attempt audit events are invalid")
    events.append(event)
    record["stage"] = stage
    record["status"] = status
    record["updated_at"] = _utc_now()
    if updates:
        record.update(updates)
    temporary = audit_path.with_name(f".{audit_path.name}.tmp-{uuid.uuid4().hex}")
    try:
        _write_json(temporary, record)
        os.replace(temporary, audit_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return record


def select_query_rows_for_formal_evaluation(
    manifest_rows: Sequence[dict[str, str]],
    full_matrix: np.ndarray,
    *,
    query_group: str,
    authorization: dict[str, object] | None = None,
) -> tuple[list[dict[str, str]], np.ndarray]:
    """Select query vectors only after the appropriate group gate has passed."""
    if query_group == FINAL_TEST_GROUP:
        if not authorization or authorization.get("authorized") is not True:
            raise PermissionError(
                "Final-test authorization must be validated before vector selection"
            )
    elif query_group != DEVELOPMENT_GROUP:
        raise ValueError(f"Unsupported formal query group: {query_group}")
    selected = [row for row in manifest_rows if row["evaluation_group"] == query_group]
    indices = np.asarray([int(row["cache_row_index"]) for row in selected], dtype=int)
    matrix = np.asarray(full_matrix, dtype=np.float64)
    if np.any(indices < 0) or np.any(indices >= len(matrix)):
        raise ValueError("Manifest cache_row_index is outside the cached matrix")
    return selected, matrix[indices].copy()


def environment_snapshot() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": importlib.metadata.version("scipy"),
    }


def validate_runtime_environment(
    executable_lock: dict[str, object],
) -> dict[str, str]:
    """Reject a formal run when critical runtime versions differ from the lock."""
    expected = executable_lock.get("runtime_environment")
    if not isinstance(expected, dict):
        raise ValueError("Executable source lock has no runtime_environment")
    actual = environment_snapshot()
    for key in ("python_version", "numpy_version", "scipy_version"):
        if str(expected.get(key, "")) != actual[key]:
            raise RuntimeError(
                f"Formal runtime environment mismatch for {key}: "
                f"expected {expected.get(key)!r}, got {actual[key]!r}"
            )
    return actual


def _bootstrap_to_dict(value: BootstrapResult) -> dict[str, object]:
    return {
        "resampling_unit": "query_sample_id",
        "unit_count": len(value.sample_id_order),
        "sample_id_order": list(value.sample_id_order),
        "sample_id_order_sha256": value.sample_id_order_sha256,
        "draw_indices_sha256": value.draw_indices_sha256,
        "replicate_count": value.replicate_count,
        "random_seed": value.random_seed,
        "shared_draw_indices_across_top_k": True,
        "quantile_function": "numpy.quantile",
        "quantiles": list(BOOTSTRAP_QUANTILES),
        "quantile_method": value.quantile_method,
        "point_estimates": {
            f"top_{k}": value.point_estimates[k] for k in TOP_K_VALUES
        },
        "percentile_intervals": {
            f"top_{k}": {
                "lower": value.intervals[k][0],
                "upper": value.intervals[k][1],
            }
            for k in TOP_K_VALUES
        },
        "interpretation": (
            "Exploratory conditional Sample_ID resampling description; not an "
            "external generalization guarantee."
        ),
    }


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")


def write_preflight_outputs_atomic(
    output_dir: Path,
    result: EvaluationResult,
    frozen_input_audit: dict[str, object],
    baseline_comparison: dict[str, object],
    method_lock_path: Path,
) -> tuple[Path, ...]:
    """Write a complete Development preflight directory without overwriting history."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Preflight output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=False, exist_ok=False)
    paths = {
        "records": temporary / "development_query_record_results.csv",
        "rankings": temporary / "development_full_label_rankings.csv",
        "samples": temporary / "development_sample_id_results.csv",
        "labels": temporary / "development_label_metrics.csv",
        "confusions": temporary / "development_confusion_pairs.csv",
        "bootstrap": temporary / "bootstrap_validation.json",
        "inputs": temporary / "frozen_input_audit.json",
        "summary": temporary / "preflight_summary.json",
        "audit": temporary / "RUN_AUDIT.json",
    }
    try:
        _write_csv(paths["records"], RECORD_FIELDS, result.record_rows)
        _write_csv(paths["rankings"], RANKING_FIELDS, result.ranking_rows)
        _write_csv(paths["samples"], SAMPLE_FIELDS, result.sample_rows)
        _write_csv(paths["labels"], LABEL_FIELDS, result.label_rows)
        _write_csv(paths["confusions"], CONFUSION_FIELDS, result.confusion_rows)
        _write_json(paths["bootstrap"], _bootstrap_to_dict(result.bootstrap))
        _write_json(paths["inputs"], frozen_input_audit)
        summary = {
            "experiment": "12-B",
            "run_type": "development_preflight_only",
            "method": METHOD_ID,
            "query_group": DEVELOPMENT_GROUP,
            "gallery_record_count": 724,
            "gallery_sample_id_count": 342,
            "development_query_record_count": len(result.record_rows),
            "development_query_sample_id_count": len(result.sample_rows),
            "strict_reference_label_count": len(result.label_rows),
            "sample_equal_metrics": {
                f"top_{k}": result.sample_equal_metrics[k] for k in TOP_K_VALUES
            },
            "label_macro_metrics": {
                f"top_{k}": result.label_macro_metrics[k] for k in TOP_K_VALUES
            },
            "baseline_comparison": baseline_comparison,
            "bootstrap": _bootstrap_to_dict(result.bootstrap),
            "method_lock": {
                "path": str(Path(method_lock_path).relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(method_lock_path),
                "approval_status": "pending_human_approval",
            },
            "environment": environment_snapshot(),
            "cache_loading_limit": (
                "The monolithic NPZ is physically loaded, but only frozen Gallery and "
                "Development rows entered evaluation operations."
            ),
            "final_test_counters": {
                "vector_selection_count": 0,
                "similarity_row_count": 0,
                "prediction_count": 0,
                "score_count": 0,
            },
            "scientific_interpretation": (
                "Development retrieval hit rates relative to candidate database reference "
                "labels; cosine scores are not probabilities or confidence."
            ),
        }
        _write_json(paths["summary"], summary)
        audit = {
            "status": "completed_development_preflight",
            "completed_stages": [
                "method_lock_validation",
                "frozen_input_hash_validation",
                "manifest_identity_and_leakage_validation",
                "cache_index_manifest_mapping_validation",
                "gallery_and_development_selection",
                "development_method_a_evaluation",
                "sample_id_and_label_aggregation",
                "deterministic_bootstrap",
                "experiment_11_2_baseline_comparison",
                "atomic_output_staging",
            ],
            "final_test_vector_selection_count": 0,
            "final_test_similarity_row_count": 0,
            "final_test_prediction_count": 0,
            "final_test_score_count": 0,
            "output_file_names": sorted(path.name for path in paths.values()),
        }
        _write_json(paths["audit"], audit)
        os.replace(temporary, output_dir)
    except Exception as exc:
        failure = {
            "status": "failed",
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "final_test_vector_selection_count": 0,
            "final_test_similarity_row_count": 0,
            "final_test_prediction_count": 0,
            "final_test_score_count": 0,
        }
        failure_path = temporary / "RUN_AUDIT.failed.json"
        if not failure_path.exists():
            try:
                _write_json(failure_path, failure)
            except Exception:
                pass
        raise
    return tuple(output_dir / path.name for path in paths.values())


def write_final_test_outputs_atomic(
    output_dir: Path,
    result: EvaluationResult,
    frozen_input_audit: dict[str, object],
    authorization: dict[str, object] | None,
    method_lock_path: Path,
    executable_lock_path: Path,
    run_id: str,
    gallery_reference_row_count: int,
) -> tuple[Path, ...]:
    """Write one authorized final-test result set without overwriting history."""
    if result.query_group != FINAL_TEST_GROUP:
        raise ValueError("Final-test output writer received a non-final result")
    if not authorization or authorization.get("authorized") is not True:
        raise PermissionError("Final-test output authorization is missing")
    if authorization.get("run_id") != run_id:
        raise PermissionError("Final-test output run_id does not match authorization")
    query_count = len(result.record_rows)
    gallery_count = int(gallery_reference_row_count)
    if query_count <= 0 or gallery_count <= 0:
        raise ValueError("Formal query and Gallery row counts must be positive")
    similarity_pair_count = query_count * gallery_count
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Final-test output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=False, exist_ok=False)
    paths = {
        "records": temporary / "final_test_query_record_results.csv",
        "rankings": temporary / "final_test_full_label_rankings.csv",
        "samples": temporary / "final_test_sample_id_results.csv",
        "labels": temporary / "final_test_label_metrics.csv",
        "confusions": temporary / "final_test_confusion_pairs.csv",
        "errors": temporary / "final_test_error_cases.csv",
        "bootstrap": temporary / "final_test_bootstrap.json",
        "inputs": temporary / "frozen_input_audit.json",
        "summary": temporary / "final_test_summary.json",
        "audit": temporary / "RUN_AUDIT.json",
    }
    try:
        _write_csv(paths["records"], RECORD_FIELDS, result.record_rows)
        _write_csv(paths["rankings"], RANKING_FIELDS, result.ranking_rows)
        _write_csv(paths["samples"], SAMPLE_FIELDS, result.sample_rows)
        _write_csv(paths["labels"], LABEL_FIELDS, result.label_rows)
        _write_csv(paths["confusions"], CONFUSION_FIELDS, result.confusion_rows)
        _write_csv(
            paths["errors"],
            RECORD_FIELDS,
            [row for row in result.record_rows if int(row["hit_at_1"]) == 0],
        )
        _write_json(paths["bootstrap"], _bootstrap_to_dict(result.bootstrap))
        _write_json(paths["inputs"], frozen_input_audit)
        summary = {
            "experiment": "13",
            "run_type": "authorized_frozen_final_test",
            "run_id": run_id,
            "method": METHOD_ID,
            "query_group": FINAL_TEST_GROUP,
            "gallery_record_count": gallery_count,
            "gallery_sample_id_count": 342,
            "final_test_query_record_count": len(result.record_rows),
            "final_test_query_sample_id_count": len(result.sample_rows),
            "strict_reference_label_count": len(result.label_rows),
            "sample_equal_metrics": {
                f"top_{k}": result.sample_equal_metrics[k] for k in TOP_K_VALUES
            },
            "label_macro_metrics": {
                f"top_{k}": result.label_macro_metrics[k] for k in TOP_K_VALUES
            },
            "bootstrap": _bootstrap_to_dict(result.bootstrap),
            "method_lock_sha256": sha256_file(method_lock_path),
            "executable_lock_sha256": sha256_file(executable_lock_path),
            "approval_sha256": str(authorization["approval_sha256"]),
            "environment": environment_snapshot(),
            "scientific_interpretation": (
                "Frozen closed-set retrieval hit rates relative to candidate database "
                "reference labels; not externally validated mineral identification accuracy."
            ),
        }
        _write_json(paths["summary"], summary)
        audit = {
            "status": "completed_authorized_final_test",
            "run_id": run_id,
            "authorization_verified_before_vector_selection": True,
            "final_test_query_row_count": query_count,
            "gallery_reference_row_count": gallery_count,
            "similarity_pair_count": similarity_pair_count,
            "prediction_count": query_count,
            "scores_computed": True,
            "output_file_names": sorted(path.name for path in paths.values()),
        }
        _write_json(paths["audit"], audit)
        os.replace(temporary, output_dir)
    except Exception as exc:
        failure = {
            "status": "failed",
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "authorization_verified": True,
            "run_id": run_id,
            "existing_outputs_preserved": True,
        }
        failure_path = temporary / "RUN_AUDIT.failed.json"
        if not failure_path.exists():
            try:
                _write_json(failure_path, failure)
            except Exception:
                pass
        raise
    return tuple(output_dir / path.name for path in paths.values())

