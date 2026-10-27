"""Leave-one-Gallery-Sample_ID-out sensitivity for frozen Development queries."""

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
AGGREGATION_SCRIPT = PROJECT_ROOT / "scripts" / "11_compare_reference_label_aggregation.py"
DEFAULT_BASELINE_DIR = PROJECT_ROOT / "results" / "exp11_development_baseline"
DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "docs" / "evaluation" / "exp11_v1"
DEFAULT_LOCK = DEFAULT_SNAPSHOT_DIR / "DATA_SOURCE_LOCK.json"
DEFAULT_CACHE = PROJECT_ROOT / "results" / "exp09_batch_preprocessing_full" / "rruff_preprocessed_300_850_step0p5.npz"
DEFAULT_INDEX = PROJECT_ROOT / "results" / "exp09_batch_preprocessing_full" / "rruff_preprocessing_index.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "exp11_gallery_identity_sensitivity"
TOP_K_VALUES = (1, 3, 5)
EXPECTED_GALLERY_FILES = 724
EXPECTED_GALLERY_SAMPLE_IDS = 342
EXPECTED_DEVELOPMENT_FILES = 118
EXPECTED_DEVELOPMENT_SAMPLE_IDS = 60
EXPECTED_LABELS = 60

SCENARIO_FIELDS = [
    "scenario_id",
    "deleted_gallery_sample_id",
    "deleted_reference_label",
    "deleted_reference_label_group_key",
    "deleted_file_count",
    "gallery_file_count_before",
    "gallery_file_count_after",
    "gallery_sample_id_count_before",
    "gallery_sample_id_count_after",
    "label_sample_id_count_before",
    "label_sample_id_count_after",
    "record_top1_hit_rate",
    "record_top3_hit_rate",
    "record_top5_hit_rate",
    "sample_equal_top1_hit_rate",
    "sample_equal_top3_hit_rate",
    "sample_equal_top5_hit_rate",
    "record_top1_delta",
    "record_top3_delta",
    "record_top5_delta",
    "sample_equal_top1_delta",
    "sample_equal_top3_delta",
    "sample_equal_top5_delta",
]

QUERY_FIELDS = [
    "query_relative_path",
    "query_sample_id",
    "query_reference_label",
    "query_reference_label_group_key",
    "baseline_top1_prediction_label",
    "baseline_correct_label_rank",
    "deletion_scenario_count",
    "top1_prediction_changed_count",
    "top1_prediction_changed_rate",
    "correct_label_rank_best",
    "correct_label_rank_median",
    "correct_label_rank_worst",
    "top1_hit_status_changed_count",
    "top3_hit_status_changed_count",
    "top5_hit_status_changed_count",
]

SAMPLE_FIELDS = [
    "query_sample_id",
    "query_reference_label",
    "query_reference_label_group_key",
    "query_spectrum_count",
    "deletion_scenario_count",
    "measurement_scenario_evaluation_count",
    "top1_prediction_changed_measurement_scenario_count",
    "top1_prediction_changed_measurement_scenario_rate",
    "scenarios_with_any_top1_prediction_change",
    "correct_label_rank_best",
    "correct_label_rank_median",
    "correct_label_rank_worst",
    "top1_hit_status_changed_measurement_scenario_count",
    "top3_hit_status_changed_measurement_scenario_count",
    "top5_hit_status_changed_measurement_scenario_count",
]

INFLUENCE_FIELDS = [
    "deleted_gallery_sample_id",
    "deleted_reference_label",
    "deleted_reference_label_group_key",
    "deleted_file_count",
    "top1_prediction_changed_spectrum_count",
    "top1_prediction_changed_query_sample_id_count",
    "top1_hit_status_changed_spectrum_count",
    "top3_hit_status_changed_spectrum_count",
    "top5_hit_status_changed_spectrum_count",
    "correct_label_rank_changed_spectrum_count",
    "record_top1_delta",
    "record_top3_delta",
    "record_top5_delta",
    "sample_equal_top1_delta",
    "sample_equal_top3_delta",
    "sample_equal_top5_delta",
]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


aggregation = _load_module("raman_exp11_aggregation_for_sensitivity", AGGREGATION_SCRIPT)
diagnostics = aggregation.diagnostics
baseline = aggregation.baseline
exp10 = aggregation.exp10
exp11 = aggregation.exp11


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_gallery_identity_map(
    records: tuple[dict[str, str], ...],
    *,
    minimum_ids_per_label: int = 3,
) -> dict[str, dict[str, object]]:
    """Map each known Gallery identity to all its files and one frozen label."""
    aggregation.validate_gallery_for_aggregation(records)
    identities: dict[str, dict[str, object]] = {}
    label_ids: dict[str, set[str]] = defaultdict(set)
    for index, record in enumerate(records):
        sample_id = aggregation._known_sample_id(record)
        sample_key = sample_id.casefold()
        display, _basis, label_key = exp10._reference_label_identity(record)
        label_ids[label_key].add(sample_key)
        identity = identities.setdefault(
            sample_key,
            {
                "sample_id": sample_id,
                "label_key": label_key,
                "label_display": display,
                "indices": [],
            },
        )
        if identity["label_key"] != label_key:
            raise ValueError(f"Gallery Sample_ID spans labels: {sample_id}")
        identity["indices"].append(index)
    too_small = {
        label_key: len(sample_ids)
        for label_key, sample_ids in label_ids.items()
        if len(sample_ids) < minimum_ids_per_label
    }
    if too_small:
        raise ValueError(
            "One or more Gallery labels have fewer than "
            f"{minimum_ids_per_label} Sample_IDs: {too_small}"
        )
    for identity in identities.values():
        identity["label_sample_id_count_before"] = len(label_ids[str(identity["label_key"])])
    return identities


def remove_gallery_identity(
    records: tuple[dict[str, str], ...],
    similarity_matrix: np.ndarray,
    identity: dict[str, object],
) -> tuple[tuple[dict[str, str], ...], np.ndarray]:
    """Remove every Gallery file belonging to exactly one Sample_ID."""
    matrix = np.asarray(similarity_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(records):
        raise ValueError("Similarity matrix columns do not match Gallery records")
    removed = {int(index) for index in identity["indices"]}
    if not removed:
        raise ValueError("Gallery identity has no reference files")
    keep = np.asarray([index for index in range(len(records)) if index not in removed], dtype=int)
    remaining_records = tuple(records[index] for index in keep)
    remaining_matrix = matrix[:, keep]
    removed_sample_key = str(identity["sample_id"]).casefold()
    if any(aggregation._known_sample_id(record).casefold() == removed_sample_key for record in remaining_records):
        raise ValueError("Not all files for the deleted Gallery Sample_ID were removed")
    return remaining_records, remaining_matrix


def compute_similarity_matrix(
    gallery_matrix: np.ndarray, development_matrix: np.ndarray
) -> np.ndarray:
    """Compute only Development-to-Gallery similarities, never pairwise all-cache."""
    gallery = np.asarray(gallery_matrix, dtype=np.float64)
    development = np.asarray(development_matrix, dtype=np.float64)
    if gallery.ndim != 2 or development.ndim != 2 or gallery.shape[1] != development.shape[1]:
        raise ValueError("Gallery and Development matrices must share one feature axis")
    rows = [exp10.compute_cosine_similarities(gallery, query) for query in development]
    matrix = np.vstack(rows)
    if matrix.shape != (len(development), len(gallery)):
        raise ValueError("Unexpected Development-to-Gallery similarity matrix shape")
    return matrix


def evaluate_method_a_matrix(
    similarity_matrix: np.ndarray,
    gallery_records: tuple[dict[str, str], ...],
    query_label_keys: list[str],
    *,
    expected_label_count: int | None = None,
) -> list[dict[str, object]]:
    """Evaluate method A for every query using a precomputed similarity matrix."""
    matrix = np.asarray(similarity_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(query_label_keys) or matrix.shape[1] != len(gallery_records):
        raise ValueError("Similarity matrix, query identities, and Gallery do not align")
    groups = aggregation.validate_gallery_for_aggregation(gallery_records)
    if expected_label_count is not None and len(groups) != expected_label_count:
        raise ValueError(
            f"Expected {expected_label_count} labels after deletion, got {len(groups)}"
        )
    output: list[dict[str, object]] = []
    for values, target_key in zip(matrix, query_label_keys, strict=True):
        ranking = aggregation.rank_reference_labels(
            values, gallery_records, groups, aggregation.METHOD_A
        )
        output.append(aggregation._ranking_diagnostic(ranking, target_key))
    return output


def aggregate_metrics(
    query_results: list[dict[str, object]],
    query_sample_ids: list[str],
) -> dict[str, float]:
    if len(query_results) != len(query_sample_ids) or not query_results:
        raise ValueError("Query results and Sample_ID values must align and be non-empty")
    sample_groups: dict[str, list[int]] = defaultdict(list)
    for index, sample_id in enumerate(query_sample_ids):
        sample_groups[sample_id.casefold()].append(index)
    metrics: dict[str, float] = {}
    for k in TOP_K_VALUES:
        record_hits = [int(row[f"hit_at_{k}"]) for row in query_results]
        sample_means = [
            float(np.mean([record_hits[index] for index in indices]))
            for indices in sample_groups.values()
        ]
        metrics[f"record_top_{k}"] = float(np.mean(record_hits))
        metrics[f"sample_equal_top_{k}"] = float(np.mean(sample_means))
    return metrics


def _validate_full_baseline(
    query_results: list[dict[str, object]],
    metrics: dict[str, float],
    baseline_records: list[dict[str, str]],
    baseline_summary: dict[str, object],
) -> None:
    if len(query_results) != len(baseline_records):
        raise ValueError("Full-Gallery result count does not match baseline")
    for result, existing in zip(query_results, baseline_records, strict=True):
        if result["top1_label"] != existing["reference_label_top1_reference_label"]:
            raise ValueError("Full Gallery Top-1 label does not reproduce experiment 11.2")
        if int(result["correct_label_rank"]) != int(existing["reference_label_matching_rank"]):
            raise ValueError("Full Gallery correct-label rank does not reproduce experiment 11.2")
        for k in TOP_K_VALUES:
            if int(result[f"hit_at_{k}"]) != int(existing[f"reference_label_hit_at_{k}"]):
                raise ValueError(f"Full Gallery Top-{k} does not reproduce experiment 11.2")
    published = baseline_summary["metrics"]
    for k in TOP_K_VALUES:
        for scope, section in (
            ("record", "record_level_118_spectra"),
            ("sample_equal", "sample_id_equal_weight_60_sample_ids"),
        ):
            expected = float(published[section][f"reference_label_top_{k}"])
            actual = metrics[f"{scope}_top_{k}"]
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Full Gallery {scope} Top-{k} does not reproduce experiment 11.2")


def run_deletion_scenarios(
    similarity_matrix: np.ndarray,
    gallery_records: tuple[dict[str, str], ...],
    development_manifest: list[dict[str, str]],
    baseline_query_results: list[dict[str, object]],
    baseline_metrics: dict[str, float],
    identities: dict[str, dict[str, object]],
    *,
    expected_label_count: int = EXPECTED_LABELS,
    minimum_remaining_ids_per_label: int = 2,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Run every leave-one-identity-out scenario and accumulate query effects."""
    query_keys = [row["reference_label_group_key"] for row in development_manifest]
    query_samples = [row["sample_id"] for row in development_manifest]
    query_trackers = [
        {
            "ranks": [],
            "top1_changed_scenarios": set(),
            "hit_changed_scenarios": {k: set() for k in TOP_K_VALUES},
        }
        for _ in development_manifest
    ]
    scenario_rows: list[dict[str, object]] = []
    influence_rows: list[dict[str, object]] = []
    ordered_identities = sorted(
        identities.values(),
        key=lambda identity: (
            str(identity["sample_id"]).casefold(),
            str(identity["sample_id"]),
        ),
    )
    for scenario_number, identity in enumerate(ordered_identities, start=1):
        remaining_records, remaining_similarities = remove_gallery_identity(
            gallery_records, similarity_matrix, identity
        )
        remaining_identity_map = build_gallery_identity_map(
            remaining_records,
            minimum_ids_per_label=minimum_remaining_ids_per_label,
        )
        if len(remaining_identity_map) != len(identities) - 1:
            raise ValueError("Deletion scenario did not remove exactly one Gallery Sample_ID")
        scenario_results = evaluate_method_a_matrix(
            remaining_similarities,
            remaining_records,
            query_keys,
            expected_label_count=expected_label_count,
        )
        metrics = aggregate_metrics(scenario_results, query_samples)
        changed_query_indices: list[int] = []
        hit_change_counts = {k: 0 for k in TOP_K_VALUES}
        rank_change_count = 0
        for index, (current, original) in enumerate(
            zip(scenario_results, baseline_query_results, strict=True)
        ):
            tracker = query_trackers[index]
            tracker["ranks"].append(int(current["correct_label_rank"]))
            if current["top1_label_group_key"] != original["top1_label_group_key"]:
                tracker["top1_changed_scenarios"].add(scenario_number)
                changed_query_indices.append(index)
            if int(current["correct_label_rank"]) != int(original["correct_label_rank"]):
                rank_change_count += 1
            for k in TOP_K_VALUES:
                changed = int(current[f"hit_at_{k}"]) != int(original[f"hit_at_{k}"])
                if changed:
                    tracker["hit_changed_scenarios"][k].add(scenario_number)
                hit_change_counts[k] += int(changed)
        deltas = {
            key: metrics[key] - baseline_metrics[key] for key in baseline_metrics
        }
        label_before = int(identity["label_sample_id_count_before"])
        scenario_rows.append(
            {
                "scenario_id": f"delete_gallery_sample_{scenario_number:03d}",
                "deleted_gallery_sample_id": identity["sample_id"],
                "deleted_reference_label": identity["label_display"],
                "deleted_reference_label_group_key": identity["label_key"],
                "deleted_file_count": len(identity["indices"]),
                "gallery_file_count_before": len(gallery_records),
                "gallery_file_count_after": len(remaining_records),
                "gallery_sample_id_count_before": len(identities),
                "gallery_sample_id_count_after": len(remaining_identity_map),
                "label_sample_id_count_before": label_before,
                "label_sample_id_count_after": label_before - 1,
                **{f"record_top{k}_hit_rate": metrics[f"record_top_{k}"] for k in TOP_K_VALUES},
                **{f"sample_equal_top{k}_hit_rate": metrics[f"sample_equal_top_{k}"] for k in TOP_K_VALUES},
                **{f"record_top{k}_delta": deltas[f"record_top_{k}"] for k in TOP_K_VALUES},
                **{f"sample_equal_top{k}_delta": deltas[f"sample_equal_top_{k}"] for k in TOP_K_VALUES},
            }
        )
        changed_sample_ids = {
            query_samples[index].casefold() for index in changed_query_indices
        }
        influence_rows.append(
            {
                "deleted_gallery_sample_id": identity["sample_id"],
                "deleted_reference_label": identity["label_display"],
                "deleted_reference_label_group_key": identity["label_key"],
                "deleted_file_count": len(identity["indices"]),
                "top1_prediction_changed_spectrum_count": len(changed_query_indices),
                "top1_prediction_changed_query_sample_id_count": len(changed_sample_ids),
                "top1_hit_status_changed_spectrum_count": hit_change_counts[1],
                "top3_hit_status_changed_spectrum_count": hit_change_counts[3],
                "top5_hit_status_changed_spectrum_count": hit_change_counts[5],
                "correct_label_rank_changed_spectrum_count": rank_change_count,
                **{f"record_top{k}_delta": deltas[f"record_top_{k}"] for k in TOP_K_VALUES},
                **{f"sample_equal_top{k}_delta": deltas[f"sample_equal_top_{k}"] for k in TOP_K_VALUES},
            }
        )
    return scenario_rows, query_trackers, influence_rows


def build_query_stability(
    development_manifest: list[dict[str, str]],
    baseline_results: list[dict[str, object]],
    trackers: list[dict[str, object]],
    scenario_count: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for manifest, original, tracker in zip(
        development_manifest, baseline_results, trackers, strict=True
    ):
        ranks = list(tracker["ranks"])
        top1_changed_count = len(tracker["top1_changed_scenarios"])
        rows.append(
            {
                "query_relative_path": manifest["relative_path"],
                "query_sample_id": manifest["sample_id"],
                "query_reference_label": manifest["reference_label_group_display"],
                "query_reference_label_group_key": manifest["reference_label_group_key"],
                "baseline_top1_prediction_label": original["top1_label"],
                "baseline_correct_label_rank": original["correct_label_rank"],
                "deletion_scenario_count": scenario_count,
                "top1_prediction_changed_count": top1_changed_count,
                "top1_prediction_changed_rate": top1_changed_count / scenario_count,
                "correct_label_rank_best": min(ranks),
                "correct_label_rank_median": statistics.median(ranks),
                "correct_label_rank_worst": max(ranks),
                "top1_hit_status_changed_count": len(tracker["hit_changed_scenarios"][1]),
                "top3_hit_status_changed_count": len(tracker["hit_changed_scenarios"][3]),
                "top5_hit_status_changed_count": len(tracker["hit_changed_scenarios"][5]),
            }
        )
    return rows


def build_sample_stability(
    development_manifest: list[dict[str, str]],
    trackers: list[dict[str, object]],
    scenario_count: int,
) -> list[dict[str, object]]:
    if len(development_manifest) != len(trackers):
        raise ValueError("Development manifest and query trackers do not align")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(development_manifest):
        grouped[str(row["sample_id"])].append(index)
    output: list[dict[str, object]] = []
    for sample_id in sorted(grouped, key=lambda value: (value.casefold(), value)):
        indices = grouped[sample_id]
        manifests = [development_manifest[index] for index in indices]
        sample_trackers = [trackers[index] for index in indices]
        ranks = [
            int(rank)
            for tracker in sample_trackers
            for rank in tracker["ranks"]
        ]
        changed_total = sum(
            len(tracker["top1_changed_scenarios"]) for tracker in sample_trackers
        )
        changed_scenario_union = set().union(
            *(tracker["top1_changed_scenarios"] for tracker in sample_trackers)
        )
        output.append(
            {
                "query_sample_id": sample_id,
                "query_reference_label": manifests[0]["reference_label_group_display"],
                "query_reference_label_group_key": manifests[0]["reference_label_group_key"],
                "query_spectrum_count": len(indices),
                "deletion_scenario_count": scenario_count,
                "measurement_scenario_evaluation_count": len(indices) * scenario_count,
                "top1_prediction_changed_measurement_scenario_count": changed_total,
                "top1_prediction_changed_measurement_scenario_rate": changed_total / (len(indices) * scenario_count),
                "scenarios_with_any_top1_prediction_change": len(changed_scenario_union),
                "correct_label_rank_best": min(ranks),
                "correct_label_rank_median": statistics.median(ranks),
                "correct_label_rank_worst": max(ranks),
                "top1_hit_status_changed_measurement_scenario_count": sum(len(tracker["hit_changed_scenarios"][1]) for tracker in sample_trackers),
                "top3_hit_status_changed_measurement_scenario_count": sum(len(tracker["hit_changed_scenarios"][3]) for tracker in sample_trackers),
                "top5_hit_status_changed_measurement_scenario_count": sum(len(tracker["hit_changed_scenarios"][5]) for tracker in sample_trackers),
            }
        )
    return output


def summarize_scenario_distributions(
    scenario_rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for scope in ("record", "sample_equal"):
        for k in TOP_K_VALUES:
            metric_field = f"{scope}_top{k}_hit_rate"
            delta_field = f"{scope}_top{k}_delta"
            values = [float(row[metric_field]) for row in scenario_rows]
            deltas = [float(row[delta_field]) for row in scenario_rows]
            tolerance = 1e-15
            summary[f"{scope}_top_{k}"] = {
                "minimum": min(values),
                "median": statistics.median(values),
                "maximum": max(values),
                "delta_minimum": min(deltas),
                "delta_median": statistics.median(deltas),
                "delta_maximum": max(deltas),
                "decline_scenario_count": sum(delta < -tolerance for delta in deltas),
                "improve_scenario_count": sum(delta > tolerance for delta in deltas),
                "unchanged_scenario_count": sum(abs(delta) <= tolerance for delta in deltas),
            }
    return summary


def _git_version() -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    return {"head": run("rev-parse", "HEAD"), "describe": run("describe", "--always", "--dirty")}


def build_report(summary: dict[str, object]) -> str:
    baseline_metrics = summary["baseline_metrics"]
    distributions = summary["scenario_metric_distributions"]
    influential = summary["most_influential_gallery_identities"]
    influence_lines = "\n".join(
        f"- {row['sample_id']} / {row['reference_label']}: "
        f"Top-1预测变化 {row['changed_query_spectra']} 条，"
        f"Sample_ID等权Top-1变化 {row['sample_equal_top1_delta']:+.4%}。"
        for row in influential[:10]
    )
    return f"""# RamanAI 实验 11.3-C：Gallery Sample_ID 删除敏感性

## 范围

固定 Gallery 为 724 条光谱、342 个 Sample_ID，Development 为 118 条查询光谱、60 个
Sample_ID，共60个严格闭集候选参考标签。共执行342个逐一删除 Gallery Sample_ID 的场景；
每次删除该身份的全部参考文件。删除后每个标签至少保留2个 Gallery Sample_ID。

完整 Gallery 的 Sample_ID 等权基线为 Top-1
{baseline_metrics['sample_equal_top_1']:.4%}、Top-3
{baseline_metrics['sample_equal_top_3']:.4%}、Top-5
{baseline_metrics['sample_equal_top_5']:.4%}，与实验11.2一致。

## 场景分布

- Sample_ID等权 Top-1：最小 {distributions['sample_equal_top_1']['minimum']:.4%}，
  中位 {distributions['sample_equal_top_1']['median']:.4%}，最大
  {distributions['sample_equal_top_1']['maximum']:.4%}；下降场景
  {distributions['sample_equal_top_1']['decline_scenario_count']} 个。
- Sample_ID等权 Top-3：最小 {distributions['sample_equal_top_3']['minimum']:.4%}，
  中位 {distributions['sample_equal_top_3']['median']:.4%}，最大
  {distributions['sample_equal_top_3']['maximum']:.4%}；下降场景
  {distributions['sample_equal_top_3']['decline_scenario_count']} 个。
- Sample_ID等权 Top-5：最小 {distributions['sample_equal_top_5']['minimum']:.4%}，
  中位 {distributions['sample_equal_top_5']['median']:.4%}，最大
  {distributions['sample_equal_top_5']['maximum']:.4%}；下降场景
  {distributions['sample_equal_top_5']['decline_scenario_count']} 个。

## 影响较大的身份（诊断排序）

{influence_lines}

这些身份仅表示在固定 Development 上删除后产生较多排名或预测变化，不代表参考谱异常、
错误或应被删除。

## 隔离与限制

实验9.3 NPZ为单体压缩缓存，验证加载器会解压完整矩阵，因此这里是逻辑隔离而非物理
文件级隔离。实际余弦矩阵严格为118×724，只含 Development 查询和 Gallery 候选；未运行
final-test，也未将 final-test 行传入相似度计算。

342个删除场景共享同一批 Development 查询，不能视为342组独立样本。本实验衡量的是
Gallery身份删除敏感性，不是独立测试准确率。余弦相似度不是概率或置信度，候选参考标签
也不是独立验证的矿物真值。
"""


def output_paths(output_dir: Path) -> tuple[Path, ...]:
    return (
        output_dir / "gallery_deletion_scenarios.csv",
        output_dir / "development_query_stability.csv",
        output_dir / "development_sample_id_stability.csv",
        output_dir / "gallery_identity_influence.csv",
        output_dir / "gallery_identity_sensitivity_summary.json",
        output_dir / "gallery_identity_sensitivity_report.md",
    )


def write_outputs(
    output_dir: Path,
    scenario_rows: list[dict[str, object]],
    query_rows: list[dict[str, object]],
    sample_rows: list[dict[str, object]],
    influence_rows: list[dict[str, object]],
    summary: dict[str, object],
    report: str,
) -> tuple[Path, ...]:
    paths = output_paths(output_dir)
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Gallery sensitivity output already exists: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, fields, rows in (
        (paths[0], SCENARIO_FIELDS, scenario_rows),
        (paths[1], QUERY_FIELDS, query_rows),
        (paths[2], SAMPLE_FIELDS, sample_rows),
        (paths[3], INFLUENCE_FIELDS, influence_rows),
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


def run_sensitivity(
    *,
    baseline_dir: Path,
    cache_path: Path,
    index_path: Path,
    snapshot_dir: Path,
    lock_path: Path,
    output_dir: Path,
) -> tuple[dict[str, object], tuple[Path, ...]]:
    baseline_records, _samples, _labels, baseline_summary, baseline_hashes = diagnostics.load_and_validate_baseline_results(baseline_dir)
    lock = baseline.validate_source_lock(lock_path, snapshot_dir, cache_path, index_path)
    manifest_rows, preflight = baseline.validate_frozen_manifest(snapshot_dir / "evaluation_group_manifest.csv", lock)
    gallery, development_manifest, development_matrix = baseline.prepare_frozen_data(cache_path, index_path, manifest_rows)
    if len(gallery.reference_records) != EXPECTED_GALLERY_FILES or len(development_manifest) != EXPECTED_DEVELOPMENT_FILES:
        raise ValueError("Frozen Gallery or Development record count differs from protocol")
    identities = build_gallery_identity_map(gallery.reference_records, minimum_ids_per_label=3)
    if len(identities) != EXPECTED_GALLERY_SAMPLE_IDS:
        raise ValueError(f"Expected 342 Gallery Sample_IDs, got {len(identities)}")
    similarity_matrix = compute_similarity_matrix(gallery.reference_matrix, development_matrix)
    if similarity_matrix.shape != (EXPECTED_DEVELOPMENT_FILES, EXPECTED_GALLERY_FILES):
        raise ValueError("Similarity matrix must be exactly 118 x 724")
    query_keys = [row["reference_label_group_key"] for row in development_manifest]
    query_samples = [row["sample_id"] for row in development_manifest]
    baseline_results = evaluate_method_a_matrix(
        similarity_matrix, gallery.reference_records, query_keys, expected_label_count=EXPECTED_LABELS
    )
    baseline_metrics = aggregate_metrics(baseline_results, query_samples)
    _validate_full_baseline(baseline_results, baseline_metrics, baseline_records, baseline_summary)
    scenario_rows, trackers, influence_rows = run_deletion_scenarios(
        similarity_matrix,
        gallery.reference_records,
        development_manifest,
        baseline_results,
        baseline_metrics,
        identities,
    )
    if len(scenario_rows) != EXPECTED_GALLERY_SAMPLE_IDS:
        raise ValueError("Deletion scenario count does not equal Gallery Sample_ID count")
    query_rows = build_query_stability(
        development_manifest, baseline_results, trackers, len(scenario_rows)
    )
    sample_rows = build_sample_stability(
        development_manifest, trackers, len(scenario_rows)
    )
    distributions = summarize_scenario_distributions(scenario_rows)
    sorted_influence = sorted(
        influence_rows,
        key=lambda row: (
            -int(row["top1_prediction_changed_spectrum_count"]),
            float(row["sample_equal_top1_delta"]),
            str(row["deleted_gallery_sample_id"]).casefold(),
        ),
    )
    summary: dict[str, object] = {
        "experiment": "11.3-C",
        "scope": "frozen_development_leave_one_gallery_sample_id_out",
        "denominators": {
            "gallery_spectra": len(gallery.reference_records),
            "gallery_sample_ids": len(identities),
            "development_query_spectra": len(development_manifest),
            "development_query_sample_ids": len({row["sample_id"].casefold() for row in development_manifest}),
            "strict_closed_set_reference_labels": EXPECTED_LABELS,
            "deletion_scenarios": len(scenario_rows),
        },
        "baseline_metrics": baseline_metrics,
        "scenario_metric_distributions": distributions,
        "query_stability": {
            "queries_ever_changing_top1_prediction": sum(int(row["top1_prediction_changed_count"]) > 0 for row in query_rows),
            "query_sample_ids_ever_changing_top1_prediction": len({row["query_sample_id"] for row in query_rows if int(row["top1_prediction_changed_count"]) > 0}),
            "total_top1_prediction_changes_across_query_scenarios": sum(int(row["top1_prediction_changed_count"]) for row in query_rows),
            "maximum_query_top1_change_count": max(int(row["top1_prediction_changed_count"]) for row in query_rows),
        },
        "most_influential_gallery_identities": [
            {
                "sample_id": row["deleted_gallery_sample_id"],
                "reference_label": row["deleted_reference_label"],
                "deleted_file_count": row["deleted_file_count"],
                "changed_query_spectra": row["top1_prediction_changed_spectrum_count"],
                "changed_query_sample_ids": row["top1_prediction_changed_query_sample_id_count"],
                "sample_equal_top1_delta": row["sample_equal_top1_delta"],
            }
            for row in sorted_influence[:20]
        ],
        "safety_checks": {
            **preflight,
            "source_lock_validated": True,
            "full_gallery_reproduced_experiment_11_2": True,
            "similarity_matrix_shape": list(similarity_matrix.shape),
            "similarity_matrix_contains_only_gallery_and_development": True,
            "minimum_gallery_sample_ids_per_label_before_deletion": min(int(identity["label_sample_id_count_before"]) for identity in identities.values()),
            "minimum_gallery_sample_ids_per_label_after_deletion": min(int(row["label_sample_id_count_after"]) for row in scenario_rows),
            "final_test_queries_executed": 0,
            "final_test_rows_selected_for_similarity": 0,
        },
        "cache_loading_limit": (
            "The experiment 9.3 NPZ is monolithic and the validated loader decompresses "
            "the full cache matrix for identity checks. Isolation is logical, not physical-file-level. "
            "The computed similarity matrix is strictly 118 Development x 724 Gallery."
        ),
        "input_hashes": {
            "baseline_files": baseline_hashes,
            "data_source_lock": sha256_file(lock_path),
            "frozen_manifest": sha256_file(snapshot_dir / "evaluation_group_manifest.csv"),
            "cache": sha256_file(cache_path),
            "index": sha256_file(index_path),
        },
        "code_version": {**_git_version(), "sensitivity_script_sha256": sha256_file(Path(__file__))},
        "scientific_interpretation": (
            "Scenarios reuse the same Development queries and measure Gallery identity deletion "
            "sensitivity, not independent test accuracy. Influence does not prove a reference is abnormal."
        ),
    }
    report = build_report(summary)
    paths = write_outputs(output_dir, scenario_rows, query_rows, sample_rows, influence_rows, summary, report)
    return summary, paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze leave-one-Gallery-Sample_ID-out sensitivity.")
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        summary, paths = run_sensitivity(
            baseline_dir=args.baseline_dir,
            cache_path=args.cache,
            index_path=args.index,
            snapshot_dir=args.snapshot_dir,
            lock_path=args.lock,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Deletion scenarios: {summary['denominators']['deletion_scenarios']}")
    for k in TOP_K_VALUES:
        distribution = summary["scenario_metric_distributions"][f"sample_equal_top_{k}"]
        print(
            f"Sample_ID Top-{k}: min={distribution['minimum']:.6f}, "
            f"median={distribution['median']:.6f}, max={distribution['maximum']:.6f}"
        )
    print("Final-test similarity rows: 0")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
