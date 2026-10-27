"""Compare frozen Development reference-label aggregation rules A and B."""

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
ERROR_ANALYSIS_SCRIPT = PROJECT_ROOT / "scripts" / "11_analyze_development_errors.py"
DEFAULT_BASELINE_DIR = PROJECT_ROOT / "results" / "exp11_development_baseline"
DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "docs" / "evaluation" / "exp11_v1"
DEFAULT_LOCK = DEFAULT_SNAPSHOT_DIR / "DATA_SOURCE_LOCK.json"
DEFAULT_CACHE = (
    PROJECT_ROOT / "results" / "exp09_batch_preprocessing_full"
    / "rruff_preprocessed_300_850_step0p5.npz"
)
DEFAULT_INDEX = (
    PROJECT_ROOT / "results" / "exp09_batch_preprocessing_full"
    / "rruff_preprocessing_index.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "exp11_reference_label_aggregation"
TOP_K_VALUES = (1, 3, 5)
METHOD_A = "A_max_spectrum_cosine"
METHOD_B = "B_sample_id_equal_mean"

METRIC_FIELDS = [
    "method",
    "score_definition",
    "query_spectrum_count",
    "query_sample_id_count",
    "gallery_spectrum_count",
    "gallery_sample_id_count",
    "reference_label_count",
    "record_top1_hit_rate",
    "record_top3_hit_rate",
    "record_top5_hit_rate",
    "sample_equal_top1_hit_rate",
    "sample_equal_top3_hit_rate",
    "sample_equal_top5_hit_rate",
]

QUERY_FIELDS = [
    "query_relative_path",
    "query_cache_row_index",
    "query_sample_id",
    "query_reference_label",
    "query_reference_label_group_key",
    "query_excitation_wavelength_nm",
    "method_a_top1_label",
    "method_a_top1_label_group_key",
    "method_a_top1_score",
    "method_a_correct_label_rank",
    "method_a_correct_label_score",
    "method_a_hit_at_1",
    "method_a_hit_at_3",
    "method_a_hit_at_5",
    "method_b_top1_label",
    "method_b_top1_label_group_key",
    "method_b_top1_score",
    "method_b_correct_label_rank",
    "method_b_correct_label_score",
    "method_b_hit_at_1",
    "method_b_hit_at_3",
    "method_b_hit_at_5",
    "top1_prediction_changed",
    "top1_outcome_change",
    "method_b_rank_minus_method_a_rank",
]

SAMPLE_FIELDS = [
    "query_sample_id",
    "query_reference_label",
    "query_reference_label_group_key",
    "query_spectrum_count",
    "method_a_top1_prediction_labels",
    "method_b_top1_prediction_labels",
    "any_top1_prediction_changed",
    "method_a_top1_hit_fraction",
    "method_b_top1_hit_fraction",
    "method_b_minus_a_top1_hit_fraction",
    "method_a_top3_hit_fraction",
    "method_b_top3_hit_fraction",
    "method_b_minus_a_top3_hit_fraction",
    "method_a_top5_hit_fraction",
    "method_b_top5_hit_fraction",
    "method_b_minus_a_top5_hit_fraction",
    "method_a_correct_rank_median",
    "method_b_correct_rank_median",
]

CONFUSION_FIELDS = [
    "method",
    "true_reference_label",
    "true_reference_label_group_key",
    "top1_wrong_reference_label",
    "top1_wrong_reference_label_group_key",
    "error_spectrum_count",
    "distinct_query_sample_id_count",
    "query_sample_ids",
]

SIZE_FIELDS = [
    "method",
    "gallery_sample_id_size_group",
    "label_count",
    "development_query_spectrum_count",
    "development_query_sample_id_count",
    "record_top1_hit_rate",
    "record_top3_hit_rate",
    "record_top5_hit_rate",
    "sample_equal_top1_hit_rate",
    "sample_equal_top3_hit_rate",
    "sample_equal_top5_hit_rate",
    "interpretation_note",
]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


diagnostics = _load_module(
    "raman_exp11_error_analysis_for_aggregation", ERROR_ANALYSIS_SCRIPT
)
baseline = diagnostics.baseline
exp10 = diagnostics.exp10
exp11 = diagnostics.exp11


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _known_sample_id(record: dict[str, str]) -> str:
    value = exp10._known_sample_id(record)
    if value is None:
        raise ValueError(
            "Frozen Gallery contains missing, unknown, or conflicted Sample_ID: "
            + record.get("relative_path", "")
        )
    return value


def validate_gallery_for_aggregation(
    records: Iterable[dict[str, str]],
) -> list[dict[str, object]]:
    """Validate formal Gallery identities and build immutable label/sample groups."""
    rows = list(records)
    if not rows:
        raise ValueError("Gallery is empty")
    paths = [row.get("relative_path", "").strip() for row in rows]
    if any(not path for path in paths) or len(set(paths)) != len(paths):
        raise ValueError("Gallery contains an empty or duplicate relative_path")
    sample_to_label: dict[str, str] = {}
    label_groups: dict[str, dict[str, object]] = {}
    for index, record in enumerate(rows):
        display, basis, key = exp10._reference_label_identity(record)
        if basis != "safe_unicode_whitespace_case_normalization":
            raise ValueError(
                "Frozen Gallery contains missing or unusable reference_label: "
                + record["relative_path"]
            )
        sample_id = _known_sample_id(record)
        sample_key = sample_id.casefold()
        previous = sample_to_label.setdefault(sample_key, key)
        if previous != key:
            raise ValueError(
                f"Gallery Sample_ID is associated with multiple labels: {sample_id}"
            )
        group = label_groups.setdefault(
            key,
            {
                "key": key,
                "display": display,
                "indices": [],
                "sample_indices": defaultdict(list),
            },
        )
        group["indices"].append(index)
        group["sample_indices"][sample_key].append(index)
    return [label_groups[key] for key in sorted(label_groups)]


def _best_index(
    indices: list[int], similarities: np.ndarray, records: tuple[dict[str, str], ...]
) -> int:
    return min(
        indices,
        key=lambda index: (
            -float(similarities[index]),
            records[index]["relative_path"].casefold(),
            records[index]["relative_path"],
        ),
    )


def rank_reference_labels(
    similarities: np.ndarray,
    gallery_records: tuple[dict[str, str], ...],
    label_groups: list[dict[str, object]],
    method: str,
) -> list[dict[str, object]]:
    """Rank labels by fixed method A or fixed Sample_ID-equal method B."""
    values = exp10.correct_cosine_boundaries(np.asarray(similarities, dtype=np.float64))
    if len(values) != len(gallery_records):
        raise ValueError("Similarity vector length does not match Gallery records")
    if method not in {METHOD_A, METHOD_B}:
        raise ValueError(f"Unknown aggregation method: {method}")
    ranked: list[dict[str, object]] = []
    for group in label_groups:
        indices = list(group["indices"])
        sample_indices = group["sample_indices"]
        if method == METHOD_A:
            score = max(float(values[index]) for index in indices)
            sample_means: dict[str, float] = {
                sample_key: float(np.mean(values[np.asarray(member_indices, dtype=int)]))
                for sample_key, member_indices in sample_indices.items()
            }
        else:
            sample_means = {
                sample_key: float(np.mean(values[np.asarray(member_indices, dtype=int)]))
                for sample_key, member_indices in sample_indices.items()
            }
            score = float(np.mean(list(sample_means.values())))
        best_index = _best_index(indices, values, gallery_records)
        best_sample_key = min(
            sample_means,
            key=lambda sample_key: (-sample_means[sample_key], sample_key),
        )
        ranked.append(
            {
                "score": score,
                "reference_label_group_key": group["key"],
                "reference_label_group_display": group["display"],
                "record": gallery_records[best_index],
                "best_individual_similarity": float(values[best_index]),
                "best_sample_id_key": best_sample_key,
                "best_sample_id_mean_similarity": sample_means[best_sample_key],
                "label_reference_file_count": len(indices),
                "label_sample_id_count": len(sample_indices),
                "score_definition": (
                    "maximum_gallery_spectrum_cosine"
                    if method == METHOD_A
                    else "mean_of_within_sample_id_mean_cosines"
                ),
            }
        )
    ranked.sort(
        key=lambda row: (
            -float(row["score"]),
            str(row["reference_label_group_display"]).casefold(),
            str(row["reference_label_group_key"]),
            str(row["record"]["relative_path"]).casefold(),
            str(row["record"]["relative_path"]),
        )
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["cosine_similarity"] = row["score"]
        row["reference_label_grouping_basis"] = (
            "safe_unicode_whitespace_case_normalization"
        )
    return ranked


def _ranking_diagnostic(
    ranking: list[dict[str, object]], target_key: str
) -> dict[str, object]:
    matching = [row for row in ranking if row["reference_label_group_key"] == target_key]
    if len(matching) != 1:
        raise ValueError("Query reference-label key must occur exactly once in ranking")
    correct = matching[0]
    top = ranking[0]
    return {
        "top1_label": top["reference_label_group_display"],
        "top1_label_group_key": top["reference_label_group_key"],
        "top1_score": float(top["score"]),
        "correct_label_rank": int(correct["rank"]),
        "correct_label_score": float(correct["score"]),
        "hit_at_1": int(int(correct["rank"]) <= 1),
        "hit_at_3": int(int(correct["rank"]) <= 3),
        "hit_at_5": int(int(correct["rank"]) <= 5),
    }


def _validate_method_a_against_baseline(
    existing: dict[str, str], result: dict[str, object]
) -> None:
    if existing["reference_label_top1_reference_label"] != result["top1_label"]:
        raise ValueError("Method A Top-1 label does not reproduce experiment 11.2")
    if int(existing["reference_label_matching_rank"]) != result["correct_label_rank"]:
        raise ValueError("Method A correct-label rank does not reproduce experiment 11.2")
    if not math.isclose(
        float(existing["reference_label_top1_cosine_similarity"]),
        float(result["top1_score"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Method A Top-1 score does not reproduce experiment 11.2")
    for k in TOP_K_VALUES:
        if int(existing[f"reference_label_hit_at_{k}"]) != int(result[f"hit_at_{k}"]):
            raise ValueError(f"Method A Top-{k} does not reproduce experiment 11.2")


def compare_queries(
    gallery_library,
    development_manifest: list[dict[str, str]],
    development_matrix: np.ndarray,
    baseline_record_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Compute one Gallery similarity vector per query and apply both rules."""
    existing = {row["query_relative_path"]: row for row in baseline_record_rows}
    if len(existing) != len(baseline_record_rows):
        raise ValueError("Baseline results contain duplicate query_relative_path")
    development_paths = {row["relative_path"] for row in development_manifest}
    if set(existing) != development_paths:
        raise ValueError("Baseline and frozen Development query identities differ")
    label_groups = validate_gallery_for_aggregation(gallery_library.reference_records)
    if len(label_groups) != 60:
        raise ValueError(f"Expected 60 frozen reference labels, got {len(label_groups)}")
    output: list[dict[str, object]] = []
    for manifest, vector in zip(development_manifest, development_matrix, strict=True):
        path = manifest["relative_path"]
        baseline_row = existing[path]
        for baseline_field, manifest_field in (
            ("query_cache_row_index", "cache_row_index"),
            ("query_sample_id", "sample_id"),
            ("query_reference_label_group_key", "reference_label_group_key"),
        ):
            if baseline_row[baseline_field] != manifest[manifest_field]:
                raise ValueError(f"Baseline/frozen identity mismatch for {path}: {baseline_field}")
        similarities = exp10.compute_cosine_similarities(
            gallery_library.reference_matrix, np.asarray(vector, dtype=np.float64)
        )
        ranking_a = rank_reference_labels(
            similarities, gallery_library.reference_records, label_groups, METHOD_A
        )
        ranking_b = rank_reference_labels(
            similarities, gallery_library.reference_records, label_groups, METHOD_B
        )
        target_key = manifest["reference_label_group_key"]
        result_a = _ranking_diagnostic(ranking_a, target_key)
        result_b = _ranking_diagnostic(ranking_b, target_key)
        _validate_method_a_against_baseline(baseline_row, result_a)
        if result_a["hit_at_1"] == result_b["hit_at_1"]:
            outcome = "unchanged_hit" if result_a["hit_at_1"] else "unchanged_miss"
        else:
            outcome = "improved_to_hit" if result_b["hit_at_1"] else "lost_hit"
        row: dict[str, object] = {
            "query_relative_path": path,
            "query_cache_row_index": manifest["cache_row_index"],
            "query_sample_id": manifest["sample_id"],
            "query_reference_label": manifest["reference_label_group_display"],
            "query_reference_label_group_key": target_key,
            "query_excitation_wavelength_nm": diagnostics.normalize_excitation_wavelength(
                manifest.get("excitation_wavelength_nm")
            ),
        }
        for prefix, result in (("method_a", result_a), ("method_b", result_b)):
            for key, value in result.items():
                row[f"{prefix}_{key}"] = value
        row["top1_prediction_changed"] = (
            result_a["top1_label_group_key"] != result_b["top1_label_group_key"]
        )
        row["top1_outcome_change"] = outcome
        row["method_b_rank_minus_method_a_rank"] = (
            int(result_b["correct_label_rank"]) - int(result_a["correct_label_rank"])
        )
        output.append(row)
    return output


def aggregate_sample_comparison(
    query_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in query_rows:
        groups[str(row["query_sample_id"])].append(row)
    output: list[dict[str, object]] = []
    for sample_id in sorted(groups, key=lambda value: (value.casefold(), value)):
        rows = groups[sample_id]
        keys = {str(row["query_reference_label_group_key"]) for row in rows}
        if len(keys) != 1:
            raise ValueError(f"Development Sample_ID spans labels: {sample_id}")
        result: dict[str, object] = {
            "query_sample_id": sample_id,
            "query_reference_label": rows[0]["query_reference_label"],
            "query_reference_label_group_key": rows[0]["query_reference_label_group_key"],
            "query_spectrum_count": len(rows),
            "method_a_top1_prediction_labels": " | ".join(sorted({str(row["method_a_top1_label"]) for row in rows}, key=str.casefold)),
            "method_b_top1_prediction_labels": " | ".join(sorted({str(row["method_b_top1_label"]) for row in rows}, key=str.casefold)),
            "any_top1_prediction_changed": any(bool(row["top1_prediction_changed"]) for row in rows),
            "method_a_correct_rank_median": statistics.median(int(row["method_a_correct_label_rank"]) for row in rows),
            "method_b_correct_rank_median": statistics.median(int(row["method_b_correct_label_rank"]) for row in rows),
        }
        for k in TOP_K_VALUES:
            a = float(np.mean([int(row[f"method_a_hit_at_{k}"]) for row in rows]))
            b = float(np.mean([int(row[f"method_b_hit_at_{k}"]) for row in rows]))
            result[f"method_a_top{k}_hit_fraction"] = a
            result[f"method_b_top{k}_hit_fraction"] = b
            result[f"method_b_minus_a_top{k}_hit_fraction"] = b - a
        output.append(result)
    return output


def calculate_metrics(
    query_rows: list[dict[str, object]],
    sample_rows: list[dict[str, object]],
    gallery_library,
) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    metric_rows: list[dict[str, object]] = []
    metric_map: dict[str, dict[str, float]] = {}
    for method, prefix, definition in (
        (METHOD_A, "method_a", "maximum cosine among all Gallery spectra in label"),
        (METHOD_B, "method_b", "mean of within-Sample_ID mean cosines, equal Sample_ID weight"),
    ):
        metrics: dict[str, float] = {}
        row: dict[str, object] = {
            "method": method,
            "score_definition": definition,
            "query_spectrum_count": len(query_rows),
            "query_sample_id_count": len(sample_rows),
            "gallery_spectrum_count": len(gallery_library.reference_records),
            "gallery_sample_id_count": len({_known_sample_id(record).casefold() for record in gallery_library.reference_records}),
            "reference_label_count": 60,
        }
        for k in TOP_K_VALUES:
            record_rate = float(np.mean([int(item[f"{prefix}_hit_at_{k}"]) for item in query_rows]))
            sample_rate = float(np.mean([float(item[f"{prefix}_top{k}_hit_fraction"]) for item in sample_rows]))
            row[f"record_top{k}_hit_rate"] = record_rate
            row[f"sample_equal_top{k}_hit_rate"] = sample_rate
            metrics[f"record_top_{k}"] = record_rate
            metrics[f"sample_equal_top_{k}"] = sample_rate
        metric_rows.append(row)
        metric_map[method] = metrics
    return metric_rows, metric_map


def build_confusion_pairs(query_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for method, prefix in ((METHOD_A, "method_a"), (METHOD_B, "method_b")):
        groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for row in query_rows:
            if int(row[f"{prefix}_hit_at_1"]) == 0:
                groups[(str(row["query_reference_label_group_key"]), str(row[f"{prefix}_top1_label_group_key"]))].append(row)
        for key, rows in groups.items():
            output.append(
                {
                    "method": method,
                    "true_reference_label": rows[0]["query_reference_label"],
                    "true_reference_label_group_key": key[0],
                    "top1_wrong_reference_label": rows[0][f"{prefix}_top1_label"],
                    "top1_wrong_reference_label_group_key": key[1],
                    "error_spectrum_count": len(rows),
                    "distinct_query_sample_id_count": len({str(row["query_sample_id"]) for row in rows}),
                    "query_sample_ids": " | ".join(sorted({str(row["query_sample_id"]) for row in rows}, key=str.casefold)),
                }
            )
    output.sort(key=lambda row: (str(row["method"]), -int(row["error_spectrum_count"]), str(row["true_reference_label"]).casefold(), str(row["top1_wrong_reference_label"]).casefold()))
    return output


def _size_group(count: int) -> str:
    return "3-4" if count <= 4 else "5-7" if count <= 7 else ">=8"


def build_gallery_size_summary(
    query_rows: list[dict[str, object]],
    sample_rows: list[dict[str, object]],
    gallery_records: tuple[dict[str, str], ...],
) -> list[dict[str, object]]:
    sizes = diagnostics.gallery_label_sizes(gallery_records)
    sample_by_key = {str(row["query_reference_label_group_key"]): row for row in sample_rows}
    query_by_key: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in query_rows:
        query_by_key[str(row["query_reference_label_group_key"])].append(row)
    output: list[dict[str, object]] = []
    for method, prefix in ((METHOD_A, "method_a"), (METHOD_B, "method_b")):
        for group_name in ("3-4", "5-7", ">=8"):
            keys = [key for key, size in sizes.items() if _size_group(int(size["sample_id_count"])) == group_name]
            records = [row for key in keys for row in query_by_key[key]]
            samples = [sample_by_key[key] for key in keys]
            row: dict[str, object] = {
                "method": method,
                "gallery_sample_id_size_group": group_name,
                "label_count": len(keys),
                "development_query_spectrum_count": len(records),
                "development_query_sample_id_count": len(samples),
                "interpretation_note": "descriptive_only_no_causal_interpretation",
            }
            for k in TOP_K_VALUES:
                row[f"record_top{k}_hit_rate"] = float(np.mean([int(item[f"{prefix}_hit_at_{k}"]) for item in records]))
                row[f"sample_equal_top{k}_hit_rate"] = float(np.mean([float(item[f"{prefix}_top{k}_hit_fraction"]) for item in samples]))
            output.append(row)
    return output


def _validate_method_a_metrics(
    metric_map: dict[str, dict[str, float]], baseline_summary: dict[str, object]
) -> None:
    published = baseline_summary["metrics"]
    for k in TOP_K_VALUES:
        for scope, published_section, published_key in (
            ("record", "record_level_118_spectra", f"reference_label_top_{k}"),
            ("sample_equal", "sample_id_equal_weight_60_sample_ids", f"reference_label_top_{k}"),
        ):
            actual = metric_map[METHOD_A][f"{scope}_top_{k}"]
            expected = float(published[published_section][published_key])
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Method A {scope} Top-{k} does not reproduce experiment 11.2")


def _git_version() -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    return {"head": run("rev-parse", "HEAD"), "describe": run("describe", "--always", "--dirty")}


def build_report(summary: dict[str, object]) -> str:
    metrics = summary["metrics"]
    a = metrics[METHOD_A]
    b = metrics[METHOD_B]
    changes = summary["prediction_changes"]
    lines = []
    for k in TOP_K_VALUES:
        lines.append(
            f"| Top-{k} | {a[f'record_top_{k}']:.4%} | {b[f'record_top_{k}']:.4%} | "
            f"{a[f'sample_equal_top_{k}']:.4%} | {b[f'sample_equal_top_{k}']:.4%} |"
        )
    table = "\n".join(lines)
    return f"""# RamanAI 实验 11.3-B：参考标签聚合规则对照

## 固定范围

本实验只使用冻结的 724 条 Gallery 和 118 条 Development 光谱，涉及 342 个 Gallery
Sample_ID、60 个 Development Sample_ID 和 60 个严格闭集候选参考标签。final-test 查询数和
进入相似度计算的 final-test 行数均为 0。

方法 A 是标签内所有 Gallery 光谱的最大余弦相似度。方法 B 先在标签内对同一 Gallery
Sample_ID 的测量余弦相似度取算术平均，再对不同 Sample_ID 的均值等权平均。方法定义在
计算前固定，未根据结果调整。

## 指标

| 指标 | A逐光谱 | B逐光谱 | A按Sample_ID等权 | B按Sample_ID等权 |
|---|---:|---:|---:|---:|
{table}

方法 B 相对方法 A 的 Sample_ID 等权变化：Top-1
{summary['metric_deltas']['sample_equal_top_1']:+.4%}、Top-3
{summary['metric_deltas']['sample_equal_top_3']:+.4%}、Top-5
{summary['metric_deltas']['sample_equal_top_5']:+.4%}。

共有 {changes['query_spectra_with_changed_top1_prediction']} 条查询光谱、
{changes['sample_ids_with_any_changed_top1_prediction']} 个查询 Sample_ID 的 Top-1 预测发生变化。

## 隔离和解释

实验 9.3 NPZ 是单体压缩缓存，现有验证加载器会为完整身份核对而解压整个矩阵，因此是
逻辑隔离而不是物理文件级隔离。实际余弦相似度计算仅接收 Gallery 和 Development 切片，
未选择 final-test 行。

两种得分是不同聚合统计量，不应直接解释为同一尺度的概率或置信度。候选参考标签不是
独立验证的矿物真值，Development 变化也不等于真实矿物识别准确率变化。Gallery 规模分组
仅用于描述，不支持因果结论。
"""


def output_paths(output_dir: Path) -> tuple[Path, ...]:
    return (
        output_dir / "aggregation_method_metrics.csv",
        output_dir / "aggregation_query_comparison.csv",
        output_dir / "aggregation_sample_id_comparison.csv",
        output_dir / "aggregation_confusion_pairs.csv",
        output_dir / "aggregation_gallery_size_summary.csv",
        output_dir / "aggregation_comparison_summary.json",
        output_dir / "aggregation_comparison_report.md",
    )


def write_outputs(
    output_dir: Path,
    metric_rows: list[dict[str, object]],
    query_rows: list[dict[str, object]],
    sample_rows: list[dict[str, object]],
    confusion_rows: list[dict[str, object]],
    size_rows: list[dict[str, object]],
    summary: dict[str, object],
    report: str,
) -> tuple[Path, ...]:
    paths = output_paths(output_dir)
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Aggregation comparison output already exists: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, fields, rows in (
        (paths[0], METRIC_FIELDS, metric_rows),
        (paths[1], QUERY_FIELDS, query_rows),
        (paths[2], SAMPLE_FIELDS, sample_rows),
        (paths[3], CONFUSION_FIELDS, confusion_rows),
        (paths[4], SIZE_FIELDS, size_rows),
    ):
        with path.open("x", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with paths[5].open("x", encoding="utf-8-sig", newline="") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    paths[6].write_text(report, encoding="utf-8-sig", newline="\n")
    return paths


def run_comparison(
    *,
    baseline_dir: Path,
    cache_path: Path,
    index_path: Path,
    snapshot_dir: Path,
    lock_path: Path,
    output_dir: Path,
) -> tuple[dict[str, object], tuple[Path, ...]]:
    baseline_records, _sample_rows, _label_rows, baseline_summary, baseline_hashes = diagnostics.load_and_validate_baseline_results(baseline_dir)
    lock = baseline.validate_source_lock(lock_path, snapshot_dir, cache_path, index_path)
    manifest_rows, preflight = baseline.validate_frozen_manifest(snapshot_dir / "evaluation_group_manifest.csv", lock)
    gallery, development_manifest, development_matrix = baseline.prepare_frozen_data(cache_path, index_path, manifest_rows)
    query_rows = compare_queries(gallery, development_manifest, development_matrix, baseline_records)
    sample_rows = aggregate_sample_comparison(query_rows)
    metric_rows, metric_map = calculate_metrics(query_rows, sample_rows, gallery)
    _validate_method_a_metrics(metric_map, baseline_summary)
    confusion_rows = build_confusion_pairs(query_rows)
    size_rows = build_gallery_size_summary(query_rows, sample_rows, gallery.reference_records)
    deltas = {
        key: metric_map[METHOD_B][key] - metric_map[METHOD_A][key]
        for key in metric_map[METHOD_A]
    }
    changed_samples = [row for row in sample_rows if bool(row["any_top1_prediction_changed"])]
    summary: dict[str, object] = {
        "experiment": "11.3-B",
        "scope": "frozen_development_reference_label_aggregation_comparison",
        "method_definitions": {
            METHOD_A: "maximum cosine similarity among all Gallery spectra in label",
            METHOD_B: "arithmetic mean within each Gallery Sample_ID, then arithmetic mean across Sample_ID means",
            "tie_break": "score descending, normalized label display, label key, best evidence relative_path",
        },
        "denominators": {
            "development_query_spectra": len(query_rows),
            "development_query_sample_ids": len(sample_rows),
            "gallery_spectra": len(gallery.reference_records),
            "gallery_sample_ids": len({_known_sample_id(record).casefold() for record in gallery.reference_records}),
            "strict_closed_set_reference_labels": 60,
        },
        "metrics": metric_map,
        "metric_deltas": deltas,
        "prediction_changes": {
            "query_spectra_with_changed_top1_prediction": sum(bool(row["top1_prediction_changed"]) for row in query_rows),
            "sample_ids_with_any_changed_top1_prediction": len(changed_samples),
            "method_b_improved_to_top1_hit_spectra": sum(row["top1_outcome_change"] == "improved_to_hit" for row in query_rows),
            "method_b_lost_top1_hit_spectra": sum(row["top1_outcome_change"] == "lost_hit" for row in query_rows),
            "unchanged_top1_hit_spectra": sum(row["top1_outcome_change"] == "unchanged_hit" for row in query_rows),
            "unchanged_top1_miss_spectra": sum(row["top1_outcome_change"] == "unchanged_miss" for row in query_rows),
            "sample_id_changes": [
                {
                    "sample_id": row["query_sample_id"],
                    "reference_label": row["query_reference_label"],
                    "method_a_top1_hit_fraction": row["method_a_top1_hit_fraction"],
                    "method_b_top1_hit_fraction": row["method_b_top1_hit_fraction"],
                    "delta": row["method_b_minus_a_top1_hit_fraction"],
                    "method_a_predictions": row["method_a_top1_prediction_labels"],
                    "method_b_predictions": row["method_b_top1_prediction_labels"],
                }
                for row in changed_samples
            ],
        },
        "gallery_size_groups": size_rows,
        "safety_checks": {
            **preflight,
            "source_lock_validated": True,
            "method_a_reproduced_experiment_11_2": True,
            "same_similarity_vector_used_for_both_methods_per_query": True,
            "final_test_queries_executed": 0,
            "final_test_rows_selected_for_similarity": 0,
            "similarity_gallery_rows": len(gallery.reference_records),
            "similarity_development_query_rows": len(query_rows),
        },
        "cache_loading_limit": (
            "The experiment 9.3 NPZ is monolithic and the validated loader decompresses "
            "the full matrix for identity checks. Isolation is logical, not physical-file-level; "
            "only frozen Gallery and Development slices are passed to cosine similarity."
        ),
        "input_hashes": {
            "baseline_files": baseline_hashes,
            "data_source_lock": sha256_file(lock_path),
            "frozen_manifest": sha256_file(snapshot_dir / "evaluation_group_manifest.csv"),
            "cache": sha256_file(cache_path),
            "index": sha256_file(index_path),
        },
        "code_version": {**_git_version(), "comparison_script_sha256": sha256_file(Path(__file__))},
        "scientific_interpretation": (
            "Candidate reference labels are not independently validated ground truth. "
            "Cosine and aggregate scores are not probabilities or confidence, and Development "
            "changes are not independent mineral-identification accuracy changes."
        ),
    }
    report = build_report(summary)
    paths = write_outputs(output_dir, metric_rows, query_rows, sample_rows, confusion_rows, size_rows, summary, report)
    return summary, paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare frozen Development reference-label aggregation rules.")
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
        summary, paths = run_comparison(
            baseline_dir=args.baseline_dir,
            cache_path=args.cache,
            index_path=args.index,
            snapshot_dir=args.snapshot_dir,
            lock_path=args.lock,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    for method in (METHOD_A, METHOD_B):
        values = summary["metrics"][method]
        print(method + ": " + ", ".join(f"Top-{k}={values[f'sample_equal_top_{k}']:.6f}" for k in TOP_K_VALUES))
    print("Final-test similarity rows: 0")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
