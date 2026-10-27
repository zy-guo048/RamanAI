"""Build traceable method-A retrieval evidence for frozen Development queries."""

from __future__ import annotations

import argparse
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
SENSITIVITY_SCRIPT = PROJECT_ROOT / "scripts" / "11_analyze_gallery_identity_sensitivity.py"
DEFAULT_BASELINE_DIR = PROJECT_ROOT / "results" / "exp11_development_baseline"
DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "docs" / "evaluation" / "exp11_v1"
DEFAULT_LOCK = DEFAULT_SNAPSHOT_DIR / "DATA_SOURCE_LOCK.json"
DEFAULT_CACHE = PROJECT_ROOT / "results" / "exp09_batch_preprocessing_full" / "rruff_preprocessed_300_850_step0p5.npz"
DEFAULT_INDEX = PROJECT_ROOT / "results" / "exp09_batch_preprocessing_full" / "rruff_preprocessing_index.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "exp11_retrieval_evidence"
TOP_K_VALUES = (1, 3, 5)

QUERY_FIELDS = [
    "query_relative_path",
    "query_cache_row_index",
    "query_sample_id",
    "query_reference_label",
    "query_reference_label_group_key",
    "query_excitation_wavelength_nm",
    "query_excitation_wavelength_source",
    "query_metadata_status",
    "query_requires_review",
    "query_review_reasons",
    "top1_label",
    "top1_label_group_key",
    "top1_score",
    "top2_label",
    "top2_label_group_key",
    "top2_score",
    "top1_minus_top2_label_score",
    "top5_labels",
    "top5_scores",
    "true_reference_label_rank",
    "true_reference_label_score",
    "top1_hit",
    "top1_candidate_gallery_file_count",
    "top1_candidate_gallery_sample_id_count",
    "top1_best_support_sample_id_score",
    "top1_second_support_sample_id_score",
    "top1_best_minus_second_support_sample_id_score",
]

CANDIDATE_FIELDS = [
    "query_relative_path",
    "query_sample_id",
    "query_reference_label",
    "candidate_rank",
    "candidate_reference_label",
    "candidate_reference_label_group_key",
    "candidate_label_score",
    "candidate_is_query_reference_label",
    "best_support_relative_path",
    "best_support_sample_id",
    "best_support_excitation_wavelength_nm",
    "best_support_cosine_similarity",
    "gallery_reference_file_count",
    "gallery_distinct_sample_id_count",
    "best_independent_sample_id",
    "best_independent_sample_id_score",
    "second_independent_sample_id",
    "second_independent_sample_id_score",
    "best_minus_second_independent_sample_id_score",
]

SUPPORT_FIELDS = [
    "query_relative_path",
    "query_sample_id",
    "query_reference_label",
    "candidate_rank",
    "candidate_reference_label",
    "candidate_reference_label_group_key",
    "support_sample_id_rank",
    "support_sample_id",
    "support_sample_id_score",
    "support_best_relative_path",
    "support_best_excitation_wavelength_nm",
    "support_file_count",
]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sensitivity = _load_module("raman_exp11_sensitivity_for_evidence", SENSITIVITY_SCRIPT)
aggregation = sensitivity.aggregation
diagnostics = sensitivity.diagnostics
baseline = sensitivity.baseline
exp10 = sensitivity.exp10
exp11 = sensitivity.exp11


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_sample_id_support(
    similarities: np.ndarray,
    records: tuple[dict[str, str], ...],
    sample_indices: dict[str, list[int]],
) -> list[dict[str, object]]:
    """Rank independent Sample_ID evidence by each identity's best spectrum."""
    values = exp10.correct_cosine_boundaries(np.asarray(similarities, dtype=np.float64))
    if len(values) != len(records):
        raise ValueError("Similarity vector length does not match Gallery records")
    output: list[dict[str, object]] = []
    for sample_key, indices in sample_indices.items():
        if not indices:
            raise ValueError("Candidate Sample_ID evidence group is empty")
        sample_ids = {aggregation._known_sample_id(records[index]) for index in indices}
        if len(sample_ids) != 1:
            raise ValueError("One Sample_ID evidence group contains conflicting identities")
        best_index = min(
            indices,
            key=lambda index: (
                -float(values[index]),
                records[index]["relative_path"].casefold(),
                records[index]["relative_path"],
            ),
        )
        best_record = records[best_index]
        output.append(
            {
                "sample_key": sample_key,
                "sample_id": next(iter(sample_ids)),
                "score": float(values[best_index]),
                "best_record": best_record,
                "file_count": len(indices),
            }
        )
    output.sort(
        key=lambda row: (
            -float(row["score"]),
            str(row["sample_id"]).casefold(),
            str(row["sample_id"]),
            str(row["best_record"]["relative_path"]).casefold(),
            str(row["best_record"]["relative_path"]),
        )
    )
    for rank, row in enumerate(output, start=1):
        row["rank"] = rank
    return output


def build_query_evidence(
    similarity_matrix: np.ndarray,
    gallery_records: tuple[dict[str, str], ...],
    development_manifest: list[dict[str, str]],
    baseline_record_rows: list[dict[str, str]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Generate query, Top-5 label, and Top-3 independent identity evidence."""
    matrix = np.asarray(similarity_matrix, dtype=np.float64)
    if matrix.shape != (len(development_manifest), len(gallery_records)):
        raise ValueError("Evidence similarity matrix does not match frozen data")
    existing = {row["query_relative_path"]: row for row in baseline_record_rows}
    if len(existing) != len(baseline_record_rows):
        raise ValueError("Baseline query results contain duplicate relative_path")
    if set(existing) != {row["relative_path"] for row in development_manifest}:
        raise ValueError("Baseline and frozen Development query identities differ")
    label_groups = aggregation.validate_gallery_for_aggregation(gallery_records)
    if len(label_groups) != 60:
        raise ValueError(f"Expected 60 frozen labels, got {len(label_groups)}")
    group_by_key = {str(group["key"]): group for group in label_groups}
    query_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    baseline_results: list[dict[str, object]] = []
    for query_index, (values, manifest) in enumerate(
        zip(matrix, development_manifest, strict=True)
    ):
        path = manifest["relative_path"]
        baseline_row = existing[path]
        for baseline_field, manifest_field in (
            ("query_cache_row_index", "cache_row_index"),
            ("query_sample_id", "sample_id"),
            ("query_reference_label_group_key", "reference_label_group_key"),
        ):
            if baseline_row[baseline_field] != manifest[manifest_field]:
                raise ValueError(
                    f"Baseline/frozen identity mismatch for {path}: {baseline_field}"
                )
        ranking = aggregation.rank_reference_labels(
            values, gallery_records, label_groups, aggregation.METHOD_A
        )
        target_key = manifest["reference_label_group_key"]
        diagnostic = aggregation._ranking_diagnostic(ranking, target_key)
        aggregation._validate_method_a_against_baseline(baseline_row, diagnostic)
        baseline_results.append(diagnostic)
        top_five = ranking[:5]
        target = next(
            row for row in ranking if row["reference_label_group_key"] == target_key
        )
        support_by_candidate: dict[str, list[dict[str, object]]] = {}
        for candidate in top_five:
            candidate_key = str(candidate["reference_label_group_key"])
            group = group_by_key[candidate_key]
            support = rank_sample_id_support(
                values, gallery_records, group["sample_indices"]
            )
            support_by_candidate[candidate_key] = support
            if len(support) < 2:
                second_id = ""
                second_score: float | str = ""
                identity_gap: float | str = ""
            else:
                second_id = support[1]["sample_id"]
                second_score = float(support[1]["score"])
                identity_gap = float(support[0]["score"]) - float(support[1]["score"])
            best_record = candidate["record"]
            candidate_rows.append(
                {
                    "query_relative_path": path,
                    "query_sample_id": manifest["sample_id"],
                    "query_reference_label": manifest["reference_label_group_display"],
                    "candidate_rank": candidate["rank"],
                    "candidate_reference_label": candidate[
                        "reference_label_group_display"
                    ],
                    "candidate_reference_label_group_key": candidate_key,
                    "candidate_label_score": candidate["score"],
                    "candidate_is_query_reference_label": candidate_key == target_key,
                    "best_support_relative_path": best_record["relative_path"],
                    "best_support_sample_id": best_record["sample_id"],
                    "best_support_excitation_wavelength_nm": diagnostics.normalize_excitation_wavelength(
                        best_record.get("excitation_wavelength_nm")
                    ),
                    "best_support_cosine_similarity": candidate["score"],
                    "gallery_reference_file_count": candidate[
                        "label_reference_file_count"
                    ],
                    "gallery_distinct_sample_id_count": candidate[
                        "label_sample_id_count"
                    ],
                    "best_independent_sample_id": support[0]["sample_id"],
                    "best_independent_sample_id_score": support[0]["score"],
                    "second_independent_sample_id": second_id,
                    "second_independent_sample_id_score": second_score,
                    "best_minus_second_independent_sample_id_score": identity_gap,
                }
            )
            for item in support[:3]:
                evidence_record = item["best_record"]
                support_rows.append(
                    {
                        "query_relative_path": path,
                        "query_sample_id": manifest["sample_id"],
                        "query_reference_label": manifest[
                            "reference_label_group_display"
                        ],
                        "candidate_rank": candidate["rank"],
                        "candidate_reference_label": candidate[
                            "reference_label_group_display"
                        ],
                        "candidate_reference_label_group_key": candidate_key,
                        "support_sample_id_rank": item["rank"],
                        "support_sample_id": item["sample_id"],
                        "support_sample_id_score": item["score"],
                        "support_best_relative_path": evidence_record[
                            "relative_path"
                        ],
                        "support_best_excitation_wavelength_nm": diagnostics.normalize_excitation_wavelength(
                            evidence_record.get("excitation_wavelength_nm")
                        ),
                        "support_file_count": item["file_count"],
                    }
                )
        top1_support = support_by_candidate[
            str(top_five[0]["reference_label_group_key"])
        ]
        top1_second_score = (
            float(top1_support[1]["score"]) if len(top1_support) >= 2 else ""
        )
        top1_identity_gap = (
            float(top1_support[0]["score"]) - float(top1_support[1]["score"])
            if len(top1_support) >= 2
            else ""
        )
        query_rows.append(
            {
                "query_relative_path": path,
                "query_cache_row_index": manifest["cache_row_index"],
                "query_sample_id": manifest["sample_id"],
                "query_reference_label": manifest[
                    "reference_label_group_display"
                ],
                "query_reference_label_group_key": target_key,
                "query_excitation_wavelength_nm": diagnostics.normalize_excitation_wavelength(
                    manifest.get("excitation_wavelength_nm")
                ),
                "query_excitation_wavelength_source": manifest.get(
                    "excitation_wavelength_source", ""
                ),
                "query_metadata_status": manifest.get("metadata_status", ""),
                "query_requires_review": manifest.get(
                    "metadata_requires_review", ""
                ),
                "query_review_reasons": manifest.get("review_reasons", ""),
                "top1_label": top_five[0]["reference_label_group_display"],
                "top1_label_group_key": top_five[0]["reference_label_group_key"],
                "top1_score": top_five[0]["score"],
                "top2_label": top_five[1]["reference_label_group_display"],
                "top2_label_group_key": top_five[1]["reference_label_group_key"],
                "top2_score": top_five[1]["score"],
                "top1_minus_top2_label_score": float(top_five[0]["score"])
                - float(top_five[1]["score"]),
                "top5_labels": " | ".join(
                    str(row["reference_label_group_display"]) for row in top_five
                ),
                "top5_scores": " | ".join(
                    f"{float(row['score']):.17g}" for row in top_five
                ),
                "true_reference_label_rank": target["rank"],
                "true_reference_label_score": target["score"],
                "top1_hit": int(int(target["rank"]) == 1),
                "top1_candidate_gallery_file_count": top_five[0][
                    "label_reference_file_count"
                ],
                "top1_candidate_gallery_sample_id_count": top_five[0][
                    "label_sample_id_count"
                ],
                "top1_best_support_sample_id_score": top1_support[0]["score"],
                "top1_second_support_sample_id_score": top1_second_score,
                "top1_best_minus_second_support_sample_id_score": top1_identity_gap,
            }
        )
    return query_rows, candidate_rows, support_rows, baseline_results


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "minimum": None, "p25": None, "median": None, "p75": None, "maximum": None}
    if not np.all(np.isfinite(array)):
        raise ValueError("Evidence score-gap distribution contains non-finite values")
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "maximum": float(np.max(array)),
    }


def _git_version() -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=PROJECT_ROOT, check=True,
                capture_output=True, text=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    return {"head": run("rev-parse", "HEAD"), "describe": run("describe", "--always", "--dirty")}


def build_report(summary: dict[str, object]) -> str:
    metrics = summary["baseline_metrics"]
    label_gap = summary["top1_minus_top2_label_score_distribution"]
    identity_gap = summary[
        "top1_best_minus_second_support_sample_id_score_distribution"
    ]
    return f"""# RamanAI 实验 11.4-A：Development 检索证据审计

## 范围与基线

本报告覆盖冻结的118条 Development 查询、60个查询 Sample_ID、724条 Gallery 光谱、
342个 Gallery Sample_ID 和60个候选参考标签。方法A的 Sample_ID 等权结果为 Top-1
{metrics['sample_equal_top_1']:.4%}、Top-3 {metrics['sample_equal_top_3']:.4%}、
Top-5 {metrics['sample_equal_top_5']:.4%}，与实验11.2一致。

每条查询均计算完整60标签排名；Top-5每个标签记录最佳支持文件，并按 Gallery Sample_ID
将同一身份的多条测量压缩为该身份的最高余弦分数，再列出最高3个不同身份。

## 分数差描述

- 第一名减第二名标签分数：最小 {label_gap['minimum']:.6f}，中位
  {label_gap['median']:.6f}，最大 {label_gap['maximum']:.6f}。
- Top-1候选最佳减第二佳独立 Sample_ID 分数：最小
  {identity_gap['minimum']:.6f}，中位 {identity_gap['median']:.6f}，最大
  {identity_gap['maximum']:.6f}。
- Top-1候选仅有一个有效支持身份的查询比例：
  {summary['top1_candidate_single_support_identity_fraction']:.4%}。

这些差值只是排序证据，不是概率、置信度或自动质量阈值。

## 数据隔离

实验9.3 NPZ是单体压缩缓存，现有验证加载器会解压完整缓存，所以隔离是逻辑性的，
不是物理文件级隔离。实际余弦矩阵严格为118×724，仅将 Development 与 Gallery 切片
传入相似度计算；未运行 final-test，也未打开 final-test 检索结果文件。

候选参考标签不是独立验证的矿物真值。本报告不自动剔除参考身份、不修改检索规则，
也不设置未经验证的分数阈值。
"""


def output_paths(output_dir: Path) -> tuple[Path, ...]:
    return (
        output_dir / "development_query_evidence.csv",
        output_dir / "candidate_label_evidence.csv",
        output_dir / "supporting_sample_id_evidence.csv",
        output_dir / "retrieval_evidence_summary.json",
        output_dir / "retrieval_evidence_report.md",
    )


def write_outputs(
    output_dir: Path,
    query_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    support_rows: list[dict[str, object]],
    summary: dict[str, object],
    report: str,
) -> tuple[Path, ...]:
    paths = output_paths(output_dir)
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Retrieval evidence output already exists: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, fields, rows in (
        (paths[0], QUERY_FIELDS, query_rows),
        (paths[1], CANDIDATE_FIELDS, candidate_rows),
        (paths[2], SUPPORT_FIELDS, support_rows),
    ):
        with path.open("x", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with paths[3].open("x", encoding="utf-8-sig", newline="") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    paths[4].write_text(report, encoding="utf-8-sig", newline="\n")
    return paths


def run_evidence_audit(
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
    manifest_rows, preflight = baseline.validate_frozen_manifest(
        snapshot_dir / "evaluation_group_manifest.csv", lock
    )
    gallery, development_manifest, development_matrix = baseline.prepare_frozen_data(
        cache_path, index_path, manifest_rows
    )
    if len(gallery.reference_records) != 724 or len(development_manifest) != 118:
        raise ValueError("Frozen Gallery or Development record count differs from protocol")
    similarity_matrix = sensitivity.compute_similarity_matrix(
        gallery.reference_matrix, development_matrix
    )
    if similarity_matrix.shape != (118, 724):
        raise ValueError("Evidence similarity matrix must be exactly 118 x 724")
    query_rows, candidate_rows, support_rows, baseline_results = build_query_evidence(
        similarity_matrix,
        gallery.reference_records,
        development_manifest,
        baseline_records,
    )
    query_sample_ids = [row["sample_id"] for row in development_manifest]
    metrics = sensitivity.aggregate_metrics(baseline_results, query_sample_ids)
    sensitivity._validate_full_baseline(
        baseline_results, metrics, baseline_records, baseline_summary
    )
    single_support_count = sum(
        int(row["top1_candidate_gallery_sample_id_count"]) == 1
        for row in query_rows
    )
    identity_gaps = [
        float(row["top1_best_minus_second_support_sample_id_score"])
        for row in query_rows
        if row["top1_best_minus_second_support_sample_id_score"] != ""
    ]
    summary: dict[str, object] = {
        "experiment": "11.4-A",
        "scope": "frozen_development_method_a_traceable_evidence",
        "denominators": {
            "development_query_spectra": len(query_rows),
            "development_query_sample_ids": len(set(query_sample_ids)),
            "gallery_spectra": len(gallery.reference_records),
            "gallery_sample_ids": len({aggregation._known_sample_id(row).casefold() for row in gallery.reference_records}),
            "reference_labels": 60,
            "candidate_label_evidence_rows": len(candidate_rows),
            "supporting_sample_id_evidence_rows": len(support_rows),
        },
        "baseline_metrics": metrics,
        "top1_minus_top2_label_score_distribution": _distribution(
            float(row["top1_minus_top2_label_score"]) for row in query_rows
        ),
        "top1_candidate_single_support_identity_count": single_support_count,
        "top1_candidate_single_support_identity_fraction": single_support_count / len(query_rows),
        "top1_best_minus_second_support_sample_id_score_distribution": _distribution(identity_gaps),
        "evidence_rules": {
            "label_score": "maximum cosine similarity among all valid Gallery spectra in label",
            "support_sample_id_score": "maximum cosine similarity among Gallery spectra sharing that Sample_ID",
            "support_sample_id_limit_per_top5_label": 3,
            "tie_break": "score descending, stable normalized label/Sample_ID/path ordering",
            "score_gap_interpretation": "descriptive ranking information, not probability or confidence",
        },
        "safety_checks": {
            **preflight,
            "source_lock_validated": True,
            "method_a_reproduced_experiment_11_2": True,
            "similarity_matrix_shape": list(similarity_matrix.shape),
            "similarity_matrix_contains_only_gallery_and_development": True,
            "all_queries_have_60_label_ranking": len(query_rows) == 118,
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
        "code_version": {
            **_git_version(),
            "evidence_script_sha256": sha256_file(Path(__file__)),
        },
        "scientific_interpretation": (
            "Candidate reference labels are not independently validated ground truth. "
            "Cosine similarities and score gaps are not probabilities or confidence."
        ),
    }
    report = build_report(summary)
    paths = write_outputs(
        output_dir, query_rows, candidate_rows, support_rows, summary, report
    )
    return summary, paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit traceable evidence for frozen Development method-A retrieval."
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
    args = parser.parse_args()
    try:
        summary, paths = run_evidence_audit(
            baseline_dir=args.baseline_dir,
            cache_path=args.cache,
            index_path=args.index,
            snapshot_dir=args.snapshot_dir,
            lock_path=args.lock,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        "Evidence rows: "
        f"queries={summary['denominators']['development_query_spectra']}, "
        f"candidate_labels={summary['denominators']['candidate_label_evidence_rows']}, "
        f"supporting_sample_ids={summary['denominators']['supporting_sample_id_evidence_rows']}"
    )
    for k in TOP_K_VALUES:
        print(
            f"Sample_ID Top-{k}: "
            f"{summary['baseline_metrics'][f'sample_equal_top_{k}']:.6f}"
        )
    print("Final-test similarity rows: 0")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
