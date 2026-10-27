"""Build a leakage-audited Sample_ID-level Raman retrieval evaluation protocol."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP10_SCRIPT = PROJECT_ROOT / "scripts" / "10_search_rruff_topk.py"
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "exp11_evaluation_protocol"
DEFAULT_SEED = 42
DEFAULT_MIN_SAMPLE_IDS_PER_LABEL = 5

GROUP_GALLERY = "reference_gallery"
GROUP_DEVELOPMENT = "development_query"
GROUP_FINAL = "final_test_query"
GROUP_SPARSE = "sparse_label_not_in_strict_closed_set"
GROUP_CONFLICT = "sample_group_conflict_exploratory_only"
GROUP_INELIGIBLE = "ineligible_exploratory_only"

MANIFEST_FIELDS = [
    "relative_path",
    "file_name",
    "cache_row_index",
    "measurement_stem",
    "sample_id",
    "sample_id_source",
    "filename_sample_id",
    "header_rruff_id",
    "reference_label",
    "reference_label_status",
    "reference_label_group_display",
    "reference_label_group_key",
    "reference_label_grouping_basis",
    "excitation_wavelength_nm",
    "excitation_wavelength_source",
    "requires_review",
    "review_reasons",
    "audit_metadata_warnings",
    "has_missing_metadata",
    "has_conflict",
    "metadata_status",
    "metadata_flags",
    "known_sample_id_usable",
    "sample_id_metadata_conflict",
    "reference_label_usable",
    "metadata_requires_review",
    "record_level_eligible",
    "record_level_ineligibility_reasons",
    "sample_group_record_count",
    "sample_group_reference_label_count",
    "sample_group_eligible",
    "sample_group_ineligibility_reasons",
    "strict_label_eligible_sample_id_count",
    "strict_closed_set_eligible",
    "formal_ineligibility_reasons",
    "evaluation_group",
    "split_seed",
    "split_algorithm",
    "reference_label_scientific_status",
]

DISTRIBUTION_FIELDS = [
    "row_type",
    "identity_key",
    "display_value",
    "reference_label_grouping_basis",
    "record_count",
    "known_sample_id_count",
    "eligible_sample_id_count",
    "sample_id_count_bucket",
    "reference_label_count",
    "reference_labels",
    "multiple_reference_labels",
    "strict_closed_set_label",
    "gallery_sample_id_count",
    "development_sample_id_count",
    "final_test_sample_id_count",
]


def _load_exp10_module():
    spec = importlib.util.spec_from_file_location(
        "raman_exp10_topk",
        EXP10_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load experiment 10 retrieval module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp10 = _load_exp10_module()


def _parse_bool(value: str | None, field_name: str) -> bool:
    normalized = (value or "").strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid boolean value for {field_name}: {value!r}")


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _sample_id_conflict(record: dict[str, str]) -> bool:
    source = record.get("sample_id_source", "").strip().casefold()
    return "conflict" in source


def _assessment_for_record(record: dict[str, str]) -> dict[str, object]:
    known_sample = exp10._known_sample_id(record)
    sample_conflict = _sample_id_conflict(record)
    label_display, label_basis, label_key = exp10._reference_label_identity(record)
    label_usable = label_basis == "safe_unicode_whitespace_case_normalization"
    requires_review = _parse_bool(record.get("requires_review"), "requires_review")
    has_missing = _parse_bool(
        record.get("has_missing_metadata"), "has_missing_metadata"
    )
    has_conflict = _parse_bool(record.get("has_conflict"), "has_conflict")
    reasons: list[str] = []
    if known_sample is None:
        reasons.append("sample_id_unknown_missing_or_conflicted")
    if sample_conflict:
        reasons.append("sample_id_metadata_conflict")
    if not label_usable:
        reasons.append("reference_label_missing_or_unusable")
    if requires_review:
        reasons.append("metadata_requires_review")
    if has_missing:
        reasons.append("metadata_missing_fields")
    if has_conflict:
        reasons.append("metadata_conflict_present")
    return {
        "record": record,
        "known_sample_id": known_sample,
        "sample_id_metadata_conflict": sample_conflict,
        "reference_label_group_display": label_display,
        "reference_label_group_key": label_key,
        "reference_label_grouping_basis": label_basis,
        "reference_label_usable": label_usable,
        "metadata_requires_review": requires_review,
        "record_level_eligible": not reasons,
        "record_level_reasons": reasons,
    }


def _bucket_sample_count(count: int) -> str:
    if count >= 5:
        return "5_plus"
    return str(count)


def build_evaluation_plan(
    reference_records: Iterable[dict[str, str]],
    *,
    seed: int = DEFAULT_SEED,
    min_sample_ids_per_label: int = DEFAULT_MIN_SAMPLE_IDS_PER_LABEL,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Assess eligibility and assign whole Sample_ID groups deterministically."""
    if min_sample_ids_per_label < 5:
        raise ValueError(
            "min_sample_ids_per_label must be at least 5 to support 3/1/1 splitting"
        )
    records = [dict(record) for record in reference_records]
    paths = [record.get("relative_path", "") for record in records]
    if any(not path for path in paths):
        raise ValueError("Reference records contain an empty relative_path")
    duplicate_paths = [path for path, count in Counter(paths).items() if count > 1]
    if duplicate_paths:
        raise ValueError("Reference records contain duplicate relative_path values")

    assessments = [_assessment_for_record(record) for record in records]
    sample_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for assessment in assessments:
        known_sample = assessment["known_sample_id"]
        if known_sample is not None:
            sample_groups[str(known_sample).casefold()].append(assessment)

    sample_group_info: dict[str, dict[str, object]] = {}
    qualified_samples_by_label: dict[str, list[str]] = defaultdict(list)
    label_display_by_key: dict[str, str] = {}
    for normalized_sample_id, group in sample_groups.items():
        label_keys = {
            str(assessment["reference_label_group_key"])
            for assessment in group
            if bool(assessment["reference_label_usable"])
        }
        label_displays = sorted(
            {
                str(assessment["reference_label_group_display"])
                for assessment in group
                if bool(assessment["reference_label_usable"])
            },
            key=str.casefold,
        )
        reasons: list[str] = []
        if any(not bool(assessment["record_level_eligible"]) for assessment in group):
            reasons.append("sample_group_contains_record_level_ineligibility")
        if len(label_keys) == 0:
            reasons.append("sample_group_has_no_usable_reference_label")
        elif len(label_keys) > 1:
            reasons.append("sample_id_associated_with_multiple_reference_labels")
        eligible = not reasons
        label_key = next(iter(label_keys)) if len(label_keys) == 1 else ""
        label_display = label_displays[0] if len(label_displays) == 1 else ""
        canonical_sample_id = str(group[0]["known_sample_id"])
        sample_group_info[normalized_sample_id] = {
            "sample_id": canonical_sample_id,
            "record_count": len(group),
            "label_keys": sorted(label_keys),
            "label_displays": label_displays,
            "label_count": len(label_keys),
            "eligible": eligible,
            "reasons": reasons,
            "label_key": label_key,
            "label_display": label_display,
        }
        if eligible:
            qualified_samples_by_label[label_key].append(normalized_sample_id)
            label_display_by_key[label_key] = label_display

    strict_label_keys = {
        label_key
        for label_key, sample_ids in qualified_samples_by_label.items()
        if len(sample_ids) >= min_sample_ids_per_label
    }
    rng = random.Random(seed)
    sample_assignment: dict[str, str] = {}
    for label_key in sorted(strict_label_keys):
        sample_ids = sorted(
            qualified_samples_by_label[label_key],
            key=lambda sample_key: (
                str(sample_group_info[sample_key]["sample_id"]).casefold(),
                str(sample_group_info[sample_key]["sample_id"]),
            ),
        )
        rng.shuffle(sample_ids)
        sample_assignment[sample_ids[0]] = GROUP_DEVELOPMENT
        sample_assignment[sample_ids[1]] = GROUP_FINAL
        for sample_id in sample_ids[2:]:
            sample_assignment[sample_id] = GROUP_GALLERY

    manifest_records: list[dict[str, object]] = []
    for assessment in assessments:
        source = assessment["record"]
        assert isinstance(source, dict)
        known_sample = assessment["known_sample_id"]
        group_info: dict[str, object] | None = None
        if known_sample is not None:
            group_info = sample_group_info[str(known_sample).casefold()]
        sample_group_eligible = bool(group_info and group_info["eligible"])
        strict_label_count = 0
        if group_info and group_info["label_key"]:
            strict_label_count = len(
                qualified_samples_by_label[str(group_info["label_key"])]
            )
        strict_eligible = bool(
            sample_group_eligible
            and str(group_info["label_key"]) in strict_label_keys
        )
        formal_reasons = list(assessment["record_level_reasons"])
        if group_info:
            formal_reasons.extend(str(reason) for reason in group_info["reasons"])
        if sample_group_eligible and not strict_eligible:
            formal_reasons.append(
                f"label_has_fewer_than_{min_sample_ids_per_label}_eligible_sample_ids"
            )
        formal_reasons = _ordered_unique(formal_reasons)

        if strict_eligible:
            evaluation_group = sample_assignment[str(known_sample).casefold()]
        elif group_info and not bool(group_info["eligible"]):
            evaluation_group = GROUP_CONFLICT
        elif sample_group_eligible:
            evaluation_group = GROUP_SPARSE
        else:
            evaluation_group = GROUP_INELIGIBLE

        manifest_records.append(
            {
                "relative_path": source.get("relative_path", ""),
                "file_name": source.get("file_name", ""),
                "cache_row_index": source.get("cache_row_index", ""),
                "measurement_stem": source.get("measurement_stem", ""),
                "sample_id": source.get("sample_id", ""),
                "sample_id_source": source.get("sample_id_source", ""),
                "filename_sample_id": source.get("filename_sample_id", ""),
                "header_rruff_id": source.get("header_rruff_id", ""),
                "reference_label": source.get("reference_label", ""),
                "reference_label_status": source.get(
                    "reference_label_status", ""
                ),
                "reference_label_group_display": assessment[
                    "reference_label_group_display"
                ],
                "reference_label_group_key": assessment[
                    "reference_label_group_key"
                ],
                "reference_label_grouping_basis": assessment[
                    "reference_label_grouping_basis"
                ],
                "excitation_wavelength_nm": source.get(
                    "excitation_wavelength_nm", ""
                ),
                "excitation_wavelength_source": source.get(
                    "excitation_wavelength_source", ""
                ),
                "requires_review": source.get("requires_review", ""),
                "review_reasons": source.get("review_reasons", ""),
                "audit_metadata_warnings": source.get(
                    "audit_metadata_warnings", ""
                ),
                "has_missing_metadata": source.get("has_missing_metadata", ""),
                "has_conflict": source.get("has_conflict", ""),
                "metadata_status": source.get("metadata_status", ""),
                "metadata_flags": source.get("metadata_flags", ""),
                "known_sample_id_usable": known_sample is not None,
                "sample_id_metadata_conflict": assessment[
                    "sample_id_metadata_conflict"
                ],
                "reference_label_usable": assessment["reference_label_usable"],
                "metadata_requires_review": assessment[
                    "metadata_requires_review"
                ],
                "record_level_eligible": assessment["record_level_eligible"],
                "record_level_ineligibility_reasons": ";".join(
                    str(reason) for reason in assessment["record_level_reasons"]
                ),
                "sample_group_record_count": (
                    int(group_info["record_count"]) if group_info else 1
                ),
                "sample_group_reference_label_count": (
                    int(group_info["label_count"]) if group_info else 0
                ),
                "sample_group_eligible": sample_group_eligible,
                "sample_group_ineligibility_reasons": (
                    ";".join(str(reason) for reason in group_info["reasons"])
                    if group_info
                    else "sample_id_not_usable_for_grouping"
                ),
                "strict_label_eligible_sample_id_count": strict_label_count,
                "strict_closed_set_eligible": strict_eligible,
                "formal_ineligibility_reasons": ";".join(formal_reasons),
                "evaluation_group": evaluation_group,
                "split_seed": seed,
                "split_algorithm": (
                    "sorted_labels_and_sample_ids_then_global_seeded_shuffle_"
                    "one_development_one_final_remainder_gallery"
                ),
                "reference_label_scientific_status": (
                    "candidate_database_label_not_independently_validated_ground_truth"
                ),
            }
        )

    label_records: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in manifest_records:
        label_records[str(record["reference_label_group_key"])].append(record)
    distribution_rows: list[dict[str, object]] = []
    for label_key in sorted(label_records):
        group = label_records[label_key]
        known_ids = {
            str(record["sample_id"]).casefold()
            for record in group
            if bool(record["known_sample_id_usable"])
        }
        eligible_ids = {
            str(record["sample_id"]).casefold()
            for record in group
            if bool(record["sample_group_eligible"])
        }
        group_counts = Counter(str(record["evaluation_group"]) for record in group)
        distribution_rows.append(
            {
                "row_type": "reference_label",
                "identity_key": label_key,
                "display_value": group[0]["reference_label_group_display"],
                "reference_label_grouping_basis": group[0][
                    "reference_label_grouping_basis"
                ],
                "record_count": len(group),
                "known_sample_id_count": len(known_ids),
                "eligible_sample_id_count": len(eligible_ids),
                "sample_id_count_bucket": _bucket_sample_count(len(known_ids)),
                "reference_label_count": 1,
                "reference_labels": group[0]["reference_label_group_display"],
                "multiple_reference_labels": False,
                "strict_closed_set_label": label_key in strict_label_keys,
                "gallery_sample_id_count": len(
                    {
                        str(record["sample_id"]).casefold()
                        for record in group
                        if record["evaluation_group"] == GROUP_GALLERY
                    }
                ),
                "development_sample_id_count": len(
                    {
                        str(record["sample_id"]).casefold()
                        for record in group
                        if record["evaluation_group"] == GROUP_DEVELOPMENT
                    }
                ),
                "final_test_sample_id_count": len(
                    {
                        str(record["sample_id"]).casefold()
                        for record in group
                        if record["evaluation_group"] == GROUP_FINAL
                    }
                ),
            }
        )
    for sample_key in sorted(sample_group_info):
        info = sample_group_info[sample_key]
        assignment = sample_assignment.get(
            sample_key,
            GROUP_SPARSE if info["eligible"] else GROUP_CONFLICT,
        )
        distribution_rows.append(
            {
                "row_type": "sample_id",
                "identity_key": "sample_id:" + sample_key,
                "display_value": info["sample_id"],
                "reference_label_grouping_basis": "",
                "record_count": info["record_count"],
                "known_sample_id_count": 1,
                "eligible_sample_id_count": int(bool(info["eligible"])),
                "sample_id_count_bucket": "",
                "reference_label_count": info["label_count"],
                "reference_labels": ";".join(info["label_displays"]),
                "multiple_reference_labels": int(info["label_count"]) > 1,
                "strict_closed_set_label": (
                    bool(info["eligible"])
                    and str(info["label_key"]) in strict_label_keys
                ),
                "gallery_sample_id_count": int(assignment == GROUP_GALLERY),
                "development_sample_id_count": int(
                    assignment == GROUP_DEVELOPMENT
                ),
                "final_test_sample_id_count": int(assignment == GROUP_FINAL),
            }
        )

    plan_context = {
        "sample_group_info": sample_group_info,
        "qualified_samples_by_label": {
            key: sorted(values) for key, values in qualified_samples_by_label.items()
        },
        "strict_label_keys": sorted(strict_label_keys),
        "label_display_by_key": label_display_by_key,
        "sample_assignment": sample_assignment,
        "min_sample_ids_per_label": min_sample_ids_per_label,
        "seed": seed,
    }
    return manifest_records, distribution_rows, plan_context


def audit_leakage(
    manifest_records: list[dict[str, object]],
) -> dict[str, object]:
    """Check identity, file, measurement, and source-role separation."""
    strict_groups = {GROUP_GALLERY, GROUP_DEVELOPMENT, GROUP_FINAL}
    sample_to_groups: dict[str, set[str]] = defaultdict(set)
    path_to_groups: dict[str, set[str]] = defaultdict(set)
    stem_to_groups: dict[str, set[str]] = defaultdict(set)
    invalid_identity_in_strict: list[str] = []
    processed_name_records: list[str] = []
    for record in manifest_records:
        group = str(record["evaluation_group"])
        sample_id = str(record["sample_id"])
        if bool(record["known_sample_id_usable"]):
            sample_to_groups[sample_id.casefold()].add(group)
        path = str(record["relative_path"])
        path_to_groups[path].add(group)
        stem = str(record["measurement_stem"])
        stem_to_groups[stem.casefold()].add(group)
        if group in strict_groups and (
            not bool(record["known_sample_id_usable"])
            or bool(record["sample_id_metadata_conflict"])
            or not bool(record["strict_closed_set_eligible"])
        ):
            invalid_identity_in_strict.append(path)
        if "raman_data_processed" in path.casefold():
            processed_name_records.append(path)

    sample_leaks = sorted(
        sample_id for sample_id, groups in sample_to_groups.items() if len(groups) > 1
    )
    path_leaks = sorted(
        path for path, groups in path_to_groups.items() if len(groups) > 1
    )
    stem_leaks = sorted(
        stem for stem, groups in stem_to_groups.items() if len(groups) > 1
    )
    strict_records = [
        record
        for record in manifest_records
        if record["evaluation_group"] in strict_groups
    ]
    gallery_labels = {
        str(record["reference_label_group_key"])
        for record in strict_records
        if record["evaluation_group"] == GROUP_GALLERY
    }
    query_labels = {
        str(record["reference_label_group_key"])
        for record in strict_records
        if record["evaluation_group"] in {GROUP_DEVELOPMENT, GROUP_FINAL}
    }
    labels_missing_gallery = sorted(query_labels - gallery_labels)
    passed = not (
        sample_leaks
        or path_leaks
        or stem_leaks
        or invalid_identity_in_strict
        or processed_name_records
        or labels_missing_gallery
    )
    return {
        "passed": passed,
        "same_known_sample_id_cross_group_count": len(sample_leaks),
        "same_known_sample_id_cross_group_examples": sample_leaks[:10],
        "relative_path_cross_group_count": len(path_leaks),
        "relative_path_cross_group_examples": path_leaks[:10],
        "measurement_stem_cross_group_count": len(stem_leaks),
        "measurement_stem_cross_group_examples": stem_leaks[:10],
        "invalid_or_conflicted_identity_in_strict_group_count": len(
            invalid_identity_in_strict
        ),
        "invalid_or_conflicted_identity_examples": invalid_identity_in_strict[:10],
        "processed_filename_in_reference_cache_count": len(processed_name_records),
        "processed_filename_examples": processed_name_records[:10],
        "query_label_missing_from_gallery_count": len(labels_missing_gallery),
        "query_label_missing_from_gallery_examples": labels_missing_gallery[:10],
        "similarity_diagnostic_note": (
            "No spectral-similarity-based identity merging was performed. Similar "
            "spectra may be inspected later only as exploratory diagnostics."
        ),
        "raw_processed_note": (
            "The source is experiment 9.3's admitted RAW-only cache; Processed "
            "derivatives are not independent records in this protocol."
        ),
    }


def require_leakage_audit_passed(leakage: dict[str, object]) -> None:
    """Refuse to publish a formal grouping after any leakage audit failure."""
    if leakage.get("passed") is not True:
        raise ValueError(
            "Leakage audit failed; formal evaluation grouping outputs will not "
            "be published"
        )


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_hash(
    manifest_records: list[dict[str, object]],
    groups: set[str] | None = None,
) -> str:
    lines = []
    for record in manifest_records:
        if groups is not None and str(record["evaluation_group"]) not in groups:
            continue
        lines.append(
            "|".join(
                (
                    str(record["sample_id"]),
                    str(record["evaluation_group"]),
                    str(record["relative_path"]),
                    str(record["reference_label_group_key"]),
                )
            )
        )
    payload = "\n".join(sorted(lines)).encode("utf-8")
    return _sha256_bytes(payload)


def build_audit_summary(
    cache_path: Path,
    index_path: Path,
    manifest_records: list[dict[str, object]],
    distribution_rows: list[dict[str, object]],
    plan_context: dict[str, object],
    leakage: dict[str, object],
) -> dict[str, object]:
    label_rows = [row for row in distribution_rows if row["row_type"] == "reference_label"]
    sample_rows = [row for row in distribution_rows if row["row_type"] == "sample_id"]
    bucket_counts = Counter(str(row["sample_id_count_bucket"]) for row in label_rows)
    group_records = Counter(str(record["evaluation_group"]) for record in manifest_records)
    group_samples: dict[str, set[str]] = defaultdict(set)
    for record in manifest_records:
        if bool(record["known_sample_id_usable"]):
            group_samples[str(record["evaluation_group"])].add(
                str(record["sample_id"]).casefold()
            )
    multiple_label_samples = [
        row for row in sample_rows if bool(row["multiple_reference_labels"])
    ]
    missing_count = sum(
        _parse_bool(str(record["has_missing_metadata"]), "has_missing_metadata")
        for record in manifest_records
    )
    conflict_count = sum(
        _parse_bool(str(record["has_conflict"]), "has_conflict")
        for record in manifest_records
    )
    review_count = sum(
        _parse_bool(str(record["requires_review"]), "requires_review")
        for record in manifest_records
    )
    strict_records = sum(
        bool(record["strict_closed_set_eligible"]) for record in manifest_records
    )
    strict_samples = {
        str(record["sample_id"]).casefold()
        for record in manifest_records
        if bool(record["strict_closed_set_eligible"])
    }
    strict_labels = {
        str(record["reference_label_group_key"])
        for record in manifest_records
        if bool(record["strict_closed_set_eligible"])
    }
    return {
        "experiment": "11.1",
        "input": {
            "cache": str(cache_path),
            "index": str(index_path),
            "reference_filter": (
                "experiment 10 filter: processing_status=success and "
                "reference_candidate=True"
            ),
        },
        "parameters": {
            "seed": plan_context["seed"],
            "minimum_eligible_sample_ids_per_label": plan_context[
                "min_sample_ids_per_label"
            ],
            "split": (
                "per strict label: 1 development Sample_ID, 1 final-test "
                "Sample_ID, all remaining Sample_IDs gallery"
            ),
        },
        "inventory": {
            "reference_spectrum_count": len(manifest_records),
            "reference_label_count": len(label_rows),
            "known_sample_id_count": len(sample_rows),
            "label_sample_id_bucket_counts": {
                "one": bucket_counts.get("1", 0),
                "two": bucket_counts.get("2", 0),
                "three": bucket_counts.get("3", 0),
                "four": bucket_counts.get("4", 0),
                "five_or_more": bucket_counts.get("5_plus", 0),
            },
            "records_with_missing_metadata": missing_count,
            "records_with_conflict_metadata": conflict_count,
            "records_requiring_review": review_count,
            "known_sample_ids_with_multiple_reference_labels": len(
                multiple_label_samples
            ),
            "multiple_label_sample_examples": [
                {
                    "sample_id": row["display_value"],
                    "reference_labels": row["reference_labels"],
                    "record_count": row["record_count"],
                }
                for row in multiple_label_samples[:20]
            ],
        },
        "strict_closed_set": {
            "reference_label_count": len(strict_labels),
            "sample_id_count": len(strict_samples),
            "record_count": strict_records,
        },
        "groups": {
            group: {
                "record_count": group_records.get(group, 0),
                "known_sample_id_count": len(group_samples.get(group, set())),
            }
            for group in (
                GROUP_GALLERY,
                GROUP_DEVELOPMENT,
                GROUP_FINAL,
                GROUP_SPARSE,
                GROUP_CONFLICT,
                GROUP_INELIGIBLE,
            )
        },
        "formal_exclusion_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for record in manifest_records
                    for reason in str(record["formal_ineligibility_reasons"]).split(";")
                    if reason
                ).items()
            )
        ),
        "leakage_checks": leakage,
        "scientific_notes": {
            "reference_label": (
                "Candidate database label, not independently validated ground truth."
            ),
            "sample_id": (
                "Distinct Sample_ID values are grouping identifiers and are not "
                "automatically proven independent physical specimens."
            ),
            "no_metrics_run": (
                "No Top-1, Top-3, or Top-5 retrieval hit rate was computed, and "
                "the final-test queries were not searched."
            ),
        },
    }


def build_protocol_markdown(summary: dict[str, object]) -> str:
    inventory = summary["inventory"]
    strict = summary["strict_closed_set"]
    groups = summary["groups"]
    return f"""# RamanAI 实验 11.1：固定检索评价协议

## 1. 协议状态

本协议由实验 9.3 成功缓存和实验 10 的参考库过滤条件确定。固定随机种子为
`{summary['parameters']['seed']}`。本阶段未计算任何 Top-1、Top-3 或 Top-5 命中率，
也未运行 final test 查询。

参考标签仅是数据库中的待核实标签，不是独立验证的矿物真值；不同 Sample_ID 也不自动
等价于已经核实的独立物理标本。

## 2. 数据资格

记录必须同时满足：具有可用且无冲突的已知 Sample_ID、具有可用参考标签、无缺失或冲突
元数据标记、无需人工复核。同一 Sample_ID 的全部记录作为不可拆分单元；若同一 Sample_ID
关联多个参考标签，或组内任一记录不合格，则整个 Sample_ID 组不进入正式闭集评价。

严格闭集标签还必须具有至少
`{summary['parameters']['minimum_eligible_sample_ids_per_label']}` 个合格 Sample_ID。
不合格及稀疏标签记录继续保留为探索性检索材料，但不进入正式指标分母。

实际参考谱 {inventory['reference_spectrum_count']} 条，参考标签
{inventory['reference_label_count']} 个，已知 Sample_ID {inventory['known_sample_id_count']} 个。
严格闭集纳入 {strict['reference_label_count']} 个标签、{strict['sample_id_count']} 个
Sample_ID、{strict['record_count']} 条记录。

## 3. 固定分组

对每个严格闭集标签，将排序后的 Sample_ID 用种子 42 的确定性随机过程打乱：1 个进入
`development_query`，1 个进入 `final_test_query`，其余全部进入 `reference_gallery`。
同一 Sample_ID 的全部测量记录始终属于同一分组。

- reference_gallery：{groups[GROUP_GALLERY]['known_sample_id_count']} 个 Sample_ID，
  {groups[GROUP_GALLERY]['record_count']} 条记录。
- development_query：{groups[GROUP_DEVELOPMENT]['known_sample_id_count']} 个 Sample_ID，
  {groups[GROUP_DEVELOPMENT]['record_count']} 条记录。
- final_test_query：{groups[GROUP_FINAL]['known_sample_id_count']} 个 Sample_ID，
  {groups[GROUP_FINAL]['record_count']} 条记录。

development 仅用于算法选择、阈值和检索规则确认。final test 在协议确认后冻结，不得根据
final-test 检索成绩重新选择样本、调整规则或优化参数。

## 4. 身份隔离与泄漏

gallery、development 和 final test 不共享已知 Sample_ID。还必须保证 relative_path 和
measurement_stem 不跨分组。实验 9.3 缓存仅包含准入 RAW；Processed 衍生文件不作为独立
参考样本。Unknown、缺失或冲突身份不得进入严格闭集分组。

光谱相似只允许作为探索性重复谱诊断，不得据此自动宣布两个文件来自同一物理样本。

## 5. 后续指标定义

后续可分别报告文件级、Sample_ID 级和参考标签级 Top-1、Top-3、Top-5 命中指标：

- 文件级：保留每条 gallery 文件作为证据，检查候选参考标签是否与查询参考标签一致。
- Sample_ID 级：每个已知 gallery Sample_ID 仅保留最高排名证据后评价。
- 参考标签级：每个标签使用其全部有效 gallery 参考谱中的最高余弦相似度。

闭集分母只包含所属标签在 gallery 中存在、且通过本协议资格检查的 query Sample_ID。
若查询标签不在 gallery，应单独报告为 out-of-gallery/open-set 情况，不计入闭集命中率分母。

同一 query Sample_ID 若有多条测量，应先在该 Sample_ID 内对各测量的命中指示求平均，
然后每个 Sample_ID 以相同权重进入总体指标，避免多测量 Sample_ID 重复计权。除总体微平均外，
必须对标签计算宏平均，使每个标签权重相同，并同时报告每个标签的 Sample_ID 数和查询数。

## 6. 标签库规模偏倚

参考标签级分数采用标签内最大相似度。参考文件越多，取得偶然高最大值的机会可能越大。
后续必须联合报告每个标签的文件数和 Sample_ID 数，并分析命中率或分数与库规模的关系。
任何按标签限额、重采样或替代聚合分数的方案只能在 development 上确定，并在 final test 前冻结。

## 7. 科学表述

所有指标只能称为“相对于当前数据库候选参考标签的检索命中率”。不得称为矿物识别概率、
置信度、独立验证准确率或真实材料分类准确率。
"""


def _output_paths(output_dir: Path) -> tuple[Path, Path, Path, Path]:
    return (
        output_dir / "reference_label_sample_distribution.csv",
        output_dir / "evaluation_group_manifest.csv",
        output_dir / "evaluation_group_audit.json",
        output_dir / "EVALUATION_PROTOCOL.md",
    )


def write_outputs(
    output_dir: Path,
    manifest_records: list[dict[str, object]],
    distribution_rows: list[dict[str, object]],
    summary: dict[str, object],
    protocol_markdown: str,
) -> tuple[Path, Path, Path, Path]:
    paths = _output_paths(output_dir)
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Evaluation protocol output already exists: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    distribution_path, manifest_path, audit_path, protocol_path = paths
    with distribution_path.open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=DISTRIBUTION_FIELDS)
        writer.writeheader()
        writer.writerows(distribution_rows)
    with manifest_path.open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_records)
    protocol_path.write_text(protocol_markdown, encoding="utf-8-sig", newline="\n")
    summary["reproducibility"] = {
        "distribution_csv_sha256": _sha256_file(distribution_path),
        "manifest_csv_sha256": _sha256_file(manifest_path),
        "protocol_markdown_sha256": _sha256_file(protocol_path),
        "all_group_identity_sha256": _identity_hash(manifest_records),
        "final_test_identity_sha256": _identity_hash(
            manifest_records, {GROUP_FINAL}
        ),
        "identity_hash_fields": (
            "sample_id|evaluation_group|relative_path|reference_label_group_key, "
            "UTF-8, lexicographically sorted, LF joined"
        ),
    }
    with audit_path.open("x", encoding="utf-8-sig", newline="") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    return paths


def run_protocol(
    cache_path: Path,
    index_path: Path,
    output_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    min_sample_ids_per_label: int = DEFAULT_MIN_SAMPLE_IDS_PER_LABEL,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    tuple[Path, Path, Path, Path],
]:
    library = exp10.load_reference_library(cache_path, index_path)
    manifest, distribution, context = build_evaluation_plan(
        library.reference_records,
        seed=seed,
        min_sample_ids_per_label=min_sample_ids_per_label,
    )
    leakage = audit_leakage(manifest)
    require_leakage_audit_passed(leakage)
    summary = build_audit_summary(
        cache_path,
        index_path,
        manifest,
        distribution,
        context,
        leakage,
    )
    protocol = build_protocol_markdown(summary)
    paths = write_outputs(output_dir, manifest, distribution, summary, protocol)
    return manifest, distribution, summary, paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit reference-label and Sample_ID structure and build a fixed, "
            "leakage-checked Raman retrieval evaluation protocol."
        )
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--min-sample-ids-per-label",
        type=int,
        default=DEFAULT_MIN_SAMPLE_IDS_PER_LABEL,
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()
    try:
        _manifest, _distribution, summary, paths = run_protocol(
            arguments.cache,
            arguments.index,
            arguments.output_dir,
            seed=arguments.seed,
            min_sample_ids_per_label=arguments.min_sample_ids_per_label,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Reference spectra: {summary['inventory']['reference_spectrum_count']}")
    print(f"Reference labels: {summary['inventory']['reference_label_count']}")
    print(f"Known Sample_IDs: {summary['inventory']['known_sample_id_count']}")
    print(
        "Strict closed-set labels: "
        f"{summary['strict_closed_set']['reference_label_count']}"
    )
    print(f"Leakage audit passed: {summary['leakage_checks']['passed']}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
