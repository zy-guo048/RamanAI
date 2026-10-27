"""Read-only RamanAI V2.0.0 release-candidate integrity check.

This command never runs final-test retrieval. It only validates files, hashes,
completed formal-run audits, dependencies declared by the project, and release
documentation against already existing results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHOD_LOCK = (
    PROJECT_ROOT / "docs" / "evaluation" / "exp12_v1" / "RETRIEVAL_METHOD_LOCK.json"
)
DEFAULT_EXECUTABLE_LOCK = (
    PROJECT_ROOT
    / "docs"
    / "evaluation"
    / "exp12_v1"
    / "FINAL_EVALUATION_EXECUTABLE_LOCK.json"
)
DEFAULT_FINAL_OUTPUT = PROJECT_ROOT / "results" / "exp13_final_test_v1"

REQUIRED_FILES = (
    "app.py",
    "requirements.txt",
    "raman_app/upload.py",
    "raman_app/pipeline.py",
    "raman_app/matching.py",
    "raman_app/final_evaluation.py",
    "scripts/13_evaluate_final_test.py",
    "docs/EXPERIMENT_13_SUMMARY.md",
    "docs/V2_RELEASE.md",
    "docs/SCIENTIFIC_LIMITATIONS.md",
    "docs/evaluation/exp13_v1/FORMAL_OUTPUT_SHA256.json",
    "tests/test_pipeline.py",
    "tests/test_matching_export.py",
    "tests/test_spectrum_matching_ui.py",
)
REQUIRED_DEPENDENCIES = {"matplotlib", "numpy", "pandas", "ramanspy", "scipy", "streamlit"}
EXPECTED_COUNTS = {
    "final_test_query_row_count": 129,
    "gallery_reference_row_count": 724,
    "similarity_pair_count": 93_396,
    "prediction_count": 129,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {Path(path).name}")
    return value


def _dependency_names(requirements_path: Path) -> set[str]:
    names: set[str] = set()
    for raw in requirements_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("==", 1)[0].split(">=", 1)[0].split("<", 1)[0].strip()
        names.add(name.casefold())
    return names


def _verify_spec(project_root: Path, spec: dict[str, Any], label: str) -> dict[str, Any]:
    path = project_root / str(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path.name}")
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != int(spec["size_bytes"]):
        raise ValueError(f"{label} size mismatch: {path.name}")
    if digest != str(spec["sha256"]):
        raise ValueError(f"{label} SHA-256 mismatch: {path.name}")
    return {"path": str(spec["path"]), "size_bytes": size, "sha256": digest}


def run_release_check(
    *,
    project_root: Path,
    method_lock_path: Path,
    executable_lock_path: Path,
    final_output_dir: Path,
    require_final_results: bool = True,
) -> dict[str, Any]:
    """Return a deterministic read-only release audit or raise on inconsistency."""
    project_root = Path(project_root).resolve()
    missing = [name for name in REQUIRED_FILES if not (project_root / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing required release files: " + ", ".join(missing))

    dependencies = _dependency_names(project_root / "requirements.txt")
    missing_dependencies = sorted(REQUIRED_DEPENDENCIES - dependencies)
    if missing_dependencies:
        raise ValueError(
            "requirements.txt is missing dependencies: " + ", ".join(missing_dependencies)
        )

    method_lock = read_json(method_lock_path)
    executable_lock = read_json(executable_lock_path)
    method_hash = sha256_file(method_lock_path)
    executable_hash = sha256_file(executable_lock_path)
    if executable_lock["method_lock"]["sha256"] != method_hash:
        raise ValueError("Executable lock does not reference the current method lock")
    source_checks = [
        _verify_spec(project_root, spec, "locked executable source")
        for spec in executable_lock["executable_sources"]
    ]
    input_checks = []
    for key in (
        "evaluation_manifest",
        "experiment_11_data_source_lock",
        "experiment_9_3_npz",
        "experiment_9_3_index",
    ):
        input_checks.append(
            _verify_spec(project_root, method_lock["confirmed_inputs"][key], f"locked input {key}")
        )

    result: dict[str, Any] = {
        "release_candidate": "V2.0.0",
        "check_mode": "read_only_no_final_test_execution",
        "required_files_present": True,
        "required_dependencies_declared": True,
        "method_lock_sha256": method_hash,
        "executable_lock_sha256": executable_hash,
        "locked_source_checks": source_checks,
        "locked_input_checks": input_checks,
        "final_test_status": "not_yet_executed",
        "ready_for_release": False,
    }

    final_output_dir = Path(final_output_dir)
    if not final_output_dir.is_dir():
        if require_final_results:
            raise FileNotFoundError(
                "Formal final-test output is absent; V2.0.0 remains a release candidate"
            )
        return result

    summary_path = final_output_dir / "final_test_summary.json"
    audit_path = final_output_dir / "RUN_AUDIT.json"
    summary = read_json(summary_path)
    audit = read_json(audit_path)
    if audit.get("status") != "completed_authorized_final_test":
        raise ValueError("Formal final-test RUN_AUDIT is not completed")
    for key, expected in EXPECTED_COUNTS.items():
        if int(audit.get(key, -1)) != expected:
            raise ValueError(f"Formal final-test count mismatch for {key}")
    if summary.get("method_lock_sha256") != method_hash:
        raise ValueError("Final-test summary method-lock hash mismatch")
    if summary.get("executable_lock_sha256") != executable_hash:
        raise ValueError("Final-test summary executable-lock hash mismatch")
    if summary.get("run_id") != audit.get("run_id"):
        raise ValueError("Final-test summary and audit run_id differ")
    run_id = str(audit["run_id"])
    attempt_path = final_output_dir.parent / "final_test_attempt_audits" / f"{run_id}.json"
    attempt = read_json(attempt_path)
    if attempt.get("status") != "completed_authorized_final_test":
        raise ValueError("Persistent formal attempt audit is not completed")

    hash_manifest = read_json(
        project_root / "docs" / "evaluation" / "exp13_v1" / "FORMAL_OUTPUT_SHA256.json"
    )
    if hash_manifest.get("run_id") != run_id:
        raise ValueError("Formal output hash manifest run_id mismatch")
    if hash_manifest.get("method_lock_sha256") != method_hash:
        raise ValueError("Formal output hash manifest method-lock mismatch")
    if hash_manifest.get("executable_lock_sha256") != executable_hash:
        raise ValueError("Formal output hash manifest executable-lock mismatch")
    hash_manifest_checks = [
        _verify_spec(project_root, spec, "formal output hash manifest entry")
        for spec in hash_manifest.get("files", [])
    ]
    if len(hash_manifest_checks) != 12:
        raise ValueError("Formal output hash manifest must contain 12 entries")

    experiment_13_text = (project_root / "docs" / "EXPERIMENT_13_SUMMARY.md").read_text(
        encoding="utf-8-sig"
    )
    required_document_tokens = [
        run_id,
        f"{float(summary['sample_equal_metrics']['top_1']):.10f}",
        f"{float(summary['sample_equal_metrics']['top_3']):.10f}",
        f"{float(summary['sample_equal_metrics']['top_5']):.10f}",
        method_hash,
        executable_hash,
    ]
    if any(token not in experiment_13_text for token in required_document_tokens):
        raise ValueError("Experiment 13 summary does not match formal results and locks")

    pipeline_tests = (project_root / "tests" / "test_pipeline.py").read_text(
        encoding="utf-8-sig"
    )
    matching_export_tests = (
        project_root / "tests" / "test_matching_export.py"
    ).read_text(encoding="utf-8-sig")
    if "test_local_sample_matches_historical_peak_results" not in pipeline_tests:
        raise ValueError("V1 Annite regression test is absent")
    if "test_export_does_not_load_library_or_rerun_retrieval" not in matching_export_tests:
        raise ValueError("Matching export no-retrieval regression test is absent")

    result.update(
        {
            "final_test_status": "completed_authorized_final_test",
            "run_id": run_id,
            "formal_counts": {key: int(audit[key]) for key in EXPECTED_COUNTS},
            "formal_output_hashes": {
                summary_path.name: sha256_file(summary_path),
                audit_path.name: sha256_file(audit_path),
                attempt_path.name: sha256_file(attempt_path),
            },
            "formal_hash_manifest_entries_verified": len(hash_manifest_checks),
            "sample_equal_metrics": summary.get("sample_equal_metrics"),
            "release_document_consistency": True,
            "v1_and_matching_export_regression_tests_present": True,
            "ready_for_release": True,
        }
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--method-lock", type=Path, default=DEFAULT_METHOD_LOCK)
    parser.add_argument("--executable-lock", type=Path, default=DEFAULT_EXECUTABLE_LOCK)
    parser.add_argument("--final-output-dir", type=Path, default=DEFAULT_FINAL_OUTPUT)
    parser.add_argument(
        "--allow-pending-final-test",
        action="store_true",
        help="Audit pre-release files without treating absent formal results as success.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = run_release_check(
            project_root=args.project_root,
            method_lock_path=args.method_lock,
            executable_lock_path=args.executable_lock,
            final_output_dir=args.final_output_dir,
            require_final_results=not args.allow_pending_final_test,
        )
    except Exception as exc:
        print(f"Release check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["ready_for_release"]:
        print("Release status: pending authorized final-test; not releasable yet.")


if __name__ == "__main__":
    main()
