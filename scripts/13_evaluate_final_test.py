"""Sealed thin entrypoint for a future authorized experiment 13 run.

The current experiment 12 locks are pending human approval, so this command
cannot pass its gate. Do not create an approval record automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raman_app.final_evaluation import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_DATA_LOCK,
    DEFAULT_INDEX,
    DEFAULT_METHOD_LOCK,
    DEFAULT_SNAPSHOT_DIR,
    begin_formal_run_attempt,
    prepare_final_test_data,
    require_final_test_approval,
    run_final_test_evaluation,
    validate_frozen_inputs,
    update_formal_run_attempt,
    write_final_test_outputs_atomic,
)


DEFAULT_EXECUTABLE_LOCK = (
    PROJECT_ROOT
    / "docs"
    / "evaluation"
    / "exp12_v1"
    / "FINAL_EVALUATION_EXECUTABLE_LOCK.json"
)


def run_authorized_final_test(
    *,
    method_lock_path: Path,
    executable_lock_path: Path,
    approval_path: Path,
    snapshot_dir: Path,
    data_lock_path: Path,
    cache_path: Path,
    index_path: Path,
    output_dir: Path,
    run_id: str,
    explicit_final_mode: bool,
):
    """Run only after the one-time approval gate succeeds."""
    attempt_audit = begin_formal_run_attempt(output_dir, run_id)
    stage = "authorization"
    authorization = None
    query_count = 0
    gallery_count = 0
    scores_computed = False
    try:
        authorization = require_final_test_approval(
            method_lock_path=method_lock_path,
            executable_lock_path=executable_lock_path,
            approval_path=approval_path,
            output_dir=output_dir,
            run_id=run_id,
            explicit_final_mode=explicit_final_mode,
        )
        update_formal_run_attempt(
            attempt_audit,
            stage=stage,
            status="authorization_verified",
            updates={
                "authorization_verified": True,
                "approval_sha256": authorization["approval_sha256"],
                "runtime_environment": authorization["runtime_environment"],
            },
        )

        stage = "frozen_input_validation"
        update_formal_run_attempt(attempt_audit, stage=stage, status="started")
        _method_lock, manifest_rows, frozen_audit = validate_frozen_inputs(
            method_lock_path=method_lock_path,
            snapshot_dir=snapshot_dir,
            data_lock_path=data_lock_path,
            cache_path=cache_path,
            index_path=index_path,
            project_root=PROJECT_ROOT,
        )

        stage = "gallery_and_query_preparation"
        update_formal_run_attempt(attempt_audit, stage=stage, status="started")
        gallery, final_manifest, final_matrix = prepare_final_test_data(
            cache_path,
            index_path,
            manifest_rows,
            authorization=authorization,
        )
        query_count = int(final_matrix.shape[0])
        gallery_count = int(gallery.reference_matrix.shape[0])
        if len(final_manifest) != query_count:
            raise ValueError("Final-test manifest and query matrix row counts differ")
        if len(gallery.reference_records) != gallery_count:
            raise ValueError("Gallery metadata and matrix row counts differ")
        update_formal_run_attempt(
            attempt_audit,
            stage=stage,
            status="completed",
            updates={
                "final_test_query_row_count": query_count,
                "gallery_reference_row_count": gallery_count,
            },
        )

        stage = "evaluation_and_bootstrap"
        update_formal_run_attempt(
            attempt_audit,
            stage=stage,
            status="started",
            updates={"score_computation_started": True},
        )
        result = run_final_test_evaluation(
            gallery,
            final_manifest,
            final_matrix,
            authorization=authorization,
        )
        if len(result.record_rows) != query_count:
            raise ValueError("Prediction count differs from final-test query row count")
        scores_computed = True
        similarity_pair_count = query_count * gallery_count
        update_formal_run_attempt(
            attempt_audit,
            stage=stage,
            status="completed",
            updates={
                "similarity_pair_count": similarity_pair_count,
                "prediction_count": len(result.record_rows),
                "scores_computed": True,
            },
        )

        stage = "output_writing"
        update_formal_run_attempt(attempt_audit, stage=stage, status="started")
        paths = write_final_test_outputs_atomic(
            output_dir,
            result,
            frozen_audit,
            authorization,
            method_lock_path,
            executable_lock_path,
            run_id,
            gallery_count,
        )
        update_formal_run_attempt(
            attempt_audit,
            stage="completed",
            status="completed_authorized_final_test",
            updates={"output_file_count": len(paths)},
        )
        return result, paths
    except Exception as exc:
        update_formal_run_attempt(
            attempt_audit,
            stage=stage,
            status=(
                "rejected_before_authorization"
                if authorization is None
                else "failed_authorized_attempt"
            ),
            updates={
                "authorization_verified": authorization is not None,
                "final_test_query_row_count": query_count,
                "gallery_reference_row_count": gallery_count,
                "similarity_pair_count": (
                    query_count * gallery_count if scores_computed else 0
                ),
                "prediction_count": query_count if scores_computed else 0,
                "scores_computed": scores_computed,
                "technical_retry_requires_new_run_id_and_human_approval": True,
            },
            exception=exc,
        )
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the approved, frozen RamanAI final-test exactly once."
    )
    parser.add_argument(
        "--execute-frozen-final-test",
        action="store_true",
        help="Required explicit formal-execution acknowledgement.",
    )
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument(
        "--run-id",
        required=True,
        help="Canonical UUID bound to the one-time human approval record.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-lock", type=Path, default=DEFAULT_METHOD_LOCK)
    parser.add_argument(
        "--executable-lock", type=Path, default=DEFAULT_EXECUTABLE_LOCK
    )
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--data-lock", type=Path, default=DEFAULT_DATA_LOCK)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    result, paths = run_authorized_final_test(
        method_lock_path=args.method_lock,
        executable_lock_path=args.executable_lock,
        approval_path=args.approval,
        snapshot_dir=args.snapshot_dir,
        data_lock_path=args.data_lock,
        cache_path=args.cache,
        index_path=args.index,
        output_dir=args.output_dir,
        run_id=args.run_id,
        explicit_final_mode=args.execute_frozen_final_test,
    )
    print("Authorized frozen final-test completed")
    print(f"Query spectra: {len(result.record_rows)}")
    print(f"Query Sample_IDs: {len(result.sample_rows)}")
    for k in (1, 3, 5):
        print(f"Top-{k}: {result.sample_equal_metrics[k]:.10f}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

