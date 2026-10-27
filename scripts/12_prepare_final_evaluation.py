"""Prepare and verify the frozen RamanAI final-evaluation implementation.

This experiment 12 command evaluates the frozen Development split only. It has
no CLI branch that selects or evaluates final-test vectors.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raman_app.final_evaluation import (  # noqa: E402
    DEFAULT_BASELINE_DIR,
    DEFAULT_CACHE,
    DEFAULT_DATA_LOCK,
    DEFAULT_INDEX,
    DEFAULT_METHOD_LOCK,
    DEFAULT_SNAPSHOT_DIR,
    prepare_development_data,
    run_development_evaluation,
    validate_frozen_inputs,
    verify_development_baseline,
    write_preflight_outputs_atomic,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "exp12_final_evaluation_preflight"


def run_preflight(
    *,
    method_lock_path: Path,
    snapshot_dir: Path,
    data_lock_path: Path,
    cache_path: Path,
    index_path: Path,
    baseline_dir: Path,
    output_dir: Path,
):
    """Validate frozen inputs and run method A on Development only."""
    if Path(output_dir).exists():
        raise FileExistsError(f"Preflight output directory already exists: {output_dir}")
    _method_lock, manifest_rows, frozen_audit = validate_frozen_inputs(
        method_lock_path=method_lock_path,
        snapshot_dir=snapshot_dir,
        data_lock_path=data_lock_path,
        cache_path=cache_path,
        index_path=index_path,
        project_root=PROJECT_ROOT,
    )
    gallery, development_manifest, development_matrix = prepare_development_data(
        cache_path, index_path, manifest_rows
    )
    result = run_development_evaluation(
        gallery, development_manifest, development_matrix
    )
    baseline_comparison = verify_development_baseline(result, baseline_dir)
    paths = write_preflight_outputs_atomic(
        output_dir,
        result,
        frozen_audit,
        baseline_comparison,
        method_lock_path,
    )
    return result, paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate frozen RamanAI evaluation inputs and reproduce the "
            "Development method-A baseline. This command cannot run final-test."
        )
    )
    parser.add_argument("--method-lock", type=Path, default=DEFAULT_METHOD_LOCK)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--data-lock", type=Path, default=DEFAULT_DATA_LOCK)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    result, paths = run_preflight(
        method_lock_path=args.method_lock,
        snapshot_dir=args.snapshot_dir,
        data_lock_path=args.data_lock,
        cache_path=args.cache,
        index_path=args.index,
        baseline_dir=args.baseline_dir,
        output_dir=args.output_dir,
    )
    print("Experiment 12-B Development preflight completed")
    print(f"Development spectra: {len(result.record_rows)}")
    print(f"Development Sample_IDs: {len(result.sample_rows)}")
    for k in (1, 3, 5):
        lower, upper = result.bootstrap.intervals[k]
        print(
            f"Top-{k}: {result.sample_equal_metrics[k]:.10f} "
            f"(bootstrap 95% percentile [{lower:.10f}, {upper:.10f}])"
        )
    print("Final-test vector selections: 0")
    print("Final-test similarity rows: 0")
    print("Final-test predictions: 0")
    print("Final-test scores: 0")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

