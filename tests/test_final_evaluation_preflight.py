from __future__ import annotations

import codecs
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
import uuid

import numpy as np

from raman_app import final_evaluation as fe


def _reference_record(path: str, sample_id: str, label: str) -> dict[str, str]:
    return {
        "relative_path": path,
        "sample_id": sample_id,
        "sample_id_source": "header_and_filename_agree",
        "measurement_stem": Path(path).stem,
        "reference_label": label,
        "excitation_wavelength_nm": "532",
    }


def _synthetic_library(matrix: np.ndarray, records: list[dict[str, str]]):
    values = np.asarray(matrix, dtype=np.float64)
    rows = tuple(records)
    return fe.exp10.ReferenceLibrary(
        grid=np.arange(values.shape[1], dtype=np.float64),
        all_success_matrix=values.copy(),
        all_success_records=rows,
        reference_matrix=values.copy(),
        reference_records=rows,
        path_to_success_row={row["relative_path"]: index for index, row in enumerate(rows)},
    )


def _query_manifest(
    *, path: str = "query.txt", sample_id: str = "Q1", label: str = "Beta"
) -> dict[str, str]:
    return {
        "relative_path": path,
        "cache_row_index": "0",
        "measurement_stem": Path(path).stem,
        "sample_id": sample_id,
        "sample_id_source": "header_and_filename_agree",
        "reference_label": label,
        "reference_label_group_display": label,
        "reference_label_group_key": "reference_label:" + label.casefold(),
        "excitation_wavelength_nm": "532",
        "evaluation_group": fe.DEVELOPMENT_GROUP,
    }


class FrozenDevelopmentIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.method_lock, cls.manifest_rows, cls.frozen_audit = fe.validate_frozen_inputs()
        cls.gallery, cls.development_manifest, cls.development_matrix = (
            fe.prepare_development_data(
                fe.DEFAULT_CACHE, fe.DEFAULT_INDEX, cls.manifest_rows
            )
        )
        cls.result = fe.run_development_evaluation(
            cls.gallery, cls.development_manifest, cls.development_matrix
        )

    def test_frozen_counts_and_no_final_test_selection(self):
        self.assertEqual(len(self.gallery.reference_records), 724)
        self.assertEqual(len(self.development_manifest), 118)
        self.assertEqual(len(self.result.sample_rows), 60)
        self.assertEqual(len(self.result.label_rows), 60)
        self.assertEqual(len(self.result.ranking_rows), 118 * 60)
        self.assertEqual(
            self.frozen_audit["manifest"]["final_test_record_count"], 129
        )
        self.assertEqual(
            self.frozen_audit["manifest"]["final_test_sample_id_count"], 60
        )
        for field in (
            "final_test_vector_selection_count",
            "final_test_similarity_row_count",
            "final_test_prediction_count",
            "final_test_score_count",
        ):
            self.assertEqual(self.frozen_audit[field], 0)

    def test_method_a_reproduces_experiment_11_2_development(self):
        expected = {
            1: 0.8805555555555556,
            3: 0.95,
            5: 0.9666666666666667,
        }
        for k, value in expected.items():
            self.assertAlmostEqual(self.result.sample_equal_metrics[k], value, places=12)
            self.assertAlmostEqual(self.result.label_macro_metrics[k], value, places=12)
        comparison = fe.verify_development_baseline(self.result)
        self.assertTrue(comparison["query_identity_match"])
        self.assertTrue(comparison["complete_correct_label_rank_match"])
        self.assertTrue(comparison["top1_evidence_match"])
        self.assertTrue(comparison["top5_label_order_match"])

    def test_real_bootstrap_is_deterministic_and_uses_60_units(self):
        first = self.result.bootstrap
        second = fe.deterministic_sample_id_bootstrap(self.result.sample_rows)
        self.assertEqual(len(first.sample_id_order), 60)
        self.assertEqual(first.sample_id_order, second.sample_id_order)
        self.assertEqual(first.sample_id_order_sha256, second.sample_id_order_sha256)
        self.assertEqual(first.draw_indices_sha256, second.draw_indices_sha256)
        self.assertEqual(first.intervals, second.intervals)


class FinalEvaluationCoreTests(unittest.TestCase):
    def test_method_a_uses_all_gallery_files_and_stable_ties(self):
        records = [
            _reference_record("z_alpha.txt", "A1", "Alpha"),
            _reference_record("a_alpha.txt", "A2", "Alpha"),
            _reference_record("b_beta.txt", "B1", "Beta"),
        ]
        records.extend(
            _reference_record(
                f"label_{index:02d}.txt", f"S{index:02d}", f"Label{index:02d}"
            )
            for index in range(2, 60)
        )
        library = _synthetic_library(
            np.asarray(
                [[0.8, 0.6], [1.0, 0.0], [1.0, 0.0]]
                + [[0.0, 1.0]] * 58
            ),
            records,
        )
        manifest = [_query_manifest()]
        rows, rankings = fe.evaluate_reference_label_queries(
            library,
            manifest,
            np.asarray([[1.0, 0.0]]),
            query_group=fe.DEVELOPMENT_GROUP,
        )
        self.assertEqual(rows[0]["top1_reference_label"], "Alpha")
        self.assertEqual(rows[0]["top1_relative_path"], "a_alpha.txt")
        self.assertEqual(rows[0]["correct_label_rank"], 2)
        self.assertEqual(
            [row["candidate_reference_label"] for row in rankings[:2]],
            ["Alpha", "Beta"],
        )

    def test_nonfinite_and_zero_norm_vectors_are_rejected(self):
        records = [
            _reference_record(f"label_{index}.txt", f"S{index}", f"Label{index}")
            for index in range(60)
        ]
        matrix = np.tile(np.asarray([[1.0, 0.0]]), (60, 1))
        library = _synthetic_library(matrix, records)
        manifest = [_query_manifest(label="Label0")]
        with self.assertRaisesRegex(ValueError, "non-finite"):
            fe.evaluate_reference_label_queries(
                library,
                manifest,
                np.asarray([[np.nan, 0.0]]),
                query_group=fe.DEVELOPMENT_GROUP,
            )
        with self.assertRaisesRegex(ValueError, "zero or invalid norm"):
            fe.evaluate_reference_label_queries(
                library,
                manifest,
                np.asarray([[0.0, 0.0]]),
                query_group=fe.DEVELOPMENT_GROUP,
            )

    def test_missing_correct_label_is_rejected(self):
        records = [
            _reference_record(f"label_{index}.txt", f"S{index}", f"Label{index}")
            for index in range(60)
        ]
        library = _synthetic_library(
            np.tile(np.asarray([[1.0, 0.0]]), (60, 1)), records
        )
        with self.assertRaisesRegex(ValueError, "Correct frozen reference label"):
            fe.evaluate_reference_label_queries(
                library,
                [_query_manifest(label="Absent")],
                np.asarray([[1.0, 0.0]]),
                query_group=fe.DEVELOPMENT_GROUP,
            )

    def test_repeated_measurements_are_averaged_before_sample_weighting(self):
        rows = [
            {
                "query_sample_id": "S1",
                "query_reference_label": "A",
                "query_reference_label_group_key": "reference_label:a",
                "hit_at_1": 1,
                "hit_at_3": 1,
                "hit_at_5": 1,
            },
            {
                "query_sample_id": "S1",
                "query_reference_label": "A",
                "query_reference_label_group_key": "reference_label:a",
                "hit_at_1": 0,
                "hit_at_3": 1,
                "hit_at_5": 1,
            },
            {
                "query_sample_id": "S2",
                "query_reference_label": "B",
                "query_reference_label_group_key": "reference_label:b",
                "hit_at_1": 0,
                "hit_at_3": 0,
                "hit_at_5": 1,
            },
        ]
        samples = fe.aggregate_query_sample_ids(rows)
        metrics = fe.compute_sample_equal_metrics(samples)
        labels = fe.build_label_macro_rows(samples)
        macro = fe.compute_label_macro_metrics(labels)
        self.assertEqual(samples[0]["query_spectrum_count"], 2)
        self.assertEqual(metrics, {1: 0.25, 3: 0.5, 5: 1.0})
        self.assertEqual(metrics, macro)

    def test_bootstrap_fixture_freezes_draws_points_and_linear_quantiles(self):
        rows = [
            {"query_sample_id": "S1", "mean_hit_at_1": 1.0, "mean_hit_at_3": 1.0, "mean_hit_at_5": 1.0},
            {"query_sample_id": "S2", "mean_hit_at_1": 0.0, "mean_hit_at_3": 0.5, "mean_hit_at_5": 1.0},
            {"query_sample_id": "S3", "mean_hit_at_1": 0.5, "mean_hit_at_3": 0.5, "mean_hit_at_5": 0.5},
            {"query_sample_id": "S4", "mean_hit_at_1": 1.0, "mean_hit_at_3": 1.0, "mean_hit_at_5": 1.0},
        ]
        result = fe.deterministic_sample_id_bootstrap(rows, expected_unit_count=4)
        self.assertEqual(result.point_estimates, {1: 0.625, 3: 0.75, 5: 0.875})
        self.assertEqual(result.intervals, {1: (0.25, 1.0), 3: (0.5, 1.0), 5: (0.625, 1.0)})
        self.assertEqual(result.quantile_method, "linear")
        self.assertEqual(
            result.sample_id_order_sha256,
            "282c8a226ad4c2584e906b96c36893b313d56d344a90f8eff3bee0a9cb004dc0",
        )
        self.assertEqual(
            result.draw_indices_sha256,
            "1142190b073ccc8af7fb1112b94b3095729ba34fd55e74e561ca31a13d343a7e",
        )

    def test_method_lock_rejects_method_b_and_modified_input_hash(self):
        source = json.loads(fe.DEFAULT_METHOD_LOCK.read_text(encoding="utf-8-sig"))
        with tempfile.TemporaryDirectory(dir=fe.PROJECT_ROOT) as temporary:
            root = Path(temporary)
            method_b = json.loads(json.dumps(source))
            method_b["method"]["method_id"] = "B_sample_id_equal_mean"
            method_b_path = root / "method_b.json"
            method_b_path.write_text(json.dumps(method_b), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "method_id"):
                fe.load_and_validate_method_lock(method_b_path)

            altered = json.loads(json.dumps(source))
            altered["confirmed_inputs"]["evaluation_manifest"]["sha256"] = "0" * 64
            altered_path = root / "altered.json"
            altered_path.write_text(json.dumps(altered), encoding="utf-8")
            parsed = fe.load_and_validate_method_lock(altered_path)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                fe.verify_locked_inputs(parsed)

    def test_pending_executable_lock_validates_hashes_but_cannot_authorize(self):
        executable_lock = (
            fe.PROJECT_ROOT
            / "docs"
            / "evaluation"
            / "exp12_v1"
            / "FINAL_EVALUATION_EXECUTABLE_LOCK.json"
        )
        verified = fe.validate_executable_source_lock(
            executable_lock,
            fe.DEFAULT_METHOD_LOCK,
            require_approved=False,
        )
        self.assertEqual(
            verified["lock"]["lock_status"],
            "frozen_candidate_pending_human_approval",
        )
        self.assertGreaterEqual(len(verified["verified_sources"]), 6)
        formally_checked = fe.validate_executable_source_lock(
            executable_lock,
            fe.DEFAULT_METHOD_LOCK,
            require_approved=True,
        )
        self.assertEqual(len(formally_checked["verified_sources"]), 6)

        altered = json.loads(executable_lock.read_text(encoding="utf-8-sig"))
        altered["executable_sources"][0]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory(dir=fe.PROJECT_ROOT) as temporary:
            altered_path = Path(temporary) / "altered-executable-lock.json"
            altered_path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                fe.validate_executable_source_lock(
                    altered_path,
                    fe.DEFAULT_METHOD_LOCK,
                    require_approved=False,
                )

    def test_final_test_is_rejected_before_matrix_access(self):
        class ExplodingMatrix:
            def __array__(self, *args, **kwargs):
                raise AssertionError("matrix must not be accessed before authorization")

        with self.assertRaisesRegex(PermissionError, "before vector selection"):
            fe.select_query_rows_for_formal_evaluation(
                [],
                ExplodingMatrix(),
                query_group=fe.FINAL_TEST_GROUP,
                authorization=None,
            )

    def test_synthetic_authorization_allows_only_requested_final_rows(self):
        manifest = [
            {"evaluation_group": fe.GALLERY_GROUP, "cache_row_index": "0"},
            {"evaluation_group": fe.FINAL_TEST_GROUP, "cache_row_index": "1"},
            {"evaluation_group": fe.DEVELOPMENT_GROUP, "cache_row_index": "2"},
        ]
        selected, matrix = fe.select_query_rows_for_formal_evaluation(
            manifest,
            np.asarray([[1.0], [2.0], [3.0]]),
            query_group=fe.FINAL_TEST_GROUP,
            authorization={"authorized": True},
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["evaluation_group"], fe.FINAL_TEST_GROUP)
        np.testing.assert_array_equal(matrix, np.asarray([[2.0]]))

    def test_sealed_experiment_13_entrypoint_stops_before_input_validation(self):
        entry = fe._load_module(
            "raman_exp13_sealed_entry_for_test",
            fe.PROJECT_ROOT / "scripts" / "13_evaluate_final_test.py",
        )
        with tempfile.TemporaryDirectory(dir=fe.PROJECT_ROOT) as temporary:
            root = Path(temporary)
            run_id = str(uuid.uuid4())
            with mock.patch.object(entry, "validate_frozen_inputs") as validate_inputs:
                with self.assertRaisesRegex(PermissionError, "Explicit formal"):
                    entry.run_authorized_final_test(
                        method_lock_path=fe.DEFAULT_METHOD_LOCK,
                        executable_lock_path=root / "missing-executable-lock.json",
                        approval_path=root / "missing-approval.json",
                        snapshot_dir=fe.DEFAULT_SNAPSHOT_DIR,
                        data_lock_path=fe.DEFAULT_DATA_LOCK,
                        cache_path=fe.DEFAULT_CACHE,
                        index_path=fe.DEFAULT_INDEX,
                        output_dir=root / "unused-final-output",
                        run_id=run_id,
                        explicit_final_mode=False,
                    )
                validate_inputs.assert_not_called()
            audit = json.loads(
                (root / fe.FORMAL_ATTEMPT_AUDIT_DIRNAME / f"{run_id}.json").read_text(
                    encoding="utf-8-sig"
                )
            )
            self.assertEqual(audit["status"], "rejected_before_authorization")
            self.assertFalse(audit["authorization_verified"])
            self.assertFalse(audit["scores_computed"])

    def test_synthetic_approval_state_machine(self):
        source = json.loads(fe.DEFAULT_METHOD_LOCK.read_text(encoding="utf-8-sig"))
        candidate = "frozen_candidate_pending_external_human_approval"
        source["approval"]["method_configuration"] = candidate
        source["approval"]["bootstrap_configuration"] = candidate
        source["approval"]["executable_source_lock"] = candidate
        source["approval"]["final_test_execution"] = "requires_valid_experiment_13_single_run_approval_record"
        with tempfile.TemporaryDirectory(dir=fe.PROJECT_ROOT) as temporary:
            root = Path(temporary)
            method_path = root / "method.json"
            method_path.write_text(json.dumps(source, sort_keys=True), encoding="utf-8")
            executable_path = root / "executable.json"
            core_relative = "raman_app/final_evaluation.py"
            core_path = fe.PROJECT_ROOT / core_relative
            executable_path.write_text(
                json.dumps(
                    {
                        "schema": "ramanai_final_evaluation_executable_lock_v1",
                        "lock_status": "frozen_candidate_pending_human_approval",
                        "method_lock": {"sha256": fe.sha256_file(method_path)},
                        "executable_sources": [
                            {
                                "path": core_relative,
                                "size_bytes": core_path.stat().st_size,
                                "sha256": fe.sha256_file(core_path),
                            }
                        ],
                        "runtime_environment": {
                            "python_version": fe.environment_snapshot()["python_version"],
                            "numpy_version": fe.environment_snapshot()["numpy_version"],
                            "scipy_version": fe.environment_snapshot()["scipy_version"],
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            output_dir = root / "unused_output"
            run_id = str(uuid.uuid4())
            approval = {
                "approval_status": "approved_for_single_final_test_attempt",
                "approved_by": "synthetic-test-only",
                "approved_at": "2000-01-01T00:00:00Z",
                "method_lock_sha256": fe.sha256_file(method_path),
                "executable_lock_sha256": fe.sha256_file(executable_path),
                "final_test_identity_sha256": source["frozen_evaluation_data"]["final_test"]["identity_sha256"],
                "npz_sha256": source["confirmed_inputs"]["experiment_9_3_npz"]["sha256"],
                "index_sha256": source["confirmed_inputs"]["experiment_9_3_index"]["sha256"],
                "evaluation_manifest_sha256": source["confirmed_inputs"]["evaluation_manifest"]["sha256"],
                "data_source_lock_sha256": source["confirmed_inputs"]["experiment_11_data_source_lock"]["sha256"],
                "run_id": run_id,
                "output_dir": str(output_dir.resolve()),
            }
            approval_path = root / "approval.json"
            approval_path.write_text(json.dumps(approval), encoding="utf-8")
            authorized = fe.require_final_test_approval(
                method_lock_path=method_path,
                executable_lock_path=executable_path,
                approval_path=approval_path,
                output_dir=output_dir,
                run_id=run_id,
                explicit_final_mode=True,
            )
            self.assertTrue(authorized["authorized"])
            approval["npz_sha256"] = "0" * 64
            approval_path.write_text(json.dumps(approval), encoding="utf-8")
            with self.assertRaisesRegex(PermissionError, "npz_sha256"):
                fe.require_final_test_approval(
                    method_lock_path=method_path,
                    executable_lock_path=executable_path,
                    approval_path=approval_path,
                    output_dir=output_dir,
                    run_id=run_id,
                    explicit_final_mode=True,
                )

    def test_runtime_environment_mismatch_is_rejected_before_approval_use(self):
        source = json.loads(fe.DEFAULT_METHOD_LOCK.read_text(encoding="utf-8-sig"))
        candidate = "frozen_candidate_pending_external_human_approval"
        source["approval"]["method_configuration"] = candidate
        source["approval"]["bootstrap_configuration"] = candidate
        source["approval"]["executable_source_lock"] = candidate
        source["approval"]["final_test_execution"] = "requires_valid_experiment_13_single_run_approval_record"
        with tempfile.TemporaryDirectory(dir=fe.PROJECT_ROOT) as temporary:
            root = Path(temporary)
            method_path = root / "method.json"
            method_path.write_text(json.dumps(source), encoding="utf-8")
            core = fe.PROJECT_ROOT / "raman_app" / "final_evaluation.py"
            executable_path = root / "executable.json"
            executable_path.write_text(
                json.dumps(
                    {
                        "schema": "ramanai_final_evaluation_executable_lock_v1",
                        "lock_status": "frozen_candidate_pending_human_approval",
                        "method_lock": {"sha256": fe.sha256_file(method_path)},
                        "executable_sources": [{
                            "path": "raman_app/final_evaluation.py",
                            "size_bytes": core.stat().st_size,
                            "sha256": fe.sha256_file(core),
                        }],
                        "runtime_environment": {
                            "python_version": "0.0.0",
                            "numpy_version": fe.environment_snapshot()["numpy_version"],
                            "scipy_version": fe.environment_snapshot()["scipy_version"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "python_version"):
                fe.require_final_test_approval(
                    method_lock_path=method_path,
                    executable_lock_path=executable_path,
                    approval_path=root / "not-read.json",
                    output_dir=root / "output",
                    run_id=str(uuid.uuid4()),
                    explicit_final_mode=True,
                )

    def test_run_id_is_single_use_and_attempt_failures_are_persistent(self):
        entry = fe._load_module(
            "raman_exp13_attempt_audit_for_test",
            fe.PROJECT_ROOT / "scripts" / "13_evaluate_final_test.py",
        )
        stages = (
            ("frozen_input_validation", "validate_frozen_inputs"),
            ("gallery_and_query_preparation", "prepare_final_test_data"),
            ("evaluation_and_bootstrap", "run_final_test_evaluation"),
            ("output_writing", "write_final_test_outputs_atomic"),
        )
        with tempfile.TemporaryDirectory(dir=fe.PROJECT_ROOT) as temporary:
            root = Path(temporary)
            for expected_stage, failing_name in stages:
                with self.subTest(stage=expected_stage):
                    run_id = str(uuid.uuid4())
                    output = root / f"output-{run_id}"
                    authorization = {
                        "authorized": True,
                        "approval_sha256": "a" * 64,
                        "run_id": run_id,
                        "runtime_environment": fe.environment_snapshot(),
                    }
                    gallery = SimpleNamespace(
                        reference_records=tuple(range(724)),
                        reference_matrix=np.ones((724, 2), dtype=float),
                    )
                    final_manifest = tuple({"relative_path": str(i)} for i in range(129))
                    final_matrix = np.ones((129, 2), dtype=float)
                    result = SimpleNamespace(record_rows=tuple(range(129)))
                    patches = {
                        "require_final_test_approval": mock.patch.object(
                            entry, "require_final_test_approval", return_value=authorization
                        ),
                        "validate_frozen_inputs": mock.patch.object(
                            entry,
                            "validate_frozen_inputs",
                            return_value=({}, [], {}),
                        ),
                        "prepare_final_test_data": mock.patch.object(
                            entry,
                            "prepare_final_test_data",
                            return_value=(gallery, final_manifest, final_matrix),
                        ),
                        "run_final_test_evaluation": mock.patch.object(
                            entry, "run_final_test_evaluation", return_value=result
                        ),
                        "write_final_test_outputs_atomic": mock.patch.object(
                            entry,
                            "write_final_test_outputs_atomic",
                            return_value=(output / "summary.json",),
                        ),
                    }
                    patches[failing_name] = mock.patch.object(
                        entry, failing_name, side_effect=RuntimeError("synthetic failure")
                    )
                    managers = [patch.start() for patch in patches.values()]
                    try:
                        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                            entry.run_authorized_final_test(
                                method_lock_path=root / "method.json",
                                executable_lock_path=root / "executable.json",
                                approval_path=root / "approval.json",
                                snapshot_dir=root,
                                data_lock_path=root / "data-lock.json",
                                cache_path=root / "cache.npz",
                                index_path=root / "index.csv",
                                output_dir=output,
                                run_id=run_id,
                                explicit_final_mode=True,
                            )
                    finally:
                        for patch in patches.values():
                            patch.stop()
                    audit_path = root / fe.FORMAL_ATTEMPT_AUDIT_DIRNAME / f"{run_id}.json"
                    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
                    self.assertEqual(audit["stage"], expected_stage)
                    self.assertEqual(audit["status"], "failed_authorized_attempt")
                    self.assertEqual(audit["exception_type"], "RuntimeError")
                    self.assertTrue(audit["technical_retry_requires_new_run_id_and_human_approval"])
                    if expected_stage == "output_writing":
                        self.assertEqual(audit["final_test_query_row_count"], 129)
                        self.assertEqual(audit["gallery_reference_row_count"], 724)
                        self.assertEqual(audit["similarity_pair_count"], 129 * 724)
                        self.assertEqual(audit["prediction_count"], 129)
                        self.assertTrue(audit["scores_computed"])
                    with self.assertRaisesRegex(PermissionError, "already started"):
                        fe.begin_formal_run_attempt(output, run_id)

    def test_atomic_outputs_use_utf8_sig_refuse_overwrite_and_keep_failure_audit(self):
        bootstrap = fe.BootstrapResult(
            sample_id_order=("S1",),
            sample_id_order_sha256="a" * 64,
            draw_indices_sha256="b" * 64,
            point_estimates={1: 1.0, 3: 1.0, 5: 1.0},
            intervals={1: (1.0, 1.0), 3: (1.0, 1.0), 5: (1.0, 1.0)},
            replicate_count=10_000,
            random_seed=42,
            quantile_method="linear",
        )
        record = {field: "" for field in fe.RECORD_FIELDS}
        ranking = {field: "" for field in fe.RANKING_FIELDS}
        sample = {field: "" for field in fe.SAMPLE_FIELDS}
        label = {field: "" for field in fe.LABEL_FIELDS}
        result = fe.EvaluationResult(
            query_group=fe.DEVELOPMENT_GROUP,
            record_rows=(record,),
            ranking_rows=(ranking,),
            sample_rows=(sample,),
            label_rows=(label,),
            confusion_rows=(),
            sample_equal_metrics={1: 1.0, 3: 1.0, 5: 1.0},
            label_macro_metrics={1: 1.0, 3: 1.0, 5: 1.0},
            bootstrap=bootstrap,
        )
        with tempfile.TemporaryDirectory(dir=fe.PROJECT_ROOT) as temporary:
            root = Path(temporary)
            output = root / "success"
            paths = fe.write_preflight_outputs_atomic(
                output,
                result,
                {"final_test_vector_selection_count": 0},
                {"query_identity_match": True},
                fe.DEFAULT_METHOD_LOCK,
            )
            self.assertTrue(paths[0].read_bytes().startswith(codecs.BOM_UTF8))
            with self.assertRaises(FileExistsError):
                fe.write_preflight_outputs_atomic(
                    output,
                    result,
                    {},
                    {},
                    fe.DEFAULT_METHOD_LOCK,
                )

            failing_output = root / "failure"
            original = fe._write_csv
            call_count = 0

            def fail_second_csv(path, fields, rows):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise OSError("synthetic write failure")
                return original(path, fields, rows)

            with mock.patch.object(fe, "_write_csv", side_effect=fail_second_csv):
                with self.assertRaisesRegex(OSError, "synthetic write failure"):
                    fe.write_preflight_outputs_atomic(
                        failing_output,
                        result,
                        {},
                        {},
                        fe.DEFAULT_METHOD_LOCK,
                    )
            failed_dirs = list(root.glob(".failure.tmp-*"))
            self.assertEqual(len(failed_dirs), 1)
            self.assertTrue((failed_dirs[0] / "RUN_AUDIT.failed.json").is_file())
            self.assertFalse(failing_output.exists())

    def test_final_output_audit_distinguishes_queries_gallery_and_pairs(self):
        bootstrap = fe.BootstrapResult(
            sample_id_order=("S1",),
            sample_id_order_sha256="a" * 64,
            draw_indices_sha256="b" * 64,
            point_estimates={1: 1.0, 3: 1.0, 5: 1.0},
            intervals={1: (1.0, 1.0), 3: (1.0, 1.0), 5: (1.0, 1.0)},
            replicate_count=10_000,
            random_seed=42,
            quantile_method="linear",
        )
        record = {field: "" for field in fe.RECORD_FIELDS}
        record["hit_at_1"] = 0
        result = fe.EvaluationResult(
            query_group=fe.FINAL_TEST_GROUP,
            record_rows=(record,),
            ranking_rows=({field: "" for field in fe.RANKING_FIELDS},),
            sample_rows=({field: "" for field in fe.SAMPLE_FIELDS},),
            label_rows=({field: "" for field in fe.LABEL_FIELDS},),
            confusion_rows=(),
            sample_equal_metrics={1: 1.0, 3: 1.0, 5: 1.0},
            label_macro_metrics={1: 1.0, 3: 1.0, 5: 1.0},
            bootstrap=bootstrap,
        )
        run_id = str(uuid.uuid4())
        authorization = {
            "authorized": True,
            "approval_sha256": "a" * 64,
            "run_id": run_id,
        }
        executable_lock = (
            fe.PROJECT_ROOT
            / "docs"
            / "evaluation"
            / "exp12_v1"
            / "FINAL_EVALUATION_EXECUTABLE_LOCK.json"
        )
        with tempfile.TemporaryDirectory(dir=fe.PROJECT_ROOT) as temporary:
            output = Path(temporary) / "formal-output"
            fe.write_final_test_outputs_atomic(
                output,
                result,
                {},
                authorization,
                fe.DEFAULT_METHOD_LOCK,
                executable_lock,
                run_id,
                2,
            )
            audit = json.loads(
                (output / "RUN_AUDIT.json").read_text(encoding="utf-8-sig")
            )
            self.assertEqual(audit["final_test_query_row_count"], 1)
            self.assertEqual(audit["gallery_reference_row_count"], 2)
            self.assertEqual(audit["similarity_pair_count"], 2)
            self.assertEqual(audit["prediction_count"], 1)


if __name__ == "__main__":
    unittest.main()

