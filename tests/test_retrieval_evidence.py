from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "11_analyze_retrieval_evidence.py"
spec = importlib.util.spec_from_file_location("retrieval_evidence_test", SCRIPT_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load experiment 11.4-A module")
evidence = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evidence
spec.loader.exec_module(evidence)


def _record(
    path: str,
    sample_id: str,
    label: str,
    wavelength: str = "532",
    source: str = "filename_header_match",
) -> dict[str, str]:
    return {
        "relative_path": path,
        "sample_id": sample_id,
        "sample_id_source": source,
        "measurement_stem": Path(path).stem,
        "reference_label": label,
        "excitation_wavelength_nm": wavelength,
    }


def _key(label: str) -> str:
    return evidence.exp10._reference_label_identity(
        _record("x.txt", "S", label)
    )[2]


def _sixty_label_fixture():
    records: list[dict[str, str]] = []
    for label_index in range(60):
        label = f"Label {label_index:02d}"
        for sample_index in range(3):
            records.append(
                _record(
                    f"l{label_index:02d}-s{sample_index}.txt",
                    f"L{label_index:02d}S{sample_index}",
                    label,
                    str(500 + label_index),
                )
            )
    scores = np.linspace(0.8, -0.8, len(records), dtype=float)
    target_label = "Label 00"
    scores[0:3] = [0.95, 0.90, 0.85]
    manifest = [{
        "relative_path": "development.txt",
        "cache_row_index": "999",
        "sample_id": "DEV001",
        "reference_label_group_display": target_label,
        "reference_label_group_key": _key(target_label),
        "excitation_wavelength_nm": "532",
        "excitation_wavelength_source": "header",
        "metadata_status": "consistent",
        "metadata_requires_review": "False",
        "review_reasons": "",
    }]
    baseline = [{
        "query_relative_path": "development.txt",
        "query_cache_row_index": "999",
        "query_sample_id": "DEV001",
        "query_reference_label_group_key": _key(target_label),
        "reference_label_top1_reference_label": target_label,
        "reference_label_matching_rank": "1",
        "reference_label_top1_cosine_similarity": "0.95",
        "reference_label_hit_at_1": "1",
        "reference_label_hit_at_3": "1",
        "reference_label_hit_at_5": "1",
    }]
    return tuple(records), scores.reshape(1, -1), manifest, baseline


class RetrievalEvidenceTests(unittest.TestCase):
    def test_method_a_baseline_and_full_ranking_are_reproduced(self):
        records, matrix, manifest, baseline = _sixty_label_fixture()
        query_rows, candidate_rows, support_rows, results = (
            evidence.build_query_evidence(matrix, records, manifest, baseline)
        )
        self.assertEqual(results[0]["correct_label_rank"], 1)
        self.assertEqual(query_rows[0]["top1_label"], "Label 00")
        self.assertEqual(len(candidate_rows), 5)
        self.assertEqual(len(support_rows), 15)
        self.assertEqual(len(evidence.aggregation.rank_reference_labels(
            matrix[0], records,
            evidence.aggregation.validate_gallery_for_aggregation(records),
            evidence.aggregation.METHOD_A,
        )), 60)

    def test_same_sample_multiple_files_count_as_one_support_identity(self):
        records = (
            _record("s1-a.txt", "S1", "A"),
            _record("s1-b.txt", "S1", "A"),
            _record("s2.txt", "S2", "A"),
        )
        ranked = evidence.rank_sample_id_support(
            np.asarray([0.8, 0.9, 0.7]), records,
            {"s1": [0, 1], "s2": [2]},
        )
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0]["sample_id"], "S1")
        self.assertEqual(ranked[0]["file_count"], 2)
        self.assertEqual(ranked[0]["best_record"]["relative_path"], "s1-b.txt")

    def test_best_evidence_identity_path_and_wavelength_are_mapped(self):
        records, matrix, manifest, baseline = _sixty_label_fixture()
        _, candidates, supports, _ = evidence.build_query_evidence(
            matrix, records, manifest, baseline
        )
        top = candidates[0]
        self.assertEqual(top["best_support_relative_path"], "l00-s0.txt")
        self.assertEqual(top["best_support_sample_id"], "L00S0")
        self.assertEqual(top["best_support_excitation_wavelength_nm"], "500")
        self.assertEqual(supports[0]["support_best_relative_path"], "l00-s0.txt")

    def test_score_gaps_are_computed_with_documented_sign(self):
        records, matrix, manifest, baseline = _sixty_label_fixture()
        queries, candidates, _, _ = evidence.build_query_evidence(
            matrix, records, manifest, baseline
        )
        self.assertAlmostEqual(
            queries[0]["top1_minus_top2_label_score"],
            queries[0]["top1_score"] - queries[0]["top2_score"],
        )
        self.assertAlmostEqual(
            candidates[0]["best_minus_second_independent_sample_id_score"],
            0.05,
        )

    def test_tied_support_scores_use_stable_sample_and_path_order(self):
        records = (
            _record("z.txt", "Z", "A"),
            _record("a.txt", "A", "A"),
            _record("a2.txt", "A", "A"),
        )
        first = evidence.rank_sample_id_support(
            np.asarray([0.5, 0.5, 0.5]), records,
            {"z": [0], "a": [1, 2]},
        )
        second = evidence.rank_sample_id_support(
            np.asarray([0.5, 0.5, 0.5]), records,
            {"z": [0], "a": [1, 2]},
        )
        self.assertEqual(first, second)
        self.assertEqual(first[0]["sample_id"], "A")
        self.assertEqual(first[0]["best_record"]["relative_path"], "a.txt")

    def test_nonfinite_conflicted_identity_and_missing_label_stop(self):
        records = (_record("a.txt", "S1", "A"),)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            evidence.rank_sample_id_support(
                np.asarray([np.nan]), records, {"s1": [0]}
            )
        with self.assertRaisesRegex(ValueError, "conflicting identities"):
            evidence.rank_sample_id_support(
                np.asarray([0.8, 0.7]),
                (_record("a.txt", "S1", "A"), _record("b.txt", "S2", "A")),
                {"conflict": [0, 1]},
            )
        with self.assertRaisesRegex(ValueError, "missing or unusable"):
            evidence.aggregation.validate_gallery_for_aggregation(
                (_record("unknown.txt", "S1", "Unknown"),)
            )

    def test_only_supplied_development_and_gallery_enter_similarity(self):
        gallery = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        development = np.asarray([[1.0, 0.0]])
        matrix = evidence.sensitivity.compute_similarity_matrix(gallery, development)
        self.assertEqual(matrix.shape, (1, 2))
        self.assertTrue(np.allclose(matrix, [[1.0, 0.0]]))

    def test_matrix_or_baseline_identity_mismatch_stops(self):
        records, matrix, manifest, baseline = _sixty_label_fixture()
        with self.assertRaisesRegex(ValueError, "does not match frozen data"):
            evidence.build_query_evidence(matrix[:, :-1], records, manifest, baseline)
        baseline[0]["query_sample_id"] = "WRONG"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            evidence.build_query_evidence(matrix, records, manifest, baseline)

    def test_hash_error_and_missing_frozen_inputs_stop(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                evidence.sha256_file(root / "missing")
            names = (
                "evaluation_group_manifest.csv",
                "reference_label_sample_distribution.csv",
                "evaluation_group_audit.json",
                "EVALUATION_PROTOCOL.md",
            )
            for name in names:
                (root / name).write_bytes(name.encode())
            cache = root / "cache.npz"
            index = root / "index.csv"
            cache.write_bytes(b"cache")
            index.write_bytes(b"index")

            def item(path: Path):
                return {
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }

            lock_data = {
                "evaluation_snapshot": {name: item(root / name) for name in names},
                "source_data": {
                    "experiment_9_3_npz": item(cache),
                    "experiment_9_3_index": item(index),
                },
                "frozen_evaluation": {},
            }
            lock_data["source_data"]["experiment_9_3_npz"]["sha256"] = "0" * 64
            lock = root / "lock.json"
            lock.write_text(json.dumps(lock_data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                evidence.baseline.validate_source_lock(lock, root, cache, index)

    def test_outputs_are_utf8_sig_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            output = Path(temporary) / "output"
            paths = evidence.write_outputs(
                output, [], [], [], {"ok": True}, "# report\n"
            )
            self.assertTrue(all(path.exists() for path in paths))
            self.assertTrue(paths[0].read_bytes().startswith(b"\xef\xbb\xbf"))
            with self.assertRaises(FileExistsError):
                evidence.write_outputs(
                    output, [], [], [], {"ok": True}, "# report\n"
                )


if __name__ == "__main__":
    unittest.main()
