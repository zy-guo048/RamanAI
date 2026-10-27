from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from raman_app.export import peak_features_to_csv
from raman_app.matching import (
    MATCHING_CSV_FIELDS,
    MatchingCandidate,
    ReferenceLibrarySourceFingerprint,
    SourceFileFingerprint,
    SpectrumMatchingResult,
    SupportingIdentity,
    build_matching_csv_filename,
    build_matching_json_filename,
    matching_candidates_to_csv,
    matching_exports_for_current_source,
    matching_record_to_json,
    matching_result_matches_request,
    load_reference_library_with_fingerprint,
    sha256_file,
    validate_reference_source_snapshot,
)


QUERY_BYTES = "# 光谱\n300,1\n850,2\n".encode("utf-8")


def _source_fingerprint(
    npz_sha: str = "a" * 64,
    index_sha: str = "b" * 64,
) -> ReferenceLibrarySourceFingerprint:
    return ReferenceLibrarySourceFingerprint(
        npz_cache=SourceFileFingerprint(
            filename="rruff_cache.npz",
            relative_identifier="project_relative:results/cache/rruff_cache.npz",
            sha256=npz_sha,
            size_bytes=1234,
        ),
        index_csv=SourceFileFingerprint(
            filename="rruff_index.csv",
            relative_identifier="project_relative:results/cache/rruff_index.csv",
            sha256=index_sha,
            size_bytes=5678,
        ),
    )


def _support(
    rank: int,
    sample_id: str,
    score: float,
    path: str,
    wavelength: str,
) -> SupportingIdentity:
    return SupportingIdentity(
        rank=rank,
        identity_basis="known_sample_id",
        identity_key=f"sample_id:{sample_id.casefold()}",
        sample_id=sample_id,
        score=score,
        relative_path=path,
        excitation_wavelength_nm=wavelength,
        file_count=1,
    )


def _result() -> SpectrumMatchingResult:
    candidates = (
        MatchingCandidate(
            rank=1,
            reference_label="石英 α",
            cosine_similarity=0.987654321,
            reference_file_count=4,
            distinguishable_identity_count=2,
            best_relative_path="Quartz_样本_RAW.txt",
            best_sample_id="R中文001",
            best_excitation_wavelength_nm="532",
            best_minus_second_identity_score=0.0125,
            supporting_identities=(
                _support(1, "R中文001", 0.987654321, "Quartz_样本_RAW.txt", "532"),
                _support(2, "R000002", 0.975154321, "Quartz_2_RAW.txt", "785"),
            ),
            best_reference_vector=np.asarray([0.1, 0.5, 1.0]),
        ),
        MatchingCandidate(
            rank=2,
            reference_label="Unknown metadata example",
            cosine_similarity=0.8,
            reference_file_count=1,
            distinguishable_identity_count=1,
            best_relative_path="unknown_RAW.txt",
            best_sample_id="",
            best_excitation_wavelength_nm="",
            best_minus_second_identity_score=None,
            supporting_identities=(
                _support(1, "", 0.8, "unknown_RAW.txt", ""),
            ),
            best_reference_vector=np.asarray([0.2, 0.4, 0.8]),
        ),
        MatchingCandidate(
            rank=3,
            reference_label="Calcite",
            cosine_similarity=0.7,
            reference_file_count=2,
            distinguishable_identity_count=2,
            best_relative_path="calcite_RAW.txt",
            best_sample_id="R3",
            best_excitation_wavelength_nm="633",
            best_minus_second_identity_score=0.03,
            supporting_identities=(
                _support(1, "R3", 0.7, "calcite_RAW.txt", "633"),
                _support(2, "R4", 0.67, "calcite_2_RAW.txt", "532"),
            ),
            best_reference_vector=np.asarray([0.3, 0.6, 0.9]),
        ),
    )
    return SpectrumMatchingResult(
        filename="查询 光谱.txt",
        query_sha256=hashlib.sha256(QUERY_BYTES).hexdigest(),
        requested_top_k=3,
        grid=np.linspace(300.0, 850.0, 1101),
        query_vector=np.zeros(1101),
        reference_library_size=5529,
        candidates=candidates,
        source_fingerprint=_source_fingerprint(),
    )


class MatchingExportTests(unittest.TestCase):
    def test_csv_row_count_equals_current_candidate_count(self):
        result = _result()
        rows = list(
            csv.DictReader(
                StringIO(matching_candidates_to_csv(result).decode("utf-8-sig"))
            )
        )
        self.assertEqual(len(rows), len(result.candidates))
        self.assertEqual(tuple(rows[0]), MATCHING_CSV_FIELDS)

    def test_csv_order_labels_and_scores_match_displayed_result(self):
        result = _result()
        rows = list(
            csv.DictReader(
                StringIO(matching_candidates_to_csv(result).decode("utf-8-sig"))
            )
        )
        self.assertEqual([int(row["candidate_rank"]) for row in rows], [1, 2, 3])
        self.assertEqual(
            [row["reference_label"] for row in rows],
            [candidate.reference_label for candidate in result.candidates],
        )
        self.assertEqual(
            [float(row["cosine_similarity"]) for row in rows],
            [candidate.cosine_similarity for candidate in result.candidates],
        )

    def test_json_records_real_sha_topk_grid_and_preprocessing(self):
        payload = json.loads(matching_record_to_json(_result()).decode("utf-8"))
        self.assertEqual(
            payload["query"]["uploaded_file_sha256"],
            hashlib.sha256(QUERY_BYTES).hexdigest(),
        )
        self.assertEqual(payload["retrieval"]["requested_top_k"], 3)
        self.assertEqual(payload["common_grid"]["point_count"], 1101)
        self.assertEqual(payload["common_grid"]["start_cm_1"], 300.0)
        self.assertEqual(payload["common_grid"]["end_cm_1"], 850.0)
        self.assertEqual(payload["common_grid"]["step_cm_1"], 0.5)
        self.assertEqual(payload["preprocessing"]["sg_window_length_points"], 21)
        self.assertEqual(payload["preprocessing"]["asls_lam"], 1e6)
        self.assertEqual(
            payload["preprocessing"]["normalization_stage"],
            "before_interpolation_on_original_axis",
        )
        self.assertIsNone(payload["reference_database"]["version_hash"])

    def test_json_contains_stable_source_fingerprint_structure(self):
        payload = json.loads(matching_record_to_json(_result()).decode("utf-8"))
        database = payload["reference_database"]
        self.assertIsNone(database["version_hash"])
        self.assertEqual(
            database["source_fingerprint"]["npz_cache"],
            {
                "filename": "rruff_cache.npz",
                "relative_identifier": (
                    "project_relative:results/cache/rruff_cache.npz"
                ),
                "sha256": "a" * 64,
                "size_bytes": 1234,
            },
        )
        self.assertEqual(
            database["source_fingerprint"]["index_csv"]["sha256"],
            "b" * 64,
        )

    def test_csv_utf8_sig_and_json_preserve_chinese_and_special_characters(self):
        result = _result()
        csv_bytes = matching_candidates_to_csv(result)
        json_bytes = matching_record_to_json(result)
        self.assertTrue(csv_bytes.startswith(b"\xef\xbb\xbf"))
        self.assertIn("查询 光谱", csv_bytes.decode("utf-8-sig"))
        self.assertIn("石英 α", csv_bytes.decode("utf-8-sig"))
        self.assertIn("石英 α", json_bytes.decode("utf-8"))

    def test_unknown_metadata_is_explicit(self):
        result = _result()
        rows = list(
            csv.DictReader(
                StringIO(matching_candidates_to_csv(result).decode("utf-8-sig"))
            )
        )
        self.assertEqual(rows[1]["best_support_sample_id"], "Unknown")
        self.assertEqual(
            rows[1]["best_support_excitation_wavelength_nm"], "Unknown"
        )
        self.assertEqual(
            rows[1]["best_minus_second_distinct_sample_id_score"], "Unknown"
        )
        payload = json.loads(matching_record_to_json(result).decode("utf-8"))
        self.assertEqual(payload["candidates"][1]["best_support_sample_id"], "Unknown")
        self.assertEqual(
            payload["candidates"][1]["best_support_excitation_wavelength_nm"],
            "Unknown",
        )

    def test_stale_result_does_not_match_changed_file_or_topk(self):
        result = _result()
        self.assertTrue(
            matching_result_matches_request(result, QUERY_BYTES, result.filename, 3)
        )
        self.assertFalse(
            matching_result_matches_request(result, b"changed", result.filename, 3)
        )
        self.assertFalse(
            matching_result_matches_request(result, QUERY_BYTES, "other.txt", 3)
        )
        self.assertFalse(
            matching_result_matches_request(result, QUERY_BYTES, result.filename, 5)
        )

    def test_export_does_not_load_library_or_rerun_retrieval(self):
        result = _result()
        with (
            mock.patch("raman_app.matching.load_reference_library") as load_library,
            mock.patch("raman_app.matching.match_uploaded_spectrum") as match,
        ):
            matching_candidates_to_csv(result)
            matching_record_to_json(result)
        load_library.assert_not_called()
        match.assert_not_called()

    def test_download_serialization_does_not_recompute_file_hashes(self):
        result = _result()
        with mock.patch("raman_app.matching.sha256_file") as hasher:
            matching_candidates_to_csv(result)
            matching_record_to_json(result)
        hasher.assert_not_called()

    def test_repeated_current_source_downloads_do_not_reload_or_rehash(self):
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        ) as temporary:
            root = Path(temporary)
            cache = root / "cache.npz"
            index = root / "index.csv"
            cache.write_bytes(b"cache-v1")
            index.write_bytes(b"index-v1")
            with mock.patch(
                "raman_app.matching.load_reference_library",
                return_value=object(),
            ):
                bundle = load_reference_library_with_fingerprint(cache, index)
            result = replace(
                _result(), source_fingerprint=bundle.source_fingerprint
            )
            with (
                mock.patch(
                    "raman_app.matching.sha256_file",
                    wraps=sha256_file,
                ) as hasher,
                mock.patch("raman_app.matching.load_reference_library") as loader,
                mock.patch("raman_app.matching.match_uploaded_spectrum") as matcher,
            ):
                prepared_downloads = matching_exports_for_current_source(
                    result, cache, index
                )
                first = prepared_downloads
                second = prepared_downloads
        self.assertEqual(first, second)
        self.assertEqual(hasher.call_count, 2)
        loader.assert_not_called()
        matcher.assert_not_called()

    def test_source_hash_size_and_filename_match_independent_calculation(self):
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        ) as temporary:
            root = Path(temporary)
            cache = root / "actual_cache.npz"
            index = root / "actual_index.csv"
            cache_content = b"NPZ test bytes\x00\x01"
            index_content = "relative_path,标签\na.txt,石英\n".encode("utf-8")
            cache.write_bytes(cache_content)
            index.write_bytes(index_content)
            with mock.patch(
                "raman_app.matching.load_reference_library",
                return_value=object(),
            ):
                bundle = load_reference_library_with_fingerprint(cache, index)
        source = bundle.source_fingerprint
        self.assertEqual(source.npz_cache.filename, "actual_cache.npz")
        self.assertEqual(source.index_csv.filename, "actual_index.csv")
        self.assertEqual(source.npz_cache.size_bytes, len(cache_content))
        self.assertEqual(source.index_csv.size_bytes, len(index_content))
        self.assertEqual(
            source.npz_cache.sha256,
            hashlib.sha256(cache_content).hexdigest(),
        )
        self.assertEqual(
            source.index_csv.sha256,
            hashlib.sha256(index_content).hexdigest(),
        )
        self.assertNotIn(str(root), source.npz_cache.relative_identifier)

    def test_missing_or_changed_source_refuses_incomplete_fingerprint(self):
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        ) as temporary:
            root = Path(temporary)
            missing = root / "missing.npz"
            index = root / "index.csv"
            index.write_bytes(b"index")
            with self.assertRaisesRegex(ValueError, "metadata"):
                load_reference_library_with_fingerprint(missing, index)

            cache_with_missing_index = root / "cache-with-missing-index.npz"
            cache_with_missing_index.write_bytes(b"cache")
            with self.assertRaisesRegex(ValueError, "metadata"):
                load_reference_library_with_fingerprint(
                    cache_with_missing_index,
                    root / "missing-index.csv",
                )

            cache = root / "cache.npz"
            cache.write_bytes(b"before")

            def mutate_source(*_args):
                cache.write_bytes(b"changed-source-size")
                return object()

            with mock.patch(
                "raman_app.matching.load_reference_library",
                side_effect=mutate_source,
            ):
                with self.assertRaisesRegex(ValueError, "changed while"):
                    load_reference_library_with_fingerprint(cache, index)

            with (
                mock.patch(
                    "raman_app.matching.load_reference_library",
                    return_value=object(),
                ),
                mock.patch(
                    "raman_app.matching.sha256_file",
                    side_effect=ValueError("Unable to fingerprint reference source"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    load_reference_library_with_fingerprint(cache, index)

        incomplete = _result()
        object.__setattr__(incomplete, "source_fingerprint", None)
        with self.assertRaisesRegex(ValueError, "no validated"):
            matching_candidates_to_csv(incomplete)
        with self.assertRaisesRegex(ValueError, "no validated"):
            matching_record_to_json(incomplete)

    def test_npz_and_index_changes_invalidate_old_result_exports(self):
        for changed_name in ("cache.npz", "index.csv"):
            with self.subTest(changed_name=changed_name):
                with tempfile.TemporaryDirectory(
                    dir=Path(__file__).resolve().parent
                ) as temporary:
                    root = Path(temporary)
                    cache = root / "cache.npz"
                    index = root / "index.csv"
                    cache.write_bytes(b"cache-v1")
                    index.write_bytes(b"index-v1")
                    with mock.patch(
                        "raman_app.matching.load_reference_library",
                        return_value=object(),
                    ):
                        bundle = load_reference_library_with_fingerprint(cache, index)
                    result = replace(
                        _result(), source_fingerprint=bundle.source_fingerprint
                    )
                    matching_exports_for_current_source(result, cache, index)

                    changed_path = root / changed_name
                    changed_path.write_bytes(changed_path.read_bytes() + b"-changed")
                    with self.assertRaisesRegex(ValueError, "no longer matches"):
                        matching_exports_for_current_source(result, cache, index)

    def test_mtime_only_change_invalidates_old_result_export(self):
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        ) as temporary:
            root = Path(temporary)
            cache = root / "cache.npz"
            index = root / "index.csv"
            cache.write_bytes(b"same-size-cache")
            index.write_bytes(b"same-size-index")
            with mock.patch(
                "raman_app.matching.load_reference_library",
                return_value=object(),
            ):
                bundle = load_reference_library_with_fingerprint(cache, index)
            result = replace(
                _result(), source_fingerprint=bundle.source_fingerprint
            )
            before = cache.stat()
            os.utime(
                cache,
                ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
            )
            self.assertEqual(cache.stat().st_size, len(b"same-size-cache"))
            with self.assertRaisesRegex(ValueError, "no longer matches"):
                matching_exports_for_current_source(result, cache, index)

    def test_same_size_content_change_is_detected_with_changed_or_restored_mtime(self):
        for changed_name in ("cache.npz", "index.csv"):
            with self.subTest(changed_name=changed_name):
                with tempfile.TemporaryDirectory(
                    dir=Path(__file__).resolve().parent
                ) as temporary:
                    root = Path(temporary)
                    cache = root / "cache.npz"
                    index = root / "index.csv"
                    cache.write_bytes(b"cache-v1")
                    index.write_bytes(b"index-v1")
                    with mock.patch(
                        "raman_app.matching.load_reference_library",
                        return_value=object(),
                    ):
                        bundle = load_reference_library_with_fingerprint(cache, index)
                    result = replace(
                        _result(), source_fingerprint=bundle.source_fingerprint
                    )
                    changed_path = root / changed_name
                    original_stat = changed_path.stat()
                    replacement_content = (
                        b"cache-v2" if changed_name.endswith(".npz") else b"index-v2"
                    )
                    self.assertEqual(
                        len(replacement_content), original_stat.st_size
                    )
                    changed_path.write_bytes(replacement_content)
                    os.utime(
                        changed_path,
                        ns=(
                            original_stat.st_atime_ns,
                            original_stat.st_mtime_ns + 1_000_000_000,
                        ),
                    )
                    with self.assertRaisesRegex(ValueError, "no longer matches"):
                        matching_exports_for_current_source(result, cache, index)

                    os.utime(
                        changed_path,
                        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
                    )
                    with self.assertRaisesRegex(ValueError, "no longer matches"):
                        matching_exports_for_current_source(result, cache, index)

    def test_same_name_size_and_mtime_replacement_is_detected(self):
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        ) as temporary:
            root = Path(temporary)
            cache = root / "cache.npz"
            index = root / "index.csv"
            cache.write_bytes(b"cache-v1")
            index.write_bytes(b"index-v1")
            with mock.patch(
                "raman_app.matching.load_reference_library",
                return_value=object(),
            ):
                bundle = load_reference_library_with_fingerprint(cache, index)
            result = replace(
                _result(), source_fingerprint=bundle.source_fingerprint
            )
            original_stat = cache.stat()
            replacement = root / "replacement.tmp"
            replacement.write_bytes(b"cache-v2")
            os.utime(
                replacement,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            os.replace(replacement, cache)
            self.assertEqual(cache.stat().st_size, original_stat.st_size)
            self.assertEqual(cache.stat().st_mtime_ns, original_stat.st_mtime_ns)
            with self.assertRaisesRegex(ValueError, "no longer matches"):
                matching_exports_for_current_source(result, cache, index)

    def test_source_failure_then_recovery_requires_new_snapshot(self):
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        ) as temporary:
            root = Path(temporary)
            cache = root / "cache.npz"
            index = root / "index.csv"
            cache.write_bytes(b"cache-v1")
            index.write_bytes(b"index-v1")
            with mock.patch(
                "raman_app.matching.load_reference_library",
                return_value=object(),
            ):
                first_bundle = load_reference_library_with_fingerprint(cache, index)
            old_result = replace(
                _result(), source_fingerprint=first_bundle.source_fingerprint
            )

            index.unlink()
            with self.assertRaisesRegex(ValueError, "metadata") as failure:
                matching_exports_for_current_source(old_result, cache, index)
            self.assertNotIn(str(root), str(failure.exception))

            index.write_bytes(b"index-v2-restored")
            with mock.patch(
                "raman_app.matching.load_reference_library",
                side_effect=[ValueError("source validation failed"), object()],
            ):
                with self.assertRaisesRegex(ValueError, "validation failed"):
                    load_reference_library_with_fingerprint(cache, index)
                recovered_bundle = load_reference_library_with_fingerprint(cache, index)
            recovered_result = replace(
                _result(), source_fingerprint=recovered_bundle.source_fingerprint
            )
            csv_bytes, json_bytes = matching_exports_for_current_source(
                recovered_result, cache, index
            )
        self.assertTrue(csv_bytes.startswith(b"\xef\xbb\xbf"))
        self.assertIn(b'"source_fingerprint"', json_bytes)
        self.assertNotEqual(
            first_bundle.source_fingerprint.index_csv.sha256,
            recovered_bundle.source_fingerprint.index_csv.sha256,
        )

    def test_source_snapshot_requires_recorded_file_state(self):
        with self.assertRaisesRegex(ValueError, "no file state"):
            validate_reference_source_snapshot(
                _source_fingerprint(), Path("cache.npz"), Path("index.csv")
            )

    def test_source_snapshot_is_bound_to_matching_result(self):
        result = _result()
        replacement = _source_fingerprint("c" * 64, "d" * 64)
        object.__setattr__(result, "source_fingerprint", replacement)
        payload = json.loads(matching_record_to_json(result).decode("utf-8"))
        self.assertEqual(
            payload["reference_database"]["source_fingerprint"]["npz_cache"]["sha256"],
            "c" * 64,
        )

    def test_v1_peak_csv_export_remains_unchanged(self):
        features = [{
            "Peak_ID": 1,
            "Position_cm-1": 520.5,
            "Intensity": 42.0,
            "Prominence": 12.0,
            "Half_Prominence_Width_cm-1": 3.5,
            "Left_cm-1": 518.0,
            "Right_cm-1": 521.5,
            "Width_Height": 36.0,
        }]
        before = peak_features_to_csv(features)
        matching_candidates_to_csv(_result())
        matching_record_to_json(_result())
        self.assertEqual(peak_features_to_csv(features), before)

    def test_export_filenames_are_query_specific(self):
        self.assertEqual(
            build_matching_csv_filename("查询 光谱.txt"),
            "查询 光谱_raman_matching_candidates.csv",
        )
        self.assertEqual(
            build_matching_json_filename("查询 光谱.txt"),
            "查询 光谱_raman_matching_record.json",
        )


if __name__ == "__main__":
    unittest.main()
