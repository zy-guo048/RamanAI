from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import app
from raman_app import matching
from raman_app.i18n import SUPPORTED_LANGUAGES, TRANSLATIONS, translate_matching_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _spectrum_bytes(start: float = 250.0, stop: float = 900.0) -> bytes:
    x = np.arange(start, stop + 1.0, 1.0)
    y = 250.0 + 0.03 * x
    y += 4000.0 * np.exp(-0.5 * ((x - 520.0) / 12.0) ** 2)
    y += 2800.0 * np.exp(-0.5 * ((x - 710.0) / 18.0) ** 2)
    return ("# synthetic matching test\n" + "\n".join(
        f"{shift:.6f},{intensity:.12f}" for shift, intensity in zip(x, y)
    ) + "\n").encode("utf-8")


def _record(
    path: str,
    sample_id: str,
    label: str,
    wavelength: str,
) -> dict[str, str]:
    return {
        "relative_path": path,
        "file_name": Path(path).name,
        "measurement_stem": Path(path).stem,
        "sample_id": sample_id,
        "sample_id_source": "filename_header_match",
        "reference_label": label,
        "reference_label_status": "available",
        "reference_candidate": "True",
        "excitation_wavelength_nm": wavelength,
        "excitation_wavelength_source": "header_and_filename_agree",
        "requires_review": "False",
        "review_reasons": "",
        "audit_metadata_warnings": "",
        "has_missing_metadata": "False",
        "has_conflict": "False",
        "metadata_status": "complete_no_flag",
        "metadata_flags": "",
    }


def _library():
    records = (
        _record("a-s1-one.txt", "A1", "Alpha", "532"),
        _record("a-s1-two.txt", "A1", "Alpha", "633"),
        _record("a-s2.txt", "A2", "Alpha", "785"),
        _record("b-s1.txt", "B1", "Beta", "514"),
        _record("c-s1.txt", "C1", "Gamma", "458"),
    )
    matrix = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return matching.exp10.ReferenceLibrary(
        grid=np.asarray([300.0, 300.5, 301.0]),
        all_success_matrix=matrix,
        all_success_records=records,
        reference_matrix=matrix,
        reference_records=records,
        path_to_success_row={row["relative_path"]: index for index, row in enumerate(records)},
    )


class SpectrumMatchingUiTests(unittest.TestCase):
    def test_reference_source_cache_identity_changes_for_npz_and_index(self):
        for changed_name in ("cache.npz", "index.csv"):
            with self.subTest(changed_name=changed_name):
                with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
                    root = Path(temporary)
                    cache = root / "cache.npz"
                    index = root / "index.csv"
                    cache.write_bytes(b"cache-v1")
                    index.write_bytes(b"index-v1")

                    def build_bundle(*arguments):
                        with mock.patch.object(
                            matching,
                            "load_reference_library",
                            return_value=_library(),
                        ):
                            return matching.load_reference_library_with_fingerprint(
                                Path(arguments[0]), Path(arguments[1])
                            )

                    with (
                        mock.patch.object(app, "DEFAULT_CACHE_PATH", cache),
                        mock.patch.object(app, "DEFAULT_INDEX_PATH", index),
                        mock.patch.object(
                            app,
                            "_load_matching_library_cached",
                            side_effect=build_bundle,
                        ) as cached_loader,
                    ):
                        first = app._get_matching_library()
                        changed_path = root / changed_name
                        changed_path.write_bytes(
                            changed_path.read_bytes() + b"-changed"
                        )
                        second = app._get_matching_library()

                self.assertEqual(cached_loader.call_count, 2)
                self.assertNotEqual(
                    cached_loader.call_args_list[0].args[2:],
                    cached_loader.call_args_list[1].args[2:],
                )
                first_file = (
                    first.source_fingerprint.npz_cache
                    if changed_name.endswith(".npz")
                    else first.source_fingerprint.index_csv
                )
                second_file = (
                    second.source_fingerprint.npz_cache
                    if changed_name.endswith(".npz")
                    else second.source_fingerprint.index_csv
                )
                self.assertNotEqual(first_file.sha256, second_file.sha256)

    def test_stale_source_clears_result_and_hides_downloads(self):
        file_bytes = b"query"
        filename = "query.txt"
        query = matching.exp10.QuerySpectrum(
            vector=np.asarray([1.0, 0.0, 0.0]),
            mode="test",
            relative_path="external:q.txt",
        )
        with mock.patch.object(
            matching, "query_from_uploaded_bytes", return_value=query
        ):
            result = matching.match_uploaded_spectrum(
                file_bytes, filename, 3, _library()
            )
        uploaded = mock.Mock()
        uploaded.name = filename
        uploaded.getvalue.return_value = file_bytes
        session_state = {
            app.MATCHING_RESULT_STATE_KEY: result,
            app.MATCHING_REQUEST_STATE_KEY: app._build_matching_request_key(
                file_bytes, filename, 3
            ),
        }
        secret_path = str(PROJECT_ROOT / "private" / "cache.npz")
        with (
            mock.patch.object(app.st, "session_state", session_state),
            mock.patch.object(app.st, "file_uploader", return_value=uploaded),
            mock.patch.object(app.st, "segmented_control", return_value=3),
            mock.patch.object(app.st, "button", return_value=False),
            mock.patch.object(app.st, "info"),
            mock.patch.object(app.st, "markdown"),
            mock.patch.object(app.st, "error") as show_error,
            mock.patch.object(
                app,
                "matching_exports_for_current_source",
                side_effect=ValueError(
                    f"NPZ reference source changed: {secret_path}"
                ),
            ),
            mock.patch.object(app, "_render_matching_result") as render_result,
        ):
            app._render_matching_page("en")
        self.assertIsNone(session_state[app.MATCHING_RESULT_STATE_KEY])
        self.assertIsNone(session_state[app.MATCHING_REQUEST_STATE_KEY])
        render_result.assert_not_called()
        show_error.assert_called_once()
        self.assertNotIn(secret_path, show_error.call_args.args[0])

    def test_content_hash_changes_cache_key_when_size_and_mtime_are_restored(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            cache = root / "cache.npz"
            index = root / "index.csv"
            cache.write_bytes(b"cache-v1")
            index.write_bytes(b"index-v1")

            def build_bundle(*arguments):
                with mock.patch.object(
                    matching,
                    "load_reference_library",
                    return_value=_library(),
                ):
                    return matching.load_reference_library_with_fingerprint(
                        Path(arguments[0]), Path(arguments[1])
                    )

            with (
                mock.patch.object(app, "DEFAULT_CACHE_PATH", cache),
                mock.patch.object(app, "DEFAULT_INDEX_PATH", index),
                mock.patch.object(
                    app,
                    "_load_matching_library_cached",
                    side_effect=build_bundle,
                ) as cached_loader,
            ):
                first = app._get_matching_library()
                original_stat = cache.stat()
                cache.write_bytes(b"cache-v2")
                os.utime(
                    cache,
                    ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
                )
                second = app._get_matching_library()

        self.assertEqual(cached_loader.call_count, 2)
        first_key = cached_loader.call_args_list[0].args
        second_key = cached_loader.call_args_list[1].args
        self.assertEqual(first_key[2:6], second_key[2:6])
        self.assertNotEqual(first_key[6], second_key[6])
        self.assertNotEqual(
            first.source_fingerprint.npz_cache.sha256,
            second.source_fingerprint.npz_cache.sha256,
        )

    def test_stale_cached_library_is_rejected_then_current_source_recovers(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            cache = root / "cache.npz"
            index = root / "index.csv"
            cache.write_bytes(b"cache-v1")
            index.write_bytes(b"index-v1")
            with mock.patch.object(
                matching,
                "load_reference_library",
                return_value=_library(),
            ):
                stale_bundle = matching.load_reference_library_with_fingerprint(
                    cache, index
                )
            original_stat = cache.stat()
            cache.write_bytes(b"cache-v2")
            os.utime(
                cache,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            with mock.patch.object(
                matching,
                "load_reference_library",
                return_value=_library(),
            ):
                current_bundle = matching.load_reference_library_with_fingerprint(
                    cache, index
                )

            with (
                mock.patch.object(app, "DEFAULT_CACHE_PATH", cache),
                mock.patch.object(app, "DEFAULT_INDEX_PATH", index),
                mock.patch.object(
                    app,
                    "_load_matching_library_cached",
                    return_value=stale_bundle,
                ) as stale_cache,
            ):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    app._get_matching_library()
                stale_cache.clear.assert_called_once()

            with (
                mock.patch.object(app, "DEFAULT_CACHE_PATH", cache),
                mock.patch.object(app, "DEFAULT_INDEX_PATH", index),
                mock.patch.object(
                    app,
                    "_load_matching_library_cached",
                    return_value=current_bundle,
                ),
            ):
                recovered = app._get_matching_library()
        self.assertEqual(recovered.source_fingerprint, current_bundle.source_fingerprint)

    def test_uploaded_query_calls_validated_v1_preprocessing_parser_flow(self):
        original = matching.exp10.exp09.parse_uploaded_spectrum
        with mock.patch.object(
            matching.exp10.exp09,
            "parse_uploaded_spectrum",
            wraps=original,
        ) as parser:
            query = matching.query_from_uploaded_bytes(
                _spectrum_bytes(), "uploaded.txt"
            )
        parser.assert_called_once()
        self.assertEqual(query.vector.shape, (1101,))
        self.assertEqual(query.sample_id, "Unknown")

    def test_uploaded_preprocessing_matches_experiment_10_external_raw(self):
        content = _spectrum_bytes()
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            path = Path(temporary) / "query.txt"
            path.write_bytes(content)
            existing = matching.exp10.query_from_external_raw(path)
            uploaded = matching.query_from_uploaded_bytes(content, path.name)
        np.testing.assert_array_equal(uploaded.vector, existing.vector)

    def test_topk_candidates_match_existing_reference_label_api(self):
        library = _library()
        query = matching.exp10.QuerySpectrum(
            vector=np.asarray([1.0, 0.0, 0.0]),
            mode="test",
            relative_path="external:q.txt",
        )
        expected = matching.exp10.search_topk(
            library,
            query,
            3,
            result_level="reference_label",
            exclude_self=False,
            exclude_same_sample=False,
        )
        with mock.patch.object(matching, "query_from_uploaded_bytes", return_value=query):
            actual = matching.match_uploaded_spectrum(b"unused", "q.txt", 3, library)
        self.assertEqual(
            [row.reference_label for row in actual.candidates],
            [row["reference_label_group_display"] for row in expected],
        )
        np.testing.assert_allclose(
            [row.cosine_similarity for row in actual.candidates],
            [row["cosine_similarity"] for row in expected],
        )

    def test_candidate_counts_and_independent_identity_gap_are_correct(self):
        library = _library()
        query = matching.exp10.QuerySpectrum(
            vector=np.asarray([1.0, 0.0, 0.0]),
            mode="test",
            relative_path="external:q.txt",
        )
        with mock.patch.object(matching, "query_from_uploaded_bytes", return_value=query):
            result = matching.match_uploaded_spectrum(b"unused", "q.txt", 3, library)
        alpha = result.candidates[0]
        self.assertEqual(alpha.reference_file_count, 3)
        self.assertEqual(alpha.distinguishable_identity_count, 2)
        self.assertEqual(len(alpha.supporting_identities), 2)
        self.assertAlmostEqual(alpha.best_minus_second_identity_score, 1.0 - (0.8 / np.sqrt(0.68)))

    def test_best_evidence_path_sample_and_wavelength_are_mapped(self):
        library = _library()
        query = matching.exp10.QuerySpectrum(
            vector=np.asarray([1.0, 0.0, 0.0]),
            mode="test",
            relative_path="external:q.txt",
        )
        with mock.patch.object(matching, "query_from_uploaded_bytes", return_value=query):
            result = matching.match_uploaded_spectrum(b"unused", "q.txt", 3, library)
        best = result.candidates[0]
        self.assertEqual(best.best_relative_path, "a-s1-one.txt")
        self.assertEqual(best.best_sample_id, "A1")
        self.assertEqual(best.best_excitation_wavelength_nm, "532")
        np.testing.assert_array_equal(best.best_reference_vector, [1.0, 0.0, 0.0])

    def test_incomplete_target_band_is_rejected_without_extrapolation(self):
        with self.assertRaisesRegex(ValueError, "without extrapolation"):
            matching.query_from_uploaded_bytes(
                _spectrum_bytes(400.0, 900.0), "incomplete.txt"
            )
        self.assertIn(
            "300–850",
            translate_matching_error(
                "zh_CN", "Uploaded RAW cannot cover the grid without extrapolation"
            ),
        )

    def test_missing_cache_and_identity_mapping_errors_are_explicit(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "NPZ cache does not exist"):
                matching.load_reference_library(root / "missing.npz", root / "missing.csv")
        with mock.patch.object(
            matching.exp10,
            "load_reference_library",
            side_effect=ValueError("Index and NPZ relative_path identity order do not match"),
        ):
            with self.assertRaisesRegex(ValueError, "identity order"):
                matching.load_reference_library(Path("cache.npz"), Path("index.csv"))

    def test_matching_does_not_modify_frozen_evaluation_manifest(self):
        manifest = PROJECT_ROOT / "docs" / "evaluation" / "exp11_v1" / "evaluation_group_manifest.csv"
        before = hashlib.sha256(manifest.read_bytes()).hexdigest()
        library = _library()
        query = matching.exp10.QuerySpectrum(
            vector=np.asarray([1.0, 0.0, 0.0]), mode="test", relative_path="external:q.txt"
        )
        with mock.patch.object(matching, "query_from_uploaded_bytes", return_value=query):
            matching.match_uploaded_spectrum(b"unused", "q.txt", 3, library)
        after = hashlib.sha256(manifest.read_bytes()).hexdigest()
        self.assertEqual(before, after)

    def test_bilingual_matching_copy_is_complete_and_accessible(self):
        required = {
            "matching_view",
            "matching_upload",
            "start_matching",
            "matching_candidates",
            "cosine_similarity",
            "matching_scientific_note",
            "matching_downloads",
            "download_matching_csv",
            "download_matching_json",
            "matching_download_note",
            "matching_error_band",
            "matching_error_cache",
        }
        for language in SUPPORTED_LANGUAGES:
            with self.subTest(language=language):
                self.assertTrue(required <= set(TRANSLATIONS[language]))
                self.assertTrue(all(TRANSLATIONS[language][key].strip() for key in required))

    def test_matching_request_identity_changes_with_file_or_topk(self):
        original = app._build_matching_request_key(b"one", "sample.txt", 5)
        self.assertNotEqual(
            original,
            app._build_matching_request_key(b"two", "sample.txt", 5),
        )
        self.assertNotEqual(
            original,
            app._build_matching_request_key(b"one", "sample.txt", 10),
        )

    def test_matching_figure_contains_query_reference_and_overlay(self):
        library = _library()
        query = matching.exp10.QuerySpectrum(
            vector=np.asarray([1.0, 0.0, 0.0]),
            mode="test",
            relative_path="external:q.txt",
        )
        with mock.patch.object(matching, "query_from_uploaded_bytes", return_value=query):
            result = matching.match_uploaded_spectrum(b"unused", "q.txt", 3, library)
        figure = app._create_matching_figure(result, "en")
        try:
            self.assertEqual(len(figure.axes), 3)
            self.assertEqual(len(figure.axes[2].lines), 2)
        finally:
            app.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
