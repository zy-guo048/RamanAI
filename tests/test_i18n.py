"""Tests for RamanAI bilingual interface behavior."""

import csv
from io import StringIO
import unittest
from unittest.mock import patch

import numpy as np

import app
from raman_app.export import PEAK_FEATURE_COLUMNS, peak_features_to_csv
from raman_app.i18n import (
    DEFAULT_LANGUAGE,
    LANGUAGE_LABELS,
    SUPPORTED_LANGUAGES,
    TRANSLATIONS,
    normalize_language,
    translate,
    translate_error,
)
from raman_app.pipeline import analyze_uploaded_spectrum


def make_spectrum_bytes() -> bytes:
    raman_shift = np.arange(100.0, 501.0)
    intensity = 200.0 + 0.2 * raman_shift
    for center, width, relative_height in (
        (180.0, 5.0, 1.0),
        (290.0, 8.0, 0.8),
        (410.0, 6.0, 1.2),
    ):
        intensity += (
            8000.0
            * relative_height
            * np.exp(-0.5 * ((raman_shift - center) / width) ** 2)
        )
    rows = (
        f"{shift:.1f},{value:.12f}"
        for shift, value in zip(raman_shift, intensity)
    )
    return ("##NAMES=Bilingual Test\n" + "\n".join(rows) + "\n").encode(
        "utf-8"
    )


class InternationalizationTests(unittest.TestCase):
    def test_simplified_chinese_and_english_are_available(self):
        self.assertEqual(SUPPORTED_LANGUAGES, ("zh_CN", "en"))
        self.assertEqual(DEFAULT_LANGUAGE, "zh_CN")
        self.assertEqual(set(LANGUAGE_LABELS), set(SUPPORTED_LANGUAGES))

    def test_all_translation_keys_have_nonempty_content_in_both_languages(self):
        expected_keys = set(TRANSLATIONS[DEFAULT_LANGUAGE])

        for language in SUPPORTED_LANGUAGES:
            with self.subTest(language=language):
                self.assertEqual(set(TRANSLATIONS[language]), expected_keys)
                self.assertTrue(
                    all(
                        isinstance(value, str) and value.strip()
                        for value in TRANSLATIONS[language].values()
                    )
                )

    def test_invalid_language_code_falls_back_to_simplified_chinese(self):
        for invalid_language in (None, "", "fr", 123):
            with self.subTest(invalid_language=invalid_language):
                self.assertEqual(
                    normalize_language(invalid_language),
                    DEFAULT_LANGUAGE,
                )
                self.assertEqual(
                    translate(invalid_language, "app_title"),
                    TRANSLATIONS[DEFAULT_LANGUAGE]["app_title"],
                )

    def test_known_input_errors_are_translated(self):
        chinese_message = "上传文件为空"

        self.assertEqual(
            translate_error("zh_CN", chinese_message),
            "上传文件为空。",
        )
        self.assertEqual(
            translate_error("en", chinese_message),
            "The uploaded file is empty.",
        )

    def test_raman_shift_axis_preserves_matplotlib_mathtext(self):
        for language in SUPPORTED_LANGUAGES:
            with self.subTest(language=language):
                self.assertEqual(
                    translate(language, "raman_shift_axis"),
                    "Raman Shift (cm$^{-1}$)",
                )

    def test_language_switch_preserves_analysis_and_widget_state(self):
        analysis_result = object()
        request_key = ("sample.txt", "digest", "float:1000.0")
        session_state = {
            app.LANGUAGE_STATE_KEY: "zh_CN",
            app.LANGUAGE_SELECTOR_STATE_KEY: "en",
            app.ANALYSIS_RESULT_STATE_KEY: analysis_result,
            app.ANALYSIS_REQUEST_STATE_KEY: request_key,
            app.MAIN_VIEW_STATE_KEY: translate("zh_CN", "matching_view"),
            "peak_prominence": 1000.0,
            "spectrum_upload": object(),
        }

        with (
            patch.object(app.st, "session_state", session_state),
            patch.object(app.st, "rerun") as mocked_rerun,
        ):
            app._apply_language_selection()

        self.assertEqual(session_state[app.LANGUAGE_STATE_KEY], "en")
        self.assertIs(
            session_state[app.ANALYSIS_RESULT_STATE_KEY],
            analysis_result,
        )
        self.assertEqual(
            session_state[app.ANALYSIS_REQUEST_STATE_KEY],
            request_key,
        )
        self.assertEqual(session_state["peak_prominence"], 1000.0)
        self.assertIn("spectrum_upload", session_state)
        self.assertEqual(
            session_state[app.MAIN_VIEW_STATE_KEY],
            translate("en", "matching_view"),
        )
        mocked_rerun.assert_called_once_with(scope="app")

    def test_language_does_not_change_scientific_results(self):
        file_bytes = make_spectrum_bytes()
        results = {
            language: analyze_uploaded_spectrum(
                file_bytes,
                filename="bilingual_test.txt",
            )
            for language in SUPPORTED_LANGUAGES
        }
        chinese_result = results["zh_CN"]
        english_result = results["en"]

        np.testing.assert_array_equal(
            chinese_result.peak_indices,
            english_result.peak_indices,
        )
        np.testing.assert_allclose(
            chinese_result.corrected_spectrum.spectral_data,
            english_result.corrected_spectrum.spectral_data,
        )
        np.testing.assert_allclose(
            chinese_result.normalized_spectrum.spectral_data,
            english_result.normalized_spectrum.spectral_data,
        )
        self.assertEqual(
            chinese_result.peak_features,
            english_result.peak_features,
        )
        self.assertEqual(chinese_result.parameters, english_result.parameters)

    def test_csv_fields_and_values_are_language_independent(self):
        result = analyze_uploaded_spectrum(
            make_spectrum_bytes(),
            filename="bilingual_test.txt",
        )
        exports = {
            language: peak_features_to_csv(result.peak_features)
            for language in SUPPORTED_LANGUAGES
        }

        self.assertEqual(exports["zh_CN"], exports["en"])
        rows = list(
            csv.DictReader(
                StringIO(exports["en"].decode("utf-8-sig"))
            )
        )
        self.assertEqual(tuple(rows[0]), PEAK_FEATURE_COLUMNS)
        self.assertEqual(len(rows), result.peak_count)


if __name__ == "__main__":
    unittest.main()
