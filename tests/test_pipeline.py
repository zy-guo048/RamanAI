"""Tests for the uploaded-spectrum analysis pipeline."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from raman_app.pipeline import analyze_uploaded_spectrum
from raman_app.upload import parse_uploaded_spectrum
from scripts.raman_utils import (
    baseline_correct_spectrum,
    denoise_spectrum,
    detect_peaks,
    extract_peak_features,
    normalize_spectrum,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_FILE = PROJECT_DIR / "data" / "raw" / "sample_raw.txt"

HISTORICAL_PEAK_POSITIONS = np.array([
    267.406,
    388.149,
    534.344,
    667.586,
    714.981,
    765.538,
    934.406,
    1018.056,
])
HISTORICAL_HALF_PROMINENCE_WIDTHS = np.array([
    18.20956778083155,
    81.21704676785515,
    69.20446559410311,
    133.08914158668335,
    18.04175173883084,
    30.37985309881867,
    74.32859379459626,
    94.10453182412891,
])


def make_peak_spectrum_bytes(peak_scale=8000.0):
    raman_shift = np.arange(100.0, 501.0)
    intensity = 200.0 + 0.2 * raman_shift
    for center, width, relative_height in (
        (180.0, 5.0, 1.0),
        (290.0, 8.0, 0.8),
        (410.0, 6.0, 1.2),
    ):
        intensity += (
            peak_scale
            * relative_height
            * np.exp(-0.5 * ((raman_shift - center) / width) ** 2)
        )
    rows = (
        f"{shift:.1f},{value:.12f}"
        for shift, value in zip(raman_shift, intensity)
    )
    return ("##NAMES=Synthetic\n" + "\n".join(rows) + "\n").encode("utf-8")


class AnalyzeUploadedSpectrumTests(unittest.TestCase):
    def test_valid_spectrum_completes_full_pipeline(self):
        result = analyze_uploaded_spectrum(
            make_peak_spectrum_bytes(),
            filename="synthetic_raw.txt",
        )

        self.assertEqual(result.filename, "synthetic_raw.txt")
        self.assertEqual(result.point_count, 401)
        self.assertGreater(result.peak_count, 0)
        self.assertAlmostEqual(
            np.max(result.normalized_spectrum.spectral_data),
            1.0,
        )

    def test_all_output_spectral_axes_match_input(self):
        result = analyze_uploaded_spectrum(make_peak_spectrum_bytes())
        expected_axis = result.raw_spectrum.spectral_axis

        for spectrum in (
            result.denoised_spectrum,
            result.corrected_spectrum,
            result.normalized_spectrum,
        ):
            with self.subTest(spectrum=spectrum):
                np.testing.assert_array_equal(
                    spectrum.spectral_axis,
                    expected_axis,
                )

    def test_parameters_match_stable_historical_settings(self):
        parameters = analyze_uploaded_spectrum(
            make_peak_spectrum_bytes()
        ).parameters

        self.assertEqual(parameters.sg_window_length, 21)
        self.assertEqual(parameters.sg_polyorder, 3)
        self.assertEqual(parameters.asls_lam, 1e6)
        self.assertEqual(parameters.asls_p, 0.01)
        self.assertEqual(parameters.peak_prominence, 1000)
        self.assertEqual(parameters.peak_distance, 10)
        self.assertEqual(parameters.peak_distance_unit, "samples")
        self.assertEqual(parameters.peak_width_method, "half_prominence")
        self.assertEqual(parameters.peak_width_rel_height, 0.5)
        self.assertEqual(parameters.normalization_method, "max")

    @unittest.skipUnless(SAMPLE_FILE.is_file(), "本地示例 RAW 文件不可用")
    def test_custom_prominence_changes_peak_detection(self):
        file_bytes = SAMPLE_FILE.read_bytes()

        default_result = analyze_uploaded_spectrum(file_bytes)
        lower_threshold_result = analyze_uploaded_spectrum(
            file_bytes,
            prominence=100,
        )

        self.assertEqual(default_result.peak_count, 8)
        self.assertEqual(default_result.parameters.peak_prominence, 1000)
        self.assertEqual(lower_threshold_result.peak_count, 17)
        self.assertEqual(lower_threshold_result.parameters.peak_prominence, 100)

    def test_low_intensity_spectrum_accepts_lower_prominence(self):
        file_bytes = make_peak_spectrum_bytes(peak_scale=100.0)

        default_result = analyze_uploaded_spectrum(file_bytes)
        lower_threshold_result = analyze_uploaded_spectrum(
            file_bytes,
            prominence=10,
        )

        self.assertEqual(default_result.peak_count, 0)
        self.assertGreater(lower_threshold_result.peak_count, 0)
        self.assertLess(
            np.max(lower_threshold_result.corrected_spectrum.spectral_data),
            120,
        )

    def test_invalid_prominence_is_rejected(self):
        file_bytes = make_peak_spectrum_bytes()

        for invalid_value in (-1, np.nan, np.inf, -np.inf):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaisesRegex(ValueError, "prominence"):
                    analyze_uploaded_spectrum(
                        file_bytes,
                        prominence=invalid_value,
                    )

        for invalid_value in (True, "100"):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaisesRegex(TypeError, "prominence"):
                    analyze_uploaded_spectrum(
                        file_bytes,
                        prominence=invalid_value,
                    )

    def test_pipeline_matches_direct_tool_calls(self):
        file_bytes = make_peak_spectrum_bytes()
        result = analyze_uploaded_spectrum(file_bytes)

        raw = parse_uploaded_spectrum(file_bytes)
        denoised = denoise_spectrum(raw, window_length=21, polyorder=3)
        corrected = baseline_correct_spectrum(denoised, lam=1e6, p=0.01)
        peaks, properties = detect_peaks(
            corrected,
            prominence=1000,
            distance=10,
        )
        features = extract_peak_features(corrected, peaks, properties)
        normalized = normalize_spectrum(corrected, method="max")

        np.testing.assert_allclose(
            result.denoised_spectrum.spectral_data,
            denoised.spectral_data,
        )
        np.testing.assert_allclose(
            result.corrected_spectrum.spectral_data,
            corrected.spectral_data,
        )
        np.testing.assert_array_equal(result.peak_indices, peaks)
        self.assertEqual(result.peak_features, features)
        np.testing.assert_allclose(
            result.normalized_spectrum.spectral_data,
            normalized.spectral_data,
        )

    def test_peak_detection_uses_corrected_not_normalized_spectrum(self):
        with patch(
            "raman_app.pipeline.detect_peaks",
            wraps=detect_peaks,
        ) as mocked_detect_peaks:
            result = analyze_uploaded_spectrum(make_peak_spectrum_bytes())

        detected_spectrum = mocked_detect_peaks.call_args.args[0]
        self.assertIs(detected_spectrum, result.corrected_spectrum)
        self.assertIsNot(detected_spectrum, result.normalized_spectrum)

        normalized_peaks, _ = detect_peaks(
            result.normalized_spectrum,
            prominence=1000,
            distance=10,
        )
        self.assertGreater(result.peak_count, 0)
        self.assertEqual(len(normalized_peaks), 0)

    def test_peak_feature_count_matches_peak_count(self):
        result = analyze_uploaded_spectrum(make_peak_spectrum_bytes())

        self.assertEqual(result.peak_count, len(result.peak_indices))
        self.assertEqual(result.peak_count, len(result.peak_features))

    def test_zero_peaks_returns_empty_features(self):
        result = analyze_uploaded_spectrum(
            make_peak_spectrum_bytes(peak_scale=50.0)
        )

        self.assertEqual(result.peak_count, 0)
        self.assertEqual(result.peak_indices.tolist(), [])
        self.assertEqual(result.peak_features, [])

    def test_invalid_upload_and_filename_raise_errors(self):
        with self.assertRaisesRegex(ValueError, "上传文件为空"):
            analyze_uploaded_spectrum(b"")
        with self.assertRaisesRegex(ValueError, "扩展名为 .txt"):
            analyze_uploaded_spectrum(
                make_peak_spectrum_bytes(),
                filename="spectrum.csv",
            )
        with self.assertRaisesRegex(ValueError, "不能包含路径"):
            analyze_uploaded_spectrum(
                make_peak_spectrum_bytes(),
                filename="../spectrum.txt",
            )

    def test_pipeline_does_not_create_result_files(self):
        original_directory = Path.cwd()
        with TemporaryDirectory(dir=PROJECT_DIR / "tests") as temporary_directory:
            temporary_path = Path(temporary_directory)
            before = set(temporary_path.iterdir())
            try:
                os.chdir(temporary_path)
                analyze_uploaded_spectrum(make_peak_spectrum_bytes())
            finally:
                os.chdir(original_directory)
            after = set(temporary_path.iterdir())

        self.assertEqual(after, before)

    @unittest.skipUnless(SAMPLE_FILE.is_file(), "本地示例 RAW 文件不可用")
    def test_local_sample_matches_historical_peak_results(self):
        result = analyze_uploaded_spectrum(
            SAMPLE_FILE.read_bytes(),
            filename=SAMPLE_FILE.name,
        )

        positions = np.array([
            feature["Position_cm-1"] for feature in result.peak_features
        ])
        widths = np.array([
            feature["Half_Prominence_Width_cm-1"]
            for feature in result.peak_features
        ])

        self.assertEqual(result.peak_count, 8)
        np.testing.assert_allclose(
            positions,
            HISTORICAL_PEAK_POSITIONS,
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            widths,
            HISTORICAL_HALF_PROMINENCE_WIDTHS,
            rtol=0,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
