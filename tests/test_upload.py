"""Tests for in-memory RRUFF spectrum upload parsing."""

from pathlib import Path
import unittest

import numpy as np

from raman_app.upload import (
    MAX_UPLOAD_SIZE_BYTES,
    parse_uploaded_spectrum,
)
from scripts.raman_utils import load_spectrum


PROJECT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_FILE = PROJECT_DIR / "data" / "raw" / "sample_raw.txt"


def make_spectrum_bytes(point_count=21, include_comment=False):
    lines = []
    if include_comment:
        lines.extend(["##NAMES=Test", "# synthetic RRUFF spectrum"])
    lines.extend(
        f"{100 + index:.1f},{1000 + index * 10:.1f}"
        for index in range(point_count)
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


class ParseUploadedSpectrumTests(unittest.TestCase):
    def test_parses_valid_two_column_spectrum(self):
        spectrum = parse_uploaded_spectrum(make_spectrum_bytes())

        expected_axis = np.arange(100.0, 121.0)
        expected_intensity = np.arange(1000.0, 1210.0, 10.0)
        np.testing.assert_array_equal(spectrum.spectral_axis, expected_axis)
        np.testing.assert_array_equal(
            spectrum.spectral_data,
            expected_intensity,
        )

    def test_ignores_hash_comment_lines(self):
        spectrum = parse_uploaded_spectrum(
            make_spectrum_bytes(include_comment=True)
        )

        self.assertEqual(len(spectrum.spectral_axis), 21)
        self.assertEqual(spectrum.spectral_axis[0], 100.0)

    def test_rejects_empty_file(self):
        for content in (b"", b" \r\n\t"):
            with self.subTest(content=content):
                with self.assertRaisesRegex(ValueError, "上传文件为空"):
                    parse_uploaded_spectrum(content)

    def test_rejects_comment_only_file(self):
        with self.assertRaisesRegex(ValueError, "不包含光谱数据"):
            parse_uploaded_spectrum(b"##NAMES=Empty\n# no numeric rows\n")

    def test_rejects_wrong_column_count(self):
        rows = "\n".join(
            f"{index},{index + 1},{index + 2}" for index in range(21)
        ).encode("utf-8")

        with self.assertRaisesRegex(ValueError, "两列"):
            parse_uploaded_spectrum(rows)

    def test_rejects_non_numeric_data(self):
        content = make_spectrum_bytes().replace(b"110.0,1100.0", b"bad,data")

        with self.assertRaisesRegex(ValueError, "格式错误"):
            parse_uploaded_spectrum(content)

    def test_rejects_nan_and_inf(self):
        for invalid_value in (b"nan", b"inf", b"-inf"):
            with self.subTest(invalid_value=invalid_value):
                content = make_spectrum_bytes().replace(
                    b"110.0,1100.0",
                    b"110.0," + invalid_value,
                )
                with self.assertRaisesRegex(ValueError, "NaN 或 Inf"):
                    parse_uploaded_spectrum(content)

    def test_rejects_duplicate_raman_shift(self):
        lines = make_spectrum_bytes().decode("utf-8").splitlines()
        lines[10] = "109.0,1100.0"

        with self.assertRaisesRegex(ValueError, "严格递增"):
            parse_uploaded_spectrum("\n".join(lines).encode("utf-8"))

    def test_rejects_descending_raman_shift(self):
        rows = "\n".join(
            f"{120 - index:.1f},{1000 + index * 10:.1f}"
            for index in range(21)
        ).encode("utf-8")

        with self.assertRaisesRegex(ValueError, "严格递增"):
            parse_uploaded_spectrum(rows)

    def test_rejects_fewer_than_21_points(self):
        with self.assertRaisesRegex(ValueError, "至少需要 21 个采样点"):
            parse_uploaded_spectrum(make_spectrum_bytes(point_count=20))

    def test_rejects_file_over_size_limit(self):
        oversized_content = b"0" * (MAX_UPLOAD_SIZE_BYTES + 1)

        with self.assertRaisesRegex(ValueError, "超过 5 MB"):
            parse_uploaded_spectrum(oversized_content)

    @unittest.skipUnless(SAMPLE_FILE.is_file(), "本地示例 RAW 文件不可用")
    def test_matches_existing_loader_for_local_sample(self):
        uploaded = parse_uploaded_spectrum(SAMPLE_FILE.read_bytes())
        existing = load_spectrum(SAMPLE_FILE)

        np.testing.assert_array_equal(
            uploaded.spectral_axis,
            existing.spectral_axis,
        )
        np.testing.assert_array_equal(
            uploaded.spectral_data,
            existing.spectral_data,
        )


if __name__ == "__main__":
    unittest.main()
