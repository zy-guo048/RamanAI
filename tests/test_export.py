"""Tests for in-memory candidate-peak CSV export."""

import codecs
import csv
from io import StringIO
import unittest

from raman_app.export import (
    PEAK_FEATURE_COLUMNS,
    build_peak_csv_filename,
    peak_features_to_csv,
)


class PeakFeatureExportTests(unittest.TestCase):
    def test_exports_all_peak_fields_as_utf8_sig(self):
        feature = {
            "Peak_ID": 1,
            "Position_cm-1": 267.406,
            "Intensity": 3028.1,
            "Prominence": 2035.4,
            "Half_Prominence_Width_cm-1": 18.2095,
            "Left_cm-1": 258.1,
            "Right_cm-1": 276.3,
            "Width_Height": 2010.2,
        }

        csv_bytes = peak_features_to_csv([feature])
        rows = list(csv.DictReader(StringIO(csv_bytes.decode("utf-8-sig"))))

        self.assertTrue(csv_bytes.startswith(codecs.BOM_UTF8))
        self.assertEqual(tuple(rows[0]), PEAK_FEATURE_COLUMNS)
        for field, value in feature.items():
            with self.subTest(field=field):
                self.assertEqual(rows[0][field], str(value))

    def test_empty_features_export_header_only_csv(self):
        csv_bytes = peak_features_to_csv([])
        decoded = csv_bytes.decode("utf-8-sig")

        self.assertTrue(csv_bytes.startswith(codecs.BOM_UTF8))
        self.assertEqual(decoded.splitlines(), [",".join(PEAK_FEATURE_COLUMNS)])

    def test_download_filename_differs_from_upload(self):
        self.assertEqual(
            build_peak_csv_filename("sample_raw.txt"),
            "sample_raw_candidate_peak_features.csv",
        )


if __name__ == "__main__":
    unittest.main()
