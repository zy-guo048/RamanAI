"""Tests for Streamlit-independent RamanAI app helpers."""

import unittest

from app import _build_analysis_request_key


class AppStateHelperTests(unittest.TestCase):
    def test_same_request_has_same_key(self):
        first = _build_analysis_request_key(b"spectrum", "sample.txt", 1000.0)
        second = _build_analysis_request_key(b"spectrum", "sample.txt", 1000.0)

        self.assertEqual(first, second)

    def test_changing_uploaded_file_invalidates_request_key(self):
        first = _build_analysis_request_key(b"first", "sample.txt", 1000.0)
        second = _build_analysis_request_key(b"second", "sample.txt", 1000.0)

        self.assertNotEqual(first, second)

    def test_changing_filename_invalidates_request_key(self):
        first = _build_analysis_request_key(b"spectrum", "first.txt", 1000.0)
        second = _build_analysis_request_key(b"spectrum", "second.txt", 1000.0)

        self.assertNotEqual(first, second)

    def test_changing_prominence_invalidates_request_key(self):
        first = _build_analysis_request_key(b"spectrum", "sample.txt", 1000.0)
        second = _build_analysis_request_key(b"spectrum", "sample.txt", 10.0)

        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
