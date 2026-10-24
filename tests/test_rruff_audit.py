"""Tests for the independent RRUFF dataset audit script."""

import codecs
import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_DIR / "scripts" / "08_audit_rruff_dataset.py"
SPEC = importlib.util.spec_from_file_location("rruff_audit", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load RRUFF audit module")
rruff_audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rruff_audit
SPEC.loader.exec_module(rruff_audit)


def make_valid_content(point_count: int = 21) -> bytes:
    rows = [
        "##NAMES=Testite",
        "##RRUFFID=R123456",
        "##FILETYPE=Raman RAW",
        "##RAMAN WAVELENGTH=532",
        "",
    ]
    rows.extend(
        f"{100 + index:.1f}, {1000 + index:.1f}"
        for index in range(point_count)
    )
    rows.append("##END=")
    return ("\n".join(rows) + "\n").encode("utf-8")


class RruffAuditTests(unittest.TestCase):
    def audit_content(
        self,
        content: bytes,
        file_name: str = (
            "Testite__R123456__Raman__532______"
            "Raman_Data_RAW__abcdef.txt"
        ),
    ):
        with TemporaryDirectory(dir=PROJECT_DIR / "tests") as temporary_directory:
            root = Path(temporary_directory)
            path = root / file_name
            path.write_bytes(content)
            return rruff_audit.audit_file(path, root=root)

    def test_valid_two_column_numeric_file(self):
        record = self.audit_content(make_valid_content())

        self.assertTrue(record.format_parse_success)
        self.assertTrue(record.v1_upload_compatible)
        self.assertEqual(record.delimiter, "comma")
        self.assertEqual(record.data_column_count, "2")
        self.assertEqual(record.point_count, 21)
        self.assertTrue(record.wavenumber_strictly_increasing)
        self.assertEqual(record.wavenumber_min, 100.0)
        self.assertEqual(record.wavenumber_max, 120.0)

    def test_empty_file(self):
        record = self.audit_content(b"")

        self.assertFalse(record.format_parse_success)
        self.assertFalse(record.v1_upload_compatible)
        self.assertIn("empty_file", record.parse_errors)

    def test_non_numeric_line(self):
        content = make_valid_content().replace(
            b"110.0, 1010.0",
            b"bad, data",
        )
        record = self.audit_content(content)

        self.assertFalse(record.format_parse_success)
        self.assertIn("non_numeric_line", record.parse_errors)

    def test_duplicate_and_unsorted_wavenumbers(self):
        duplicate = make_valid_content().replace(
            b"110.0, 1010.0",
            b"109.0, 1010.0",
        )
        unsorted = make_valid_content().replace(
            b"110.0, 1010.0",
            b"90.0, 1010.0",
        )

        duplicate_record = self.audit_content(duplicate)
        unsorted_record = self.audit_content(unsorted)

        self.assertTrue(duplicate_record.format_parse_success)
        self.assertTrue(duplicate_record.has_duplicate_wavenumber)
        self.assertFalse(duplicate_record.v1_upload_compatible)
        self.assertTrue(unsorted_record.format_parse_success)
        self.assertTrue(unsorted_record.has_descending_wavenumber)
        self.assertFalse(unsorted_record.v1_upload_compatible)

    def test_nonfinite_values(self):
        content = make_valid_content().replace(
            b"110.0, 1010.0",
            b"110.0, nan",
        )
        record = self.audit_content(content)

        self.assertTrue(record.format_parse_success)
        self.assertEqual(record.nonfinite_value_count, 1)
        self.assertFalse(record.v1_upload_compatible)
        self.assertIn(
            "nonfinite_values",
            record.v1_incompatibility_reasons,
        )

    def test_file_over_v1_size_limit_is_not_v1_compatible(self):
        padding_size = (
            rruff_audit.V1_MAX_UPLOAD_SIZE_BYTES
            - len(make_valid_content())
            + 1
        )
        content = make_valid_content() + b"#" + (b"x" * padding_size)
        record = self.audit_content(content)

        self.assertTrue(record.format_parse_success)
        self.assertFalse(record.v1_upload_compatible)
        self.assertIn(
            "file_exceeds_5_mb",
            record.v1_incompatibility_reasons,
        )

    def test_raw_and_processed_filename_classification(self):
        raw_name = (
            "Testite__R123456__Raman__532______"
            "Raman_Data_RAW__abcdef.txt"
        )
        processed_name = (
            "Testite__R123456__Raman__532______"
            "Raman_Data_Processed__abcdef.txt"
        )

        self.assertEqual(
            rruff_audit.classify_filename(raw_name),
            rruff_audit.FILE_TYPE_RAW,
        )
        self.assertEqual(
            rruff_audit.classify_filename(processed_name),
            rruff_audit.FILE_TYPE_PROCESSED,
        )

    def test_missing_filename_metadata_is_unknown(self):
        metadata = rruff_audit.extract_filename_metadata("unstructured.txt")

        self.assertEqual(metadata["mineral"], "Unknown")
        self.assertEqual(metadata["sample_id"], "Unknown")
        self.assertEqual(metadata["suspected_excitation_nm"], "Unknown")

    def test_non_r_sample_id_prefix_is_extracted_when_pattern_is_reliable(self):
        file_name = (
            "Stolzite__X060016__Raman__514______"
            "Raman_Data_RAW__abcdef.txt"
        )

        metadata = rruff_audit.extract_filename_metadata(file_name)

        self.assertEqual(metadata["mineral"], "Stolzite")
        self.assertEqual(metadata["sample_id"], "X060016")
        self.assertEqual(metadata["suspected_excitation_nm"], "514")

    def test_numeric_three_column_file_parses_but_is_not_v1_compatible(self):
        rows = [f"{index}, {index + 1}, {index + 2}" for index in range(21)]
        record = self.audit_content(("\n".join(rows) + "\n").encode())

        self.assertTrue(record.format_parse_success)
        self.assertEqual(record.data_column_count, "3")
        self.assertFalse(record.v1_upload_compatible)
        self.assertIn(
            "column_count_not_two",
            record.v1_incompatibility_reasons,
        )

    def test_fixed_seed_sampling_is_reproducible_and_balanced(self):
        with TemporaryDirectory(dir=PROJECT_DIR / "tests") as temporary_directory:
            root = Path(temporary_directory)
            paths = []
            for index in range(10):
                raw_path = root / f"Mineral{index}_Raman_Data_RAW_{index}.txt"
                processed_path = (
                    root / f"Mineral{index}_Raman_Data_Processed_{index}.txt"
                )
                raw_path.write_bytes(b"")
                processed_path.write_bytes(b"")
                paths.extend([raw_path, processed_path])

            first = rruff_audit.select_sample(paths, limit=10, seed=42)
            second = rruff_audit.select_sample(paths, limit=10, seed=42)

        self.assertEqual(
            [path.name for path in first],
            [path.name for path in second],
        )
        classifications = [
            rruff_audit.classify_filename(path.name) for path in first
        ]
        self.assertEqual(classifications.count(rruff_audit.FILE_TYPE_RAW), 5)
        self.assertEqual(
            classifications.count(rruff_audit.FILE_TYPE_PROCESSED),
            5,
        )

    def test_damaged_utf8_bom_is_a_single_file_decode_failure(self):
        content = codecs.BOM_UTF8 + b"\xffbroken"

        record = self.audit_content(content)

        self.assertEqual(record.file_size_bytes, len(content))
        self.assertEqual(record.encoding, "utf-8-sig")
        self.assertTrue(record.has_bom)
        self.assertEqual(record.sample_id, "R123456")
        self.assertFalse(record.format_parse_success)
        self.assertFalse(record.v1_upload_compatible)
        self.assertIn("decode_error", record.parse_errors)
        self.assertIn("decode_error", record.v1_incompatibility_reasons)

    def test_truncated_utf16_is_a_single_file_decode_failure(self):
        content = codecs.BOM_UTF16_LE + b"A"

        record = self.audit_content(content)

        self.assertEqual(record.encoding, "utf-16-le")
        self.assertTrue(record.has_bom)
        self.assertFalse(record.format_parse_success)
        self.assertFalse(record.v1_upload_compatible)
        self.assertIn("decode_error", record.parse_errors)

    def test_batch_continues_when_one_file_has_decode_error(self):
        with TemporaryDirectory(dir=PROJECT_DIR / "tests") as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "output"
            valid_names = [
                "Alpha__R123456__Raman__532______Raman_Data_RAW__a.txt",
                (
                    "Beta__R234567__Raman__532______"
                    "Raman_Data_Processed__b.txt"
                ),
            ]
            for file_name in valid_names:
                (root / file_name).write_bytes(make_valid_content())
            broken_name = (
                "Gamma__R345678__Raman__532______"
                "Raman_Data_RAW__broken.txt"
            )
            (root / broken_name).write_bytes(codecs.BOM_UTF8 + b"\xff")

            records, summary, csv_path, json_path = rruff_audit.run_audit(
                root=root,
                limit=3,
                seed=42,
                output_dir=output_dir,
            )

            self.assertEqual(len(records), 3)
            self.assertEqual(summary["format_parse"]["success"], 2)
            self.assertEqual(summary["format_parse"]["failure"], 1)
            self.assertTrue(csv_path.is_file())
            self.assertTrue(json_path.is_file())
            failed = [record for record in records if record.parse_errors]
            self.assertEqual(len(failed), 1)
            self.assertIn("decode_error", failed[0].parse_errors)

    def test_audit_scope_distinguishes_sample_from_full_dataset(self):
        with TemporaryDirectory(dir=PROJECT_DIR / "tests") as temporary_directory:
            root = Path(temporary_directory)
            for index in range(4):
                file_name = (
                    f"Mineral{index}__R{index:06d}__Raman__532______"
                    f"Raman_Data_RAW__{index}.txt"
                )
                (root / file_name).write_bytes(make_valid_content())

            sample_records, sample_summary, _, _ = rruff_audit.run_audit(
                root=root,
                limit=2,
                seed=42,
                output_dir=root / "sample-output",
            )
            full_records, full_summary, _, _ = rruff_audit.run_audit(
                root=root,
                limit=10,
                seed=42,
                output_dir=root / "full-output",
            )

            self.assertEqual(len(sample_records), 2)
            self.assertEqual(
                sample_summary["audit_scope"],
                "sample_only_not_full_dataset",
            )
            self.assertEqual(sample_summary["discovered_txt_files"], 4)
            self.assertEqual(sample_summary["audited_files"], 2)
            self.assertEqual(len(full_records), 4)
            self.assertEqual(full_summary["audit_scope"], "full_dataset")
            self.assertEqual(full_summary["discovered_txt_files"], 4)
            self.assertEqual(full_summary["audited_files"], 4)

    def test_filename_header_metadata_conflicts_generate_warnings(self):
        content = make_valid_content()
        content = content.replace(b"##NAMES=Testite", b"##NAMES=Aliasite")
        content = content.replace(b"##RRUFFID=R123456", b"##RRUFFID=R654321")
        content = content.replace(
            b"##RAMAN WAVELENGTH=532",
            b"##RAMAN WAVELENGTH=785",
        )

        record = self.audit_content(content)

        warnings = record.metadata_warnings.split(";")
        self.assertIn("sample_id_conflict", warnings)
        self.assertIn("mineral_name_mismatch_requires_review", warnings)
        self.assertIn("excitation_wavelength_conflict", warnings)
        self.assertEqual(record.header_names, "Aliasite")
        self.assertEqual(record.header_rruff_id, "R654321")
        self.assertEqual(record.header_raman_wavelength, "785")

    def test_metadata_comparison_accepts_case_and_numeric_format(self):
        content = make_valid_content()
        content = content.replace(b"##NAMES=Testite", b"##NAMES=testite")
        content = content.replace(b"##RRUFFID=R123456", b"##RRUFFID=r123456")
        content = content.replace(
            b"##RAMAN WAVELENGTH=532",
            b"##RAMAN WAVELENGTH=532.0 nm",
        )

        record = self.audit_content(content)

        self.assertEqual(record.metadata_warnings, "")

    def test_missing_header_metadata_is_not_reported_as_conflict(self):
        content = make_valid_content()
        content = content.replace(b"##NAMES=Testite\n", b"")
        content = content.replace(b"##RRUFFID=R123456\n", b"")
        content = content.replace(b"##RAMAN WAVELENGTH=532\n", b"")

        record = self.audit_content(content)

        warnings = record.metadata_warnings.split(";")
        self.assertIn("sample_id_header_missing", warnings)
        self.assertIn("mineral_header_missing", warnings)
        self.assertIn("excitation_wavelength_header_missing", warnings)
        self.assertFalse(any("conflict" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
