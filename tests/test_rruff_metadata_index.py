"""Tests for the audit-CSV-only RRUFF metadata index builder."""

import csv
import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_DIR / "scripts" / "08_build_rruff_metadata_index.py"
SPEC = importlib.util.spec_from_file_location("rruff_metadata_index", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load RRUFF metadata index module")
rruff_index = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rruff_index
SPEC.loader.exec_module(rruff_index)


def make_audit_row(
    file_name: str,
    file_type: str,
    *,
    mineral: str = "Testite",
    reference_label: str = "Testite",
    sample_id: str = "R123456",
    header_rruff_id: str = "R123456",
    filename_wavelength: str = "532",
    header_wavelength: str = "532",
    metadata_warnings: str = "",
) -> dict[str, str]:
    return {
        "file_name": file_name,
        "relative_path": file_name,
        "file_type": file_type,
        "filename_type": file_type,
        "header_file_type": file_type,
        "mineral": mineral,
        "sample_id": sample_id,
        "suspected_excitation_nm": filename_wavelength,
        "header_names": reference_label,
        "header_rruff_id": header_rruff_id,
        "header_raman_wavelength": header_wavelength,
        "format_parse_success": "True",
        "v1_upload_compatible": "True",
        "metadata_warnings": metadata_warnings,
    }


def measurement_name(stem: str, file_type: str, file_hash: str) -> str:
    return f"{stem}_{file_type}__{file_hash}.txt"


def write_audit_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted(rruff_index.REQUIRED_INPUT_COLUMNS)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class RruffMetadataIndexTests(unittest.TestCase):
    def test_measurement_stem_removes_terminal_type_and_hash(self):
        raw = (
            "nested/Testite__R123456__Raman__532______"
            "Raman_Data_RAW__abc123.txt"
        )
        processed = raw.replace("RAW__abc123", "Processed__def456")

        raw_result = rruff_index.derive_measurement_stem(raw)
        processed_result = rruff_index.derive_measurement_stem(processed)

        self.assertEqual(raw_result[0], processed_result[0])
        self.assertEqual(raw_result[1], rruff_index.FILE_TYPE_RAW)
        self.assertEqual(
            processed_result[1],
            rruff_index.FILE_TYPE_PROCESSED,
        )
        self.assertEqual(raw_result[2], "abc123")
        self.assertTrue(raw_result[3])

    def test_file_index_preserves_both_metadata_sources(self):
        file_name = measurement_name(
            "Filenameite__R123456__Raman__532______Raman_Data",
            "RAW",
            "abc123",
        )
        row = make_audit_row(
            file_name,
            "RAW",
            mineral="Filenameite",
            reference_label="Headerite",
            sample_id="R123456",
            header_rruff_id="R654321",
        )

        record = rruff_index.build_file_index([row])[0]

        self.assertEqual(record["filename_mineral"], "Filenameite")
        self.assertEqual(record["reference_label"], "Headerite")
        self.assertEqual(record["filename_sample_id"], "R123456")
        self.assertEqual(record["header_rruff_id"], "R654321")
        self.assertTrue(record["has_conflict"])
        self.assertTrue(record["needs_manual_review"])

    def test_pair_index_distinguishes_both_and_single_sided_stems(self):
        both_stem = "Bothite__R100001__Raman__532______Raman_Data"
        raw_stem = "Rawite__R100002__Raman__532______Raman_Data"
        processed_stem = "Processite__R100003__Raman__532______Raman_Data"
        rows = [
            make_audit_row(measurement_name(both_stem, "RAW", "a"), "RAW"),
            make_audit_row(
                measurement_name(both_stem, "Processed", "b"),
                "Processed",
            ),
            make_audit_row(measurement_name(raw_stem, "RAW", "c"), "RAW"),
            make_audit_row(
                measurement_name(processed_stem, "Processed", "d"),
                "Processed",
            ),
        ]

        pairs = rruff_index.build_pair_index(
            rruff_index.build_file_index(rows)
        )
        statuses = {pair["measurement_stem"]: pair["pair_status"] for pair in pairs}

        self.assertEqual(statuses[both_stem], rruff_index.PAIR_STATUS_BOTH)
        self.assertEqual(statuses[raw_stem], rruff_index.PAIR_STATUS_RAW_ONLY)
        self.assertEqual(
            statuses[processed_stem],
            rruff_index.PAIR_STATUS_PROCESSED_ONLY,
        )

    def test_missing_conflict_and_review_flags_are_explicit(self):
        file_name = measurement_name(
            "Testite__R123456__Raman__532______Raman_Data",
            "RAW",
            "abc123",
        )
        row = make_audit_row(
            file_name,
            "RAW",
            mineral="Testite",
            reference_label="Possible Alias",
            sample_id="R123456",
            header_rruff_id="Unknown",
            filename_wavelength="532",
            header_wavelength="785.0 nm",
        )

        record = rruff_index.build_file_index([row])[0]
        flags = str(record["metadata_flags"]).split(";")

        self.assertIn("sample_id_header_missing", flags)
        self.assertIn("mineral_name_mismatch_requires_review", flags)
        self.assertIn("excitation_wavelength_conflict", flags)
        self.assertTrue(record["has_missing_metadata"])
        self.assertTrue(record["has_conflict"])
        self.assertIn("requires_review", str(record["metadata_status"]))

    def test_outputs_are_deterministic_utf8_sig_and_not_overwritten(self):
        stem = "Testite__R123456__Raman__532______Raman_Data"
        rows = [
            make_audit_row(
                measurement_name(stem, "Processed", "def456"),
                "Processed",
            ),
            make_audit_row(measurement_name(stem, "RAW", "abc123"), "RAW"),
        ]
        with TemporaryDirectory(dir=PROJECT_DIR / "tests") as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "audit.csv"
            write_audit_csv(input_path, rows)

            first = rruff_index.run_index(input_path, root / "first")
            second = rruff_index.run_index(input_path, root / "second")

            for first_path, second_path in zip(first[3], second[3]):
                self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
                self.assertTrue(first_path.read_bytes().startswith(b"\xef\xbb\xbf"))
            with self.assertRaises(FileExistsError):
                rruff_index.run_index(input_path, root / "first")

    def test_missing_required_columns_are_rejected(self):
        with TemporaryDirectory(dir=PROJECT_DIR / "tests") as temporary_directory:
            input_path = Path(temporary_directory) / "incomplete.csv"
            input_path.write_text("file_name\nexample.txt\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                rruff_index.load_audit_csv(input_path)


if __name__ == "__main__":
    unittest.main()
