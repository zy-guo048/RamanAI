"""Tests for the experiment 9.1 RAW coverage profiler."""

import csv
import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_DIR / "scripts" / "09_profile_raw_coverage.py"
SPEC = importlib.util.spec_from_file_location("raw_coverage", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load RAW coverage profiler")
raw_coverage = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = raw_coverage
SPEC.loader.exec_module(raw_coverage)


def make_audit_row(
    relative_path: str = "Testite_Raman_Data_RAW__abc.txt",
    file_type: str = "RAW",
    **overrides: str,
) -> dict[str, str]:
    row = {
        "file_name": Path(relative_path).name,
        "relative_path": relative_path,
        "file_type": file_type,
        "filename_type": file_type,
        "header_file_type": file_type,
        "header_names": "Testite",
        "header_rruff_id": "R123456",
        "header_raman_wavelength": "532",
        "point_count": "1000",
        "format_parse_success": "True",
        "data_column_count": "2",
        "nonfinite_value_count": "0",
        "v1_upload_compatible": "True",
        "wavenumber_min": "100",
        "wavenumber_max": "1500",
        "wavenumber_strictly_increasing": "True",
        "parse_errors": "",
        "v1_incompatibility_reasons": "",
        "metadata_warnings": "",
    }
    row.update(overrides)
    return row


def make_manifest_row(
    relative_path: str = "Testite_Raman_Data_RAW__abc.txt",
    file_type: str = "RAW",
    **overrides: str,
) -> dict[str, str]:
    is_raw = file_type == "RAW"
    row = {
        "relative_path": relative_path,
        "file_name": Path(relative_path).name,
        "measurement_stem": relative_path.split("_RAW__", 1)[0],
        "file_type": file_type,
        "filename_type": file_type,
        "header_file_type": file_type,
        "is_raw": str(is_raw),
        "is_processed": str(file_type == "Processed"),
        "preprocessing_candidate": str(is_raw),
        "reference_label": "Testite",
        "filename_sample_id": "R123456",
        "header_rruff_id": "R123456",
        "header_raman_wavelength": "532",
        "format_parse_success": "True",
        "data_column_count": "2",
        "nonfinite_value_count": "0",
        "wavenumber_strictly_increasing": "True",
        "v1_upload_compatible": "True",
        "parse_errors": "",
        "v1_incompatibility_reasons": "",
        "audit_metadata_warnings": "",
    }
    row.update(overrides)
    return row


def make_candidate(
    index: int,
    excitation: str = "532",
    span: float = 1000.0,
    points: int = 1000,
) -> dict[str, object]:
    return {
        "relative_path": f"sample_{index}.txt",
        "file_name": f"sample_{index}.txt",
        "measurement_stem": f"sample_{index}",
        "sample_id": f"R{index:06d}",
        "sample_id_source": "header_rruff_id",
        "filename_sample_id": f"R{index:06d}",
        "header_rruff_id": f"R{index:06d}",
        "reference_label": f"Mineral {index}",
        "excitation_wavelength_group": excitation,
        "header_raman_wavelength": excitation,
        "wavenumber_min_cm-1": 100.0,
        "wavenumber_max_cm-1": 100.0 + span,
        "wavenumber_span_cm-1": span,
        "point_count": points,
    }


def write_csv(path: Path, fields: set[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=sorted(fields))
        writer.writeheader()
        writer.writerows(rows)


class RawCoverageTests(unittest.TestCase):
    def test_input_record_sets_and_common_fields_must_match(self):
        with self.assertRaisesRegex(ValueError, "record sets do not match"):
            raw_coverage.join_inputs(
                [make_audit_row("audit.txt")],
                [make_manifest_row("manifest.txt")],
            )
        with self.assertRaisesRegex(ValueError, "disagree"):
            raw_coverage.join_inputs(
                [make_audit_row()],
                [make_manifest_row(header_raman_wavelength="785")],
            )

    def test_duplicate_relative_path_is_rejected(self):
        audit = make_audit_row()
        manifest = make_manifest_row()
        with self.assertRaisesRegex(ValueError, "duplicate relative_path"):
            raw_coverage.join_inputs([audit, dict(audit)], [manifest])
        with self.assertRaisesRegex(ValueError, "duplicate relative_path"):
            raw_coverage.join_inputs([audit], [manifest, dict(manifest)])

    def test_only_admitted_raw_records_are_profiled(self):
        raw_audit = make_audit_row("raw.txt")
        raw_manifest = make_manifest_row("raw.txt")
        rejected_audit = make_audit_row("rejected.txt")
        rejected_manifest = make_manifest_row(
            "rejected.txt",
            preprocessing_candidate="False",
        )
        processed_audit = make_audit_row("processed.txt", "Processed")
        processed_manifest = make_manifest_row("processed.txt", "Processed")
        joined = raw_coverage.join_inputs(
            [raw_audit, rejected_audit, processed_audit],
            [raw_manifest, rejected_manifest, processed_manifest],
        )

        records = raw_coverage.build_candidate_records(joined)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["relative_path"], "raw.txt")

    def test_endpoint_band_coverage_rule(self):
        self.assertTrue(raw_coverage.fully_covers_band(200, 900, 300, 850))
        self.assertTrue(raw_coverage.fully_covers_band(300, 850, 300, 850))
        self.assertFalse(raw_coverage.fully_covers_band(301, 900, 300, 850))
        self.assertFalse(raw_coverage.fully_covers_band(200, 849, 300, 850))

    def test_invalid_wavenumber_ranges_are_rejected(self):
        cases = (
            {"wavenumber_min": "nan"},
            {"wavenumber_max": "inf"},
            {"wavenumber_min": "bad"},
            {"wavenumber_min": "500", "wavenumber_max": "500"},
            {"wavenumber_min": "600", "wavenumber_max": "500"},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                audit = make_audit_row(**overrides)
                joined = raw_coverage.join_inputs(
                    [audit],
                    [make_manifest_row()],
                )
                with self.assertRaisesRegex(ValueError, "Invalid wavenumber"):
                    raw_coverage.build_candidate_records(joined)

    def test_seeded_diversity_selection_is_reproducible(self):
        records = [
            make_candidate(
                index,
                excitation=("532", "785", "633")[index % 3],
                span=500.0 + index * 50,
                points=500 + index * 20,
            )
            for index in range(40)
        ]

        first, first_summary = raw_coverage.select_diverse_trial_set(
            records, 30, 42
        )
        second, second_summary = raw_coverage.select_diverse_trial_set(
            list(reversed(records)), 30, 42
        )

        self.assertEqual(first, second)
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(
            set(first_summary["represented_excitation_groups"]),
            {"532", "633", "785"},
        )

    def test_small_dataset_selects_all_and_reports_shortfall(self):
        records = [make_candidate(index) for index in range(5)]

        selected, summary = raw_coverage.select_diverse_trial_set(
            records, 30, 42
        )

        self.assertEqual(len(selected), 5)
        self.assertEqual(summary["shortfall"], 25)
        self.assertIn("Only 5 eligible", summary["compromise_note"])

    def test_outputs_refuse_overwrite(self):
        with TemporaryDirectory(dir=PROJECT_DIR / "tests") as temporary_directory:
            root = Path(temporary_directory)
            audit_path = root / "audit.csv"
            manifest_path = root / "manifest.csv"
            output_dir = root / "output"
            write_csv(
                audit_path,
                raw_coverage.REQUIRED_AUDIT_COLUMNS,
                [make_audit_row()],
            )
            write_csv(
                manifest_path,
                raw_coverage.REQUIRED_MANIFEST_COLUMNS,
                [make_manifest_row()],
            )

            result = raw_coverage.run_profile(
                audit_path,
                manifest_path,
                output_dir,
                seed=42,
                sample_size=30,
            )

            self.assertEqual(len(result[1]), 1)
            for output_path in result[3]:
                self.assertTrue(output_path.read_bytes().startswith(b"\xef\xbb\xbf"))
            with self.assertRaises(FileExistsError):
                raw_coverage.run_profile(
                    audit_path,
                    manifest_path,
                    output_dir,
                    seed=42,
                    sample_size=30,
                )


if __name__ == "__main__":
    unittest.main()
