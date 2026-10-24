"""Tests for the RRUFF RAW candidate manifest selector."""

import csv
import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_DIR / "scripts" / "08_select_rruff_candidates.py"
SPEC = importlib.util.spec_from_file_location("rruff_candidates", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load RRUFF candidate selector")
rruff_candidates = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rruff_candidates
SPEC.loader.exec_module(rruff_candidates)


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
        "mineral": "Testite",
        "sample_id": "R123456",
        "suspected_excitation_nm": "532",
        "header_names": "Testite",
        "header_rruff_id": "R123456",
        "header_raman_wavelength": "532",
        "format_parse_success": "True",
        "data_column_count": "2",
        "nonfinite_value_count": "0",
        "has_duplicate_wavenumber": "False",
        "wavenumber_strictly_increasing": "True",
        "v1_upload_compatible": "True",
        "parse_errors": "",
        "v1_incompatibility_reasons": "",
        "metadata_warnings": "",
    }
    row.update(overrides)
    return row


def make_index_row(
    relative_path: str = "Testite_Raman_Data_RAW__abc.txt",
    file_type: str = "RAW",
    **overrides: str,
) -> dict[str, str]:
    row = {
        "relative_path": relative_path,
        "file_name": Path(relative_path).name,
        "file_type": file_type,
        "filename_type": file_type,
        "header_file_type": file_type,
        "measurement_stem": "Testite_Raman_Data",
        "filename_mineral": "Testite",
        "reference_label": "Testite",
        "filename_sample_id": "R123456",
        "header_rruff_id": "R123456",
        "filename_suspected_excitation_nm": "532",
        "header_raman_wavelength": "532",
        "format_parse_success": "True",
        "v1_upload_compatible": "True",
        "audit_metadata_warnings": "",
        "has_missing_metadata": "False",
        "has_conflict": "False",
        "needs_manual_review": "False",
        "metadata_status": "complete_no_flag",
        "metadata_flags": "",
    }
    row.update(overrides)
    return row


def write_csv(path: Path, fieldnames: set[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=sorted(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


class RruffCandidateTests(unittest.TestCase):
    def build_one(
        self,
        audit_row: dict[str, str] | None = None,
        index_row: dict[str, str] | None = None,
    ) -> dict[str, object]:
        if audit_row is None and index_row is None:
            audit_row = make_audit_row()
            index_row = make_index_row()
        elif audit_row is not None and index_row is None:
            warnings = audit_row.get("metadata_warnings", "")
            index_row = make_index_row(
                audit_row["relative_path"],
                audit_row["file_type"],
                filename_mineral=audit_row["mineral"],
                filename_sample_id=audit_row["sample_id"],
                reference_label=audit_row["header_names"],
                header_rruff_id=audit_row["header_rruff_id"],
                filename_suspected_excitation_nm=audit_row[
                    "suspected_excitation_nm"
                ],
                header_raman_wavelength=audit_row[
                    "header_raman_wavelength"
                ],
                format_parse_success=audit_row["format_parse_success"],
                v1_upload_compatible=audit_row["v1_upload_compatible"],
                audit_metadata_warnings=warnings,
                metadata_flags=warnings,
                has_conflict=(
                    "True" if "conflict" in warnings else "False"
                ),
                needs_manual_review=("True" if warnings else "False"),
            )
        elif audit_row is None and index_row is not None:
            audit_row = make_audit_row(
                index_row["relative_path"],
                index_row["file_type"],
                mineral=index_row["filename_mineral"],
                sample_id=index_row["filename_sample_id"],
                header_names=index_row["reference_label"],
                header_rruff_id=index_row["header_rruff_id"],
                suspected_excitation_nm=index_row[
                    "filename_suspected_excitation_nm"
                ],
                header_raman_wavelength=index_row[
                    "header_raman_wavelength"
                ],
                format_parse_success=index_row["format_parse_success"],
                v1_upload_compatible=index_row["v1_upload_compatible"],
                metadata_warnings=index_row["audit_metadata_warnings"],
            )
        joined = rruff_candidates.join_inputs(
            [audit_row],
            [index_row],
        )
        return rruff_candidates.build_candidate_manifest(joined)[0]

    def test_valid_raw_is_preprocessing_and_reference_candidate(self):
        record = self.build_one()

        self.assertTrue(record["is_raw"])
        self.assertTrue(record["preprocessing_candidate"])
        self.assertTrue(record["reference_candidate"])
        self.assertFalse(record["requires_review"])
        self.assertIn(
            "not_independently_validated",
            str(record["reference_label_status"]),
        )

    def test_processed_record_is_retained_but_not_independent_candidate(self):
        path = "Testite_Raman_Data_Processed__def.txt"
        record = self.build_one(
            make_audit_row(path, "Processed"),
            make_index_row(path, "Processed"),
        )

        self.assertTrue(record["is_processed"])
        self.assertFalse(record["preprocessing_candidate"])
        self.assertFalse(record["reference_candidate"])
        self.assertFalse(record["requires_review"])
        self.assertEqual(
            record["sample_unit_role"],
            "processed_derivative_not_independent_sample",
        )

    def test_each_preprocessing_rule_can_reject_raw_record(self):
        cases = {
            "format_parse_success": (
                {"format_parse_success": "False"},
                "format_parse_failed",
            ),
            "two_columns": (
                {"data_column_count": "3"},
                "data_column_count_not_two",
            ),
            "finite_values": (
                {"nonfinite_value_count": "1"},
                "nonfinite_values",
            ),
            "increasing_axis": (
                {"wavenumber_strictly_increasing": "False"},
                "wavenumber_not_strictly_increasing",
            ),
            "duplicate_axis": (
                {"has_duplicate_wavenumber": "True"},
                "duplicate_wavenumber",
            ),
            "v1_compatibility": (
                {"v1_upload_compatible": "False"},
                "v1_static_compatibility_failed",
            ),
        }
        for name, (overrides, expected_reason) in cases.items():
            with self.subTest(rule=name):
                record = self.build_one(make_audit_row(**overrides))
                self.assertFalse(record["preprocessing_candidate"])
                self.assertIn(
                    expected_reason,
                    str(record["preprocessing_rejection_reasons"]),
                )
                self.assertTrue(record["requires_review"])

    def test_invalid_nonfinite_counts_are_rejected_conservatively(self):
        cases = {
            "": "nonfinite_value_count_missing",
            "invalid": "nonfinite_value_count_invalid",
            "-1": "nonfinite_value_count_negative",
        }
        for value, expected_reason in cases.items():
            with self.subTest(value=value):
                record = self.build_one(
                    make_audit_row(nonfinite_value_count=value)
                )
                self.assertFalse(record["preprocessing_candidate"])
                self.assertIn(
                    expected_reason,
                    str(record["preprocessing_rejection_reasons"]),
                )

    def test_missing_metadata_is_traceable_and_requires_review(self):
        index_row = make_index_row(
            reference_label="Unknown",
            header_rruff_id="Unknown",
            has_missing_metadata="True",
            needs_manual_review="True",
            metadata_status="missing",
            metadata_flags=(
                "mineral_header_missing;sample_id_header_missing"
            ),
        )

        record = self.build_one(index_row=index_row)

        self.assertFalse(record["reference_candidate"])
        self.assertTrue(record["requires_review"])
        self.assertIn(
            "reference_label_missing",
            str(record["reference_rejection_reasons"]),
        )
        self.assertEqual(record["header_rruff_id"], "Unknown")

    def test_metadata_conflict_is_traceable_and_requires_review(self):
        index_row = make_index_row(
            filename_sample_id="R123456",
            header_rruff_id="R654321",
            has_conflict="True",
            needs_manual_review="True",
            metadata_status="conflict",
            metadata_flags="sample_id_conflict",
            audit_metadata_warnings="sample_id_conflict",
        )
        audit_row = make_audit_row(
            header_rruff_id="R654321",
            metadata_warnings="sample_id_conflict",
        )

        record = self.build_one(audit_row, index_row)

        self.assertFalse(record["reference_candidate"])
        self.assertTrue(record["requires_review"])
        self.assertIn("sample_id_conflict", str(record["review_reasons"]))
        self.assertEqual(record["filename_sample_id"], "R123456")
        self.assertEqual(record["header_rruff_id"], "R654321")

    def test_audit_conflict_cannot_be_hidden_by_index_flags(self):
        audit_row = make_audit_row(
            metadata_warnings="sample_id_conflict",
        )
        index_row = make_index_row(
            audit_metadata_warnings="sample_id_conflict",
            metadata_flags="sample_id_conflict",
            has_conflict="False",
        )

        with self.assertRaisesRegex(ValueError, "has_conflict"):
            rruff_candidates.join_inputs([audit_row], [index_row])

    def test_all_repeated_metadata_fields_must_match(self):
        cases = {
            "mineral": ("filename_mineral", "Otherite"),
            "sample_id": ("filename_sample_id", "R654321"),
            "header_names": ("reference_label", "Otherite"),
            "header_rruff_id": ("header_rruff_id", "R654321"),
            "filename_wavelength": (
                "filename_suspected_excitation_nm",
                "785",
            ),
            "header_wavelength": ("header_raman_wavelength", "785"),
            "format_parse_success": ("format_parse_success", "False"),
            "v1_upload_compatible": ("v1_upload_compatible", "False"),
            "metadata_warnings": ("audit_metadata_warnings", "unexpected"),
        }
        for name, (index_field, index_value) in cases.items():
            with self.subTest(field=name):
                index_row = make_index_row(**{index_field: index_value})
                with self.assertRaisesRegex(ValueError, "disagree"):
                    rruff_candidates.join_inputs(
                        [make_audit_row()],
                        [index_row],
                    )

    def test_processed_quality_exceptions_require_review(self):
        path = "Testite_Raman_Data_Processed__def.txt"
        cases = {
            "format": {"format_parse_success": "False"},
            "nonfinite": {"nonfinite_value_count": "1"},
            "duplicate": {"has_duplicate_wavenumber": "True"},
            "axis": {"wavenumber_strictly_increasing": "False"},
            "v1": {"v1_upload_compatible": "False"},
        }
        for name, overrides in cases.items():
            with self.subTest(issue=name):
                audit_row = make_audit_row(path, "Processed", **overrides)
                index_overrides = {
                    field: value
                    for field, value in overrides.items()
                    if field in {"format_parse_success", "v1_upload_compatible"}
                }
                index_row = make_index_row(
                    path,
                    "Processed",
                    **index_overrides,
                )
                record = self.build_one(audit_row, index_row)
                self.assertFalse(record["preprocessing_candidate"])
                self.assertFalse(record["reference_candidate"])
                self.assertTrue(record["requires_review"])
                self.assertTrue(record["spectral_quality_review_reasons"])

    def test_duplicate_paths_are_rejected_in_either_input(self):
        audit = make_audit_row()
        index = make_index_row()
        with self.assertRaisesRegex(ValueError, "duplicate relative_path"):
            rruff_candidates.join_inputs([audit, dict(audit)], [index])
        with self.assertRaisesRegex(ValueError, "duplicate relative_path"):
            rruff_candidates.join_inputs([audit], [index, dict(index)])

    def test_mismatched_record_sets_and_identity_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "record sets do not match"):
            rruff_candidates.join_inputs(
                [make_audit_row("audit-only.txt")],
                [make_index_row("index-only.txt")],
            )
        with self.assertRaisesRegex(ValueError, "disagree"):
            rruff_candidates.join_inputs(
                [make_audit_row()],
                [make_index_row(file_type="Processed")],
            )

    def test_run_retains_every_record_and_refuses_overwrite(self):
        raw_path = "Testite_Raman_Data_RAW__abc.txt"
        processed_path = "Testite_Raman_Data_Processed__def.txt"
        audit_rows = [
            make_audit_row(raw_path, "RAW"),
            make_audit_row(processed_path, "Processed"),
        ]
        index_rows = [
            make_index_row(raw_path, "RAW"),
            make_index_row(processed_path, "Processed"),
        ]
        with TemporaryDirectory(dir=PROJECT_DIR / "tests") as temporary_directory:
            root = Path(temporary_directory)
            audit_path = root / "audit.csv"
            index_path = root / "index.csv"
            output_dir = root / "output"
            write_csv(
                audit_path,
                rruff_candidates.REQUIRED_AUDIT_COLUMNS,
                audit_rows,
            )
            write_csv(
                index_path,
                rruff_candidates.REQUIRED_INDEX_COLUMNS,
                index_rows,
            )

            records, summary, output_paths = rruff_candidates.run_selection(
                audit_path,
                index_path,
                output_dir,
            )

            self.assertEqual(len(records), 2)
            self.assertEqual(summary["joined_record_count"], 2)
            self.assertEqual(summary["file_type_counts"]["RAW"], 1)
            self.assertEqual(summary["file_type_counts"]["Processed"], 1)
            for output_path in output_paths:
                self.assertTrue(output_path.read_bytes().startswith(b"\xef\xbb\xbf"))
            with self.assertRaises(FileExistsError):
                rruff_candidates.run_selection(
                    audit_path,
                    index_path,
                    output_dir,
                )


if __name__ == "__main__":
    unittest.main()
