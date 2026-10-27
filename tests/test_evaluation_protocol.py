from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "11_build_evaluation_protocol.py"
SPEC = importlib.util.spec_from_file_location("evaluation_protocol", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
protocol = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = protocol
SPEC.loader.exec_module(protocol)


def _record(
    path: str,
    sample_id: str,
    label: str,
    *,
    sample_source: str = "header_and_filename_agree",
    review: str = "False",
    missing: str = "False",
    conflict: str = "False",
    stem: str | None = None,
) -> dict[str, str]:
    return {
        "relative_path": path,
        "file_name": Path(path).name,
        "cache_row_index": "0",
        "measurement_stem": stem or Path(path).stem,
        "sample_id": sample_id,
        "sample_id_source": sample_source,
        "filename_sample_id": sample_id,
        "header_rruff_id": sample_id,
        "reference_label": label,
        "reference_label_status": "candidate_reference_label_not_independently_validated",
        "reference_candidate": "True",
        "excitation_wavelength_nm": "532",
        "excitation_wavelength_source": "header_and_filename_agree",
        "filename_suspected_excitation_nm": "532",
        "header_raman_wavelength": "532",
        "requires_review": review,
        "review_reasons": "metadata_requires_review" if review == "True" else "",
        "audit_metadata_warnings": "",
        "has_missing_metadata": missing,
        "has_conflict": conflict,
        "metadata_status": "complete_no_flag",
        "metadata_flags": "",
        "processing_status": "success",
        "failure_code": "",
    }


def _label_records(label: str, sample_count: int, measurements: int = 1):
    records = []
    for sample_number in range(sample_count):
        sample_id = f"{label}_R{sample_number:06d}"
        for measurement in range(measurements):
            records.append(
                _record(
                    f"{label}_{sample_id}_{measurement}.txt",
                    sample_id,
                    label,
                    stem=f"{label}_{sample_id}_{measurement}",
                )
            )
    return records


class EvaluationProtocolTests(unittest.TestCase):
    def test_same_sample_multiple_measurements_never_cross_groups(self):
        records = _label_records("LabelA", 5, measurements=2)
        manifest, _distribution, _context = protocol.build_evaluation_plan(records)
        groups_by_sample = {}
        for record in manifest:
            groups_by_sample.setdefault(record["sample_id"], set()).add(
                record["evaluation_group"]
            )
        self.assertTrue(all(len(groups) == 1 for groups in groups_by_sample.values()))

    def test_same_sample_multiple_labels_is_excluded_as_conflict(self):
        records = _label_records("LabelA", 5)
        records.extend(
            [
                _record("conflict_a.txt", "R999999", "LabelA"),
                _record("conflict_b.txt", "R999999", "LabelB"),
            ]
        )
        manifest, _distribution, _context = protocol.build_evaluation_plan(records)
        conflict_rows = [row for row in manifest if row["sample_id"] == "R999999"]
        self.assertEqual(
            {row["evaluation_group"] for row in conflict_rows},
            {protocol.GROUP_CONFLICT},
        )
        self.assertTrue(
            all(
                "sample_id_associated_with_multiple_reference_labels"
                in row["formal_ineligibility_reasons"]
                for row in conflict_rows
            )
        )

    def test_unknown_identity_is_exploratory_and_not_grouped(self):
        records = _label_records("LabelA", 5)
        records.extend(
            [
                _record(
                    "unknown_a.txt",
                    "Unknown",
                    "LabelA",
                    sample_source="missing",
                ),
                _record(
                    "unknown_b.txt",
                    "Unknown",
                    "LabelA",
                    sample_source="filename_header_conflict_unresolved",
                ),
            ]
        )
        manifest, _distribution, _context = protocol.build_evaluation_plan(records)
        unknown_rows = [row for row in manifest if row["sample_id"] == "Unknown"]
        self.assertTrue(
            all(row["evaluation_group"] == protocol.GROUP_INELIGIBLE for row in unknown_rows)
        )
        self.assertTrue(all(not row["known_sample_id_usable"] for row in unknown_rows))

    def test_sparse_labels_one_to_four_samples_are_not_forced_into_strict_set(self):
        records = []
        for count in range(1, 5):
            records.extend(_label_records(f"Sparse{count}", count))
        manifest, _distribution, _context = protocol.build_evaluation_plan(records)
        self.assertTrue(
            all(row["evaluation_group"] == protocol.GROUP_SPARSE for row in manifest)
        )
        self.assertTrue(all(not row["strict_closed_set_eligible"] for row in manifest))

    def test_gallery_has_at_least_three_sample_ids_for_every_strict_label(self):
        records = _label_records("A", 5) + _label_records("B", 7)
        manifest, distribution, _context = protocol.build_evaluation_plan(records)
        strict_labels = [
            row
            for row in distribution
            if row["row_type"] == "reference_label"
            and row["strict_closed_set_label"]
        ]
        self.assertEqual(len(strict_labels), 2)
        self.assertTrue(all(row["gallery_sample_id_count"] >= 3 for row in strict_labels))
        self.assertTrue(all(row["development_sample_id_count"] == 1 for row in strict_labels))
        self.assertTrue(all(row["final_test_sample_id_count"] == 1 for row in strict_labels))
        self.assertTrue(all(row["strict_closed_set_eligible"] for row in manifest))

    def test_development_and_final_do_not_share_sample_ids_with_gallery(self):
        records = _label_records("A", 8, measurements=2)
        manifest, _distribution, _context = protocol.build_evaluation_plan(records)
        groups = {}
        for row in manifest:
            groups.setdefault(row["evaluation_group"], set()).add(row["sample_id"])
        self.assertTrue(groups[protocol.GROUP_GALLERY].isdisjoint(groups[protocol.GROUP_DEVELOPMENT]))
        self.assertTrue(groups[protocol.GROUP_GALLERY].isdisjoint(groups[protocol.GROUP_FINAL]))
        self.assertTrue(groups[protocol.GROUP_DEVELOPMENT].isdisjoint(groups[protocol.GROUP_FINAL]))

    def test_fixed_seed_is_reproducible(self):
        records = _label_records("A", 10) + _label_records("B", 10)
        first = protocol.build_evaluation_plan(records, seed=42)[0]
        second = protocol.build_evaluation_plan(records, seed=42)[0]
        first_map = [(row["relative_path"], row["evaluation_group"]) for row in first]
        second_map = [(row["relative_path"], row["evaluation_group"]) for row in second]
        self.assertEqual(first_map, second_map)

    def test_file_and_measurement_stem_do_not_cross_groups(self):
        records = _label_records("A", 6, measurements=2)
        manifest, _distribution, _context = protocol.build_evaluation_plan(records)
        leakage = protocol.audit_leakage(manifest)
        self.assertTrue(leakage["passed"])
        self.assertEqual(leakage["relative_path_cross_group_count"], 0)
        self.assertEqual(leakage["measurement_stem_cross_group_count"], 0)

    def test_failed_leakage_audit_blocks_formal_publication(self):
        protocol.require_leakage_audit_passed({"passed": True})
        with self.assertRaisesRegex(ValueError, "Leakage audit failed"):
            protocol.require_leakage_audit_passed(
                {"passed": False, "same_known_sample_id_cross_group_count": 1}
            )

    def test_metadata_review_makes_whole_sample_group_ineligible(self):
        records = _label_records("A", 5)
        records.append(
            _record("reviewed_extra.txt", "A_R000000", "A", review="True")
        )
        manifest, _distribution, _context = protocol.build_evaluation_plan(records)
        affected = [row for row in manifest if row["sample_id"] == "A_R000000"]
        self.assertTrue(all(not row["sample_group_eligible"] for row in affected))
        self.assertTrue(
            all(row["evaluation_group"] == protocol.GROUP_CONFLICT for row in affected)
        )

    def test_csv_json_markdown_outputs_and_refuse_overwrite(self):
        records = _label_records("A", 5)
        manifest, distribution, context = protocol.build_evaluation_plan(records)
        leakage = protocol.audit_leakage(manifest)
        summary = protocol.build_audit_summary(
            Path("cache.npz"),
            Path("index.csv"),
            manifest,
            distribution,
            context,
            leakage,
        )
        markdown = protocol.build_protocol_markdown(summary)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as temporary:
            output_dir = Path(temporary) / "output"
            paths = protocol.write_outputs(
                output_dir, manifest, distribution, summary, markdown
            )
            self.assertTrue(all(path.is_file() for path in paths))
            audit = json.loads(paths[2].read_text(encoding="utf-8-sig"))
            self.assertIn("manifest_csv_sha256", audit["reproducibility"])
            self.assertIn("final_test_identity_sha256", audit["reproducibility"])
            self.assertIn("final_test_query", paths[3].read_text(encoding="utf-8-sig"))
            with self.assertRaises(FileExistsError):
                protocol.write_outputs(
                    output_dir, manifest, distribution, summary, markdown
                )


if __name__ == "__main__":
    unittest.main()
