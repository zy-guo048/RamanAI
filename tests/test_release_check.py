from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "14_release_check.py"
SPEC = importlib.util.spec_from_file_location("raman_release_check_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
release = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release
SPEC.loader.exec_module(release)


class ReleaseCheckTests(unittest.TestCase):
    def test_real_project_pending_check_is_read_only_and_explicit(self):
        result = release.run_release_check(
            project_root=PROJECT_ROOT,
            method_lock_path=release.DEFAULT_METHOD_LOCK,
            executable_lock_path=release.DEFAULT_EXECUTABLE_LOCK,
            final_output_dir=PROJECT_ROOT / "results" / "never-created-release-test-output",
            require_final_results=False,
        )
        self.assertEqual(result["final_test_status"], "not_yet_executed")
        self.assertFalse(result["ready_for_release"])

    def test_absent_final_results_fail_release_mode(self):
        with self.assertRaisesRegex(FileNotFoundError, "release candidate"):
            release.run_release_check(
                project_root=PROJECT_ROOT,
                method_lock_path=release.DEFAULT_METHOD_LOCK,
                executable_lock_path=release.DEFAULT_EXECUTABLE_LOCK,
                final_output_dir=PROJECT_ROOT / "results" / "never-created-release-test-output",
                require_final_results=True,
            )

    def test_changed_locked_source_is_rejected(self):
        executable = json.loads(
            release.DEFAULT_EXECUTABLE_LOCK.read_text(encoding="utf-8-sig")
        )
        executable["executable_sources"][0]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            altered = Path(temporary) / "executable.json"
            altered.write_text(json.dumps(executable), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                release.run_release_check(
                    project_root=PROJECT_ROOT,
                    method_lock_path=release.DEFAULT_METHOD_LOCK,
                    executable_lock_path=altered,
                    final_output_dir=Path(temporary) / "absent",
                    require_final_results=False,
                )


if __name__ == "__main__":
    unittest.main()
