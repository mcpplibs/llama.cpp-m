import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/check_ci_run.py"
SHA = "a" * 40


class CheckCiRunTest(unittest.TestCase):
    def run_checker(self, runs):
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "runs.json"
            payload.write_text(json.dumps({"workflow_runs": runs}), encoding="utf-8")
            return subprocess.run(
                [sys.executable, SCRIPT, "--runs", payload, "--sha", SHA],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

    @staticmethod
    def workflow_run(
        run_id, *, sha=SHA, status="completed", conclusion="success"
    ):
        return {
            "id": run_id,
            "head_sha": sha,
            "status": status,
            "conclusion": conclusion,
        }

    def test_accepts_latest_successful_ci_run_for_exact_commit(self):
        completed = self.run_checker(
            [
                self.workflow_run(10, conclusion="failure"),
                self.workflow_run(11, sha="b" * 40, conclusion="success"),
                self.workflow_run(12, conclusion="success"),
            ]
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("CI run 12 succeeded", completed.stdout)

    def test_rejects_latest_failure_even_if_older_run_succeeded(self):
        completed = self.run_checker(
            [
                self.workflow_run(20, conclusion="success"),
                self.workflow_run(21, conclusion="failure"),
            ]
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("latest CI run 21 is completed/failure", completed.stderr)

    def test_rejects_latest_pending_run(self):
        completed = self.run_checker(
            [
                self.workflow_run(30, conclusion="success"),
                self.workflow_run(31, status="in_progress", conclusion=None),
            ]
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("latest CI run 31 is in_progress/none", completed.stderr)

    def test_rejects_payload_without_exact_commit_run(self):
        completed = self.run_checker([self.workflow_run(40, sha="b" * 40)])

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("no CI workflow run found", completed.stderr)


if __name__ == "__main__":
    unittest.main()
