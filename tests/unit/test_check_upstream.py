import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import urllib.error

from tools import check_upstream
from tools.import_upstream import UpstreamLock


class CheckUpstreamTest(unittest.TestCase):
    def setUp(self):
        self.lock = UpstreamLock(
            repository="https://github.com/ggml-org/llama.cpp",
            tag="b10069",
            commit="a" * 40,
            archive_url="https://github.com/ggml-org/llama.cpp/archive/refs/tags/b10069.tar.gz",
            archive_sha256="b" * 64,
            imported_at_utc="2026-06-14T00:00:00Z",
        )
        self.now = dt.datetime(2026, 7, 29, tzinfo=dt.timezone.utc)

    def github_response(self, url):
        if url.endswith("/tags?per_page=1"):
            return [{"name": "b10123", "commit": {"sha": "c" * 40}}]
        if url.endswith(f"/commits/{self.lock.commit}"):
            return {"commit": {"committer": {"date": "2026-06-14T00:00:00Z"}}}
        self.fail(f"unexpected GitHub URL: {url}")

    def test_report_contains_current_latest_age_and_no_commit_count_trigger(self):
        with mock.patch.object(
            check_upstream, "github_json", side_effect=self.github_response
        ) as github_json:
            status = check_upstream.collect_status(self.lock, now=self.now)

        report = check_upstream.render_report(status)
        self.assertIn("b10069", report)
        self.assertIn(self.lock.commit, report)
        self.assertIn("b10123", report)
        self.assertIn("c" * 40, report)
        self.assertIn("45 days", report)
        self.assertEqual(status.candidate_reason, "none")
        self.assertEqual(github_json.call_count, 2)

    def test_age_candidate_requires_complete_green_release_gate(self):
        with mock.patch.object(
            check_upstream, "github_json", side_effect=self.github_response
        ):
            without_gate = check_upstream.collect_status(self.lock, now=self.now)
            with_gate = check_upstream.collect_status(
                self.lock, now=self.now, release_gate_passed=True
            )

        self.assertEqual(without_gate.candidate_reason, "none")
        self.assertEqual(with_gate.candidate_reason, "checkpoint-age")

    def test_explicit_policy_trigger_is_reported_without_release_mutation(self):
        with mock.patch.object(
            check_upstream, "github_json", side_effect=self.github_response
        ):
            status = check_upstream.collect_status(
                self.lock, now=self.now, triggers=("security-correctness",)
            )

        self.assertEqual(status.candidate_reason, "security-correctness")

    def test_github_output_contains_only_documented_fields(self):
        with mock.patch.object(
            check_upstream, "github_json", side_effect=self.github_response
        ):
            status = check_upstream.collect_status(self.lock, now=self.now)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "github-output"
            check_upstream.append_github_output(output, status)
            lines = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(
            {line.split("=", 1)[0] for line in lines},
            {
                "current_tag",
                "current_commit",
                "latest_tag",
                "latest_commit",
                "age_days",
                "candidate_reason",
            },
        )

    def test_github_json_retries_transient_network_failure(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            [{"name": "b10123"}]
        ).encode()
        with mock.patch.object(
            check_upstream.urllib.request,
            "urlopen",
            side_effect=[urllib.error.URLError("TLS EOF"), response],
        ) as urlopen, mock.patch.object(check_upstream.time, "sleep") as sleep:
            result = check_upstream.github_json("https://api.github.com/example")

        self.assertEqual(result, [{"name": "b10123"}])
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once()

    def test_github_json_does_not_retry_stable_local_failure(self):
        with mock.patch.object(
            check_upstream.urllib.request,
            "urlopen",
            side_effect=PermissionError("sandbox denied"),
        ) as urlopen, mock.patch.object(check_upstream.time, "sleep") as sleep:
            with self.assertRaises(PermissionError):
                check_upstream.github_json("https://api.github.com/example")

        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()

    def test_cli_can_run_as_a_repository_script(self):
        completed = subprocess.run(
            [sys.executable, "tools/check_upstream.py", "--help"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
