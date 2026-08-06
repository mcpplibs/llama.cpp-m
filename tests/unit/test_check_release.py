from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from tools import check_release


def write_release_tree(root: Path) -> None:
    (root / "mcpp.toml").write_text(
        '[package]\nname = "llamacpp"\nversion = "b10069"\n',
        encoding="utf-8",
    )
    (root / "upstream.lock").write_text(
        """[upstream]
repository = "https://github.com/ggml-org/llama.cpp"
tag = "b10069"
commit = "178a6c44937154dc4c4eff0d166f4a044c4fceba"
archive_url = "https://github.com/ggml-org/llama.cpp/archive/refs/tags/b10069.tar.gz"
archive_sha256 = "293a7c65a11e2203c5468a06d0d0e8d21dfff16ad08712b16c61efbe0d93e097"
imported_at_utc = "2026-07-29T00:00:00Z"
""",
        encoding="utf-8",
    )
    (root / "third_party/llama.cpp").mkdir(parents=True)
    (root / "snapshots").mkdir()
    (root / "snapshots/b10069.json").write_text("{}\n", encoding="utf-8")


class CheckReleaseTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        write_release_tree(self.root)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_tag_matches_package_version_and_lock_mapping(self):
        identity = check_release.load_identity(self.root, "b10069")

        self.assertEqual(str(identity.version), "b10069")
        self.assertEqual(identity.upstream_tag, "b10069")
        self.assertEqual(
            identity.upstream_commit,
            "178a6c44937154dc4c4eff0d166f4a044c4fceba",
        )

    def test_rejects_tag_version_mismatch(self):
        with self.assertRaisesRegex(ValueError, "tag.*version"):
            check_release.load_identity(self.root, "b10069.1")

    def test_rejects_noncanonical_archive_mapping(self):
        lock = self.root / "upstream.lock"
        lock.write_text(
            lock.read_text(encoding="utf-8").replace(
                "/archive/refs/tags/b10069.tar.gz", "/archive/refs/heads/master.tar.gz"
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "archive URL"):
            check_release.load_identity(self.root, "b10069")

    def test_rejects_consistent_mapping_to_upstream_fork(self):
        lock = self.root / "upstream.lock"
        lock.write_text(
            lock.read_text(encoding="utf-8").replace(
                "https://github.com/ggml-org/llama.cpp",
                "https://github.com/example/llama.cpp",
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "official upstream repository"):
            check_release.load_identity(self.root, "b10069")

    def test_changed_existing_tag_mapping_fails(self):
        current = check_release.load_identity(self.root, "b10069")
        existing = current._replace(upstream_commit="f" * 40)

        with self.assertRaisesRegex(ValueError, "immutable"):
            check_release.validate_existing_release(current, existing)

    def test_existing_tag_with_incomplete_identity_fails_closed(self):
        completed = lambda arguments, code, output="": subprocess.CompletedProcess(
            arguments, code, stdout=output, stderr="missing"
        )

        def git(_root, *arguments):
            if arguments[0] == "rev-parse":
                return completed(arguments, 0, "d" * 40)
            if arguments[-1].endswith(":mcpp.toml"):
                return completed(arguments, 0, '[package]\nversion = "b10069"\n')
            return completed(arguments, 1)

        with mock.patch.object(check_release, "_git", side_effect=git):
            with self.assertRaisesRegex(ValueError, "existing release tag"):
                check_release._tag_identity(self.root, "b10069")

    def test_same_checkpoint_wrapper_fix_requires_a_revision_bump(self):
        # The build number names the checkpoint, so a same-checkpoint release can
        # only move the revision. Moving the build number instead would claim a
        # checkpoint that is not the one vendored.
        previous = check_release.ReleaseIdentity.for_test("b10069", "b10069")
        revision = check_release.ReleaseIdentity.for_test("b10069.1", "b10069")
        wrong_build = check_release.ReleaseIdentity.for_test("b10123", "b10069")

        check_release.validate_version_transition(previous, revision)
        with self.assertRaisesRegex(ValueError, "build number"):
            check_release.validate_version_transition(previous, wrong_build)

    def test_changed_checkpoint_takes_that_checkpoints_build_number(self):
        previous = check_release.ReleaseIdentity.for_test("b10069", "b10069")
        forward = check_release.ReleaseIdentity.for_test("b10123", "b10123")
        backward = check_release.ReleaseIdentity.for_test("b10069.1", "b10123")

        check_release.validate_version_transition(previous, forward)
        with self.assertRaisesRegex(ValueError, "higher build number"):
            check_release.validate_version_transition(previous, backward)

    def test_rejects_candidate_older_than_an_existing_release(self):
        current = check_release.ReleaseIdentity.for_test("b10069.1", "b10069")
        higher = check_release.ReleaseIdentity.for_test("b10123", "b10123")
        lower = check_release.ReleaseIdentity.for_test("b10069", "b10069")
        tags = subprocess.CompletedProcess(
            [], 0, stdout="b10123\nb10069\n", stderr=""
        )

        with mock.patch.object(check_release, "_git", return_value=tags):
            with mock.patch.object(
                check_release,
                "_tag_identity",
                side_effect=lambda _root, tag: {
                    "b10123": higher,
                    "b10069": lower,
                }[tag],
            ):
                with self.assertRaisesRegex(ValueError, "newer release already exists"):
                    check_release._previous_identity(self.root, current, "b10069.1")

    def test_vendored_source_rejects_git_metadata_and_submodules(self):
        check_release.validate_vendor_boundary(self.root)
        nested_git = self.root / "third_party/llama.cpp/.git"
        nested_git.mkdir()
        with self.assertRaisesRegex(ValueError, "Git metadata"):
            check_release.validate_vendor_boundary(self.root)
        nested_git.rmdir()
        (self.root / ".gitmodules").write_text("[submodule]\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "submodule"):
            check_release.validate_vendor_boundary(self.root)

    def test_repository_checks_invoke_import_export_and_snapshot_check_modes(self):
        runner = mock.Mock(return_value=subprocess.CompletedProcess([], 0))

        check_release.run_repository_checks(self.root, "b10069", runner=runner)

        commands = [call.args[0] for call in runner.call_args_list]
        self.assertEqual(len(commands), 3)
        self.assertIn("tools/import_upstream.py", commands[0][1])
        self.assertIn("--verify-tag", commands[0])
        self.assertIn("tools/gen_exports.py", commands[1][1])
        self.assertIn("--check", commands[1])
        self.assertIn("tools/audit_snapshot.py", commands[2][1])
        self.assertTrue(commands[2][-1].endswith("snapshots/b10069.json"))
        for call in runner.call_args_list:
            self.assertTrue(call.kwargs["check"])
            self.assertEqual(call.kwargs["cwd"], self.root)

    def test_mapping_line_is_stable(self):
        identity = check_release.load_identity(self.root, "b10069")
        self.assertEqual(
            check_release.format_mapping(identity),
            "llama.cpp-m b10069 -> llama.cpp b10069 "
            "(178a6c44937154dc4c4eff0d166f4a044c4fceba)",
        )

    def test_cli_can_run_as_a_repository_script(self):
        completed = subprocess.run(
            [sys.executable, "tools/check_release.py", "--help"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
