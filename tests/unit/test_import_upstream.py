import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

from tools import import_upstream


def make_archive(path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, contents in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            archive.addfile(info, io.BytesIO(contents))


def archive_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ImportUpstreamTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def lock_for(self, archive: Path, digest: str | None = None):
        return import_upstream.UpstreamLock(
            repository="https://github.com/example/project",
            tag="v1.0.0",
            commit="a" * 40,
            archive_url=archive.as_uri(),
            archive_sha256=digest or archive_sha256(archive),
            imported_at_utc="2026-07-29T00:00:00Z",
        )

    def test_rejects_archive_sha_mismatch_and_preserves_destination(self):
        archive = self.root / "source.tar.gz"
        make_archive(archive, {"project-v1/file.txt": b"new"})
        destination = self.root / "vendor"
        destination.mkdir()
        (destination / "file.txt").write_bytes(b"old")

        with self.assertRaisesRegex(ValueError, "SHA-256"):
            import_upstream.import_upstream(
                self.lock_for(archive, "0" * 64),
                destination,
                check=False,
                verify_tag=False,
            )

        self.assertEqual((destination / "file.txt").read_bytes(), b"old")
        self.assertFalse(any(self.root.glob("*.tmp")))

    def test_rejects_absolute_and_parent_tar_members(self):
        for member in ("/escape", "project-v1/../escape"):
            with self.subTest(member=member):
                archive = self.root / ("absolute.tar.gz" if member.startswith("/") else "parent.tar.gz")
                make_archive(archive, {member: b"escape"})
                output = self.root / "extract"

                with self.assertRaisesRegex(ValueError, "unsafe archive member"):
                    import_upstream.safe_extract(archive, output)

                self.assertFalse(output.exists())

    def test_rejects_links_and_special_members(self):
        archive = self.root / "links.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            info = tarfile.TarInfo("project-v1/link")
            info.type = tarfile.SYMTYPE
            info.linkname = "target"
            tar.addfile(info)

        with self.assertRaisesRegex(ValueError, "unsupported archive member"):
            import_upstream.safe_extract(archive, self.root / "extract")

    def test_requires_one_archive_wrapper_directory(self):
        archive = self.root / "two-roots.tar.gz"
        make_archive(
            archive,
            {"project-v1/a.txt": b"a", "other-v1/b.txt": b"b"},
        )

        with self.assertRaisesRegex(ValueError, "one wrapper"):
            import_upstream.safe_extract(archive, self.root / "extract")

    def test_replaces_only_destination_tree(self):
        archive = self.root / "source.tar.gz"
        make_archive(
            archive,
            {"project-v1/new.txt": b"new", "project-v1/sub/item": b"item"},
        )
        destination = self.root / "third_party" / "project"
        destination.mkdir(parents=True)
        (destination / "old.txt").write_text("old", encoding="utf-8")
        sibling = destination.parent / "keep.txt"
        sibling.write_text("keep", encoding="utf-8")

        import_upstream.import_upstream(
            self.lock_for(archive), destination, check=False, verify_tag=False
        )

        self.assertFalse((destination / "old.txt").exists())
        self.assertEqual((destination / "new.txt").read_bytes(), b"new")
        self.assertEqual((destination / "sub/item").read_bytes(), b"item")
        self.assertEqual(sibling.read_text(encoding="utf-8"), "keep")

    def test_check_mode_detects_added_removed_and_changed_files(self):
        expected = self.root / "expected"
        actual = self.root / "actual"
        expected.mkdir()
        actual.mkdir()
        (expected / "removed.txt").write_text("removed", encoding="utf-8")
        (expected / "changed.txt").write_text("before", encoding="utf-8")
        (actual / "added.txt").write_text("added", encoding="utf-8")
        (actual / "changed.txt").write_text("after", encoding="utf-8")

        changes = import_upstream.compare_trees(expected, actual)

        self.assertEqual(
            changes,
            ["added: added.txt", "changed: changed.txt", "removed: removed.txt"],
        )

    def test_check_mode_reports_drift_without_modifying_destination(self):
        archive = self.root / "source.tar.gz"
        make_archive(archive, {"project-v1/file.txt": b"official"})
        destination = self.root / "vendor"
        destination.mkdir()
        (destination / "file.txt").write_bytes(b"local")

        with self.assertRaisesRegex(ValueError, "changed: file.txt"):
            import_upstream.import_upstream(
                self.lock_for(archive), destination, check=True, verify_tag=False
            )

        self.assertEqual((destination / "file.txt").read_bytes(), b"local")

    def test_resolves_lightweight_tag_to_commit(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"object": {"type": "commit", "sha": "b" * 40}}
        ).encode()
        with mock.patch("urllib.request.urlopen", return_value=response) as urlopen:
            commit = import_upstream.resolve_tag_commit(
                "https://github.com/example/project", "v1.0.0"
            )

        self.assertEqual(commit, "b" * 40)
        self.assertEqual(urlopen.call_count, 1)

    def test_resolves_annotated_tag_to_commit(self):
        ref_response = mock.MagicMock()
        ref_response.__enter__.return_value.read.return_value = json.dumps(
            {"object": {"type": "tag", "sha": "c" * 40}}
        ).encode()
        tag_response = mock.MagicMock()
        tag_response.__enter__.return_value.read.return_value = json.dumps(
            {"object": {"type": "commit", "sha": "d" * 40}}
        ).encode()
        with mock.patch(
            "urllib.request.urlopen", side_effect=[ref_response, tag_response]
        ) as urlopen:
            commit = import_upstream.resolve_tag_commit(
                "https://github.com/example/project", "v1.0.0"
            )

        self.assertEqual(commit, "d" * 40)
        self.assertEqual(urlopen.call_count, 2)


if __name__ == "__main__":
    unittest.main()
