#!/usr/bin/env python3

import argparse
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from typing import NamedTuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.import_upstream import OFFICIAL_UPSTREAM_REPOSITORY, load_lock


class Version(NamedTuple):
    """The wrapper version IS upstream's build number.

    `b10069` is llama.cpp b10069, and a wrapper-only fix on the same checkpoint
    adds a revision: `b10069.1`. This replaced a separate semantic version whose
    README had to carry a decoder ring next to it ("Version 0.1.0 maps to
    llama.cpp b10069"); see docs/upstream-update-policy.md and mcpp-index#163.

    Ordering still works — the build number is an integer — so the "must not
    release something older" checks below are unchanged. What is gone is the
    major/minor/patch contract; the drift it used to police is now impossible by
    construction, because the version and the checkpoint are the same string.
    """

    build: int          # upstream llama.cpp build number
    revision: int       # wrapper revision on that checkpoint; 0 when absent

    @classmethod
    def parse(cls, value: str) -> "Version":
        match = re.fullmatch(r"b(\d+)(?:\.([1-9]\d*))?", value)
        if not match:
            raise ValueError(
                f"invalid wrapper version: {value!r} — expected upstream's build "
                f"number (b10069), optionally with a wrapper revision (b10069.1)"
            )
        return cls(int(match.group(1)), int(match.group(2) or 0))

    def __str__(self) -> str:
        return f"b{self.build}" + (f".{self.revision}" if self.revision else "")

    @property
    def upstream_tag(self) -> str:
        """The checkpoint this version names."""
        return f"b{self.build}"


class ReleaseIdentity(NamedTuple):
    version: Version
    upstream_repository: str
    upstream_tag: str
    upstream_commit: str
    archive_url: str
    archive_sha256: str

    @classmethod
    def for_test(cls, version: str, upstream_tag: str) -> "ReleaseIdentity":
        commit = "a" * 40 if upstream_tag == "b10069" else "b" * 40
        repository = "https://github.com/ggml-org/llama.cpp"
        return cls(
            Version.parse(version),
            repository,
            upstream_tag,
            commit,
            f"{repository}/archive/refs/tags/{upstream_tag}.tar.gz",
            "c" * 64,
        )


def _identity_from_text(manifest_text: str, lock_text: str, tag: str) -> ReleaseIdentity:
    manifest = tomllib.loads(manifest_text)
    lock_data = tomllib.loads(lock_text)
    try:
        version_text = manifest["package"]["version"]
        upstream = lock_data["upstream"]
        identity = ReleaseIdentity(
            version=Version.parse(version_text),
            upstream_repository=upstream["repository"].rstrip("/"),
            upstream_tag=upstream["tag"],
            upstream_commit=upstream["commit"],
            archive_url=upstream["archive_url"],
            archive_sha256=upstream["archive_sha256"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"invalid release identity: missing {error}") from error

    # The tag is the version verbatim, which is also upstream's own tag form
    # (ggml-org/llama.cpp tags releases `b10069`, not `vb10069`).
    if tag != str(identity.version):
        raise ValueError(
            f"release tag {tag} does not match package version {identity.version}"
        )
    # The version NAMES the vendored checkpoint. Before, this was a convention
    # the transition rules policed after the fact; now a mismatch is simply not
    # a valid identity.
    if identity.version.upstream_tag != identity.upstream_tag:
        raise ValueError(
            f"package version {identity.version} names checkpoint "
            f"{identity.version.upstream_tag}, but the lock vendors "
            f"{identity.upstream_tag}"
        )
    if identity.upstream_repository != OFFICIAL_UPSTREAM_REPOSITORY:
        raise ValueError(
            f"official upstream repository must be {OFFICIAL_UPSTREAM_REPOSITORY}"
        )
    canonical_archive = (
        f"{identity.upstream_repository}/archive/refs/tags/"
        f"{identity.upstream_tag}.tar.gz"
    )
    if identity.archive_url != canonical_archive:
        raise ValueError(
            f"upstream archive URL must be the immutable tag archive: {canonical_archive}"
        )
    if len(identity.upstream_commit) != 40 or any(
        character not in "0123456789abcdef" for character in identity.upstream_commit
    ):
        raise ValueError("invalid upstream commit")
    if len(identity.archive_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in identity.archive_sha256
    ):
        raise ValueError("invalid upstream archive SHA-256")
    return identity


def load_identity(root: Path, tag: str) -> ReleaseIdentity:
    identity = _identity_from_text(
        (root / "mcpp.toml").read_text(encoding="utf-8"),
        (root / "upstream.lock").read_text(encoding="utf-8"),
        tag,
    )
    lock = load_lock(root / "upstream.lock")
    if (
        lock.tag != identity.upstream_tag
        or lock.commit != identity.upstream_commit
        or lock.archive_url != identity.archive_url
        or lock.archive_sha256 != identity.archive_sha256
    ):
        raise ValueError("upstream lock does not match release identity")
    snapshot = root / "snapshots" / f"{identity.upstream_tag}.json"
    if not snapshot.is_file():
        raise ValueError(f"missing upstream snapshot: {snapshot}")
    return identity


def validate_existing_release(
    current: ReleaseIdentity, existing: ReleaseIdentity
) -> None:
    if current != existing:
        raise ValueError("an existing release tag has an immutable mapping")


def validate_version_transition(
    previous: ReleaseIdentity, current: ReleaseIdentity
) -> None:
    if current.version <= previous.version:
        raise ValueError("release version must increase")
    same_checkpoint = (
        current.upstream_tag == previous.upstream_tag
        and current.upstream_commit == previous.upstream_commit
    )
    if same_checkpoint:
        # Same checkpoint means the same build number, so the only thing left to
        # move is the wrapper revision.
        if current.version.build != previous.version.build:
            raise ValueError(
                "a same-checkpoint release must keep the checkpoint's build number"
            )
        if current.version.revision <= previous.version.revision:
            raise ValueError(
                "a same-checkpoint wrapper fix requires a revision bump "
                "(b10069 -> b10069.1)"
            )
        return
    # A new checkpoint IS a new build number; `_identity_from_text` has already
    # checked that the version names the checkpoint actually vendored, so the
    # only thing left is that releases move forward.
    if current.version.build <= previous.version.build:
        raise ValueError(
            "a changed upstream checkpoint requires its own, higher build number"
        )


def validate_vendor_boundary(root: Path) -> None:
    if (root / ".gitmodules").exists():
        raise ValueError("release source must not use a submodule")
    vendor = root / "third_party/llama.cpp"
    if not vendor.is_dir():
        raise ValueError("vendored llama.cpp source is missing")
    if any(path.name == ".git" for path in vendor.rglob(".git")):
        raise ValueError("vendored source contains Git metadata")


def run_repository_checks(
    root: Path,
    upstream_tag: str,
    *,
    runner=subprocess.run,
) -> None:
    commands = (
        [
            sys.executable,
            "tools/import_upstream.py",
            "--lock",
            "upstream.lock",
            "--check",
            "--verify-tag",
        ],
        [sys.executable, "tools/gen_exports.py", "--check"],
        [
            sys.executable,
            "tools/audit_snapshot.py",
            "--check",
            f"snapshots/{upstream_tag}.json",
        ],
    )
    for command in commands:
        runner(command, cwd=root, check=True)


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )


def _tag_identity(root: Path, tag: str) -> ReleaseIdentity | None:
    reference = _git(root, "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}")
    if reference.returncode != 0:
        return None
    manifest = _git(root, "show", f"refs/tags/{tag}:mcpp.toml")
    lock = _git(root, "show", f"refs/tags/{tag}:upstream.lock")
    if manifest.returncode != 0 or lock.returncode != 0:
        raise ValueError(f"existing release tag {tag} has an incomplete identity")
    return _identity_from_text(manifest.stdout, lock.stdout, tag)


def _previous_identity(
    root: Path, current: ReleaseIdentity, current_tag: str
) -> ReleaseIdentity | None:
    tags = _git(root, "tag", "--list", "v[0-9]*", "--sort=-version:refname")
    if tags.returncode != 0:
        raise ValueError(f"cannot list release tags: {tags.stderr.strip()}")
    for tag in tags.stdout.splitlines():
        if tag == current_tag:
            continue
        try:
            version = Version.parse(tag)
        except ValueError:
            continue
        if version > current.version:
            raise ValueError(f"newer release already exists: {tag}")
        if version == current.version:
            raise ValueError(f"release version already exists under tag {tag}")
        if version < current.version:
            identity = _tag_identity(root, tag)
            if identity:
                return identity
    return None


def format_mapping(identity: ReleaseIdentity) -> str:
    return (
        f"llama.cpp-m {identity.version} -> llama.cpp {identity.upstream_tag} "
        f"({identity.upstream_commit})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify an immutable llama.cpp-m release mapping"
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()

    current = load_identity(root, args.tag)
    validate_vendor_boundary(root)
    if existing := _tag_identity(root, args.tag):
        validate_existing_release(current, existing)
    if previous := _previous_identity(root, current, args.tag):
        validate_version_transition(previous, current)
    run_repository_checks(root, current.upstream_tag)
    print(format_mapping(current))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
