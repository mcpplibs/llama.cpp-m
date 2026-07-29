#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import tomllib
import urllib.parse
import urllib.request
import uuid


@dataclass(frozen=True)
class UpstreamLock:
    repository: str
    tag: str
    commit: str
    archive_url: str
    archive_sha256: str
    imported_at_utc: str


def load_lock(path: Path) -> UpstreamLock:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    try:
        upstream = data["upstream"]
        lock = UpstreamLock(
            repository=upstream["repository"],
            tag=upstream["tag"],
            commit=upstream["commit"],
            archive_url=upstream["archive_url"],
            archive_sha256=upstream["archive_sha256"],
            imported_at_utc=upstream["imported_at_utc"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"invalid upstream lock: missing {error}") from error
    if len(lock.commit) != 40 or any(c not in "0123456789abcdef" for c in lock.commit):
        raise ValueError("invalid upstream commit")
    if len(lock.archive_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in lock.archive_sha256
    ):
        raise ValueError("invalid upstream archive SHA-256")
    return lock


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    url: str,
    output: Path,
    timeout: int = 120,
    max_bytes: int = 536870912,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    total = 0
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "llama.cpp-m-importer"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(
                            f"download exceeds maximum size of {max_bytes} bytes"
                        )
                    stream.write(chunk)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _validated_members(archive: tarfile.TarFile) -> tuple[list[tarfile.TarInfo], str]:
    members = archive.getmembers()
    if not members:
        raise ValueError("archive is empty")

    roots: set[str] = set()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError(f"unsafe archive member: {member.name}")
        if not (member.isdir() or member.isfile()):
            raise ValueError(f"unsupported archive member: {member.name}")
        roots.add(path.parts[0])

    if len(roots) != 1:
        raise ValueError("archive must contain exactly one wrapper directory")
    wrapper = next(iter(roots))
    if any(len(PurePosixPath(member.name).parts) == 1 and member.isfile() for member in members):
        raise ValueError("archive root must be one wrapper directory")
    return members, wrapper


def safe_extract(archive: Path, output: Path) -> Path:
    if output.exists():
        raise ValueError(f"extraction output already exists: {output}")

    with tarfile.open(archive, "r:*") as source:
        members, wrapper = _validated_members(source)
        output.mkdir(parents=True)
        try:
            for member in members:
                destination = output.joinpath(*PurePosixPath(member.name).parts)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                extracted = source.extractfile(member)
                if extracted is None:
                    raise ValueError(f"cannot read archive member: {member.name}")
                with extracted, destination.open("wb") as target:
                    shutil.copyfileobj(extracted, target)
                os.chmod(destination, member.mode & 0o777)
        except Exception:
            shutil.rmtree(output, ignore_errors=True)
            raise
    return output / wrapper


def _github_repository(repository: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(repository)
    if parsed.scheme != "https" or parsed.netloc.lower() != "github.com":
        raise ValueError(f"unsupported GitHub repository URL: {repository}")
    parts = [part for part in parsed.path.removesuffix(".git").split("/") if part]
    if len(parts) != 2:
        raise ValueError(f"unsupported GitHub repository URL: {repository}")
    return parts[0], parts[1]


def _github_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "llama.cpp-m-importer",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def resolve_tag_commit(repository: str, tag: str) -> str:
    owner, name = _github_repository(repository)
    api = f"https://api.github.com/repos/{owner}/{name}/git"
    quoted_tag = urllib.parse.quote(tag, safe="")
    obj = _github_json(f"{api}/ref/tags/{quoted_tag}")["object"]
    seen: set[str] = set()
    while obj["type"] == "tag":
        sha = obj["sha"]
        if sha in seen:
            raise ValueError("annotated tag cycle detected")
        seen.add(sha)
        obj = _github_json(f"{api}/tags/{sha}")["object"]
    if obj["type"] != "commit":
        raise ValueError(f"tag resolves to unsupported object type: {obj['type']}")
    return obj["sha"]


def _tree_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        return {}
    return {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
    }


def compare_trees(expected: Path, actual: Path) -> list[str]:
    expected_files = _tree_files(expected)
    actual_files = _tree_files(actual)
    changes = [f"added: {name}" for name in actual_files.keys() - expected_files.keys()]
    changes.extend(
        f"changed: {name}"
        for name in expected_files.keys() & actual_files.keys()
        if sha256_file(expected_files[name]) != sha256_file(actual_files[name])
    )
    changes.extend(f"removed: {name}" for name in expected_files.keys() - actual_files.keys())
    return sorted(changes)


def _replace_tree(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = uuid.uuid4().hex
    staging = destination.parent / f".{destination.name}.{suffix}.tmp"
    backup = destination.parent / f".{destination.name}.{suffix}.backup"
    shutil.copytree(source, staging)
    moved_existing = False
    try:
        if destination.exists():
            os.replace(destination, backup)
            moved_existing = True
        try:
            os.replace(staging, destination)
        except Exception:
            if moved_existing:
                os.replace(backup, destination)
            raise
        if moved_existing:
            shutil.rmtree(backup)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if backup.exists() and destination.exists():
            shutil.rmtree(backup, ignore_errors=True)


def import_upstream(
    lock: UpstreamLock,
    destination: Path,
    check: bool,
    verify_tag: bool,
) -> None:
    if verify_tag:
        resolved = resolve_tag_commit(lock.repository, lock.tag)
        if resolved != lock.commit:
            raise ValueError(
                f"tag {lock.tag} resolves to {resolved}, expected {lock.commit}"
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.import-", dir=destination.parent
    ) as temporary:
        work = Path(temporary)
        archive = work / "upstream.tar.gz"
        download(lock.archive_url, archive)
        actual_sha256 = sha256_file(archive)
        if actual_sha256 != lock.archive_sha256:
            raise ValueError(
                f"archive SHA-256 mismatch: expected {lock.archive_sha256}, "
                f"got {actual_sha256}"
            )
        wrapper = safe_extract(archive, work / "extracted")

        if check:
            changes = compare_trees(wrapper, destination)
            if changes:
                raise ValueError("snapshot differs:\n" + "\n".join(changes))
            print("Snapshot matches.")
            return

        _replace_tree(wrapper, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description="Import a pinned llama.cpp archive")
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path("third_party/llama.cpp"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--verify-tag", action="store_true")
    args = parser.parse_args()

    import_upstream(
        load_lock(args.lock),
        args.destination,
        check=args.check,
        verify_tag=args.verify_tag,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
