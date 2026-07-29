#!/usr/bin/env python3
"""Fetch the pinned GGUF test model with cryptographic verification."""
import argparse
import hashlib
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


_MODEL_URL = (
    "https://huggingface.co/ggml-org/models-moved/resolve/"
    "499bc8821c6b12b4e53c5bffcb21ec206f212d81/tinyllamas/"
    "stories15M-q4_0.gguf"
)
_MODEL_SIZE = 19077344
_MODEL_SHA256 = "66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739"
_DOWNLOAD_TIMEOUT = 60
_DOWNLOAD_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 1


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(
    output: str,
    *,
    url: str = _MODEL_URL,
    expected_size: int = _MODEL_SIZE,
    expected_sha256: str = _MODEL_SHA256,
) -> str:
    output = os.path.abspath(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    if os.path.isfile(output):
        size = os.path.getsize(output)
        digest = sha256_file(output)
        if size == expected_size and digest == expected_sha256:
            print(f"Model already present and verified: {output}", file=sys.stderr)
            return output
        print(
            f"Existing file {output} (size={size}, sha256={digest}) "
            f"does not match expected (size={expected_size}, "
            f"sha256={expected_sha256}); re-downloading",
            file=sys.stderr,
        )

    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{os.path.basename(output)}.",
            suffix=".tmp",
            dir=os.path.dirname(output),
        )
        try:
            print(
                f"Downloading {url} (attempt {attempt}/{_DOWNLOAD_ATTEMPTS}) ...",
                file=sys.stderr,
            )
            downloaded = 0
            with os.fdopen(descriptor, "wb") as target:
                descriptor = -1
                with urllib.request.urlopen(
                    url, timeout=_DOWNLOAD_TIMEOUT
                ) as response:
                    while chunk := response.read(
                        min(1 << 20, expected_size - downloaded + 1)
                    ):
                        downloaded += len(chunk)
                        if downloaded > expected_size:
                            raise RuntimeError(
                                "Downloaded model exceeds expected size "
                                f"{expected_size}"
                            )
                        target.write(chunk)
            actual_size = os.path.getsize(temporary)
            if actual_size != expected_size:
                raise RuntimeError(
                    f"Downloaded model size {actual_size} != expected "
                    f"{expected_size}"
                )
            actual_sha = sha256_file(temporary)
            if actual_sha != expected_sha256:
                raise RuntimeError(
                    f"Downloaded model SHA-256 {actual_sha} != expected "
                    f"{expected_sha256}"
                )
            os.replace(temporary, output)
            print(f"Model verified and saved to {output}", file=sys.stderr)
            break
        except OSError as error:
            if descriptor >= 0:
                os.close(descriptor)
            if os.path.isfile(temporary):
                os.unlink(temporary)
            if attempt == _DOWNLOAD_ATTEMPTS:
                raise
            print(
                f"Download failed: {error}; retrying",
                file=sys.stderr,
            )
            time.sleep(_RETRY_DELAY_SECONDS)
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            if os.path.isfile(temporary):
                os.unlink(temporary)
            raise
    return output


def self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        payload = b"local pinned model payload\0" * 64
        source = os.path.join(directory, "source.gguf")
        with open(source, "wb") as stream:
            stream.write(payload)
        url = Path(source).as_uri()
        size = len(payload)
        digest = hashlib.sha256(payload).hexdigest()

        output = os.path.join(directory, "model.gguf")
        fetch(output, url=url, expected_size=size, expected_sha256=digest)
        fetch(
            output,
            url="https://invalid.example/reuse.gguf",
            expected_size=size,
            expected_sha256=digest,
        )

        for label, rejected_size, rejected_digest in (
            ("size", size + 1, digest),
            ("digest", size, "0" * 64),
        ):
            rejected = os.path.join(directory, f"reject-{label}.gguf")
            try:
                fetch(
                    rejected,
                    url=url,
                    expected_size=rejected_size,
                    expected_sha256=rejected_digest,
                )
            except RuntimeError:
                if os.path.exists(rejected) or os.path.exists(rejected + ".tmp"):
                    raise AssertionError(f"{label} rejection left output bytes")
            else:
                raise AssertionError(f"{label} mismatch was accepted")

        with open(output, "wb") as stream:
            stream.write(b"invalid existing output")
        fetch(output, url=url, expected_size=size, expected_sha256=digest)
        if sha256_file(output) != digest or os.path.exists(output + ".tmp"):
            raise AssertionError("atomic replacement did not produce verified output")

        print("Self-test passed", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch pinned GGUF test model")
    parser.add_argument("--output", help="Output path for the model file")
    parser.add_argument(
        "--self-test", action="store_true", help="Run local self-test only"
    )
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.output:
        parser.error("--output is required (or use --self-test)")
    path = fetch(args.output)
    print(os.path.abspath(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
