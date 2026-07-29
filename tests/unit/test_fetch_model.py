"""Tests for immutable, atomic llama.cpp GGUF provisioning."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "support" / "fetch_model.py"
SPEC = importlib.util.spec_from_file_location("fetch_model", SCRIPT)
assert SPEC and SPEC.loader
fetch_model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch_model)


class TestFetchModel(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.payload = b"pinned gguf fixture\0" * 32
        self.source = self.root / "source.gguf"
        self.source.write_bytes(self.payload)
        self.digest = hashlib.sha256(self.payload).hexdigest()

    def fetch(self, output: Path, *, size=None, digest=None, url=None):
        return fetch_model.fetch(
            str(output),
            url=url or self.source.as_uri(),
            expected_size=len(self.payload) if size is None else size,
            expected_sha256=self.digest if digest is None else digest,
        )

    def test_pins_immutable_model_identity(self):
        self.assertEqual(
            fetch_model._MODEL_URL,
            "https://huggingface.co/ggml-org/models-moved/resolve/"
            "499bc8821c6b12b4e53c5bffcb21ec206f212d81/"
            "tinyllamas/stories15M-q4_0.gguf",
        )
        self.assertEqual(fetch_model._MODEL_SIZE, 19077344)
        self.assertEqual(
            fetch_model._MODEL_SHA256,
            "66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739",
        )

    def test_accepts_local_source_through_fetch_path(self):
        output = self.root / "models" / "model.gguf"
        result = self.fetch(output)
        self.assertTrue(Path(result).is_absolute())
        self.assertTrue(Path(result).samefile(output))
        self.assertEqual(output.read_bytes(), self.payload)
        self.assertFalse(Path(f"{output}.tmp").exists())

    def test_rejects_wrong_size_without_replacing_output(self):
        output = self.root / "model.gguf"
        output.write_bytes(b"existing")
        with self.assertRaisesRegex(RuntimeError, "size"):
            self.fetch(output, size=len(self.payload) + 1)
        self.assertEqual(output.read_bytes(), b"existing")
        self.assertFalse(Path(f"{output}.tmp").exists())

    def test_rejects_wrong_digest_without_replacing_output(self):
        output = self.root / "model.gguf"
        output.write_bytes(b"existing")
        with self.assertRaisesRegex(RuntimeError, "SHA-256"):
            self.fetch(output, digest="0" * 64)
        self.assertEqual(output.read_bytes(), b"existing")
        self.assertFalse(Path(f"{output}.tmp").exists())

    def test_reuses_only_valid_existing_output(self):
        output = self.root / "model.gguf"
        output.write_bytes(self.payload)
        with mock.patch.object(fetch_model.urllib.request, "urlopen") as urlopen:
            result = self.fetch(output, url="https://invalid.example/model.gguf")
        urlopen.assert_not_called()
        self.assertTrue(Path(result).is_absolute())
        self.assertTrue(Path(result).samefile(output))

    def test_invalid_output_is_atomically_replaced(self):
        output = self.root / "model.gguf"
        output.write_bytes(b"existing")
        self.fetch(output)
        self.assertEqual(output.read_bytes(), self.payload)
        self.assertFalse(Path(f"{output}.tmp").exists())

    def test_streams_and_cleans_partial_temp_on_network_failure(self):
        timeout = getattr(fetch_model, "_DOWNLOAD_TIMEOUT", None)
        self.assertIsNotNone(timeout)
        output = self.root / "model.gguf"
        unrelated_tmp = Path(f"{output}.tmp")
        unrelated_tmp.write_bytes(b"another invocation")
        response = io.BytesIO(self.payload)
        with mock.patch.object(
            fetch_model.urllib.request, "urlopen", return_value=response
        ) as urlopen:
            self.fetch(output, url="https://example.invalid/model.gguf")
        urlopen.assert_called_once_with(
            "https://example.invalid/model.gguf", timeout=timeout
        )
        self.assertEqual(output.read_bytes(), self.payload)
        self.assertEqual(unrelated_tmp.read_bytes(), b"another invocation")
        self.assertEqual(list(self.root.glob(".model.gguf.*.tmp")), [])

        with mock.patch.object(
            fetch_model.urllib.request,
            "urlopen",
            side_effect=urllib.error.URLError("network down"),
        ):
            with self.assertRaisesRegex(urllib.error.URLError, "network down"):
                self.fetch(
                    output,
                    url="https://example.invalid/model.gguf",
                    digest="1" * 64,
                )
        self.assertEqual(unrelated_tmp.read_bytes(), b"another invocation")
        self.assertEqual(list(self.root.glob(".model.gguf.*.tmp")), [])

    def test_retries_transient_network_failure(self):
        output = self.root / "model.gguf"
        with mock.patch.object(
            fetch_model.urllib.request,
            "urlopen",
            side_effect=[
                urllib.error.URLError("TLS EOF"),
                io.BytesIO(self.payload),
            ],
        ) as urlopen, mock.patch.object(fetch_model.time, "sleep") as sleep:
            self.fetch(output, url="https://example.invalid/model.gguf")

        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once()
        self.assertEqual(output.read_bytes(), self.payload)
        self.assertEqual(list(self.root.glob(".model.gguf.*.tmp")), [])

    def test_does_not_retry_stable_http_client_error(self):
        output = self.root / "model.gguf"
        error = urllib.error.HTTPError(
            "https://example.invalid/model.gguf", 404, "Not Found", {}, None
        )
        self.addCleanup(error.close)
        with mock.patch.object(
            fetch_model.urllib.request, "urlopen", side_effect=error
        ) as urlopen, mock.patch.object(fetch_model.time, "sleep") as sleep:
            with self.assertRaises(urllib.error.HTTPError):
                self.fetch(output, url="https://example.invalid/model.gguf")

        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()

    def test_rejects_response_larger_than_pinned_size(self):
        output = self.root / "model.gguf"
        oversized = io.BytesIO(self.payload + b"unexpected trailing bytes")
        with mock.patch.object(
            fetch_model.urllib.request, "urlopen", return_value=oversized
        ):
            with self.assertRaisesRegex(RuntimeError, "exceeds expected size"):
                self.fetch(
                    output,
                    url="https://example.invalid/model.gguf",
                    size=len(self.payload),
                )
        self.assertFalse(output.exists())
        self.assertFalse(Path(f"{output}.tmp").exists())

    def test_self_test_exercises_fetch(self):
        stderr = io.StringIO()
        with mock.patch.object(
            fetch_model, "fetch", wraps=fetch_model.fetch
        ) as fetch, contextlib.redirect_stderr(stderr):
            fetch_model.self_test()
        self.assertGreaterEqual(fetch.call_count, 4)
        self.assertIn("Self-test passed", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
