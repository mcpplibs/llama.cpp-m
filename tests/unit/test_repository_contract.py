from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]


class RepositoryContractTest(unittest.TestCase):
    def test_required_package_files_exist(self):
        for name in (
            "mcpp.toml", "build.mcpp", "upstream.lock",
            "src/llamacpp.cppm", "third_party/llama.cpp/include/llama.h",
        ):
            self.assertTrue((ROOT / name).is_file(), name)

    def test_no_submodule_or_consumer_time_upstream_fetch(self):
        self.assertFalse((ROOT / ".gitmodules").exists())
        build_helper = (ROOT / "build.mcpp").read_text(encoding="utf-8")
        self.assertNotRegex(
            build_helper,
            r"\b(system|popen|exec[lv]?[pe]?|posix_spawn[p]?)\s*\(",
        )
        self.assertNotIn("urllib", build_helper)
        self.assertNotIn("curl", build_helper)

    def test_package_and_module_identity(self):
        manifest = (ROOT / "mcpp.toml").read_text(encoding="utf-8")
        module = (ROOT / "src/llamacpp.cppm").read_text(encoding="utf-8")
        self.assertRegex(manifest, r'(?m)^name\s*=\s*"llamacpp"$')
        self.assertRegex(manifest, r'(?m)^version\s*=\s*"0\.1\.0"$')
        self.assertIn("export module llamacpp;", module)

    def test_vendored_tree_is_not_locally_patched(self):
        patches = ROOT / "patches"
        self.assertFalse(patches.exists() and any(patches.iterdir()))

    def test_metal_runtime_decodes_and_samples_before_cleanup(self):
        source = (ROOT / "tests/metal_decode.cpp").read_text(encoding="utf-8")
        calls = (
            "llama_decode(",
            "llama_sampler_chain_init(",
            "llama_sampler_chain_add(",
            "llama_sampler_sample(",
            "llama_sampler_free(",
            "llama_free(context)",
        )
        positions = []
        for call in calls:
            start = positions[-1] + 1 if positions else 0
            position = source.find(call, start)
            self.assertNotEqual(position, -1, f"missing runtime call: {call}")
            positions.append(position)

        self.assertIn("sampled < 0 || sampled >= vocabulary_size", source)

    def test_chat_examples_keep_next_token_alive_and_report_runtime_errors(self):
        for example in ("chat-cpu", "chat-metal"):
            source = (
                ROOT / "examples" / example / "src" / "main.cpp"
            ).read_text(encoding="utf-8")
            self.assertRegex(
                source,
                r"llama_token\s+sampled\s*=\s*LLAMA_TOKEN_NULL;"
                r"[\s\S]*for\s*\([^)]*\)\s*\{"
                r"[\s\S]*sampled\s*=\s*llama_sampler_sample",
            )
            self.assertNotRegex(
                source,
                r"for\s*\([^)]*\)\s*\{[\s\S]*?"
                r"llama_token\s+sampled\s*=\s*llama_sampler_sample",
            )
            self.assertIn("int exit_code = 0;", source)
            self.assertRegex(
                source,
                r"decode failed[\s\S]*exit_code\s*=\s*[1-9][0-9]*;",
            )
            self.assertRegex(
                source,
                r"failed to render sampled token"
                r"[\s\S]*exit_code\s*=\s*[1-9][0-9]*;",
            )
            self.assertIn("return exit_code;", source)

    def test_ci_covers_supported_platform_and_runtime_boundaries(self):
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        for marker in (
            "ubuntu-24.04",
            "ubuntu-24.04-arm",
            "windows-latest",
            "macos-15",
            "LLAMACPP_CPU_TEST=PASS",
            "LLAMACPP_METAL_TEST=PASS",
            "build.mcpp.bin",
            "PT_INTERP",
            "compile_commands.json",
        ):
            self.assertIn(marker, workflow)

        self.assertIn("mirrors.ustc.edu.cn/ubuntu-ports", workflow)
        self.assertIn("--retry 5 --retry-all-errors", workflow)
        self.assertIn("MCPP_VENDORED_XLINGS", workflow)
        self.assertNotIn("set -o pipefail", workflow)
        self.assertGreaterEqual(workflow.count("set -euo pipefail"), 5)


if __name__ == "__main__":
    unittest.main()
