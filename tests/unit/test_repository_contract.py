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

    def test_metal_runtime_segments_probe_and_model_evidence(self):
        source = (ROOT / "tests/metal_decode.cpp").read_text(encoding="utf-8")
        probe = source.find("run_metal_add_probe(device)")
        reset = source.find("logs.clear()", probe)
        model_load = source.find("llama_model_load_from_file", reset)
        self.assertNotEqual(probe, -1)
        self.assertNotEqual(reset, -1)
        self.assertNotEqual(model_load, -1)
        self.assertLess(probe, reset)
        self.assertLess(reset, model_load)

        self.assertIn("has_positive_metal_model_buffer(logs)", source)
        self.assertIn("has_positive_metal_compute_buffer(logs)", source)
        self.assertIn(
            "model_params.n_gpu_layers = std::numeric_limits<int>::max();",
            source,
        )

    def test_chat_cpu_exposes_only_the_cpu_backend(self):
        source = (
            ROOT / "examples" / "chat-cpu" / "src" / "main.cpp"
        ).read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertNotIn("backend_kind", source)
        self.assertNotIn("cpu|metal", source)
        self.assertNotIn('requested == "metal"', source)
        self.assertIn("model_params.n_gpu_layers = 0;", source)
        self.assertIn('"backend=cpu', source)
        self.assertNotRegex(readme, r"chat-cpu[\s\S]{0,120}mcpp run -- [^\n]+ cpu(?:\s|$)")

    def test_chat_metal_requires_positive_model_and_compute_buffers(self):
        source = (
            ROOT / "examples" / "chat-metal" / "src" / "main.cpp"
        ).read_text(encoding="utf-8")

        self.assertIn("has_positive_metal_model_buffer(logs)", source)
        self.assertIn("has_positive_metal_compute_buffer(logs)", source)
        report = source.index('"backend=" << requested')
        self.assertLess(
            source.index("has_positive_metal_compute_buffer(logs)", 1),
            report,
        )

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

    def test_policy_workflows_are_report_only_and_tag_gated(self):
        upstream_path = ROOT / ".github/workflows/upstream-check.yml"
        self.assertTrue(upstream_path.is_file())
        upstream = upstream_path.read_text(encoding="utf-8")
        for marker in (
            'cron: "17 3 * * 1"',
            "contents: read",
            "python3 tools/check_upstream.py",
            "GITHUB_STEP_SUMMARY",
        ):
            self.assertIn(marker, upstream)
        for forbidden in ("git push", "gh issue", "gh release", "mcpp-index"):
            self.assertNotIn(forbidden, upstream)

        release_path = ROOT / ".github/workflows/release.yml"
        self.assertTrue(release_path.is_file())
        release = release_path.read_text(encoding="utf-8")
        for marker in (
            'tags: ["v*"]',
            "fetch-depth: 0",
            "python3 -m unittest discover",
            "tools/check_release.py",
            "actions/workflows/ci.yml/runs",
            "tools/check_ci_run.py",
            "gh release create",
        ):
            self.assertIn(marker, release)
        self.assertIn("actions: read", release)
        self.assertNotIn("check-runs", release)
        self.assertNotIn("upload-release-asset", release)


if __name__ == "__main__":
    unittest.main()
