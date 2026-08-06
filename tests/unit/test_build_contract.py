from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "mcpp.toml"
BUILD_HELPER = ROOT / "build.mcpp"
UPSTREAM = ROOT / "third_party/llama.cpp"
SNAPSHOT = ROOT / "snapshots/b10069.json"

GENERATED_SOURCE_MAP = {
    "generated/ggml_cpp.cpp": "ggml/src/ggml.cpp",
    "generated/ggml-cpu_cpp.cpp": "ggml/src/ggml-cpu/ggml-cpu.cpp",
    "generated/ggml_metal_device_m.m":
        "ggml/src/ggml-metal/ggml-metal-device.m",
}


def load_manifest() -> dict:
    with MANIFEST.open("rb") as stream:
        return tomllib.load(stream)


def upstream_source(path: str) -> str | None:
    if path in GENERATED_SOURCE_MAP:
        return GENERATED_SOURCE_MAP[path]
    prefix = "third_party/llama.cpp/"
    if path.startswith(prefix):
        return path.removeprefix(prefix)
    return None


def translation_units(paths: list[str]) -> set[str]:
    return {
        normalized
        for path in paths
        if (normalized := upstream_source(path)) is not None
        and Path(normalized).suffix in {".c", ".cc", ".cpp", ".m", ".mm"}
    }


def snapshot_sources(snapshot: dict, *groups: str) -> set[str]:
    sources: set[str] = set()
    for group in groups:
        for path in snapshot["sources"][group]:
            if group == "models":
                path = f"src/{path}"
            if Path(path).suffix in {".c", ".cc", ".cpp", ".m", ".mm"}:
                sources.add(path)
    return sources


class BuildManifestContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_manifest()
        cls.snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

    def base_sources(self) -> list[str]:
        return self.manifest["build"]["sources"]

    def target_sources(self, predicate: str) -> list[str]:
        return self.manifest["target"][predicate]["build"]["sources"]

    def test_identity_and_public_module_are_cxx23(self):
        package = self.manifest["package"]
        self.assertEqual(package["name"], "llamacpp")
        # The namespace names whose library this is (mcpp-index#163); the version
        # IS the upstream checkpoint, so it must agree with the build info the
        # manifest generates for ggml.
        self.assertEqual(package["namespace"], "ggml-org")
        self.assertEqual(package["version"], "b10069")
        self.assertIn(
            f'#define GGML_VERSION "{package["version"]}"',
            self.manifest["generated_files"]["generated/ggml_build_info.h"],
        )
        self.assertEqual(package["standard"], "c++23")
        self.assertIn("src/llamacpp.cppm", self.base_sources())
        self.assertEqual(self.manifest["targets"], {"llama": {"kind": "lib"}})

    def test_default_sources_match_the_audited_cpu_and_model_boundary(self):
        expected = snapshot_sources(
            self.snapshot,
            "ggml_base",
            "ggml_registry",
            "ggml_cpu_common",
            "llama_core",
            "models",
        )
        self.assertEqual(translation_units(self.base_sources()), expected)

    def test_target_architecture_sources_are_exact_and_disjoint(self):
        for arch, group, rejected in (
            ("x86_64", "ggml_cpu_x86", "ggml_cpu_arm"),
            ("aarch64", "ggml_cpu_arm", "ggml_cpu_x86"),
        ):
            predicate = f'cfg(arch = "{arch}")'
            selected = translation_units(
                self.base_sources() + self.target_sources(predicate)
            )
            self.assertEqual(
                selected,
                snapshot_sources(
                    self.snapshot,
                    "ggml_base",
                    "ggml_registry",
                    "ggml_cpu_common",
                    "llama_core",
                    "models",
                    group,
                ),
            )
            self.assertTrue(
                selected.isdisjoint(snapshot_sources(self.snapshot, rejected))
            )

    def test_every_model_translation_unit_is_selected(self):
        models = {
            path.removeprefix("third_party/llama.cpp/src/models/")
            for path in self.base_sources()
            if path.startswith("third_party/llama.cpp/src/models/")
        }
        self.assertEqual(models, {
            Path(path).name for path in self.snapshot["sources"]["models"]
        })
        self.assertIn("qwen35.cpp", models)

    def test_cpu_is_default_and_metal_is_the_only_optional_backend(self):
        features = self.manifest["features"]
        self.assertEqual(set(features), {
            "default", "backend-cpu", "backend-metal"
        })
        default = features["default"]
        implied = default.get("implies", default)
        self.assertEqual(implied, ["backend-cpu"])
        self.assertEqual(features["backend-cpu"], {})
        metal = features["backend-metal"]
        self.assertEqual(
            translation_units(metal["sources"]),
            snapshot_sources(self.snapshot, "ggml_metal"),
        )

    def test_six_targeted_cxx20_overrides_are_preserved(self):
        actual = sorted(
            upstream_source(rule["glob"])
            for rule in self.manifest["build"]["flags"]
            if "-std=c++20" in rule.get("cxxflags", [])
        )
        self.assertEqual(
            actual, self.snapshot["dialect_exceptions"]["c++20"]
        )

    def test_paths_are_repository_relative_without_archive_wrapper_globs(self):
        paths = list(self.base_sources())
        for target in self.manifest.get("target", {}).values():
            paths.extend(target.get("build", {}).get("sources", []))
        paths.extend(self.manifest["features"]["backend-metal"]["sources"])
        paths.extend(self.manifest["build"]["include_dirs"])
        for path in paths:
            with self.subTest(path=path):
                self.assertFalse(Path(path).is_absolute())
                self.assertNotIn("*/", path)
                self.assertNotIn("..", Path(path).parts)


class BuildHelperContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = BUILD_HELPER.read_text(encoding="utf-8")
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        if compiler is None:
            raise unittest.SkipTest("host C++ compiler is unavailable")
        cls.temp = tempfile.TemporaryDirectory(prefix="llamacpp-build-helper-")
        cls.executable = Path(cls.temp.name) / "build-helper"
        completed = subprocess.run(
            [
                compiler,
                "-std=c++20",
                "-x",
                "c++",
                str(BUILD_HELPER),
                "-o",
                cls.executable,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "temp"):
            cls.temp.cleanup()

    def run_helper(
        self, *, features: tuple[str, ...], target_os: str, target_arch: str
    ) -> subprocess.CompletedProcess[str]:
        output = Path(self.temp.name) / "out"
        output.mkdir(exist_ok=True)
        env = os.environ.copy()
        for name in tuple(env):
            if name == "MCPP_FEATURES" or name.startswith("MCPP_FEATURE_"):
                env.pop(name)
        env.update({
            "MCPP_FEATURES": ",".join(features),
            "MCPP_TARGET_OS": target_os,
            "MCPP_TARGET_ARCH": target_arch,
            "MCPP_MANIFEST_DIR": str(ROOT),
            "MCPP_OUT_DIR": str(output),
        })
        for feature in features:
            env[f"MCPP_FEATURE_{feature.upper().replace('-', '_')}"] = "1"
        return subprocess.run(
            [self.executable],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_contains_no_subprocess_launch_api(self):
        self.assertNotRegex(
            self.source,
            r"\b(system|popen|exec[lv]?[pe]?|posix_spawn[p]?)\s*\(",
        )

    def test_accepts_only_cpu_and_metal_features(self):
        accepted = self.run_helper(
            features=("backend-cpu",), target_os="linux", target_arch="x86_64"
        )
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        rejected = self.run_helper(
            features=("backend-cpu", "backend-vulkan"),
            target_os="linux",
            target_arch="x86_64",
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("unsupported feature 'backend-vulkan'", rejected.stderr)

    def test_metal_is_rejected_outside_macos_aarch64(self):
        for target_os, target_arch in (
            ("linux", "aarch64"),
            ("macos", "x86_64"),
        ):
            result = self.run_helper(
                features=("backend-cpu", "backend-metal"),
                target_os=target_os,
                target_arch=target_arch,
            )
            with self.subTest(target_os=target_os, target_arch=target_arch):
                self.assertNotEqual(result.returncode, 0)

    def test_metal_assembly_has_one_embedded_library_boundary(self):
        result = self.run_helper(
            features=("backend-cpu", "backend-metal"),
            target_os="macos",
            target_arch="aarch64",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        match = re.search(r"(?m)^mcpp:generated=(.+)$", result.stdout)
        self.assertIsNotNone(match, result.stdout)
        assembly = Path(match.group(1)).read_text(encoding="utf-8")
        self.assertEqual(assembly.count("__DATA,__ggml_metallib"), 1)
        for symbol in ("_ggml_metallib_start", "_ggml_metallib_end"):
            self.assertEqual(
                len(re.findall(rf"(?m)^\.globl {re.escape(symbol)}$", assembly)),
                1,
                symbol,
            )
            self.assertEqual(
                len(re.findall(rf"(?m)^{re.escape(symbol)}:$", assembly)),
                1,
                symbol,
            )


if __name__ == "__main__":
    unittest.main()
