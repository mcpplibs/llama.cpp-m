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
BUILD_HELPER_BINARY = ROOT / "target/.build-mcpp/build.mcpp.bin"
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

    def test_cpu_is_default_and_the_optional_backends_are_metal_and_vulkan(self):
        features = self.manifest["features"]
        self.assertEqual(set(features), {
            "default", "backend-cpu", "backend-metal", "backend-vulkan"
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

    def test_vulkan_is_additive_and_costs_a_cpu_consumer_nothing(self):
        """A consumer that does not name the feature acquires none of it.

        The dependencies and the tools a Vulkan build needs are declared under
        the feature's own tables, never unconditionally. Asserting only that
        the feature works would pass on a manifest that made every consumer
        download a shader compiler.
        """
        vulkan = self.manifest["features"]["backend-vulkan"]
        self.assertEqual(
            vulkan["sources"],
            ["third_party/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp"],
        )

        # The registry compiles a backend in only when it is told the backend
        # exists; without this the library builds, links, and reports no
        # Vulkan device -- a failure with no error in it.
        registry_defines = [
            rule["defines"]
            for rule in vulkan["flags"]
            if rule["glob"].endswith("ggml-backend-reg.cpp")
        ]
        self.assertEqual(registry_defines, [["GGML_USE_VULKAN"]])

        feature_deps = self.manifest["feature-deps"]["backend-vulkan"]
        self.assertEqual(set(feature_deps["compat"]), {
            "vulkan", "spirv-headers", "vulkan-runtime"
        })
        self.assertNotIn("compat", self.manifest.get("dependencies", {}))

        feature_tools = self.manifest["feature-xlings"]["backend-vulkan"]
        self.assertIn("xim:shaderc", feature_tools)
        # A Vulkan device that needs no GPU exists for THIS repository's own
        # test. `dev` is the one tier that does not propagate, so a consumer
        # of the feature does not download it.
        self.assertEqual(feature_tools["xim:mesa-lavapipe"]["when"], "dev")
        self.assertEqual(self.manifest.get("xlings", {}).get("workspace", {}), {})

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
        paths.extend(self.manifest["features"]["backend-vulkan"]["sources"])
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
        # THE HELPER IS COMPILED BY mcpp, NOT BY THIS TEST.
        #
        # `build.mcpp` is a modules program -- `import std; import mcpp;` --
        # and the `mcpp` module is bundled in the mcpp binary and staged on
        # demand. A plain `c++ -x c++ build.mcpp` therefore cannot compile it,
        # and a hand-written stand-in for that module would be a second
        # implementation of the wire protocol, which is the thing the module
        # exists to prevent this package from owning.
        #
        # So the test uses the binary mcpp already produced. It is statically
        # linked, so running it outside a build is exactly what this class
        # needs: the same program, with an environment this test chooses.
        cls.temp = tempfile.TemporaryDirectory(prefix="llamacpp-build-helper-")
        cls.executable = BUILD_HELPER_BINARY
        if not cls.executable.exists():
            message = (
                f"{cls.executable} is absent -- run `mcpp build` first so mcpp "
                "compiles the build program"
            )
            # A SKIP AND A FAILURE MUST NOT READ THE SAME. On a developer's
            # machine the binary may legitimately not exist yet; in CI its
            # absence means the job ran in the wrong order and these contracts
            # measured nothing. The variable is what tells the two apart.
            if os.environ.get("LLAMACPP_REQUIRE_BUILD_HELPER") == "1":
                raise AssertionError(message)
            raise unittest.SkipTest(message)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "temp"):
            cls.temp.cleanup()

    def run_helper(
        self,
        *,
        features: tuple[str, ...],
        target_os: str,
        target_arch: str,
        extra_env: dict[str, str] | None = None,
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
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            [self.executable],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_the_only_subprocess_is_the_capability_probe(self):
        """The build program asks the machine a question; it does not build.

        This used to be a blanket ban on every process-launching call, written
        when the only backend needing help was Metal and that help was pure
        file manipulation. The ban was the wrong shape rather than too strict:
        what it protected against is a build program PERFORMING the work, and
        what it forbade was also the one thing such a program legitimately
        does -- ask the toolchain what it supports, before flags are fixed.
        The CUDA rule in mcpp:plugins asks the same kind of question when it
        states `mcpp:fact=cuda.driver=...`.

        So the ban is narrowed to its intent: exactly one call, and it is the
        glslc extension probe. `test_vulkan_shader_generation_is_declared`
        below carries the property the ban actually existed for.
        """
        launches = re.findall(
            r"\b(system|popen|exec[lv]?[pe]?|posix_spawn[p]?)\s*\(", self.source
        )
        self.assertEqual(launches, ["system"], self.source)
        probe = re.search(
            r"bool glslc_supports\(.*?\n\}", self.source, re.S
        )
        self.assertIsNotNone(probe)
        self.assertIn("std::system(", probe.group(0))

    def vulkan_environment(self) -> dict[str, str] | None:
        """The three answers mcpp gives the program that this test must supply.

        Discovered from the payload store rather than written down: the store
        layout is mcpp's internal business, but a test is allowed to know it
        (e2e 131 in the mcpp repository reads the same paths), and hardcoding
        versions here would turn an ecosystem bump into a failing test about
        features.
        """
        home = Path(os.environ.get("MCPP_HOME", Path.home() / ".mcpp"))
        store = home / "registry/data/xpkgs"
        shaderc = sorted((store / "xim-x-shaderc").glob("*"))
        if not shaderc or not (shaderc[-1] / "bin/glslc").exists():
            return None
        toolchains = [
            entry
            for pattern in ("xim-x-gcc", "xim-x-llvm")
            for entry in sorted((store / pattern).glob("*"))
            if (entry / "bin").is_dir()
        ]
        if not toolchains:
            return None
        return {
            "MCPP_XPKG_XIM_SHADERC_DIR": str(shaderc[-1]),
            "MCPP_TOOLCHAIN_DIR": str(toolchains[-1]),
            # The program refuses to cross-compile, and equal values are what
            # a native build gives it.
            "MCPP_HOST": "x86_64-unknown-linux-gnu",
            "MCPP_TARGET": "x86_64-unknown-linux-gnu",
        }

    def test_vulkan_shader_generation_is_declared_not_performed(self):
        """134 shader sets are build-graph edges, not a loop in this program.

        This is what the subprocess ban above was really protecting. A build
        program that generated the sources itself would do it serially, once
        per prepare, and report any failure as "build.mcpp exited 1". The
        criterion is behavioural: one declared action per shader plus two (the
        generator and the shared header), and NOT ONE generated source written
        by the program itself.
        """
        shaders = sorted(
            (UPSTREAM / "ggml/src/ggml-vulkan/vulkan-shaders").glob("*.comp")
        )
        self.assertGreater(len(shaders), 100, "vendored shader set is missing")

        environment = self.vulkan_environment()
        if environment is None:
            message = (
                "xim:shaderc or a toolchain payload is absent from the store, "
                "so the shader pipeline cannot be exercised here"
            )
            # Same discipline as the missing helper above: on a developer's
            # machine this is a legitimate skip, and in the job that exists to
            # measure Vulkan it is a job that measured nothing.
            if os.environ.get("LLAMACPP_REQUIRE_VULKAN_HELPER") == "1":
                raise AssertionError(message)
            self.skipTest(message)

        result = self.run_helper(
            features=("backend-cpu", "backend-vulkan"),
            target_os="linux",
            target_arch="x86_64",
            extra_env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        actions = re.findall(r"(?m)^mcpp:action=", result.stdout)
        self.assertEqual(len(actions), len(shaders) + 2, result.stdout)

        written = list(Path(self.temp.name).rglob("*.comp.cpp"))
        self.assertEqual(written, [], "the program generated sources itself")

    def test_a_feature_the_package_does_not_have_is_refused(self):
        """A misspelt feature is an error, not a no-op.

        The name below has to be one the package genuinely does not have --
        this test previously used `backend-vulkan`, which the package now HAS,
        so the assertion would have inverted silently as the feature landed.
        """
        accepted = self.run_helper(
            features=("backend-cpu",), target_os="linux", target_arch="x86_64"
        )
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        rejected = self.run_helper(
            features=("backend-cpu", "backend-webgpu"),
            target_os="linux",
            target_arch="x86_64",
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("unsupported feature 'backend-webgpu'", rejected.stderr)

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
