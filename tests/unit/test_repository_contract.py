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


if __name__ == "__main__":
    unittest.main()
