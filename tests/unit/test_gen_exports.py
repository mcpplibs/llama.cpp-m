from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = ROOT / "tools"
sys.path.insert(0, str(TOOL_DIR))

import gen_exports


class TestTypeFingerprint(unittest.TestCase):
    def test_fixed_width_integer_alias_is_platform_independent(self):
        macos = {
            "type": {
                "qualType": "int64_t",
                "desugaredQualType": "long long",
            }
        }
        linux = {
            "type": {
                "qualType": "int64_t",
                "desugaredQualType": "long",
            }
        }

        self.assertEqual(
            gen_exports._type_fingerprint(macos),
            gen_exports._type_fingerprint(linux),
        )
        self.assertEqual(
            gen_exports._type_fingerprint(macos),
            {
                "qualType": "int64_t",
                "desugaredQualType": "signed 64-bit integer",
            },
        )


class TestGenerateExports(unittest.TestCase):
    def setUp(self):
        for name, value in {
            "REQUIRED_GGML_TYPES": {
                "ggml_backend_dev_t",
                "ggml_backend_reg_t",
                "ggml_context",
                "ggml_log_callback",
                "ggml_log_level",
            },
            "REQUIRED_GGML_ENUM_MEMBERS": {
                "GGML_BACKEND_DEVICE_TYPE_ACCEL",
                "GGML_BACKEND_DEVICE_TYPE_CPU",
                "GGML_BACKEND_DEVICE_TYPE_GPU",
            },
            "REQUIRED_GGML_FUNCTIONS": {
                "ggml_backend_alloc_ctx_tensors",
                "ggml_backend_dev_type",
                "ggml_backend_reg_by_name",
            },
        }.items():
            patcher = mock.patch.object(gen_exports, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        os.makedirs(os.path.join(self.root, "include"))
        self.llama_header = Path(self.root) / "include" / "llama.h"
        with open(
            self.llama_header,
            "w",
            encoding="utf-8",
        ) as fixture:
            fixture.write(
                """\
#define LLAMA_API
#define DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
struct llama_model;
struct llama_options {
    int threads;
    long context;
    union { int small; long large; };
};
typedef struct llama_batch { int tokens; } llama_batch;
union llama_value { int integer; double real; };
enum llama_mode { LLAMA_MODE_A, LLAMA_MODE_B };
LLAMA_API void llama_live(struct llama_model *);
DEPRECATED(LLAMA_API void llama_legacy(struct llama_model *), "use llama_live");
#define LLAMA_NUMBER 7
#define LLAMA_FUNCTION_LIKE_MACRO(value) ((value) + 1)
#define LLAMA_DEFAULT_SEED 0xFFFFFFFF
#define LLAMA_TOKEN_NULL -1
#define LLAMA_FILE_MAGIC_GGLA 0x67676c61u
#define LLAMA_FILE_MAGIC_GGSN 0x6767736eu
#define LLAMA_FILE_MAGIC_GGSQ 0x67677371u
#define LLAMA_SESSION_MAGIC LLAMA_FILE_MAGIC_GGSN
#define LLAMA_SESSION_VERSION 9
#define LLAMA_STATE_SEQ_MAGIC LLAMA_FILE_MAGIC_GGSQ
#define LLAMA_STATE_SEQ_VERSION 2
#define LLAMA_STATE_SEQ_FLAGS_NONE 0
#define LLAMA_STATE_SEQ_FLAGS_SWA_ONLY 1
#define LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY 1
#define LLAMA_STATE_SEQ_FLAGS_ON_DEVICE 2
"""
            )
        os.makedirs(os.path.join(self.root, "ggml", "include"))
        with open(
            os.path.join(self.root, "ggml", "include", "ggml.h"),
            "w",
            encoding="utf-8",
        ) as fixture:
            fixture.write(
                """\
enum ggml_log_level { GGML_LOG_LEVEL_NONE, GGML_LOG_LEVEL_INFO };
typedef void (*ggml_log_callback)(enum ggml_log_level, const char *, void *);
                """
            )
        with open(
            os.path.join(self.root, "ggml", "include", "ggml-backend.h"),
            "w",
            encoding="utf-8",
        ) as fixture:
            fixture.write(
                """\
typedef struct ggml_backend_reg * ggml_backend_reg_t;
typedef struct ggml_backend_device * ggml_backend_dev_t;
enum ggml_backend_dev_type {
    GGML_BACKEND_DEVICE_TYPE_CPU,
    GGML_BACKEND_DEVICE_TYPE_GPU,
    GGML_BACKEND_DEVICE_TYPE_ACCEL,
};
ggml_backend_reg_t ggml_backend_reg_by_name(const char *);
enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t);
"""
            )
        with open(
            os.path.join(self.root, "ggml", "include", "ggml-alloc.h"),
            "w",
            encoding="utf-8",
        ) as fixture:
            fixture.write(
                """\
struct ggml_context;
struct ggml_backend;
struct ggml_backend_buffer;
typedef struct ggml_backend * ggml_backend_t;
struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors(
    struct ggml_context *, ggml_backend_t);
"""
            )

    def tearDown(self):
        self.tmp.cleanup()

    def test_generate_exports_finds_llama_types(self):
        llama, _, _, _ = gen_exports.generate_exports(
            upstream_dir=self.root,
            include_dirs=[
                os.path.join(self.root, "include"),
                os.path.join(self.root, "ggml", "include"),
            ],
        )
        self.assertIn("export using ::llama_model;", llama)
        self.assertIn("export using ::llama_live;", llama)
        self.assertIn("export using ::LLAMA_MODE_A;", llama)

    def test_generate_exports_includes_deprecated_llama_api(self):
        llama, _, skipped, _ = gen_exports.generate_exports(
            upstream_dir=self.root,
            include_dirs=[
                os.path.join(self.root, "include"),
                os.path.join(self.root, "ggml", "include"),
            ],
        )
        self.assertIn("export using ::llama_legacy;", llama)
        self.assertNotIn("deprecated function 'llama_legacy'", skipped)

    def test_collect_api_snapshot_records_signatures_and_macros(self):
        api = gen_exports.collect_api_snapshot(
            upstream_dir=self.root,
            include_dirs=[
                os.path.join(self.root, "include"),
                os.path.join(self.root, "ggml", "include"),
            ],
        )

        self.assertIn("llama_live", api["declarations"])
        self.assertIn("void", api["declarations"]["llama_live"])
        self.assertIn("llama_legacy", api["declarations"])
        self.assertEqual(api["macros"]["LLAMA_NUMBER"], "7")
        self.assertNotIn("LLAMA_API", api["macros"])
        self.assertEqual(
            api["manual_decisions"], ["LLAMA_FUNCTION_LIKE_MACRO"]
        )

    def test_collect_api_snapshot_records_exported_ggml_signatures(self):
        before = gen_exports.collect_api_snapshot(
            upstream_dir=self.root,
            include_dirs=[
                os.path.join(self.root, "include"),
                os.path.join(self.root, "ggml", "include"),
            ],
        )
        self.assertIn("ggml_backend_reg_by_name", before["declarations"])

        backend = Path(self.root) / "ggml/include/ggml-backend.h"
        backend.write_text(
            backend.read_text(encoding="utf-8").replace(
                "ggml_backend_reg_by_name(const char *);",
                "ggml_backend_reg_by_name(const char *, int);",
            ),
            encoding="utf-8",
        )
        after = gen_exports.collect_api_snapshot(
            upstream_dir=self.root,
            include_dirs=[
                os.path.join(self.root, "include"),
                os.path.join(self.root, "ggml", "include"),
            ],
        )

        self.assertNotEqual(
            before["declarations"]["ggml_backend_reg_by_name"],
            after["declarations"]["ggml_backend_reg_by_name"],
        )

    def test_main_generates_typed_constant_replacements(self):
        output = Path(self.root) / "generated"
        result = gen_exports.main(
            ["--upstream", self.root, "--output-dir", str(output)]
        )

        self.assertEqual(result, 0)
        generated = output / "typed_constants.inc"
        self.assertTrue(generated.is_file())
        content = generated.read_text(encoding="utf-8")
        self.assertIn("LLAMA_DEFAULT_SEED = 0xFFFFFFFF", content)
        self.assertIn("LLAMA_SESSION_MAGIC = 0x6767736eu", content)
        self.assertIn("LLAMA_STATE_SEQ_FLAGS_ON_DEVICE", content)

    def test_collect_api_snapshot_tracks_struct_field_layout(self):
        before = gen_exports.collect_api_snapshot(
            upstream_dir=self.root,
            include_dirs=[
                os.path.join(self.root, "include"),
                os.path.join(self.root, "ggml", "include"),
            ],
        )
        self.llama_header.write_text(
            self.llama_header.read_text(encoding="utf-8").replace(
                "int threads;", "long threads;"
            ),
            encoding="utf-8",
        )
        after = gen_exports.collect_api_snapshot(
            upstream_dir=self.root,
            include_dirs=[
                os.path.join(self.root, "include"),
                os.path.join(self.root, "ggml", "include"),
            ],
        )

        self.assertNotEqual(
            before["declarations"]["llama_options"],
            after["declarations"]["llama_options"],
        )

    def test_collect_api_snapshot_preserves_typedef_struct_layout(self):
        before = gen_exports.collect_api_snapshot(
            upstream_dir=self.root,
            include_dirs=[
                os.path.join(self.root, "include"),
                os.path.join(self.root, "ggml", "include"),
            ],
        )
        self.llama_header.write_text(
            self.llama_header.read_text(encoding="utf-8").replace(
                "int tokens;", "long tokens;"
            ),
            encoding="utf-8",
        )
        after = gen_exports.collect_api_snapshot(
            upstream_dir=self.root,
            include_dirs=[
                os.path.join(self.root, "include"),
                os.path.join(self.root, "ggml", "include"),
            ],
        )

        self.assertNotEqual(
            before["declarations"]["llama_batch"],
            after["declarations"]["llama_batch"],
        )

    def test_collect_api_snapshot_tracks_union_field_layout(self):
        before = gen_exports.collect_api_snapshot(
            upstream_dir=self.root,
            include_dirs=[
                os.path.join(self.root, "include"),
                os.path.join(self.root, "ggml", "include"),
            ],
        )
        self.llama_header.write_text(
            self.llama_header.read_text(encoding="utf-8").replace(
                "double real;", "long double real;"
            ),
            encoding="utf-8",
        )
        after = gen_exports.collect_api_snapshot(
            upstream_dir=self.root,
            include_dirs=[
                os.path.join(self.root, "include"),
                os.path.join(self.root, "ggml", "include"),
            ],
        )

        self.assertNotEqual(
            before["declarations"]["llama_value"],
            after["declarations"]["llama_value"],
        )

    def test_collect_api_snapshot_tracks_anonymous_union_layout(self):
        before = gen_exports.collect_api_snapshot(
            upstream_dir=self.root,
            include_dirs=[
                os.path.join(self.root, "include"),
                os.path.join(self.root, "ggml", "include"),
            ],
        )
        self.llama_header.write_text(
            self.llama_header.read_text(encoding="utf-8").replace(
                "int small;", "long small;"
            ),
            encoding="utf-8",
        )
        after = gen_exports.collect_api_snapshot(
            upstream_dir=self.root,
            include_dirs=[
                os.path.join(self.root, "include"),
                os.path.join(self.root, "ggml", "include"),
            ],
        )

        self.assertNotEqual(
            before["declarations"]["llama_options"],
            after["declarations"]["llama_options"],
        )

    def test_record_fingerprint_is_independent_of_checkout_path(self):
        first = gen_exports.collect_api_snapshot(
            upstream_dir=self.root,
            include_dirs=[
                os.path.join(self.root, "include"),
                os.path.join(self.root, "ggml", "include"),
            ],
        )
        with tempfile.TemporaryDirectory() as other_root:
            shutil.copytree(self.root, other_root, dirs_exist_ok=True)
            second = gen_exports.collect_api_snapshot(
                upstream_dir=other_root,
                include_dirs=[
                    os.path.join(other_root, "include"),
                    os.path.join(other_root, "ggml", "include"),
                ],
            )

        self.assertEqual(
            first["declarations"]["llama_options"],
            second["declarations"]["llama_options"],
        )

    def test_collect_api_snapshot_resolves_implicit_enum_values(self):
        before = gen_exports.collect_api_snapshot(
            upstream_dir=self.root,
            include_dirs=[
                os.path.join(self.root, "include"),
                os.path.join(self.root, "ggml", "include"),
            ],
        )
        self.llama_header.write_text(
            self.llama_header.read_text(encoding="utf-8").replace(
                "LLAMA_MODE_A, LLAMA_MODE_B",
                "LLAMA_MODE_A, LLAMA_MODE_INSERTED, LLAMA_MODE_B",
            ),
            encoding="utf-8",
        )
        after = gen_exports.collect_api_snapshot(
            upstream_dir=self.root,
            include_dirs=[
                os.path.join(self.root, "include"),
                os.path.join(self.root, "ggml", "include"),
            ],
        )

        self.assertTrue(
            before["declarations"]["LLAMA_MODE_B"].endswith(":1")
        )
        self.assertTrue(
            after["declarations"]["LLAMA_MODE_B"].endswith(":2")
        )
        self.assertNotEqual(
            before["declarations"]["LLAMA_MODE_B"],
            after["declarations"]["LLAMA_MODE_B"],
        )

    def test_generate_exports_skips_macros(self):
        _, _, skipped, _ = gen_exports.generate_exports(
            upstream_dir=self.root,
            include_dirs=[
                os.path.join(self.root, "include"),
                os.path.join(self.root, "ggml", "include"),
            ],
        )
        self.assertIn("LLAMA_NUMBER", skipped)

    def test_generate_exports_includes_required_ggml_types(self):
        _, ggml, _, _ = gen_exports.generate_exports(
            upstream_dir=self.root,
            include_dirs=[
                os.path.join(self.root, "include"),
                os.path.join(self.root, "ggml", "include"),
            ],
        )
        self.assertIn("export using ::ggml_log_level;", ggml)
        self.assertIn("export using ::GGML_LOG_LEVEL_INFO;", ggml)
        self.assertIn("export using ::ggml_log_callback;", ggml)

    def test_generate_exports_includes_required_backend_api(self):
        _, ggml, _, _ = gen_exports.generate_exports(
            upstream_dir=self.root,
            include_dirs=[
                os.path.join(self.root, "include"),
                os.path.join(self.root, "ggml", "include"),
            ],
        )
        self.assertIn("export using ::ggml_backend_reg_t;", ggml)
        self.assertIn("export using ::GGML_BACKEND_DEVICE_TYPE_GPU;", ggml)
        self.assertIn("export using ::ggml_backend_reg_by_name;", ggml)
        self.assertIn("export using ::ggml_backend_alloc_ctx_tensors;", ggml)

    def test_generate_exports_rejects_missing_required_api(self):
        with mock.patch.object(
            gen_exports,
            "REQUIRED_GGML_FUNCTIONS",
            gen_exports.REQUIRED_GGML_FUNCTIONS | {"ggml_missing"},
        ), self.assertRaisesRegex(RuntimeError, "ggml_missing"):
            gen_exports.generate_exports(
                upstream_dir=self.root,
                include_dirs=[
                    os.path.join(self.root, "include"),
                    os.path.join(self.root, "ggml", "include"),
                ],
            )

    def test_generate_exports_rejects_failed_macro_dump(self):
        real_run = subprocess.run

        def run(command, *args, **kwargs):
            if "-dM" in command:
                return subprocess.CompletedProcess(
                    command, 1, stdout="", stderr="macro failure"
                )
            return real_run(command, *args, **kwargs)

        with mock.patch.object(gen_exports.subprocess, "run", side_effect=run), \
                self.assertRaisesRegex(RuntimeError, "macro failure"):
            gen_exports.generate_exports(
                upstream_dir=self.root,
                include_dirs=[
                    os.path.join(self.root, "include"),
                    os.path.join(self.root, "ggml", "include"),
                ],
            )


class TestCheckMode(unittest.TestCase):
    GENERATED = {
        "llama.inc": "export using ::llama_model;\n",
        "required_ggml.inc": "export using ::ggml_context;\n",
        "llama.skipped.txt": "macro LLAMA_API\n",
        "typed_constants.inc": (
            "export inline constexpr uint32_t LLAMA_DEFAULT_SEED = 0xFFFFFFFF;\n"
        ),
    }

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmp.name) / "generated"
        self.output_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_output_is_the_checked_in_module_snapshot(self):
        self.assertEqual(
            gen_exports.DEFAULT_OUTPUT_DIR,
            ROOT / "src/gen_exports",
        )

    def test_default_upstream_is_the_vendored_tree(self):
        self.assertEqual(
            gen_exports.DEFAULT_UPSTREAM_DIR,
            ROOT / "third_party/llama.cpp",
        )

    def write_outputs(self, names=None):
        selected = names if names is not None else self.GENERATED.keys()
        for name in selected:
            (self.output_dir / name).write_text(
                self.GENERATED[name], encoding="utf-8"
            )

    def run_check(self):
        generated_tuple = (
            self.GENERATED["llama.inc"],
            self.GENERATED["required_ggml.inc"],
            self.GENERATED["llama.skipped.txt"],
            self.GENERATED["typed_constants.inc"],
        )
        stderr = io.StringIO()
        with mock.patch.object(
            gen_exports, "generate_exports", return_value=generated_tuple
        ), contextlib.redirect_stderr(stderr):
            result = gen_exports.main(
                [
                    "--upstream",
                    self.tmp.name,
                    "--output-dir",
                    str(self.output_dir),
                    "--check",
                ]
            )
        return result, stderr.getvalue()

    def snapshot(self):
        return {
            path.name: path.read_bytes()
            for path in sorted(self.output_dir.iterdir())
            if path.is_file()
        }

    def test_check_accepts_matching_outputs_without_writing(self):
        self.write_outputs()
        before = self.snapshot()
        result, stderr = self.run_check()
        self.assertEqual(result, 0)
        self.assertIn("All exports match.", stderr)
        self.assertEqual(self.snapshot(), before)

    def test_check_rejects_stale_output_without_writing(self):
        self.write_outputs()
        stale = self.output_dir / "llama.inc"
        stale.write_text("stale\n", encoding="utf-8")
        before = self.snapshot()
        result, stderr = self.run_check()
        self.assertEqual(result, 1)
        self.assertIn("llama.inc differs", stderr)
        self.assertEqual(self.snapshot(), before)

    def test_check_rejects_missing_output_without_writing(self):
        self.write_outputs(["llama.inc", "required_ggml.inc"])
        before = self.snapshot()
        result, stderr = self.run_check()
        self.assertEqual(result, 1)
        self.assertIn("llama.skipped.txt does not exist", stderr)
        self.assertEqual(self.snapshot(), before)

    def test_check_requires_typed_constants_without_writing(self):
        self.write_outputs(
            ["llama.inc", "required_ggml.inc", "llama.skipped.txt"]
        )
        before = self.snapshot()
        result, stderr = self.run_check()
        self.assertEqual(result, 1)
        self.assertIn("typed_constants.inc does not exist", stderr)
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
