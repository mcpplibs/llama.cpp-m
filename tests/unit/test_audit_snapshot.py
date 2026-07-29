"""Unit tests for audit_snapshot.py against a miniature CMake tree."""
import contextlib
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import audit_snapshot


class TestRepositoryDefaults(unittest.TestCase):
    def test_defaults_use_vendored_source_and_upstream_lock(self):
        self.assertEqual(
            audit_snapshot.DEFAULT_SOURCE,
            ROOT / "third_party/llama.cpp",
        )
        self.assertEqual(audit_snapshot.DEFAULT_LOCK, ROOT / "upstream.lock")


class TestApiDrift(unittest.TestCase):
    def setUp(self):
        self.previous = {
            "declarations": {
                "llama_old": "void ()",
                "llama_changed": "int (int)",
            },
            "macros": {"LLAMA_VERSION": "1"},
            "manual_decisions": [],
        }
        self.current = {
            "declarations": {
                "llama_new": "void ()",
                "llama_changed": "long (int)",
            },
            "macros": {"LLAMA_VERSION": "2"},
            "manual_decisions": ["LLAMA_FUNCTION_LIKE_MACRO"],
        }

    def test_classifies_public_api_drift(self):
        report = audit_snapshot.classify_api_drift(self.previous, self.current)

        self.assertEqual(report["added_declarations"], ["llama_new"])
        self.assertEqual(report["removed_declarations"], ["llama_old"])
        self.assertEqual(report["changed_declarations"], ["llama_changed"])
        self.assertEqual(report["changed_macros"], ["LLAMA_VERSION"])
        self.assertEqual(
            report["manual_decisions"], ["LLAMA_FUNCTION_LIKE_MACRO"]
        )

    def test_breaking_api_requires_matching_acceptance_report(self):
        report = audit_snapshot.classify_api_drift(self.previous, self.current)
        with self.assertRaisesRegex(ValueError, "--accept-breaking-api"):
            audit_snapshot.require_breaking_api_acceptance(report, None)

        with tempfile.TemporaryDirectory() as directory:
            acceptance = Path(directory) / "accepted.json"
            acceptance.write_text(json.dumps(report) + "\n", encoding="utf-8")
            audit_snapshot.require_breaking_api_acceptance(report, acceptance)

    def test_breaking_api_rejects_stale_acceptance_report(self):
        report = audit_snapshot.classify_api_drift(self.previous, self.current)
        with tempfile.TemporaryDirectory() as directory:
            acceptance = Path(directory) / "accepted.json"
            acceptance.write_text(
                json.dumps({**report, "removed_declarations": []}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                audit_snapshot.require_breaking_api_acceptance(report, acceptance)

    def test_breaking_api_rejects_stale_signature_acceptance(self):
        accepted_report = audit_snapshot.classify_api_drift(
            self.previous, self.current
        )
        later_current = {
            **self.current,
            "declarations": {
                **self.current["declarations"],
                "llama_changed": "double (int)",
            },
        }
        later_report = audit_snapshot.classify_api_drift(
            self.previous, later_current
        )

        with tempfile.TemporaryDirectory() as directory:
            acceptance = Path(directory) / "accepted.json"
            acceptance.write_text(
                json.dumps(accepted_report) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                audit_snapshot.require_breaking_api_acceptance(
                    later_report, acceptance
                )


class TestExtractCmakeCall(unittest.TestCase):
    """Balanced-paren CMake extraction."""

    def test_simple_add_library(self):
        text = 'add_library(ggml-base ggml.c ggml.cpp ggml-backend.cpp)'
        name, sources = audit_snapshot.extract_cmake_call(text, 'add_library', 'ggml-base')
        self.assertEqual(name, 'ggml-base')
        self.assertEqual(sources, ['ggml.c', 'ggml.cpp', 'ggml-backend.cpp'])

    def test_ggml_add_backend_library(self):
        text = '''ggml_add_backend_library(ggml-metal
    ggml-metal.cpp
    ggml-metal-device.m
    ggml-metal-device.cpp)'''
        name, sources = audit_snapshot.extract_cmake_call(text, 'ggml_add_backend_library', 'ggml-metal')
        self.assertEqual(name, 'ggml-metal')
        self.assertEqual(sources, ['ggml-metal.cpp', 'ggml-metal-device.m', 'ggml-metal-device.cpp'])

    def test_multiline_file_glob(self):
        text = '''file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")
add_library(llama llama.cpp ${LLAMA_MODELS_SOURCES})'''
        # Should handle the GLOB but not expand it here
        name, sources = audit_snapshot.extract_cmake_call(text, 'add_library', 'llama')
        self.assertEqual(name, 'llama')
        # GLOB variable reference is preserved as-is for later expansion
        self.assertIn('${LLAMA_MODELS_SOURCES}', ' '.join(sources))

    def test_balanced_nested_parens(self):
        text = 'target_compile_definitions(ggml PRIVATE GGML_BUILD=1 GGML_SHARED=0 $<$<CONFIG:Debug>:GGML_DEBUG>)'
        name, args = audit_snapshot.extract_cmake_call(text, 'target_compile_definitions', 'ggml')
        self.assertIn('GGML_BUILD=1', args)
        self.assertIn('$<$<CONFIG:Debug>:GGML_DEBUG>', ' '.join(args))


class TestCollectSnapshotMiniTree(unittest.TestCase):
    """Full snapshot from a synthetic tree."""

    maxDiff = None
    API = {
        "declarations": {"llama_fixture": "FunctionDecl:void ()"},
        "macros": {"LLAMA_FIXTURE": "1"},
        "manual_decisions": [],
    }

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        os.makedirs(os.path.join(self.root, 'ggml', 'src'), exist_ok=True)
        os.makedirs(os.path.join(self.root, 'ggml', 'src', 'ggml-cpu', 'arch', 'x86'), exist_ok=True)
        os.makedirs(os.path.join(self.root, 'ggml', 'src', 'ggml-cpu', 'arch', 'arm'), exist_ok=True)
        os.makedirs(os.path.join(self.root, 'ggml', 'src', 'ggml-metal'), exist_ok=True)
        os.makedirs(os.path.join(self.root, 'ggml', 'include'), exist_ok=True)
        os.makedirs(os.path.join(self.root, 'src', 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.root, 'include'), exist_ok=True)

        # Write CMake snippet
        with open(os.path.join(self.root, 'CMakeLists.txt'), 'w') as f:
            f.write('''add_library(ggml-base ggml.c ggml.cpp ggml-backend.cpp)
add_library(ggml ggml-backend-dl.cpp ggml-backend-reg.cpp)
ggml_add_backend_library(ggml-metal
    ggml-metal.cpp
    ggml-metal-device.m
    ggml-metal-device.cpp)
file(GLOB LLAMA_MODELS_SOURCES "src/models/*.cpp")
add_library(llama llama.cpp ${LLAMA_MODELS_SOURCES})
''')
        cpu_cmake = os.path.join(
            self.root, 'ggml', 'src', 'ggml-cpu', 'CMakeLists.txt')
        with open(cpu_cmake, 'w') as f:
            f.write('''function(ggml_add_cpu_backend_variant_impl tag_name)
    list(APPEND GGML_CPU_SOURCES
        ggml-cpu/ggml-cpu.c
        ggml-cpu/ggml-cpu.cpp
        ggml-cpu/ops.cpp)
    if (GGML_SYSTEM_ARCH STREQUAL "ARM")
        list(APPEND GGML_CPU_SOURCES ggml-cpu/arch/arm/quants.c)
    elseif (GGML_SYSTEM_ARCH STREQUAL "x86")
        list(APPEND GGML_CPU_SOURCES ggml-cpu/arch/x86/quants.c)
    endif()
endfunction()
''')
        # Create source files
        for src in ['ggml.c', 'ggml.cpp', 'ggml-backend.cpp',
                    'ggml-backend-dl.cpp', 'ggml-backend-reg.cpp',
                    'ggml-metal.cpp', 'ggml-metal-device.m', 'ggml-metal-device.cpp',
                    'llama.cpp']:
            open(os.path.join(self.root, src), 'w').close()
        for src in [
                'ggml/src/ggml-cpu/ggml-cpu.c',
                'ggml/src/ggml-cpu/ggml-cpu.cpp',
                'ggml/src/ggml-cpu/ops.cpp',
                'ggml/src/ggml-cpu/arch/x86/quants.c',
                'ggml/src/ggml-cpu/arch/arm/quants.c']:
            open(os.path.join(self.root, src), 'w').close()
        # Create model files (sorted names for deterministic order)
        for name in ['a.cpp', 'z.cpp']:
            open(os.path.join(self.root, 'src', 'models', name), 'w').close()
        # Create registry fixture
        reg_path = os.path.join(self.root, 'ggml-backend-reg.cpp')
        with open(reg_path, 'w') as f:
            f.write('''
#ifdef GGML_USE_CPU
    register_backend(ggml_backend_cpu_reg());
#endif
#ifdef GGML_USE_METAL
    register_backend(ggml_backend_metal_reg());
#endif
''')
        # Create metal shader input files
        for name in ['ggml-common.h', 'ggml-metal.metal', 'ggml-metal-impl.h']:
            sub = 'ggml-metal' if name != 'ggml-common.h' else ''
            p = os.path.join(self.root, 'ggml', 'src', sub, name)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'w') as f:
                f.write('/* placeholder */\n')
        # Create public headers
        for name, directory in [
                ('llama.h', 'include'),
                ('ggml.h', 'ggml/include'),
                ('ggml-cpu.h', 'ggml/include')]:
            with open(os.path.join(self.root, directory, name), 'w') as f:
                f.write(f'/* {name} placeholder */\n')

        # Create the metal shader with markers
        metal_path = os.path.join(self.root, 'ggml', 'src', 'ggml-metal', 'ggml-metal.metal')
        with open(metal_path, 'w') as f:
            f.write('''
#include "ggml-common.h"
// ... metal shader code ...
// replacement marker for build system
''')

    def tearDown(self):
        self.tmp.cleanup()

    def test_collect_snapshot_sources(self):
        report = audit_snapshot.collect_snapshot(
            self.root,
            tag='test',
            commit='deadbeef',
            url='https://example.com/test.tar.gz',
            archive_sha256='abc123',
            api_snapshot=self.API)
        # sorted() gives ASCII order: '-' < '.', so ggml-backend*.cpp comes first
        self.assertEqual(report['sources']['ggml_base'],
                         ['ggml-backend.cpp', 'ggml.c', 'ggml.cpp'])
        self.assertEqual(report['sources']['ggml_registry'],
                         ['ggml-backend-dl.cpp', 'ggml-backend-reg.cpp'])
        # Models: glob expanded and sorted
        self.assertEqual(report['sources']['models'],
                         ['src/models/a.cpp', 'src/models/z.cpp'])
        # Metal: 3 source files
        # sorted order for ggml_metal sources
        self.assertEqual(report['sources']['ggml_metal'],
                         ['ggml-metal-device.cpp', 'ggml-metal-device.m', 'ggml-metal.cpp'])
        self.assertEqual(report['sources']['ggml_cpu_common'], [
            'ggml/src/ggml-cpu/ggml-cpu.c',
            'ggml/src/ggml-cpu/ggml-cpu.cpp',
            'ggml/src/ggml-cpu/ops.cpp',
        ])
        self.assertEqual(report['sources']['ggml_cpu_x86'], [
            'ggml/src/ggml-cpu/arch/x86/quants.c',
        ])
        self.assertEqual(report['sources']['ggml_cpu_arm'], [
            'ggml/src/ggml-cpu/arch/arm/quants.c',
        ])
        # Registry markers
        self.assertEqual(report['registry']['GGML_USE_CPU'], 'ggml_backend_cpu_reg')
        self.assertEqual(report['registry']['GGML_USE_METAL'], 'ggml_backend_metal_reg')
        # Shader inputs (sorted, deduplicated)
        self.assertIn('ggml/src/ggml-common.h', report['metal']['shader_inputs'])
        self.assertIn('ggml/src/ggml-metal/ggml-metal.metal', report['metal']['shader_inputs'])
        self.assertIn('ggml/src/ggml-metal/ggml-metal-impl.h', report['metal']['shader_inputs'])
        self.assertEqual(
            set(report['public_header_sha256']),
            {'include/llama.h', 'ggml/include/ggml.h', 'ggml/include/ggml-cpu.h'},
        )
        self.assertEqual(report['api'], self.API)

    def test_check_exports_uses_verified_upstream_tree(self):
        import gen_exports

        checker = getattr(audit_snapshot, 'check_exports', None)
        self.assertIsNotNone(checker)
        generated = ('llama exports\n', 'ggml exports\n', 'skipped\n')
        output_dir = Path(self.root) / 'exports'
        with mock.patch.object(
                gen_exports, 'generate_exports', return_value=generated) as generate, \
                mock.patch.object(
                    gen_exports, 'sync_outputs', return_value=0) as sync:
            self.assertEqual(checker(self.root, output_dir), 0)
        generate.assert_called_once_with(self.root)
        sync.assert_called_once_with(
            output_dir,
            {
                'llama.inc': generated[0],
                'required_ggml.inc': generated[1],
                'llama.skipped.txt': generated[2],
            },
            True,
        )

    def test_rejects_unknown_cpu_source_variable(self):
        cpu_cmake = Path(self.root) / 'ggml/src/ggml-cpu/CMakeLists.txt'
        with cpu_cmake.open('a') as f:
            f.write('\nlist(APPEND GGML_CPU_SOURCES ${NEW_DEFAULT_CPU_SOURCES})\n')

        with self.assertRaisesRegex(
                ValueError, 'unexpected CPU source variable.*NEW_DEFAULT_CPU_SOURCES'):
            audit_snapshot.collect_snapshot(
                self.root,
                tag='test',
                commit='deadbeef',
                url='https://example.com/test.tar.gz',
                archive_sha256='abc123',
                api_snapshot=self.API)

    def test_rejects_unresolved_cpu_translation_unit(self):
        cpu_cmake = Path(self.root) / 'ggml/src/ggml-cpu/CMakeLists.txt'
        with cpu_cmake.open('a') as f:
            f.write(
                '\nlist(APPEND GGML_CPU_SOURCES ggml-cpu/generated-default.cpp)\n')

        with self.assertRaisesRegex(
                ValueError, 'unresolved CPU source.*generated-default.cpp'):
            audit_snapshot.collect_snapshot(
                self.root,
                tag='test',
                commit='deadbeef',
                url='https://example.com/test.tar.gz',
                archive_sha256='abc123',
                api_snapshot=self.API)

    def _write_archive_and_report(self):
        fixture_dir = tempfile.TemporaryDirectory()
        archive = Path(fixture_dir.name) / 'llama.cpp-test.tar.gz'
        with tarfile.open(archive, 'w:gz') as tf:
            tf.add(self.root, arcname='llama.cpp-test')
        digest = audit_snapshot.sha256_file(archive)
        url = archive.as_uri()
        report = audit_snapshot.collect_snapshot(
            self.root,
            tag='test',
            commit='d' * 40,
            url=url,
            archive_sha256=digest,
            api_snapshot=self.API,
        )
        report_path = Path(fixture_dir.name) / 'snapshot.json'
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        lock_path = Path(fixture_dir.name) / 'upstream.lock'
        lock_path.write_text(
            '[upstream]\n'
            'repository = "https://github.com/example/project"\n'
            f'tag = "{report["upstream"]["tag"]}"\n'
            f'commit = "{report["upstream"]["commit"]}"\n'
            f'archive_url = "{report["upstream"]["url"]}"\n'
            f'archive_sha256 = "{report["upstream"]["sha256"]}"\n'
            'imported_at_utc = "2026-07-29T00:00:00Z"\n',
            encoding='utf-8',
        )
        return fixture_dir, report_path, lock_path

    def test_check_uses_local_source_and_lock_without_download(self):
        fixture_dir, report_path, lock_path = self._write_archive_and_report()
        self.addCleanup(fixture_dir.cleanup)
        with mock.patch.object(
                sys, 'argv', ['audit_snapshot.py', '--check', str(report_path),
                              '--source', self.root, '--lock', str(lock_path)]), \
                mock.patch(
                    'gen_exports.collect_api_snapshot', return_value=self.API
                ), \
                self.subTest("audit has no download path"):
            self.assertFalse(hasattr(audit_snapshot, 'download_file'))
            self.assertEqual(audit_snapshot.main(), 0)

    def test_check_rejects_incomplete_report_identity(self):
        fixture_dir, report_path, lock_path = self._write_archive_and_report()
        self.addCleanup(fixture_dir.cleanup)
        report = json.loads(report_path.read_text())
        del report['upstream']['sha256']
        report_path.write_text(json.dumps(report))
        stderr = io.StringIO()
        with mock.patch.object(
                sys, 'argv', ['audit_snapshot.py', '--check', str(report_path),
                              '--source', self.root, '--lock', str(lock_path)]), \
                contextlib.redirect_stderr(stderr), \
                self.assertRaises(SystemExit):
            audit_snapshot.main()
        self.assertIn('missing upstream identity fields: sha256', stderr.getvalue())

    def test_check_rejects_snapshot_identity_conflicting_with_lock(self):
        fixture_dir, report_path, lock_path = self._write_archive_and_report()
        self.addCleanup(fixture_dir.cleanup)
        report = json.loads(report_path.read_text(encoding='utf-8'))
        report['upstream']['tag'] = 'other'
        report_path.write_text(json.dumps(report), encoding='utf-8')
        stderr = io.StringIO()
        with mock.patch.object(
                sys, 'argv', ['audit_snapshot.py', '--check', str(report_path),
                              '--source', self.root, '--lock', str(lock_path)]), \
                contextlib.redirect_stderr(stderr), \
                self.assertRaises(SystemExit):
            audit_snapshot.main()
        self.assertIn('conflicts with upstream.lock', stderr.getvalue())

    def test_output_requires_acceptance_before_replacing_breaking_api(self):
        fixture_dir, report_path, lock_path = self._write_archive_and_report()
        self.addCleanup(fixture_dir.cleanup)
        released = json.loads(report_path.read_text(encoding='utf-8'))
        released['api']['declarations']['llama_removed'] = 'FunctionDecl:void ()'
        report_path.write_text(json.dumps(released) + '\n', encoding='utf-8')
        original = report_path.read_bytes()
        argv = [
            'audit_snapshot.py', '--output', str(report_path),
            '--source', self.root, '--lock', str(lock_path),
        ]

        with mock.patch.object(sys, 'argv', argv), \
                mock.patch(
                    'gen_exports.collect_api_snapshot', return_value=self.API
                ):
            self.assertEqual(audit_snapshot.main(), 1)
        self.assertEqual(report_path.read_bytes(), original)

        drift = audit_snapshot.classify_api_drift(released['api'], self.API)
        acceptance = Path(fixture_dir.name) / 'accepted.json'
        acceptance.write_text(json.dumps(drift) + '\n', encoding='utf-8')
        with mock.patch.object(
                sys, 'argv', argv + [
                    '--accept-breaking-api', str(acceptance)
                ]), mock.patch(
                    'gen_exports.collect_api_snapshot', return_value=self.API
                ):
            self.assertEqual(audit_snapshot.main(), 0)
        updated = json.loads(report_path.read_text(encoding='utf-8'))
        self.assertEqual(updated['api'], self.API)


if __name__ == '__main__':
    unittest.main()
