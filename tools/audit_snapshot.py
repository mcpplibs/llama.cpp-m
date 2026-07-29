#!/usr/bin/env python3
"""Read-only audit of a llama.cpp upstream snapshot."""
from __future__ import annotations

import argparse, hashlib, json, os, re, sys
from collections import OrderedDict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "third_party/llama.cpp"
DEFAULT_LOCK = ROOT / "upstream.lock"


def classify_api_drift(previous: dict, current: dict) -> dict:
    previous_declarations = previous.get("declarations", {})
    current_declarations = current.get("declarations", {})
    previous_macros = previous.get("macros", {})
    current_macros = current.get("macros", {})
    previous_typed_constants = previous.get("typed_constants", {})
    current_typed_constants = current.get("typed_constants", {})
    removed_declarations = sorted(
        previous_declarations.keys() - current_declarations.keys()
    )
    changed_declarations = sorted(
        name
        for name in previous_declarations.keys() & current_declarations.keys()
        if previous_declarations[name] != current_declarations[name]
    )
    return {
        "added_declarations": sorted(
            current_declarations.keys() - previous_declarations.keys()
        ),
        "removed_declarations": removed_declarations,
        "changed_declarations": changed_declarations,
        "breaking_declarations": {
            name: {
                "previous": previous_declarations[name],
                "current": current_declarations.get(name),
            }
            for name in removed_declarations + changed_declarations
        },
        "changed_macros": sorted(
            name
            for name in previous_macros.keys() | current_macros.keys()
            if previous_macros.get(name) != current_macros.get(name)
        ),
        "added_typed_constants": sorted(
            current_typed_constants.keys() - previous_typed_constants.keys()
        ),
        "removed_typed_constants": sorted(
            previous_typed_constants.keys() - current_typed_constants.keys()
        ),
        "changed_typed_constants": sorted(
            name
            for name in previous_typed_constants.keys()
            & current_typed_constants.keys()
            if previous_typed_constants[name] != current_typed_constants[name]
        ),
        "manual_decisions": sorted(set(current.get("manual_decisions", []))),
    }


def require_breaking_api_acceptance(report: dict, acceptance: Path | None) -> None:
    breaking = any(
        report.get(key, [])
        for key in (
            "removed_declarations",
            "changed_declarations",
            "removed_typed_constants",
            "changed_typed_constants",
        )
    )
    if not breaking:
        return
    if acceptance is None:
        raise ValueError(
            "removed or changed declarations or typed constants require "
            "--accept-breaking-api"
        )
    try:
        accepted = json.loads(acceptance.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read breaking API acceptance: {error}") from error
    if accepted != report:
        raise ValueError("breaking API acceptance does not match the drift report")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _tokenize_cmake(text: str):
    lines = []
    for ln in text.splitlines():
        idx = ln.find('#')
        lines.append(ln[:idx] if idx >= 0 else ln)
    text = ' '.join(lines)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c == '"':
            j = i + 1
            while j < n:
                if text[j] == '\\':
                    j += 2
                elif text[j] == '"':
                    j += 1
                    break
                else:
                    j += 1
            yield ('STR', text[i+1:j-1])
            i = j
            continue
        if c in '()':
            yield (c, c)
            i += 1
            continue
        j = i
        while j < n and not text[j].isspace() and text[j] not in '()"':
            j += 1
        yield ('WORD', text[i:j])
        i = j


def extract_cmake_call(text: str, callee: str,
                       first_arg: str | None = None
                       ) -> tuple[str | None, list[str]]:
    tokens = list(_tokenize_cmake(text))
    for i, (typ, val) in enumerate(tokens):
        if typ != 'WORD' or val != callee:
            continue
        if i + 1 >= len(tokens) or tokens[i + 1] != ('(', '('):
            continue
        depth = 1
        j = i + 2
        args: list[str] = []
        while j < len(tokens) and depth > 0:
            typ2, val2 = tokens[j]
            if typ2 == '(':
                depth += 1
                if depth == 1:
                    j += 1
                    continue
            elif typ2 == ')':
                depth -= 1
                if depth == 0:
                    break
            if depth == 1 and typ2 in ('WORD', 'STR'):
                args.append(val2)
            j += 1
        if not args:
            continue
        if first_arg is not None and args[0] != first_arg:
            continue
        return (args[0], args[1:])
    return (None, [])


def extract_cmake_list_appends(text: str, variable: str) -> list[str]:
    """Return literal sources appended to a CMake list variable."""
    tokens = list(_tokenize_cmake(text))
    result: list[str] = []
    for i, (typ, val) in enumerate(tokens):
        if typ != 'WORD' or val.lower() != 'list':
            continue
        if i + 1 >= len(tokens) or tokens[i + 1] != ('(', '('):
            continue
        depth = 1
        j = i + 2
        args: list[str] = []
        while j < len(tokens) and depth > 0:
            typ2, val2 = tokens[j]
            if typ2 == '(':
                depth += 1
            elif typ2 == ')':
                depth -= 1
                if depth == 0:
                    break
            elif depth == 1 and typ2 in ('WORD', 'STR'):
                args.append(val2)
            j += 1
        if len(args) >= 2 and args[0].upper() == 'APPEND' and args[1] == variable:
            result.extend(args[2:])
    return result


def _expand_glob(base: str, pattern: str) -> list[str]:
    directory = os.path.dirname(pattern)
    glob_pat = os.path.basename(pattern)
    full_dir = os.path.join(base, directory)
    if not os.path.isdir(full_dir):
        return []
    import fnmatch
    result = []
    for fn in sorted(os.listdir(full_dir)):
        if fnmatch.fnmatch(fn, glob_pat):
            result.append(os.path.join(directory, fn))
    return sorted(result)


def _read_cmake_forest(root: str) -> tuple[str, dict[str, str]]:
    top = os.path.join(root, 'CMakeLists.txt')
    with open(top, 'r') as f:
        top_text = f.read() if os.path.isfile(top) else ''
    subs: dict[str, str] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        if 'CMakeLists.txt' in filenames:
            sub = os.path.relpath(dirpath, root)
            if sub == '.':
                continue
            fp = os.path.join(dirpath, 'CMakeLists.txt')
            with open(fp, 'r') as f:
                subs[sub] = f.read()
    return top_text, subs


def _find_gpu_backend_calls(text: str, name: str) -> list[str]:
    """Find ggml_add_backend_library(NAME ...) calls, skipping function defs."""
    results = []
    for m in re.finditer(
        r'^\s*ggml_add_backend_library\s*\(\s*(' + re.escape(name) + r')\b',
        text, re.MULTILINE,
    ):
        before = text[:m.start()]
        opens = before.count('function(')
        closes = before.count('endfunction()')
        if opens == closes:  # not inside a function definition
            _, args = extract_cmake_call(
                text[m.start():], 'ggml_add_backend_library', name)
            if args:
                results.extend(args)
    return results


def _find_files(root: str, ext: str) -> list[str]:
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(ext):
                rel = os.path.relpath(os.path.join(dirpath, fn), root)
                found.append(rel)
    return sorted(found)


IGNORED_CPU_SOURCE_VARIABLES = {
    '${GGML_KLEIDIAI_SME_SOURCES}',
    '${GGML_KLEIDIAI_SME2_SOURCES}',
    '${GGML_KLEIDIAI_SOURCES}',
}


def _collect_cpu_sources(root: str, top_text: str,
                         sub_cmakes: dict[str, str]) -> tuple[list[str], list[str], list[str]]:
    common: set[str] = set()
    x86: set[str] = set()
    arm: set[str] = set()
    cmakes = [('.', top_text), *sub_cmakes.items()]
    for subdir, text in cmakes:
        if 'GGML_CPU_SOURCES' not in text:
            continue
        for source in extract_cmake_list_appends(text, 'GGML_CPU_SOURCES'):
            if '$' in source:
                if source in IGNORED_CPU_SOURCE_VARIABLES:
                    continue
                raise ValueError(f'unexpected CPU source variable: {source}')
            # Literal conditional sources are conservatively included. This audit is
            # not a full CMake evaluator, so unknown variable expansions fail closed.
            if Path(source).suffix not in {'.c', '.cc', '.cpp', '.m', '.mm'}:
                continue
            candidates = (
                Path(root) / source,
                Path(root) / subdir / source,
                Path(root) / subdir / '..' / source,
            )
            resolved = next((path.resolve() for path in candidates if path.is_file()), None)
            if resolved is None:
                raise ValueError(f'unresolved CPU source: {source}')
            relative = resolved.relative_to(Path(root).resolve()).as_posix()
            prefix = 'ggml/src/ggml-cpu/'
            if not relative.startswith(prefix):
                continue
            cpu_relative = relative.removeprefix(prefix)
            if cpu_relative.startswith('arch/x86/'):
                x86.add(relative)
            elif cpu_relative.startswith('arch/arm/'):
                arm.add(relative)
            elif (cpu_relative.startswith('arch/')
                  or cpu_relative.startswith('kleidiai/')
                  or cpu_relative.startswith('spacemit/')):
                continue
            else:
                common.add(relative)
    if not common or not x86 or not arm:
        raise ValueError('cannot derive common/x86/arm GGML CPU sources from CMake')
    return sorted(common), sorted(x86), sorted(arm)


def collect_snapshot(
        root, tag, commit, url, archive_sha256, api_snapshot=None):
    top_text, sub_cmakes = _read_cmake_forest(root)
    if not top_text and not sub_cmakes:
        raise FileNotFoundError(f"No CMakeLists.txt found under {root}")

    def _extract_from_forest(callee, first_arg, cmake_dir='.'):
        for subdir, text in sub_cmakes.items():
            name, args = extract_cmake_call(text, callee, first_arg)
            if name is not None:
                return name, [os.path.join(subdir, a) for a in args]
        name, args = extract_cmake_call(top_text, callee, first_arg)
        if name is not None:
            return name, args
        return None, []

    _, ggml_base = _extract_from_forest('add_library', 'ggml-base')
    _, ggml_registry = _extract_from_forest('add_library', 'ggml')
    _, ggml_metal = _extract_from_forest('ggml_add_backend_library', 'ggml-metal')
    _, llama = _extract_from_forest('add_library', 'llama')

    # Expand GLOB in llama sources
    llama_expanded: list[str] = []
    model_files: list[str] = []
    for src in llama:
        if '${LLAMA_MODELS_SOURCES}' in src:
            for subdir, text in sub_cmakes.items():
                _, glob_args = extract_cmake_call(text, 'file', 'GLOB')
                if (glob_args and len(glob_args) >= 2 and
                        glob_args[0] == 'LLAMA_MODELS_SOURCES'):
                    pattern = glob_args[1].strip('"')
                    model_dir = os.path.join(root, subdir)
                    model_files = _expand_glob(model_dir, pattern)
                    break
            if not model_files:
                _, glob_args = extract_cmake_call(top_text, 'file', 'GLOB')
                if glob_args and len(glob_args) >= 2:
                    model_files = _expand_glob(root, glob_args[1])
        elif src.endswith('.cpp') or src.endswith('.c'):
            llama_expanded.append(src)

    cpu_common, cpu_arch_x86, cpu_arch_arm = _collect_cpu_sources(
        root, top_text, sub_cmakes)

    # Registry macros
    registry: dict[str, str] = {}
    for candidate in ['ggml/src/ggml-backend-reg.cpp', 'ggml-backend-reg.cpp']:
        reg_path = os.path.join(root, candidate)
        if os.path.isfile(reg_path):
            with open(reg_path, 'r') as f:
                reg_text = f.read()
            for macro in ['GGML_USE_CPU', 'GGML_USE_METAL', 'GGML_USE_CUDA',
                           'GGML_USE_VULKAN', 'GGML_USE_SYCL', 'GGML_USE_CANN']:
                # Match within a single #ifdef ... #endif block only
                block_pat = re.compile(
                    r'#ifdef\s+' + macro + r'\b[^\n]*\n(.*?)#endif',
                    re.DOTALL)
                for block_m in block_pat.finditer(reg_text):
                    inner = block_m.group(1)
                    rm = re.search(r'register_backend\((\w+)\(\)\)', inner)
                    if rm:
                        registry[macro] = rm.group(1)
                        break
            break

    # Metal shader inputs
    metal_inputs: list[str] = []
    for candidate_dir in [os.path.join(root, 'ggml', 'src', 'ggml-metal'),
                          os.path.join(root, 'ggml-metal')]:
        if os.path.isdir(candidate_dir):
            for pat in ['ggml-common.h', 'ggml-metal.metal', 'ggml-metal-impl.h']:
                for base in [candidate_dir, os.path.dirname(candidate_dir)]:
                    full = os.path.join(base, pat)
                    if os.path.isfile(full):
                        rel = os.path.relpath(full, root)
                        if rel not in metal_inputs:
                            metal_inputs.append(rel)
                        break
            if metal_inputs:
                break
    metal_inputs.sort()

    # C++20 exceptions
    cpp20_exceptions = []
    for p in ['src/models/dflash.cpp', 'src/models/eagle3.cpp',
              'src/models/hunyuan-dense.cpp', 'src/models/llama-embed.cpp',
              'src/models/minimax-m2.cpp',
              'src/models/t5.cpp']:
        if os.path.isfile(os.path.join(root, p)):
            cpp20_exceptions.append(p)

    # Public header hashes
    header_hashes = {}
    for hdr in [
            'include/llama.h',
            'ggml/include/ggml.h',
            'ggml/include/ggml-cpu.h',
            'ggml/include/ggml-backend.h',
            'ggml/include/ggml-alloc.h']:
        p = os.path.join(root, hdr)
        if os.path.isfile(p):
            header_hashes[hdr] = sha256_file(p)

    if api_snapshot is None:
        from gen_exports import collect_api_snapshot
        api_snapshot = collect_api_snapshot(root)

    return OrderedDict([
        ('schema', 1),
        ('upstream', OrderedDict([
            ('tag', tag), ('commit', commit),
            ('url', url), ('sha256', archive_sha256),
        ])),
        ('sources', OrderedDict([
            ('ggml_base', sorted(set(ggml_base))),
            ('ggml_registry', sorted(set(ggml_registry))),
            ('ggml_cpu_common', cpu_common),
            ('ggml_cpu_x86', cpu_arch_x86),
            ('ggml_cpu_arm', cpu_arch_arm),
            ('llama_core', sorted(set(llama_expanded))),
            ('models', sorted(set(model_files))),
            ('ggml_metal', sorted(set(ggml_metal))),
        ])),
        ('registry', registry),
        ('metal', OrderedDict([
            ('frameworks', ['Foundation', 'Metal', 'MetalKit']),
            ('shader_inputs', metal_inputs),
        ])),
        ('dialect_exceptions', OrderedDict([
            ('c++20', cpp20_exceptions),
        ])),
        ('platform_links', {'windows_cpu': ['advapi32']}),
        ('public_header_sha256', header_hashes),
        ('api', api_snapshot),
    ])


def compare_reports(old: dict, new: dict) -> list[str]:
    diffs = []
    for key in ['upstream', 'sources', 'registry', 'metal', 'dialect_exceptions',
                 'platform_links', 'public_header_sha256', 'api']:
        if old.get(key) != new.get(key):
            diffs.append(f"{key} changed")
    return diffs


def check_exports(root: str, output_dir: Path) -> int:
    from gen_exports import generate_exports, sync_outputs

    llama_inc, ggml_inc, skipped_txt, typed_constants = generate_exports(root)
    return sync_outputs(
        output_dir,
        {
            'llama.inc': llama_inc,
            'required_ggml.inc': ggml_inc,
            'llama.skipped.txt': skipped_txt,
            'typed_constants.inc': typed_constants,
        },
        True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description='Audit a llama.cpp snapshot')
    ap.add_argument(
        '--source', '--upstream', dest='source', type=Path,
        default=DEFAULT_SOURCE, help='Local llama.cpp source tree')
    ap.add_argument(
        '--lock', type=Path, default=DEFAULT_LOCK,
        help='Pinned upstream lock file')
    out = ap.add_mutually_exclusive_group()
    out.add_argument('--output', help='Write JSON report to this file')
    out.add_argument('--check', help='Regenerate and compare with this report file')
    out.add_argument('--compare', help='Compare OLD_REPORT with a new generation')
    ap.add_argument(
        '--check-exports', type=Path,
        help='Compare generated module exports using the same verified source tree')
    ap.add_argument(
        '--accept-breaking-api', type=Path,
        help='Reviewed JSON drift report accepting removals/signature changes')
    args = ap.parse_args()

    if args.check_exports and not args.check:
        ap.error('--check-exports requires --check')

    try:
        from import_upstream import load_lock
        lock = load_lock(args.lock)
    except (OSError, ValueError) as error:
        ap.error(f'cannot load upstream lock: {error}')
    identity = OrderedDict([
        ('tag', lock.tag),
        ('commit', lock.commit),
        ('url', lock.archive_url),
        ('sha256', lock.archive_sha256),
    ])

    root = args.source.resolve()
    if not (root / 'include/llama.h').is_file():
        ap.error(f'mcpplibs:llamacpp source tree not found: {root}')

    expected = None
    if args.check:
        try:
            with open(args.check, 'r') as f:
                expected = json.load(f, object_pairs_hook=OrderedDict)
        except (OSError, json.JSONDecodeError) as error:
            ap.error(f'cannot load check report: {error}')
        expected_identity = expected.get('upstream')
        if not isinstance(expected_identity, dict):
            ap.error('check report is missing upstream identity')
        required = ('tag', 'commit', 'url', 'sha256')
        missing = [key for key in required
                   if not isinstance(expected_identity.get(key), str)
                   or not expected_identity[key]]
        if missing:
            ap.error('check report is missing upstream identity fields: '
                     + ', '.join(missing))
        if expected_identity != identity:
            ap.error('snapshot upstream identity conflicts with upstream.lock')

    report = collect_snapshot(
        root=os.fspath(root),
        tag=lock.tag,
        commit=lock.commit,
        url=lock.archive_url,
        archive_sha256=lock.archive_sha256,
    )
    exports_result = (
        check_exports(os.fspath(root), args.check_exports)
        if args.check_exports else 0
    )

    if args.output:
        output = Path(args.output)
        if output.is_file():
            try:
                previous = json.loads(output.read_text(encoding='utf-8'))
                drift = classify_api_drift(
                    previous.get('api', {}), report.get('api', {}))
                require_breaking_api_acceptance(
                    drift, args.accept_breaking_api)
            except (OSError, json.JSONDecodeError, ValueError) as error:
                print(error, file=sys.stderr)
                return 1
        import tempfile
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=output.parent,
                prefix=f'.{output.name}.', suffix='.tmp', delete=False) as f:
            temporary_output = Path(f.name)
            json.dump(report, f, indent=2)
            f.write('\n')
        try:
            os.replace(temporary_output, output)
        finally:
            temporary_output.unlink(missing_ok=True)
        print(f"Snapshot written to {output}", file=sys.stderr)
    elif args.check:
        assert expected is not None
        if expected == report:
            print("Snapshot matches.", file=sys.stderr)
        else:
            drift = classify_api_drift(
                expected.get('api', {}), report.get('api', {}))
            try:
                require_breaking_api_acceptance(
                    drift, args.accept_breaking_api)
            except ValueError as error:
                print(error, file=sys.stderr)
                return 1
            print("Snapshot differs:", file=sys.stderr)
            for diff in compare_reports(expected, report):
                print(f"  - {diff}", file=sys.stderr)
            print(json.dumps(drift, indent=2), file=sys.stderr)
            return 1
        if exports_result != 0:
            return exports_result
    elif args.compare:
        with open(args.compare, 'r') as f:
            old = json.load(f, object_pairs_hook=OrderedDict)
        diffs = compare_reports(old, report)
        if diffs:
            for d in diffs:
                print(d)
        else:
            print("No differences.")
    else:
        json.dump(report, sys.stdout, indent=2)
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
