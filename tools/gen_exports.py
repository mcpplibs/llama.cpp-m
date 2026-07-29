#!/usr/bin/env python3
"""Deterministic API export generator using Clang JSON AST dump."""
from __future__ import annotations

import argparse, json, os, re, subprocess, sys, tempfile
from pathlib import Path

REQUIRED_GGML_TYPES = {
    "ggml_abort_callback", "ggml_backend_buffer_t",
    "ggml_backend_buffer_type_t", "ggml_backend_get_features_t",
    "ggml_backend_reg_t", "ggml_backend_t",
    "ggml_backend_dev_t", "ggml_backend_sched_eval_callback",
    "ggml_cgraph", "ggml_context", "ggml_log_callback",
    "ggml_init_params", "ggml_log_level", "ggml_numa_strategy",
    "ggml_opt_dataset_t",
    "ggml_opt_epoch_callback", "ggml_opt_get_optimizer_params",
    "ggml_opt_optimizer_type", "ggml_opt_result_t", "ggml_status",
    "ggml_tensor", "ggml_threadpool_t", "ggml_type",
}

REQUIRED_GGML_ENUM_MEMBERS = {
    "GGML_BACKEND_DEVICE_TYPE_ACCEL",
    "GGML_BACKEND_DEVICE_TYPE_CPU",
    "GGML_BACKEND_DEVICE_TYPE_GPU",
    "GGML_STATUS_SUCCESS",
}

REQUIRED_GGML_FUNCTIONS = {
    "ggml_add", "ggml_backend_alloc_ctx_tensors",
    "ggml_backend_buffer_free", "ggml_backend_dev_init",
    "ggml_backend_dev_type", "ggml_backend_free",
    "ggml_backend_graph_compute", "ggml_backend_reg_by_name",
    "ggml_backend_reg_dev_count", "ggml_backend_reg_dev_get",
    "ggml_backend_reg_get_proc_address", "ggml_backend_synchronize",
    "ggml_backend_tensor_get", "ggml_backend_tensor_set",
    "ggml_build_forward_expand", "ggml_free", "ggml_init",
    "ggml_new_graph", "ggml_new_tensor_1d",
}

GGML_ENUM_TYPES = {"ggml_log_level", "ggml_numa_strategy",
                    "ggml_opt_optimizer_type", "ggml_type"}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM_DIR = ROOT / "third_party/llama.cpp"
DEFAULT_OUTPUT_DIR = ROOT / "src/gen_exports"
NON_API_LLAMA_MACROS = {"LLAMA_API", "LLAMA_H"}
TYPED_LLAMA_CONSTANTS = {
    "LLAMA_DEFAULT_SEED": "uint32_t",
    "LLAMA_TOKEN_NULL": "llama_token",
    "LLAMA_FILE_MAGIC_GGLA": "uint32_t",
    "LLAMA_FILE_MAGIC_GGSN": "uint32_t",
    "LLAMA_FILE_MAGIC_GGSQ": "uint32_t",
    "LLAMA_SESSION_MAGIC": "uint32_t",
    "LLAMA_SESSION_VERSION": "uint32_t",
    "LLAMA_STATE_SEQ_MAGIC": "uint32_t",
    "LLAMA_STATE_SEQ_VERSION": "uint32_t",
    "LLAMA_STATE_SEQ_FLAGS_NONE": "llama_state_seq_flags",
    "LLAMA_STATE_SEQ_FLAGS_SWA_ONLY": "llama_state_seq_flags",
    "LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY": "llama_state_seq_flags",
    "LLAMA_STATE_SEQ_FLAGS_ON_DEVICE": "llama_state_seq_flags",
}
FIXED_WIDTH_INTEGER = re.compile(
    r"\b(?P<unsigned>u?)int(?P<bits>8|16|32|64)_t\b"
)


def _find_clang():
    for candidate in [os.environ.get("CLANG", ""), "clang++-22",
                      "clang++-20", "clang++-19", "clang++"]:
        if not candidate:
            continue
        r = subprocess.run(["which", candidate], capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout.strip()
    raise RuntimeError("clang++ not found")


def _scan_headers(upstream_dir, include_dirs=None):
    if include_dirs is None:
        include_dirs = [os.path.join(upstream_dir, "include"),
                        os.path.join(upstream_dir, "ggml", "include")]
    inc_flags = " ".join(f"-I{p}" for p in include_dirs)
    clang = _find_clang()

    # Find header paths
    llama_h = None
    ggml_h = None
    for d in include_dirs:
        for name, var in [("llama.h", "llama_h"), ("ggml.h", "ggml_h")]:
            p = os.path.join(d, name)
            if os.path.isfile(p) and not locals()[var]:
                if var == "llama_h":
                    llama_h = p
                else:
                    ggml_h = p
    assert llama_h, "llama.h not found"

    # Read source lines
    with open(llama_h) as f:
        llama_lines = f.readlines()

    # Create probe and run Clang AST dump
    probe = tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False)
    probe.write(
        '#include "llama.h"\n'
        '#include "ggml.h"\n'
        '#include "ggml-backend.h"\n'
        '#include "ggml-alloc.h"\n'
    )
    probe.close()
    try:
        cmd = [clang, "-std=c++20", "-fsyntax-only",
               "-Xclang", "-ast-dump=json"] + inc_flags.split() + [probe.name]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Clang AST dump failed:\n{r.stderr[:2000]}")
        ast_data = json.loads(r.stdout)
    finally:
        os.unlink(probe.name)

    # Macro dump
    macro_r = subprocess.run(
        [clang, "-dM", "-E"] + inc_flags.split() + ["-"],
        input='#include "llama.h"\n', capture_output=True, text=True)
    if macro_r.returncode != 0:
        raise RuntimeError(f"Clang macro dump failed:\n{macro_r.stderr[:2000]}")
    return ast_data, macro_r.stdout, llama_lines


def _node_line(node, source_lines):
    loc = node.get("loc", {})
    if not isinstance(loc, dict):
        return ""
    line_no = loc.get("line", loc.get("spellingLoc", {}).get("line", 0))
    return source_lines[line_no - 1] if 1 <= line_no <= len(source_lines) else ""


def _enum_value(node):
    if "value" in node:
        return str(node["value"])
    for child in node.get("inner", []):
        if value := _enum_value(child):
            return value
    return ""


def _type_fingerprint(node):
    type_info = node.get("type", {})
    fingerprint = {
        key: re.sub(
            r"\(anonymous (struct|union|class) at [^)]+:\d+:\d+\)",
            r"(anonymous \1)",
            type_info[key],
        )
        for key in ("qualType", "desugaredQualType")
        if key in type_info
    }
    qual_type = fingerprint.get("qualType", "")
    if "desugaredQualType" in fingerprint and FIXED_WIDTH_INTEGER.search(
        qual_type
    ):
        fingerprint["desugaredQualType"] = FIXED_WIDTH_INTEGER.sub(
            lambda match: (
                f"{'unsigned' if match.group('unsigned') else 'signed'} "
                f"{match.group('bits')}-bit integer"
            ),
            qual_type,
        )
    return fingerprint


def _layout_attributes(node):
    attributes = []
    for child in node.get("inner", []):
        kind = child.get("kind", "")
        if not kind.endswith("Attr"):
            continue
        attribute = {"kind": kind}
        value = _enum_value(child)
        if value:
            attribute["value"] = value
        for key in ("alignment", "spelling"):
            if key in child:
                attribute[key] = child[key]
        attributes.append(attribute)
    return attributes


def _record_layout(node):
    fields = []
    for child in node.get("inner", []):
        if child.get("kind") != "FieldDecl":
            continue
        field = {
            "name": child.get("name", ""),
            "type": _type_fingerprint(child),
        }
        if child.get("isBitfield"):
            field["bit_width"] = _enum_value(child)
        attributes = _layout_attributes(child)
        if attributes:
            field["attributes"] = attributes
        fields.append(field)

    layout = {
        "complete": bool(node.get("completeDefinition")),
        "tag": node.get("tagUsed", ""),
        "fields": fields,
    }
    attributes = _layout_attributes(node)
    if attributes:
        layout["attributes"] = attributes
    nested_records = []
    for child in node.get("inner", []):
        if (child.get("kind") not in ("RecordDecl", "CXXRecordDecl")
                or child.get("isImplicit")
                or not child.get("completeDefinition")):
            continue
        nested_records.append({
            "name": child.get("name", ""),
            "layout": _record_layout(child),
        })
    if nested_records:
        layout["nested_records"] = nested_records
    bases = []
    for base in node.get("bases", []):
        fingerprint = {
            key: base[key]
            for key in ("access", "writtenAccess", "isVirtual")
            if key in base
        }
        if "type" in base:
            fingerprint["type"] = _type_fingerprint(base)
        bases.append(fingerprint)
    if bases:
        layout["bases"] = bases
    return layout


def _record_fingerprint(node):
    return json.dumps(
        _record_layout(node), sort_keys=True, separators=(",", ":")
    )


def _resolved_enum_values(ast_data):
    values = {}

    def walk(node):
        if not isinstance(node, dict):
            return
        if node.get("kind") == "EnumDecl":
            previous = -1
            for child in node.get("inner", []):
                if child.get("kind") != "EnumConstantDecl":
                    continue
                value = _enum_value(child)
                if value:
                    previous = int(value, 0)
                else:
                    previous += 1
                    value = str(previous)
                values[child.get("name", "")] = value
        for child in node.get("inner", []):
            walk(child)

    if isinstance(ast_data, dict):
        walk(ast_data)
    else:
        for node in ast_data:
            walk(node)
    return values


def _object_macros(macro_output):
    macros = {}
    manual_decisions = set()
    for line in macro_output.splitlines():
        match = re.match(r"#define\s+(LLAMA_[^\s]*)(?:\s+(.*))?$", line)
        if not match:
            continue
        token, value = match.group(1), match.group(2) or ""
        if "(" in token:
            manual_decisions.add(token.split("(", 1)[0])
        elif token not in NON_API_LLAMA_MACROS:
            macros[token] = value.strip()
    return macros, manual_decisions


def _expand_macro(name, macros, stack=()):
    if name in stack:
        raise RuntimeError(f"cyclic public macro definition: {name}")
    if name not in macros:
        raise RuntimeError(f"required public macro is missing: {name}")
    value = macros[name]

    def replace(match):
        dependency = match.group(0)
        if dependency not in macros:
            return dependency
        return _expand_macro(dependency, macros, (*stack, name))

    return re.sub(r"\bLLAMA_[A-Z0-9_]+\b", replace, value)


def _typed_constant_snapshot(macros):
    return {
        name: f"{type_name}:{_expand_macro(name, macros)}"
        for name, type_name in TYPED_LLAMA_CONSTANTS.items()
    }


def _generate_typed_constants(macros):
    lines = []
    for name, type_name in TYPED_LLAMA_CONSTANTS.items():
        value = _expand_macro(name, macros)
        if type_name == "llama_state_seq_flags":
            value = f"static_cast<{type_name}>({value})"
        lines.append(f"export inline constexpr {type_name} {name} = {value};")
    return "\n".join(lines) + "\n"


def collect_api_snapshot(upstream_dir, include_dirs=None):
    ast_data, macro_output, llama_lines = _scan_headers(
        upstream_dir, include_dirs
    )
    declarations = {}
    complete_records = set()
    enum_values = _resolved_enum_values(ast_data)
    ggml_enum_members = set()

    def walk(node):
        if not isinstance(node, dict):
            return
        kind = node.get("kind", "")
        name = node.get("name", "")
        type_name = node.get("type", {}).get("qualType", "")
        if kind == "EnumDecl" and name in GGML_ENUM_TYPES:
            ggml_enum_members.update(
                child.get("name", "")
                for child in node.get("inner", [])
                if child.get("kind") == "EnumConstantDecl"
            )
        if (kind in ("FunctionDecl", "CXXMethodDecl")
                and name.startswith("llama_")
                and "LLAMA_API" in _node_line(node, llama_lines)):
            declarations[name] = f"{kind}:{type_name}"
        elif (kind in ("FunctionDecl", "CXXMethodDecl")
              and name in REQUIRED_GGML_FUNCTIONS):
            declarations[name] = f"{kind}:{type_name}"
        elif (kind in ("RecordDecl", "CXXRecordDecl")
              and (name.startswith("llama_") or name in REQUIRED_GGML_TYPES)):
            if node.get("completeDefinition"):
                declarations[name] = f"{kind}:{_record_fingerprint(node)}"
                complete_records.add(name)
            elif name not in complete_records and not node.get("isImplicit"):
                declarations[name] = f"{kind}:{_record_fingerprint(node)}"
        elif (kind in ("TypedefDecl", "ClassTemplateDecl", "EnumDecl")
              and (name.startswith("llama_") or name in REQUIRED_GGML_TYPES)):
            if kind != "TypedefDecl" or name not in complete_records:
                declarations[name] = f"{kind}:{type_name or name}"
        elif kind == "EnumConstantDecl" and name.startswith("LLAMA_"):
            declarations[name] = f"{kind}:{type_name}:{enum_values[name]}"
        elif (kind == "EnumConstantDecl"
              and (name in REQUIRED_GGML_ENUM_MEMBERS
                   or name in ggml_enum_members)):
            declarations[name] = f"{kind}:{type_name}:{enum_values[name]}"
        for child in node.get("inner", []):
            walk(child)

    if isinstance(ast_data, dict):
        walk(ast_data)
    else:
        for node in ast_data:
            walk(node)

    macros, manual_decisions = _object_macros(macro_output)

    return {
        "declarations": dict(sorted(declarations.items())),
        "macros": dict(sorted(macros.items())),
        "manual_decisions": sorted(manual_decisions),
        "typed_constants": _typed_constant_snapshot(macros),
    }


def generate_exports(upstream_dir, include_dirs=None):
    if include_dirs is None:
        include_dirs = [os.path.join(upstream_dir, "include"),
                        os.path.join(upstream_dir, "ggml", "include")]
    ast_data, macro_output, llama_lines = _scan_headers(
        upstream_dir, include_dirs
    )

    # Collect declarations.
    llama_exports = []
    ggml_exports = []
    ggml_enum_members = set()
    skipped = []

    def _walk(node):
        if not isinstance(node, dict):
            return
        kind = node.get("kind", "")
        name = node.get("name", "")
        loc = node.get("loc", {})
        fpath = ""
        if isinstance(loc, dict):
            fpath = loc.get("file", loc.get("spellingLoc", {}).get("file", ""))

        # Clang's JSON -ast-dump does not reliably include the source file name
        # via `loc.file` or `loc.includedFrom.file`; it may point to the probe
        # .cpp rather than the actual header.  Use naming heuristics instead:
        # FunctionDecl / RecordDecl / EnumDecl / EnumConstantDecl names that
        #   start with `llama_` (or `LLAMA_`) are from llama.h.
        # Types prefixed `ggml_` (or `GGML_`) are from ggml.h.
        in_llama = name.startswith("llama_") or name.startswith("LLAMA_")
        # namespace-like prefixes that belong to ggml
        in_ggml = name.startswith("ggml_") or name.startswith("GGML_")

        if kind in ("FunctionDecl", "CXXMethodDecl") and name in REQUIRED_GGML_FUNCTIONS:
            ggml_exports.append(f"export using ::{name};")

        elif kind in ("FunctionDecl", "CXXMethodDecl") and in_llama and name.startswith("llama_"):
            line_no = 0
            if isinstance(loc, dict):
                line_no = loc.get("line", loc.get("spellingLoc", {}).get("line", 1))
            if 1 <= line_no <= len(llama_lines):
                src_line = llama_lines[line_no - 1]
            else:
                src_line = ""
            if "LLAMA_API" in src_line:
                llama_exports.append(f"export using ::{name};")
            # else: static/inline helpers, skip

        elif kind in ("RecordDecl", "CXXRecordDecl", "TypedefDecl", "ClassTemplateDecl"):
            if in_llama and name.startswith("llama_"):
                llama_exports.append(f"export using ::{name};")
            elif in_ggml and name in REQUIRED_GGML_TYPES:
                ggml_exports.append(f"export using ::{name};")

        elif kind == "EnumDecl":
            if in_llama and name.startswith("llama_"):
                llama_exports.append(f"export using ::{name};")
            elif in_ggml and name in REQUIRED_GGML_TYPES:
                ggml_exports.append(f"export using ::{name};")
                if name in GGML_ENUM_TYPES:
                    for child in node.get("inner", []):
                        if child.get("kind") == "EnumConstantDecl":
                            cname = child.get("name", "")
                            if cname:
                                ggml_enum_members.add(cname)

        elif kind == "EnumConstantDecl":
            if name in REQUIRED_GGML_ENUM_MEMBERS:
                ggml_exports.append(f"export using ::{name};")
            elif in_llama and name.startswith("LLAMA_"):
                llama_exports.append(f"export using ::{name};")

        for child in node.get("inner", []):
            _walk(child)

    if isinstance(ast_data, dict):
        _walk(ast_data)
    elif isinstance(ast_data, list):
        for n in ast_data:
            _walk(n)

    # Add GGML enumerator exports
    for ename in sorted(ggml_enum_members):
        ggml_exports.append(f"export using ::{ename};")

    # Collect LLAMA_ macros
    for line in macro_output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "#define" and parts[1].startswith("LLAMA_"):
            skipped.append(f"macro {parts[1]}")

    # Deduplicate and sort
    llama_exports = sorted(set(llama_exports), key=lambda x: x.rsplit("::", 1)[-1])
    ggml_exports = sorted(set(ggml_exports), key=lambda x: x.rsplit("::", 1)[-1])
    skipped = sorted(set(skipped))

    exported_ggml = {
        line.rsplit("::", 1)[-1].removesuffix(";")
        for line in ggml_exports
    }
    missing_ggml = sorted(
        (REQUIRED_GGML_TYPES
         | REQUIRED_GGML_ENUM_MEMBERS
         | REQUIRED_GGML_FUNCTIONS)
        - exported_ggml
    )
    if missing_ggml:
        raise RuntimeError(
            "required GGML API missing from Clang AST: "
            + ", ".join(missing_ggml)
        )

    macros, _ = _object_macros(macro_output)
    return ("\n".join(llama_exports) + "\n",
            "\n".join(ggml_exports) + "\n",
            "\n".join(skipped) + "\n",
            _generate_typed_constants(macros))


def sync_outputs(output_dir, outputs, check):
    output_dir = Path(output_dir)
    if check:
        for name, content in outputs.items():
            existing = output_dir / name
            if not existing.exists():
                print(f"{name} does not exist", file=sys.stderr)
                return 1
            if existing.read_text(encoding="utf-8") != content:
                print(f"{name} differs", file=sys.stderr)
                return 1
        print("All exports match.", file=sys.stderr)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (output_dir / name).write_text(content, encoding="utf-8")
        print(f"Wrote {name} ({len(content)} bytes)", file=sys.stderr)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", help="llama.cpp checkout dir")
    ap.add_argument("--check", action="store_true")
    ap.add_argument(
        "--output-dir",
        type=Path,
        help="generated export directory (default: src/gen_exports)",
    )
    args = ap.parse_args(argv)

    upstream = Path(args.upstream) if args.upstream else DEFAULT_UPSTREAM_DIR
    if not args.upstream and not (upstream / "include/llama.h").is_file():
        print(
            f"mcpplibs:llamacpp upstream tree not found: {upstream}",
            file=sys.stderr,
        )
        return 1

    llama_inc, ggml_inc, skipped_txt, typed_constants = generate_exports(upstream)

    outputs = {
        "llama.inc": llama_inc,
        "required_ggml.inc": ggml_inc,
        "llama.skipped.txt": skipped_txt,
        "typed_constants.inc": typed_constants,
    }
    return sync_outputs(args.output_dir or DEFAULT_OUTPUT_DIR, outputs, args.check)


if __name__ == "__main__":
    sys.exit(main())
