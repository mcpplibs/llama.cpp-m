#!/usr/bin/env python3
"""Verify native ELF artifacts use static or explicitly managed runtimes."""
import argparse
from pathlib import Path
import re
import subprocess


def output(command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", required=True, type=Path)
    parser.add_argument("--arch", required=True)
    parser.add_argument("binaries", nargs="+", type=Path)
    args = parser.parse_args()
    registry = (args.home / "registry").resolve()
    for binary in args.binaries:
        binary = binary.resolve()
        description = output(["file", str(binary)])
        print(description, end="")
        if args.arch not in description:
            raise SystemExit(f"Wrong artifact architecture: {binary}")
        headers = output(["readelf", "-l", str(binary)])
        interpreter = re.search(r"Requesting program interpreter: ([^\]]+)", headers)
        if not interpreter:
            if "statically linked" not in description:
                raise SystemExit(f"Artifact has neither a managed PT_INTERP nor static linkage: {binary}")
            print(f"PT_INTERP absent: {binary}")
            continue
        loader = Path(interpreter[1]).resolve()
        if not loader.is_relative_to(registry) or "xim-x-glibc" not in loader.parts:
            raise SystemExit(f"Unmanaged ELF interpreter: {loader}")
        dependencies = output([str(loader), "--list", str(binary)])
        print(dependencies, end="")
        for path in re.findall(r"(/[^\s()]+)", dependencies):
            if not Path(path).resolve().is_relative_to(registry):
                raise SystemExit(f"Host library escaped the managed runtime: {path}")
        print(f"PT_INTERP and dependency closure are managed: {binary}")


if __name__ == "__main__":
    main()
