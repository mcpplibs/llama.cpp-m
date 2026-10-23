#!/usr/bin/env python3
"""Materialize stable manifest-generated build inputs for path consumers too."""
import argparse
from pathlib import Path
import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = tomllib.loads((root / "mcpp.toml").read_text())
    for relative, content in manifest["generated_files"].items():
        path = root / relative
        if args.check:
            if not path.is_file() or path.read_text() != content:
                raise SystemExit(f"Stale or absent build input: {relative}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    print("Stable generated inputs match the manifest")


if __name__ == "__main__":
    main()
