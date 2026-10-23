#!/usr/bin/env python3
"""Render an installed candidate's default mcpp template, then build and run it."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcpp", default="mcpp")
    parser.add_argument("--toolchain", default="llvm@23.1.3")
    parser.add_argument("--run-arg", action="append", default=[], help="argument passed to the generated application")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "mcpp.toml").read_text())["package"]
    namespace, name, version = (metadata[key] for key in ("namespace", "name", "version"))
    identity = namespace + "." + name
    with tempfile.TemporaryDirectory(prefix=name + "-template-") as directory:
        temporary = Path(directory)
        home = temporary / "home"
        package = home / "registry/data/xpkgs" / (namespace + "-x-" + name) / version / (name + "-" + version)
        shutil.copytree(root, package, ignore=shutil.ignore_patterns(
            ".git", "target", ".mcpp", ".xlings", ".candidate-index", "__pycache__", "mcpp.lock", "compile_commands.json", "mcpp-2026.*"))
        index = home / "indices/mcpplibs/pkgs/mwrap"
        index.mkdir(parents=True)
        entry = '{url="https://example.invalid/candidate.tar.gz",sha256="fixture"}'
        descriptor = ('package={spec="1",namespace=' + json.dumps(namespace) + ',name=' + json.dumps(name) +
                      ',description="Installed template fixture",type="package",mcpp="mcpp.toml",xpm={')
        descriptor += ",".join(platform + "={[" + json.dumps(version) + "]=" + entry + "}"
                               for platform in ("linux", "macosx", "windows")) + "}}\n"
        (index / (identity + ".lua")).write_text(descriptor)
        env = dict(os.environ, MCPP_HOME=str(home))
        selector = identity + "@" + version
        for command in ([args.mcpp, "new", "--list-templates", selector],
                        [args.mcpp, "new", "quickstart", "--template", selector]):
            print("+", " ".join(command), flush=True)
            subprocess.run(command, cwd=temporary, env=env, check=True)
        project = temporary / "quickstart"
        manifest = project / "mcpp.toml"
        text = manifest.read_text()
        dependency = tomllib.loads(text)["dependencies"][identity]
        options = {"version": dependency} if isinstance(dependency, str) else dependency
        assert options["version"] == version, options
        replacement = '{ path = ' + json.dumps(package.as_posix())
        if options.get("features"):
            replacement += ', features = ' + json.dumps(options["features"])
        replacement += ' }'
        lines = text.splitlines()
        matches = [i for i, line in enumerate(lines) if line.startswith(json.dumps(identity) + " =")]
        assert len(matches) == 1, "Expected exactly one qualified dependency from self.name"
        lines[matches[0]] = json.dumps(identity) + " = " + replacement
        manifest.write_text("\n".join(lines) + "\n")
        command = [args.mcpp, "run", "--toolchain", args.toolchain]
        if args.run_arg:
            command += ["--", *args.run_arg]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, cwd=project, check=True)


if __name__ == "__main__":
    main()
