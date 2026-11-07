#!/usr/bin/env python3
import argparse
import ast
import base64
import gzip
import hashlib
import io
import os
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools" / "sparkpipe_cli.py"
LICENSE = ROOT / "LICENSE"
NAME = "sparkpipe"
MODULE = "sparkpipe_cli"
SUMMARY = "Drive a SparkPipe fleet from your own machine: sign in, book, set up Claude Code, Codex and OpenCode, fetch and back up."
LICENSE_NAME = "FSL-1.1-ALv2"
REQUIRES_PYTHON = ">=3.9"
ENTRY = "sparkpipe = sparkpipe_cli:main"
TAG_PREFIX = "cli-v"
EPOCH = 315532800
URLS = (
    ("Homepage", "https://sparkpipe.ai"),
    ("Documentation", "https://github.com/sparkpipe/sparkpipe/blob/main/docs/FLEET_API.md"),
    ("Source", "https://github.com/sparkpipe/sparkpipe/blob/main/tools/sparkpipe_cli.py"),
)
CLASSIFIERS = (
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Developers",
    "Operating System :: MacOS",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Software Development",
)
README = """# sparkpipe

The command-line client for a SparkPipe fleet. It is one Python file with no
dependencies outside the standard library.

```sh
pip install sparkpipe
sparkpipe login
sparkpipe status
sparkpipe setup claude-code
```

- `sparkpipe login` signs in with an email code and creates a key for this machine.
- `sparkpipe status` shows what the account holds, its deadlines and the fleet's working state.
- `sparkpipe book week F3 4`, `sparkpipe book hours F1 2027-01-05T10 2` and `sparkpipe waitlist lock 144` book fleets.
- `sparkpipe keys ls`, `keys create` and `keys revoke` manage API keys.
- `sparkpipe setup claude-code`, `setup codex` and `setup opencode` configure the three coding agents for the fleet.
- `sparkpipe tokens set hf` keeps a source token in the operating system's keychain only.
- `sparkpipe fetch hf://org/model@revision models/x/` has the fleet download straight into its storage.
- `sparkpipe storage sync models/ ./backup/` copies what changed since the last sync.
- `sparkpipe proxy` serves a local endpoint for tools that cannot set headers.

The full specification is the Fleet API: https://github.com/sparkpipe/sparkpipe/blob/main/docs/FLEET_API.md
"""


def version():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == ["VERSION"]:
            return node.value.value
    raise SystemExit(f"{SOURCE} has no VERSION")


def metadata(release):
    lines = ["Metadata-Version: 2.1", f"Name: {NAME}", f"Version: {release}", f"Summary: {SUMMARY}", f"License: {LICENSE_NAME}", f"Requires-Python: {REQUIRES_PYTHON}"]
    lines += [f"Project-URL: {label}, {url}" for label, url in URLS]
    lines += [f"Classifier: {name}" for name in CLASSIFIERS]
    lines += ["Description-Content-Type: text/markdown", "", README]
    return "\n".join(lines)


def pyproject(release):
    quoted = ",\n".join(f'    "{name}"' for name in CLASSIFIERS)
    links = "\n".join(f'{label} = "{url}"' for label, url in URLS)
    return f"""[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "{NAME}"
version = "{release}"
description = "{SUMMARY}"
readme = "README.md"
requires-python = "{REQUIRES_PYTHON}"
license = {{text = "{LICENSE_NAME}"}}
classifiers = [
{quoted},
]

[project.urls]
{links}

[project.scripts]
{ENTRY.split(" = ")[0]} = "{ENTRY.split(" = ")[1]}"

[tool.setuptools]
py-modules = ["{MODULE}"]
"""


def record_line(name, data):
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{name},sha256={digest},{len(data)}"


def wheel_files(release):
    info = f"{NAME}-{release}.dist-info"
    files = [
        (f"{MODULE}.py", SOURCE.read_bytes()),
        (f"{info}/METADATA", metadata(release).encode()),
        (f"{info}/WHEEL", b"Wheel-Version: 1.0\nGenerator: sparkpipe_cli_build\nRoot-Is-Purelib: true\nTag: py3-none-any\n"),
        (f"{info}/entry_points.txt", f"[console_scripts]\n{ENTRY}\n".encode()),
        (f"{info}/LICENSE", LICENSE.read_bytes()),
    ]
    record = "\n".join([record_line(name, data) for name, data in files] + [f"{info}/RECORD,,"]) + "\n"
    return files + [(f"{info}/RECORD", record.encode())]


def build_wheel(out, release):
    path = out / f"{NAME}-{release}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in wheel_files(release):
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.external_attr = 0o644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
    return path


def build_sdist(out, release):
    top = f"{NAME}-{release}"
    files = [
        ("PKG-INFO", metadata(release).encode()),
        ("pyproject.toml", pyproject(release).encode()),
        ("README.md", README.encode()),
        ("LICENSE", LICENSE.read_bytes()),
        (f"{MODULE}.py", SOURCE.read_bytes()),
    ]
    path = out / f"{top}.tar.gz"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, data in files:
            entry = tarfile.TarInfo(f"{top}/{name}")
            entry.size, entry.mtime, entry.mode = len(data), EPOCH, 0o644
            archive.addfile(entry, io.BytesIO(data))
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=EPOCH) as stream:
        stream.write(buffer.getvalue())
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the sparkpipe sdist and wheel from tools/sparkpipe_cli.py. It never uploads.")
    parser.add_argument("--out", required=True, help="the directory for the sdist and the wheel")
    parser.add_argument("--tag", help=f"refuse unless this release tag is {TAG_PREFIX}VERSION")
    args = parser.parse_args(argv)
    release = version()
    if args.tag is not None and args.tag != TAG_PREFIX + release:
        print(f"the tag {args.tag} does not match {SOURCE.name} VERSION {release}; tag {TAG_PREFIX}{release}", file=sys.stderr)
        return 1
    out = Path(args.out)
    os.makedirs(out, exist_ok=True)
    for path in (build_sdist(out, release), build_wheel(out, release)):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
