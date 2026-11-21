#!/usr/bin/env python3
"""Build the nvrm backend's cubin fixtures and the compiler's own report of them.

tests/nvrm/fixture_kernels.cu is compiled with the release build's flags into
tests/fixtures/nvrm/fixture_kernels.<arch>.cubin. The manifest beside it holds
the compiler version, the flags, both digests and, per kernel, ptxas's own
resource report (registers, barriers, stack frame, cumulative stack, static
shared memory), the kernel's parameters from the PTX of the same compile
(type, alignment, bytes and offset, in declaration order, as the PTX ISA lays
out .param space) and its launch-bound and cluster directives: the oracle the
host tests hold the cubin parser to, and what the runtime checks a cubin
against at load. The manifest is written by tools/nvrm_kernel_manifest.py's
own code, loaded by path, so fixtures and engine modules share one parser.

    python3 -I tools/nvrm_cubin_fixtures.py --nvcc PATH [--arch sm_121a] [--check]

--check compiles into a scratch directory and fails when the cubin or the
manifest in the tree differ.
"""
import argparse
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path("tests/nvrm/fixture_kernels.cu")
FIXTURES = Path("tests/fixtures/nvrm")
INCLUDE_FLAGS = ("-I.", "-Iinclude", "-Isrc")
MANIFEST_TOOL = ROOT / "tools/nvrm_kernel_manifest.py"


class FixtureError(Exception):
    pass


def release_flags(arch):
    compute = arch.replace("sm_", "compute_", 1)
    return ["-O3", "-gencode", f"arch={compute},code={arch}"]


def manifest_tool():
    spec = importlib.util.spec_from_file_location("nvrm_kernel_manifest", MANIFEST_TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(nvcc, arch, directory):
    cubin = directory / f"fixture_kernels.{arch}.cubin"
    keep = directory / "keep"
    keep.mkdir()
    flags = release_flags(arch) + ["-cubin", "-Xptxas", "-v", *INCLUDE_FLAGS]
    run = subprocess.run([nvcc, *flags, "-keep", "-keep-dir", str(keep), "-o", str(cubin), str(SOURCE)], cwd=ROOT, capture_output=True, text=True)
    if run.returncode != 0:
        raise FixtureError(f"nvcc failed: {(run.stdout + run.stderr).strip()[-600:]}")
    ptx = sorted(keep.glob("*.ptx"))
    if len(ptx) != 1:
        raise FixtureError(f"the compile kept {len(ptx)} PTX files, not one")
    tool = manifest_tool()
    blob = cubin.read_bytes()
    try:
        text = tool.manifest(tool.nvcc_version(nvcc), " ".join(flags), str(SOURCE), (ROOT / SOURCE).read_bytes(), blob, ptx[0].read_text(),
                             run.stdout + run.stderr, arch)
    except tool.ManifestError as error:
        raise FixtureError(str(error)) from error
    return blob, text


def main(argv):
    parser = argparse.ArgumentParser(description="build the nvrm cubin fixtures with the release compiler flags")
    parser.add_argument("--nvcc", required=True, help="the nvcc of the release build")
    parser.add_argument("--arch", default="sm_121a")
    parser.add_argument("--check", action="store_true", help="fail when the fixtures in the tree differ from a fresh build")
    arguments = parser.parse_args(argv)
    try:
        with tempfile.TemporaryDirectory(prefix="nvrm-fixtures-") as scratch:
            blob, manifest = build(arguments.nvcc, arguments.arch, Path(scratch))
    except (FixtureError, OSError) as error:
        print(f"FAIL {error}")
        return 1
    cubin_path = ROOT / FIXTURES / f"fixture_kernels.{arguments.arch}.cubin"
    manifest_path = ROOT / FIXTURES / f"fixture_kernels.{arguments.arch}.manifest"
    if arguments.check:
        failures = []
        if not cubin_path.is_file() or cubin_path.read_bytes() != blob:
            failures.append(f"{cubin_path.relative_to(ROOT)} differs from a fresh build")
        if not manifest_path.is_file() or manifest_path.read_text() != manifest:
            failures.append(f"{manifest_path.relative_to(ROOT)} differs from a fresh build")
        for failure in failures:
            print(f"FAIL {failure}")
        if failures:
            return 1
        print(f"PASS the {arguments.arch} fixtures match a fresh build")
        return 0
    cubin_path.parent.mkdir(parents=True, exist_ok=True)
    cubin_path.write_bytes(blob)
    manifest_path.write_text(manifest)
    print(f"wrote {cubin_path.relative_to(ROOT)} ({len(blob)} bytes) and {manifest_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
