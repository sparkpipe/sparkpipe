#!/usr/bin/env python3
"""Build the nvrm backend's cubin fixtures and the compiler's own report of them.

tests/nvrm/fixture_kernels.cu is compiled with the release build's flags into
tests/fixtures/nvrm/fixture_kernels.<arch>.cubin. The manifest beside it holds
the compiler version, the flags, both digests and, per kernel, ptxas's own
resource report (registers, barriers, stack frame, static shared memory), the
kernel's parameters from the PTX of the same compile (type, alignment, bytes
and offset, in declaration order, as the PTX ISA lays out .param space) and
its launch-bound and cluster directives: the oracle the host tests hold the
cubin parser to, and what the runtime checks a cubin against at load.

    python3 -I tools/nvrm_cubin_fixtures.py --nvcc PATH [--arch sm_121a] [--check]

--check compiles into a scratch directory and fails when the cubin or the
manifest in the tree differ.
"""
import argparse
import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path("tests/nvrm/fixture_kernels.cu")
FIXTURES = Path("tests/fixtures/nvrm")
INCLUDE_FLAGS = ("-I.", "-Iinclude", "-Isrc")
ENTRY = re.compile(r"^ptxas info\s+: Compiling entry function '([^']+)' for '([^']+)'$")
FRAME = re.compile(r"^\s+(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads$")
USED = re.compile(r"^ptxas info\s+: Used (\d+) registers, used (\d+) barriers(?:, (\d+) bytes smem)?(?:, .*)?$")
VERSION = re.compile(r"release [0-9.]+, V([0-9.]+)")
PTX_ENTRY = re.compile(r"\.entry\s+([A-Za-z_$][\w$]*)\s*\(([^)]*)\)([^{]*)\{")
PTX_PARAM = re.compile(r"\.param\s+(?:\.align\s+(\d+)\s+)?\.([a-z]\d+)(?:\s+\.ptr(?:\s+\.[a-z]+)*(?:\s+\.align\s+\d+)?)?\s+[\w$]+(?:\[(\d+)\])?")
PTX_DIRECTIVE = re.compile(r"\.(maxntid|reqntid|minnctapersm|maxnreg|reqnctapercluster|explicitcluster|maxclusterrank)\b([^.]*)")


class FixtureError(Exception):
    pass


def release_flags(arch):
    compute = arch.replace("sm_", "compute_", 1)
    return ["-O3", "-gencode", f"arch={compute},code={arch}"]


def nvcc_version(nvcc):
    run = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
    found = VERSION.search(run.stdout)
    if run.returncode != 0 or found is None:
        raise FixtureError(f"{nvcc} --version did not name a release: {run.stderr.strip()}")
    return found.group(1)


def report(text, arch):
    kernels = {}
    current = None
    for line in text.splitlines():
        entry = ENTRY.match(line)
        if entry:
            if entry.group(2) != arch:
                raise FixtureError(f"ptxas compiled {entry.group(1)} for {entry.group(2)}, not {arch}")
            current = entry.group(1)
            kernels[current] = {}
            continue
        frame = FRAME.match(line)
        if frame and current is not None:
            kernels[current].update(stack=int(frame.group(1)), spill_stores=int(frame.group(2)), spill_loads=int(frame.group(3)))
            continue
        used = USED.match(line)
        if used and current is not None:
            kernels[current].update(registers=int(used.group(1)), barriers=int(used.group(2)), smem=int(used.group(3) or 0))
            current = None
    wanted = ("registers", "barriers", "stack", "spill_stores", "spill_loads", "smem")
    for name, fields in kernels.items():
        if any(field not in fields for field in wanted):
            raise FixtureError(f"ptxas reported only {sorted(fields)} for {name}")
    if not kernels:
        raise FixtureError("ptxas reported no entry function")
    return kernels


def ptx_parameters(text):
    kernels = {}
    for entry in PTX_ENTRY.finditer(text):
        name, parameters, directives = entry.group(1), entry.group(2), entry.group(3)
        layout = []
        offset = 0
        for parameter in PTX_PARAM.finditer(parameters):
            kind = parameter.group(2)
            element = int(kind[1:]) // 8
            count = int(parameter.group(3) or 1)
            align = int(parameter.group(1) or element)
            offset = (offset + align - 1) // align * align
            layout.append((kind, align, element * count, offset))
            offset += element * count
        bounds = []
        for directive in PTX_DIRECTIVE.finditer(directives):
            values = [value for value in re.split(r"[\s,]+", directive.group(2).strip()) if value]
            bounds.append((directive.group(1), values))
        kernels[name] = (layout, bounds)
    if not kernels:
        raise FixtureError("the PTX of the compile declares no entry function")
    return kernels


def build(nvcc, arch, directory):
    cubin = directory / f"fixture_kernels.{arch}.cubin"
    keep = directory / "keep"
    keep.mkdir()
    flags = release_flags(arch) + ["-cubin", "-Xptxas", "-v", *INCLUDE_FLAGS]
    run = subprocess.run([nvcc, *flags, "-keep", "-keep-dir", str(keep), "-o", str(cubin), str(SOURCE)], cwd=ROOT, capture_output=True, text=True)
    if run.returncode != 0:
        raise FixtureError(f"nvcc failed: {(run.stdout + run.stderr).strip()[-600:]}")
    kernels = report(run.stdout + run.stderr, arch)
    ptx = sorted(keep.glob("*.ptx"))
    if len(ptx) != 1:
        raise FixtureError(f"the compile kept {len(ptx)} PTX files, not one")
    parameters = ptx_parameters(ptx[0].read_text())
    if sorted(parameters) != sorted(kernels):
        raise FixtureError("the PTX entry functions and ptxas's report name different kernels")
    blob = cubin.read_bytes()
    lines = [
        f"nvcc {nvcc_version(nvcc)}",
        "flags " + " ".join(flags),
        f"source {SOURCE} {hashlib.sha256((ROOT / SOURCE).read_bytes()).hexdigest()}",
        f"cubin {hashlib.sha256(blob).hexdigest()} {len(blob)}",
    ]
    for name in sorted(kernels):
        fields = kernels[name]
        lines.append(f"kernel {name} registers {fields['registers']} barriers {fields['barriers']} stack {fields['stack']} "
                     f"spill_stores {fields['spill_stores']} spill_loads {fields['spill_loads']} smem {fields['smem']}")
        layout, bounds = parameters[name]
        for index, (kind, align, size, offset) in enumerate(layout):
            lines.append(f"param {name} {index} {kind} align {align} bytes {size} offset {offset}")
        for directive, values in bounds:
            lines.append(f"bound {name} {directive}" + "".join(f" {value}" for value in values))
    return blob, "\n".join(lines) + "\n"


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
