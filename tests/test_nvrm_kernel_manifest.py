#!/usr/bin/env python3
"""tools/nvrm_kernel_manifest.py writes the manifest the nvrm backend checks a cubin against.

Synthetic inputs, no compiler: a stand-in nvcc that only answers --version, a
PTX with two kernels (one with a mixed, aligned parameter list and a launch
bound), and a ptxas log that also holds another translation unit's report and
a report for another target. The manifest must list this unit's kernels with
ptxas's numbers and the PTX ISA layout, and nothing is written when a kernel
has no report for the target.
"""
import hashlib
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/nvrm_kernel_manifest.py"

PTX = """
.version 9.0
.target sm_121a
.visible .entry Alpha(
	.param .u64 .ptr .global .align 1 Alpha_param_0,
	.param .u32 Alpha_param_1
)
{
	ret;
}
.visible .entry Beta(
	.param .u8 Beta_param_0,
	.param .align 8 .b8 Beta_param_1[24],
	.param .f32 Beta_param_2
)
.maxntid 128, 1, 1
.minnctapersm 2
{
	ret;
}
"""

LOG = """ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'Gamma' for 'sm_121a'
ptxas info    : Function properties for Gamma
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 9 registers, used 0 barriers
ptxas info    : Compiling entry function 'Alpha' for 'sm_90a'
ptxas info    : Function properties for Alpha
    64 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 99 registers, used 0 barriers
ptxas info    : Compiling entry function 'Alpha' for 'sm_121a'
ptxas info    : Function properties for Alpha
    16 bytes stack frame, 4 bytes spill stores, 4 bytes spill loads
ptxas info    : Used 12 registers, used 0 barriers, 8 bytes cumulative stack size
ptxas info    : Compiling entry function 'Beta' for 'sm_121a'
ptxas info    : Function properties for Beta
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 22 registers, used 1 barriers, 4000 bytes smem
"""


def run(directory, log, output):
    cubin = directory / "unit.cubin"
    source = directory / "unit.cu"
    return subprocess.run([sys.executable, "-I", str(TOOL), "--cubin", str(cubin), "--ptx", str(directory / "unit.ptx"),
                           "--ptxas-log", str(log), "--nvcc", str(directory / "nvcc"), "--flags", "-O3 -arch=sm_121a",
                           "--source", str(source), "--arch", "sm_121a", "--output", str(output)], capture_output=True, text=True)


def main():
    failures = []
    with tempfile.TemporaryDirectory(prefix="nvrm-manifest-") as scratch:
        directory = Path(scratch)
        nvcc = directory / "nvcc"
        nvcc.write_text("#!/bin/sh\necho 'Cuda compilation tools, release 13.0, V13.0.88'\n")
        nvcc.chmod(nvcc.stat().st_mode | stat.S_IXUSR)
        (directory / "unit.cubin").write_bytes(b"\x7fELF fake cubin")
        (directory / "unit.cu").write_text("source\n")
        (directory / "unit.ptx").write_text(PTX)
        log = directory / "nvcc.log"
        log.write_text(LOG)
        output = directory / "unit.manifest"
        result = run(directory, log, output)
        cubin_digest = hashlib.sha256(b"\x7fELF fake cubin").hexdigest()
        source_digest = hashlib.sha256(b"source\n").hexdigest()
        expected = "\n".join([
            "nvcc 13.0.88",
            "flags -O3 -arch=sm_121a",
            f"source {directory / 'unit.cu'} {source_digest}",
            f"cubin {cubin_digest} 15",
            "kernel Alpha registers 12 barriers 0 stack 16 spill_stores 4 spill_loads 4 smem 0",
            "param Alpha 0 u64 align 8 bytes 8 offset 0",
            "param Alpha 1 u32 align 4 bytes 4 offset 8",
            "kernel Beta registers 22 barriers 1 stack 0 spill_stores 0 spill_loads 0 smem 4000",
            "param Beta 0 u8 align 1 bytes 1 offset 0",
            "param Beta 1 b8 align 8 bytes 24 offset 8",
            "param Beta 2 f32 align 4 bytes 4 offset 32",
            "bound Beta maxntid 128 1 1",
            "bound Beta minnctapersm 2",
        ]) + "\n"
        if result.returncode != 0 or not output.is_file():
            failures.append(f"the tool failed on complete inputs: {result.stdout.strip()} {result.stderr.strip()}")
        elif output.read_text() != expected:
            failures.append("the manifest differs from the expected lines:\n" + output.read_text())
        missing = directory / "missing.log"
        missing.write_text(LOG.replace("for 'sm_121a'\nptxas info    : Function properties for Beta", "for 'sm_90a'\nptxas info    : Function properties for Beta"))
        refused = directory / "refused.manifest"
        result = run(directory, missing, refused)
        if result.returncode != 1 or "no sm_121a report for Beta" not in result.stdout or refused.exists():
            failures.append(f"a kernel with only another target's report must be refused and nothing written: {result.stdout.strip()}")
        result = run(directory, directory / "absent.log", refused)
        if result.returncode != 1 or refused.exists():
            failures.append("an unreadable log must be refused and nothing written")
        (directory / "unit.ptx").write_text(".version 9.0\n")
        result = run(directory, log, refused)
        if result.returncode != 1 or "no entry function" not in result.stdout or refused.exists():
            failures.append("a PTX without entry functions must be refused and nothing written")
    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        return 1
    print("PASS the kernel manifest tool writes this unit's kernels with ptxas's report and the PTX parameter layout, and refuses incomplete inputs")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.exit(main())
