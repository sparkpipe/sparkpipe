#!/usr/bin/env python3
"""MTP>0 pack-load regression guard driver.

Compiles tests/test_dsv4_mtp_head_scale_pack_load.c twice against the real
DSV4 module translation unit - once with the Flash defines and once with
-DSPARK_DSV4_PRO_BUILD=1 (the Pro alias of the shared model header) - so
both model variants prove that SparkDsv4ModuleLoadPack seeds
mtp_hc_head_scale_value from the pack's 1x1 MTP_HC_HEAD_SCALE tensor and
that readiness refuses a zero or non-finite draft gate scale.

The link uses the repository CUDA stub for cudaMemcpy and generates
never-called no-op stubs for the device launchers the module references
but never calls on this host path; any other leftover symbol fails the
gate loudly.
"""

from __future__ import annotations

import os
import pathlib
import re
import shlex
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST_C = ROOT / "tests" / "test_dsv4_mtp_head_scale_pack_load.c"
CUDA_STUB_C = ROOT / "tests" / "cuda_stub" / "cuda_runtime_stub.c"
# runtime/stage_module_common.c is deliberately NOT linked: the test TU
# carries its own hermetic stand-ins for PackRead/LoadDeviceRegion/
# CudaStatus so the multi-GB tensor regions load lazily instead of being
# fread into resident memory. Linking the real common here would collide
# with those definitions.
SCRATCH = ROOT / "build" / "tmp"

INCLUDES = [
    "-I.",
    "-Iinclude",
    "-Isrc",
    "-Itests/cuda_stub",
    "-Imodel-families/common/include",
    "-Imodel-families/dsv4/include",
    "-Imodules/dsv4_resident_decode_stage/include",
    "-Imodules/dsv4_resident_decode_stage/source",
]
BASE_FLAGS = [
    "-std=c11",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-O1",
    "-D_POSIX_C_SOURCE=200809L",
    "-D_FILE_OFFSET_BITS=64",
    "-DSPARK_BATCH_BUCKET=1024u",
    "-include",
    "model-families/dsv4/include/sparkpipe/spark_dsv4_model.h",
]
COMMON_FLAGS = [
    "-std=c11",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-O1",
    "-D_POSIX_C_SOURCE=200809L",
    "-D_FILE_OFFSET_BITS=64",
]

# TEST_VARIANT_NAME selects the pack path and the success marker.
# TEST_VARIANT_NAME selects the pack path and success marker.
VARIANTS = (
    ('flash', ['-DTEST_VARIANT_NAME="flash"']),
    ('pro', ['-DSPARK_DSV4_PRO_BUILD=1',
             '-DTEST_VARIANT_NAME="pro"']),
)

UNDEFINED_PATTERNS = (
    re.compile(r'"_?([A-Za-z_][A-Za-z0-9_]*)", referenced from'),
    re.compile("undefined[ ]reference[ ]to[ ][" + chr(96) + "']?([A-Za-z_][A-Za-z0-9_]*)"),
)
STUB_PREFIXES = ("Spark", "cuda")
MAX_LINK_ROUNDS = 3


def compiler() -> str:
    configured = os.environ.get("CC")
    if configured:
        return shlex.split(configured)[0]
    return "cc"


def run(command: list[str], what: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command, cwd=str(ROOT), text=True, capture_output=True,
        timeout=600, check=False)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        print(f"FAIL {what}")
    return result


def undefined_symbols(link_stderr: str) -> list[str]:
    symbols: set[str] = set()
    for pattern in UNDEFINED_PATTERNS:
        symbols.update(pattern.findall(link_stderr))
    return sorted(symbols)


def build_and_run(variant: str, extra_flags: list[str]) -> int:
    cc = compiler()
    stem = f"test_dsv4_mtp_head_scale_pack_load_{variant}"
    objects = []
    for source, flags, suffix in (
        (TEST_C, BASE_FLAGS + extra_flags, "test.o"),
        (CUDA_STUB_C, COMMON_FLAGS, "cuda_stub.o"),
    ):
        obj = SCRATCH / f"{stem}_{suffix}"
        build = run([cc] + flags + INCLUDES + [
            "-c", str(source), "-o", str(obj)], f"{stem} {suffix} build")
        if build.returncode != 0:
            return 1
        objects.append(obj)

    stub_c = SCRATCH / f"{stem}_link_stubs.c"
    stub_o = SCRATCH / f"{stem}_link_stubs.o"
    binary = SCRATCH / f"{stem}.bin"
    link_base = [cc] + [str(obj) for obj in objects] + ["-lm", "-lpthread"]

    linked = False
    for _round_index in range(MAX_LINK_ROUNDS):
        link = subprocess.run(
            link_base + ["-o", str(binary)], cwd=str(ROOT), text=True,
            capture_output=True, timeout=600, check=False)
        if link.returncode == 0:
            linked = True
            break
        symbols = undefined_symbols(link.stderr)
        unexpected = [
            symbol for symbol in symbols
            if not symbol.startswith(STUB_PREFIXES)]
        if not symbols or unexpected:
            print(link.stdout)
            print(link.stderr, file=sys.stderr)
            print(f"FAIL {variant} link: unexpected undefined symbols "
                  f"{unexpected or '(none parsed)'}")
            return 1
        stub_c.write_text(
            "".join(
                f"long {symbol}(void) {{ return 0L; }}\n"
                for symbol in symbols),
            encoding="utf-8")
        build = run([cc] + ["-std=c11", "-O0", "-c", str(stub_c),
                            "-o", str(stub_o)], f"{variant} link stubs build")
        if build.returncode != 0:
            return 1
        link_base.append(str(stub_o))
    if not linked:
        print(f"FAIL {variant} link: unresolved after "
              f"{MAX_LINK_ROUNDS} rounds")
        return 1

    executed = subprocess.run(
        [str(binary)], cwd=str(ROOT), text=True, capture_output=True,
        timeout=120, check=False)
    output = (executed.stdout + executed.stderr).strip()
    if executed.returncode != 0 or             f"PASS dsv4 mtp_hc_head_scale_value seeded on MTP>0 pack load ({variant})" \
            not in executed.stdout:
        print(output)
        print(f"FAIL {variant} execution")
        return 1
    print(output.replace("\n", " | "))
    return 0


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    status = 0
    for variant, extra_flags in VARIANTS:
        if build_and_run(variant, extra_flags) != 0:
            status = 1
    if status == 0:
        print("PASS dsv4 mtp head-scale pack-load guard (flash+pro)")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
