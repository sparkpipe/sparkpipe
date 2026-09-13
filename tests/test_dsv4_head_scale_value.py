#!/usr/bin/env python3
"""P0-A release-blocker regression guard: hc_head_scale_value is non-zero
after pack load.

Builds and RUNS tests/test_dsv4_head_scale_value.c, which compiles the real
DSV4 resident-decode-stage module translation unit in place so its static
bind/coverage seam is executable on the host:

- binding the pack 1x1 HC_HEAD_SCALE / MTP_HC_HEAD_SCALE tensors seeds
  hc_head_scale_value / mtp_hc_head_scale_value by value (the fix for the
  declared-but-never-assigned blocker that emitted every token at scale
  0.0f), and
- readiness refuses zero/non-finite head scales, so a regression to
  "never assigned" fails here instead of zero-gating every token.

The module references device-only kernel launchers it never calls on this
path; after a failed link the driver generates never-called no-op stubs
for the leftover Spark*/cuda* symbols and relinks. Any other leftover
symbol is an unexpected dependency and fails the gate loudly.
"""
from __future__ import annotations

import os
import pathlib
import re
import shlex
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST_C = ROOT / "tests" / "test_dsv4_head_scale_value.c"
CUDA_STUB_C = ROOT / "tests" / "cuda_stub" / "cuda_runtime_stub.c"
COMMON_C = ROOT / "runtime" / "stage_module_common.c"
SCRATCH = ROOT / "build" / "tmp"

INCLUDES = [
    "-I.",
    "-Iinclude",
    "-Isrc",
    "-Itests/cuda_stub",
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

# Apple ld quotes mangled names ("_Sym", referenced from); GNU ld writes
# an undefined-reference diagnostic naming the bare symbol. The GNU form
# quotes the name with a backtick-plus-quote pair, spliced in via chr(96).
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


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    cc = compiler()

    objects = []
    for source, flags, obj_name in (
        (TEST_C, BASE_FLAGS, "test_dsv4_head_scale_value.o"),
        (CUDA_STUB_C, COMMON_FLAGS, "test_dsv4_head_scale_value_cuda_stub.o"),
        (COMMON_C, COMMON_FLAGS, "test_dsv4_head_scale_value_common.o"),
    ):
        obj = SCRATCH / obj_name
        build = run([cc] + flags + INCLUDES + [
            "-c", str(source), "-o", str(obj)], f"{obj_name} build")
        if build.returncode != 0:
            return 1
        objects.append(obj)

    stub_c = SCRATCH / "test_dsv4_head_scale_value_link_stubs.c"
    stub_o = SCRATCH / "test_dsv4_head_scale_value_link_stubs.o"
    binary = SCRATCH / "test_dsv4_head_scale_value.bin"
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
            print("FAIL head-scale unit link: unexpected undefined symbols "
                  f"{unexpected or '(none parsed)'}")
            return 1
        # Never-called no-ops: they exist purely so the host linker accepts
        # the device-launcher references the real module TU carries.
        stub_c.write_text(
            "".join(
                f"long {symbol}(void) {{ return 0L; }}\n"
                for symbol in symbols),
            encoding="utf-8")
        build = run([cc] + ["-std=c11", "-O0", "-c", str(stub_c),
                          "-o", str(stub_o)], "link stubs build")
        if build.returncode != 0:
            return 1
        link_base.append(str(stub_o))
    if not linked:
        print("FAIL head-scale unit link: unresolved after "
              f"{MAX_LINK_ROUNDS} rounds")
        return 1

    executed = subprocess.run(
        [str(binary)], cwd=str(ROOT), text=True, capture_output=True,
        timeout=120, check=False)
    output = (executed.stdout + executed.stderr).strip()
    if executed.returncode != 0 or "PASS dsv4 hc_head_scale_value" \
            not in executed.stdout:
        print(output)
        print("FAIL head-scale unit execution")
        return 1
    print(output.replace("\n", " | "))
    print("PASS dsv4 head-scale pack-load unit (P0-A guard executed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
