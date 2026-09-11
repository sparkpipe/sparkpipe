#!/usr/bin/env python3
"""CI wrapper for the dsv41-flash layer-0 host-oracle anchor.

Generates the synth layer-0 TP8 stagepack + per-piece expectations with
tools/dsv41_flash_layer0_oracle.py, compiles the independent C anchor
(test_dsv41_flash_layer0_anchor.c) against the same pack bytes, and requires
every piece to agree. Skipped only if no host C compiler exists.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORACLE = ROOT / "tools" / "dsv41_flash_layer0_oracle.py"
ANCHOR_C = ROOT / "tests" / "test_dsv41_flash_layer0_anchor.c"


def main() -> int:
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        print("SKIP no host C compiler")
        return 0
    with tempfile.TemporaryDirectory(prefix="dsv41_l0_") as tmp:
        result = subprocess.run(
            [sys.executable, str(ORACLE), "--out", tmp],
            capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            print("FAIL oracle generation")
            return 1
        binary = Path(tmp) / "layer0_anchor"
        compile_result = subprocess.run(
            [cc, "-O2", "-std=c11", "-Wall", "-Wextra", "-o", str(binary),
             str(ANCHOR_C), "-lm"],
            capture_output=True, text=True)
        if compile_result.returncode != 0:
            print(compile_result.stderr)
            print("FAIL anchor compile")
            return 1
        run_result = subprocess.run(
            [str(binary),
             str(Path(tmp) / "dsv41_flash_layer0_synth_tp8.spstage"),
             str(Path(tmp) / "layer0_piece_table.bin"),
             str(Path(tmp) / "layer0_expectations.bin")],
            capture_output=True, text=True, timeout=600)
        print(run_result.stderr)
        print(run_result.stdout)
        if run_result.returncode != 0:
            print("FAIL anchor disagree")
            return 1
    print("PASS dsv41 layer0 host-oracle anchor CI")
    return 0


if __name__ == "__main__":
    sys.exit(main())
