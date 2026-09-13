#!/usr/bin/env python3
"""Fail-closed source contract for exact mHC residual preservation."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / "modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_cuda.cu"
MODULE = ROOT / "modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c"


def body(source: str, name: str) -> str:
    match = re.search(r"\b" + re.escape(name) + r"\s*\([^;]*?\)\s*\{", source, re.S)
    if match is None:
        raise AssertionError(f"missing function {name}")
    start = match.end()
    depth = 1
    for index in range(start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index]
    raise AssertionError(f"unterminated function {name}")


def main() -> int:
    cuda = CUDA.read_text(encoding="utf-8")
    module = MODULE.read_text(encoding="utf-8")
    kernel = body(cuda, "SparkDsv4HcPreReduceKernel")
    enter = body(module, "SparkDsv4ModuleHcEnter")
    if "((__nv_bfloat16 *)residual_bf16)[index] = raw;" not in kernel:
        raise AssertionError("pre-reduce must preserve raw BF16 residual bits")
    if "__bfloat162float(raw)" not in kernel:
        raise AssertionError("pre-reduce must accumulate the same preserved input")
    if "cudaMemcpyAsync" in enter:
        raise AssertionError("mHC enter must not launch a separate residual copy")
    if "slot->reduced_bf16,slot->residual_bf16" not in enter:
        raise AssertionError("mHC enter must pass caller-owned residual storage")
    print("DSV4 mHC residual fusion source contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
