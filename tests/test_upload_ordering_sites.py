#!/usr/bin/env python3
"""Production code orders every device copy and fill explicitly (docs/UPLOAD_ORDERING.md): no asynchronous copy, fill or event record
runs on the legacy stream 0, the CUDA flavour's asynchronous wrappers name their stream without a default, and every synchronous
SparkGpuMemcpy or SparkGpuMemset site is one that docs/UPLOAD_ORDERING.md classifies, counted per file here."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "UPLOAD_ORDERING.md"
FLAVOUR = ROOT / "runtime" / "device" / "cuda" / "spark_gpu_cuda.h"
ASYNC = {"MemcpyAsync": 5, "Memcpy2DAsync": 8, "MemsetAsync": 4, "Memset2DAsync": 6, "EventRecord": 2}
SYNC = ("Memcpy", "Memset")
LEGACY = {"0", "0u", "NULL", "nullptr", "(SparkGpuStream)0"}
SITES = {
    "inference/kernels/weight_prefetch.cuh": 1,
    "inference/runner/pass_prefill.cuh": 1,
    "inference/runner/stage_model_cuda.cuh": 2,
    "inference/runner/stage_runner.cu": 1,
    "model-families/dsv41_flash/source/dsv41_flash_stage.cuh": 19,
    "model-families/gemma4/source/gemma4_stage.cuh": 1,
    "model-families/glm52/source/glm52_stage.cuh": 1,
    "model-families/glm5_next/source/glm5_next_stage_model.cuh": 1,
    "model-families/k3/source/k3_resident_decode_stage_cuda.cu": 10,
    "model-families/k3/source/k3_stage.cuh": 2,
    "model-families/k3/source/k3_streamed_prefill.cuh": 1,
    "model-families/mimo26/source/mimo26_stage.cuh": 2,
    "model-families/motif/source/motif_stage.cuh": 1,
    "model-families/qwen38/source/qwen38_hybrid_stage.cuh": 4,
    "ring/transport/tp_device_collective.c": 28,
    "runtime/spark_weightd.c": 4,
    "runtime/spark_weightd_spine.c": 1,
}
CALL = re.compile(r"\bSparkGpu(" + "|".join(sorted(list(ASYNC) + list(SYNC), key=len, reverse=True)) + r")\s*\(")


def arguments(text, start):
    depth, current, parts = 0, "", []
    for index in range(start, len(text)):
        ch = text[index]
        if ch == "(":
            depth += 1
            if depth == 1:
                continue
        elif ch == ")":
            depth -= 1
            if depth == 0:
                parts.append(current.strip())
                return parts
        if ch == "," and depth == 1:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    return parts


def scan(name, text):
    sync_sites, legacy = 0, []
    for m in CALL.finditer(text):
        kind = m.group(1)
        parts = arguments(text, m.end() - 1)
        line = text.count("\n", 0, m.start()) + 1
        if kind in SYNC:
            sync_sites += 1
        elif len(parts) < ASYNC[kind] or parts[ASYNC[kind] - 1] in LEGACY:
            legacy.append(f"{name}:{line}: SparkGpu{kind} runs on the legacy stream; name the stream whose work it is ordered with")
    return sync_sites, legacy


def production_files():
    listed = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard"], capture_output=True)
    if listed.returncode != 0:
        return None
    names = [n for n in listed.stdout.decode("utf-8", "surrogateescape").split("\0") if n]
    return [n for n in names if n.endswith((".c", ".h", ".cu", ".cuh", ".cc", ".cpp")) and not n.startswith(("tests/", "tools/", "runtime/device/"))
            and (ROOT / n).is_file()]


def main():
    failures = []
    planted = 'SparkGpuMemsetAsync(control,0,64u,0); SparkGpuEventRecord(event); SparkGpuMemcpy(a,b,8u,kind); SparkGpuMemcpyAsync(a,b,8u,kind,stream);'
    sites, legacy = scan("planted", planted)
    if sites != 1 or len(legacy) != 2:
        failures.append(f"the scan sees {sites} synchronous sites and {len(legacy)} legacy-stream calls in the planted line, not 1 and 2")
    flavour = FLAVOUR.read_text()
    for kind in ASYNC:
        m = re.search(r"SparkGpuError SparkGpu" + kind + r"\(([^)]*)\)", flavour)
        if m is None:
            failures.append(f"the CUDA flavour defines no SparkGpu{kind}")
        elif re.search(r"SparkGpuStream\s+stream\s+SPARK_GPU_DEFAULT", m.group(1)):
            failures.append(f"the CUDA flavour's SparkGpu{kind} defaults its stream; a caller that leaves it out runs on the legacy stream")
    files = production_files()
    if files is None:
        failures.append("git ls-files failed")
        files = []
    found = {}
    for name in files:
        sites, legacy = scan(name, (ROOT / name).read_text(errors="replace"))
        failures += legacy
        if sites:
            found[name] = sites
    for name in sorted(set(found) | set(SITES)):
        if found.get(name, 0) != SITES.get(name, 0):
            failures.append(f"{name} has {found.get(name, 0)} synchronous copy or fill sites, the record says {SITES.get(name, 0)}: classify the change in "
                            f"docs/UPLOAD_ORDERING.md and update SITES")
    doc = DOC.read_text()
    for name in SITES:
        if f"`{name}`" not in doc:
            failures.append(f"docs/UPLOAD_ORDERING.md does not classify {name}")
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        print(f"\nFAIL ({len(failures)})")
        return 1
    print(f"PASS upload ordering sites: no asynchronous copy, fill or event record on the legacy stream, no default stream on those wrappers, and "
          f"{sum(found.values())} synchronous copy and fill sites in {len(found)} files, each classified in docs/UPLOAD_ORDERING.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
