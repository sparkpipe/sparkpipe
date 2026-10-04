#!/usr/bin/env python3
"""Source contracts for the asynchronous KV binding: no synchronous copies, no removed entry points, a completion entry that never takes the KV lock, and the execution fence before every frame's stream work."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
GLM52 = ROOT / "modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_module.c"
GLM5_NEXT = ROOT / "modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_module.c"
GLM_STAGE = ROOT / "common/common_glm_stage_module/spark_glm_stage_module.h"
RUNTIME_FILES = [ROOT / "runtime/stage_kv_binding.c", ROOT / "runtime/kv_device_copy.c"]
REMOVED = [
    r"SparkKvPageStoreCopyResidentPage",
    r"SparkKvPageCacheSaveSequence",
    r"SparkKvPageCacheSaveChain",
    r"SparkKvPageCacheSaveEntry",
    r"\bSparkStageKvBindingFinish\s*\(",
    r"write_back_degraded_block_count",
]


def body(text, name):
    match = re.search(r"^[A-Za-z_][^\n;]*\b" + re.escape(name) + r"\s*\([^;{]*\)\s*\{", text, re.M)
    if match is None:
        raise AssertionError(f"{name} definition not found")
    depth, index = 1, match.end()
    while depth:
        depth += {"{": 1, "}": -1}.get(text[index], 0)
        index += 1
    return text[match.end():index - 1]


def scanned_files():
    for directory in ("cache", "runtime", "include", "modules", "model-families", "common"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and path.suffix in (".c", ".h", ".cu", ".cuh", ".inc"):
                yield path
    for pattern in ("*.c", "*.py"):
        for path in (ROOT / "tests").glob(pattern):
            if path.resolve() != SELF:
                yield path


def main():
    failures = []
    for path in RUNTIME_FILES + sorted((ROOT / "cache").glob("*.c")):
        if re.search(r"\bcudaMemcpy\s*\(", path.read_text()):
            failures.append(f"synchronous cudaMemcpy in {path.relative_to(ROOT)}")
    for path in scanned_files():
        text = path.read_text(errors="replace")
        for pattern in REMOVED:
            if re.search(pattern, text):
                failures.append(f"{pattern} still present in {path.relative_to(ROOT)}")
    binding = (ROOT / "runtime/stage_kv_binding.c").read_text()
    finish = body(binding, "SparkStageKvBindingFinishAsync")
    for forbidden in ("binding->mutex", "SparkStageKvBindingLock", "cuda"):
        if forbidden in finish:
            failures.append(f"SparkStageKvBindingFinishAsync touches {forbidden}")
    glm52 = GLM52.read_text()
    complete = body(glm52, "SparkGlm52CompleteAsync")
    if re.search(r"cuda[A-Z]\w*\s*\(", complete):
        failures.append("SparkGlm52CompleteAsync calls CUDA")
    if "SparkStageKvBindingFinishAsync(" not in complete:
        failures.append("SparkGlm52CompleteAsync does not hand the completion to SparkStageKvBindingFinishAsync")
    execute = body(glm52, "SparkGlm52ExecuteBatch")
    fence, upload = execute.find("SparkStageKvBindingFenceExecution("), execute.find("SparkStageKvBindingUploadPageTables(")
    if fence < 0 or upload < 0 or fence > upload:
        failures.append("SparkGlm52ExecuteBatch does not fence the execution stream before uploading page tables")
    glm5_next = GLM5_NEXT.read_text()
    if "SparkKvDeviceCopierFence(" not in body(glm5_next, "SparkGlm5NextStartSlot"):
        failures.append("SparkGlm5NextStartSlot does not fence the execution stream on the KV copier")
    for path, text in ((GLM_STAGE, GLM_STAGE.read_text()), (GLM5_NEXT, glm5_next)):
        if "frame->execution_stream != state->execution_stream" not in text:
            failures.append(f"the frame stream check is missing from {path.relative_to(ROOT)}")
    for path in RUNTIME_FILES:
        text = path.read_text()
        for pattern in (r"\bgetenv\s*\(", r"\busleep\s*\(", r"\bnanosleep\s*\(", r"\bsleep\s*\("):
            if re.search(pattern, text):
                failures.append(f"{pattern} in {path.relative_to(ROOT)}")
    if failures:
        for failure in failures:
            print("FAIL " + failure, file=sys.stderr)
        sys.exit(1)
    print("PASS kv binding async source contracts: no synchronous copies, removed entry points gone, completion entry lock-free, execution fence before page tables, frame stream checks kept, no sleeps or env switches")


if __name__ == "__main__":
    main()
