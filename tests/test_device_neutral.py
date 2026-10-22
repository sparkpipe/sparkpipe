"""Keep device APIs out of shared host code.

Host files (.c .h .cc .cpp .hpp .m .mm) outside the backend directories may
not include a device API header, call or name a device API (CUDA, NCCL, Metal,
Objective-C runtime, MLX), carry a type-erased device stream, graph or event
handle, or include a device translation unit. Device translation units (.cu
.cuh .metal) live only in kernel, model and backend directories and contain
device code. Production build rules never take sources, include paths or
variables from tests/.

PENDING records today's debt per file and rule, and PENDING_BUILD the
production rules that still use test stubs. Every entry must match the tree
exactly, so the debt can only shrink: a file that gets cleaner fails until its
entry is lowered or removed, and new contamination fails at once. The gate
checks itself on every run by injecting each kind of contamination into a
scratch tree.
"""
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GATED_ROOTS = (
    "api",
    "cache",
    "common",
    "deployment/src",
    "include",
    "inference",
    "model-families",
    "modules",
    "node",
    "ring",
    "runtime",
    "scheduler",
    "serving",
    "src",
    "text",
)
BACKEND_DIRS = ("runtime/device/cuda", "runtime/device/metal")
DEVICE_DIRS = ("inference/kernels", "inference/llms", "inference/runner", "model-families", "modules") + BACKEND_DIRS
EXCLUDED_PARTS = ("validation", "tools", "__pycache__")
HOST_SUFFIXES = {".c", ".h", ".cc", ".cpp", ".hpp", ".m", ".mm"}
DEVICE_SUFFIXES = {".cu", ".cuh", ".metal"}

HEADER = re.compile(
    r'^\s*#\s*(?:include|import)\s*[<"]'
    r"(?:cuda[^>\"]*|cu[A-Z][^>\"]*|mma\.h|cublas[^>\"]*|cudnn[^>\"]*|nccl\.h|nvrtc\.h|cooperative_groups[^>\"]*"
    r"|Metal/[^>\"]*|MetalKit/[^>\"]*|MetalPerformanceShaders[^>\"]*|MetalPerformancePrimitives/[^>\"]*"
    r"|Foundation/[^>\"]*|objc/[^>\"]*|mlx/[^>\"]*)[>\"]",
    re.MULTILINE,
)
TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])(?:cuda[A-Z]\w*|cu[A-Z][a-z]\w*(?=\s*\()|CU[a-z]\w*|CU_[A-Z0-9_]+"
    r"|__CUDACC__|__CUDA_ARCH__|__global__|__device__|__host__|__shared__|__nv_bfloat16\w*|__half\w*"
    r"|nccl[A-Z]\w*|MTL[A-Z]\w*|__METAL_VERSION__)"
    r"|mlx::|<<<|\bid<\s*MTL|@autoreleasepool"
)
DEVICE_INCLUDE = re.compile(r'^\s*#\s*include\s*[<"][^>"]+\.(?:cu|cuh|metal)[>"]', re.MULTILINE)
HANDLE = re.compile(r"\bvoid\s*\*+\s*(?:const\s*\**\s*)?(?:\w*_)?(?:stream|streams|graph|graphs|event|events)\b(?!\s*\()", re.IGNORECASE)
DEVICE_CODE = re.compile(
    r"__global__|__device__|__host__|__forceinline__|__shared__|<<<|\bLM_LAUNCH|\basm\s*(?:volatile\s*)?\(|"
    r"threadIdx|blockIdx|__syncthreads|__shfl|\bcuda[A-Z]|\bCU[a-zA-Z_]|\b(?:u?int|float|half|bfloat)[24]\b|__nv_|"
    r"\bkernel\s+void\b|\[\[\s*kernel\s*\]\]"
)
DEVICE_UNIT_INCLUDE = re.compile(r'^\s*#\s*include\s*[<"][^>"]+\.cuh[>"]', re.MULTILINE)

PRODUCTION_OBJECT_SETS = ("CORE", "COMPILER", "RUNTIME", "TRANSPORT", "CACHE", "MODEL_COMMON", "DEPLOYMENT")
PRODUCTION_TARGET = re.compile(
    r"^(?:build/sparkpipe_\w+|\$\((?:%s)_(?:LIBRARY|OBJECTS)\)|build/obj/\S+\.o)$" % "|".join(PRODUCTION_OBJECT_SETS)
)
ASSIGNMENT = re.compile(r"^(?:override\s+|export\s+)?([A-Za-z0-9_]+)\s*(?::=|\+=|\?=|=)\s*(.*)$")
RULE = re.compile(r"^([^\t#=:][^=:]*?)\s*:(?![=])\s*(.*)$")
VARIABLE_REFERENCE = re.compile(r"\$\(([A-Za-z0-9_]+)\)")

PENDING = {
    "include/sparkpipe/family/glm/spark_glm_head_maxloc.cuh": {"placement": 1},
    "include/sparkpipe/family/module/spark_module_combine.h": {"handle": 4, "token": 8},
    "include/sparkpipe/spark_graph_reloc.h": {"handle": 1},
    "include/sparkpipe/spark_hidden_transport.h": {"handle": 3},
    "include/sparkpipe/spark_kv_device_copy.h": {"handle": 5},
    "include/sparkpipe/spark_kv_shard.h": {"token": 3},
    "include/sparkpipe/spark_memory_buffer.h": {"handle": 1},
    "include/sparkpipe/spark_model_driver.h": {"handle": 2},
    "include/sparkpipe/spark_model_serving_adapter.h": {"handle": 1},
    "include/sparkpipe/spark_module_abi.h": {"handle": 1},
    "include/sparkpipe/spark_stage_kv_binding.h": {"handle": 7},
    "include/sparkpipe/spark_stage_module_common.h": {"header": 1, "token": 22},
    "include/sparkpipe/spark_stage_runner.h": {"handle": 3},
    "include/sparkpipe/spark_stage_runner_model.h": {"handle": 7},
    "include/sparkpipe/spark_tp_chain_graph.h": {"handle": 3, "header": 1, "token": 23},
    "include/sparkpipe/spark_tp_device_collective.h": {"handle": 22},
    "include/sparkpipe/spark_weightd_map.h": {"header": 1, "token": 1},
    "inference/kernels/layer_kind.cuh": {"rename": 1},
    "inference/kernels/tensor_map.cuh": {"rename": 1},
    "inference/llms/kimi_k3/config.h": {"device-include": 1},
    "inference/runner/stage_serving_adapter.c": {"handle": 4, "header": 1, "token": 12},
    "model-families/common/include/sparkpipe/spark_lm_certified_launch.h": {"device-include": 1, "token": 8},
    "model-families/common/include/sparkpipe/spark_tp_mesh_register.h": {"header": 1, "token": 8},
    "model-families/common/include/sparkpipe/spark_tp_mesh_round_control.h": {"token": 48},
    "model-families/glm5_next/include/sparkpipe/spark_glm5_next_index_cp.h": {"token": 3},
    "model-families/glm5_next/include/sparkpipe/spark_glm5_next_kv_shard.h": {"token": 3},
    "model-families/k3/include/sparkpipe/spark_k3_kv_shard.h": {"token": 3},
    "modules/k3_resident_decode_stage/include/sparkpipe/spark_k3_resident_decode_stage_cuda.h": {"device-include": 1, "header": 1, "token": 3},
    "node/model_residentd.c": {"header": 1, "token": 29},
    "node/weightd_mesh.c": {"header": 1, "token": 3},
    "ring/transport/hidden_transport.c": {"handle": 1},
    "ring/transport/host_staged_tcp.c": {"header": 1, "token": 21},
    "ring/transport/rdma.cu": {"placement": 1, "rename": 1},
    "ring/transport/tp_device_collective.c": {"handle": 14, "token": 100},
    "runtime/gemm.cuh": {"placement": 1},
    "runtime/gemm_descriptor_cache.h": {"token": 5},
    "runtime/kv_device_copy.c": {"handle": 1, "header": 1, "token": 50},
    "runtime/launch.h": {"device-include": 1, "token": 23},
    "runtime/memory_buffer.c": {"handle": 2, "header": 1, "token": 23},
    "runtime/spark_graph_reloc_cuda.c": {"handle": 1, "header": 2, "token": 88},
    "runtime/spark_weightd.c": {"header": 2, "token": 172},
    "runtime/spark_weightd_attach.c": {"header": 2, "token": 29},
    "runtime/spark_weightd_kv_pool.c": {"header": 2, "token": 44},
    "runtime/spark_weightd_lazy_pack.c": {"token": 4},
    "runtime/spark_weightd_map.c": {"header": 1, "token": 67},
    "runtime/spark_weightd_spine.c": {"header": 1, "token": 3},
    "runtime/spark_weightd_worker.c": {"header": 1, "token": 4},
    "runtime/stage_kv_binding.c": {"handle": 6, "token": 58},
    "runtime/stage_module_common.c": {"token": 130},
    "runtime/tensor_map.h": {"device-include": 1, "header": 1, "token": 14},
    "src/spark_sha256.c": {"token": 1},
}
PENDING_BUILD = (
    "$(MODEL_COMMON_LIBRARY)",
    "$(MODEL_COMMON_OBJECTS)",
    "build/obj/runtime/spark_weightd.o",
    "build/sparkpipe_glm52_batchplane_model",
    "build/sparkpipe_glm52_tokenize",
    "build/sparkpipe_mesh_status",
    "build/sparkpipe_model_api",
    "build/sparkpipe_model_batch",
    "build/sparkpipe_model_residentd",
    "build/sparkpipe_weightd",
    "build/sparkpipe_weightsd",
)


def relative_parts(relative):
    return relative.split("/")


def gated(relative):
    parts = relative_parts(relative)
    if not any(relative == root or relative.startswith(root + "/") for root in GATED_ROOTS):
        return False
    if any(part in EXCLUDED_PARTS for part in parts):
        return False
    return True


def in_dirs(relative, dirs):
    return any(relative.startswith(prefix + "/") for prefix in dirs)


def tree_files(root):
    for top in GATED_ROOTS:
        base = root / top
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            relative = str(path.relative_to(root))
            if gated(relative):
                yield path, relative


def scan_file(relative, text):
    suffix = Path(relative).suffix
    found = {}
    if suffix in HOST_SUFFIXES and not in_dirs(relative, BACKEND_DIRS):
        for rule, pattern in (("header", HEADER), ("token", TOKEN), ("device-include", DEVICE_INCLUDE), ("handle", HANDLE)):
            count = len(pattern.findall(text))
            if count:
                found[rule] = count
    elif suffix in DEVICE_SUFFIXES:
        if not in_dirs(relative, DEVICE_DIRS):
            found["placement"] = 1
        if not DEVICE_CODE.search(text) and not DEVICE_UNIT_INCLUDE.search(text):
            found["rename"] = 1
    return found


def scan_tree(root):
    findings = {}
    for path, relative in tree_files(root):
        found = scan_file(relative, path.read_text(errors="surrogateescape"))
        if found:
            findings[relative] = found
    return findings


def makefile_lines(text):
    joined = []
    current = ""
    for line in text.split("\n"):
        if line.endswith("\\"):
            current += line[:-1] + " "
            continue
        joined.append(current + line)
        current = ""
    if current:
        joined.append(current)
    return joined


def stub_variables(lines):
    definitions = {}
    for line in lines:
        if line.startswith("\t"):
            continue
        match = ASSIGNMENT.match(line.strip())
        if match:
            definitions.setdefault(match.group(1), []).append(match.group(2))
    stubs = {name for name, values in definitions.items() if any("tests/" in value for value in values)}
    changed = True
    while changed:
        changed = False
        for name, values in definitions.items():
            if name in stubs:
                continue
            if any(set(VARIABLE_REFERENCE.findall(value)) & stubs for value in values):
                stubs.add(name)
                changed = True
    return stubs


def build_findings(makefile_text):
    lines = makefile_lines(makefile_text)
    stubs = stub_variables(lines)
    findings = set()
    target = None
    production = False
    for line in lines:
        if line.startswith("\t"):
            body = line
        else:
            stripped = line.strip()
            match = None if not stripped or stripped.startswith("#") or ASSIGNMENT.match(stripped) else RULE.match(stripped)
            if not match:
                target = None
                production = False
                continue
            targets = match.group(1).split()
            target = targets[0]
            production = bool(targets) and all(PRODUCTION_TARGET.match(item) for item in targets) and "test" not in match.group(1)
            body = match.group(2)
        if not production or target is None:
            continue
        references = set(VARIABLE_REFERENCE.findall(body))
        if "tests/" in body or references & stubs:
            findings.add(target)
    return findings


def compare(findings, pending, built, pending_build):
    failures = []
    for relative in sorted(findings):
        expected = pending.get(relative)
        actual = findings[relative]
        if expected is None:
            failures.append(f"{relative}: device API in shared host code {actual}")
            continue
        for rule in sorted(set(actual) | set(expected)):
            have, want = actual.get(rule, 0), expected.get(rule, 0)
            if have > want:
                failures.append(f"{relative}: {rule} grew from {want} to {have}")
            elif have < want:
                failures.append(f"{relative}: {rule} fell from {want} to {have}; lower its PENDING entry")
    for relative in sorted(set(pending) - set(findings)):
        failures.append(f"{relative}: PENDING entry is clean or gone; remove it")
    for target in sorted(built - set(pending_build)):
        failures.append(f"build rule {target}: production rule takes a source, include path or variable from tests/")
    for target in sorted(set(pending_build) - built):
        failures.append(f"build rule {target}: PENDING_BUILD entry is clean or gone; remove it")
    return failures


CONTROL_HOST = "include/sparkpipe/spark_control.h"
CONTROLS = (
    ("header", "#include <cuda_runtime.h>\n"),
    ("header", '#import <Metal/Metal.h>\n'),
    ("token", "extern int cudaMalloc(void **pointer, unsigned long bytes);\n"),
    ("token", "static unsigned long buffer_bytes(MTLBuffer *buffer);\n"),
    ("token", "#if defined(__CUDACC__)\n#endif\n"),
    ("token", "static void run(void) { mlx::core::eval(); }\n"),
    ("device-include", '#include "inference/kernels/norm.cuh"\n'),
    ("handle", "typedef struct SparkControl { void *execution_stream; } SparkControl;\n"),
    ("handle", "int spark_control_wait(void *cuda_graph);\n"),
    ("handle", "int spark_control_join(void *const *streams, unsigned count);\n"),
)


def self_check():
    failures = []
    with tempfile.TemporaryDirectory(prefix="device-neutral-") as scratch:
        root = Path(scratch)
        host = root / CONTROL_HOST
        host.parent.mkdir(parents=True)
        clean = "#pragma once\n#include <stdint.h>\ntypedef struct SparkControl { uint32_t value; } SparkControl;\n"
        host.write_text(clean)
        if scan_tree(root):
            failures.append("self-check: a clean host header was flagged")
        for rule, injection in CONTROLS:
            host.write_text(clean + injection)
            if rule not in scan_tree(root).get(CONTROL_HOST, {}):
                failures.append(f"self-check: injected {rule} contamination was not caught: {injection.strip()}")
        host.write_text(clean + "typedef struct SparkControlEvents { void *event_context; } SparkControlEvents;\nstatic void *stream_reader(void *argument);\n")
        if scan_tree(root):
            failures.append("self-check: a host callback context or a function named after a stream was flagged")
        host.write_text(clean)
        misplaced = root / "ring/transport/control.cu"
        misplaced.parent.mkdir(parents=True)
        misplaced.write_text("#include <stdint.h>\nint spark_control(void) { return 0; }\n")
        found = scan_tree(root).get("ring/transport/control.cu", {})
        if "placement" not in found or "rename" not in found:
            failures.append("self-check: host code renamed to .cu under ring/ was not caught")
        misplaced.unlink()
        kernel = root / "inference/kernels/control.cuh"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("template<int T> __global__ void ControlKernel(float *x) { x[threadIdx.x] = 0.0f; }\n")
        if scan_tree(root):
            failures.append("self-check: a real kernel in a kernel directory was flagged")
        backend = root / "runtime/device/metal/device_metal.m"
        backend.parent.mkdir(parents=True)
        backend.write_text("#import <Metal/Metal.h>\nid<MTLDevice> device;\n")
        if scan_tree(root):
            failures.append("self-check: a backend file was flagged")
    makefile = (
        "STUB_FLAGS := -Itests/cuda_stub\n"
        "COMMON_FLAGS := -Iinclude $(STUB_FLAGS)\n"
        "CLEAN_FLAGS := -Iinclude\n"
        "build/sparkpipe_control: node/control.c $(CORE_LIBRARY)\n\t$(CC) $(COMMON_FLAGS) $^ -o $@\n"
        "build/sparkpipe_clean: node/clean.c $(CORE_LIBRARY)\n\t$(CC) $(CLEAN_FLAGS) $^ -o $@\n"
        "$(RUNTIME_OBJECTS): SP_INCLUDE_FLAGS = $(COMMON_FLAGS)\n"
        "build/test_control: tests/test_control.c $(STUB_FLAGS)\n\t$(CC) $(STUB_FLAGS) $^ -o $@\n"
    )
    built = build_findings(makefile)
    if built != {"build/sparkpipe_control", "$(RUNTIME_OBJECTS)"}:
        failures.append(f"self-check: build rule scan found {sorted(built)}")
    if not compare({"a.h": {"token": 2}}, {"a.h": {"token": 3}}, set(), ()):
        failures.append("self-check: a PENDING count above the tree was accepted")
    if not compare({}, {"a.h": {"token": 1}}, set(), ()):
        failures.append("self-check: a PENDING entry for a clean file was accepted")
    if not compare({"a.h": {"token": 1}}, {}, set(), ()):
        failures.append("self-check: new contamination was accepted")
    if not compare({}, {}, {"build/sparkpipe_x"}, ()):
        failures.append("self-check: a new stub build rule was accepted")
    return failures


def print_pending(findings, built):
    print("PENDING = {")
    for relative in sorted(findings):
        rules = ", ".join(f'"{rule}": {count}' for rule, count in sorted(findings[relative].items()))
        print(f'    "{relative}": {{{rules}}},')
    print("}")
    print("PENDING_BUILD = (")
    for target in sorted(built):
        print(f'    "{target}",')
    print(")")


def main():
    findings = scan_tree(ROOT)
    built = build_findings((ROOT / "Makefile").read_text())
    if "--print-pending" in sys.argv[1:]:
        print_pending(findings, built)
        return 0
    failures = self_check() + compare(findings, PENDING, built, PENDING_BUILD)
    for failure in failures:
        print(f"  FAIL {failure}")
    debt = sum(sum(rules.values()) for rules in findings.values())
    print(f"device-neutral: {len(findings)} pending files ({debt} hits), {len(built)} pending build rules")
    if failures:
        print(f"\nFAIL ({len(failures)})")
        return 1
    print("\nPASS shared host code is device-neutral apart from the recorded debt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
