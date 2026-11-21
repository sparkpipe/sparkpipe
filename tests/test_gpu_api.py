"""GPU runtime calls go through spark_gpu.h.

include/sparkpipe/spark_gpu.h names every GPU runtime function, type and
constant the code uses in our own names, and a device backend implements
them; runtime/device/cuda/spark_gpu_cuda.h is the CUDA one. This test checks:

- every name tools/gpu_api_sites.py writes is defined by the CUDA flavour,
  and the front header names no device API itself;
- the header compiles as C and as C++ against the CUDA stub, with every
  function addressable from C;
- CUDA runtime names (cuda followed by a capital letter) outside comments,
  strings and includes in production code. PENDING records today's count per
  file; every entry must match the tree exactly, so the count can only
  shrink as directories move to spark_gpu.h, and a new CUDA runtime name
  outside the device backend fails at once.
"""
import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("gpu_api_sites", ROOT / "tools" / "gpu_api_sites.py")
gpu_api_sites = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gpu_api_sites)

FRONT = ROOT / "include" / "sparkpipe" / "spark_gpu.h"
CUDA_FLAVOUR = ROOT / "runtime" / "device" / "cuda" / "spark_gpu_cuda.h"
STUB = ROOT / "tests" / "cuda_stub"
ROOTS = (
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
EXCLUDED_PARTS = ("validation", "tools", "__pycache__")

PENDING = {
    "include/sparkpipe/family/glm/spark_glm_head_maxloc.cuh": 2,
    "include/sparkpipe/family/module/spark_module_combine.h": 8,
    "include/sparkpipe/spark_tp_chain_graph.h": 23,
    "include/sparkpipe/spark_weightd_map.h": 1,
    "inference/runner/pass_keys.cuh": 2,
    "inference/runner/pass_prefill.cuh": 144,
    "inference/runner/stage_model_cuda.cuh": 33,
    "inference/runner/stage_runner.cu": 407,
    "model-families/common/include/sparkpipe/spark_lm_certified_launch.h": 7,
    "model-families/common/include/sparkpipe/spark_lm_kernels.cuh": 12,
    "model-families/common/include/sparkpipe/spark_tp_mesh_kernels.cuh": 266,
    "model-families/common/include/sparkpipe/spark_tp_mesh_register.h": 8,
    "model-families/dsv41_flash/source/dsv41_flash_attend.cuh": 9,
    "model-families/dsv41_flash/source/dsv41_flash_kernels.cuh": 76,
    "model-families/dsv41_flash/source/dsv41_flash_stage.cuh": 361,
    "model-families/dsv41_flash/source/dsv41_flash_stage_kernels.cuh": 3,
    "model-families/gemma4/source/gemma4_stage.cuh": 16,
    "model-families/glm52/source/glm52_row_shard.cuh": 1,
    "model-families/glm52/source/glm52_stage.cuh": 119,
    "model-families/glm5_next/source/glm5_next_stage.cuh": 29,
    "model-families/glm5_next/source/glm5_next_stage_model.cuh": 18,
    "model-families/hy4/source/hy4_stage.cuh": 14,
    "model-families/k3/include/sparkpipe/spark_k3_resident_decode_stage_cuda.h": 3,
    "model-families/k3/source/bind.cu": 4,
    "model-families/k3/source/k3_resident_decode_stage_cuda.cu": 70,
    "model-families/k3/source/k3_stage.cuh": 124,
    "model-families/k3/source/k3_streamed_prefill.cuh": 47,
    "model-families/k3/source/layer.cuh": 79,
    "model-families/k3/source/slice.cuh": 39,
    "model-families/k3/source/streamed.cuh": 80,
    "model-families/k3/source/unity.cu": 2,
    "model-families/ling/source/ling_stage.cuh": 27,
    "model-families/mimo26/source/mimo26_stage.cuh": 18,
    "model-families/motif/source/motif_stage.cuh": 31,
    "model-families/qwen38/source/qwen38_hybrid_stage.cuh": 62,
    "node/model_residentd.c": 29,
    "ring/transport/tp_device_collective.c": 98,
    "runtime/gemm.cuh": 27,
    "runtime/launch.h": 20,
    "runtime/spark_weightd.c": 34,
    "runtime/spark_weightd_attach.c": 2,
    "runtime/spark_weightd_kv_pool.c": 6,
    "runtime/spark_weightd_lazy_pack.c": 4,
    "runtime/spark_weightd_map.c": 17,
    "runtime/spark_weightd_spine.c": 3,
}


def production_files(root):
    for top in ROOTS:
        base = root / top
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in gpu_api_sites.SUFFIXES:
                continue
            relative = path.relative_to(root).as_posix()
            if any(relative.startswith(prefix + "/") for prefix in BACKEND_DIRS):
                continue
            if any(part in EXCLUDED_PARTS for part in relative.split("/")):
                continue
            yield path, relative


def scan(root):
    found = {}
    for path, relative in production_files(root):
        count = sum(gpu_api_sites.remaining(path.read_text(errors="surrogateescape")).values())
        if count:
            found[relative] = count
    return found


def compare(found, pending):
    failures = []
    for relative in sorted(set(found) | set(pending)):
        have, want = found.get(relative, 0), pending.get(relative, 0)
        if relative not in pending:
            failures.append("%s: %d CUDA runtime names outside spark_gpu.h" % (relative, have))
        elif have > want:
            failures.append("%s: CUDA runtime names grew from %d to %d" % (relative, want, have))
        elif have < want:
            failures.append("%s: CUDA runtime names fell from %d to %d; lower its PENDING entry" % (relative, want, have))
    return failures


def check_definitions():
    failures = []
    text = CUDA_FLAVOUR.read_text()
    for cuda_name, name in sorted(gpu_api_sites.NAMES.items()):
        if name.startswith("SPARK_GPU_"):
            pattern = r"^#define %s " % re.escape(name)
        elif cuda_name[4:] in gpu_api_sites.FUNCTIONS:
            pattern = r"^static inline [^\n(]*\b%s\(" % re.escape(name)
        else:
            pattern = r"^typedef [^\n;]*\b%s;" % re.escape(name)
        if not re.search(pattern, text, re.M):
            failures.append("%s: %s is not defined for %s" % (CUDA_FLAVOUR.relative_to(ROOT), name, cuda_name))
    front = FRONT.read_text()
    if gpu_api_sites.remaining(front) or re.search(r"#\s*include\s*<", front):
        failures.append("%s names a device API itself" % FRONT.relative_to(ROOT))
    return failures


def check_rewrite():
    failures = []
    source = (
        "#include <stdint.h>\n"
        "#include \"cuda_runtime_check.h\"\n"
        "static cudaError_t Copy(void *d, const void *s, size_t n, cudaStream_t q)\n"
        "{\n"
        "\tif ( cudaMemcpyAsync(d, s, n, cudaMemcpyDeviceToHost, q) != cudaSuccess )\n"
        "\t\treturn cudaErrorInvalidValue;\n"
        "\tprintf(\"cudaMemcpyAsync failed: %s\\n\", cudaGetErrorString(cudaGetLastError()));\n"
        "\tstruct cudaDeviceProp p; enum cudaMemcpyKind k = cudaMemcpyDefault;\n"
        "\treturn cudaLaunchKernelEx(0, 0);\n"
        "}\n"
    )
    expected = (
        "#include <stdint.h>\n"
        "#include \"cuda_runtime_check.h\"\n"
        "#include \"sparkpipe/spark_gpu.h\"\n"
        "static SparkGpuError Copy(void *d, const void *s, size_t n, SparkGpuStream q)\n"
        "{\n"
        "\tif ( SparkGpuMemcpyAsync(d, s, n, SPARK_GPU_COPY_DEVICE_TO_HOST, q) != SPARK_GPU_SUCCESS )\n"
        "\t\treturn SPARK_GPU_ERROR_INVALID_VALUE;\n"
        "\tprintf(\"cudaMemcpyAsync failed: %s\\n\", SparkGpuGetErrorString(SparkGpuGetLastError()));\n"
        "\tSparkGpuDeviceProperties p; SparkGpuCopyKind k = SPARK_GPU_COPY_DEFAULT;\n"
        "\treturn cudaLaunchKernelEx(0, 0);\n"
        "}\n"
    )
    result = gpu_api_sites.add_include(gpu_api_sites.rewrite_text(source))
    if result != expected:
        failures.append("rewrite gave:\n%s\nexpected:\n%s" % (result, expected))
    if gpu_api_sites.remaining(result) != {"cudaLaunchKernelEx": 1}:
        failures.append("an unmirrored name was not reported: %r" % gpu_api_sites.remaining(result))
    return failures


def compile_probe(compiler, language, source):
    with tempfile.TemporaryDirectory(prefix="gpu-api-") as scratch:
        probe = Path(scratch) / ("probe." + ("c" if language == "c" else "cc"))
        probe.write_text(source)
        command = [compiler, "-x", language, "-std=c11" if language == "c" else "-std=c++17", "-Wall", "-Wextra", "-Werror",
                   "-I" + str(STUB), "-I" + str(ROOT), "-I" + str(ROOT / "include"), "-fsyntax-only", str(probe)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            return ["%s probe does not compile against the CUDA stub:\n%s" % (language, result.stderr.strip())]
    return []


def check_compile():
    functions = ["SparkGpu" + name for name in gpu_api_sites.FUNCTIONS]
    c_source = "#include \"sparkpipe/spark_gpu.h\"\ntypedef void (*GpuApiProbeFunction)(void);\nGpuApiProbeFunction gpu_api_probe[] = {\n"
    c_source += "".join("\t(GpuApiProbeFunction)%s,\n" % name for name in functions) + "};\n"
    c_source += "int gpu_api_probe_constants[] = {\n" + "".join(
        "\t(int)%s,\n" % name for name in sorted(set(gpu_api_sites.CONSTANTS.values()))) + "};\n"
    cc_source = "#include \"sparkpipe/spark_gpu.h\"\nstatic SparkGpuError Probe(SparkGpuStream stream)\n{\n\tint *pointer = 0;\n"
    cc_source += "\treturn SparkGpuMalloc(&pointer, 4u) != SPARK_GPU_SUCCESS ? SparkGpuGetLastError() : SparkGpuMemsetAsync(pointer, 0, 4u, stream);\n}\n"
    cc_source += "SparkGpuError (*gpu_api_probe)(SparkGpuStream) = Probe;\n"
    failures = []
    compiler = shutil.which("cc")
    cxx = shutil.which("c++")
    if compiler is None or cxx is None:
        return ["no C or C++ compiler to check spark_gpu.h"]
    failures += compile_probe(compiler, "c", c_source)
    failures += compile_probe(cxx, "c++", cc_source)
    return failures


def self_check():
    failures = []
    with tempfile.TemporaryDirectory(prefix="gpu-api-tree-") as scratch:
        root = Path(scratch)
        host = root / "ring" / "control.c"
        host.parent.mkdir(parents=True)
        host.write_text("#include \"sparkpipe/spark_gpu.h\"\nint control(void) { return SparkGpuDeviceSynchronize(); }\n")
        backend = root / "runtime" / "device" / "cuda" / "control.h"
        backend.parent.mkdir(parents=True)
        backend.write_text("static inline int control(void) { return cudaDeviceSynchronize(); }\n")
        if scan(root):
            failures.append("self-check: spark_gpu.h code or the CUDA backend was counted")
        host.write_text("int control(void) { /* cudaMalloc */ return cudaDeviceSynchronize() == cudaSuccess; }\n")
        if scan(root) != {"ring/control.c": 2}:
            failures.append("self-check: CUDA runtime names in host code counted as %r" % scan(root))
    if not compare({"a.c": 3}, {"a.c": 2}):
        failures.append("self-check: a count above PENDING was accepted")
    if not compare({"a.c": 1}, {"a.c": 2}):
        failures.append("self-check: a count below PENDING was accepted")
    if not compare({}, {"a.c": 1}):
        failures.append("self-check: a PENDING entry for a clean file was accepted")
    if not compare({"a.c": 1}, {}):
        failures.append("self-check: new CUDA runtime names were accepted")
    return failures


def print_pending(found):
    print("PENDING = {")
    for relative in sorted(found):
        print('    "%s": %d,' % (relative, found[relative]))
    print("}")


def main():
    found = scan(ROOT)
    if "--print-pending" in sys.argv[1:]:
        print_pending(found)
        return 0
    failures = check_definitions() + check_rewrite() + check_compile() + self_check() + compare(found, PENDING)
    for failure in failures:
        print("  FAIL %s" % failure)
    print("gpu api: %d files with %d CUDA runtime names outside spark_gpu.h" % (len(found), sum(found.values())))
    if failures:
        print("\nFAIL (%d)" % len(failures))
        return 1
    print("\nPASS GPU runtime names go through spark_gpu.h apart from the recorded debt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
