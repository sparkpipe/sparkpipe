"""Exercise actual GLM index geometry and shared KV address calculation."""
import pathlib
import subprocess
import tempfile
from host_cuda_compiler import host_cuda_cxx

ROOT = pathlib.Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="glm-index-kv-") as directory:
    binary = str(pathlib.Path(directory) / "probe")
    cuda = (ROOT / "modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_cuda.cu").read_text()
    begin = cuda.index("static void SparkGlm5NextBuildKvView(")
    end = cuda.index("\nstatic uint32_t index_ordinal_of(", begin)
    (pathlib.Path(directory) / "glm_kv_view_body.h").write_text(cuda[begin:end])
    layer = (ROOT / "modules/glm5_next_resident_decode_stage/source/cuda/layer.cuh").read_text()
    begin = layer.index("template<uint32_t THREADS, uint32_t DIM, uint32_t KPOOL, uint32_t HEADS>")
    end = layer.index('\n#include "modules/', begin)
    (pathlib.Path(directory) / "glm_pool_kernels.h").write_text(layer[begin:end])
    subprocess.run([host_cuda_cxx(), "-std=c++17", "-O0", "-I.",
                    "-Itests/host_cuda", "-Iinclude", "-I" + directory, "-Imodel-families/glm5_next/include",
                    "-x", "c++", "tests/host_cuda/glm_index_kv_host.cu",
                    "-o", binary], cwd=ROOT, check=True)
    subprocess.run([binary], check=True)
