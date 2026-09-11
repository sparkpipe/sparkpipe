"""Exercise actual GLM index geometry and shared KV address calculation."""
import pathlib
import subprocess
import tempfile
from host_cuda_compiler import host_cuda_cxx

ROOT = pathlib.Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="glm-index-kv-") as directory:
    binary = str(pathlib.Path(directory) / "probe")
    subprocess.run([host_cuda_cxx(), "-std=c++17", "-O0", "-I.",
                    "-Itests/host_cuda", "-Imodel-families/glm5_next/include",
                    "-x", "c++", "tests/host_cuda/glm_index_kv_host.cu",
                    "-o", binary], cwd=ROOT, check=True)
    subprocess.run([binary], check=True)
