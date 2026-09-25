"""Run the grouped expert skinny kernels on host threads."""
import pathlib
import subprocess
import tempfile
from host_cuda_compiler import host_cuda_cxx

ROOT = pathlib.Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="skinny-grouped-") as directory:
    binary = str(pathlib.Path(directory) / "probe")
    subprocess.run([host_cuda_cxx(), "-std=c++17", "-O2", "-pthread", "-Itests/host_cuda/shim",
                    "-Itests/host_cuda", "-I.", "-Iinclude", "-Imodel-families/common/include",
                    "-x", "c++", "tests/host_cuda/skinny_grouped_host.cu",
                    "-o", binary], cwd=ROOT, check=True)
    subprocess.run([binary], check=True, timeout=900)
