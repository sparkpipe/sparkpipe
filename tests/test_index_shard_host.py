"""Run the shared index-shard merge, KV gather and query pack kernels on the CPU shim against replicated oracles."""
import pathlib
import subprocess
import sys
import tempfile

from host_cuda_compiler import host_cuda_cxx

ROOT = pathlib.Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="index-shard-host-") as directory:
    binary = str(pathlib.Path(directory) / "probe")
    subprocess.run([host_cuda_cxx(), "-std=c++17", "-O2", "-pthread", "-Itests/host_cuda/shim", "-Itests/host_cuda", "-I.", "-Iinclude", "-Imodel-families/common/include",
                    "-x", "c++", "tests/host_cuda/index_shard_host.cu", "-o", binary, "-lpthread"],
                   cwd=ROOT, check=True)
    subprocess.run([binary] + sys.argv[1:], check=True, timeout=1800)
