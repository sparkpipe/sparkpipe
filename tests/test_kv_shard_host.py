"""Run the shared latent KV shard kernels on the CPU shim against the replicated-storage oracle."""
import pathlib
import subprocess
import sys
import tempfile

from host_cuda_compiler import host_cuda_cxx

ROOT = pathlib.Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="kv-shard-host-") as directory:
    binary = str(pathlib.Path(directory) / "probe")
    subprocess.run([host_cuda_cxx(), "-std=c++17", "-O1", "-I.", "-Itests/host_cuda", "-Iinclude",
                    "-x", "c++", "tests/host_cuda/kv_shard_host.cu", "-o", binary, "-lpthread"],
                   cwd=ROOT, check=True, capture_output=True)
    subprocess.run([binary] + sys.argv[1:], check=True, timeout=1800)
