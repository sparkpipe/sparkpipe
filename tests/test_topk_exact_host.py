#!/usr/bin/env python3
"""Run the DSA top-k on host threads: exact top-k, lowest index on ties, ascending index order, the same answer on every run."""
import pathlib
import subprocess
import tempfile
from host_cuda_compiler import host_cuda_cxx

ROOT = pathlib.Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="topk-exact-") as directory:
    binary = str(pathlib.Path(directory) / "probe")
    subprocess.run([host_cuda_cxx(), "-std=c++17", "-O2", "-pthread", "-Itests/host_cuda/shim",
                    "-Itests/host_cuda", "-I.", "-Iinclude", "-Imodel-families/common/include",
                    "-x", "c++", "tests/host_cuda/topk_exact_host.cu", "-o", binary], cwd=ROOT, check=True)
    subprocess.run([binary], check=True, timeout=600)
