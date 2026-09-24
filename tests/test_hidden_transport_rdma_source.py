#!/usr/bin/env python3
"""Host C++ syntax check for both builds of the weightd-proxy RDMA transport TU."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

for device_direct in (0, 1):
    subprocess.run(
        [
            "g++", "-x", "c++", "-fsyntax-only", "-std=c++17",
            "-Wall", "-Wextra", "-Werror",
            f"-DSPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT={device_direct}",
            "-Itests/rdma_syntax_stub", "-Itests/cuda_stub", "-Iinclude",
            "-Isrc", "ring/transport/rdma.cu",
        ],
        cwd=ROOT,
        check=True,
    )
print("PASS rdma.cu host syntax (host and device-direct builds)")
