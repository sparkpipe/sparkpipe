#!/usr/bin/env python3
"""Host C syntax check of the weightd-proxy RDMA transport TU."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

subprocess.run(
    [
        "cc", "-x", "c", "-fsyntax-only", "-std=c11", "-D_GNU_SOURCE",
        "-Wall", "-Wextra", "-Werror",
        "-Itests/rdma_syntax_stub", "-Iinclude",
        "-Isrc", "ring/transport/rdma.c",
    ],
    cwd=ROOT,
    check=True,
)
print("PASS rdma.c host syntax")
