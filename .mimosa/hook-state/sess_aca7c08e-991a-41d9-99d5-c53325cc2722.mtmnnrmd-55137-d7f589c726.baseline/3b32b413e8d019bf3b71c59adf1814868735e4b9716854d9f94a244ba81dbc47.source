#!/usr/bin/env python3
"""Select the compiler used by CPU CUDA-kernel harnesses."""

import os
import platform
import shutil


def host_cuda_cxx():
    configured = os.environ.get("SPARKPIPE_HOST_CUDA_CXX")
    if configured:
        if shutil.which(configured) is None:
            raise RuntimeError(
                f"SPARKPIPE_HOST_CUDA_CXX is not executable: {configured}")
        return configured
    if platform.system() != "Darwin":
        return "g++"
    for version in range(20, 10, -1):
        candidate = f"g++-{version}"
        if shutil.which(candidate) is not None:
            return candidate
    raise RuntimeError(
        "the CUDA CPU harnesses require GNU g++ on macOS; install Homebrew "
        "gcc or set SPARKPIPE_HOST_CUDA_CXX")
