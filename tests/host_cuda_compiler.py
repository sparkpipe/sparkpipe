#!/usr/bin/env python3
"""Select the compiler and standard flags for CPU CUDA-kernel harnesses."""

import os
import platform
import shutil


def repository_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def host_cuda_include_flags():
    """Standard include flags for harness compilations.

    The kernel headers include <sparkpipe/...> and each other relative to the
    repository root, so every host compilation needs -Iinclude; without it the
    builds fail inside inference/kernels/*.cuh on unrelated topology headers.
    """
    return ["-I" + os.path.join(repository_root(), "include")]


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
