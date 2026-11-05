#!/usr/bin/env python3
"""Run the tensor-core latent shard attention against the scalar partial kernel on selected-key lists, split lists,
pair-grouped records, padded head groups and empty rows on an sm_121 device, and check that it is row-invariant."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_test_runner import run_cuda_source  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_cuda_source("test_latent_shard_sparse_cuda", "tests/cuda/latent_shard_sparse_cuda.cu"))
