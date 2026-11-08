#!/usr/bin/env python3
"""Run the latent shard prefill attention on an sm_121 device beside a copy of the kernel it replaced, on the same inputs, for every
launcher its callers use: dense scatter waves, decode and one-row splits, compacted and split selection lists with pair-grouped records,
padded head groups and empty rows, and landed-key gathers with key windows and blocks that span two sequences. The records must be the
same bits and the access errors the same codes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_test_runner import run_cuda_source  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_cuda_source("test_latent_shard_prefill_bits_cuda", "tests/cuda/latent_shard_prefill_bits_cuda.cu"))
