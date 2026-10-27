"""Run the grouped small top-k on an sm_121 device against a host reference, ties included."""
import sys

import gpu_test_runner

sys.exit(gpu_test_runner.run_cuda_source("test_topk_small_cuda", "tests/cuda/topk_small_cuda.cu"))
