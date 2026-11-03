"""Check the BF16 conversion of the CUDA kernels on every float bit pattern on an sm_121 device."""
import sys

import gpu_test_runner

sys.exit(gpu_test_runner.run_cuda_source("test_bf16_convert_cuda", "tests/cuda/bf16_convert_cuda.cu"))
