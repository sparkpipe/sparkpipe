"""Check on an sm_121 device that every row of a 2-16 row skinny frame equals that row launched alone, bit for bit.

For bf16, fp8, mxfp8, mxfp4 and nvfp4 weights at DSV4.1, GLM 5.3 and A.X K2 per-rank projection shapes and at shapes that
reach every lane width, chunk tail and split path, it runs each frame through LmSkinnyLaunch, LmSkinnyLaunchRows,
LmSkinnyDenseMulti and every rows-per-task and neurons-per-task shape of the frame kernel, compares each row with its
one-row launch and checks that no output outside the frame is written. Speculative verify frames rely on this to
reproduce decode exactly.
"""
import sys

import gpu_test_runner

sys.exit(gpu_test_runner.run_cuda_source("test_skinny_frame_rows_cuda", "tests/cuda/skinny_frame_rows_cuda.cu"))
