import ctypes
import hashlib
import os
import subprocess

import torch

SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "draftd_kernels.cu")
ARCH = os.environ.get("DRAFTD_CUDA_ARCH", "sm_120")
NVCC = os.environ.get("DRAFTD_NVCC", "/usr/local/cuda/bin/nvcc")


class DraftdKernels:
    def __init__(self, library):
        self.lib = ctypes.CDLL(library)
        self.lib.spark_draftd_gemv_bf16.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        self.lib.spark_draftd_gemv_fp8_block.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
        self.wide = int(os.environ.get("DRAFTD_ACCUMULATE", "f32") == "f64")

    @staticmethod
    def _stream():
        return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    def gemv_bf16(self, w, x, out):
        if w.dtype != torch.bfloat16 or x.dtype != torch.float32 or out.dtype != torch.float32:
            raise TypeError("gemv_bf16 takes bf16 weights and f32 input/output")
        if not (w.is_contiguous() and x.is_contiguous() and out.is_contiguous()):
            raise ValueError("gemv_bf16 needs contiguous tensors")
        rows, cols = w.shape
        if x.numel() != cols or out.numel() != rows:
            raise ValueError(f"gemv_bf16 shape mismatch {tuple(w.shape)} x {x.numel()} -> {out.numel()}")
        status = self.lib.spark_draftd_gemv_bf16(w.data_ptr(), x.data_ptr(), out.data_ptr(), rows, cols, self.wide,
                                                 self._stream())
        if status:
            raise RuntimeError(f"spark_draftd_gemv_bf16 status {status}")

    def gemv_fp8_block(self, codes, scale, ids, x, out):
        if codes.dtype != torch.uint8 or scale.dtype != torch.float32 or ids.dtype != torch.int64:
            raise TypeError("gemv_fp8_block takes u8 codes, f32 block scales and i64 ids")
        if not all(t.is_contiguous() for t in (codes, scale, ids, x, out)):
            raise ValueError("gemv_fp8_block needs contiguous tensors")
        _, rows, cols = codes.shape
        count = ids.numel()
        if scale.shape[1:] != (rows // 128, cols // 128) or x.shape[-1] != cols or out.numel() != count * rows:
            raise ValueError("gemv_fp8_block shape mismatch")
        stride = cols if x.dim() == 2 and x.shape[0] == count else 0
        status = self.lib.spark_draftd_gemv_fp8_block(codes.data_ptr(), scale.data_ptr(), ids.data_ptr(), x.data_ptr(),
                                                      out.data_ptr(), rows, cols, stride, count, self.wide,
                                                      self._stream())
        if status:
            raise RuntimeError(f"spark_draftd_gemv_fp8_block status {status}")


_KERNELS = None


def build(cache_dir=None):
    with open(SOURCE, "rb") as fh:
        digest = hashlib.sha256(fh.read() + ARCH.encode()).hexdigest()[:16]
    cache_dir = cache_dir or os.environ.get("DRAFTD_KERNEL_CACHE", os.path.expanduser("~/.cache/draftd"))
    os.makedirs(cache_dir, exist_ok=True)
    library = os.path.join(cache_dir, f"draftd_kernels_{ARCH}_{digest}.so")
    if not os.path.exists(library):
        partial = library + ".partial"
        subprocess.run([NVCC, "-O3", f"-arch={ARCH}", "-shared", "-Xcompiler", "-fPIC", "-o", partial, SOURCE], check=True)
        os.replace(partial, library)
    return library


def kernels():
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = DraftdKernels(build())
    return _KERNELS
