import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

try:
    import torch
    HAVE_CUDA = torch.cuda.is_available()
except ImportError:
    HAVE_CUDA = False


def bf16_values(rng, shape):
    raw = rng.standard_normal(shape).astype(np.float32)
    bits = raw.view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).view(np.float32)


@unittest.skipUnless(HAVE_CUDA, "needs a CUDA device")
class DraftdKernelTest(unittest.TestCase):
    def setUp(self):
        from draftd_kernels import kernels
        self.k = kernels()
        self.rng = np.random.default_rng(7)

    def test_gemv_bf16_matches_f32_matvec(self):
        w = bf16_values(self.rng, (300, 1536))
        x = bf16_values(self.rng, (1536,))
        out = torch.empty(300, dtype=torch.float32, device="cuda")
        self.k.gemv_bf16(torch.as_tensor(w).to(torch.bfloat16).cuda(), torch.as_tensor(x).cuda(), out)
        want = w.astype(np.float64) @ x.astype(np.float64)
        np.testing.assert_allclose(out.cpu().numpy(), want, rtol=2e-5, atol=2e-4)

    def test_gemv_fp8_block_decodes_every_code_and_selects_experts(self):
        from t1_reference_common import _E4M3_LUT
        experts, rows, cols = 5, 256, 384
        codes = self.rng.integers(0, 256, size=(experts, rows, cols), dtype=np.uint8)
        codes[codes == 0x7F] = 0x7E
        codes[codes == 0xFF] = 0xFE
        codes[0, 0, :256] = np.arange(256, dtype=np.uint8)
        codes[0, 0, 0x7F] = 0
        codes[0, 0, 0xFF] = 0
        scale = self.rng.uniform(0.001, 0.01, size=(experts, rows // 128, cols // 128)).astype(np.float32)
        ids = np.array([3, 0, 3], dtype=np.int64)
        x = bf16_values(self.rng, (3, cols))
        out = torch.empty((3, rows), dtype=torch.float32, device="cuda")
        self.k.gemv_fp8_block(torch.as_tensor(codes).cuda(), torch.as_tensor(scale).cuda(), torch.as_tensor(ids).cuda(),
                              torch.as_tensor(x).cuda(), out)
        for i, e in enumerate(ids):
            dense = _E4M3_LUT[codes[e]].reshape(rows // 128, 128, cols // 128, 128) * scale[e][:, None, :, None]
            want = dense.reshape(rows, cols).astype(np.float64) @ x[i].astype(np.float64)
            np.testing.assert_allclose(out[i].cpu().numpy(), want, rtol=1e-4, atol=1e-4)

    def test_gemv_fp8_block_broadcasts_one_input_row(self):
        codes = self.rng.integers(0, 0x70, size=(2, 128, 128), dtype=np.uint8)
        scale = np.ones((2, 1, 1), dtype=np.float32)
        x = bf16_values(self.rng, (128,))
        out = torch.empty((2, 128), dtype=torch.float32, device="cuda")
        self.k.gemv_fp8_block(torch.as_tensor(codes).cuda(), torch.as_tensor(scale).cuda(),
                              torch.as_tensor(np.array([1, 0])).cuda(), torch.as_tensor(x).cuda(), out)
        from t1_reference_common import _E4M3_LUT
        np.testing.assert_allclose(out[0].cpu().numpy(), _E4M3_LUT[codes[1]] @ x, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(out[1].cpu().numpy(), _E4M3_LUT[codes[0]] @ x, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
