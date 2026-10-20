"""Check the fused latent split attention against the f32 reference on a CUDA device: split tables over several parts, an empty part, a lone tensor, a peaked score row and a query count that is not a tile multiple."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "spec_offline"))

try:
    import torch
    HAVE_CUDA = torch.cuda.is_available()
except ImportError:
    HAVE_CUDA = False

if not HAVE_CUDA:
    print("SKIP test_latent_attention_kernel_cuda (needs torch with a CUDA device)")
    raise SystemExit(0)

import latent_attention_kernel as kernel  # noqa: E402
from latent_block_drafter import latent_attention  # noqa: E402

KV_RANK, ROPE, SCALE = 512, 64, 0.0721


def relative_error(got, want):
    return ((got - want).norm() / want.norm()).item()


class FusedLatentAttention(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator(device="cuda").manual_seed(11)

    def rows(self, count):
        return torch.randn((count, KV_RANK + ROPE), generator=self.generator, device="cuda").to(torch.bfloat16)

    def queries(self, blocks, heads):
        absorbed = torch.randn((blocks, heads, KV_RANK), generator=self.generator, device="cuda") * 0.6
        rope = torch.randn((blocks, heads, ROPE), generator=self.generator, device="cuda") * 0.6
        return absorbed, rope

    def test_parts_match_reference(self):
        absorbed, rope = self.queries(8, 64)
        for sizes in ((4096, 8), (65536, 65536, 3000, 8), (65536, 0, 17, 8), (1,)):
            parts = [self.rows(n) for n in sizes]
            want = latent_attention(absorbed, rope, [p for p in parts if p.shape[0]], KV_RANK, SCALE)
            got = kernel.fused_latent_attention(absorbed, rope, parts, KV_RANK, SCALE)
            self.assertEqual(got.shape, want.shape)
            self.assertEqual(got.dtype, torch.float32)
            self.assertLess(relative_error(got, want), 5e-3, sizes)

    def test_lone_tensor_and_ragged_queries(self):
        absorbed, rope = self.queries(3, 5)
        rows = self.rows(9000)
        want = latent_attention(absorbed, rope, rows, KV_RANK, SCALE)
        self.assertLess(relative_error(kernel.fused_latent_attention(absorbed, rope, rows, KV_RANK, SCALE), want), 5e-3)

    def test_peaked_row_selects_its_latent(self):
        rows = self.rows(5000)
        target = rows[1234].float()
        absorbed = target[:KV_RANK].expand(8, 64, KV_RANK) * 4.0
        rope = target[KV_RANK:].expand(8, 64, ROPE) * 4.0
        got = kernel.fused_latent_attention(absorbed, rope, [rows], KV_RANK, SCALE)
        self.assertLess(relative_error(got, target[:KV_RANK].expand(8, 64, KV_RANK)), 1e-3)

    def test_split_table_covers_every_row_once(self):
        for total in (8, 4104, 137728, 737280):
            split_rows = kernel.split_rows_for(total, 16, kernel.CONFIG["BLOCK_N"])
            self.assertEqual(split_rows % kernel.CONFIG["BLOCK_N"], 0)
            self.assertGreaterEqual(-(-total // split_rows) * 16, min(kernel.TARGET_PROGRAMS // 2, 16 * -(-total // kernel.CONFIG["BLOCK_N"])))


if __name__ == "__main__":
    unittest.main()
