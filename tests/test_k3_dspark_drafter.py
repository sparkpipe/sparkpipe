#!/usr/bin/env python3
"""The K3.1 DSpark drafter's attention is causal inside the block and merges with the context exactly.

A row's latent output matches a brute-force softmax over the context and the block rows up to its own, with and without
a context, and no later block row changes it. A checkpoint of another architecture is refused by name.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "spec_offline"))
try:
    import torch
except ImportError:
    print("SKIP test_k3_dspark_drafter: torch is not installed")
    sys.exit(0)
try:
    import k3_dspark_drafter as drafter
except ImportError as error:
    print(f"SKIP test_k3_dspark_drafter: {error}")
    sys.exit(0)

KV_RANK, ROPE = 16, 8
ROWS, HEADS = 4, 3
SCALE = 0.07


def brute(absorbed, query_rope, context, block):
    out = torch.zeros(ROWS, HEADS, KV_RANK, dtype=torch.float64)
    for row in range(ROWS):
        keys = torch.cat((context, block[:row + 1]), 0).double()
        for head in range(HEADS):
            scores = (keys[:, :KV_RANK] @ absorbed[row, head].double() + keys[:, KV_RANK:] @ query_rope[row, head].double()) * SCALE
            out[row, head] = torch.softmax(scores, 0) @ keys[:, :KV_RANK]
    return out


class CausalAttention(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(7)
        self.absorbed = torch.randn(ROWS, HEADS, KV_RANK, generator=generator)
        self.query_rope = torch.randn(ROWS, HEADS, ROPE, generator=generator)
        self.block = torch.randn(ROWS, KV_RANK + ROPE, generator=generator)
        self.context = torch.randn(40, KV_RANK + ROPE, generator=generator)

    def test_matches_brute_force_with_context(self):
        carried = drafter.context_partials(self.absorbed, self.query_rope, [self.context[:13], self.context[13:]], KV_RANK, SCALE)
        out = drafter.causal_block_attention(self.absorbed, self.query_rope, self.block, KV_RANK, SCALE, carried)
        self.assertLess((out.double() - brute(self.absorbed, self.query_rope, self.context, self.block)).abs().max().item(), 1e-4)

    def test_matches_brute_force_without_context(self):
        out = drafter.causal_block_attention(self.absorbed, self.query_rope, self.block, KV_RANK, SCALE)
        empty = torch.zeros(0, KV_RANK + ROPE)
        self.assertLess((out.double() - brute(self.absorbed, self.query_rope, empty, self.block)).abs().max().item(), 1e-4)

    def test_a_later_block_row_changes_no_earlier_row(self):
        carried = drafter.context_partials(self.absorbed, self.query_rope, [self.context], KV_RANK, SCALE)
        first = drafter.causal_block_attention(self.absorbed, self.query_rope, self.block, KV_RANK, SCALE, carried)
        changed = self.block.clone()
        changed[3] += 5.0
        second = drafter.causal_block_attention(self.absorbed, self.query_rope, changed, KV_RANK, SCALE, carried)
        self.assertTrue(torch.equal(first[:3], second[:3]))
        self.assertFalse(torch.equal(first[3], second[3]))

    def test_another_architecture_is_refused(self):
        with self.assertRaises(drafter.DrafterContractError):
            drafter.refuse_unsupported({"architectures": ["DFlash2DraftModel"]})


if __name__ == "__main__":
    unittest.main()
