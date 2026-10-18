import json
import os
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "spec_offline"))

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, INTER, VOCAB, BLOCK = 32, 4, 2, 8, 24, 40, 4
TARGET_LAYERS = [3, 7]


def write_safetensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, tensor in tensors.items():
        data = tensor.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()
        header[name] = {"dtype": "BF16", "shape": list(tensor.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    encoded = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def tiny_block_checkpoint(directory, seed=11):
    generator = torch.Generator().manual_seed(seed)

    def r(*shape, scale=0.3):
        return torch.randn(*shape, generator=generator) * scale

    def norm(width):
        return 1.0 + 0.1 * torch.randn(width, generator=generator)
    tensors = {"fc.weight": r(HIDDEN, HIDDEN * len(TARGET_LAYERS), scale=0.2), "hidden_norm.weight": norm(HIDDEN), "norm.weight": norm(HIDDEN)}
    for index in range(2):
        p = f"layers.{index}."
        tensors.update({
            p + "input_layernorm.weight": norm(HIDDEN), p + "post_attention_layernorm.weight": norm(HIDDEN),
            p + "self_attn.q_proj.weight": r(HEADS * HEAD_DIM, HIDDEN), p + "self_attn.k_proj.weight": r(KV_HEADS * HEAD_DIM, HIDDEN),
            p + "self_attn.v_proj.weight": r(KV_HEADS * HEAD_DIM, HIDDEN), p + "self_attn.o_proj.weight": r(HIDDEN, HEADS * HEAD_DIM),
            p + "self_attn.q_norm.weight": norm(HEAD_DIM), p + "self_attn.k_norm.weight": norm(HEAD_DIM),
            p + "mlp.gate_proj.weight": r(INTER, HIDDEN), p + "mlp.up_proj.weight": r(INTER, HIDDEN), p + "mlp.down_proj.weight": r(HIDDEN, INTER)})
    config = {"num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS, "head_dim": HEAD_DIM, "rms_norm_eps": 1e-5, "num_hidden_layers": 2,
              "rope_theta": 1000.0, "dflash_config": {"block_size": BLOCK, "target_layer_ids": TARGET_LAYERS, "mask_token_id": VOCAB - 1}}
    with open(os.path.join(directory, "config.json"), "w") as handle:
        json.dump(config, handle)
    write_safetensors(os.path.join(directory, "model.safetensors"), tensors)
    write_safetensors(os.path.join(directory, "target.safetensors"), {"embed": r(VOCAB, HIDDEN), "head": r(VOCAB, HIDDEN), "norm": norm(HIDDEN)})


@unittest.skipUnless(HAVE_TORCH, "torch is not installed")
class BlockDrafterContextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from block_drafter import BlockDrafter
        cls.temporary = tempfile.TemporaryDirectory()
        tiny_block_checkpoint(cls.temporary.name)
        cls.drafter = BlockDrafter(cls.temporary.name, os.path.join(cls.temporary.name, "target.safetensors"), {"embed": "embed", "head": "head", "norm": "norm"}, "cpu")
        generator = torch.Generator().manual_seed(3)
        cls.taps = (torch.randn(37, HIDDEN * len(TARGET_LAYERS), generator=generator)).to(torch.bfloat16)
        cls.positions = torch.arange(500, 537)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_chunked_context_matches_one_pass(self):
        whole_keys, whole_values = self.drafter.context(self.taps, self.positions, chunk=1 << 20)
        keys, values = self.drafter.context(self.taps, self.positions, chunk=5)
        for whole, chunked in zip(whole_keys + whole_values, keys + values):
            self.assertEqual(tuple(chunked.shape), (37, KV_HEADS, HEAD_DIM))
            torch.testing.assert_close(chunked.float(), whole.float(), rtol=1e-2, atol=1e-2)

    def test_chunked_context_rotates_each_row_at_its_own_position(self):
        keys, _ = self.drafter.context(self.taps, self.positions, chunk=5)
        shifted, _ = self.drafter.context(self.taps, self.positions + 7, chunk=5)
        self.assertGreater((keys[0].float() - shifted[0].float()).abs().max().item(), 0.05)
        tail, _ = self.drafter.context(self.taps[20:], self.positions[20:], chunk=5)
        torch.testing.assert_close(keys[0][20:].float(), tail[0].float(), rtol=1e-2, atol=1e-2)

    def test_window_attends_to_the_last_positions_only(self):
        keys, values = self.drafter.context(self.taps, self.positions, chunk=8)
        count, window = 30, 9
        self.drafter.window = None
        sliced = self.drafter.draft(5, 530, [k[count - window:count] for k in keys], [v[count - window:count] for v in values], window)
        self.drafter.window = window
        try:
            windowed = self.drafter.draft(5, 530, keys, values, count)
        finally:
            self.drafter.window = None
        full = self.drafter.draft(5, 530, keys, values, count)
        for a, b in zip(sliced, windowed):
            self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(full[1], windowed[1]))

    def test_window_longer_than_the_context_sees_all_of_it(self):
        keys, values = self.drafter.context(self.taps, self.positions, chunk=8)
        full = self.drafter.draft(2, 512, keys, values, 12)
        self.drafter.window = 64
        try:
            windowed = self.drafter.draft(2, 512, keys, values, 12)
        finally:
            self.drafter.window = None
        for a, b in zip(full, windowed):
            self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
