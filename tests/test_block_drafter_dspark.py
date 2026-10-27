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

HIDDEN, HEADS, HEAD_DIM, INTER, VOCAB, DRAFT_VOCAB, RANK, BLOCK, WINDOW = 32, 4, 8, 24, 48, 12, 6, 4, 5
TARGET_LAYERS = [2, 5, 7]


def write_safetensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, tensor in tensors.items():
        if tensor.dtype == torch.int64:
            data, dtype = tensor.contiguous().numpy().tobytes(), "I64"
        else:
            data, dtype = tensor.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes(), "BF16"
        header[name] = {"dtype": dtype, "shape": list(tensor.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    encoded = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def tiny_dspark_checkpoint(directory, seed=5):
    generator = torch.Generator().manual_seed(seed)

    def r(*shape, scale=0.3):
        return torch.randn(*shape, generator=generator) * scale

    def norm(width):
        return 1.0 + 0.1 * torch.randn(width, generator=generator)
    targets = torch.sort(torch.randperm(VOCAB, generator=generator)[:DRAFT_VOCAB]).values
    tensors = {"fc.weight": r(HIDDEN, HIDDEN * len(TARGET_LAYERS), scale=0.2), "hidden_norm.weight": norm(HIDDEN), "norm.weight": norm(HIDDEN),
               "embed_tokens.weight": r(VOCAB, HIDDEN), "lm_head.weight": r(DRAFT_VOCAB, HIDDEN, scale=1.0),
               "d2t": (targets - torch.arange(DRAFT_VOCAB)).to(torch.int64),
               "markov_head.markov_w1.weight": r(VOCAB, RANK, scale=1.0), "markov_head.markov_w2.weight": r(DRAFT_VOCAB, RANK, scale=1.0),
               "confidence_head.proj.weight": r(1, HIDDEN + RANK), "confidence_head.proj.bias": r(1)}
    for index in range(2):
        p = f"layers.{index}."
        tensors.update({
            p + "input_layernorm.weight": norm(HIDDEN), p + "post_attention_layernorm.weight": norm(HIDDEN),
            p + "self_attn.q_proj.weight": r(HEADS * HEAD_DIM, HIDDEN), p + "self_attn.k_proj.weight": r(HEADS * HEAD_DIM, HIDDEN),
            p + "self_attn.v_proj.weight": r(HEADS * HEAD_DIM, HIDDEN), p + "self_attn.o_proj.weight": r(HIDDEN, HEADS * HEAD_DIM),
            p + "self_attn.q_norm.weight": norm(HEAD_DIM), p + "self_attn.k_norm.weight": norm(HEAD_DIM),
            p + "mlp.gate_proj.weight": r(INTER, HIDDEN), p + "mlp.up_proj.weight": r(INTER, HIDDEN), p + "mlp.down_proj.weight": r(HIDDEN, INTER)})
    config = {"architectures": ["DSparkDraftModel"], "aux_hidden_state_layer_ids": TARGET_LAYERS, "block_size": BLOCK, "mask_token_id": VOCAB - 1,
              "draft_vocab_size": DRAFT_VOCAB, "markov_rank": RANK,
              "transformer_layer_config": {"num_attention_heads": HEADS, "num_key_value_heads": HEADS, "head_dim": HEAD_DIM, "hidden_size": HIDDEN,
                                           "rms_norm_eps": 1e-6, "num_hidden_layers": 2, "sliding_window": WINDOW, "use_sliding_window": True,
                                           "rope_parameters": {"rope_theta": 1000.0, "rope_type": "default"}, "vocab_size": VOCAB}}
    with open(os.path.join(directory, "config.json"), "w") as handle:
        json.dump(config, handle)
    write_safetensors(os.path.join(directory, "model.safetensors"), tensors)
    write_safetensors(os.path.join(directory, "head.safetensors"), {"lm_head.weight": r(VOCAB, HIDDEN)})
    write_safetensors(os.path.join(directory, "norm.safetensors"), {"model.norm.weight": norm(HIDDEN)})
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as handle:
        json.dump({"weight_map": {"lm_head.weight": "head.safetensors", "model.norm.weight": "norm.safetensors"}}, handle)
    return tensors, targets


def tiny_target_dspark_checkpoint(directory, seed=6):
    generator = torch.Generator().manual_seed(seed)

    def r(*shape, scale=0.3):
        return torch.randn(*shape, generator=generator) * scale

    def norm(width):
        return 1.0 + 0.1 * torch.randn(width, generator=generator)
    draft = {"fc.weight": r(HIDDEN, HIDDEN * len(TARGET_LAYERS), scale=0.2), "hidden_norm.weight": norm(HIDDEN), "norm.weight": norm(HIDDEN),
             "markov_head.markov_w1.weight": r(VOCAB, RANK, scale=1.0), "markov_head.markov_w2.weight": r(VOCAB, RANK, scale=1.0),
             "confidence_head.proj.weight": r(1, HIDDEN + RANK), "confidence_head.proj.bias": r(1)}
    for index in range(3):
        p = f"layers.{index}."
        draft.update({
            p + "input_layernorm.weight": norm(HIDDEN), p + "post_attention_layernorm.weight": norm(HIDDEN),
            p + "self_attn.q_proj.weight": r(HEADS * HEAD_DIM, HIDDEN), p + "self_attn.k_proj.weight": r(2 * HEAD_DIM, HIDDEN),
            p + "self_attn.v_proj.weight": r(2 * HEAD_DIM, HIDDEN), p + "self_attn.o_proj.weight": r(HIDDEN, HEADS * HEAD_DIM),
            p + "self_attn.q_norm.weight": norm(HEAD_DIM), p + "self_attn.k_norm.weight": norm(HEAD_DIM),
            p + "mlp.gate_proj.weight": r(INTER, HIDDEN), p + "mlp.up_proj.weight": r(INTER, HIDDEN), p + "mlp.down_proj.weight": r(HIDDEN, INTER)})
    config = {"architectures": ["ExaoneMoeForCausalLM"], "num_attention_heads": HEADS, "num_key_value_heads": 2, "head_dim": HEAD_DIM, "hidden_size": HIDDEN,
              "rms_norm_eps": 1e-5, "num_hidden_layers": 9, "sliding_window": WINDOW, "use_sliding_window": True,
              "rope_parameters": {"rope_theta": 1000.0, "rope_type": "default"}, "vocab_size": VOCAB,
              "dspark_target_layer_ids": TARGET_LAYERS, "dspark_num_hidden_layers": 3, "dspark_block_size": BLOCK + 1, "dspark_noise_token_id": 7,
              "dspark_markov_rank": RANK}
    with open(os.path.join(directory, "config.json"), "w") as handle:
        json.dump(config, handle)
    write_safetensors(os.path.join(directory, "model-dspark.safetensors"), {"mtp." + name: tensor for name, tensor in draft.items()})
    target = {"model.embed_tokens.weight": r(VOCAB, HIDDEN), "lm_head.weight": r(VOCAB, HIDDEN, scale=1.0), "model.norm.weight": norm(HIDDEN)}
    write_safetensors(os.path.join(directory, "target.safetensors"), target)
    return draft, target


@unittest.skipUnless(HAVE_TORCH, "torch is not installed")
class BlockDrafterTargetDSparkTest(unittest.TestCase):
    def test_a_drafter_inside_the_target_checkpoint_drafts_over_the_target_head(self):
        from block_drafter import BlockDrafter
        with tempfile.TemporaryDirectory() as directory:
            draft, target = tiny_target_dspark_checkpoint(directory)
            d = BlockDrafter(directory, os.path.join(directory, "target.safetensors"),
                             {"embed": "model.embed_tokens.weight", "head": "lm_head.weight", "norm": "model.norm.weight"}, "cpu")
            self.assertEqual((len(d.layers), d.block, d.mask, d.layer_ids, d.window, d.kv_heads), (3, BLOCK + 1, 7, TARGET_LAYERS, None, 2))
            self.assertIsNone(d.targets)
            torch.testing.assert_close(d.embed.float(), target["model.embed_tokens.weight"].to(torch.bfloat16).float())
            taps = torch.randn(20, HIDDEN * len(TARGET_LAYERS), generator=torch.Generator().manual_seed(4)).to(torch.bfloat16)
            keys, values = d.context(taps, torch.arange(20))
            y, base = d.draft(11, 15, keys, values, 15)
            self.assertEqual(tuple(base.shape), (BLOCK + 1, VOCAB))
            drafts, confidence = d.sequential(y, base, 11, list(range(BLOCK + 1)))
            w1 = draft["markov_head.markov_w1.weight"].to(torch.bfloat16).float()
            w2 = draft["markov_head.markov_w2.weight"].to(torch.bfloat16).float()
            previous, expected = 11, []
            for slot in range(BLOCK + 1):
                previous = int(torch.argmax(base[slot] + w2 @ w1[previous]))
                expected.append(previous)
            self.assertEqual(drafts, expected)
            self.assertEqual(len(confidence), BLOCK + 1)


@unittest.skipUnless(HAVE_TORCH, "torch is not installed")
class BlockDrafterDSparkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from block_drafter import BlockDrafter
        cls.temporary = tempfile.TemporaryDirectory()
        cls.tensors, cls.targets = tiny_dspark_checkpoint(cls.temporary.name)
        cls.drafter = BlockDrafter(cls.temporary.name, os.path.join(cls.temporary.name, "model.safetensors.index.json"),
                                   {"embed": "model.embed_tokens.weight", "head": "lm_head.weight", "norm": "model.norm.weight"}, "cpu")
        generator = torch.Generator().manual_seed(9)
        cls.taps = torch.randn(23, HIDDEN * len(TARGET_LAYERS), generator=generator).to(torch.bfloat16)
        cls.positions = torch.arange(100, 123)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_layout_is_read_from_the_nested_config_and_the_checkpoint(self):
        d = self.drafter
        self.assertEqual((d.heads, d.kv_heads, d.head_dim, d.block, d.window), (HEADS, HEADS, HEAD_DIM, BLOCK, WINDOW))
        self.assertEqual(d.layer_ids, TARGET_LAYERS)
        torch.testing.assert_close(d.embed.float(), self.tensors["embed_tokens.weight"].to(torch.bfloat16).float())
        self.assertEqual(tuple(d.head.shape), (VOCAB, HIDDEN))
        self.assertEqual(d.targets.tolist(), self.targets.tolist())

    def test_drafts_are_target_ids_of_the_markov_biased_draft_argmax(self):
        d = self.drafter
        keys, values = d.context(self.taps, self.positions)
        anchor = 17
        y, base = d.draft(int(self.targets[3]), 100 + anchor, keys, values, anchor)
        self.assertEqual(tuple(base.shape), (BLOCK, DRAFT_VOCAB))
        drafts, confidence = d.sequential(y, base, int(self.targets[3]), list(range(BLOCK)))
        w1 = self.tensors["markov_head.markov_w1.weight"].to(torch.bfloat16).float()
        w2 = self.tensors["markov_head.markov_w2.weight"].to(torch.bfloat16).float()
        previous, expected = int(self.targets[3]), []
        for slot in range(BLOCK):
            draft_id = int(torch.argmax(base[slot] + w2 @ w1[previous]))
            previous = int(self.targets[draft_id])
            expected.append(previous)
        self.assertEqual(drafts, expected)
        self.assertTrue(set(drafts) <= set(self.targets.tolist()))
        self.assertEqual(len(confidence), BLOCK)
        tokens, scores = d.sequential_device(y, base, int(self.targets[3]), BLOCK)
        self.assertEqual(tokens, expected)

    def test_the_sliding_window_limits_the_context(self):
        d = self.drafter
        keys, values = d.context(self.taps, self.positions)
        anchor = 17
        clipped = [k.clone() for k in keys]
        for k in clipped:
            k[:anchor - WINDOW] = 0
        y, base = d.draft(int(self.targets[0]), 100 + anchor, keys, values, anchor)
        y_clipped, base_clipped = d.draft(int(self.targets[0]), 100 + anchor, clipped, values, anchor)
        torch.testing.assert_close(base_clipped, base)


if __name__ == "__main__":
    unittest.main()
