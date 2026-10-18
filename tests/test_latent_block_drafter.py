import json
import math
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

HIDDEN, HEADS, NOPE, ROPE, VALUE, Q_RANK, KV_RANK, INTER, VOCAB, BLOCK, GROUP, TAPS, RANK, TOP_K, WINDOW = 32, 2, 8, 4, 8, 12, 6, 24, 40, 4, 4, 2, 4, 5, 10
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


def tiny_config(**overrides):
    config = {
        "architectures": ["DFlash2DraftModel"], "hidden_size": HIDDEN, "intermediate_size": INTER, "num_hidden_layers": 2,
        "num_attention_heads": HEADS, "num_key_value_heads": HEADS, "q_lora_rank": Q_RANK, "kv_lora_rank": KV_RANK,
        "qk_nope_head_dim": NOPE, "qk_rope_head_dim": ROPE, "v_head_dim": VALUE, "mla_use_output_gate": False, "vocab_size": VOCAB,
        "rms_norm_eps": 1e-5, "fc_norm": False, "layer_types": ["sliding_attention", "full_attention"], "is_causal": False,
        "sliding_window": WINDOW,
        "rope_parameters": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 64, "rope_theta": 500.0,
                            "beta_fast": 32, "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 1.0},
        "dflash_config": {"block_size": BLOCK, "attention_mode": "mla", "conv_kernel_size": TAPS, "conv_group_size": GROUP,
                          "selector_rank": RANK, "selector_top_k": TOP_K, "target_layer_ids": TARGET_LAYERS, "mask_token_id": VOCAB - 1},
    }
    config.update(overrides)
    return config


def tiny_checkpoint(directory, seed=5):
    generator = torch.Generator().manual_seed(seed)

    def r(*shape, scale=0.4):
        return torch.randn(*shape, generator=generator) * scale

    def norm(width):
        return 1.0 + 0.1 * torch.randn(width, generator=generator)
    tensors = {"fc.weight": r(HIDDEN, HIDDEN * len(TARGET_LAYERS), scale=0.2), "hidden_norm.weight": norm(HIDDEN), "norm.weight": norm(HIDDEN),
               "candidate_selector.hidden_projection.weight": r(RANK, HIDDEN), "candidate_selector.predecessor_codebook": r(VOCAB, RANK),
               "candidate_selector.successor_codebook": r(VOCAB, RANK), "embed_tokens.weight": r(VOCAB, HIDDEN)}
    for index in range(2):
        p = f"layers.{index}."
        tensors.update({
            p + "input_layernorm.weight": norm(HIDDEN), p + "post_attention_layernorm.weight": norm(HIDDEN),
            p + "self_attn.q_a_proj.weight": r(Q_RANK, HIDDEN), p + "self_attn.q_a_layernorm.weight": norm(Q_RANK),
            p + "self_attn.q_b_proj.weight": r(HEADS * (NOPE + ROPE), Q_RANK), p + "self_attn.kv_a_proj_with_mqa.weight": r(KV_RANK + ROPE, HIDDEN),
            p + "self_attn.kv_a_layernorm.weight": norm(KV_RANK), p + "self_attn.kv_b_proj.weight": r(HEADS * (NOPE + VALUE), KV_RANK),
            p + "self_attn.o_proj.weight": r(HIDDEN, HEADS * VALUE), p + "mlp.gate_proj.weight": r(INTER, HIDDEN), p + "mlp.up_proj.weight": r(INTER, HIDDEN),
            p + "mlp.down_proj.weight": r(HIDDEN, INTER), p + "attention_conv.base_kernel": r(2, TAPS, HIDDEN), p + "mlp_conv.base_kernel": r(2, TAPS, HIDDEN),
            p + "attention_conv.kernel_projection.weight": r(2 * TAPS * (HIDDEN // GROUP), HIDDEN, scale=0.1),
            p + "mlp_conv.kernel_projection.weight": r(2 * TAPS * (HIDDEN // GROUP), HIDDEN, scale=0.1)})
    with open(os.path.join(directory, "config.json"), "w") as handle:
        json.dump(tiny_config(), handle)
    write_safetensors(os.path.join(directory, "model.safetensors"), tensors)
    target = {"embed": r(VOCAB, HIDDEN), "head": r(VOCAB, HIDDEN), "norm": norm(HIDDEN)}
    write_safetensors(os.path.join(directory, "target.safetensors"), target)
    return {name: tensor.to(torch.bfloat16).double() for name, tensor in list(tensors.items()) + list(target.items())}


def reference_rms(x, weight, eps=1e-5):
    return x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps) * weight


def reference_rope_half_split(x, positions, inverse, gain):
    permutation = list(range(0, x.shape[-1], 2)) + list(range(1, x.shape[-1], 2))
    split = x[..., permutation]
    half = x.shape[-1] // 2
    angles = positions.double()[:, None] * inverse.double()[None, :]
    shape = [angles.shape[0]] + [1] * (x.dim() - 2) + [half]
    cos, sin = (torch.cos(angles) * gain).view(shape), (torch.sin(angles) * gain).view(shape)
    first, second = split[..., :half], split[..., half:]
    rotated = torch.cat((first * cos - second * sin, first * sin + second * cos), -1)
    inverse_permutation = [0] * x.shape[-1]
    for target, source in enumerate(permutation):
        inverse_permutation[source] = target
    return rotated[..., inverse_permutation]


def reference_conv(z, base, delta):
    out = torch.zeros_like(z)
    for i in range(z.shape[0]):
        for c in range(z.shape[1]):
            for tap in range(base.shape[0]):
                if i - tap >= 0:
                    out[i, c] += (base[tap, c] + delta[i, tap, c // GROUP]) * z[i - tap, c]
    return out


def reference_block(w, drafter, anchor, position, taps_rows, context_positions, count):
    inverse, gain, scale = drafter.inverse.double(), drafter.rotary_gain, drafter.scale
    features = reference_rms(taps_rows @ w["fc.weight"].T, w["hidden_norm.weight"])
    tokens = [anchor] + [VOCAB - 1] * (BLOCK - 1)
    positions = torch.arange(position, position + BLOCK)
    residual = w["embed"][tokens]
    for index in range(2):
        p = f"layers.{index}."
        begin = 0 if index == 1 else max(0, count - (WINDOW - BLOCK))
        x = reference_rms(residual, w[p + "input_layernorm.weight"])
        delta = (x @ w[p + "attention_conv.kernel_projection.weight"].T).view(BLOCK, 2, TAPS, -1)
        xp = reference_conv(x, w[p + "attention_conv.base_kernel"][0], delta[:, 0])
        query = (reference_rms(xp @ w[p + "self_attn.q_a_proj.weight"].T, w[p + "self_attn.q_a_layernorm.weight"]) @ w[p + "self_attn.q_b_proj.weight"].T).view(BLOCK, HEADS, NOPE + ROPE)
        q_rope = reference_rope_half_split(query[..., NOPE:], positions, inverse, gain)
        context_latent = features[begin:count] @ w[p + "self_attn.kv_a_proj_with_mqa.weight"].T
        block_latent = xp @ w[p + "self_attn.kv_a_proj_with_mqa.weight"].T
        latent = torch.cat((context_latent, block_latent), 0)
        all_positions = torch.cat((context_positions[begin:count], positions), 0)
        compressed = reference_rms(latent[:, :KV_RANK], w[p + "self_attn.kv_a_layernorm.weight"])
        k_rope = reference_rope_half_split(latent[:, KV_RANK:], all_positions, inverse, gain)
        kv = (compressed @ w[p + "self_attn.kv_b_proj.weight"].T).view(-1, HEADS, NOPE + VALUE)
        keys, values = kv[..., :NOPE], kv[..., NOPE:]
        out = torch.zeros(BLOCK, HEADS, VALUE, dtype=torch.float64)
        for h in range(HEADS):
            scores = (query[:, h, :NOPE] @ keys[:, h].T + q_rope[:, h] @ k_rope.T) * scale
            out[:, h] = torch.softmax(scores, -1) @ values[:, h]
        attended = out.reshape(BLOCK, -1) @ w[p + "self_attn.o_proj.weight"].T
        residual = residual + reference_conv(attended, w[p + "attention_conv.base_kernel"][1], delta[:, 1])
        x = reference_rms(residual, w[p + "post_attention_layernorm.weight"])
        delta = (x @ w[p + "mlp_conv.kernel_projection.weight"].T).view(BLOCK, 2, TAPS, -1)
        xp = reference_conv(x, w[p + "mlp_conv.base_kernel"][0], delta[:, 0])
        mlp = (torch.nn.functional.silu(xp @ w[p + "mlp.gate_proj.weight"].T) * (xp @ w[p + "mlp.up_proj.weight"].T)) @ w[p + "mlp.down_proj.weight"].T
        residual = residual + reference_conv(mlp, w[p + "mlp_conv.base_kernel"][1], delta[:, 1])
    return reference_rms(residual, w["norm.weight"])


def reference_walk(w, y, anchor):
    logits = y[1:] @ w["head"].T
    hidden = y[1:] @ w["candidate_selector.hidden_projection.weight"].T
    previous_token, tokens = anchor, []
    for step in range(BLOCK - 1):
        candidates = torch.topk(logits[step], TOP_K).indices.tolist()
        edges = [float(logits[step, c]) + float((w["candidate_selector.predecessor_codebook"][previous_token] * hidden[step] * w["candidate_selector.successor_codebook"][c]).sum()) for c in candidates]
        previous_token = candidates[max(range(TOP_K), key=lambda k: edges[k])]
        tokens.append(previous_token)
    return tokens


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class LatentBlockDrafterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from latent_block_drafter import LatentBlockDrafter
        cls.directory = tempfile.mkdtemp(prefix="sparkpipe-latent-drafter-")
        cls.weights = tiny_checkpoint(cls.directory)
        cls.drafter = LatentBlockDrafter(cls.directory, os.path.join(cls.directory, "target.safetensors"), {"embed": "embed", "head": "head", "norm": "norm"}, "cpu")
        generator = torch.Generator().manual_seed(11)
        cls.taps = (torch.randn(24, HIDDEN * len(TARGET_LAYERS), generator=generator) * 0.5).to(torch.bfloat16)
        cls.positions = torch.arange(100, 124)

    def test_block_conv_matches_its_equation_and_row_zero_reads_no_earlier_row(self):
        from latent_block_drafter import block_conv
        generator = torch.Generator().manual_seed(3)
        z, base, delta = torch.randn(BLOCK, HIDDEN, generator=generator), torch.randn(TAPS, HIDDEN, generator=generator), torch.randn(BLOCK, TAPS, HIDDEN // GROUP, generator=generator)
        self.assertTrue(torch.allclose(block_conv(z, base, delta, GROUP).double(), reference_conv(z.double(), base.double(), delta.double()), atol=1e-5))
        shifted = z.clone()
        shifted[1:] += 1.0
        self.assertTrue(torch.equal(block_conv(z, base, delta, GROUP)[0], block_conv(shifted, base, delta, GROUP)[0]))

    def test_interleaved_rope_equals_half_split_rope_on_permuted_pairs(self):
        from latent_block_drafter import rotate_pairs
        x = torch.randn(6, HEADS, ROPE, dtype=torch.float32)
        positions = torch.arange(30, 36)
        self.assertTrue(torch.allclose(rotate_pairs(x, positions, self.drafter.inverse, self.drafter.rotary_gain).double(),
                                       reference_rope_half_split(x.double(), positions, self.drafter.inverse, self.drafter.rotary_gain), atol=1e-5))

    def test_softmax_scale_carries_the_squared_yarn_mscale_and_rotation_carries_none(self):
        self.assertAlmostEqual(self.drafter.rotary_gain, 1.0)
        self.assertAlmostEqual(self.drafter.scale, (0.1 * math.log(4.0) + 1.0) ** 2 / math.sqrt(NOPE + ROPE), places=6)

    def test_chunked_absorbed_attention_equals_one_softmax(self):
        from latent_block_drafter import latent_attention
        generator = torch.Generator().manual_seed(9)
        absorbed, rope = torch.randn(BLOCK, HEADS, KV_RANK, generator=generator), torch.randn(BLOCK, HEADS, ROPE, generator=generator)
        rows = torch.randn(17, KV_RANK + ROPE, generator=generator).to(torch.bfloat16)
        whole = latent_attention(absorbed, rope, rows, KV_RANK, 0.3, chunk=1 << 20)
        self.assertTrue(torch.allclose(latent_attention(absorbed, rope, rows, KV_RANK, 0.3, chunk=3), whole, atol=1e-5))

    def test_block_matches_the_float64_expanded_reference(self):
        keys, _ = self.drafter.context(self.taps, self.positions)
        for count in (0, 5, 20):
            y, _ = self.drafter.draft(17, int(self.positions[0]) + count, keys, None, count)
            expected = reference_block(self.weights, self.drafter, 17, int(self.positions[0]) + count, self.taps.double(), self.positions, count)
            error = (y.double() - expected).norm() / expected.norm()
            self.assertLess(float(error), 0.04, f"count {count}")

    def test_sliding_layers_see_exactly_the_window_before_the_block_end(self):
        keys, _ = self.drafter.context(self.taps, self.positions)
        count = 20
        first_visible = count - (WINDOW - BLOCK)
        baseline, _ = self.drafter.draft(5, 120, keys, None, count)
        hidden_far = [rows.clone() for rows in keys]
        hidden_far[0][:first_visible] += 3.0
        far, _ = self.drafter.draft(5, 120, hidden_far, None, count)
        self.assertTrue(torch.equal(baseline, far))
        hidden_near = [rows.clone() for rows in keys]
        hidden_near[0][first_visible] += 3.0
        near, _ = self.drafter.draft(5, 120, hidden_near, None, count)
        self.assertFalse(torch.equal(baseline, near))
        full = [rows.clone() for rows in keys]
        full[1][0] += 3.0
        changed, _ = self.drafter.draft(5, 120, full, None, count)
        self.assertFalse(torch.equal(baseline, changed))

    def test_selector_walk_takes_the_best_edge_after_each_previous_choice(self):
        keys, _ = self.drafter.context(self.taps, self.positions)
        y, _ = self.drafter.draft(17, 120, keys, None, 20)
        tokens, scores = self.drafter.sequential(y, None, 17, range(1, BLOCK))
        self.assertEqual(tokens, reference_walk(self.weights, y.double(), 17))
        self.assertTrue(all(0.0 < score <= 1.0 for score in scores))

    def test_coupled_choice_is_the_keyed_sample_of_the_lattice_row(self):
        from block_drafter import keyed_gumbel
        from latent_block_drafter import coupled_candidates
        generator = torch.Generator().manual_seed(21)
        for trial in range(20):
            row = torch.randn(TOP_K, generator=generator) * 2
            tokens = torch.randperm(VOCAB, generator=generator)[:TOP_K]
            seed, position = torch.tensor(4242 + trial), torch.tensor(500 + trial)
            sampling = (seed, torch.tensor(0.7), torch.tensor(0.0), torch.tensor(0))
            choice = int(coupled_candidates(row, tokens, sampling, position))
            self.assertEqual(choice, int(torch.argmax(row * 0.7 + keyed_gumbel(seed, position, tokens))))

    def test_refuses_checkpoints_it_cannot_run(self):
        from latent_block_drafter import DrafterContractError, refuse_unsupported
        refuse_unsupported(tiny_config())
        for overrides in ({"fc_norm": True}, {"mla_use_output_gate": True}, {"is_causal": True}, {"architectures": ["DFlashDraftModel"]},
                          {"layer_types": ["sliding_attention"]}, {"dflash_config": dict(tiny_config()["dflash_config"], attention_mode="gqa")},
                          {"dflash_config": dict(tiny_config()["dflash_config"], aux_hidden_stream="attn_res")},
                          {"dflash_config": dict(tiny_config()["dflash_config"], output_multiplier=2.0)}):
            with self.assertRaises(DrafterContractError):
                refuse_unsupported(tiny_config(**overrides))


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class LatentContextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from latent_block_drafter import LatentBlockDrafter, LatentContext
        cls.directory = tempfile.mkdtemp(prefix="sparkpipe-latent-context-")
        tiny_checkpoint(cls.directory)
        cls.drafter = LatentBlockDrafter(cls.directory, os.path.join(cls.directory, "target.safetensors"), {"embed": "embed", "head": "head", "norm": "norm"}, "cpu")
        cls.Context = LatentContext
        generator = torch.Generator().manual_seed(17)
        cls.taps = (torch.randn(40, HIDDEN * len(TARGET_LAYERS), generator=generator) * 0.5).to(torch.bfloat16)
        cls.tokens = torch.randint(0, VOCAB - 1, (40,), generator=generator).numpy().astype("uint32")

    def context(self, frames, page_positions=8):
        context = self.Context(self.drafter)
        context.page_positions = page_positions
        for first, last in frames:
            context.write(first, self.tokens[first:last], self.taps[first:last])
        return context

    def dense(self, position):
        keys, _ = self.drafter.context(self.taps[:position], torch.arange(0, position))
        y, _ = self.drafter.draft(int(self.tokens[position]), position, keys, None, position)
        return y

    def test_ring_and_pages_draft_like_dense_context_at_each_frame_end(self):
        context = self.context([])
        for first, last in ((0, 3), (3, 13), (13, 29), (29, 31), (31, 37)):
            context.write(first, self.tokens[first:last], self.taps[first:last])
            contexts, complete = context.contexts(last)
            self.assertTrue(complete)
            y = self.drafter.forward_block(int(self.tokens[last]), last, contexts)
            self.assertLess(float((y.double() - self.dense(last).double()).abs().max()), 2e-2, f"position {last}")

    def test_a_later_stream_fills_from_a_donor_that_shares_its_prefix(self):
        donor = self.context([(0, 37)])
        late = self.context([(20, 30)])
        self.assertFalse(late.complete(30))
        self.assertGreater(late.copy_prefix(donor, self.tokens, 30), 0)
        self.assertTrue(late.complete(30) and late.matches(self.tokens, 30))
        y = self.drafter.forward_block(int(self.tokens[30]), 30, late.contexts(30)[0])
        self.assertLess(float((y.double() - self.dense(30).double()).abs().max()), 2e-2)
        other = self.tokens.copy()
        other[5] = (other[5] + 1) % (VOCAB - 1)
        stranger = self.context([(20, 30)])
        self.assertEqual(stranger.copy_prefix(donor, other, 30), 5)
        self.assertFalse(stranger.complete(30))

    def test_a_committed_token_the_context_does_not_hold_is_refused(self):
        context = self.context([(0, 37)])
        committed = self.tokens.copy()
        self.assertTrue(context.matches(committed, 25))
        committed[30] = (committed[30] + 1) % (VOCAB - 1)
        self.assertFalse(context.matches(committed, 33))
        context.write(28, self.tokens[28:37], self.taps[28:37])
        self.assertTrue(context.matches(self.tokens, 33))


if __name__ == "__main__":
    unittest.main(verbosity=1)
