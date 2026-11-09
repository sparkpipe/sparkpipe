#!/usr/bin/env python3
"""The DSpine drafter (tools/spec_offline/dspine_drafter.py) on a tiny random checkpoint this test writes.

With the feature writes, the predecessor gate and the gate bias zero, the code bridge leaves every row unchanged and the
draft equals a float64 DFlash backbone with causal block attention and per-row sliding windows; with them nonzero it
equals a float64 DSpine forward and differs from the backbone. The serial-table chain (greedy tables from the LM head)
equals a brute-force left-to-right decode that redoes the last write with each row's exact sender, the FP32 sampled
table equals a float64 recomputation, and the draftd graph's segments (pre_attention, attend_window, post_attention,
walk on the graph's own last norm) draft the same tokens as the offline path. Checkpoint detection picks DSpineDrafter
and refuses a DSpine config without causal block attention.
"""
import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "spec_offline"))
try:
    import numpy as np
    import torch
except ImportError:
    print("SKIP test_dspine_drafter: torch is not installed")
    sys.exit(0)
try:
    import block_drafter
    import dspine_drafter as dspine
    import selector_block_drafter
except ImportError as error:
    print(f"SKIP test_dspine_drafter: {error}")
    sys.exit(0)

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, FFN, LAYERS, VOCAB = 64, 4, 2, 16, 96, 2, 200
BLOCK, CODE, RANK, TOP_K, WINDOW, THETA, EPS = 8, 32, 16, 4, 12, 10000.0, 1e-6
TAP_LAYERS = [0, 2]
MASK = VOCAB - 1
NAMES = {"embed": "embed.weight", "head": "head.weight", "norm": "norm.weight"}


def config(**overrides):
    dflash = {"block_size": BLOCK, "mask_token_id": MASK, "target_layer_ids": TAP_LAYERS, "draft_block_causal": True,
              "selected_prefix_code_bridge": {"code_size": CODE, "rank": RANK, "eps": EPS, "kappa": 10.0}, "selected_prefix_decode_mode": "serial",
              "selected_prefix_feature_layers": list(range(1, LAYERS + 1)), "selected_prefix_feature_write_mode": "independent",
              "selected_prefix_feature_writes": True, "selected_prefix_last_write_refine": {"head_topk": TOP_K, "kind": "exact_sender", "version": 1, "weight": 1.0},
              "selected_prefix_terminal": False}
    dflash.update(overrides)
    return {"architectures": ["DFlashDraftModel"], "block_size": BLOCK, "code_bridge_second_pass": {"first_exact": LAYERS - 1, "head_topk": TOP_K, "mode": "serial_table"},
            "dflash_config": dflash, "head_dim": HEAD_DIM, "hidden_size": HIDDEN, "intermediate_size": FFN, "is_causal": True,
            "layer_types": ["sliding_attention"] * LAYERS, "num_attention_heads": HEADS, "num_hidden_layers": LAYERS, "num_key_value_heads": KV_HEADS,
            "rms_norm_eps": EPS, "rope_parameters": {"rope_theta": THETA, "rope_type": "default"}, "rope_theta": THETA, "sliding_window": WINDOW,
            "use_sliding_window": True, "vocab_size": VOCAB}


def write_safetensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, tensor in tensors.items():
        if tensor.dtype == torch.bfloat16:
            dtype, blob = "BF16", tensor.contiguous().view(torch.int16).numpy().tobytes()
        elif tensor.dtype == torch.int64:
            dtype, blob = "I64", tensor.numpy().tobytes()
        else:
            dtype, blob = "F32", tensor.float().contiguous().numpy().tobytes()
        header[name] = {"dtype": dtype, "shape": list(tensor.shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    encoded = json.dumps(header).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def random_weights(generator, bridge_on):
    def normal(*shape, scale=1.0):
        return (torch.randn(*shape, generator=generator) * scale).to(torch.bfloat16)

    def gain(size):
        return (1.0 + 0.1 * torch.randn(size, generator=generator)).to(torch.bfloat16)
    tensors = {"fc.weight": normal(HIDDEN, HIDDEN * len(TAP_LAYERS), scale=HIDDEN ** -0.5), "hidden_norm.weight": gain(HIDDEN), "norm.weight": gain(HIDDEN),
               "draft_block_causal_version": torch.tensor(1), "last_write_refine_version": torch.tensor(1)}
    for index in range(LAYERS):
        p = f"layers.{index}."
        tensors.update({p + "input_layernorm.weight": gain(HIDDEN), p + "post_attention_layernorm.weight": gain(HIDDEN),
                        p + "self_attn.q_proj.weight": normal(HEADS * HEAD_DIM, HIDDEN, scale=HIDDEN ** -0.5),
                        p + "self_attn.k_proj.weight": normal(KV_HEADS * HEAD_DIM, HIDDEN, scale=HIDDEN ** -0.5),
                        p + "self_attn.v_proj.weight": normal(KV_HEADS * HEAD_DIM, HIDDEN, scale=HIDDEN ** -0.5),
                        p + "self_attn.o_proj.weight": normal(HIDDEN, HEADS * HEAD_DIM, scale=(HEADS * HEAD_DIM) ** -0.5),
                        p + "self_attn.q_norm.weight": gain(HEAD_DIM), p + "self_attn.k_norm.weight": gain(HEAD_DIM),
                        p + "mlp.gate_proj.weight": normal(FFN, HIDDEN, scale=HIDDEN ** -0.5), p + "mlp.up_proj.weight": normal(FFN, HIDDEN, scale=HIDDEN ** -0.5),
                        p + "mlp.down_proj.weight": normal(HIDDEN, FFN, scale=FFN ** -0.5)})
    b = "selected_prefix_bridge."
    write_scale = 1.5 * RANK ** -0.5 if bridge_on else 0.0
    tensors.update({b + "codebook": normal(VOCAB, CODE, scale=CODE ** -0.5), b + "read_weight": normal(CODE, HIDDEN, scale=HIDDEN ** -0.5),
                    b + "early_read_weights": normal(LAYERS - 1, CODE, HIDDEN, scale=HIDDEN ** -0.5),
                    b + "feature_write_weights": normal(LAYERS, HIDDEN, RANK, scale=write_scale),
                    b + "gate.weight": normal(RANK, CODE, scale=CODE ** -0.5), b + "gate.bias": normal(RANK, scale=0.3 if bridge_on else 0.0),
                    b + "predecessor_gate.weight": normal(RANK, CODE, scale=CODE ** -0.5 if bridge_on else 0.0), b + "prev.weight": normal(RANK, CODE, scale=CODE ** -0.5),
                    b + "out.weight": torch.zeros(CODE, RANK, dtype=torch.bfloat16), b + "independent_write_version": torch.tensor(1),
                    b + "terminal_disabled_version": torch.tensor(1)})
    return tensors


def write_checkpoint(directory, seed, bridge_on=True, overrides=None):
    directory.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    tensors = random_weights(generator, bridge_on)
    (directory / "config.json").write_text(json.dumps(config(**(overrides or {}))))
    write_safetensors(directory / "model.safetensors", tensors)
    target = {"embed.weight": (torch.randn(VOCAB, HIDDEN, generator=generator)).to(torch.bfloat16),
              "head.weight": (torch.randn(VOCAB, HIDDEN, generator=generator) * 3 * HIDDEN ** -0.5).to(torch.bfloat16),
              "norm.weight": 1.0 + 0.1 * torch.randn(HIDDEN, generator=generator)}
    write_safetensors(directory / "target.safetensors", target)
    taps = torch.randn(40, HIDDEN * len(TAP_LAYERS), generator=generator).to(torch.bfloat16)
    return {name: t.double() for name, t in tensors.items() if t.is_floating_point()}, {name: t.double() for name, t in target.items()}, taps


def norm64(x, weight):
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * weight


def rope64(x, positions):
    inverse = 1.0 / THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM)
    angles = positions.double()[:, None] * inverse[None, :]
    cos, sin = torch.cat((angles.cos(), angles.cos()), -1)[:, None, :], torch.cat((angles.sin(), angles.sin()), -1)[:, None, :]
    half = HEAD_DIM // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), -1) * sin


def reference(w, target, taps, count, anchor, bridge_on):
    ctx = norm64(taps[:count].double() @ w["fc.weight"].t(), w["hidden_norm.weight"])
    ctx_positions = torch.arange(count)
    positions = torch.arange(count, count + BLOCK)
    tokens = torch.full((BLOCK,), MASK)
    tokens[0] = anchor
    h = target["embed.weight"][tokens]
    b = "selected_prefix_bridge."
    for index in range(LAYERS):
        p = f"layers.{index}."
        x = norm64(h, w[p + "input_layernorm.weight"])
        q = rope64(norm64((x @ w[p + "self_attn.q_proj.weight"].t()).view(BLOCK, HEADS, HEAD_DIM), w[p + "self_attn.q_norm.weight"]), positions)
        kb = rope64(norm64((x @ w[p + "self_attn.k_proj.weight"].t()).view(BLOCK, KV_HEADS, HEAD_DIM), w[p + "self_attn.k_norm.weight"]), positions)
        vb = (x @ w[p + "self_attn.v_proj.weight"].t()).view(BLOCK, KV_HEADS, HEAD_DIM)
        kc = rope64(norm64((ctx @ w[p + "self_attn.k_proj.weight"].t()).view(count, KV_HEADS, HEAD_DIM), w[p + "self_attn.k_norm.weight"]), ctx_positions)
        vc = (ctx @ w[p + "self_attn.v_proj.weight"].t()).view(count, KV_HEADS, HEAD_DIM)
        out = torch.zeros(BLOCK, HEADS, HEAD_DIM, dtype=torch.float64)
        for row in range(BLOCK):
            seen = [j for j in range(count) if j >= count + row - (WINDOW - 1)]
            keys = torch.cat((kc[seen], kb[:row + 1]))
            values = torch.cat((vc[seen], vb[:row + 1]))
            for head in range(HEADS):
                group = head // (HEADS // KV_HEADS)
                weights = torch.softmax(keys[:, group] @ q[row, head] / math.sqrt(HEAD_DIM), 0)
                out[row, head] = weights @ values[:, group]
        h = h + out.reshape(BLOCK, -1) @ w[p + "self_attn.o_proj.weight"].t()
        x = norm64(h, w[p + "post_attention_layernorm.weight"])
        h = h + (torch.nn.functional.silu(x @ w[p + "mlp.gate_proj.weight"].t()) * (x @ w[p + "mlp.up_proj.weight"].t())) @ w[p + "mlp.down_proj.weight"].t()
        if not bridge_on:
            continue
        scale = h.pow(2).mean(-1).sqrt()
        read = w[b + "read_weight"] if index == LAYERS - 1 else w[b + "early_read_weights"][index]
        projected = (h / (scale[:, None] + EPS)) @ read.t()
        codes = math.sqrt(CODE) * projected / (projected.norm(dim=-1, keepdim=True) + EPS)
        codes[0] = math.sqrt(CODE) * w[b + "codebook"][anchor]
        query, sender, value = codes @ w[b + "gate.weight"].t(), codes @ w[b + "predecessor_gate.weight"].t(), codes @ w[b + "prev.weight"].t()
        message = torch.zeros(BLOCK, RANK, dtype=torch.float64)
        message[1:] = torch.sigmoid(query[1:] + sender[:-1] + w[b + "gate.bias"]) * value[:-1]
        update = scale[:, None] * (message @ w[b + "feature_write_weights"][index].t())
        update[0] = 0.0
        h = h + update
    return norm64(h, w["norm.weight"])


def relative(a, b):
    return float((a.double() - b.double()).abs().max() / b.double().abs().max())


def drafted(directory, taps, count, anchor):
    drafter = dspine.DSpineDrafter(directory, directory / "target.safetensors", NAMES, "cpu")
    keys, values = drafter.context(taps[:count], torch.arange(count))
    y, base = drafter.draft(anchor, count, keys, values, count)
    return drafter, keys, values, y, base


class InjectionIdentity(unittest.TestCase):
    def test_zero_write_gate_and_bias_reduce_to_the_dflash_backbone(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            weights, target, taps = write_checkpoint(directory, 11, bridge_on=False)
            drafter, _, _, y, _ = drafted(directory, taps, 20, 17)
            raw = torch.randn(BLOCK, HIDDEN).to(torch.bfloat16)
            for index in range(LAYERS):
                self.assertTrue(torch.equal(drafter.bridge(index, raw), raw))
            backbone = reference(weights, target, taps, 20, 17, bridge_on=False)
            self.assertLess(relative(y, backbone), 0.03)

    def test_nonzero_bridge_matches_float64_dspine_and_leaves_the_backbone(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            weights, target, taps = write_checkpoint(directory, 12, bridge_on=True)
            _, _, _, y, _ = drafted(directory, taps, 20, 17)
            self.assertLess(relative(y, reference(weights, target, taps, 20, 17, bridge_on=True)), 0.03)
            self.assertGreater(relative(y, reference(weights, target, taps, 20, 17, bridge_on=False)), 0.2)


def brute_chain(raw, scale, query, anchor, candidates, head, norm, codebook, sender_weight, value_weight, gate_bias, write):
    tokens, confidence = [], []
    sender = anchor
    for row in range(candidates.shape[0]):
        code = math.sqrt(CODE) * codebook[sender].double()
        message = torch.sigmoid(query[row].double() + sender_weight.double() @ code + gate_bias.double()) * (value_weight.double() @ code)
        state = raw[row].double() + scale[row].double() * (write.double() @ message)
        scores = torch.stack([(state * norm.double()) @ head[c].double() for c in candidates[row].tolist()])
        best = int(torch.argmax(scores))
        confidence.append(float(1.0 / torch.exp((scores - scores[best]) / max(float(scale[row]), 1e-6)).sum()))
        tokens.append(int(candidates[row, best]))
        sender = tokens[-1]
    return tokens, confidence


class SerialTable(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(5)
        rows = BLOCK - 1
        self.raw = (torch.randn(rows, HIDDEN, generator=g) * 2).to(torch.bfloat16)
        self.scale = torch.rand(rows, generator=g) + 0.5
        self.query = torch.randn(rows, RANK, generator=g).to(torch.bfloat16)
        self.head = (torch.randn(VOCAB, HIDDEN, generator=g) * 0.5).to(torch.bfloat16)
        self.norm = (1.0 + 0.1 * torch.randn(HIDDEN, generator=g)).to(torch.bfloat16)
        self.codebook = (torch.randn(VOCAB, CODE, generator=g) * CODE ** -0.5).to(torch.bfloat16)
        self.sender_weight = (torch.randn(RANK, CODE, generator=g) * CODE ** -0.5).to(torch.bfloat16)
        self.value_weight = (torch.randn(RANK, CODE, generator=g) * CODE ** -0.5).to(torch.bfloat16)
        self.gate_bias = (torch.randn(RANK, generator=g) * 0.3).to(torch.bfloat16)
        self.write = (torch.randn(HIDDEN, RANK, generator=g) * 2 * RANK ** -0.5).to(torch.bfloat16)
        self.candidates = torch.stack([torch.sort(torch.randperm(VOCAB, generator=g)[:TOP_K]).values for _ in range(rows)])
        self.anchor = 7

    def test_greedy_chain_equals_brute_force_decode(self):
        tg, tsv = dspine.serial_tables(self.head, self.norm, self.write, torch.cat((self.sender_weight, self.value_weight)), self.codebook)
        senders = dspine.senders_of(torch.tensor(self.anchor), self.candidates)
        table, confidence = dspine.greedy_table(self.raw, self.scale, self.query, senders, self.candidates, self.head, self.norm, tg, tsv, self.gate_bias)
        tokens, scores = dspine.serial_chain(table, self.candidates, confidence)
        expected, expected_confidence = brute_chain(self.raw, self.scale, self.query, self.anchor, self.candidates, self.head, self.norm, self.codebook,
                                                    self.sender_weight, self.value_weight, self.gate_bias, self.write)
        self.assertEqual(tokens.tolist(), expected)
        self.assertTrue(np.allclose(scores.tolist(), expected_confidence, rtol=0.05, atol=1e-3))

    def test_refined_table_equals_float64_recomputation(self):
        senders = dspine.senders_of(torch.tensor(self.anchor), self.candidates)
        codes = math.sqrt(CODE) * self.codebook[senders].float()
        table = dspine.refined_table(self.raw, self.scale, self.query, codes, self.candidates, self.sender_weight.float(), self.value_weight.float(),
                                     self.gate_bias.float(), self.write.float(), self.norm.float(), self.head, EPS)
        for row in range(self.candidates.shape[0]):
            for lane in range(TOP_K):
                code = codes[row, lane].double()
                message = torch.sigmoid(self.query[row].double() + self.sender_weight.double() @ code + self.gate_bias.double()) * (self.value_weight.double() @ code)
                state = self.raw[row].double() + self.scale[row].double() * (self.write.double() @ message)
                expected = norm64(state, self.norm.double()) @ self.head[self.candidates[row]].double().t()
                self.assertTrue(torch.allclose(table[row, lane].double(), expected, rtol=1e-4, atol=1e-4))


class DraftPaths(unittest.TestCase):
    def test_graph_segments_draft_the_offline_tokens(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            _, _, taps = write_checkpoint(directory, 13, bridge_on=True)
            count, anchor = WINDOW - BLOCK, 23
            drafter, keys, values, y, base = drafted(directory, taps, count, anchor)
            offline, offline_scores = drafter.sequential(y, base, anchor, range(1, BLOCK))
            self.assertEqual(len(offline), BLOCK - 1)
            ring_keys = [torch.zeros((KV_HEADS, count + BLOCK, HEAD_DIM), dtype=torch.bfloat16) for _ in drafter.layers]
            ring_values = [torch.zeros_like(k) for k in ring_keys]
            for index in range(LAYERS):
                ring_keys[index][:, :count] = keys[index].transpose(0, 1)
                ring_values[index][:, :count] = values[index].transpose(0, 1)
            slots = torch.arange(BLOCK)
            positions = count + slots
            with torch.no_grad():
                h = drafter.embed[torch.where(slots == 0, torch.tensor(anchor), torch.tensor(MASK))].float()
                for index, layer in enumerate(drafter.layers):
                    q, kb, vb = drafter.pre_attention(layer, h, positions)
                    h = drafter.post_attention(layer, h, drafter.attend_window(index, q, kb, vb, ring_keys, ring_values, 0, count, None))
                graph_y = block_drafter.rms(h, drafter.norm, drafter.eps).to(torch.bfloat16)
                tokens, scores = drafter.walk(graph_y, (graph_y @ drafter.head.t()).float(), torch.tensor(anchor), BLOCK)
            self.assertEqual(tokens.tolist(), offline)
            self.assertTrue(np.allclose(scores.tolist(), offline_scores, rtol=1e-5))


class Detection(unittest.TestCase):
    def test_open_drafter_picks_dspine_and_refuses_a_non_causal_block(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / "dspine"
            write_checkpoint(directory, 14)
            self.assertTrue(dspine.is_dspine_checkpoint(directory))
            self.assertFalse(selector_block_drafter.is_selector_checkpoint(directory))
            self.assertIsInstance(block_drafter.open_drafter(directory, directory / "target.safetensors", NAMES, "cpu"), dspine.DSpineDrafter)
            plain = Path(folder) / "dflash"
            plain.mkdir()
            (plain / "config.json").write_text(json.dumps({"architectures": ["DFlashDraftModel"], "dflash_config": {"block_size": 16, "target_layer_ids": [1]}}))
            self.assertFalse(dspine.is_dspine_checkpoint(plain))
            with self.assertRaises(dspine.DrafterContractError):
                dspine.refuse_unsupported(config(draft_block_causal=False))


if __name__ == "__main__":
    unittest.main()
