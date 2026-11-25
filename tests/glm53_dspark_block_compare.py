#!/usr/bin/env python3
"""PyTorch reference and parity comparator for GLM-5.3 RedHat AI DSpark drafter module.

    glm53_dspark_block_compare.py synthetic --first F --rows N --out DIR [--seed S]
    glm53_dspark_block_compare.py compare --checkpoint DIR --target SHARD --out DIR [--precise]

Generates synthetic taps and tokens (including sliding-window wrap-around cases beyond 2048),
then compares build/modules/glm53_dspark_draft_stage.so outputs dumped by build/glm53_dspark_block_probe
against an eager PyTorch reference matching RedHatAI/GLM-5.3-speculator.dspark (DSparkDraftModel).
"""
import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "spec_offline"))
from block_drafter import keyed_gumbel  # noqa: E402

HIDDEN = 6144
TAPS = 5
LAYERS = 3
HEADS = 64
KV_HEADS = 64
HEAD_DIM = 64
MLP = 12288
BLOCK = 8
SLIDING = 2048
WINDOW = SLIDING - BLOCK
COLUMNS = WINDOW + BLOCK
VOCAB = 154880
MARKOV_RANK = 256
MARKOV_CANDIDATES = 2048
SAMPLED_CANDIDATES = 32
FRAME_ROWS = 512
RULES = ((0, 0.0, 0.0, 0), (12345, 1.0, 0.95, 0), (777, np.float32(1.0) / np.float32(0.6), 1.0, 20))


def bf16_file(path, shape):
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16).astype(np.int16)).view(torch.bfloat16).view(*shape)


def load_safetensor(path, name, device):
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    entry = header[name]
    begin, end = entry["data_offsets"]
    base = 8 + size
    raw = np.memmap(path, dtype=np.uint16, mode="r", offset=base + begin, shape=((end - begin) // 2,))
    return torch.from_numpy(np.array(raw)).view(torch.bfloat16).reshape(entry["shape"]).to(device)


def qwen3_rms(x, weight, eps):
    x32 = x.float()
    normed = (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)).to(torch.bfloat16)
    return (normed.float() * weight.float()).to(torch.bfloat16)


def rope_neox(x, positions, inv):
    angles = positions.float()[:, None] * inv[None, :]
    cos = torch.cat((angles.cos(), angles.cos()), -1)
    sin = torch.cat((angles.sin(), angles.sin()), -1)
    half = x.shape[-1] // 2
    x32 = x.float()
    rot = torch.cat((-x32[..., half:], x32[..., :half]), -1)
    return (x32 * cos[:, None, :] + rot * sin[:, None, :]).to(torch.bfloat16)


def coupled_candidates(values, tokens, sampling, position):
    seed, inv_temp, nucleus, top_k = sampling
    scaled = values * inv_temp
    probs = torch.softmax(scaled, -1)
    before = torch.cumsum(probs, 0) - probs
    rank = torch.arange(values.shape[0], device=values.device)
    keep = ((nucleus <= 0.0) | (before < nucleus)) & ((top_k <= 0) | (rank < top_k))
    keep = keep | (rank == 0)
    score = torch.where(keep, scaled + keyed_gumbel(seed, position, tokens), torch.full_like(scaled, float("-inf")))
    return torch.argmax(score)


class Glm53DsparkReference:
    def __init__(self, checkpoint_dir, target_shard, device="cuda"):
        self.device = torch.device(device)
        cfg = json.loads((Path(checkpoint_dir) / "config.json").read_text())
        self.config = cfg
        tc = cfg["transformer_layer_config"]
        assert cfg["architectures"] == ["DSparkDraftModel"]
        assert cfg["block_size"] == BLOCK
        assert cfg["sample_from_anchor"] is True
        assert cfg["sliding_window_non_causal"] is False
        assert cfg["markov_head_type"] == "vanilla"
        assert cfg["markov_rank"] == MARKOV_RANK
        assert tc["hidden_size"] == HIDDEN
        assert tc["num_hidden_layers"] == LAYERS
        assert tc["num_attention_heads"] == HEADS
        assert tc["num_key_value_heads"] == KV_HEADS
        assert tc["head_dim"] == HEAD_DIM
        assert tc["intermediate_size"] == MLP
        assert tc["sliding_window"] == SLIDING
        self.mask = int(cfg["mask_token_id"])
        self.eps = float(tc["rms_norm_eps"])
        theta = float(tc["rope_parameters"]["rope_theta"])
        self.inv = (1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))).to(self.device)
        self.scale = HEAD_DIM ** -0.5

        st = Path(checkpoint_dir) / "model.safetensors"
        self.fc = load_safetensor(st, "fc.weight", self.device)
        self.hidden_norm = load_safetensor(st, "hidden_norm.weight", self.device)
        self.norm = load_safetensor(st, "norm.weight", self.device)
        self.markov_w1 = load_safetensor(st, "markov_head.markov_w1.weight", self.device)
        self.markov_w2 = load_safetensor(st, "markov_head.markov_w2.weight", self.device)
        self.conf_weight = load_safetensor(st, "confidence_head.proj.weight", self.device).float()
        self.conf_bias = load_safetensor(st, "confidence_head.proj.bias", self.device).float()
        self.layers = []
        for idx in range(LAYERS):
            p = f"layers.{idx}."
            self.layers.append({name: load_safetensor(st, p + name + ".weight", self.device) for name in (
                "input_layernorm", "post_attention_layernorm", "self_attn.q_proj", "self_attn.k_proj",
                "self_attn.v_proj", "self_attn.o_proj", "self_attn.q_norm", "self_attn.k_norm",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")})
        self.embed = load_safetensor(target_shard, "model.embed_tokens.weight", self.device)
        self.head = load_safetensor(target_shard, "lm_head.weight", self.device)

    def ingest_chunk(self, taps, positions):
        features = qwen3_rms(taps @ self.fc.t(), self.hidden_norm, self.eps)
        keys, values = [], []
        for layer in self.layers:
            k = (features @ layer["self_attn.k_proj"].t()).view(-1, KV_HEADS, HEAD_DIM)
            k = rope_neox(qwen3_rms(k, layer["self_attn.k_norm"], self.eps), positions, self.inv)
            v = (features @ layer["self_attn.v_proj"].t()).view(-1, KV_HEADS, HEAD_DIM)
            keys.append(k)
            values.append(v)
        return keys, values

    def layer_forward(self, layer_idx, h, block_positions, ctx_k, ctx_v, valid_mask):
        layer = self.layers[layer_idx]
        x = qwen3_rms(h, layer["input_layernorm"], self.eps)
        q = (x @ layer["self_attn.q_proj"].t()).view(BLOCK, HEADS, HEAD_DIM)
        q = rope_neox(qwen3_rms(q, layer["self_attn.q_norm"], self.eps), block_positions, self.inv)
        kb = (x @ layer["self_attn.k_proj"].t()).view(BLOCK, KV_HEADS, HEAD_DIM)
        kb = rope_neox(qwen3_rms(kb, layer["self_attn.k_norm"], self.eps), block_positions, self.inv)
        vb = (x @ layer["self_attn.v_proj"].t()).view(BLOCK, KV_HEADS, HEAD_DIM)

        k_all = torch.cat((ctx_k, kb.transpose(0, 1)), dim=1).float()
        v_all = torch.cat((ctx_v, vb.transpose(0, 1)), dim=1).float()
        scores = torch.matmul(q.transpose(0, 1).float(), k_all.transpose(1, 2)) * self.scale

        ctx_len = ctx_k.shape[1]
        mask = torch.ones((BLOCK, ctx_len + BLOCK), dtype=torch.bool, device=self.device)
        if valid_mask is not None:
            mask[:, :ctx_len] = valid_mask[None, :]
        causal_block = torch.tril(torch.ones((BLOCK, BLOCK), dtype=torch.bool, device=self.device))
        mask[:, ctx_len:] = causal_block
        scores = scores.masked_fill(~mask[None, :, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v_all).transpose(0, 1).reshape(BLOCK, -1).to(torch.bfloat16)

        h = (h.float() + (attended @ layer["self_attn.o_proj"].t()).float()).to(torch.bfloat16)
        x = qwen3_rms(h, layer["post_attention_layernorm"], self.eps)
        gate = (x @ layer["mlp.gate_proj"].t()).float()
        up = (x @ layer["mlp.up_proj"].t()).float()
        act = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        h = (h.float() + (act @ layer["mlp.down_proj"].t()).float()).to(torch.bfloat16)
        return h

    def propose(self, y, anchor, position, sampling=None):
        base = (y @ self.head.t()).float()
        unary, candidates = torch.topk(base, MARKOV_CANDIDATES, dim=-1)
        edges = torch.index_select(self.markov_w2, 0, candidates.view(-1)).view(BLOCK, MARKOV_CANDIDATES, -1).float()
        hidden = y.float()
        prev = torch.tensor(int(anchor), dtype=torch.long, device=self.device)
        tokens, scores = [], []
        for step in range(BLOCK):
            vec = self.markov_w1[prev].float()
            row = unary[step] + edges[step] @ vec
            if sampling is not None:
                top_vals, top_idx = torch.topk(row, SAMPLED_CANDIDATES)
                picked = coupled_candidates(top_vals, candidates[step, top_idx], sampling, torch.tensor(position + step, device=self.device))
                slot = top_idx[picked]
            else:
                slot = torch.argmax(row)
            choice = candidates[step, slot]
            feat = torch.cat((hidden[step], vec))
            conf = torch.sigmoid(self.conf_weight @ feat + self.conf_bias).squeeze()
            tokens.append(int(choice.item()))
            scores.append(float(conf.item()))
            prev = choice
        return tokens, scores, base


def relative(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def write_case(out, start, taps, tokens):
    out.mkdir(parents=True, exist_ok=True)
    np.array([start], dtype=np.uint64).tofile(out / "first.bin")
    taps.tofile(out / "taps.bin")
    tokens.tofile(out / "tokens.bin")


def synthetic(options):
    gen = torch.Generator().manual_seed(options.seed)
    taps = (torch.randn(options.rows - options.first, TAPS * HIDDEN, generator=gen) * 1.5).to(torch.bfloat16)
    tokens = torch.randint(0, 150000, (options.rows + 1,), generator=gen).numpy().astype(np.uint32)
    write_case(options.out, options.first, taps.view(torch.int16).numpy().view(np.uint16), tokens)
    print(f"wrote {options.rows - options.first} random tap rows from position {options.first} to {options.out}")
    return 0


def compare(options):
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not options.precise
    out = options.out
    tokens = np.fromfile(out / "tokens.bin", dtype=np.uint32)
    rows = tokens.shape[0] - 1
    start = int(np.fromfile(out / "first.bin", dtype=np.uint64)[0])
    taps = bf16_file(out / "taps.bin", (rows - start, TAPS * HIDDEN))
    ref = Glm53DsparkReference(options.checkpoint, options.target, "cuda")
    failures = []

    with torch.inference_mode():
        ref_keys = [torch.zeros((KV_HEADS, COLUMNS, HEAD_DIM), dtype=torch.bfloat16, device=ref.device) for _ in range(LAYERS)]
        ref_vals = [torch.zeros((KV_HEADS, COLUMNS, HEAD_DIM), dtype=torch.bfloat16, device=ref.device) for _ in range(LAYERS)]
        ref_pos = np.full(WINDOW, -1, dtype=np.int64)
        for first in range(start, rows, FRAME_ROWS):
            count = min(FRAME_ROWS, rows - first)
            skip = max(0, count - WINDOW)
            f0 = first + skip
            c0 = count - skip
            pos = np.arange(f0, f0 + c0)
            slots = pos % WINDOW
            t_chunk = taps[first - start + skip:first - start + count].to(ref.device)
            p_chunk = torch.from_numpy(pos).to(ref.device)
            ks, vs = ref.ingest_chunk(t_chunk, p_chunk)
            slot_idx = torch.from_numpy(slots).to(ref.device)
            for layer in range(LAYERS):
                ref_keys[layer].index_copy_(1, slot_idx, ks[layer].transpose(0, 1))
                ref_vals[layer].index_copy_(1, slot_idx, vs[layer].transpose(0, 1))
            ref_pos[ref_pos >= f0] = -1
            ref_pos[slots] = pos

        mod_pos = np.fromfile(out / "cache_positions.bin", dtype=np.int64)
        held = mod_pos >= 0
        if not np.array_equal(held, ref_pos >= 0) or not np.array_equal(mod_pos[held], ref_pos[held]):
            failures.append("cache positions differ")

        held_slots = torch.from_numpy(np.flatnonzero(held)).to(ref.device)
        mod_keys, mod_vals = [], []
        for layer in range(LAYERS):
            mk = bf16_file(out / f"keys{layer}.bin", (KV_HEADS, COLUMNS, HEAD_DIM)).to(ref.device)
            mv = bf16_file(out / f"values{layer}.bin", (KV_HEADS, COLUMNS, HEAD_DIM)).to(ref.device)
            k_gap = relative(mk[:, held_slots], ref_keys[layer][:, held_slots])
            v_gap = relative(mv[:, held_slots], ref_vals[layer][:, held_slots])
            k_eq = (mk[:, held_slots] == ref_keys[layer][:, held_slots]).float().mean().item()
            v_eq = (mv[:, held_slots] == ref_vals[layer][:, held_slots]).float().mean().item()
            print(f"layer {layer} cache: keys eq {100 * k_eq:.2f}% (rel {k_gap:.5f}), values eq {100 * v_eq:.2f}% (rel {v_gap:.5f})")
            if max(k_gap, v_gap) > 2e-3:
                failures.append(f"layer {layer} cache rel diff too large: k={k_gap:.5f} v={v_gap:.5f}")
            mod_keys.append(mk)
            mod_vals.append(mv)

        win_first = max(start, rows - WINDOW)
        win_positions = np.arange(win_first, rows)
        held_mask = ref_pos[win_positions % WINDOW] == win_positions
        if not held_mask.all():
            win_positions = win_positions[int(np.flatnonzero(~held_mask)[-1]) + 1:]
        if len(win_positions) == WINDOW:
            c_start, c_end, valid = 0, WINDOW, None
        else:
            f_slot = int(win_positions[0]) % WINDOW if len(win_positions) else 0
            e_slot = f_slot + len(win_positions)
            if e_slot <= WINDOW:
                c_start, c_end, valid = f_slot, e_slot, None
            else:
                c_start, c_end = 0, WINDOW
                valid = torch.zeros(WINDOW, dtype=torch.bool, device=ref.device)
                valid[torch.from_numpy(win_positions % WINDOW).to(ref.device)] = True

        anchor = int(tokens[rows])
        block_tokens = torch.full((BLOCK,), ref.mask, dtype=torch.long, device=ref.device)
        block_tokens[0] = anchor
        block_pos = torch.arange(rows, rows + BLOCK, device=ref.device)
        h_ref = ref.embed[block_tokens]
        h_step = h_ref.clone()
        for layer in range(LAYERS):
            ctx_k = mod_keys[layer][:, c_start:c_end]
            ctx_v = mod_vals[layer][:, c_start:c_end]
            h_step = ref.layer_forward(layer, h_step, block_pos, ctx_k, ctx_v, valid)
            h_ref = ref.layer_forward(layer, h_ref, block_pos, ref_keys[layer][:, c_start:c_end], ref_vals[layer][:, c_start:c_end], valid)
            mod_h = bf16_file(out / f"hidden{layer + 1}.bin", (BLOCK, HIDDEN)).to(ref.device)
            gap_step = relative(mod_h, h_step)
            gap_e2e = relative(mod_h, h_ref)
            print(f"layer {layer}: step-matched rel {gap_step:.5f}, end-to-end rel {gap_e2e:.5f}")
            if gap_step > 5e-3:
                failures.append(f"layer {layer} step-matched rel diff {gap_step:.5f} > 5e-3")
            h_step = mod_h

        y_ref = qwen3_rms(h_ref, ref.norm, ref.eps)
        mod_y = bf16_file(out / "y.bin", (BLOCK, HIDDEN)).to(ref.device)
        y_gap = relative(mod_y, y_ref)
        base_mod = (mod_y @ ref.head.t()).float()
        base_ref = (y_ref @ ref.head.t()).float()
        logit_rel = relative(base_mod, base_ref)
        top_mod = base_mod.argmax(-1).tolist()
        top_ref = base_ref.argmax(-1).tolist()
        top_matches = sum(int(a == b) for a, b in zip(top_mod, top_ref))
        print(f"final y rel {y_gap:.5f}, base logits rel {logit_rel:.5f}, base top-1 match {top_matches}/{BLOCK}: mod={top_mod} ref={top_ref}")
        if top_matches != BLOCK:
            failures.append(f"base top-1 matched {top_matches}/{BLOCK}")

        drafted = np.fromfile(out / "drafts.bin", dtype=np.uint32).reshape(len(RULES), 2 * BLOCK + 1)
        for idx, rule in enumerate(RULES):
            sampling = None
            if rule[1] > 0.0:
                seed = rule[0] - (1 << 64) if rule[0] >= (1 << 63) else rule[0]
                sampling = (
                    torch.tensor(seed, device=ref.device),
                    torch.tensor(np.float32(rule[1]), device=ref.device),
                    torch.tensor(np.float32(rule[2] if rule[2] < 1.0 else 0.0), device=ref.device),
                    torch.tensor(rule[3], device=ref.device),
                )
            ref_toks_on_mod, ref_confs_on_mod, _ = ref.propose(mod_y, anchor, rows, sampling)
            ref_toks_e2e, ref_confs_e2e, _ = ref.propose(y_ref, anchor, rows, sampling)
            mod_toks = drafted[idx, 1:1 + BLOCK].tolist()
            mod_confs = drafted[idx, 1 + BLOCK:].view(np.float32).tolist()
            match_head = sum(int(a == b) for a, b in zip(mod_toks, ref_toks_on_mod))
            match_e2e = sum(int(a == b) for a, b in zip(mod_toks, ref_toks_e2e))
            conf_diff_head = max(abs(a - b) for a, b in zip(mod_confs, ref_confs_on_mod))
            conf_diff_e2e = max(abs(a - b) for a, b in zip(mod_confs, ref_confs_e2e))
            print(f"rule {idx}: head-exact {match_head}/{BLOCK} (conf max diff {conf_diff_head:.2e}), "
                  f"e2e {match_e2e}/{BLOCK} (conf max diff {conf_diff_e2e:.2e}); mod={mod_toks} ref={ref_toks_e2e}")
            if match_head != BLOCK or conf_diff_head > 1e-3:
                failures.append(f"rule {idx} Markov/confidence head mismatch: {match_head}/{BLOCK}, conf_diff={conf_diff_head:.2e}")

    print("PASS" if not failures else "FAIL " + "; ".join(failures))
    return 0 if not failures else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    rand_cmd = commands.add_parser("synthetic")
    rand_cmd.add_argument("--first", type=int, required=True)
    rand_cmd.add_argument("--rows", type=int, required=True)
    rand_cmd.add_argument("--out", type=Path, required=True)
    rand_cmd.add_argument("--seed", type=int, default=7)
    cmp_cmd = commands.add_parser("compare")
    cmp_cmd.add_argument("--checkpoint", required=True)
    cmp_cmd.add_argument("--target", required=True)
    cmp_cmd.add_argument("--out", type=Path, required=True)
    cmp_cmd.add_argument("--precise", action="store_true")
    options = parser.parse_args()
    return synthetic(options) if options.command == "synthetic" else compare(options)


if __name__ == "__main__":
    sys.exit(main())
