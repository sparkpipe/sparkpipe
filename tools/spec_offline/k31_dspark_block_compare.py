#!/usr/bin/env python3
"""Compare the K3.1 Pro DSpark drafter module's block forward with the Python drafter and with float64.

    k31_dspark_block_compare.py extract --recording FILE --rows N --out DIR
    k31_dspark_block_compare.py compare --checkpoint DIR --target SHARD --out DIR

`extract` finds the first sequence of a tap stream recording that holds positions 0..N from one engine session and
request, and writes taps.bin (N rows of the five taps, bf16) and tokens.bin (the N + 1 input tokens, u32) to DIR.
`build/k31_dspark_block_probe` then ingests the taps, runs the block forward at position N with token N as the anchor,
and writes y.bin (the seven block rows after the final norm, bf16), rows0..4.bin (the module's context rows) and
trace0..4.bin (per layer, the module's attention queries, its attention output and the block's latent rows).
`compare` checks two things:
  - per layer, the module's attention on its own queries, block rows and context rows against the Python drafter's
    attention (K3DsparkDrafter.attend) on the same inputs: mean relative difference below 1e-4 and largest below 1e-2;
  - per block row, the top token of y under the target's head equals the Python drafter's.
It also prints how far y of the module and of K3DsparkDrafter.forward_block (on the module's context rows) lie from a
float64 forward of the same weights and rows. Both are bf16 pipelines with the same rounding points, so their distance
from float64 is rounding noise that grows through the five layers; it is reported, not gated.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k3_dspark_drafter import K3DsparkDrafter  # noqa: E402
from tapstream import frames  # noqa: E402

NAMES = {"embed": "language_model.model.embed_tokens.weight", "head": "language_model.lm_head.weight", "norm": "language_model.model.norm.weight"}
TAPS = ["o75", "o79", "o83", "o87", "o91"]


def bf16_file(path, shape):
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16).astype(np.int16)).view(torch.bfloat16).view(*shape)


def frame_key(frame):
    return frame["control_generation"], frame["request_id"], frame["sequence_id"]


def longest_prefix_owner(recording, wanted):
    held = {}
    for _, frame in frames(recording):
        key, first = frame_key(frame), frame["first_position"]
        if first == 0:
            held[key] = 0
        if key not in held or first > held[key]:
            continue
        held[key] = first + frame["rows"]
        if held[key] >= wanted:
            return key
    return None


def extract(options):
    wanted = options.rows + 1
    owner = longest_prefix_owner(options.recording, wanted)
    if owner is None:
        raise SystemExit(f"{options.recording}: no sequence holds positions 0..{options.rows} from one session and request")
    held, taps = None, None
    tokens = np.zeros(wanted, dtype=np.uint32)
    with open(options.recording, "rb") as handle:
        for hello, frame in frames(options.recording):
            first = frame["first_position"]
            if frame_key(frame) != owner or (held is None and first != 0) or (held is not None and first > held):
                continue
            columns = [hello["entries"].index(name) for name in TAPS]
            if taps is None:
                taps = np.zeros((wanted, len(TAPS) * hello["hidden"]), dtype=np.uint16)
            count = min(frame["rows"], wanted - first)
            handle.seek(frame["offset"])
            data = np.frombuffer(handle.read(count * frame["row_bytes"]), dtype=np.uint16).reshape(count, len(hello["entries"]), hello["hidden"])
            taps[first:first + count] = data[:, columns].reshape(count, -1)
            tokens[first:first + count] = frame["tokens"][:count]
            held = first + count
            if held >= wanted:
                break
    options.out.mkdir(parents=True, exist_ok=True)
    taps[:options.rows].tofile(options.out / "taps.bin")
    tokens.tofile(options.out / "tokens.bin")
    print(f"wrote {options.rows} tap rows and {wanted} tokens of sequence {owner[2]} to {options.out}")
    return 0


def rms64(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.double()


def rotate64(x, positions, inverse, gain):
    angles = positions.double()[:, None] * inverse.double()[None, :]
    shape = [angles.shape[0]] + [1] * (x.dim() - 2) + [angles.shape[1]]
    cos, sin = (angles.cos() * gain).view(shape), (angles.sin() * gain).view(shape)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


def forward64(d, anchor, position, contexts, inverse64):
    tokens = torch.full((d.block,), d.mask, dtype=torch.long, device=d.device)
    tokens[0] = anchor
    positions = torch.arange(position, position + d.block, device=d.device)
    residual = d.embed[tokens].double()
    allowed = torch.tril(torch.ones(d.block, d.block, dtype=torch.bool, device=d.device))[:, None, :]
    for layer, context in zip(d.layers, contexts):
        x = rms64(residual, layer["input_layernorm"], d.eps)
        query = rms64(x @ layer["self_attn.q_a_proj"].double().t(), layer["self_attn.q_a_layernorm"], d.eps) @ layer["self_attn.q_b_proj"].double().t()
        query = query.view(d.block, d.heads, d.nope + d.rope_dim)
        absorbed = torch.einsum("ihn,hnr->ihr", query[..., :d.nope], layer["absorb_key"].double())
        query_rope = rotate64(query[..., d.nope:], positions, inverse64, d.rotary_gain)
        latent = x @ layer["self_attn.kv_a_proj_with_mqa"].double().t()
        block = torch.cat((rms64(latent[:, :d.kv_rank], layer["self_attn.kv_a_layernorm"], d.eps),
                           rotate64(latent[:, d.kv_rank:], positions, inverse64, d.rotary_gain)), -1)
        rows = context.double()
        context_scores = (torch.einsum("ihr,pr->ihp", absorbed, rows[:, :d.kv_rank]) + torch.einsum("ihd,pd->ihp", query_rope, rows[:, d.kv_rank:])) * d.scale
        block_scores = (torch.einsum("ihr,pr->ihp", absorbed, block[:, :d.kv_rank]) + torch.einsum("ihd,pd->ihp", query_rope, block[:, d.kv_rank:])) * d.scale
        block_scores = torch.where(allowed, block_scores, torch.full_like(block_scores, float("-inf")))
        weights = torch.softmax(torch.cat((context_scores, block_scores), -1), -1)
        values = torch.cat((rows[:, :d.kv_rank], block[:, :d.kv_rank]), 0)
        attended = torch.einsum("ihp,pr->ihr", weights, values)
        heads = torch.einsum("ihr,hvr->ihv", attended, layer["absorb_value"].double()).reshape(d.block, -1)
        residual = residual + heads @ layer["self_attn.o_proj"].double().t()
        x = rms64(residual, layer["post_attention_layernorm"], d.eps)
        gate, up = x @ layer["mlp.gate_proj"].double().t(), x @ layer["mlp.up_proj"].double().t()
        residual = residual + (torch.nn.functional.silu(gate) * up) @ layer["mlp.down_proj"].double().t()
    return rms64(residual, d.norm, d.eps)


def yarn_inverse64(rope, dim):
    base = float(rope["rope_theta"])
    extrapolated = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    if str(rope.get("rope_type", "default")).lower() not in ("yarn", "deepseek_yarn"):
        return extrapolated
    factor, original = float(rope["factor"]), float(rope["original_max_position_embeddings"])
    fast, slow = float(rope.get("beta_fast", 32)), float(rope.get("beta_slow", 1))
    low = max(math.floor(dim * math.log(original / (fast * 2 * math.pi)) / (2 * math.log(base))), 0)
    high = min(math.ceil(dim * math.log(original / (slow * 2 * math.pi)) / (2 * math.log(base))), dim - 1)
    high = high + 0.001 if low == high else high
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float64) - low) / (high - low), 0, 1)
    return extrapolated / factor * ramp + extrapolated * (1.0 - ramp)


def layer_attention(d, path, context):
    queries_bytes = d.block * d.heads * (d.kv_rank + d.rope_dim) * np.dtype(np.float32).itemsize
    latent_bytes = d.block * d.heads * d.kv_rank * np.dtype(np.float32).itemsize
    raw = np.fromfile(path, dtype=np.uint8)
    queries = torch.from_numpy(raw[:queries_bytes].view(np.float32).copy()).view(d.block, d.heads, -1).cuda()
    latent = torch.from_numpy(raw[queries_bytes:queries_bytes + latent_bytes].view(np.float32).copy()).view(d.block, d.heads, -1).cuda()
    block = torch.from_numpy(raw[queries_bytes + latent_bytes:].view(np.uint16).astype(np.int16).copy()).view(torch.bfloat16).view(d.block, -1).cuda()
    with torch.no_grad():
        python = d.attend(queries[..., :d.kv_rank].contiguous(), queries[..., d.kv_rank:].contiguous(), block, context)
    return latent.double(), python.double()


RULES = ((0, 0.0, 0.0, 0), (12345, 1.0, 0.95, 0), (777, 1.0 / 0.6, 1.0, 20))


def compare_drafts(d, path, y, anchor, position):
    raw = np.fromfile(path, dtype=np.uint32).reshape(len(RULES), 1 + 2 * d.block)
    ok = True
    for index, (seed, inverse_temperature, nucleus, top_k) in enumerate(RULES):
        tokens = raw[index, 1:1 + d.block].astype(np.int64)
        scores = raw[index, 1 + d.block:].view(np.float32)
        with torch.no_grad():
            if inverse_temperature > 0.0:
                rule = (torch.tensor(seed, device="cuda"), torch.tensor(np.float32(inverse_temperature), device="cuda"),
                        torch.tensor(np.float32(nucleus), device="cuda"), torch.tensor(top_k, device="cuda"))
                python_tokens, python_scores, _, _ = d.propose(y, anchor, rule, torch.tensor(position, device="cuda"), torch.tensor(True, device="cuda"))
            else:
                python_tokens, python_scores, _, _ = d.propose(y, anchor)
        python_tokens, python_scores = python_tokens.cpu().numpy(), python_scores.float().cpu().numpy()
        equal = int((tokens == python_tokens).sum())
        worst = float(np.abs(scores - python_scores).max())
        ok = ok and equal == d.block and worst < 1e-3
        print(f"rule {index} (seed {seed}, inverse temperature {inverse_temperature:.3f}, nucleus {nucleus}, top-k {top_k}): drafts equal on {equal} of"
              f" {d.block} slots, largest confidence difference {worst:.2e}; module {tokens.tolist()} python {python_tokens.tolist()}")
    return ok


def compare(options):
    d = K3DsparkDrafter(options.checkpoint, options.target, NAMES, "cuda")
    tokens = np.fromfile(options.out / "tokens.bin", dtype=np.uint32)
    position, anchor = len(tokens) - 1, int(tokens[-1])
    width = d.kv_rank + d.rope_dim
    contexts = [bf16_file(options.out / f"rows{index}.bin", (position, width)).cuda() for index in range(len(d.layers))]
    module = bf16_file(options.out / "y.bin", (d.block, d.fc.shape[0])).double().cuda()
    inverse64 = yarn_inverse64(dict(d.config["rope_parameters"]), d.rope_dim).cuda()
    with torch.no_grad():
        python = d.forward_block(anchor, position, contexts).double()
        truth = forward64(d, anchor, position, contexts, inverse64)
    module_error = ((module - truth).norm(dim=-1) / truth.norm(dim=-1)).cpu()
    python_error = ((python - truth).norm(dim=-1) / truth.norm(dim=-1)).cpu()
    with torch.no_grad():
        top = [(y.to(torch.bfloat16) @ d.head.t()).float().argmax(-1).cpu() for y in (module, python, truth)]
    for row in range(d.block):
        print(f"block row {row}: y against float64 module {module_error[row]:.5f} python {python_error[row]:.5f};"
              f" top token module {int(top[0][row])} python {int(top[1][row])} float64 {int(top[2][row])}")
    attention_ok = True
    for index in range(len(d.layers)):
        latent, python_latent = layer_attention(d, options.out / f"trace{index}.bin", contexts[index])
        difference = ((latent - python_latent).norm(dim=-1) / python_latent.norm(dim=-1))
        mean, worst = difference.mean().item(), difference.max().item()
        attention_ok = attention_ok and mean < 1e-4 and worst < 1e-2
        print(f"layer {index}: attention against the Python drafter on the same inputs, relative difference mean {mean:.2e} largest {worst:.2e}")
    tokens_ok = bool((top[0] == top[1]).all())
    drafts_ok = compare_drafts(d, options.out / "drafts.bin", module.to(torch.bfloat16), anchor, position)
    passed = attention_ok and tokens_ok and drafts_ok
    print(("PASS" if passed else "FAIL") + f" position {position} anchor {anchor}: attention {'matches' if attention_ok else 'differs'},"
          f" top tokens equal on {int((top[0] == top[1]).sum())} of {d.block} rows; y against float64 mean module"
          f" {module_error.mean().item():.5f} python {python_error.mean().item():.5f}")
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    pick = commands.add_parser("extract")
    pick.add_argument("--recording", required=True)
    pick.add_argument("--rows", required=True, type=int)
    pick.add_argument("--out", required=True, type=Path)
    check = commands.add_parser("compare")
    check.add_argument("--checkpoint", required=True)
    check.add_argument("--target", required=True)
    check.add_argument("--out", required=True, type=Path)
    options = parser.parse_args()
    return extract(options) if options.command == "extract" else compare(options)


if __name__ == "__main__":
    sys.exit(main())
