#!/usr/bin/env python3
"""Torch reference of the Qwen3.8-27B text decoder that records tap streams for offline drafter acceptance.

The math is model_contracts/qwen38_27b_authoritative.json as tools/qwen38_27b_reference.c computes it: BF16 weights
and activations rounded at the same points, FP32 accumulation, FP32 GatedDeltaNet state. Layer norms, the final norm
and the attention q/k norms scale by 1 + w; the GatedDeltaNet output norm by w. Full attention (every fourth layer)
has a sigmoid output gate and rotates the first 64 of 256 head dimensions (theta 1e7, halves paired). GatedDeltaNet
layers run a causal 4-tap depthwise convolution with SiLU, L2-normalised q and k (q / sqrt(128)), the gated delta
rule with decay exp(-exp(A_log) softplus(a + dt_bias)) and beta sigmoid(b), then the gated RMS norm with SiLU(z).
Prefill runs the delta rule in 64-row chunks (WY form, triangular solve per chunk) and decode runs it one row at a
time, so a prompt of a few thousand tokens prefills in seconds.

Greedy picks the argmax of the BF16 logits of norm(final) @ head, the token block_drafter.py's final_head_agreement
predicts. Sampling picks block_drafter.coupled_choice on the same logits: the verifier's keyed Gumbel noise (seed,
row position, token) over the nucleus and top-k, so a drafter scored with the same --seed is coupled to the stream.

Taps follow the stage adapter's lane-record syntax, e.g. "o1,o16,o31,o46,o61,final": oL is the residual after
0-based layer L (DFlash target_layer_ids index these, the hidden_states[L + 1] of HF), final the last residual
before the norm. The stream is the tools/spec_offline/tapstream.py layout: one HELLO, a prefill FRAME per prompt
chunk (flag 1) and a decode FRAME per generated token that is fed back (flag 2), request and sequence id 100001 +
prompt index. The last generated token is never fed back, as in serving.

Subcommands:
  head-shard CHECKPOINT OUT     embed, LM head and final norm in one safetensors file, the --target-shard of
                                block_drafter.py and block_draftd.py (names unchanged; the norm is stored F32 with
                                its 1 + w offset folded in, since the drafters multiply by the stored weight)
  run CHECKPOINT                prefill and decode a prompts JSON (prompt_token_ids per prompt, as
                                tools/qwen38_27b_gate_prompts.py writes for its gsm8k, speculation and agent sets)
                                into a tap stream and a replies JSON
  score CHECKPOINT              teacher-forced top-1 next-token agreement and mean NLL (its texts set)
  accuracy REPLIES              GSM8K accuracy of a replies JSON: last \\boxed{} after </think> against the answer
"""
import argparse
import json
import math
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from block_drafter import SafeTensors, coupled_choice, rms
from tapstream import FRAME, HELLO, MAGIC

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "model_contracts" / "qwen38_27b_authoritative.json"
PREFIX = "model.language_model."
NAMES = {"embed": PREFIX + "embed_tokens.weight", "head": "lm_head.weight", "norm": PREFIX + "norm.weight"}
EOS = (248046, 248044)
THINK_CLOSE = "</think>"
CONV_TAPS = 4
CHUNK = 64
PREFILL_ROWS = 4096
SCORE_ROWS = 512
REQUEST_BASE = 100001
CONTROL_GENERATION = 1
FORM_OUTPUT, FORM_FINAL = 1, 3
BF16 = torch.bfloat16


class Checkpoint:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.where = json.loads((self.directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.files = {}

    def load(self, name, device):
        shard = self.where[name]
        if shard not in self.files:
            self.files[shard] = SafeTensors(self.directory / shard)
        return self.files[shard].load(name, device)


def refuse_geometry(contract, config):
    m, g, a, h = contract["model"], contract["gdn"], contract["attention"], contract["hybrid_attention"]
    kinds = config["layer_types"]
    checks = (
        (config["hidden_size"] == m["hidden_dimension"], "hidden_size"),
        (config["num_hidden_layers"] == m["layer_count"] == len(kinds), "num_hidden_layers"),
        (config["vocab_size"] == m["vocabulary_size"], "vocab_size"),
        (config["rms_norm_eps"] == m["rms_norm_epsilon"], "rms_norm_eps"),
        (config["intermediate_size"] == m["dense_intermediate_dimension"], "intermediate_size"),
        (config["linear_num_key_heads"] == g["key_head_count"], "linear_num_key_heads"),
        (config["linear_num_value_heads"] == g["value_head_count"], "linear_num_value_heads"),
        (config["linear_key_head_dim"] == g["key_dimension"] == config["linear_value_head_dim"] == g["value_dimension"], "linear head dims"),
        (config["linear_conv_kernel_dim"] == g["short_conv_kernel"] == CONV_TAPS, "linear_conv_kernel_dim"),
        (config["num_attention_heads"] == a["query_head_count"], "num_attention_heads"),
        (config["num_key_value_heads"] == a["kv_head_count"], "num_key_value_heads"),
        (config["head_dim"] == a["head_dimension"], "head_dim"),
        (round(config["partial_rotary_factor"] * config["head_dim"]) == a["rope_dimension"], "partial_rotary_factor"),
        (float(config["rope_parameters"]["rope_theta"]) == a["rope_theta"], "rope_theta"),
        (config.get("attn_output_gate") is True and a["output_gate"], "attn_output_gate"),
        (all((kind == "full_attention") == (index % h["period"] == h["full_phase"]) for index, kind in enumerate(kinds)), "layer_types"),
    )
    failed = [name for ok, name in checks if not ok]
    if failed:
        raise SystemExit(f"checkpoint config differs from {CONTRACT.name}: {', '.join(failed)}")


def chunked_delta_rule(q, k, v, g, beta, state):
    rows, heads, key_dim = k.shape
    pad = (-rows) % CHUNK
    q, k, v = (F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (q, k, v))
    g, beta = (F.pad(t, (0, 0, 0, pad)) for t in (g, beta))
    chunks = (rows + pad) // CHUNK
    q, k, v = (t.view(chunks, CHUNK, heads, -1).permute(2, 0, 1, 3) for t in (q, k, v))
    g = g.view(chunks, CHUNK, heads).permute(2, 0, 1).cumsum(-1)
    beta = beta.view(chunks, CHUNK, heads).permute(2, 0, 1)[..., None]
    lower = torch.tril(torch.ones(CHUNK, CHUNK, dtype=torch.bool, device=q.device))
    decay = torch.where(lower, (g[..., :, None] - g[..., None, :]).masked_fill(~lower, 0.0).exp(), torch.zeros((), device=q.device))
    strict = lower & ~torch.eye(CHUNK, dtype=torch.bool, device=q.device)
    system = torch.eye(CHUNK, device=q.device) + ((beta * k) @ k.transpose(-1, -2) * decay).masked_fill(~strict, 0.0)
    identity = torch.eye(CHUNK, device=q.device).expand_as(system)
    solve = torch.linalg.solve_triangular(system, identity, upper=False, unitriangular=True)
    values = solve @ (beta * v)
    weighted = solve @ (beta * k * g.exp()[..., None])
    out = torch.empty_like(values)
    for index in range(chunks):
        qc, kc, gc = q[:, index], k[:, index], g[:, index]
        fresh = values[:, index] - weighted[:, index] @ state
        inner = (qc @ kc.transpose(-1, -2) * decay[:, index]).masked_fill(~lower, 0.0)
        out[:, index] = (qc * gc.exp()[..., None]) @ state + inner @ fresh
        state = state * gc[:, -1].exp()[:, None, None] + (kc * (gc[:, -1:] - gc).exp()[..., None]).transpose(-1, -2) @ fresh
    return out.permute(1, 2, 0, 3).reshape(chunks * CHUNK, heads, -1)[:rows], state


class Cache:
    def __init__(self, model, slots, length, device):
        attention = sum(model.full)
        recurrent = len(model.full) - attention
        self.length = length
        self.keys = torch.zeros((attention, slots, model.kv_heads, length, model.head_dim), dtype=BF16, device=device)
        self.values = torch.zeros_like(self.keys)
        self.state = torch.zeros((recurrent, slots, model.value_heads, model.dim, model.dim), dtype=torch.float32, device=device)
        self.window = torch.zeros((recurrent, slots, model.channels, CONV_TAPS - 1), dtype=torch.float32, device=device)

    def reset(self, slot):
        self.state[:, slot] = 0.0
        self.window[:, slot] = 0.0


class Qwen38Target:
    def __init__(self, directory, device):
        contract = json.loads(CONTRACT.read_text())
        config = json.loads((Path(directory) / "config.json").read_text())["text_config"]
        refuse_geometry(contract, config)
        m, g, a = contract["model"], contract["gdn"], contract["attention"]
        self.device = device
        self.hidden, self.eps, self.vocab = m["hidden_dimension"], m["rms_norm_epsilon"], m["vocabulary_size"]
        self.key_heads, self.value_heads, self.dim = g["key_head_count"], g["value_head_count"], g["key_dimension"]
        self.key_width, self.value_width = self.key_heads * self.dim, self.value_heads * self.dim
        self.channels = 2 * self.key_width + self.value_width
        self.heads, self.kv_heads, self.head_dim, self.rope = a["query_head_count"], a["kv_head_count"], a["head_dimension"], a["rope_dimension"]
        self.query_width, self.kv_width = self.heads * self.head_dim, self.kv_heads * self.head_dim
        self.full = [kind == "full_attention" for kind in config["layer_types"]]
        self.slot_of = [sum(self.full[:index]) if full else index - sum(self.full[:index]) for index, full in enumerate(self.full)]
        weights = Checkpoint(directory)
        self.layers = []
        for index in range(len(self.full)):
            self.layers.append(self.load_layer(weights, index, device))
            if index % 16 == 15:
                print(f"loaded {index + 1} layers", file=sys.stderr, flush=True)
        self.embed = weights.load(NAMES["embed"], device)
        self.head = weights.load(NAMES["head"], device)
        self.norm = weights.load(NAMES["norm"], device).float() + 1.0
        self.inverse = (1.0 / a["rope_theta"] ** (torch.arange(0, self.rope, 2, dtype=torch.float64) / self.rope)).float().to(device)

    def load_layer(self, weights, index, device):
        prefix = f"{PREFIX}layers.{index}."

        def load(name):
            return weights.load(prefix + name, device)
        layer = {"input_norm": load("input_layernorm.weight").float() + 1.0, "post_norm": load("post_attention_layernorm.weight").float() + 1.0,
                 "gate_up": torch.cat((load("mlp.gate_proj.weight"), load("mlp.up_proj.weight"))), "down": load("mlp.down_proj.weight")}
        if self.full[index]:
            layer.update(qkv=torch.cat((load("self_attn.q_proj.weight"), load("self_attn.k_proj.weight"), load("self_attn.v_proj.weight"))),
                         o=load("self_attn.o_proj.weight"), q_norm=load("self_attn.q_norm.weight").float() + 1.0,
                         k_norm=load("self_attn.k_norm.weight").float() + 1.0)
        else:
            layer.update(mix=torch.cat((load("linear_attn.in_proj_qkv.weight"), load("linear_attn.in_proj_z.weight"), load("linear_attn.in_proj_b.weight"),
                                        load("linear_attn.in_proj_a.weight"))),
                         conv=load("linear_attn.conv1d.weight").float().view(self.channels, CONV_TAPS), rate=load("linear_attn.A_log").float().exp(),
                         dt_bias=load("linear_attn.dt_bias").float(), gdn_norm=load("linear_attn.norm.weight").float(), out=load("linear_attn.out_proj.weight"))
        return layer

    def norm_rows(self, x, weight):
        return rms(x, weight, self.eps).to(BF16)

    def mlp(self, layer, x):
        gate, up = (x @ layer["gate_up"].t()).chunk(2, -1)
        return (F.silu(gate.float()) * up.float()).to(BF16) @ layer["down"].t()

    def head_rope(self, x, weight, positions):
        normed = rms(x, weight, self.eps).to(BF16).float()
        angles = positions.float()[:, None] * self.inverse[None, :]
        cos, sin = angles.cos()[:, None, :], angles.sin()[:, None, :]
        half = self.rope // 2
        first, second = normed[..., :half], normed[..., half:self.rope]
        return torch.cat((first * cos - second * sin, second * cos + first * sin, normed[..., self.rope:]), -1).to(BF16)

    def attention_inputs(self, layer, x, positions):
        rows = x.shape[0]
        proj = x @ layer["qkv"].t()
        gated = proj[:, :2 * self.query_width].view(rows, self.heads, 2 * self.head_dim)
        q = self.head_rope(gated[..., :self.head_dim], layer["q_norm"], positions)
        k = self.head_rope(proj[:, 2 * self.query_width:2 * self.query_width + self.kv_width].view(rows, self.kv_heads, self.head_dim), layer["k_norm"], positions)
        v = proj[:, 2 * self.query_width + self.kv_width:].view(rows, self.kv_heads, self.head_dim)
        return q, k, v, gated[..., self.head_dim:]

    def attention_output(self, layer, out, gate):
        gated = (out.float() * torch.sigmoid(gate.float())).to(BF16)
        return gated.reshape(gated.shape[0], -1) @ layer["o"].t()

    def attend_rows(self, layer, slot_index, x, cache, slot, first):
        rows = x.shape[0]
        positions = torch.arange(first, first + rows, device=self.device)
        q, k, v, gate = self.attention_inputs(layer, x, positions)
        cache.keys[slot_index, slot, :, first:first + rows] = k.transpose(0, 1)
        cache.values[slot_index, slot, :, first:first + rows] = v.transpose(0, 1)
        keys, values = cache.keys[slot_index, slot, :, :first + rows][None], cache.values[slot_index, slot, :, :first + rows][None]
        mask = None if first == 0 else torch.arange(first + rows, device=self.device)[None, :] <= positions[:, None]
        out = F.scaled_dot_product_attention(q.transpose(0, 1)[None], keys, values, attn_mask=mask, is_causal=first == 0,
                                             scale=1.0 / math.sqrt(self.head_dim), enable_gqa=True)
        return self.attention_output(layer, out[0].transpose(0, 1), gate)

    def attend_step(self, layer, slot_index, x, cache, positions, span):
        q, k, v, gate = self.attention_inputs(layer, x, positions)
        slots = torch.arange(x.shape[0], device=self.device)
        cache.keys[slot_index][slots, :, positions] = k
        cache.values[slot_index][slots, :, positions] = v
        mask = (torch.arange(span, device=self.device)[None, :] <= positions[:, None])[:, None, None, :]
        out = F.scaled_dot_product_attention(q[:, :, None, :], cache.keys[slot_index, :, :, :span], cache.values[slot_index, :, :, :span], attn_mask=mask,
                                             scale=1.0 / math.sqrt(self.head_dim), enable_gqa=True)
        return self.attention_output(layer, out[:, :, 0], gate)

    def recurrent_inputs(self, layer, mixed, a, b):
        rows = mixed.shape[0]
        q = mixed[:, :self.key_width].view(rows, self.key_heads, self.dim)
        k = mixed[:, self.key_width:2 * self.key_width].view(rows, self.key_heads, self.dim)
        v = mixed[:, 2 * self.key_width:].view(rows, self.value_heads, self.dim)
        q = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + self.eps) / math.sqrt(self.dim)
        k = k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + self.eps)
        share = self.value_heads // self.key_heads
        g = -layer["rate"] * F.softplus(a.float() + layer["dt_bias"])
        return q.repeat_interleave(share, 1), k.repeat_interleave(share, 1), v, g, torch.sigmoid(b.float())

    def recurrent_output(self, layer, core, z):
        rows = core.shape[0]
        o = core.to(BF16).float()
        normed = ((o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + self.eps)).to(BF16).float() * layer["gdn_norm"]).to(BF16).float()
        gated = (normed * F.silu(z.float().view(rows, self.value_heads, self.dim))).to(BF16)
        return gated.reshape(rows, -1) @ layer["out"].t()

    def split_mix(self, layer, x):
        return (x @ layer["mix"].t()).split((self.channels, self.value_width, self.value_heads, self.value_heads), -1)

    def recur_rows(self, layer, slot_index, x, cache, slot):
        rows = x.shape[0]
        mix, z, b, a = self.split_mix(layer, x)
        sequence = torch.cat((cache.window[slot_index, slot].t(), mix.float()), 0)
        total = sum(sequence[tap:tap + rows] * layer["conv"][:, tap] for tap in range(CONV_TAPS))
        cache.window[slot_index, slot] = sequence[rows:].t()
        mixed = F.silu(total.to(BF16).float()).to(BF16).float()
        q, k, v, g, beta = self.recurrent_inputs(layer, mixed, a, b)
        core, cache.state[slot_index, slot] = chunked_delta_rule(q, k, v, g, beta, cache.state[slot_index, slot])
        return self.recurrent_output(layer, core, z)

    def recur_step(self, layer, slot_index, x, cache):
        mix, z, b, a = self.split_mix(layer, x)
        sequence = torch.cat((cache.window[slot_index], mix.float()[..., None]), -1)
        total = sum(sequence[..., tap] * layer["conv"][:, tap] for tap in range(CONV_TAPS))
        cache.window[slot_index] = sequence[..., 1:]
        mixed = F.silu(total.to(BF16).float()).to(BF16).float()
        q, k, v, g, beta = self.recurrent_inputs(layer, mixed, a, b)
        state = cache.state[slot_index]
        state.mul_(g.exp()[..., None, None])
        predicted = torch.matmul(k[:, :, None, :], state)[:, :, 0]
        update = beta[..., None] * (v - predicted)
        state.view(-1, self.dim, self.dim).baddbmm_(k.reshape(-1, self.dim, 1), update.reshape(-1, 1, self.dim))
        core = torch.matmul(q[:, :, None, :], state)[:, :, 0]
        return self.recurrent_output(layer, core, z)

    def layers_forward(self, hidden, mixer, columns):
        captured = torch.empty((hidden.shape[0], len(columns) + 1, self.hidden), dtype=BF16, device=self.device) if columns is not None else None
        for index, layer in enumerate(self.layers):
            hidden = hidden + mixer(index, layer, self.norm_rows(hidden, layer["input_norm"]))
            hidden = hidden + self.mlp(layer, self.norm_rows(hidden, layer["post_norm"]))
            if columns is not None and index in columns:
                captured[:, columns[index]] = hidden
        if captured is not None:
            captured[:, -1] = hidden
        return hidden, captured

    def prefill(self, cache, slot, tokens, columns, keep_hidden=False):
        cache.reset(slot)
        finals, rows = [], []
        for first in range(0, len(tokens), PREFILL_ROWS):
            ids = torch.tensor(tokens[first:first + PREFILL_ROWS], device=self.device)

            def mixer(index, layer, x):
                if self.full[index]:
                    return self.attend_rows(layer, self.slot_of[index], x, cache, slot, first)
                return self.recur_rows(layer, self.slot_of[index], x, cache, slot)
            hidden, captured = self.layers_forward(self.embed[ids], mixer, columns)
            finals.append(hidden if keep_hidden else hidden[-1:])
            if captured is not None:
                rows.append(captured.cpu())
        return torch.cat(finals), (torch.cat(rows) if rows else None)

    def step(self, cache, tokens, positions, span, columns):
        def mixer(index, layer, x):
            if self.full[index]:
                return self.attend_step(layer, self.slot_of[index], x, cache, positions, span)
            return self.recur_step(layer, self.slot_of[index], x, cache)
        return self.layers_forward(self.embed[tokens], mixer, columns)

    def logits(self, final):
        return (self.norm_rows(final, self.norm) @ self.head.t()).float()


def parse_entries(text):
    entries = []
    for name in text.split(","):
        if name == "final":
            entries.append((FORM_FINAL, 0))
        elif re.fullmatch(r"o\d+", name):
            entries.append((FORM_OUTPUT, int(name[1:])))
        else:
            raise SystemExit(f"taps entry {name!r} is not o<layer> or final")
    if entries.count((FORM_FINAL, 0)) != 1 or entries[-1] != (FORM_FINAL, 0) or len(set(entries)) != len(entries):
        raise SystemExit("taps must name distinct o<layer> entries and end with final")
    return entries


def fnv(hash_value, data):
    for byte in data:
        hash_value = ((hash_value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return hash_value


class TapWriter:
    def __init__(self, path, entries, hidden, layers, tag, identity):
        self.handle = open(path, "wb")
        self.serial = 0
        self.entries = entries
        packed = b"".join(struct.pack("<HH", form, layer) for form, layer in entries)
        name = tag.encode()[:32].ljust(32, b"\0")
        identity = identity[:32].ljust(32, b"\0")
        self.fingerprint = fnv(fnv(fnv(0xCBF29CE484222325, name), identity), struct.pack("<4I", hidden, layers, 1, len(entries)) + packed)
        self.handle.write(HELLO.pack(MAGIC, 1, 1, HELLO.size, name, identity, hidden, layers, 1, len(entries), 0, 1, self.fingerprint, packed.ljust(512, b"\0")))

    def frame(self, request, first, tokens, flags, tail, rows):
        rows = np.ascontiguousarray(rows)
        payload = len(tokens) * 4 + rows.nbytes
        self.handle.write(FRAME.pack(MAGIC, 1, 2, FRAME.size, self.fingerprint, CONTROL_GENERATION, request, 0, request, first, self.serial, len(tokens), flags,
                                     tail, len(self.entries), payload))
        self.handle.write(np.asarray(tokens, dtype=np.uint32).tobytes())
        self.handle.write(rows.tobytes())
        self.serial += 1

    def close(self):
        self.handle.close()


def bf16_bits(tensor):
    return tensor.contiguous().view(torch.int16).numpy().view(np.uint16)


def byte_decoder():
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\u00a1"), ord("\u00ac") + 1)) + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    codes = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            codes.append(256 + extra)
            extra += 1
    return {chr(code): byte for byte, code in zip(printable, codes)}


class Detokenizer:
    def __init__(self, path):
        tokenizer = json.loads(Path(path).read_text())
        self.pieces = {index: piece for piece, index in tokenizer["model"]["vocab"].items()}
        self.special = {entry["id"]: entry["content"] for entry in tokenizer["added_tokens"]}
        self.bytes = byte_decoder()

    def text(self, ids):
        out = bytearray()
        for index in ids:
            if index in self.special:
                out += self.special[index].encode()
            else:
                out += bytes(self.bytes[ch] for ch in self.pieces.get(index, ""))
        return out.decode("utf-8", "replace")


def choose(logits, rule, positions):
    if rule is None:
        return torch.argmax(logits, -1).tolist()
    return [int(coupled_choice(logits[row], rule, torch.tensor(int(positions[row]), device=logits.device))) for row in range(logits.shape[0])]


def sampling_rule(args, device):
    if args.temperature <= 0.0:
        return None
    seed = args.seed - (1 << 64) if args.seed >= (1 << 63) else args.seed
    return (torch.tensor(seed, device=device), torch.tensor(1.0 / args.temperature, device=device),
            torch.tensor(args.top_p if args.top_p < 1.0 else 0.0, device=device), torch.tensor(args.top_k, device=device))


def exact_matmuls():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False


def run(args):
    device = "cuda"
    exact_matmuls()
    prompts = json.loads(Path(args.prompts).read_text())["prompts"][args.skip:args.skip + args.limit if args.limit else None]
    entries = parse_entries(args.taps)
    columns = {layer: column for column, (form, layer) in enumerate(entries) if form == FORM_OUTPUT}
    started = time.monotonic()
    model = Qwen38Target(args.checkpoint, device)
    print(f"TARGET-LOADED {time.monotonic() - started:.1f} s", flush=True)
    detokenizer = Detokenizer(Path(args.checkpoint) / "tokenizer.json")
    longest = max(len(p["prompt_token_ids"]) for p in prompts) + args.max_new
    cache = Cache(model, args.slots, longest, device)
    writer = TapWriter(args.taps_out, entries, model.hidden, len(model.layers), "qwen3.8-27b", b"torch-reference")
    rule = sampling_rule(args, device)
    pending = list(range(len(prompts)))
    active = [None] * args.slots
    replies = [None] * len(prompts)
    steps = decode_rows = 0
    decode_seconds = 0.0

    def finish(slot, reason):
        item = active[slot]
        prompt = prompts[item["index"]]
        replies[item["index"]] = {"name": prompt.get("name", str(item["index"])), "request_id": REQUEST_BASE + item["index"], "prompt_tokens": len(prompt["prompt_token_ids"]),
                                  "generated": item["generated"], "stop": reason, "answer": prompt.get("answer"), "text": detokenizer.text(item["generated"]),
                                  "seconds": round(time.monotonic() - item["started"], 1)}
        print(f"DONE {replies[item['index']]['name']} prompt {len(prompt['prompt_token_ids'])} generated {len(item['generated'])} stop {reason}", flush=True)
        active[slot] = None

    while pending or any(item is not None for item in active):
        for slot in range(args.slots):
            if active[slot] is not None or not pending:
                continue
            index = pending.pop(0)
            tokens = prompts[index]["prompt_token_ids"]
            with torch.no_grad():
                final, rows = model.prefill(cache, slot, tokens, columns)
                opening = choose(model.logits(final[-1:]), rule, [len(tokens) - 1])[0]
            request = REQUEST_BASE + index
            for begin in range(0, len(tokens), PREFILL_ROWS):
                tail = tokens[begin + PREFILL_ROWS] if begin + PREFILL_ROWS < len(tokens) else opening
                writer.frame(request, begin, tokens[begin:begin + PREFILL_ROWS], 1, tail, bf16_bits(rows[begin:begin + PREFILL_ROWS]))
            active[slot] = {"index": index, "position": len(tokens), "generated": [opening], "started": time.monotonic()}
            if opening in EOS or args.max_new <= 1:
                finish(slot, "eos" if opening in EOS else "length")
        live = [slot for slot in range(args.slots) if active[slot] is not None]
        if not live:
            continue
        tokens = [active[slot]["generated"][-1] if active[slot] is not None else 0 for slot in range(args.slots)]
        positions = [active[slot]["position"] if active[slot] is not None else 0 for slot in range(args.slots)]
        begin = time.monotonic()
        with torch.no_grad():
            final, captured = model.step(cache, torch.tensor(tokens, device=device), torch.tensor(positions, device=device), max(positions) + 1, columns)
            picked = choose(model.logits(final), rule, positions)
            captured = captured.cpu()
        decode_seconds += time.monotonic() - begin
        steps += 1
        decode_rows += len(live)
        for slot in live:
            item = active[slot]
            request = REQUEST_BASE + item["index"]
            writer.frame(request, item["position"], [tokens[slot]], 2, picked[slot], bf16_bits(captured[slot:slot + 1]))
            item["generated"].append(picked[slot])
            item["position"] += 1
            if picked[slot] in EOS:
                finish(slot, "eos")
            elif len(item["generated"]) >= args.max_new:
                finish(slot, "length")
        if steps % 100 == 0:
            print(f"STEP {steps} live {len(live)} decode {decode_rows / max(decode_seconds, 1e-9):.1f} rows/s {decode_seconds / steps * 1e3:.0f} ms/step", flush=True)
    writer.close()
    report = {"checkpoint": str(args.checkpoint), "prompts": str(args.prompts), "taps": args.taps, "taps_out": str(args.taps_out),
              "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k, "seed": args.seed, "max_new": args.max_new, "slots": args.slots,
              "decode_rows_per_second": decode_rows / max(decode_seconds, 1e-9), "seconds": time.monotonic() - started, "replies": replies}
    Path(args.replies).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")


def write_safetensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, tensor in tensors.items():
        if tensor.dtype == BF16:
            dtype, blob = "BF16", bf16_bits(tensor.cpu()).tobytes()
        elif tensor.dtype == torch.float32:
            dtype, blob = "F32", tensor.cpu().contiguous().numpy().tobytes()
        else:
            raise SystemExit(f"{name}: {tensor.dtype} is neither bf16 nor f32")
        header[name] = {"dtype": dtype, "shape": list(tensor.shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for blob in blobs:
            handle.write(blob)


def head_shard(args):
    weights = Checkpoint(args.checkpoint)
    tensors = {NAMES["embed"]: weights.load(NAMES["embed"], "cpu"), NAMES["head"]: weights.load(NAMES["head"], "cpu"),
               NAMES["norm"]: weights.load(NAMES["norm"], "cpu").float() + 1.0}
    write_safetensors(args.out, tensors)
    print(f"HEAD-SHARD {args.out}: " + ", ".join(f"{name} {tuple(t.shape)} {t.dtype}" for name, t in tensors.items()))


def score(args):
    device = "cuda"
    exact_matmuls()
    prompts = json.loads(Path(args.prompts).read_text())["prompts"]
    model = Qwen38Target(args.checkpoint, device)
    cache = Cache(model, 1, max(len(p["prompt_token_ids"]) for p in prompts), device)
    results = []
    for prompt in prompts:
        tokens = prompt["prompt_token_ids"]
        with torch.no_grad():
            final, _ = model.prefill(cache, 0, tokens, None, keep_hidden=True)
            target = torch.tensor(tokens[1:], device=device)
            hits, loss = 0, 0.0
            for begin in range(0, len(tokens) - 1, SCORE_ROWS):
                logits = model.logits(final[begin:min(begin + SCORE_ROWS, len(tokens) - 1)])
                truth = target[begin:begin + logits.shape[0]]
                hits += int((torch.argmax(logits, -1) == truth).sum())
                loss += float(F.cross_entropy(logits, truth, reduction="sum"))
        count = len(tokens) - 1
        results.append({"name": prompt.get("name"), "positions": count, "top1": hits / count, "nll": loss / count, "perplexity": math.exp(loss / count)})
        print(json.dumps(results[-1]), flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1) + "\n")


def boxed(text):
    found, start = [], 0
    while True:
        start = text.find("\\boxed{", start)
        if start < 0:
            return found[-1] if found else None
        depth, cursor = 1, start + len("\\boxed{")
        while cursor < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[cursor], 0)
            cursor += 1
        found.append(text[start + len("\\boxed{"):cursor - 1])
        start = cursor


def numeric(text):
    if text is None:
        return None
    cleaned = re.sub(r"\\text\{[^}]*\}|\\[,!; ]|\$|\\%|%|,", "", text).replace("\\dfrac", "\\frac").strip()
    match = re.fullmatch(r"-?\d+(?:\.\d+)?", cleaned)
    if match:
        return float(cleaned)
    fraction = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(\d+)\}", cleaned)
    return (-1 if fraction.group(1) else 1) * int(fraction.group(2)) / int(fraction.group(3)) if fraction else None


def accuracy(args):
    report = json.loads(Path(args.replies).read_text())
    correct = total = closed = 0
    for reply in report["replies"]:
        answer = reply["text"].split(THINK_CLOSE)[-1] if THINK_CLOSE in reply["text"] else None
        closed += answer is not None
        got = numeric(boxed(answer)) if answer is not None else None
        correct += got is not None and abs(got - float(reply["answer"])) < 1e-6
        total += 1
    lengths = [len(reply["generated"]) for reply in report["replies"]]
    print(json.dumps({"replies": total, "correct": correct, "accuracy": correct / max(1, total), "think_closed": closed, "mean_generated": sum(lengths) / max(1, total),
                      "hit_length_limit": sum(reply["stop"] == "length" for reply in report["replies"])}))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    shard = commands.add_parser("head-shard")
    shard.add_argument("checkpoint")
    shard.add_argument("out")
    generate = commands.add_parser("run")
    generate.add_argument("checkpoint")
    generate.add_argument("--prompts", required=True)
    generate.add_argument("--skip", type=int, default=0)
    generate.add_argument("--limit", type=int, default=0)
    generate.add_argument("--taps", default="o1,o16,o31,o46,o61,final")
    generate.add_argument("--taps-out", required=True)
    generate.add_argument("--replies", required=True)
    generate.add_argument("--max-new", type=int, default=4096)
    generate.add_argument("--slots", type=int, default=16, help="sequences decoded together")
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--top-p", type=float, default=1.0)
    generate.add_argument("--top-k", type=int, default=0)
    generate.add_argument("--seed", type=int, default=0)
    scoring = commands.add_parser("score")
    scoring.add_argument("checkpoint")
    scoring.add_argument("--prompts", required=True)
    scoring.add_argument("--out", required=True)
    graded = commands.add_parser("accuracy")
    graded.add_argument("replies")
    args = parser.parse_args()
    {"head-shard": head_shard, "run": run, "score": score, "accuracy": accuracy}[args.command](args)


if __name__ == "__main__":
    main()
