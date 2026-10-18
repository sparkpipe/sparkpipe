#!/usr/bin/env python3
"""Offline acceptance of a block drafter (DFlash backbone, optional DSpark heads)
fed target taps recorded by a stage adapter (tools/spec_offline/tapstream.py).

For every anchor position a of a request's reply, the drafter sees the
context features of positions < a only, drafts a block from the anchor
token and is scored against the tokens the target committed. Arms:
  --delta        tap layer offset added to the drafter's target layer ids;
  --convention   shift: query slot j predicts position a+1+j (block drafts);
                 inplace: slot j >= 1 predicts position a+j (block-1 drafts).
The target head on norm(final) must reproduce the committed next token
(check printed as final_head_agreement).
"""
import argparse
import json
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tapstream import frames  # noqa: E402

DTYPES = {"BF16": (np.uint16, torch.bfloat16), "F32": (np.float32, torch.float32)}


SPLITMIX_INCREMENT = 0x9e3779b97f4a7c15 - (1 << 64)
SPLITMIX_A = 0xbf58476d1ce4e5b9 - (1 << 64)
SPLITMIX_B = 0x94d049bb133111eb - (1 << 64)
GUMBEL_UNIFORM_BITS = 24
COUPLED_CANDIDATES = 64


def shift_right(x, bits):
    return (x >> bits) & ((1 << (64 - bits)) - 1)


def splitmix64(x):
    x = x + SPLITMIX_INCREMENT
    x = (x ^ shift_right(x, 30)) * SPLITMIX_A
    x = (x ^ shift_right(x, 27)) * SPLITMIX_B
    return x ^ shift_right(x, 31)


def keyed_gumbel(seed, position, tokens):
    """The verifier's sampling noise (inference/kernels/head.cuh LmGumbelNoise) for int64 tensors."""
    bits = splitmix64(seed ^ splitmix64((position << 32) | tokens))
    uniform = (shift_right(bits, 64 - GUMBEL_UNIFORM_BITS).to(torch.float32) + 0.5) / float(1 << GUMBEL_UNIFORM_BITS)
    return -torch.log(-torch.log(uniform))


def coupled_choice(logits, sampling, position):
    """Draft the token the verifier would sample if the drafter's distribution were the target's: the same keyed
    Gumbel noise over the drafter's own nucleus and top-k candidates."""
    seed, inverse_temperature, nucleus, top_k = sampling
    scaled = logits * inverse_temperature
    values, candidates = torch.topk(scaled, COUPLED_CANDIDATES)
    probabilities = torch.softmax(scaled, -1)[candidates]
    before = torch.cumsum(probabilities, 0) - probabilities
    rank = torch.arange(COUPLED_CANDIDATES, device=logits.device)
    keep = ((nucleus <= 0.0) | (before < nucleus)) & ((top_k <= 0) | (rank < top_k))
    keep = keep | (rank == 0)
    score = torch.where(keep, values + keyed_gumbel(seed, position, candidates), torch.full_like(values, float("-inf")))
    return torch.index_select(candidates, 0, torch.argmax(score).view(1)).view(())


class SafeTensors:
    def __init__(self, path):
        self.path = Path(path)
        with open(self.path, "rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            self.header = json.loads(handle.read(size))
        self.base = 8 + size

    def load(self, name, device):
        entry = self.header[name]
        numpy_type, torch_type = DTYPES[entry["dtype"]]
        begin, end = entry["data_offsets"]
        array = np.memmap(self.path, dtype=numpy_type, mode="r", offset=self.base + begin, shape=((end - begin) // np.dtype(numpy_type).itemsize,))
        tensor = torch.from_numpy(np.array(array))
        if torch_type == torch.bfloat16:
            tensor = tensor.view(torch.bfloat16)
        return tensor.reshape(entry["shape"]).to(device)


def rms(x, weight, eps):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)) * weight.float()


def yarn_frequencies(config, device):
    rope = config.get("rope_parameters") or config.get("rope_scaling") or {}
    dim = config["head_dim"]
    base = rope.get("rope_theta", config.get("rope_theta", 10000.0))
    pos = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    if rope.get("rope_type") != "yarn":
        return (1.0 / pos).to(device), 1.0
    factor = rope["factor"]
    original = rope["original_max_position_embeddings"]
    fast, slow = rope.get("beta_fast") or 32, rope.get("beta_slow") or 1

    def correction(rotations):
        return (dim * math.log(original / (rotations * 2 * math.pi))) / (2 * math.log(base))
    low, high = max(math.floor(correction(fast)), 0), min(math.ceil(correction(slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    extrapolation = 1 - ramp
    inv = (1.0 / (factor * pos)) * (1 - extrapolation) + (1.0 / pos) * extrapolation
    attention = rope.get("attention_factor") or (0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0)
    return inv.to(device), attention


def rotate(x, positions, inv, scale):
    angles = positions.float()[:, None] * inv[None, :]
    cos = torch.cat([angles.cos(), angles.cos()], -1) * scale
    sin = torch.cat([angles.sin(), angles.sin()], -1) * scale
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], -1)
    return x * cos[:, None, :] + rotated * sin[:, None, :]


class BlockDrafter:
    convention = "shift"

    def __init__(self, directory, shard, names, device):
        self.device = device
        self.config = json.load(open(Path(directory) / "config.json"))
        c = self.config
        weights = SafeTensors(Path(directory) / "model.safetensors")
        target = SafeTensors(shard)
        self.eps = c["rms_norm_eps"]
        self.heads, self.kv_heads, self.head_dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        dflash = c.get("dflash_config", {})
        self.block = c.get("block_size") or dflash["block_size"]
        self.layer_ids = dflash.get("target_layer_ids") or c.get("aux_hidden_state_layer_ids")
        self.mask = dflash.get("mask_token_id", c.get("mask_token_id"))
        self.fc = weights.load("fc.weight", device)
        self.fc32 = self.fc.float().t().contiguous()
        self.hidden_norm = weights.load("hidden_norm.weight", device)
        self.norm = weights.load("norm.weight", device)
        self.layers = []
        for index in range(c["num_hidden_layers"]):
            prefix = f"layers.{index}."
            self.layers.append({name: weights.load(prefix + name + ".weight", device) for name in (
                "input_layernorm", "post_attention_layernorm", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "self_attn.o_proj", "self_attn.q_norm", "self_attn.k_norm", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")})
        has = weights.header.keys()
        self.markov_w1 = weights.load("markov_head.markov_w1.weight", device) if "markov_head.markov_w1.weight" in has else None
        self.markov_w2 = weights.load("markov_head.markov_w2.weight", device) if "markov_head.markov_w2.weight" in has else None
        self.confidence = (weights.load("confidence_head.proj.weight", device), weights.load("confidence_head.proj.bias", device)) \
            if "confidence_head.proj.weight" in has else None
        self.embed = target.load(names["embed"], device)
        self.head = target.load(names["head"], device)
        self.target_norm = target.load(names["norm"], device)
        self.inv, self.attention = yarn_frequencies(c, device)

    def context(self, taps, positions):
        features = rms(taps.float() @ self.fc32, self.hidden_norm, self.eps).to(torch.bfloat16)
        keys, values = [], []
        for layer in self.layers:
            k = (features @ layer["self_attn.k_proj"].t()).view(-1, self.kv_heads, self.head_dim)
            k = rotate(rms(k, layer["self_attn.k_norm"], self.eps), positions, self.inv, self.attention).to(torch.bfloat16)
            v = (features @ layer["self_attn.v_proj"].t()).view(-1, self.kv_heads, self.head_dim)
            keys.append(k)
            values.append(v)
        return keys, values

    def draft(self, anchor_token, anchor_position, keys, values, count):
        tokens = torch.full((self.block,), self.mask, dtype=torch.long, device=self.device)
        tokens[0] = anchor_token
        h = self.embed[tokens].float()
        positions = torch.arange(anchor_position, anchor_position + self.block, device=self.device)
        group = self.heads // self.kv_heads
        for index, layer in enumerate(self.layers):
            x = rms(h, layer["input_layernorm"], self.eps).to(torch.bfloat16)
            q = (x @ layer["self_attn.q_proj"].t()).view(self.block, self.heads, self.head_dim)
            q = rotate(rms(q, layer["self_attn.q_norm"], self.eps), positions, self.inv, self.attention)
            kb = (x @ layer["self_attn.k_proj"].t()).view(self.block, self.kv_heads, self.head_dim)
            kb = rotate(rms(kb, layer["self_attn.k_norm"], self.eps), positions, self.inv, self.attention)
            vb = (x @ layer["self_attn.v_proj"].t()).view(self.block, self.kv_heads, self.head_dim).float()
            k = torch.cat([keys[index][:count].float(), kb], 0)
            v = torch.cat([values[index][:count].float(), vb], 0)
            k = k.repeat_interleave(group, dim=1).transpose(0, 1)
            v = v.repeat_interleave(group, dim=1).transpose(0, 1)
            scores = torch.matmul(q.transpose(0, 1), k.transpose(1, 2)) / math.sqrt(self.head_dim)
            attended = torch.matmul(torch.softmax(scores, -1), v).transpose(0, 1).reshape(self.block, -1)
            h = h + (attended.to(torch.bfloat16) @ layer["self_attn.o_proj"].t()).float()
            x = rms(h, layer["post_attention_layernorm"], self.eps).to(torch.bfloat16)
            gate = x @ layer["mlp.gate_proj"].t()
            up = x @ layer["mlp.up_proj"].t()
            h = h + ((torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16) @ layer["mlp.down_proj"].t()).float()
        y = rms(h, self.norm, self.eps).to(torch.bfloat16)
        base = (y @ self.head.t()).float()
        return y, base

    def draft_range(self, anchor_token, anchor_position, keys, values, begin, end):
        block, heads, kv_heads, dim = self.block, self.heads, self.kv_heads, self.head_dim
        group = heads // kv_heads
        tokens = torch.full((block,), self.mask, dtype=torch.long, device=self.device)
        tokens[0] = anchor_token
        h = self.embed[tokens].float()
        positions = torch.arange(anchor_position, anchor_position + block, device=self.device)
        for index, layer in enumerate(self.layers):
            x = rms(h, layer["input_layernorm"], self.eps).to(torch.bfloat16)
            q = (x @ layer["self_attn.q_proj"].t()).view(block, heads, dim)
            q = rotate(rms(q, layer["self_attn.q_norm"], self.eps), positions, self.inv, self.attention)
            kb = (x @ layer["self_attn.k_proj"].t()).view(block, kv_heads, dim)
            kb = rotate(rms(kb, layer["self_attn.k_norm"], self.eps), positions, self.inv, self.attention)
            vb = (x @ layer["self_attn.v_proj"].t()).view(block, kv_heads, dim).float()
            k = torch.cat([keys[index][begin:end].float(), kb], 0).permute(1, 2, 0)
            v = torch.cat([values[index][begin:end].float(), vb], 0).transpose(0, 1)
            grouped = q.view(block, kv_heads, group, dim).permute(1, 2, 0, 3).reshape(kv_heads, group * block, dim)
            scores = torch.bmm(grouped, k) / math.sqrt(dim)
            attended = torch.bmm(torch.softmax(scores, -1), v).view(kv_heads, group, block, dim).permute(2, 0, 1, 3).reshape(block, -1)
            h = h + (attended.to(torch.bfloat16) @ layer["self_attn.o_proj"].t()).float()
            x = rms(h, layer["post_attention_layernorm"], self.eps).to(torch.bfloat16)
            gate = x @ layer["mlp.gate_proj"].t()
            up = x @ layer["mlp.up_proj"].t()
            h = h + ((torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16) @ layer["mlp.down_proj"].t()).float()
        y = rms(h, self.norm, self.eps).to(torch.bfloat16)
        return y, (y @ self.head.t()).float()

    def pre_attention(self, layer, h, positions):
        block, heads, kv_heads, dim = self.block, self.heads, self.kv_heads, self.head_dim
        x = rms(h, layer["input_layernorm"], self.eps).to(torch.bfloat16)
        q = (x @ layer["self_attn.q_proj"].t()).view(block, heads, dim)
        q = rotate(rms(q, layer["self_attn.q_norm"], self.eps), positions, self.inv, self.attention).to(torch.bfloat16)
        kb = (x @ layer["self_attn.k_proj"].t()).view(block, kv_heads, dim)
        kb = rotate(rms(kb, layer["self_attn.k_norm"], self.eps), positions, self.inv, self.attention).to(torch.bfloat16)
        vb = (x @ layer["self_attn.v_proj"].t()).view(block, kv_heads, dim)
        return q.transpose(0, 1), kb.transpose(0, 1), vb.transpose(0, 1)

    def post_attention(self, layer, h, attended):
        h = h + (attended @ layer["self_attn.o_proj"].t()).float()
        x = rms(h, layer["post_attention_layernorm"], self.eps).to(torch.bfloat16)
        gate = x @ layer["mlp.gate_proj"].t()
        up = x @ layer["mlp.up_proj"].t()
        return h + ((torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16) @ layer["mlp.down_proj"].t()).float()

    def attend_window(self, index, q, kb, vb, keys, values, start, block_at, mask):
        stop = block_at + self.block
        keys[index][:, block_at:stop] = kb
        values[index][:, block_at:stop] = vb
        attended = torch.nn.functional.scaled_dot_product_attention(q[None], keys[index][None, :, start:stop], values[index][None, :, start:stop],
                                                                    attn_mask=mask, enable_gqa=True)
        return attended[0].transpose(0, 1).reshape(self.block, -1)

    def draft_window(self, anchor_token, anchor_position, keys, values, start, block_at, mask):
        block = self.block
        tokens = torch.full((block,), self.mask, dtype=torch.long, device=self.device)
        tokens[0] = anchor_token
        h = self.embed[tokens].float()
        positions = torch.arange(anchor_position, anchor_position + block, device=self.device)
        for index, layer in enumerate(self.layers):
            q, kb, vb = self.pre_attention(layer, h, positions)
            h = self.post_attention(layer, h, self.attend_window(index, q, kb, vb, keys, values, start, block_at, mask))
        y = rms(h, self.norm, self.eps).to(torch.bfloat16)
        return y, (y @ self.head.t()).float()

    def heads32(self):
        if not hasattr(self, "_heads32"):
            self._heads32 = (self.markov_w2.float() if self.markov_w2 is not None else None,
                             (self.confidence[0].float(), self.confidence[1].float()) if self.confidence is not None else None)
        return self._heads32

    def sequential(self, y, base, anchor_token, slots):
        drafts, confidence = [], []
        markov_w2, confidence_head = self.heads32()
        previous = int(anchor_token)
        for slot in slots:
            logits = base[slot]
            latent = None
            if self.markov_w1 is not None:
                latent = self.markov_w1[previous].float()
                logits = logits + (markov_w2 @ latent)
            token = int(torch.argmax(logits))
            if confidence_head is not None:
                feature = torch.cat([y[slot].float(), latent]) if latent is not None else y[slot].float()
                confidence.append(float(torch.sigmoid(confidence_head[0] @ feature + confidence_head[1])))
            drafts.append(token)
            previous = token
        return drafts, confidence

    def walk(self, y, base, previous, count, sampling=None, position=None, sampled=None):
        markov_w2, confidence_head = self.heads32()
        tokens, scores = [], []
        for slot in range(count):
            logits = base[slot]
            latent = None
            if self.markov_w1 is not None:
                latent = torch.index_select(self.markov_w1, 0, previous.view(1))[0].float()
                logits = logits + (markov_w2 @ latent)
            token = torch.argmax(logits)
            if sampling is not None:
                token = torch.where(sampled, coupled_choice(logits, sampling, position + slot), token)
            if confidence_head is not None:
                feature = torch.cat([y[slot].float(), latent]) if latent is not None else y[slot].float()
                scores.append(torch.sigmoid(confidence_head[0] @ feature + confidence_head[1]).reshape(1))
            tokens.append(token.reshape(1))
            previous = token
        return torch.cat(tokens), (torch.cat(scores) if scores else None)

    def sequential_device(self, y, base, anchor_token, count):
        tokens, scores = self.walk(y, base, torch.tensor(int(anchor_token), device=self.device), count)
        return tokens.tolist(), (scores.tolist() if scores is not None else [])


def open_drafter(directory, shard, names, device):
    from latent_block_drafter import LatentBlockDrafter, is_latent_checkpoint
    return (LatentBlockDrafter if is_latent_checkpoint(directory) else BlockDrafter)(directory, shard, names, device)


class DraftGraph:
    def __init__(self, drafter):
        d = self.drafter = drafter
        self.host = torch.zeros(5, dtype=torch.long, pin_memory=True)
        self.inputs = torch.zeros(5, dtype=torch.long, device=d.device)
        self.host_rates = torch.zeros(2, dtype=torch.float32, pin_memory=True)
        self.rates = torch.zeros(2, dtype=torch.float32, device=d.device)
        self.slots = torch.arange(d.block, device=d.device)
        self.attended = [torch.zeros((d.block, d.heads * d.head_dim), dtype=torch.bfloat16, device=d.device) for _ in d.layers]
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs = []
        self.staged = []
        self.outputs = None
        for capturing in (False, True):
            self.build(capturing)

    def segment(self, index):
        d = self.drafter
        if index == 0:
            tokens = torch.where(self.slots == 0, self.inputs[0], d.mask)
            self.positions = self.inputs[1] + self.slots
            h = d.embed[tokens].float()
        else:
            h = d.post_attention(d.layers[index - 1], self.hidden, self.attended[index - 1])
        self.hidden = h
        if index < len(d.layers):
            return d.pre_attention(d.layers[index], h, self.positions)
        y = rms(h, d.norm, d.eps).to(torch.bfloat16)
        sampling = (self.inputs[2], self.rates[0], self.rates[1], self.inputs[3])
        return d.walk(y, (y @ d.head.t()).float(), self.inputs[0], d.block, sampling, self.inputs[1], self.inputs[4] != 0)

    def build(self, capturing):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graphs, staged = [], []
        with torch.cuda.stream(stream), torch.no_grad():
            for index in range(len(self.drafter.layers) + 1):
                if capturing:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=self.pool, stream=stream):
                        staged.append(self.segment(index))
                    graphs.append(graph)
                else:
                    staged.append(self.segment(index))
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        if capturing:
            self.graphs, self.staged = graphs, staged

    def draft(self, anchor_token, anchor_position, keys, values, start, block_at, valid, count, sampling=None):
        d = self.drafter
        seed, inverse_temperature, nucleus, top_k = sampling if sampling is not None else (0, 0.0, 0.0, 0)
        self.host[0], self.host[1] = int(anchor_token), int(anchor_position)
        self.host[2] = seed - (1 << 64) if seed >= (1 << 63) else seed
        self.host[3], self.host[4] = int(top_k), 1 if inverse_temperature > 0.0 else 0
        self.host_rates[0], self.host_rates[1] = float(inverse_temperature), float(nucleus)
        self.inputs.copy_(self.host, non_blocking=True)
        self.rates.copy_(self.host_rates, non_blocking=True)
        mask = None if valid is None else torch.from_numpy(valid).to(d.device).view(1, 1, 1, -1)
        with torch.no_grad():
            for index in range(len(d.layers)):
                self.graphs[index].replay()
                q, kb, vb = self.staged[index]
                self.attended[index].copy_(d.attend_window(index, q, kb, vb, keys, values, start, block_at, mask))
            self.graphs[-1].replay()
            tokens, scores = self.staged[-1]
            tokens = tokens[:count].tolist()
        return tokens, (scores[:count].tolist() if scores is not None else [])


def requests(paths, entry_names):
    streams = defaultdict(lambda: {"positions": {}, "tail": {}})
    hello = None
    for path in paths:
        for hello, frame in frames(path):
            key = (frame["control_generation"], frame["request_id"], frame["sequence_id"])
            item = streams[key]
            for k in range(frame["rows"]):
                item["positions"][frame["first_position"] + k] = (path, frame["offset"] + k * frame["row_bytes"], frame["tokens"][k], frame["flags"])
            item["tail"][frame["first_position"] + frame["rows"]] = frame["tail_token"]
    return hello, streams


def load_rows(hello, records, columns):
    hidden = hello["hidden"]
    count = len(hello["entries"])
    out = np.empty((len(records), len(columns), hidden), dtype=np.uint16)
    handles = {}
    for index, (path, offset, _, _) in enumerate(records):
        if path not in handles:
            handles[path] = np.memmap(path, dtype=np.uint16, mode="r")
        row = handles[path][offset // 2: offset // 2 + count * hidden].reshape(count, hidden)
        out[index] = row[columns]
    return torch.from_numpy(out).view(torch.bfloat16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafter", required=True)
    parser.add_argument("--target-shard", required=True)
    parser.add_argument("--embed", default="language_model.model.embed_tokens.weight")
    parser.add_argument("--head", default="language_model.lm_head.weight")
    parser.add_argument("--norm", default="language_model.model.norm.weight")
    parser.add_argument("--taps", nargs="+", required=True)
    parser.add_argument("--delta", type=int, default=0)
    parser.add_argument("--convention", choices=("shift", "inplace"), help="defaults to the drafter's own")
    parser.add_argument("--max-anchors", type=int, default=400, help="per request, spread over the reply")
    parser.add_argument("--out", required=True)
    parser.add_argument("--anchors-out", help="JSON lines, one per anchor: request, position, accepted, drafts, confidence")
    args = parser.parse_args()
    device = "cuda"
    drafter = open_drafter(args.drafter, args.target_shard, {"embed": args.embed, "head": args.head, "norm": args.norm}, device)
    args.convention = args.convention or drafter.convention
    hello, streams = requests(args.taps, None)
    wanted = [f"o{layer + args.delta}" for layer in drafter.layer_ids]
    missing = [name for name in wanted + ["final"] if name not in hello["entries"]]
    if missing:
        raise SystemExit(f"tap stream lacks {missing}; it has {hello['entries']}")
    columns = [hello["entries"].index(name) for name in wanted]
    final_column = hello["entries"].index("final")
    slots = list(range(drafter.block)) if args.convention == "shift" else list(range(1, drafter.block))
    report = {"arm": {"delta": args.delta, "convention": args.convention, "layers": wanted}, "requests": []}
    totals = defaultdict(int)
    slot_hits = [0] * len(slots)
    slot_tries = [0] * len(slots)
    calibration = defaultdict(lambda: [0, 0])
    head_agree = head_total = 0
    reply_agree = reply_total = 0
    anchor_log = open(args.anchors_out, "w") if args.anchors_out else None
    for key, item in sorted(streams.items()):
        end = max(item["positions"]) + 1
        start = end - 1
        while start - 1 in item["positions"]:
            start -= 1
        records = [item["positions"][p] for p in range(start, end)]
        tokens = torch.tensor([record[2] for record in records], device=device)
        flags = [record[3] for record in records]
        reply = [index for index, flag in enumerate(flags) if flag & 0x6]
        if not reply:
            continue
        taps = load_rows(hello, records, columns).to(device)
        final = load_rows(hello, records, [final_column]).to(device)[:, 0]
        pos = torch.arange(start, end, device=device)
        keys, values = drafter.context(taps.reshape(len(records), -1), pos)
        with torch.no_grad():
            predicted = torch.argmax((rms(final, drafter.target_norm, drafter.eps).to(torch.bfloat16) @ drafter.head.t()).float(), -1)
        agree = (predicted[:-1] == tokens[1:]).sum().item()
        head_agree += agree
        head_total += len(records) - 1
        first = reply[0]
        reply_hits = (predicted[first - 1:-1] == tokens[first:]).sum().item() if first > 0 else 0
        reply_count = len(records) - first if first > 0 else 0
        reply_agree += reply_hits
        reply_total += reply_count
        anchors = list(range(first, len(records)))
        if len(anchors) > args.max_anchors:
            step = len(anchors) / args.max_anchors
            anchors = [anchors[int(i * step)] for i in range(args.max_anchors)]
        lengths = []
        for anchor in anchors:
            y, base = drafter.draft(int(tokens[anchor]), start + anchor, keys, values, anchor)
            drafts, confidence = drafter.sequential(y, base, int(tokens[anchor]), slots)
            truth = tokens[anchor + 1: anchor + 1 + len(drafts)].tolist()
            accepted = 0
            for index, token in enumerate(drafts):
                if index >= len(truth):
                    break
                slot_tries[index] += 1
                hit = token == truth[index]
                if confidence:
                    calibration[min(9, int(confidence[index] * 10))][0] += 1
                    calibration[min(9, int(confidence[index] * 10))][1] += int(hit)
                if not hit:
                    break
                slot_hits[index] += 1
                accepted += 1
            lengths.append(accepted)
            if anchor_log:
                anchor_log.write(json.dumps({"request": key[1], "position": start + anchor, "reply": anchor >= first, "accepted": accepted,
                                             "drafts": drafts, "confidence": [round(c, 4) for c in confidence]}) + "\n")
        totals["anchors"] += len(lengths)
        totals["accepted"] += sum(lengths)
        report["requests"].append({"request_id": key[1], "positions": len(records), "reply": len(records) - first, "anchors": len(lengths),
                                   "tokens_per_round": 1 + sum(lengths) / max(1, len(lengths)),
                                   "histogram": np.bincount(lengths, minlength=len(slots) + 1).tolist(), "final_head_agreement": agree / max(1, len(records) - 1),
                                   "reply_head_agreement": reply_hits / max(1, reply_count)})
        print(f"request {key[1]} positions {len(records)} reply {reply_count} anchors {len(lengths)} tokens/round {1 + sum(lengths) / max(1, len(lengths)):.2f}"
              f" head {agree / max(1, len(records) - 1):.4f} reply-head {reply_hits / max(1, reply_count):.4f}", flush=True)
    report["tokens_per_round"] = 1 + totals["accepted"] / max(1, totals["anchors"])
    report["slot_conditional"] = [hits / tries if tries else None for hits, tries in zip(slot_hits, slot_tries)]
    report["final_head_agreement"] = head_agree / max(1, head_total)
    report["reply_head_agreement"] = reply_agree / max(1, reply_total)
    if anchor_log:
        anchor_log.close()
    report["confidence_bins"] = {str(b): {"n": v[0], "accepted": v[1] / v[0] if v[0] else None} for b, v in sorted(calibration.items())}
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("arm", "tokens_per_round", "slot_conditional", "final_head_agreement", "reply_head_agreement")}))


if __name__ == "__main__":
    main()
