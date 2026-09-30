import argparse
import collections
import json
import math
import os
import struct
import sys
import time

import numpy as np
import torch

HIDDEN = 0
VOCAB = 0
HEADS = 64
NOPE = 192
ROPE = 64
VDIM = 256
KV_LORA = 512
EXPERTS = 256
TOP_K = 8
ROUTED_SCALE = 2.5
EPS = 1e-5
ROPE_THETA = 8000000.0
DENSE_CONTEXT = 2048
LAYER = "model.layers.78."
TAP_MAGIC = 0x31505447
TAP_HEADER = struct.Struct("<5I")
DTYPES = {"BF16": np.uint16, "F32": np.float32, "F8_E4M3": np.uint8}


def configure(root):
    global HIDDEN, VOCAB
    config = json.load(open(os.path.join(root, "config.json")))
    HIDDEN, VOCAB = int(config["hidden_size"]), int(config["vocab_size"])


class Checkpoint:
    def __init__(self, root):
        weight_map = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
        self.entries = {}
        for shard in sorted(set(weight_map.values())):
            path = os.path.join(root, shard)
            with open(path, "rb") as fh:
                size = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(size))
            header.pop("__metadata__", None)
            for name, entry in header.items():
                self.entries[name] = (path, 8 + size, entry)

    def array(self, name):
        path, base, entry = self.entries[name]
        start, _ = entry["data_offsets"]
        return np.memmap(path, dtype=DTYPES[entry["dtype"]], mode="r", offset=base + start, shape=tuple(entry["shape"]))

    def dtype(self, name):
        return self.entries[name][2]["dtype"]


def rmsnorm(x, weight):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS) * weight).to(torch.bfloat16)


def fp8_dequant(codes, scale_inv):
    values = codes.view(torch.float8_e4m3fn).float()
    out_blocks, in_blocks = scale_inv.shape[-2], scale_inv.shape[-1]
    expanded = scale_inv.repeat_interleave(128, dim=-2).repeat_interleave(128, dim=-1)
    expanded = expanded[..., :values.shape[-2], :values.shape[-1]]
    return (values * expanded).to(torch.bfloat16)


def rope_pairs(x, positions):
    half = ROPE // 2
    inv = ROPE_THETA ** (-torch.arange(0, half, device=x.device, dtype=torch.float32) * 2.0 / ROPE)
    angle = positions.float()[:, None, None] * inv[None, None, :]
    cos, sin = torch.cos(angle), torch.sin(angle)
    shape = x.shape
    pairs = x.float().reshape(shape[0], -1, half, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.reshape(shape).to(torch.bfloat16)


class GlmFullMtp:
    def __init__(self, root, device="cuda"):
        configure(root)
        self.device = device
        ckpt = Checkpoint(root)
        self.ckpt = ckpt
        dev = torch.device(device)

        def bf16(name):
            return torch.from_numpy(np.array(ckpt.array(name))).view(torch.bfloat16).to(dev)

        def f32(name):
            return torch.from_numpy(np.array(ckpt.array(name))).to(dev).float()

        def linear(name):
            if ckpt.dtype(name + ".weight") == "F8_E4M3":
                codes = torch.from_numpy(np.array(ckpt.array(name + ".weight"))).to(dev)
                return fp8_dequant(codes, f32(name + ".weight_scale_inv"))
            return bf16(name + ".weight")

        def norm(name):
            return bf16(name).float() if ckpt.dtype(name) == "BF16" else f32(name)

        self.embed = bf16("model.embed_tokens.weight")
        self.head = bf16("lm_head.weight")
        self.enorm = norm(LAYER + "enorm.weight")
        self.hnorm = norm(LAYER + "hnorm.weight")
        self.eh_proj = linear(LAYER + "eh_proj")
        self.input_ln = norm(LAYER + "input_layernorm.weight")
        self.post_ln = norm(LAYER + "post_attention_layernorm.weight")
        self.head_norm = norm(LAYER + "shared_head.norm.weight")
        attn = LAYER + "self_attn."
        self.q_a = linear(attn + "q_a_proj")
        self.q_a_ln = norm(attn + "q_a_layernorm.weight")
        self.q_b = linear(attn + "q_b_proj")
        self.kv_a = linear(attn + "kv_a_proj_with_mqa")
        self.kv_a_ln = norm(attn + "kv_a_layernorm.weight")
        self.kv_b = linear(attn + "kv_b_proj").view(HEADS, NOPE + VDIM, KV_LORA)
        self.kv_b_f32 = self.kv_b.float()
        self.o_proj = linear(attn + "o_proj")
        mlp = LAYER + "mlp."
        self.router = f32(mlp + "gate.weight") if ckpt.dtype(mlp + "gate.weight") == "F32" else bf16(mlp + "gate.weight").float()
        self.router_bias = f32(mlp + "gate.e_score_correction_bias")
        self.shared = [linear(mlp + "shared_experts." + p) for p in ("gate_proj", "up_proj", "down_proj")]
        self.expert_codes, self.expert_scales = [], []
        for part in ("gate_proj", "up_proj", "down_proj"):
            codes = [np.array(ckpt.array(f"{mlp}experts.{e}.{part}.weight")) for e in range(EXPERTS)]
            scales = [np.array(ckpt.array(f"{mlp}experts.{e}.{part}.weight_scale_inv")) for e in range(EXPERTS)]
            self.expert_codes.append(torch.from_numpy(np.stack(codes)).to(dev))
            self.expert_scales.append(torch.from_numpy(np.stack(scales)).to(dev).float())
        self.scale = (NOPE + ROPE) ** -0.5
        self.kernels = None
        if os.environ.get("DRAFTD_GLM53FULL_TORCH_EXPERTS") is None:
            from draftd_kernels import kernels
            self.kernels = kernels()

    def expert(self, part, index):
        return fp8_dequant(self.expert_codes[part][index], self.expert_scales[part][index])

    def join(self, tokens, hidden):
        e = rmsnorm(self.embed[tokens], self.enorm)
        h = rmsnorm(hidden, self.hnorm)
        return torch.cat((e, h), dim=-1) @ self.eh_proj.t()

    def project(self, x, positions):
        a = rmsnorm(x, self.input_ln)
        q = (rmsnorm(a @ self.q_a.t(), self.q_a_ln) @ self.q_b.t()).view(-1, HEADS, NOPE + ROPE)
        q_nope, q_pe = q[..., :NOPE], rope_pairs(q[..., NOPE:], positions)
        kv = a @ self.kv_a.t()
        latent = rmsnorm(kv[:, :KV_LORA], self.kv_a_ln)
        k_pe = rope_pairs(kv[:, KV_LORA:], positions)
        return q_nope, q_pe, latent, k_pe

    def keys(self, latent):
        kv = torch.einsum("pc,hdc->phd", latent.float(), self.kv_b_f32).to(torch.bfloat16)
        return kv[..., :NOPE], kv[..., NOPE:]

    def attend(self, q_nope, q_pe, k_nope, k_pe, v, mask):
        scores = torch.einsum("rhd,khd->rhk", q_nope.float(), k_nope.float()) + torch.einsum("rhd,kd->rhk", q_pe.float(), k_pe.float())
        scores = scores * self.scale
        scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("rhk,khd->rhd", probs, v.float()).to(torch.bfloat16)
        return out.reshape(out.shape[0], HEADS * VDIM) @ self.o_proj.t()

    def moe(self, x):
        m = rmsnorm(x, self.post_ln)
        logits = m.float() @ self.router.t()
        scores = torch.sigmoid(logits)
        chosen = torch.topk(scores + self.router_bias, TOP_K, dim=-1).indices
        weights = torch.gather(scores, 1, chosen)
        weights = weights / weights.sum(-1, keepdim=True) * ROUTED_SCALE
        if self.kernels is not None:
            n = x.shape[0]
            ids = chosen.reshape(-1).contiguous()
            xin = m.float().repeat_interleave(TOP_K, dim=0).contiguous()
            g = torch.empty(n * TOP_K * 2048, dtype=torch.float32, device=x.device)
            u = torch.empty_like(g)
            self.kernels.gemv_fp8_block(self.expert_codes[0], self.expert_scales[0], ids, xin, g)
            self.kernels.gemv_fp8_block(self.expert_codes[1], self.expert_scales[1], ids, xin, u)
            act = (torch.nn.functional.silu(g) * u).view(n * TOP_K, 2048).contiguous()
            y = torch.empty(n * TOP_K * HIDDEN, dtype=torch.float32, device=x.device)
            self.kernels.gemv_fp8_block(self.expert_codes[2], self.expert_scales[2], ids, act, y)
            out = (y.view(n, TOP_K, HIDDEN) * weights[..., None]).sum(1)
            g_, u_, d_ = self.shared
            shared = (torch.nn.functional.silu((m @ g_.t()).float()) * (m @ u_.t()).float()).to(torch.bfloat16) @ d_.t()
            return (x.float() + out + shared.float()).to(torch.bfloat16)
        out = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float32, device=x.device)
        for e in torch.unique(chosen).tolist():
            rows, slot = torch.nonzero(chosen == e, as_tuple=True)
            g = m[rows] @ self.expert(0, e).t()
            u = m[rows] @ self.expert(1, e).t()
            y = (torch.nn.functional.silu(g.float()) * u.float()).to(torch.bfloat16) @ self.expert(2, e).t()
            out.index_add_(0, rows, y.float() * weights[rows, slot][:, None])
        g, u, d = self.shared
        shared = (torch.nn.functional.silu((m @ g.t()).float()) * (m @ u.t()).float()).to(torch.bfloat16) @ d.t()
        return (x.float() + out + shared.float()).to(torch.bfloat16)

    def logits_argmax(self, hidden):
        return torch.argmax((rmsnorm(hidden, self.head_norm) @ self.head.t()).float(), dim=-1)

    def logits_top(self, hidden):
        probs = torch.softmax((rmsnorm(hidden, self.head_norm) @ self.head.t()).float(), dim=-1)
        conf, token = torch.max(probs, dim=-1)
        return token, conf

    def main_head_argmax(self, normed):
        return torch.argmax((normed @ self.head.t()).float(), dim=-1)

    def cache(self, capacity):
        dev = self.device
        return {"k_nope": torch.zeros((capacity, HEADS, NOPE), dtype=torch.bfloat16, device=dev),
                "k_pe": torch.zeros((capacity, ROPE), dtype=torch.bfloat16, device=dev),
                "v": torch.zeros((capacity, HEADS, VDIM), dtype=torch.bfloat16, device=dev),
                "out": torch.zeros((capacity, HIDDEN), dtype=torch.bfloat16, device=dev),
                "pred": torch.zeros((capacity,), dtype=torch.int64, device=dev),
                "conf": torch.zeros((capacity,), dtype=torch.float32, device=dev), "length": 0}

    @torch.no_grad()
    def extend(self, cache, next_tokens, taps):
        start, n = cache["length"], taps.shape[0]
        if start + n > cache["pred"].shape[0]:
            raise IndexError(f"MTP cache rows {start}..{start + n - 1} exceed {cache['pred'].shape[0]}")
        positions = torch.arange(start, start + n, device=self.device)
        x = self.join(next_tokens, taps)
        q_nope, q_pe, latent, k_pe = self.project(x, positions)
        k_nope, v = self.keys(latent)
        cache["k_nope"][start:start + n] = k_nope
        cache["k_pe"][start:start + n] = k_pe
        cache["v"][start:start + n] = v
        total = start + n
        keys = torch.arange(total, device=self.device)
        mask = keys[None, :] <= positions[:, None]
        h = x + self.attend(q_nope, q_pe, cache["k_nope"][:total], cache["k_pe"][:total], cache["v"][:total], mask)
        out = self.moe(h)
        cache["out"][start:total] = out
        cache["pred"][start:total], cache["conf"][start:total] = self.logits_top(out)
        cache["length"] = total

    @torch.no_grad()
    def chain(self, cache, anchors, depth, min_conf=0.0):
        rows = torch.tensor([p - 1 for p in anchors], device=self.device)
        if depth <= 0:
            return [[] for _ in anchors]
        if int(rows.max()) >= cache["length"] or int(rows.min()) < 0:
            raise IndexError("an anchor needs its MTP row p-1 in the cache")
        total = cache["length"]
        keys = torch.arange(total, device=self.device)
        k_nope, k_pe, v = cache["k_nope"][:total], cache["k_pe"][:total], cache["v"][:total]
        chain_tokens = [cache["pred"][rows]]
        alive = cache["conf"][rows] >= min_conf
        lengths = alive.long()
        hidden = cache["out"][rows]
        chain_pos = rows.clone()
        own_nope, own_pe, own_v = [], [], []
        base_mask = keys[None, :] <= rows[:, None]
        for step in range(1, depth):
            if min_conf > 0.0 and not bool(alive.any()):
                break
            chain_pos = chain_pos + 1
            xs = self.join(chain_tokens[-1], hidden)
            qn, qp, lat, kp = self.project(xs, chain_pos)
            kn, vv = self.keys(lat)
            own_nope.append(kn)
            own_pe.append(kp)
            own_v.append(vv)
            scores_base = torch.einsum("rhd,khd->rhk", qn.float(), k_nope.float()) + torch.einsum("rhd,kd->rhk", qp.float(), k_pe.float())
            scores_base = (scores_base * self.scale).masked_fill(~base_mask[:, None, :], float("-inf"))
            on, op, ov = torch.stack(own_nope, dim=1), torch.stack(own_pe, dim=1), torch.stack(own_v, dim=1)
            scores_own = (torch.einsum("rhd,rshd->rhs", qn.float(), on.float()) + torch.einsum("rhd,rsd->rhs", qp.float(), op.float())) * self.scale
            probs = torch.softmax(torch.cat((scores_base, scores_own), dim=-1), dim=-1)
            att = torch.einsum("rhk,khd->rhd", probs[..., :total], v.float()) + torch.einsum("rhs,rshd->rhd", probs[..., total:], ov.float())
            att = att.to(torch.bfloat16).reshape(att.shape[0], HEADS * VDIM) @ self.o_proj.t()
            hidden = self.moe(xs + att)
            token, conf = self.logits_top(hidden)
            chain_tokens.append(token)
            alive = alive & (conf >= min_conf)
            lengths = lengths + alive.long()
        chains = torch.stack(chain_tokens, dim=1).tolist()
        if min_conf <= 0.0:
            return chains
        return [chain[:n] for chain, n in zip(chains, lengths.tolist())]

    @torch.no_grad()
    def sequence(self, tokens, taps, depth, anchors):
        cache = self.cache(taps.shape[0])
        self.extend(cache, tokens[1:taps.shape[0] + 1], taps)
        drafts = {}
        for begin in range(0, len(anchors), 256):
            part = anchors[begin:begin + 256]
            for p, chain in zip(part, self.chain(cache, part, depth)):
                drafts[p] = chain
        return cache["pred"], drafts


def read_taps(path):
    record = TAP_HEADER.size + HIDDEN * 2
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % record != 0:
        raise SystemExit(f"TAP-FAIL {path}: {raw.size} bytes is not a whole number of {record}-byte records")
    count = raw.size // record
    table = raw.reshape(count, record)
    heads = table[:, :TAP_HEADER.size].copy().view(np.uint32).reshape(count, 5)
    if np.any(heads[:, 0] != TAP_MAGIC):
        raise SystemExit(f"TAP-FAIL {path}: bad record magic")
    hidden = table[:, TAP_HEADER.size:].copy().view(np.uint16).reshape(count, HIDDEN)
    return heads, hidden


def split_streams(heads):
    streams, start = [], 0
    for index in range(1, heads.shape[0] + 1):
        if index == heads.shape[0] or heads[index, 2] == 0:
            streams.append((start, index))
            start = index
    return streams


def ordered_stream(heads, hidden, start, end):
    rows = {}
    for index in range(start, end):
        rows[int(heads[index, 2])] = index
    length = max(rows) + 1
    if sorted(rows) != list(range(length)):
        raise SystemExit(f"TAP-FAIL stream at record {start}: positions are not contiguous from 0")
    order = [rows[p] for p in range(length)]
    tokens = [int(heads[i, 3]) for i in order] + [int(heads[order[-1], 4])]
    outputs = [int(heads[i, 4]) for i in order]
    return tokens, outputs, hidden[order]


def simulate(drafts, tokens, first_anchor, last_anchor, depth):
    rounds, produced, position, accepted_at, reached_at = 0, 0, first_anchor, [0] * depth, [0] * depth
    while position <= last_anchor:
        chain = drafts.get(position, [])[:depth]
        chain = chain[:last_anchor - position + 1]
        accepted = 0
        for i, token in enumerate(chain):
            reached_at[i] += 1
            if token == tokens[position + 1 + i]:
                accepted += 1
                accepted_at[i] += 1
            else:
                break
        committed = accepted + 1
        rounds += 1
        produced += committed
        position += committed
    return rounds, produced, accepted_at, reached_at



RELAY_COMMIT = 0x31435347
RELAY_DRAFT = 0x31445347
RELAY_ANSWER = 0x31415347
RELAY_HEADER = struct.Struct("<IIQII")
RELAY_ROW = struct.Struct("<II")
RELAY_SEQUENCES = 32


class RelaySequence:
    def __init__(self, model, capacity):
        self.cache = model.cache(capacity)
        self.inputs = {}
        self.taps = torch.zeros((capacity, HIDDEN), dtype=torch.bfloat16, device=model.device)
        self.tapped = 0
        self.drafts = {}
        self.grams = {}
        self.indexed = 2

    def lookup(self, position, depth):
        inputs = self.inputs
        for end in range(self.indexed, position):
            gram = (inputs.get(end - 2), inputs.get(end - 1), inputs.get(end))
            if None in gram:
                return []
            self.grams[gram] = end
        self.indexed = max(self.indexed, position)
        end = self.grams.get((inputs.get(position - 2), inputs.get(position - 1), inputs.get(position)))
        if end is None:
            return []
        return [inputs[index] for index in range(end + 1, min(end + 1 + depth, position + 1))]


class Relay:
    def __init__(self, model, capacity, log, table=None, max_depth=7, control=""):
        self.min_conf = 0.0
        self.lookup_depth = 0
        self.table = table
        self.max_depth = max_depth
        self.control = control
        self.control_mtime = None
        self.model = model
        self.capacity = capacity
        self.sequences = collections.OrderedDict()
        self.changed = None
        self.log = log
        self.stats = {"commits": 0, "rows": 0, "drafts": 0, "draft_ms": 0.0, "extend_ms": 0.0, "tokens": 0, "lookups": 0, "waits": 0}

    def advance(self, state):
        start = state.cache["length"]
        end = start
        while end < state.tapped and (end + 1) in state.inputs:
            end += 1
        if end > start:
            next_tokens = torch.tensor([state.inputs[r + 1] for r in range(start, end)], device=self.model.device)
            self.model.extend(state.cache, next_tokens, state.taps[start:end])

    def reload(self):
        if not self.control or not os.path.exists(self.control):
            return
        mtime = os.path.getmtime(self.control)
        if mtime == self.control_mtime:
            return
        self.control_mtime = mtime
        setting = json.load(open(self.control))
        from spec_recorded_drafts import read_table
        self.table = read_table(setting["table"])["entries"] if setting.get("table") else None
        self.max_depth = int(setting.get("max_depth", 7))
        self.min_conf = float(setting.get("min_conf", 0.0))
        self.lookup_depth = int(setting.get("lookup_depth", 0))
        self.log(f"RELAY-MODE {'table ' + setting['table'] if self.table is not None else 'mtp'} max_depth={self.max_depth} min_conf={self.min_conf} lookup_depth={self.lookup_depth}")

    def commit(self, sequence, first, rows):
        self.stats["commits"] += 1
        self.stats["rows"] += len(rows)
        if first == 0:
            self.reload()
        if self.model is None:
            return
        state = self.sequences.get(sequence)
        if state is not None:
            self.sequences.move_to_end(sequence)
        if first == 0 or state is None:
            self.sequences.pop(sequence, None)
            while len(self.sequences) >= RELAY_SEQUENCES:
                self.sequences.pop(next(iter(self.sequences)))
            state = RelaySequence(self.model, self.capacity)
            self.sequences[sequence] = state
        if first != state.tapped:
            raise ValueError(f"sequence {sequence}: commit at {first} but {state.tapped} rows are tapped")
        for index, (token, hidden) in enumerate(rows):
            state.inputs[first + index] = token
        block = torch.frombuffer(bytearray(b"".join(h for _, h in rows)), dtype=torch.bfloat16).view(len(rows), HIDDEN)
        state.taps[first:first + len(rows)] = block.to(self.model.device)
        state.tapped = first + len(rows)

    def ready(self, sequence, position):
        if self.model is None or self.table is not None:
            return True
        state = self.sequences.get(sequence)
        return state is not None and state.tapped >= position

    def draft(self, sequence, position, anchor, requested):
        depth = min(requested, self.max_depth)
        if self.table is not None:
            self.stats["drafts"] += 1
            return self.table.get((sequence, position), [])[:depth]
        state = self.sequences[sequence]
        key = (position, requested)
        if key not in state.drafts:
            known = state.inputs.get(position)
            if known is not None and known != anchor:
                raise ValueError(f"sequence {sequence} position {position}: anchor {anchor} differs from committed {known}")
            state.inputs[position] = anchor
            t0 = time.perf_counter()
            self.advance(state)
            torch.cuda.synchronize()
            self.stats["extend_ms"] += (time.perf_counter() - t0) * 1000.0
            found = state.lookup(position, min(requested, self.lookup_depth)) if self.lookup_depth > 0 and position > 0 else []
            if found and found[0] == int(state.cache["pred"][position - 1]):
                self.stats["lookups"] += 1
                state.drafts[key] = found
            else:
                state.drafts[key] = self.model.chain(state.cache, [position], depth, self.min_conf)[0]
            torch.cuda.synchronize()
            self.stats["tokens"] += len(state.drafts[key])
            self.stats["drafts"] += 1
            self.stats["draft_ms"] += (time.perf_counter() - t0) * 1000.0
        return state.drafts[key]


async def relay_connection(relay, reader, writer):
    import asyncio
    peer = writer.get_extra_info("peername")
    try:
        while True:
            head = await reader.readexactly(RELAY_HEADER.size)
            magic, rank, sequence, position, count = RELAY_HEADER.unpack(head)
            if magic == RELAY_COMMIT:
                rows = []
                for _ in range(count):
                    token, _output = RELAY_ROW.unpack(await reader.readexactly(RELAY_ROW.size))
                    rows.append((token, await reader.readexactly(HIDDEN * 2)))
                relay.commit(sequence, position, rows)
                async with relay.changed:
                    relay.changed.notify_all()
            elif magic == RELAY_DRAFT:
                anchor, depth = struct.unpack("<II", await reader.readexactly(8))
                if not relay.ready(sequence, position):
                    relay.stats["waits"] += 1
                    async with relay.changed:
                        await relay.changed.wait_for(lambda: relay.ready(sequence, position))
                tokens = relay.draft(sequence, position, anchor, count)
                writer.write(struct.pack(f"<II{len(tokens)}I", RELAY_ANSWER, len(tokens), *tokens))
                await writer.drain()
            else:
                raise ValueError(f"bad relay magic {magic:#x} from {peer}")
    except asyncio.IncompleteReadError:
        pass
    except Exception as error:
        relay.log(f"RELAY-ERROR {peer}: {error!r}")
    finally:
        writer.close()


def serve(args):
    import asyncio

    def log(text):
        print(time.strftime("%H:%M:%SZ", time.gmtime()), text, flush=True)

    table = None
    if args.table:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from spec_recorded_drafts import read_table
        table = read_table(args.table)["entries"]
        log(f"RELAY-TABLE {args.table} entries={len(table)}")
    configure(args.checkpoint)
    model = GlmFullMtp(args.checkpoint)
    relay = Relay(model, args.capacity, log, table, args.max_depth, args.control)
    relay.reload()

    async def run():
        relay.changed = asyncio.Condition()
        server = await asyncio.start_server(lambda r, w: relay_connection(relay, r, w), args.host, args.port)
        for sock in server.sockets:
            sock.setsockopt(__import__("socket").IPPROTO_TCP, __import__("socket").TCP_NODELAY, 1)
        log(f"RELAY-READY {args.host}:{args.port} capacity={args.capacity} mode={'table' if table is not None else 'mtp'} max_depth={args.max_depth}")

        async def report():
            while True:
                await asyncio.sleep(30)
                st = relay.stats
                log(f"RELAY-STATS commits={st['commits']} rows={st['rows']} drafts={st['drafts']} waits={st['waits']} draft_ms_mean={st['draft_ms'] / max(1, st['drafts']):.2f} extend_ms_mean={st['extend_ms'] / max(1, st['drafts']):.2f} tokens_mean={st['tokens'] / max(1, st['drafts']):.2f} lookups={st['lookups']}")

        asyncio.ensure_future(report())
        async with server:
            await server.serve_forever()

    asyncio.run(run())


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        serving = argparse.ArgumentParser(description="GLM-5.3 Full MTP draft relay for the glm52 verify (every rank asks, rank 0 commits taps)")
        serving.add_argument("serve")
        serving.add_argument("--checkpoint", required=True, help="the MTP checkpoint; its config.json also gives the hidden width of the committed taps")
        serving.add_argument("--table", default="", help="answer from a recorded draft table instead of the MTP layer")
        serving.add_argument("--max-depth", type=int, default=7, help="longest chain returned; 0 answers no drafts (plain decode through the same frames)")
        serving.add_argument("--control", default="", help="json {table, max_depth, min_conf, lookup_depth} re-read when a sequence starts; switches arms without dropping the fleet connections; min_conf ends a chain at the first MTP token whose probability is below it; lookup_depth > 0 answers with the continuation of the latest earlier occurrence of the last three tokens when its first token equals the MTP draft")
        serving.add_argument("--host", default="0.0.0.0")
        serving.add_argument("--port", type=int, required=True)
        serving.add_argument("--capacity", type=int, default=DENSE_CONTEXT)
        serve(serving.parse_args())
        return
    parser = argparse.ArgumentParser(description="GLM-5.3 Full MTP (layer 78) drafts from glm52 final-norm taps: acceptance per class and a recorded draft table")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--taps", required=True)
    parser.add_argument("--results", required=True, help="window_runner results.jsonl of the no-spec run that wrote the taps")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--classes", default="", help="json map label -> class")
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--table", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from spec_recorded_drafts import write_table
    record = None
    for line in open(args.results):
        item = json.loads(line)
        if item["tag"] == args.tag and item["session"] == args.session and item["rc"] == 0:
            record = item
    if record is None:
        raise SystemExit(f"no rc=0 {args.tag} {args.session} in {args.results}")
    classes = json.loads(open(args.classes).read()) if args.classes else {}
    heads, hidden_all = read_taps(args.taps)
    spans = split_streams(heads)
    requests = record["per_request"]
    if len(spans) != len(requests):
        raise SystemExit(f"TAP-FAIL {len(spans)} tap streams for {len(requests)} requests")
    t0 = time.time()
    model = GlmFullMtp(args.checkpoint)
    load_s = time.time() - t0
    entries, report = {}, {"load_s": load_s, "streams": [], "classes": {}}
    for (start, end), request in zip(spans, requests):
        tokens, outputs, taps = ordered_stream(heads, hidden_all, start, end)
        prompt = request["prompt_len"]
        generated = request["token_ids"]
        if tokens[prompt:] != generated[:len(tokens) - prompt] or len(tokens) - prompt != len(generated):
            raise SystemExit(f"TAP-FAIL {request['label']}: tapped stream does not match the served tokens")
        if len(tokens) > DENSE_CONTEXT:
            raise SystemExit(f"{request['label']}: {len(tokens)} tokens; this drafter runs dense MTP attention only up to {DENSE_CONTEXT}")
        device_taps = torch.from_numpy(np.ascontiguousarray(taps).view(np.int16)).view(torch.bfloat16).cuda()
        check = model.main_head_argmax(device_taps[prompt - 1:]).tolist()
        head_agree = sum(int(a == b) for a, b in zip(check, outputs[prompt - 1:]))
        anchors = list(range(prompt, len(tokens) - 1))
        t1 = time.time()
        first, drafts = model.sequence(torch.tensor(tokens, device="cuda"), device_taps[:len(tokens) - 1], args.depth, anchors)
        torch.cuda.synchronize()
        elapsed = time.time() - t1
        sequence_id = request["request_id"]
        for p, chain in drafts.items():
            entries[(sequence_id, p)] = chain
        label = request["label"]
        klass = classes.get(label, label.rsplit("-", 1)[0])
        stream = {"label": label, "class": klass, "sequence_id": sequence_id, "prompt": prompt, "generated": len(generated),
                  "head_agree": f"{head_agree}/{len(check)}", "mtp_s": elapsed, "anchors": len(anchors), "by_depth": {}}
        for k in range(1, args.depth + 1):
            rounds, produced, accepted_at, reached_at = simulate(drafts, tokens, prompt, len(tokens) - 2, k)
            stream["by_depth"][k] = {"rounds": rounds, "tokens": produced, "tokens_per_round": produced / rounds if rounds else 0.0,
                                     "p_accept": [accepted_at[i] / reached_at[i] if reached_at[i] else None for i in range(k)]}
        agg = report["classes"].setdefault(klass, {k: [0, 0] for k in range(1, args.depth + 1)})
        for k in range(1, args.depth + 1):
            agg[k][0] += stream["by_depth"][k]["rounds"]
            agg[k][1] += stream["by_depth"][k]["tokens"]
        report["streams"].append(stream)
        print(json.dumps({"label": label, "class": klass, "head_agree": stream["head_agree"], "mtp_s": round(elapsed, 2),
                          "p1": stream["by_depth"][1]["p_accept"][0], "tpr": {k: round(stream["by_depth"][k]["tokens_per_round"], 3) for k in (1, 3, 5, 7) if k <= args.depth}}), flush=True)
    for klass, agg in report["classes"].items():
        report["classes"][klass] = {k: {"rounds": r, "tokens": t, "tokens_per_round": t / r if r else 0.0} for k, (r, t) in agg.items()}
        print(json.dumps({"class": klass, "tokens_per_round": {k: round(v["tokens_per_round"], 3) for k, v in report["classes"][klass].items()}}))
    count = write_table(args.table, entries, args.depth, VOCAB)
    report["table"] = {"path": args.table, "entries": count, "depth": args.depth}
    json.dump(report, open(args.report, "w"), indent=1)
    print(f"MTP-TABLE {args.table} entries={count} depth={args.depth}")


if __name__ == "__main__":
    main()
