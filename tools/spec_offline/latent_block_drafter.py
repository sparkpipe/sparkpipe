"""Block drafter with latent attention, block convolutions and a top-k transition selector
(docs/RTX5090_SPECULATION_NODE.md, "Latent block drafter"), offline and in draftd."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from block_drafter import SafeTensors, keyed_gumbel, rms  # noqa: E402

LATENT_ATTENTION_CHUNK = 32768


class DrafterContractError(ValueError):
    pass


def refuse_unsupported(config):
    dflash = config.get("dflash_config") or {}
    checks = (
        (config.get("architectures") == ["DFlash2DraftModel"], "architectures must be [DFlash2DraftModel]"),
        (str(dflash.get("attention_mode", "")).lower() == "mla", "attention_mode must be mla"),
        (dflash.get("aux_hidden_stream", config.get("aux_hidden_stream")) in (None, "prefix"), "aux_hidden_stream must be the prefix stream"),
        (not config.get("fc_norm", False), "fc_norm must be false"),
        (not config.get("mla_use_output_gate", False), "mla_use_output_gate must be false"),
        (float(dflash.get("input_embedding_scale", config.get("input_embedding_scale", 1.0))) == 1.0, "input_embedding_scale must be 1"),
        (float(dflash.get("output_multiplier", 1.0)) == 1.0, "output_multiplier must be 1"),
        (not dflash.get("final_logit_softcapping"), "final_logit_softcapping is unsupported"),
        (not config.get("is_causal", False), "the block must be non-causal"),
        (len(config.get("layer_types", [])) == config.get("num_hidden_layers"), "layer_types must name every layer"),
        (set(config.get("layer_types", [])) <= {"sliding_attention", "full_attention"}, "layer_types must be sliding_attention or full_attention"),
        (config["hidden_size"] % int(dflash.get("conv_group_size", 1)) == 0, "conv_group_size must divide hidden_size"),
        (1 <= int(dflash.get("conv_kernel_size", 0)) <= int(dflash.get("block_size", 0)), "conv_kernel_size must be within the block"),
    )
    failed = [message for ok, message in checks if not ok]
    if failed:
        raise DrafterContractError("latent block drafter refuses this checkpoint: " + "; ".join(failed))


def yarn_mscale(scale, weight):
    return 1.0 if scale <= 1.0 or not weight else 0.1 * weight * math.log(scale) + 1.0


def rope_tables(config, dim, device):
    rope = dict(config.get("rope_parameters") or config.get("rope_scaling") or {})
    base = float(rope.get("rope_theta", config.get("rope_theta", 10000.0)))
    extrapolated = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    if str(rope.get("rope_type", "default")).lower() not in ("yarn", "deepseek_yarn"):
        return extrapolated.float().to(device), 1.0, 1.0
    factor = float(rope["factor"])
    original = float(rope["original_max_position_embeddings"])
    fast, slow = float(rope.get("beta_fast", 32)), float(rope.get("beta_slow", 1))

    def correction(rotations):
        return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))
    low, high = max(math.floor(correction(fast)), 0), min(math.ceil(correction(slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float64) - low) / (high - low), 0, 1)
    keep = 1.0 - ramp
    inverse = extrapolated / factor * (1.0 - keep) + extrapolated * keep
    rotary_gain = yarn_mscale(factor, rope.get("mscale", 1.0)) / yarn_mscale(factor, rope.get("mscale_all_dim", 0.0))
    softmax_gain = yarn_mscale(factor, rope.get("mscale_all_dim", 0.0)) ** 2
    return inverse.float().to(device), rotary_gain, softmax_gain


def rotate_pairs(x, positions, inverse, gain):
    angles = positions.to(torch.float32)[:, None] * inverse[None, :]
    shape = [angles.shape[0]] + [1] * (x.dim() - 2) + [angles.shape[1]]
    cos, sin = (angles.cos() * gain).view(shape), (angles.sin() * gain).view(shape)
    even, odd = x[..., 0::2].float(), x[..., 1::2].float()
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


def block_conv(z, base, delta, group):
    coefficients = base.float()[None] + delta.repeat_interleave(group, -1)
    out = coefficients[:, 0] * z
    for tap in range(1, base.shape[0]):
        shifted = torch.cat((torch.zeros_like(z[:tap]), z[:-tap]), 0)
        out = out + coefficients[:, tap] * shifted
    return out


def power_of_two(value):
    return value > 0 and value & (value - 1) == 0


def latent_chunks(rows, chunk):
    for part in (rows if isinstance(rows, (list, tuple)) else [rows]):
        for begin in range(0, part.shape[0], chunk):
            yield part[begin:begin + chunk]


def latent_attention(absorbed, query_rope, rows, kv_rank, scale, chunk=LATENT_ATTENTION_CHUNK):
    best = total = accumulated = None
    for part in latent_chunks(rows, chunk):
        part = part.float()
        scores = (torch.einsum("ihr,pr->ihp", absorbed, part[:, :kv_rank]) + torch.einsum("ihd,pd->ihp", query_rope, part[:, kv_rank:])) * scale
        peak = scores.amax(-1, keepdim=True)
        if best is None:
            weights = torch.exp(scores - peak)
            total, accumulated, best = weights.sum(-1, keepdim=True), torch.einsum("ihp,pr->ihr", weights, part[:, :kv_rank]), peak
            continue
        new = torch.maximum(best, peak)
        carry = torch.exp(best - new)
        weights = torch.exp(scores - new)
        total = total * carry + weights.sum(-1, keepdim=True)
        accumulated = accumulated * carry + torch.einsum("ihp,pr->ihr", weights, part[:, :kv_rank])
        best = new
    return accumulated / total


def coupled_candidates(scores, tokens, sampling, position):
    seed, inverse_temperature, nucleus, top_k = sampling
    scaled = scores * inverse_temperature
    values, order = torch.sort(scaled, descending=True)
    probabilities = torch.softmax(values, -1)
    before = torch.cumsum(probabilities, 0) - probabilities
    rank = torch.arange(values.shape[0], device=scores.device)
    keep = ((nucleus <= 0.0) | (before < nucleus)) & ((top_k <= 0) | (rank < top_k)) | (rank == 0)
    candidate_tokens = tokens[order]
    keyed = torch.where(keep, values + keyed_gumbel(seed, position, candidate_tokens), torch.full_like(values, float("-inf")))
    return torch.index_select(order, 0, torch.argmax(keyed).view(1)).view(())


class LatentBlockDrafter:
    convention = "inplace"

    def __init__(self, directory, shard, names, device):
        self.device = device
        with open(Path(directory) / "config.json") as handle:
            self.config = c = json.load(handle)
        refuse_unsupported(c)
        dflash = c["dflash_config"]
        weights = SafeTensors(Path(directory) / "model.safetensors")
        target = SafeTensors(shard)
        self.eps = c["rms_norm_eps"]
        self.heads, self.nope, self.rope_dim, self.value_dim = c["num_attention_heads"], c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
        self.kv_rank = c["kv_lora_rank"]
        self.block = dflash["block_size"]
        self.max_drafts = self.block - 1
        self.layer_ids = list(dflash["target_layer_ids"])
        self.mask = dflash["mask_token_id"]
        self.group = dflash["conv_group_size"]
        self.taps = dflash["conv_kernel_size"]
        self.top_k = dflash["selector_top_k"]
        self.windows = [c["sliding_window"] if kind == "sliding_attention" else None for kind in c["layer_types"]]
        self.fc = weights.load("fc.weight", device)
        self.hidden_norm = weights.load("hidden_norm.weight", device)
        self.norm = weights.load("norm.weight", device)
        self.layers = []
        for index in range(c["num_hidden_layers"]):
            prefix = f"layers.{index}."
            layer = {name: weights.load(prefix + name + ".weight", device) for name in (
                "input_layernorm", "post_attention_layernorm", "self_attn.q_a_proj", "self_attn.q_a_layernorm", "self_attn.q_b_proj",
                "self_attn.kv_a_proj_with_mqa", "self_attn.kv_a_layernorm", "self_attn.kv_b_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "attention_conv.kernel_projection", "mlp_conv.kernel_projection")}
            layer["attention_conv.base_kernel"] = weights.load(prefix + "attention_conv.base_kernel", device)
            layer["mlp_conv.base_kernel"] = weights.load(prefix + "mlp_conv.base_kernel", device)
            kv_b = layer["self_attn.kv_b_proj"].view(self.heads, self.nope + self.value_dim, self.kv_rank)
            layer["absorb_key"] = kv_b[:, :self.nope].float().contiguous()
            layer["absorb_value"] = kv_b[:, self.nope:].float().contiguous()
            self.layers.append(layer)
        self.selector_hidden = weights.load("candidate_selector.hidden_projection.weight", device)
        self.predecessors = weights.load("candidate_selector.predecessor_codebook", device)
        self.successors = weights.load("candidate_selector.successor_codebook", device)
        self.embed = target.load(names["embed"], device)
        self.head = target.load(names["head"], device)
        self.target_norm = target.load(names["norm"], device)
        self.inverse, self.rotary_gain, softmax_gain = rope_tables(c, self.rope_dim, device)
        self.scale = softmax_gain / math.sqrt(self.nope + self.rope_dim)
        self.attention, self.fused = "reference", None
        if str(device).startswith("cuda") and min(self.kv_rank, self.rope_dim) >= 16 and power_of_two(self.kv_rank) and power_of_two(self.rope_dim):
            from latent_attention_kernel import fused_latent_attention
            self.attention, self.fused = "fused", fused_latent_attention

    def cache_layout(self):
        return [((self.kv_rank + self.rope_dim,), window) for window in self.windows]

    def latent_rows(self, layer, x, positions):
        latent = (x @ layer["self_attn.kv_a_proj_with_mqa"].t()).float()
        compressed = rms(latent[:, :self.kv_rank], layer["self_attn.kv_a_layernorm"], self.eps)
        rotated = rotate_pairs(latent[:, self.kv_rank:], positions, self.inverse, self.rotary_gain)
        return torch.cat((compressed, rotated), -1).to(torch.bfloat16)

    def context(self, taps, positions, chunk=4096):
        rows = [[] for _ in self.layers]
        for begin in range(0, taps.shape[0], chunk):
            part = taps[begin:begin + chunk].to(self.device, torch.bfloat16)
            features = rms((part @ self.fc.t()).float(), self.hidden_norm, self.eps).to(torch.bfloat16)
            where = positions[begin:begin + chunk].to(self.device)
            for index, layer in enumerate(self.layers):
                rows[index].append(self.latent_rows(layer, features, where))
        return [torch.cat(parts, 0) for parts in rows], None

    def visible(self, index, count):
        window = self.windows[index]
        return 0 if window is None else max(0, count - (window - self.block))

    def pre_attention(self, layer, residual, positions):
        x = rms(residual, layer["input_layernorm"], self.eps).to(torch.bfloat16)
        prepared, finish = self.conv_pair(layer, "attention_conv", x)
        query = (rms((prepared @ layer["self_attn.q_a_proj"].t()).float(), layer["self_attn.q_a_layernorm"], self.eps).to(torch.bfloat16) @ layer["self_attn.q_b_proj"].t())
        query = query.view(prepared.shape[0], self.heads, self.nope + self.rope_dim)
        absorbed = torch.einsum("ihn,hnr->ihr", query[..., :self.nope].float(), layer["absorb_key"])
        query_rope = rotate_pairs(query[..., self.nope:], positions, self.inverse, self.rotary_gain)
        return absorbed, query_rope, self.latent_rows(layer, prepared, positions), finish

    def attend(self, absorbed, query_rope, block_rows, context):
        rows = list(context) + [block_rows] if isinstance(context, (list, tuple)) else [context, block_rows]
        if self.fused is not None:
            return self.fused(absorbed, query_rope, rows, self.kv_rank, self.scale)
        return latent_attention(absorbed, query_rope, rows, self.kv_rank, self.scale)

    def post_attention(self, layer, residual, latent, finish):
        values = torch.einsum("ihr,hvr->ihv", latent, layer["absorb_value"]).reshape(latent.shape[0], -1)
        attended = (values.to(torch.bfloat16) @ layer["self_attn.o_proj"].t()).float()
        residual = residual + block_conv(attended, layer["attention_conv.base_kernel"][1], finish, self.group)
        x = rms(residual, layer["post_attention_layernorm"], self.eps).to(torch.bfloat16)
        prepared, finish = self.conv_pair(layer, "mlp_conv", x)
        return residual + block_conv(self.mlp(layer, prepared), layer["mlp_conv.base_kernel"][1], finish, self.group)

    def mlp(self, layer, x):
        gate = (x @ layer["mlp.gate_proj"].t()).float()
        up = (x @ layer["mlp.up_proj"].t()).float()
        return ((torch.nn.functional.silu(gate) * up).to(torch.bfloat16) @ layer["mlp.down_proj"].t()).float()

    def conv_pair(self, layer, name, x):
        coefficients = (x @ layer[name + ".kernel_projection"].t()).float().view(x.shape[0], 2, self.taps, -1)
        return block_conv(x.float(), layer[name + ".base_kernel"][0], coefficients[:, 0], self.group).to(torch.bfloat16), coefficients[:, 1]

    def forward_block(self, anchor_token, anchor_position, contexts):
        tokens = torch.full((self.block,), self.mask, dtype=torch.long, device=self.device)
        tokens[0] = int(anchor_token)
        positions = torch.arange(int(anchor_position), int(anchor_position) + self.block, device=self.device)
        residual = self.embed[tokens].float()
        for layer, context in zip(self.layers, contexts):
            absorbed, query_rope, block_rows, finish = self.pre_attention(layer, residual, positions)
            residual = self.post_attention(layer, residual, self.attend(absorbed, query_rope, block_rows, context), finish)
        return rms(residual, self.norm, self.eps).to(torch.bfloat16)

    def draft(self, anchor_token, anchor_position, keys, values, count):
        contexts = [rows[self.visible(index, count):count] for index, rows in enumerate(keys)]
        with torch.no_grad():
            return self.forward_block(anchor_token, anchor_position, contexts), None

    def lattice(self, rows, anchor_token):
        logits = (rows @ self.head.t()).float()
        unary, candidates = torch.topk(logits, self.top_k, -1)
        total = torch.logsumexp(logits, -1)
        hidden = (rows @ self.selector_hidden.t()).float()
        anchor = torch.as_tensor(anchor_token, device=rows.device).view(1, 1).expand(1, self.top_k)
        previous = torch.cat((anchor, candidates[:-1]), 0)
        edges = unary[:, None, :] + torch.einsum("spr,sr,scr->spc", self.predecessors[previous].float(), hidden, self.successors[candidates].float())
        return candidates, unary, total, edges

    def walk(self, candidates, unary, total, edges, sampling=None, position=None, sampled=None):
        previous = torch.zeros((), dtype=torch.long, device=candidates.device)
        tail = total + torch.log1p(-torch.exp(torch.logsumexp(unary, -1) - total).clamp(max=1.0 - 1e-7))
        tokens, scores = [], []
        for step in range(candidates.shape[0]):
            row = torch.index_select(edges[step], 0, previous.view(1))[0]
            choice = torch.argmax(row)
            if sampling is not None:
                choice = torch.where(sampled, coupled_candidates(row, candidates[step], sampling, position + step), choice)
            normalizer = torch.logaddexp(torch.logsumexp(row, -1), tail[step])
            picked = choice.view(1)
            scores.append(torch.exp(torch.index_select(row, 0, picked) - normalizer))
            tokens.append(torch.index_select(candidates[step], 0, picked))
            previous = choice
        return torch.cat(tokens), torch.cat(scores)

    def propose(self, y, anchor_token, sampling=None, position=None, sampled=None):
        return self.walk(*self.lattice(y[1:], anchor_token), sampling, position, sampled)

    def sequential_sampled(self, y, base, anchor_token, slots, sampling, position):
        rows = y[torch.as_tensor(list(slots), device=y.device)]
        with torch.no_grad():
            tokens, _ = self.walk(*self.lattice(rows, int(anchor_token)), sampling, torch.tensor(int(position), device=self.device), torch.tensor(True, device=self.device))
        return tokens.tolist()

    def sequential(self, y, base, anchor_token, slots):
        rows = y[torch.as_tensor(list(slots), device=y.device)]
        with torch.no_grad():
            tokens, scores = self.walk(*self.lattice(rows, int(anchor_token)))
        return tokens.tolist(), scores.tolist()


def is_latent_checkpoint(directory):
    with open(Path(directory) / "config.json") as handle:
        config = json.load(handle)
    return str((config.get("dflash_config") or {}).get("attention_mode", "")).lower() == "mla"


class LatentDraftGraph:
    def __init__(self, drafter):
        d = self.drafter = drafter
        self.host = torch.zeros(5, dtype=torch.long, pin_memory=True)
        self.inputs = torch.zeros(5, dtype=torch.long, device=d.device)
        self.host_rates = torch.zeros(2, dtype=torch.float32, pin_memory=True)
        self.rates = torch.zeros(2, dtype=torch.float32, device=d.device)
        self.slots = torch.arange(d.block, device=d.device)
        self.latent = [torch.zeros((d.block, d.heads, d.kv_rank), dtype=torch.float32, device=d.device) for _ in d.layers]
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs, self.staged = [], []
        for capturing in (False, True):
            self.build(capturing)

    def segment(self, index):
        d = self.drafter
        if index == 0:
            tokens = torch.where(self.slots == 0, self.inputs[0], d.mask)
            self.positions = self.inputs[1] + self.slots
            residual = d.embed[tokens].float()
        else:
            residual = d.post_attention(d.layers[index - 1], self.residual, self.latent[index - 1], self.finish)
        self.residual = residual
        if index < len(d.layers):
            absorbed, query_rope, block_rows, self.finish = d.pre_attention(d.layers[index], residual, self.positions)
            return absorbed, query_rope, block_rows
        y = rms(residual, d.norm, d.eps).to(torch.bfloat16)
        return d.propose(y, self.inputs[0], (self.inputs[2], self.rates[0], self.rates[1], self.inputs[3]), self.inputs[1], self.inputs[4] != 0)

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

    def draft(self, anchor_token, anchor_position, contexts, count, sampling=None):
        d = self.drafter
        seed, inverse_temperature, nucleus, top_k = sampling if sampling is not None else (0, 0.0, 0.0, 0)
        self.host[0], self.host[1] = int(anchor_token), int(anchor_position)
        self.host[2] = seed - (1 << 64) if seed >= (1 << 63) else seed
        self.host[3], self.host[4] = int(top_k), 1 if inverse_temperature > 0.0 else 0
        self.host_rates[0], self.host_rates[1] = float(inverse_temperature), float(nucleus)
        self.inputs.copy_(self.host, non_blocking=True)
        self.rates.copy_(self.host_rates, non_blocking=True)
        with torch.no_grad():
            for index in range(len(d.layers)):
                self.graphs[index].replay()
                absorbed, query_rope, block_rows = self.staged[index]
                self.latent[index].copy_(d.attend(absorbed, query_rope, block_rows, contexts[index]))
            self.graphs[-1].replay()
            tokens, scores = self.staged[-1]
            return tokens[:count].tolist(), scores[:count].tolist()


class LatentContext:
    page_positions = 65536

    def __init__(self, drafter, window=None):
        d = self.drafter = drafter
        self.width = d.kv_rank + d.rope_dim
        windows = {w for w in d.windows if w is not None}
        if len(windows) > 1:
            raise DrafterContractError("latent draftd context needs one sliding window for all sliding layers")
        self.window = windows.pop() if windows else None
        self.sliding = [index for index, w in enumerate(d.windows) if w is not None]
        self.full = [index for index, w in enumerate(d.windows) if w is None]
        self.rings = {index: torch.zeros((self.window, self.width), dtype=torch.bfloat16, device=d.device) for index in self.sliding}
        self.ring_positions = np.full(self.window or 1, -1, dtype=np.int64)
        self.pages = {index: [] for index in self.full}
        self.held = np.zeros(0, dtype=bool)
        self.tokens = np.zeros(0, dtype=np.uint32)
        self.verified = 0
        self.used = time.time()

    def grow(self, end):
        if end <= self.held.shape[0]:
            return
        size = -(-end // self.page_positions) * self.page_positions
        self.held = np.concatenate((self.held, np.zeros(size - self.held.shape[0], dtype=bool)))
        self.tokens = np.concatenate((self.tokens, np.zeros(size - self.tokens.shape[0], dtype=np.uint32)))
        for pages in self.pages.values():
            while len(pages) * self.page_positions < size:
                pages.append(torch.zeros((self.page_positions, self.width), dtype=torch.bfloat16, device=self.drafter.device))

    def write(self, first, tokens, taps):
        end = first + len(tokens)
        positions = np.arange(first, end)
        with torch.no_grad():
            rows, _ = self.drafter.context(taps, torch.from_numpy(positions).to(self.drafter.device))
        self.grow(end)
        self.held[first:] = False
        self.verified = min(self.verified, first)
        if self.window is not None:
            self.ring_positions[self.ring_positions >= first] = -1
            keep = positions[-self.window:]
            slots = torch.from_numpy(keep % self.window).to(self.drafter.device)
            for index in self.sliding:
                self.rings[index].index_copy_(0, slots, rows[index][-len(keep):])
            self.ring_positions[keep % self.window] = keep
        for index in self.full:
            for page in range(first // self.page_positions, (end - 1) // self.page_positions + 1):
                lo, hi = max(first, page * self.page_positions), min(end, (page + 1) * self.page_positions)
                self.pages[index][page][lo - page * self.page_positions:hi - page * self.page_positions] = rows[index][lo - first:hi - first]
        self.held[first:end] = True
        self.tokens[first:end] = tokens
        self.used = time.time()

    def ring_rows(self, position):
        begin = max(0, position - (self.window - self.drafter.block))
        positions = np.arange(begin, position)
        held = self.ring_positions[positions % self.window] == positions
        index = torch.from_numpy((positions[held] % self.window).astype(np.int64)).to(self.drafter.device)
        return index, bool(held.all())

    def full_rows(self, index, position):
        if self.held[:position].all():
            pages = self.pages[index]
            return [pages[page][:min(self.page_positions, position - page * self.page_positions)] for page in range(-(-position // self.page_positions))], True
        held = np.flatnonzero(self.held[:position])
        rows = [self.pages[index][page][torch.from_numpy(held[(held >= page * self.page_positions) & (held < (page + 1) * self.page_positions)] - page * self.page_positions).to(self.drafter.device)]
                for page in range(-(-position // self.page_positions))]
        return [part for part in rows if part.shape[0] != 0], False

    def contexts(self, position):
        complete = True
        out = [None] * len(self.drafter.layers)
        if self.window is not None:
            ring_index, ring_complete = self.ring_rows(position)
            complete &= ring_complete
            for index in self.sliding:
                out[index] = self.rings[index].index_select(0, ring_index)
        for index in self.full:
            out[index], full_complete = self.full_rows(index, position)
            complete &= full_complete
        return out, complete

    def matches(self, committed, position):
        if position <= self.verified:
            return True
        if position > self.tokens.shape[0]:
            return False
        held = self.held[self.verified:position]
        if not np.array_equal(self.tokens[self.verified:position][held], committed[self.verified:position][held]):
            return False
        if held.all():
            self.verified = position
        return True

    def complete(self, position):
        return position <= self.held.shape[0] and bool(self.held[:position].all())

    def covers(self, position):
        return position == 0 or (position - 1 < self.held.shape[0] and bool(self.held[position - 1]))

    def copy_prefix(self, donor, committed, position):
        shared = min(position, donor.held.shape[0])
        same = np.flatnonzero(donor.tokens[:shared] != committed[:shared])
        prefix = int(same[0]) if same.size else shared
        if prefix == 0:
            return 0
        self.grow(prefix)
        usable = np.flatnonzero(donor.held[:prefix] & ~self.held[:prefix])
        if usable.size == 0:
            return 0
        for index in self.full:
            for page in range(-(-prefix // self.page_positions)):
                lo, hi = page * self.page_positions, min(prefix, (page + 1) * self.page_positions)
                self.pages[index][page][:hi - lo].copy_(donor.pages[index][page][:hi - lo])
        if self.window is not None:
            ring = usable[usable >= max(0, prefix - self.window)]
            slots = ring % self.window
            donor_held = donor.ring_positions[slots] == ring
            slots, ring = slots[donor_held], ring[donor_held]
            index = torch.from_numpy(slots.astype(np.int64)).to(self.drafter.device)
            for layer in self.sliding:
                self.rings[layer].index_copy_(0, index, donor.rings[layer].index_select(0, index))
            self.ring_positions[slots] = ring
        self.held[usable] = donor.held[usable]
        self.tokens[:prefix] = donor.tokens[:prefix]
        return int(usable.size)

    def bytes(self):
        return sum(t.numel() * t.element_size() for t in self.rings.values()) + sum(p.numel() * p.element_size() for pages in self.pages.values() for p in pages)

