"""Block drafter with latent attention, block convolutions and a top-k transition selector
(docs/RTX5090_SPECULATION_NODE.md, "Latent block drafter"), offline and in draftd."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

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


def latent_attention(absorbed, query_rope, rows, kv_rank, scale, chunk=LATENT_ATTENTION_CHUNK):
    best = total = accumulated = None
    for begin in range(0, rows.shape[0], chunk):
        part = rows[begin:begin + chunk].float()
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

    def attend(self, layer, x, positions, context):
        query = (rms((x @ layer["self_attn.q_a_proj"].t()).float(), layer["self_attn.q_a_layernorm"], self.eps).to(torch.bfloat16) @ layer["self_attn.q_b_proj"].t())
        query = query.view(x.shape[0], self.heads, self.nope + self.rope_dim)
        absorbed = torch.einsum("ihn,hnr->ihr", query[..., :self.nope].float(), layer["absorb_key"])
        query_rope = rotate_pairs(query[..., self.nope:], positions, self.inverse, self.rotary_gain)
        rows = torch.cat((context, self.latent_rows(layer, x, positions)), 0)
        latent = latent_attention(absorbed, query_rope, rows, self.kv_rank, self.scale)
        values = torch.einsum("ihr,hvr->ihv", latent, layer["absorb_value"]).reshape(x.shape[0], -1)
        return (values.to(torch.bfloat16) @ layer["self_attn.o_proj"].t()).float()

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
            x = rms(residual, layer["input_layernorm"], self.eps).to(torch.bfloat16)
            prepared, finish = self.conv_pair(layer, "attention_conv", x)
            attended = self.attend(layer, prepared, positions, context)
            residual = residual + block_conv(attended, layer["attention_conv.base_kernel"][1], finish, self.group)
            x = rms(residual, layer["post_attention_layernorm"], self.eps).to(torch.bfloat16)
            prepared, finish = self.conv_pair(layer, "mlp_conv", x)
            residual = residual + block_conv(self.mlp(layer, prepared), layer["mlp_conv.base_kernel"][1], finish, self.group)
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

    def sequential(self, y, base, anchor_token, slots):
        rows = y[torch.as_tensor(list(slots), device=y.device)]
        with torch.no_grad():
            tokens, scores = self.walk(*self.lattice(rows, int(anchor_token)))
        return tokens.tolist(), scores.tolist()


def is_latent_checkpoint(directory):
    with open(Path(directory) / "config.json") as handle:
        config = json.load(handle)
    return str((config.get("dflash_config") or {}).get("attention_mode", "")).lower() == "mla"
