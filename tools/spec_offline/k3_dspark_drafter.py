"""Kimi K3.1 Pro's DSpark drafter (K3DSparkModel) on the hub, offline and in draftd.

Five MLA layers over five target taps, like the latent block drafter, with no block convolutions. Attention is causal:
a block row sees the context and the block rows up to its own. The block has seven rows and every row drafts one token,
starting from the anchor. Each row adds a low-rank Markov term from the token before it, and a confidence head reads the
row and that token's Markov vector. The embedding and the head are the target's.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from block_drafter import SafeTensors, rms  # noqa: E402
from latent_block_drafter import LATENT_ATTENTION_CHUNK, DrafterContractError, LatentBlockDrafter, coupled_candidates, latent_chunks, power_of_two, rope_tables, rotate_pairs  # noqa: E402

LOG2E = 1.4426950408889634
SAMPLED_CANDIDATES = 32
MARKOV_CANDIDATES = 2048


def refuse_unsupported(config):
    checks = (
        (config.get("architectures") == ["K3DSparkModel"], "architectures must be [K3DSparkModel]"),
        (config.get("full_attention_causal") is True, "full_attention_causal must be true"),
        (config.get("sample_from_anchor") is True, "sample_from_anchor must be true"),
        (config.get("enable_confidence_head") is True and config.get("confidence_head_with_markov") is True, "the confidence head must read the Markov vector"),
        (config.get("markov_head_type") == "vanilla", "markov_head_type must be vanilla"),
        (config.get("use_sliding_window") is False, "the context must be unwindowed"),
        (set(config.get("layer_types", [])) == {"full_attention"}, "every layer must be full_attention"),
        (len(config.get("target_layer_ids", [])) == config.get("num_target_layers"), "target_layer_ids must name every tap"),
    )
    failed = [message for ok, message in checks if not ok]
    if failed:
        raise DrafterContractError("K3 DSpark drafter refuses this checkpoint: " + "; ".join(failed))


def causal_block_attention(absorbed, query_rope, block_rows, kv_rank, scale, carried=None):
    block = block_rows.float()
    scores = (torch.einsum("ihr,pr->ihp", absorbed, block[:, :kv_rank]) + torch.einsum("ihd,pd->ihp", query_rope, block[:, kv_rank:])) * scale * LOG2E
    rows = scores.shape[0]
    allowed = torch.tril(torch.ones(rows, block.shape[0], dtype=torch.bool, device=scores.device))[:, None, :]
    scores = torch.where(allowed, scores, torch.full_like(scores, float("-inf")))
    peak = scores.amax(-1)
    if carried is None:
        weights = torch.exp2(scores - peak[..., None])
        return torch.einsum("ihp,pr->ihr", weights, block[:, :kv_rank]) / weights.sum(-1, keepdim=True)
    accumulated, best, total = carried
    new = torch.maximum(best, peak)
    keep = torch.exp2(best - new)
    weights = torch.exp2(scores - new[..., None])
    merged = accumulated * keep[..., None] + torch.einsum("ihp,pr->ihr", weights, block[:, :kv_rank])
    return merged / (total * keep + weights.sum(-1))[..., None]


def context_partials(absorbed, query_rope, rows, kv_rank, scale, chunk=LATENT_ATTENTION_CHUNK):
    best = total = accumulated = None
    for part in latent_chunks(rows, chunk):
        part = part.float()
        scores = (torch.einsum("ihr,pr->ihp", absorbed, part[:, :kv_rank]) + torch.einsum("ihd,pd->ihp", query_rope, part[:, kv_rank:])) * scale * LOG2E
        peak = scores.amax(-1)
        if best is None:
            weights = torch.exp2(scores - peak[..., None])
            total, accumulated, best = weights.sum(-1), torch.einsum("ihp,pr->ihr", weights, part[:, :kv_rank]), peak
            continue
        new = torch.maximum(best, peak)
        keep = torch.exp2(best - new)
        weights = torch.exp2(scores - new[..., None])
        total = total * keep + weights.sum(-1)
        accumulated = accumulated * keep[..., None] + torch.einsum("ihp,pr->ihr", weights, part[:, :kv_rank])
        best = new
    return None if best is None else (accumulated, best, total)


class K3DsparkDrafter(LatentBlockDrafter):
    convention = "shift"

    def __init__(self, directory, shard, names, device, rope_attn_factor=True):
        self.device = device
        with open(Path(directory) / "config.json") as handle:
            self.config = c = json.load(handle)
        refuse_unsupported(c)
        weights = SafeTensors(Path(directory) / "model.safetensors")
        target = SafeTensors(shard)
        self.eps = c["rms_norm_eps"]
        self.heads, self.nope, self.rope_dim, self.value_dim = c["num_attention_heads"], c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
        self.kv_rank = c["kv_lora_rank"]
        self.block = c["block_size"]
        self.max_drafts = self.block
        self.layer_ids = list(c["target_layer_ids"])
        self.mask = c["mask_token_id"]
        self.windows = [None] * c["num_hidden_layers"]
        self.fc = weights.load("context_proj.weight", device)
        self.hidden_norm = weights.load("context_norm.weight", device)
        self.norm = weights.load("final_norm.weight", device)
        self.layers = []
        for index in range(c["num_hidden_layers"]):
            prefix = f"layers.{index}."
            layer = {name: weights.load(prefix + name + ".weight", device) for name in (
                "input_layernorm", "post_attention_layernorm", "self_attn.q_a_proj", "self_attn.q_a_layernorm", "self_attn.q_b_proj",
                "self_attn.kv_a_proj_with_mqa", "self_attn.kv_a_layernorm", "self_attn.kv_b_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
            kv_b = layer["self_attn.kv_b_proj"].view(self.heads, self.nope + self.value_dim, self.kv_rank)
            layer["absorb_key"] = kv_b[:, :self.nope].float().contiguous()
            layer["absorb_value"] = kv_b[:, self.nope:].float().contiguous()
            self.layers.append(layer)
        self.markov_previous = weights.load("markov_head.markov_w1.weight", device)
        self.markov_next = weights.load("markov_head.markov_w2.weight", device)
        self.confidence_weight = weights.load("confidence_head.proj.weight", device).float()
        self.confidence_bias = weights.load("confidence_head.proj.bias", device).float()
        self.embed = target.load(names["embed"], device)
        self.head = target.load(names["head"], device)
        self.target_norm = target.load(names["norm"], device)
        self.inverse, self.rotary_gain, softmax_gain = rope_tables(c, self.rope_dim, device)
        if rope_attn_factor:
            self.rotary_gain *= float(c["rope_parameters"].get("attn_factor", 1.0))
        self.scale = softmax_gain / math.sqrt(self.nope + self.rope_dim)
        self.attention, self.fused = "reference", None
        if str(device).startswith("cuda") and min(self.kv_rank, self.rope_dim) >= 16 and power_of_two(self.kv_rank) and power_of_two(self.rope_dim):
            from latent_attention_kernel import fused_latent_partials
            self.attention, self.fused = "fused", fused_latent_partials

    def pre_attention(self, layer, residual, positions):
        x = rms(residual, layer["input_layernorm"], self.eps).to(torch.bfloat16)
        query = (rms((x @ layer["self_attn.q_a_proj"].t()).float(), layer["self_attn.q_a_layernorm"], self.eps).to(torch.bfloat16) @ layer["self_attn.q_b_proj"].t()).float()
        query = query.view(x.shape[0], self.heads, self.nope + self.rope_dim)
        absorbed = torch.einsum("ihn,hnr->ihr", query[..., :self.nope].float(), layer["absorb_key"])
        query_rope = rotate_pairs(query[..., self.nope:], positions, self.inverse, self.rotary_gain)
        return absorbed, query_rope, self.latent_rows(layer, x, positions), None

    def attend(self, absorbed, query_rope, block_rows, context):
        parts = [part for part in (context if isinstance(context, (list, tuple)) else [context]) if part.shape[0] != 0]
        carried = None
        if parts and self.fused is not None:
            accumulated, best, total = self.fused(absorbed, query_rope, parts, self.kv_rank, self.scale)
            shape = (absorbed.shape[0], absorbed.shape[1])
            carried = accumulated.view(*shape, self.kv_rank), best.view(shape), total.view(shape)
        elif parts:
            carried = context_partials(absorbed, query_rope, parts, self.kv_rank, self.scale)
        return causal_block_attention(absorbed, query_rope, block_rows, self.kv_rank, self.scale, carried)

    def post_attention(self, layer, residual, latent, finish):
        values = torch.einsum("ihr,hvr->ihv", latent, layer["absorb_value"]).reshape(latent.shape[0], -1)
        residual = residual + (values.to(torch.bfloat16) @ layer["self_attn.o_proj"].t()).float()
        x = rms(residual, layer["post_attention_layernorm"], self.eps).to(torch.bfloat16)
        return residual + self.mlp(layer, x)

    def forward_block(self, anchor_token, anchor_position, contexts):
        tokens = torch.full((self.block,), self.mask, dtype=torch.long, device=self.device)
        tokens[0] = int(anchor_token)
        positions = torch.arange(int(anchor_position), int(anchor_position) + self.block, device=self.device)
        residual = self.embed[tokens].float()
        for layer, context in zip(self.layers, contexts):
            absorbed, query_rope, block_rows, finish = self.pre_attention(layer, residual, positions)
            residual = self.post_attention(layer, residual, self.attend(absorbed, query_rope, block_rows, context), finish)
        return rms(residual, self.norm, self.eps).to(torch.bfloat16)

    def propose(self, y, anchor_token, sampling=None, position=None, sampled=None):
        count = min(MARKOV_CANDIDATES, self.head.shape[0])
        unary, candidates = torch.topk((y @ self.head.t()).float(), count, -1)
        edges = torch.index_select(self.markov_next, 0, candidates.view(-1)).view(y.shape[0], count, -1).float()
        hidden = y.float()
        previous = torch.as_tensor(anchor_token, device=y.device).view(())
        tokens, scores = [], []
        for step in range(self.block):
            vector = torch.index_select(self.markov_previous, 0, previous.view(1))[0].float()
            row = unary[step] + edges[step] @ vector
            slot = torch.argmax(row)
            if sampling is not None:
                values, order = torch.topk(row, min(SAMPLED_CANDIDATES, count))
                picked = coupled_candidates(values, torch.index_select(candidates[step], 0, order), sampling, position + step)
                slot = torch.where(sampled, torch.index_select(order, 0, picked.view(1))[0], slot)
            choice = torch.index_select(candidates[step], 0, slot.view(1))[0]
            features = torch.cat((hidden[step], vector))
            scores.append(torch.sigmoid(self.confidence_weight @ features + self.confidence_bias).view(1))
            tokens.append(choice.view(1))
            previous = choice
        return torch.cat(tokens), torch.cat(scores)

    def slot_logits(self, y, slot, previous):
        vector = self.markov_previous[int(previous)].float()
        return (y[slot] @ self.head.t()).float() + vector @ self.markov_next.float().t()

    def sequential(self, y, base, anchor_token, slots):
        with torch.no_grad():
            tokens, scores = self.propose(y, int(anchor_token))
        chosen = list(slots)
        return tokens[chosen].tolist(), scores[chosen].tolist()

    def sequential_sampled(self, y, base, anchor_token, slots, sampling, position):
        with torch.no_grad():
            tokens, _ = self.propose(y, int(anchor_token), sampling, torch.tensor(int(position), device=self.device), torch.tensor(True, device=self.device))
        return tokens[list(slots)].tolist()


def is_k3_dspark_checkpoint(directory):
    with open(Path(directory) / "config.json") as handle:
        return json.load(handle).get("architectures") == ["K3DSparkModel"]
