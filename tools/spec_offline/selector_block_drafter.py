"""DFlash2 block drafter with grouped-query attention: the DFlash backbone with the two-tap block convolutions around
each sublayer and the top-k transition selector (latent_block_drafter.py has both for latent attention), offline and
in draftd.

Block rows 0..T-1 sit at positions a..a+T-1 (anchor, then mask tokens) and row j proposes the token at a+j. The block
attention is not causal; a sliding_attention layer limits row j to context positions a+j-W+1 and later. Per layer:
  x = norm(h); x', f = attention_conv.prepare(x); h += attention_conv.finish(attention(x'), f)
  x = norm(h); x', f = mlp_conv.prepare(x); h += mlp_conv.finish(mlp(x'), f)
with block_conv taps that stay inside the block. The selector keeps the head's top-k per row, scores each transition
from the previous row's candidate (the anchor for row 1) with the low-rank predecessor/successor codebooks under the
row's hidden gate, and walks the best path; a sampled walk draws with the verifier's keyed Gumbel noise.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from block_drafter import SafeTensors, rms, rotate, yarn_frequencies
from latent_block_drafter import LatentBlockDrafter, block_conv

BACKBONE = ("input_layernorm", "post_attention_layernorm", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "self_attn.q_norm", "self_attn.k_norm", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "attention_conv.kernel_projection",
            "mlp_conv.kernel_projection")


class DrafterContractError(ValueError):
    pass


def refuse_unsupported(config):
    dflash = config.get("dflash_config") or {}
    kinds = set(config.get("layer_types") or ["full_attention"])
    checks = (
        (config.get("architectures") == ["DFlash2DraftModel"], "architectures must be [DFlash2DraftModel]"),
        (str(dflash.get("attention_mode", config.get("attention_mode", "gqa"))).lower() == "gqa", "attention_mode must be gqa"),
        (not config.get("is_causal", False), "the block must be non-causal"),
        (int(dflash.get("selector_top_k", 0)) > 0 and int(dflash.get("selector_rank", 0)) > 0, "the selector needs selector_top_k and selector_rank"),
        (config["hidden_size"] % int(dflash.get("conv_group_size", 1)) == 0, "conv_group_size must divide hidden_size"),
        (1 <= int(dflash.get("conv_kernel_size", 0)) <= int(dflash.get("block_size", 0)), "conv_kernel_size must be within the block"),
        (kinds in ({"full_attention"}, {"sliding_attention"}), "layers must be all full or all sliding attention"),
        (kinds == {"full_attention"} or isinstance(config.get("sliding_window"), int), "sliding layers need sliding_window"),
        (float(config.get("partial_rotary_factor") or 1.0) == 1.0, "partial rotary is unsupported"),
    )
    failed = [message for ok, message in checks if not ok]
    if failed:
        raise DrafterContractError("grouped-query DFlash2 drafter refuses this checkpoint: " + "; ".join(failed))


def is_selector_checkpoint(directory):
    with open(Path(directory) / "config.json") as handle:
        config = json.load(handle)
    mode = str((config.get("dflash_config") or {}).get("attention_mode", config.get("attention_mode", "gqa"))).lower()
    return config.get("architectures") == ["DFlash2DraftModel"] and mode == "gqa"


class SelectorBlockDrafter:
    convention = "inplace"
    lattice = LatentBlockDrafter.lattice
    select = LatentBlockDrafter.walk
    mlp = LatentBlockDrafter.mlp
    conv_pair = LatentBlockDrafter.conv_pair

    def __init__(self, directory, shard, names, device):
        self.device = device
        with open(Path(directory) / "config.json") as handle:
            self.config = c = json.load(handle)
        refuse_unsupported(c)
        dflash = c["dflash_config"]
        weights = SafeTensors(Path(directory) / "model.safetensors")
        target = SafeTensors(shard)
        self.eps = c["rms_norm_eps"]
        self.heads, self.kv_heads, self.head_dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.block = dflash["block_size"]
        self.max_drafts = self.block - 1
        self.layer_ids = list(dflash["target_layer_ids"])
        self.mask = dflash["mask_token_id"]
        self.group = dflash["conv_group_size"]
        self.taps = dflash["conv_kernel_size"]
        self.top_k = dflash["selector_top_k"]
        self.sliding = c["sliding_window"] if set(c.get("layer_types") or []) == {"sliding_attention"} else None
        self.window = None
        self.fc = weights.load("fc.weight", device)
        self.fc32 = self.fc.float().t().contiguous()
        self.hidden_norm = weights.load("hidden_norm.weight", device)
        self.norm = weights.load("norm.weight", device)
        self.layers = []
        for index in range(c["num_hidden_layers"]):
            prefix = f"layers.{index}."
            layer = {name: weights.load(prefix + name + ".weight", device) for name in BACKBONE}
            layer["attention_conv.base_kernel"] = weights.load(prefix + "attention_conv.base_kernel", device)
            layer["mlp_conv.base_kernel"] = weights.load(prefix + "mlp_conv.base_kernel", device)
            self.layers.append(layer)
        self.selector_hidden = weights.load("candidate_selector.hidden_projection.weight", device)
        self.predecessors = weights.load("candidate_selector.predecessor_codebook", device)
        self.successors = weights.load("candidate_selector.successor_codebook", device)
        self.embed = target.load(names["embed"], device)
        self.head = target.load(names["head"], device)
        self.target_norm = target.load(names["norm"], device)
        self.inv, self.attention = yarn_frequencies(c, device)
        self.finish = None

    def context(self, taps, positions, chunk=4096):
        keys, values = [[] for _ in self.layers], [[] for _ in self.layers]
        for begin in range(0, taps.shape[0], chunk):
            part = taps[begin:begin + chunk].to(self.device)
            where = positions[begin:begin + chunk].to(self.device)
            features = rms(part.float() @ self.fc32, self.hidden_norm, self.eps).to(torch.bfloat16)
            for index, layer in enumerate(self.layers):
                k = (features @ layer["self_attn.k_proj"].t()).view(-1, self.kv_heads, self.head_dim)
                keys[index].append(rotate(rms(k, layer["self_attn.k_norm"], self.eps), where, self.inv, self.attention).to(torch.bfloat16))
                values[index].append((features @ layer["self_attn.v_proj"].t()).view(-1, self.kv_heads, self.head_dim))
        return [torch.cat(parts, 0) for parts in keys], [torch.cat(parts, 0) for parts in values]

    def pre_attention(self, layer, h, positions):
        block, heads, kv_heads, dim = self.block, self.heads, self.kv_heads, self.head_dim
        x = rms(h, layer["input_layernorm"], self.eps).to(torch.bfloat16)
        prepared, self.finish = self.conv_pair(layer, "attention_conv", x)
        q = rotate(rms((prepared @ layer["self_attn.q_proj"].t()).view(block, heads, dim), layer["self_attn.q_norm"], self.eps), positions, self.inv, self.attention)
        kb = rotate(rms((prepared @ layer["self_attn.k_proj"].t()).view(block, kv_heads, dim), layer["self_attn.k_norm"], self.eps), positions, self.inv, self.attention)
        vb = (prepared @ layer["self_attn.v_proj"].t()).view(block, kv_heads, dim)
        return q.to(torch.bfloat16).transpose(0, 1), kb.to(torch.bfloat16).transpose(0, 1), vb.transpose(0, 1)

    def post_attention(self, layer, h, attended):
        h = h + block_conv((attended @ layer["self_attn.o_proj"].t()).float(), layer["attention_conv.base_kernel"][1], self.finish, self.group)
        x = rms(h, layer["post_attention_layernorm"], self.eps).to(torch.bfloat16)
        prepared, finish = self.conv_pair(layer, "mlp_conv", x)
        return h + block_conv(self.mlp(layer, prepared), layer["mlp_conv.base_kernel"][1], finish, self.group)

    def attend_window(self, index, q, kb, vb, keys, values, start, block_at, mask):
        stop = block_at + self.block
        keys[index][:, block_at:stop] = kb
        values[index][:, block_at:stop] = vb
        attended = torch.nn.functional.scaled_dot_product_attention(q[None], keys[index][None, :, start:stop], values[index][None, :, start:stop],
                                                                    attn_mask=mask, enable_gqa=True)
        return attended[0].transpose(0, 1).reshape(self.block, -1)

    def draft(self, anchor_token, anchor_position, keys, values, count):
        tokens = torch.full((self.block,), self.mask, dtype=torch.long, device=self.device)
        tokens[0] = int(anchor_token)
        positions = torch.arange(anchor_position, anchor_position + self.block, device=self.device)
        first = 0 if self.window is None else max(0, count - self.window)
        if self.sliding is not None:
            first = max(first, count - (self.sliding - 1))
        visible = torch.ones((self.block, count - first + self.block), dtype=torch.bool, device=self.device)
        if self.sliding is not None:
            context = torch.arange(first, count, device=self.device)[None, :]
            rows = torch.arange(self.block, device=self.device)[:, None]
            visible[:, :count - first] = context >= count + rows - (self.sliding - 1)
        with torch.no_grad():
            h = self.embed[tokens].float()
            for index, layer in enumerate(self.layers):
                q, kb, vb = self.pre_attention(layer, h, positions)
                k = torch.cat((keys[index][first:count].transpose(0, 1), kb), 1)
                v = torch.cat((values[index][first:count].transpose(0, 1), vb), 1)
                attended = torch.nn.functional.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=visible[None, None], enable_gqa=True)
                h = self.post_attention(layer, h, attended[0].transpose(0, 1).reshape(self.block, -1))
            y = rms(h, self.norm, self.eps).to(torch.bfloat16)
            return y, None

    def walk(self, y, base, previous, count, sampling=None, position=None, sampled=None):
        return self.select(*self.lattice(y[1:], previous), sampling, position, sampled)

    def slot_logits(self, y, slot, previous):
        row = y[slot + 1]
        unary, candidates = torch.topk((row @ self.head.t()).float(), self.top_k)
        gate = self.predecessors[int(previous)].float() * (row @ self.selector_hidden.t()).float()
        full = torch.full((self.head.shape[0],), float("-inf"), device=self.device)
        full[candidates] = unary + self.successors[candidates].float() @ gate
        return full

    def sequential(self, y, base, anchor_token, slots):
        rows = y[torch.as_tensor(list(slots), device=y.device)]
        with torch.no_grad():
            tokens, scores = self.select(*self.lattice(rows, int(anchor_token)))
        return tokens.tolist(), scores.tolist()

    def sequential_sampled(self, y, base, anchor_token, slots, sampling, position):
        rows = y[torch.as_tensor(list(slots), device=y.device)]
        with torch.no_grad():
            tokens, _ = self.select(*self.lattice(rows, int(anchor_token)), sampling, torch.tensor(int(position), device=self.device),
                                    torch.tensor(True, device=self.device))
        return tokens.tolist()
