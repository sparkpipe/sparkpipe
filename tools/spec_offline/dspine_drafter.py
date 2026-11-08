"""DSpine block drafter (Tencent DSpine, https://github.com/Tencent/DSpine, Apache-2.0): a DFlash backbone with
causal block attention, adjacent causal injection after every layer and the serial last-write refinement, offline and
in draftd.

Block rows 0..T-1 sit at positions a..a+T-1: row 0 holds the anchor token a, rows 1..T-1 the mask token, and row j
proposes the token at a+j (the inplace convention). Every row attends to the context keys (the fc projection of the
target taps, as DFlash) and to the block rows up to its own; a sliding_attention layer limits row j to positions
a+j-W+1 and later. The residual stream is BF16 at every add, as the reference serving path rounds it.

Code bridge, after layer l (every layer is active; the reference's _forward_fused / ref_bridge_forward):
  u = the layer output, r = sqrt(max(mean(u^2), tiny)), n = bf16(u / (r + eps))
  z = sqrt(C) normalize(R_l n) for rows 1..T-1, z_0 = sqrt(C) codebook[anchor]; C = code_size
  [q; s; m] = [G_q; G_p; A] bf16(z)                 one BF16 GEMM, packed per row
  g_t = sigmoid(q_t + s_{t-1} + b), m_t = g_t * m_{t-1} (BF16), zero on row 0
  h = bf16(u + r W_l m), row 0 unchanged; the next input is bf16(bf16(h rsqrt(mean(h^2) + eps)) w_next)
The codebook ships in the checkpoint (selected_prefix_bridge.codebook [vocab, C]); the greedy tables are built at load
from the target LM head: TG[v] = bf16(head[v] * w_norm) W_last, TSV[v] = [G_p; A] bf16(sqrt(C) codebook[v]).

Refinement (code_bridge_second_pass serial_table, the last write only): the K candidates of row j are the K largest
logits of norm(h_j) @ head, ascending by token id (ties to the lower id). For every candidate i of row j-1 (the anchor
for row 1) the last write is redone with the exact sender code of that candidate:
  S_j[i, c] = (u_j w_norm) . head[c] + r_j (sigmoid(q_j + TSV[i][:R] + b) * TSV[i][R:]) . TG[c]
the greedy chain starts at row 1 and takes argmax_c S_j[sender, c], the chosen lane becoming row j+1's sender. The
confidence of a choice is 1 / sum_c exp((S_j[sender, c] - max) / max(r_j, 1e-6)). A sampled walk draws over the
exact FP32 table of the redone write, normalized and through the head (serial_rs_table), with the verifier's keyed
Gumbel noise (latent_block_drafter.coupled_candidates), so the drafter is coupled to a keyed verify.

The draftd graph (block_drafter.DraftGraph) recomputes the last norm without the bridge's intermediate rounding; walk
re-ranks the 64 best of those logits with the bridge's own normed rows, so both paths pick candidates the same way.
Serve with --window at most sliding_window - block_size (8176 for the Qwen3.8-27B checkpoint): draftd keeps one
context window for every row, and that bound keeps it inside each row's sliding window.
"""

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from block_drafter import SafeTensors, rms, rotate, yarn_frequencies
from latent_block_drafter import coupled_candidates

BRIDGE = "selected_prefix_bridge."
F32_TINY = torch.finfo(torch.float32).tiny
CANDIDATE_POOL = 64
CONFIDENCE_SCALE_FLOOR = 1e-6
TABLE_CHUNK = 8192
BACKBONE = ("input_layernorm", "post_attention_layernorm", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "self_attn.q_norm", "self_attn.k_norm", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


class DrafterContractError(ValueError):
    pass


def refuse_unsupported(config):
    dflash = config.get("dflash_config") or {}
    bridge = dflash.get("selected_prefix_code_bridge") or {}
    layers = config.get("num_hidden_layers", 0)
    refine = dflash.get("selected_prefix_last_write_refine") or {}
    second = config.get("code_bridge_second_pass") or {}
    widths = {int(item["head_topk"]) for item in (refine, second) if "head_topk" in item}
    kinds = set(config.get("layer_types") or ["full_attention"])
    rope = config.get("rope_parameters") or {}
    checks = (
        (all(key in bridge for key in ("code_size", "rank", "eps")), "dflash_config.selected_prefix_code_bridge needs code_size, rank and eps"),
        (dflash.get("selected_prefix_feature_writes") is True, "selected_prefix_feature_writes must be true"),
        (dflash.get("selected_prefix_feature_write_mode") == "independent", "selected_prefix_feature_write_mode must be independent"),
        (dflash.get("selected_prefix_terminal") is False, "selected_prefix_terminal must be false"),
        (dflash.get("selected_prefix_decode_mode", "serial") == "serial", "selected_prefix_decode_mode must be serial"),
        (dflash.get("draft_block_causal") is True, "draft_block_causal must be true"),
        (dflash.get("selected_prefix_feature_layers") == list(range(1, layers + 1)), "every layer must carry a feature write"),
        (not dflash.get("selected_prefix_terminal_residual_rank"), "a terminal token residual is unsupported"),
        (dflash.get("selected_prefix_feature_reliability") is None, "reliability probes are unsupported"),
        (not dflash.get("selected_prefix_feature_next_token_rows", False), "next-token rows are a DSpark layout"),
        (not any(dflash.get(key) for key in ("conv_kernel_size", "selector_rank", "selector_top_k", "projector_type", "attention_sink_bias")),
         "convolutions, selectors, projectors and sinks are not DSpine"),
        (dflash.get("attention_value_scale") is None, "attention_value_scale is unsupported"),
        (not refine or (refine.get("kind") == "exact_sender" and refine.get("version", 1) == 1 and float(refine.get("weight", 1.0)) >= 0),
         "the last-write refinement must be exact_sender v1"),
        (second.get("mode") == "serial_table" and second.get("first_exact") == layers - 1, "code_bridge_second_pass must be serial_table on the last layer"),
        (len(widths) == 1 and all(k & (k - 1) == 0 and 2 <= k <= 64 for k in widths), "refinement widths must agree on a power of two in [2, 64]"),
        (kinds in ({"full_attention"}, {"sliding_attention"}), "layers must be all full or all sliding attention"),
        (kinds == {"full_attention"} or isinstance(config.get("sliding_window"), int), "sliding layers need sliding_window"),
        (rope.get("rope_type", "default") == "default" and not config.get("rope_scaling"), "only default RoPE is supported"),
        (float(config.get("partial_rotary_factor") or 1.0) == 1.0, "partial rotary is unsupported"),
    )
    failed = [message for ok, message in checks if not ok]
    if failed:
        raise DrafterContractError("DSpine drafter refuses this checkpoint: " + "; ".join(failed))


def is_dspine_checkpoint(directory):
    with open(Path(directory) / "config.json") as handle:
        return "selected_prefix_code_bridge" in (json.load(handle).get("dflash_config") or {})


def serial_tables(head, norm, write, senders, codebook, chunk=TABLE_CHUNK):
    rank = write.shape[1]
    scale = float(codebook.shape[1]) ** 0.5
    tg = torch.empty((head.shape[0], rank), dtype=torch.bfloat16, device=head.device)
    tsv = torch.empty((head.shape[0], 2 * rank), dtype=torch.bfloat16, device=head.device)
    weight = norm.float()
    for begin in range(0, head.shape[0], chunk):
        end = min(begin + chunk, head.shape[0])
        tg[begin:end] = (head[begin:end].float() * weight).to(torch.bfloat16) @ write.to(torch.bfloat16)
        tsv[begin:end] = (scale * codebook[begin:end].float()).to(torch.bfloat16) @ senders.t()
    return tg, tsv


def pooled_candidates(normed, logits, head, count, pool=CANDIDATE_POOL):
    widest = min(pool, logits.shape[-1])
    pooled = torch.sort(torch.topk(logits, widest, -1).indices, -1).values
    exact = torch.bmm(normed[:, None, :], head[pooled].transpose(1, 2))[:, 0].float()
    chosen = torch.sort(exact, dim=-1, descending=True, stable=True).indices[:, :count]
    return torch.sort(torch.gather(pooled, 1, chosen), -1).values


def senders_of(anchor, candidates):
    first = torch.as_tensor(anchor, device=candidates.device).view(1, 1).expand(1, candidates.shape[1])
    return torch.cat((first, candidates[:-1]), 0)


def greedy_table(raw, scale, query, senders, candidates, head, norm, tg, tsv, gate_bias):
    rank = tg.shape[1]
    rows = head[candidates].float()
    base = torch.bmm(rows, (raw.float() * norm.float())[..., None])[..., 0]
    pair = tsv[senders].float()
    message = torch.sigmoid(query.float()[:, None, :] + pair[..., :rank] + gate_bias.float()) * pair[..., rank:]
    scores = base[:, None, :] + scale[:, None, None] * torch.bmm(message, tg[candidates].float().transpose(1, 2))
    best = scores.max(-1, keepdim=True).values
    confidence = 1.0 / torch.exp((scores - best) / scale.clamp_min(CONFIDENCE_SCALE_FLOOR)[:, None, None]).sum(-1)
    return torch.argmax(scores, -1), confidence


def serial_chain(table, candidates, confidence):
    lane = table[0, 0]
    previous = torch.zeros((), dtype=torch.long, device=table.device)
    tokens, scores = [], []
    for row in range(table.shape[0]):
        scores.append(torch.index_select(confidence[row], 0, previous.view(1)))
        tokens.append(torch.index_select(candidates[row], 0, lane.view(1)))
        previous = lane
        if row + 1 < table.shape[0]:
            lane = torch.index_select(table[row + 1], 0, lane.view(1))[0]
    return torch.cat(tokens), torch.cat(scores)


def refined_table(raw, scale, query, codes, candidates, gate_weight, value_weight, gate_bias, write, norm, head, eps):
    gate = torch.sigmoid(query.float()[:, None, :] + codes @ gate_weight.t() + gate_bias)
    message = gate * (codes @ value_weight.t())
    refined = raw.float()[:, None, :] + scale[:, None, None] * (message @ write.t())
    normalized = refined / torch.sqrt(refined.pow(2).mean(-1, keepdim=True) + eps) * norm
    return torch.bmm(normalized, head[candidates].float().transpose(1, 2))


class AnchoredEmbedding:
    def __init__(self, weight, drafter):
        self.weight = weight
        self.drafter = drafter

    def __getitem__(self, tokens):
        self.drafter.anchor = tokens[0]
        return self.weight[tokens]


class DSpineDrafter:
    convention = "inplace"

    def __init__(self, directory, shard, names, device):
        self.device = device
        with open(Path(directory) / "config.json") as handle:
            self.config = c = json.load(handle)
        refuse_unsupported(c)
        dflash = c["dflash_config"]
        bridge = dflash["selected_prefix_code_bridge"]
        weights = SafeTensors(Path(directory) / "model.safetensors")
        target = SafeTensors(shard)
        self.eps = c["rms_norm_eps"]
        self.heads, self.kv_heads, self.head_dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.block = c.get("block_size") or dflash["block_size"]
        self.max_drafts = self.block - 1
        self.layer_ids = list(dflash["target_layer_ids"])
        self.mask = dflash["mask_token_id"]
        self.code_size, self.rank, self.bridge_eps = int(bridge["code_size"]), int(bridge["rank"]), float(bridge["eps"])
        self.top_k = int(c["code_bridge_second_pass"]["head_topk"])
        self.sliding = c["sliding_window"] if set(c.get("layer_types") or []) == {"sliding_attention"} else None
        self.window = None
        for name in ("draft_block_causal_version", "last_write_refine_version", BRIDGE + "independent_write_version", BRIDGE + "terminal_disabled_version"):
            if name in weights.header and int(weights.load(name, "cpu")) != 1:
                raise DrafterContractError(f"DSpine drafter refuses {name} {int(weights.load(name, 'cpu'))}")
        self.fc = weights.load("fc.weight", device)
        self.hidden_norm = weights.load("hidden_norm.weight", device)
        self.norm = weights.load("norm.weight", device)
        self.layers = []
        for index in range(c["num_hidden_layers"]):
            layer = {name: weights.load(f"layers.{index}.{name}.weight", device) for name in BACKBONE}
            layer["index"] = index
            self.layers.append(layer)
        last = len(self.layers) - 1
        early = weights.load(BRIDGE + "early_read_weights", device)
        self.read = [early[index] for index in range(last)] + [weights.load(BRIDGE + "read_weight", device)]
        self.write = weights.load(BRIDGE + "feature_write_weights", device)
        self.codebook = weights.load(BRIDGE + "codebook", device)
        self.gate_weight = weights.load(BRIDGE + "gate.weight", device)
        self.gate_bias = weights.load(BRIDGE + "gate.bias", device)
        self.sender_weight = weights.load(BRIDGE + "predecessor_gate.weight", device)
        self.value_weight = weights.load(BRIDGE + "prev.weight", device)
        if weights.load(BRIDGE + "out.weight", device).count_nonzero().item():
            raise DrafterContractError("DSpine drafter refuses a terminal-disabled checkpoint with a nonzero terminal projection")
        self.packed = torch.cat((self.gate_weight, self.sender_weight, self.value_weight)).contiguous()
        self.exact = (self.sender_weight.float(), self.value_weight.float(), self.gate_bias.float(), self.write[last].float(), self.norm.float())
        self.embed = AnchoredEmbedding(target.load(names["embed"], device), self)
        self.head = target.load(names["head"], device)
        self.target_norm = target.load(names["norm"], device)
        self.tg, self.tsv = serial_tables(self.head, self.norm, self.write[last], self.packed[self.rank:], self.codebook)
        self.inv, self.attention = yarn_frequencies(c, device)
        self.anchor = None
        self.stash = None

    def context(self, taps, positions, chunk=4096):
        keys, values = [[] for _ in self.layers], [[] for _ in self.layers]
        for begin in range(0, taps.shape[0], chunk):
            part = taps[begin:begin + chunk].to(self.device, torch.bfloat16)
            where = positions[begin:begin + chunk].to(self.device)
            features = rms(part @ self.fc.t(), self.hidden_norm, self.eps).to(torch.bfloat16)
            for index, layer in enumerate(self.layers):
                k = rms((features @ layer["self_attn.k_proj"].t()).view(-1, self.kv_heads, self.head_dim), layer["self_attn.k_norm"], self.eps).to(torch.bfloat16)
                keys[index].append(rotate(k.float(), where, self.inv, self.attention).to(torch.bfloat16))
                values[index].append((features @ layer["self_attn.v_proj"].t()).view(-1, self.kv_heads, self.head_dim))
        return [torch.cat(parts, 0) for parts in keys], [torch.cat(parts, 0) for parts in values]

    def fused_norm(self, x, weight):
        x32 = x.float()
        y = (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).to(torch.bfloat16)
        return (y.float() * weight.float()).to(torch.bfloat16)

    def pre_attention(self, layer, h, positions):
        block, heads, kv_heads, dim = self.block, self.heads, self.kv_heads, self.head_dim
        x = rms(h, layer["input_layernorm"], self.eps).to(torch.bfloat16) if layer["index"] == 0 else self.fused_norm(h, layer["input_layernorm"])
        q = rms((x @ layer["self_attn.q_proj"].t()).view(block, heads, dim), layer["self_attn.q_norm"], self.eps).to(torch.bfloat16)
        kb = rms((x @ layer["self_attn.k_proj"].t()).view(block, kv_heads, dim), layer["self_attn.k_norm"], self.eps).to(torch.bfloat16)
        q = rotate(q.float(), positions, self.inv, self.attention).to(torch.bfloat16)
        kb = rotate(kb.float(), positions, self.inv, self.attention).to(torch.bfloat16)
        vb = (x @ layer["self_attn.v_proj"].t()).view(block, kv_heads, dim)
        return q.transpose(0, 1), kb.transpose(0, 1), vb.transpose(0, 1)

    def bridge(self, index, raw):
        f = raw.float()
        scale = f.pow(2).mean(-1).clamp_min(F32_TINY).sqrt()
        normalized = (f / (scale[:, None] + self.bridge_eps)).to(torch.bfloat16)
        projected = (normalized @ self.read[index].t()).float()
        codes = math.sqrt(self.code_size) * projected / (projected.norm(dim=-1, keepdim=True) + self.bridge_eps)
        anchor = math.sqrt(self.code_size) * torch.index_select(self.codebook, 0, self.anchor.view(1)).float()
        packed = torch.cat((anchor, codes[1:])).to(torch.bfloat16) @ self.packed.t()
        query, sender, value = packed.float().split(self.rank, -1)
        gate = torch.sigmoid(query[1:] + sender[:-1] + self.gate_bias.float())
        message = torch.cat((torch.zeros_like(query[:1]), gate * value[:-1])).to(torch.bfloat16)
        delta = (message @ self.write[index].t()).float()
        out = torch.cat((f[:1], f[1:] + scale[1:, None] * delta[1:])).to(torch.bfloat16)
        if index == len(self.layers) - 1:
            self.stash = {"raw": raw, "rms": scale, "query": packed[:, :self.rank], "normed": self.fused_norm(out, self.norm)}
        return out

    def post_attention(self, layer, h, attended):
        h = (h.to(torch.bfloat16) + attended @ layer["self_attn.o_proj"].t())
        x = rms(h, layer["post_attention_layernorm"], self.eps).to(torch.bfloat16)
        gate = x @ layer["mlp.gate_proj"].t()
        up = x @ layer["mlp.up_proj"].t()
        raw = h + (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16) @ layer["mlp.down_proj"].t()
        return self.bridge(layer["index"], raw)

    def block_mask(self, width, context_columns):
        rows = torch.arange(self.block, device=self.device)[:, None]
        columns = torch.arange(width, device=self.device)[None, :]
        return (columns < context_columns) | (columns - context_columns <= rows)

    def attend_window(self, index, q, kb, vb, keys, values, start, block_at, mask):
        stop = block_at + self.block
        keys[index][:, block_at:stop] = kb
        values[index][:, block_at:stop] = vb
        visible = self.block_mask(stop - start, block_at - start)
        if mask is not None:
            visible = visible & mask.view(-1)[start:stop][None, :]
        attended = torch.nn.functional.scaled_dot_product_attention(q[None], keys[index][None, :, start:stop], values[index][None, :, start:stop],
                                                                    attn_mask=visible[None, None], enable_gqa=True)
        return attended[0].transpose(0, 1).reshape(self.block, -1)

    def draft(self, anchor_token, anchor_position, keys, values, count):
        self.anchor = torch.tensor(int(anchor_token), device=self.device)
        tokens = torch.full((self.block,), self.mask, dtype=torch.long, device=self.device)
        tokens[0] = int(anchor_token)
        positions = torch.arange(anchor_position, anchor_position + self.block, device=self.device)
        first = 0 if self.window is None else max(0, count - self.window)
        if self.sliding is not None:
            first = max(first, count - (self.sliding - 1))
        visible = self.block_mask(count - first + self.block, count - first)
        if self.sliding is not None:
            context = torch.arange(first, count, device=self.device)[None, :]
            rows = torch.arange(self.block, device=self.device)[:, None]
            visible[:, :count - first] &= context >= count + rows - (self.sliding - 1)
        with torch.no_grad():
            h = self.embed.weight[tokens].float()
            for index, layer in enumerate(self.layers):
                q, kb, vb = self.pre_attention(layer, h, positions)
                k = torch.cat((keys[index][first:count].transpose(0, 1), kb), 1)
                v = torch.cat((values[index][first:count].transpose(0, 1), vb), 1)
                attended = torch.nn.functional.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=visible[None, None], enable_gqa=True)
                h = self.post_attention(layer, h, attended[0].transpose(0, 1).reshape(self.block, -1))
            y = self.stash["normed"]
            return y, (y @ self.head.t()).float()

    def candidates(self, base):
        stash = self.stash
        if "candidates" not in stash:
            stash["candidates"] = pooled_candidates(stash["normed"][1:], base[1:], self.head, self.top_k)
        return stash["candidates"]

    def exact_codes(self, senders):
        return math.sqrt(self.code_size) * self.codebook[senders].float()

    def walk(self, y, base, previous, count, sampling=None, position=None, sampled=None):
        stash = self.stash
        candidates = self.candidates(base)
        senders = senders_of(previous, candidates)
        raw, scale, query = stash["raw"][1:], stash["rms"][1:], stash["query"][1:]
        table, confidence = greedy_table(raw, scale, query, senders, candidates, self.head, self.norm, self.tg, self.tsv, self.gate_bias)
        tokens, scores = serial_chain(table, candidates, confidence)
        if sampling is None:
            return tokens, scores
        sender_weight, value_weight, gate_bias, write, norm = self.exact
        refined = refined_table(raw, scale, query, self.exact_codes(senders), candidates, sender_weight, value_weight, gate_bias, write, norm, self.head, self.eps)
        lane = torch.zeros((), dtype=torch.long, device=refined.device)
        drawn = []
        for row in range(refined.shape[0]):
            lane = coupled_candidates(torch.index_select(refined[row], 0, lane.view(1))[0], candidates[row], sampling, position + row)
            drawn.append(torch.index_select(candidates[row], 0, lane.view(1)))
        return torch.where(sampled, torch.cat(drawn), tokens), scores

    def slot_logits(self, y, slot, previous):
        stash = self.stash
        candidates = self.candidates((y @ self.head.t()).float())
        row = slice(slot + 1, slot + 2)
        sender_weight, value_weight, gate_bias, write, norm = self.exact
        codes = self.exact_codes(torch.tensor([[int(previous)]], device=self.device))
        logits = refined_table(stash["raw"][row], stash["rms"][row], stash["query"][row], codes, candidates[slot:slot + 1], sender_weight, value_weight,
                               gate_bias, write, norm, self.head, self.eps)[0, 0]
        full = torch.full((self.head.shape[0],), float("-inf"), device=self.device)
        full[candidates[slot]] = logits
        return full

    def sequential(self, y, base, anchor_token, slots):
        with torch.no_grad():
            tokens, scores = self.walk(y, base, torch.tensor(int(anchor_token), device=self.device), self.block)
        chosen = [slot - 1 for slot in slots]
        return tokens[chosen].tolist(), scores[chosen].tolist()

    def sequential_sampled(self, y, base, anchor_token, slots, sampling, position):
        with torch.no_grad():
            tokens, _ = self.walk(y, base, torch.tensor(int(anchor_token), device=self.device), self.block, sampling,
                                  torch.tensor(int(position), device=self.device), torch.tensor(True, device=self.device))
        return tokens[[slot - 1 for slot in slots]].tolist()
