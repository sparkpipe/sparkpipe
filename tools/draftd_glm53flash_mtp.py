import json
import os
import struct

import numpy as np
import torch

from draftd_kernels import kernels

MTP_PREFIX = "model.language_model.layers.{layer}."
DTYPES = {"BF16": np.uint16, "F32": np.float32, "F8_E4M3": np.uint8, "U8": np.uint8}


def e4m3_table():
    table = np.zeros(256, dtype=np.float32)
    for i in range(256):
        sign = -1.0 if i & 0x80 else 1.0
        e, m = (i >> 3) & 0xF, i & 0x7
        if e == 0:
            value = m / 8.0 * 2.0 ** -6
        elif e == 15 and m == 7:
            value = np.nan
        else:
            value = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        table[i] = sign * value
    return table


class Checkpoint:
    def __init__(self, root):
        self.root = root
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
        start, end = entry["data_offsets"]
        return np.memmap(path, dtype=DTYPES[entry["dtype"]], mode="r", offset=base + start,
                         shape=tuple(entry["shape"]))

    def dtype(self, name):
        return self.entries[name][2]["dtype"]


def bf16r(x):
    return x.to(torch.bfloat16).to(torch.float32)


class Glm53FlashMtpDrafter:
    def __init__(self, checkpoint_dir, device="cuda", max_chain=7, head_rows=None):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = torch.device(device)
        self.ck = Checkpoint(checkpoint_dir)
        config = json.load(open(os.path.join(checkpoint_dir, "config.json")))
        config = config.get("text_config", config)
        self.config = config
        self.layer = int(config["num_hidden_layers"])
        self.hidden = int(config["hidden_size"])
        self.eps = float(config["rms_norm_eps"])
        self.heads = int(config["num_attention_heads"])
        self.latent = int(config["kv_lora_rank"])
        self.nope = int(config["qk_nope_head_dim"])
        self.vdim = int(config["v_head_dim"])
        self.topk = int(config["num_experts_per_tok"])
        self.experts = int(config["n_routed_experts"])
        self.norm_topk = int(config["norm_topk_prob"])
        self.scaling = float(config["routed_scaling_factor"])
        self.limit = float(config["swiglu_limit"])
        self.max_chain = max_chain
        self.lut = torch.from_numpy(e4m3_table()).to(self.device)
        self.k = kernels()
        self.prefix = MTP_PREFIX.format(layer=self.layer)
        self._load(head_rows)
        self.graphs = {}

    def _raw(self, name):
        return torch.from_numpy(np.ascontiguousarray(self.ck.array(name))).to(self.device)

    def _vector(self, name):
        raw = self._raw(name)
        if self.ck.dtype(name) == "BF16":
            return raw.view(torch.int16).view(torch.bfloat16).to(torch.float32)
        return raw.to(torch.float32)

    def _spine(self, name):
        raw = self._raw(name)
        if self.ck.dtype(name) == "BF16":
            return raw.view(torch.int16).view(torch.bfloat16).contiguous()
        if self.ck.dtype(name) != "F8_E4M3":
            raise ValueError(f"{name}: unsupported spine dtype {self.ck.dtype(name)}")
        scale = self._raw(name + "_scale_inv").to(torch.float32)
        rows, cols = raw.shape
        tiled = scale.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :cols]
        return (self.lut[raw.long()] * tiled).to(torch.bfloat16).contiguous()

    def _experts(self, kind):
        names = [f"{self.prefix}mlp.experts.{e}.{kind}.weight" for e in range(self.experts)]
        first = self.ck.array(names[0])
        codes = torch.empty((self.experts,) + first.shape, dtype=torch.uint8, device=self.device)
        scales = []
        for e, name in enumerate(names):
            if self.ck.dtype(name) != "F8_E4M3":
                raise ValueError(f"{name}: routed experts must be F8_E4M3 block-128")
            codes[e].copy_(torch.from_numpy(np.ascontiguousarray(self.ck.array(name))))
            scales.append(torch.from_numpy(np.ascontiguousarray(self.ck.array(name + "_scale_inv"))).to(torch.float32))
        return codes, torch.stack(scales).to(self.device).contiguous()

    def _load(self, head_rows):
        p = self.prefix
        self.enorm = self._vector(p + "enorm.weight")
        self.hnorm = self._vector(p + "hnorm.weight")
        self.eh_proj = self._spine(p + "eh_proj.weight")
        self.input_norm = self._vector(p + "input_layernorm.weight")
        self.post_norm = self._vector(p + "post_attention_layernorm.weight")
        self.head_norm = self._vector(p + "shared_head.norm.weight")
        self.q_a = self._spine(p + "self_attn.q_a_proj.weight")
        self.q_a_norm = self._vector(p + "self_attn.q_a_layernorm.weight")
        self.q_b = self._spine(p + "self_attn.q_b_proj.weight")
        self.kv_a = self._spine(p + "self_attn.kv_a_proj_with_mqa.weight")
        self.kv_a_norm = self._vector(p + "self_attn.kv_a_layernorm.weight")
        kvb = self._spine(p + "self_attn.kv_b_proj.weight").to(torch.float32)
        kvb = kvb.view(self.heads, self.nope + self.vdim, self.latent)
        self.wk = kvb[:, :self.nope].contiguous()
        self.wv = kvb[:, self.nope:].contiguous()
        self.o_proj = self._spine(p + "self_attn.o_proj.weight")
        self.router = self._vector(p + "mlp.gate.weight").contiguous()
        self.router_bias = self._vector(p + "mlp.gate.e_score_correction_bias")
        self.shared_gate = self._spine(p + "mlp.shared_experts.gate_proj.weight")
        self.shared_up = self._spine(p + "mlp.shared_experts.up_proj.weight")
        self.shared_down = self._spine(p + "mlp.shared_experts.down_proj.weight")
        self.gate_codes, self.gate_scale = self._experts("gate_proj")
        self.up_codes, self.up_scale = self._experts("up_proj")
        self.down_codes, self.down_scale = self._experts("down_proj")
        self.embed = self._raw("model.language_model.embed_tokens.weight").view(torch.int16).view(torch.bfloat16)
        head = self._raw("lm_head.weight").view(torch.int16).view(torch.bfloat16)
        self.head_ids = None
        if head_rows is not None:
            self.head_ids = torch.as_tensor(np.asarray(head_rows, dtype=np.int64), device=self.device)
            head = head[self.head_ids].contiguous()
        self.lm_head = head.contiguous()
        self.logits = torch.empty(self.lm_head.shape[0], dtype=torch.float32, device=self.device)

    def device_bytes(self):
        total = 0
        for value in vars(self).values():
            if isinstance(value, torch.Tensor) and value.is_cuda:
                total += value.numel() * value.element_size()
        return total

    def linear(self, weight, x):
        out = torch.empty(weight.shape[0], dtype=torch.float32, device=self.device)
        self.k.gemv_bf16(weight, x.contiguous(), out)
        return bf16r(out)

    def rmsnorm(self, x, weight):
        return x / torch.sqrt(torch.mean(x * x) + self.eps) * weight

    def attention(self, x, cache, length):
        q_norm = bf16r(self.rmsnorm(self.linear(self.q_a, x), self.q_a_norm))
        q = self.linear(self.q_b, q_norm).view(self.heads, self.nope)
        kv = self.linear(self.kv_a, x)
        cache[length] = bf16r(self.rmsnorm(kv[:self.latent], self.kv_a_norm))
        slots = cache[:length + 1]
        ql = bf16r(torch.einsum("hn,hnl->hl", q, self.wk))
        scores = (ql @ slots.T) * (self.nope ** -0.5)
        weights = torch.softmax(scores, dim=1)
        al = bf16r(weights @ slots)
        attn = bf16r(torch.einsum("hl,hvl->hv", al, self.wv))
        return self.linear(self.o_proj, attn.reshape(-1))

    def moe(self, x):
        scores = torch.sigmoid(self.router @ x)
        choice = scores + self.router_bias
        selected = torch.sort(torch.topk(choice, self.topk).indices).values
        picked = scores[selected]
        if self.norm_topk:
            weights = picked / (picked.sum() + 1e-20) * self.scaling
        else:
            weights = picked * self.scaling
        gate = torch.empty((self.topk, self.gate_codes.shape[1]), dtype=torch.float32, device=self.device)
        up = torch.empty_like(gate)
        self.k.gemv_fp8_block(self.gate_codes, self.gate_scale, selected, x, gate)
        self.k.gemv_fp8_block(self.up_codes, self.up_scale, selected, x, up)
        gate = torch.clamp(bf16r(gate), max=self.limit)
        up = torch.clamp(bf16r(up), -self.limit, self.limit)
        act = bf16r(bf16r(gate * torch.sigmoid(gate)) * up).contiguous()
        down = torch.empty((self.topk, self.down_codes.shape[1]), dtype=torch.float32, device=self.device)
        self.k.gemv_fp8_block(self.down_codes, self.down_scale, selected, act, down)
        down = bf16r(down)
        routed = torch.zeros_like(x)
        for i in range(self.topk):
            routed = bf16r(routed + bf16r(down[i] * weights[i]))
        sg = torch.clamp(self.linear(self.shared_gate, x), max=self.limit)
        su = torch.clamp(self.linear(self.shared_up, x), -self.limit, self.limit)
        sa = bf16r(bf16r(sg * torch.sigmoid(sg)) * su)
        shared = self.linear(self.shared_down, sa)
        return bf16r(routed + shared), selected

    def step(self, hidden, token, cache, length):
        embed = self.embed[token].to(torch.float32).reshape(-1)
        e = bf16r(self.rmsnorm(embed, self.enorm))
        h = bf16r(self.rmsnorm(hidden, self.hnorm))
        x = self.linear(self.eh_proj, torch.cat([e, h]))
        x = bf16r(x + self.attention(bf16r(self.rmsnorm(x, self.input_norm)), cache, length))
        mlp, _ = self.moe(bf16r(self.rmsnorm(x, self.post_norm)))
        x = bf16r(x + mlp)
        head_in = bf16r(self.rmsnorm(x, self.head_norm))
        self.k.gemv_bf16(self.lm_head, head_in, self.logits)
        best = torch.argmax(self.logits)
        if self.head_ids is not None:
            best = self.head_ids[best]
        return x, head_in, best

    def chain_eager(self, hidden, token, depth):
        cache = torch.zeros((self.max_chain, self.latent), dtype=torch.float32, device=self.device)
        h = hidden.to(self.device, torch.float32)
        t = torch.as_tensor([int(token)], device=self.device)
        drafts = []
        for d in range(depth):
            h, _, best = self.step(h, t, cache, d)
            t = best.reshape(1)
            drafts.append(best)
        return torch.stack(drafts)

    def capture(self, depth):
        state = {"hidden": torch.zeros(self.hidden, dtype=torch.float32, device=self.device),
                 "token": torch.zeros(1, dtype=torch.int64, device=self.device),
                 "cache": torch.zeros((self.max_chain, self.latent), dtype=torch.float32, device=self.device)}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._chain_body(state, depth)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            state["drafts"] = self._chain_body(state, depth)
        self.graphs[depth] = (graph, state)

    def _chain_body(self, state, depth):
        h, t = state["hidden"], state["token"]
        drafts = []
        for d in range(depth):
            h, _, best = self.step(h, t, state["cache"], d)
            t = best.reshape(1)
            drafts.append(best)
        return torch.stack(drafts)

    def chain(self, hidden, token, depth):
        if depth not in self.graphs:
            self.capture(depth)
        graph, state = self.graphs[depth]
        state["hidden"].copy_(hidden.to(torch.float32), non_blocking=True)
        state["token"].fill_(int(token))
        graph.replay()
        return state["drafts"]

