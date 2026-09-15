import numpy as np

from t1_reference_common import (Safetensors, bf16_to_f32, define_float,
                                 define_uint, rmsnorm, sigmoid)

PREFIX = "model.layers."
FP8_GROUP = 32
WSUM_FLOOR = 6.103515625e-5
MAX_INDEXER_CONTEXT = 2048

_E4M3 = np.zeros(256, dtype=np.float32)
for _i in range(256):
    _sign = -1.0 if _i & 0x80 else 1.0
    _e = (_i >> 3) & 0xF
    _m = _i & 0x7
    if _e == 0:
        _v = _m / 8.0 * 2.0 ** -6
    elif _e == 15 and _m == 7:
        _v = np.nan
    else:
        _v = (1.0 + _m / 8.0) * 2.0 ** (_e - 7)
    _E4M3[_i] = _sign * _v

_E8M0 = (2.0 ** np.arange(-127, 128, dtype=np.float64)).astype(np.float32)

DEFINES_VS_CONFIG = [
    ("HIDDEN_DIMENSION", "hidden_size", "uint"),
    ("LAYER_COUNT", "num_hidden_layers", "uint"),
    ("OUTPUT_VOCAB_COUNT", "vocab_size", "uint"),
    ("RMS_NORM_EPSILON", "rms_norm_eps", "float"),
    ("MLA_HEAD_COUNT", "num_attention_heads", "uint"),
    ("MLA_Q_LORA_RANK", "q_lora_rank", "uint"),
    ("MLA_LATENT_DIMENSION", "kv_lora_rank", "uint"),
    ("MLA_QK_HEAD_DIMENSION", "qk_head_dim", "uint"),
    ("MLA_QK_NOPE_HEAD_DIMENSION", "qk_nope_head_dim", "uint"),
    ("MLA_QK_ROPE_HEAD_DIMENSION", "qk_rope_head_dim", "uint"),
    ("MLA_VALUE_HEAD_DIMENSION", "v_head_dim", "uint"),
    ("HC_STREAM_COUNT", "hc_mult", "uint"),
    ("HC_EPSILON", "hc_eps", "float"),
    ("HC_MAGNITUDE", "hc_magnitude", "float"),
    ("MOE_EXPERT_COUNT", "n_routed_experts", "uint"),
    ("MOE_TOP_K", "num_experts_per_tok", "uint"),
    ("MOE_SHARED_EXPERT_COUNT", "n_shared_experts", "uint"),
    ("MOE_INTERMEDIATE_DIMENSION", "moe_intermediate_size", "uint"),
    ("MOE_ROUTED_SCALING_FACTOR", "routed_scaling_factor", "float"),
    ("MOE_NORM_TOPK_PROB", "norm_topk_prob", "uint"),
    ("DENSE_INTERMEDIATE_DIMENSION", "intermediate_size", "uint"),
    ("SWIGLU_LIMIT", "swiglu_limit", "float"),
    ("INDEX_HEAD_COUNT", "index_n_heads", "uint"),
    ("INDEX_HEAD_DIMENSION", "index_head_dim", "uint"),
    ("INDEX_TOP_K", "index_topk", "uint"),
    ("END_OF_TEXT_TOKEN_ID", "eos_token_id", "uint"),
]

DEFINES_RECORDED_ONLY = [
    ("MLA_ROPE_THETA", "rope_parameters.rope_theta", "float"),
]


class Hy4ConfigError(ValueError):
    pass


def define_of(kind, defines, dname):
    if kind == "uint":
        return define_uint(defines, dname)
    return define_float(defines, dname)


def config_of(kind, config, cname):
    node = config
    while "." in cname:
        head, cname = cname.split(".", 1)
        if not isinstance(node, dict) or head not in node:
            raise Hy4ConfigError(f"config key {head} missing")
        node = node[head]
    if cname not in node:
        raise Hy4ConfigError(f"config key {cname} missing")
    value = node[cname]
    return float(value) if kind == "float" else int(value)


def cross_check(defines, config):
    mismatches = []
    for dname, cname, kind in DEFINES_VS_CONFIG:
        want = define_of(kind, defines, dname)
        try:
            got = config_of(kind, config, cname)
        except Hy4ConfigError as error:
            mismatches.append({"define": f"SPARK_LLM_{dname}",
                               "config": cname, "error": str(error)})
            continue
        if want != got:
            raise Hy4ConfigError(
                f"SPARK_LLM_{dname}={want} disagrees with config "
                f"{cname}={got}")
    for dname, cname, kind in DEFINES_RECORDED_ONLY:
        want = define_of(kind, defines, dname)
        got = config_of(kind, config, cname)
        if want != got:
            mismatches.append({"define": f"SPARK_LLM_{dname}", "config": cname,
                               "define_value": want, "config_value": got})
    return mismatches


def dequant_fp8(payload, scale):
    if payload.dtype != np.uint8 or scale.dtype != np.uint8:
        raise ValueError("fp8 plane requires U8 payload and U8 scale")
    rows, cols = payload.shape
    if cols % FP8_GROUP:
        raise ValueError(f"column count {cols} not a multiple of {FP8_GROUP}")
    groups = cols // FP8_GROUP
    if scale.shape != (rows, groups):
        raise ValueError(f"scale shape {scale.shape} disagrees with "
                         f"payload {(rows, cols)} group size {FP8_GROUP}")
    weights = _E4M3[payload.reshape(rows, groups, FP8_GROUP)]
    scales = _E8M0[scale.astype(np.int32)].astype(np.float32)
    return (weights * scales[:, :, None]).reshape(rows, cols)


class Hy4Engine:
    def __init__(self, checkpoint, defines, config):
        self.config_mismatches = cross_check(defines, config)
        self.st = Safetensors(checkpoint)
        self.hidden = define_uint(defines, "HIDDEN_DIMENSION")
        self.layers = define_uint(defines, "LAYER_COUNT")
        self.vocab = define_uint(defines, "OUTPUT_VOCAB_COUNT")
        self.eps = define_float(defines, "RMS_NORM_EPSILON")
        self.heads = define_uint(defines, "MLA_HEAD_COUNT")
        self.q_lora = define_uint(defines, "MLA_Q_LORA_RANK")
        self.kv_lora = define_uint(defines, "MLA_LATENT_DIMENSION")
        self.qk_dim = define_uint(defines, "MLA_QK_HEAD_DIMENSION")
        self.nope = define_uint(defines, "MLA_QK_NOPE_HEAD_DIMENSION")
        self.rot = define_uint(defines, "MLA_QK_ROPE_HEAD_DIMENSION")
        self.v_dim = define_uint(defines, "MLA_VALUE_HEAD_DIMENSION")
        self.hc = define_uint(defines, "HC_STREAM_COUNT")
        self.hc_eps = define_float(defines, "HC_EPSILON")
        self.hc_magnitude = define_float(defines, "HC_MAGNITUDE")
        self.experts = define_uint(defines, "MOE_EXPERT_COUNT")
        self.top_k = define_uint(defines, "MOE_TOP_K")
        self.shared_count = define_uint(defines, "MOE_SHARED_EXPERT_COUNT")
        self.moe_inter = define_uint(defines, "MOE_INTERMEDIATE_DIMENSION")
        self.route_scale = define_float(defines, "MOE_ROUTED_SCALING_FACTOR")
        self.norm_topk = define_uint(defines, "MOE_NORM_TOPK_PROB")
        self.dense_inter = define_uint(defines, "DENSE_INTERMEDIATE_DIMENSION")
        self.swiglu_limit = define_float(defines, "SWIGLU_LIMIT")
        self.eot = define_uint(defines, "END_OF_TEXT_TOKEN_ID")
        self.cache = {}
        self.kq_scale = 1.0 / np.sqrt(np.float32(self.qk_dim))
        self.rope_theta = define_float(defines, "MLA_ROPE_THETA")
        self.rope_freqs = np.power(
            np.float32(self.rope_theta),
            -np.arange(0, self.rot, 2, dtype=np.float32) / self.rot)
        self.sinks = self._plane_f32("model.layers.0.self_attn."
                                     "learnable_sink_param", (self.heads,))
        self._geometry_checks()

    def _plane_f32(self, name, shape):
        raw = self.st.raw(name)
        if raw.dtype != np.float32:
            raise ValueError(f"{name} must be F32, got {raw.dtype}")
        if tuple(raw.shape) != shape:
            raise ValueError(f"{name} shape {raw.shape} disagrees with "
                             f"{shape}")
        return raw

    def _geometry_checks(self):
        q_b = self.st.entry(f"{PREFIX}0.self_attn.q_b_proj.weight")
        if list(q_b["shape"]) != [self.heads * self.qk_dim, self.q_lora]:
            raise Hy4ConfigError(
                f"q_b_proj {q_b['shape']} disagrees with heads*qk "
                f"{self.heads * self.qk_dim} x q_lora {self.q_lora}")
        kv_b = self.st.entry(f"{PREFIX}0.self_attn.kv_b_proj.weight")
        per_head = kv_b["shape"][0] // self.heads
        if per_head != self.nope + self.v_dim:
            raise Hy4ConfigError(
                f"kv_b_proj per-head stride {per_head} disagrees with "
                f"nope+v {self.nope + self.v_dim}")
        if kv_b["shape"][1] != self.kv_lora:
            raise Hy4ConfigError(
                f"kv_b_proj {kv_b['shape']} disagrees with kv_lora "
                f"{self.kv_lora}")
        gu = self.st.entry(f"{PREFIX}1.mlp.experts.gate_up_proj")
        if list(gu["shape"]) != [self.experts, 2 * self.moe_inter,
                                 self.hidden]:
            raise Hy4ConfigError(
                f"gate_up_proj {gu['shape']} disagrees with experts "
                f"{self.experts} x 2*inter {2 * self.moe_inter} x hidden "
                f"{self.hidden}")
        hc_fn = self.st.entry(f"{PREFIX}0.hc_attn_layer.hc_pre.hc_fn")
        if list(hc_fn["shape"]) != [2 * self.hc, self.hc * self.hidden]:
            raise Hy4ConfigError(
                f"hc_attn fn {hc_fn['shape']} disagrees with 2*hc "
                f"{2 * self.hc} x hc*hidden {self.hc * self.hidden}")
        head_fn = self.st.entry("model.hc_head.hc_head_fn")
        if list(head_fn["shape"]) != [self.hc, self.hc * self.hidden]:
            raise Hy4ConfigError(
                f"hc_head fn {head_fn['shape']} disagrees with hc "
                f"{self.hc} x hc*hidden {self.hc * self.hidden}")

    def _weight(self, name):
        if name not in self.cache:
            self.cache[name] = self.st.raw(name)
        return self.cache[name]

    def _plane_bf16(self, name):
        raw = self._weight(name)
        if raw.dtype != np.uint16:
            raise ValueError(f"{name} must be BF16, got {raw.dtype}")
        return bf16_to_f32(raw)

    def _matmul_fp8(self, name, vector):
        payload = self._weight(name + ".weight")
        if payload.dtype != np.uint8:
            raise ValueError(f"{name} payload must be F8_E4M3 bytes")
        scale = self._weight(name + ".weight_scale")
        rows = payload.shape[0]
        out = np.empty(rows, dtype=np.float32)
        block = 8192
        for start in range(0, rows, block):
            stop = min(start + block, rows)
            plane = dequant_fp8(payload[start:stop], scale[start:stop])
            out[start:stop] = plane @ vector
        return out

    def _expert(self, il, kind, expert, vector):
        payload = self.st.raw_slab(f"{PREFIX}{il}.mlp.experts.{kind}", expert, 1)
        scale = self.st.raw_slab(f"{PREFIX}{il}.mlp.experts.{kind}_scale",
                                 expert, 1)
        return dequant_fp8(payload[0], scale[0]) @ vector

    def _rope_(self, value, position):
        half = value.shape[-1] // 2
        angles = np.float32(position) * self.rope_freqs
        cos = np.cos(angles)
        sin = np.sin(angles)
        a = value[..., :half].copy()
        b = value[..., half:].copy()
        value[..., :half] = a * cos - b * sin
        value[..., half:] = a * sin + b * cos

    def _hc_pre(self, streams, fn, scale, base):
        flat = streams.reshape(-1).astype(np.float32)
        flat_norm = flat / np.sqrt(np.mean(flat * flat) + self.eps)
        mixes = fn @ flat_norm
        count = self.hc
        pre = sigmoid(mixes[:count] * float(scale[0]) +
                      base[:count].astype(np.float32)) + self.hc_eps
        post = sigmoid(mixes[count:] * float(scale[1]) +
                       base[count:].astype(np.float32)) * self.hc_magnitude \
            + self.hc_eps
        return pre, post

    def _hc_post(self, streams, branch, post):
        return streams + branch[None, :] * post[:, None]

    def _attention(self, il, position, cur, caches):
        p = f"{PREFIX}{il}.self_attn."
        qr = self._matmul_fp8(p + "q_a_proj", cur)
        qr = rmsnorm(qr, self._plane_bf16(p + "q_a_layernorm.weight"), self.eps)
        q = self._matmul_fp8(p + "q_b_proj", qr).reshape(self.heads, self.qk_dim)
        q_pe = q[:, self.nope:].copy()
        self._rope_(q_pe, position)
        kv = self._matmul_fp8(p + "kv_a_proj_with_mqa", cur)
        k_pe = kv[self.kv_lora:].copy()
        self._rope_(k_pe, position)
        k_latent = rmsnorm(kv[:self.kv_lora],
                           self._plane_bf16(p + "kv_a_layernorm.weight"),
                           self.eps)
        if il not in caches:
            caches[il] = ([], [])
        caches[il][0].append(k_latent)
        caches[il][1].append(k_pe)
        latents = np.stack(caches[il][0])
        pes = np.stack(caches[il][1])
        kv_b = dequant_fp8(self._weight(p + "kv_b_proj.weight"),
                           self._weight(p + "kv_b_proj.weight_scale"))
        kv_b = kv_b.reshape(self.heads, self.nope + self.v_dim, self.kv_lora)
        q_abs = np.einsum("hnl,hn->hl", kv_b[:, :self.nope, :],
                          q[:, :self.nope], optimize=True)
        scores = (np.einsum("hk,tk->ht", q_abs, latents, optimize=True) +
                  np.einsum("hk,tk->ht", q_pe, pes, optimize=True)) \
            * self.kq_scale
        ceiling = np.maximum(scores.max(axis=1),
                             self.sinks.astype(np.float32))
        weights = np.exp(scores - ceiling[:, None])
        denominator = weights.sum(axis=1) + np.exp(
            self.sinks.astype(np.float32) - ceiling)
        probs = weights / denominator[:, None]
        context = np.einsum("ht,tn->hn", probs, latents, optimize=True)
        head_out = np.einsum("ht,thv->hv", probs,
                             np.einsum("tn,hvn->thv", latents,
                                       kv_b[:, self.nope:, :],
                                       optimize=True),
                             optimize=True)
        gate = sigmoid(self._plane_bf16(p + "linear_gate.weight") @ cur)
        return self._matmul_fp8(p + "o_proj", head_out.reshape(-1) * gate)

    def _dense_ffn(self, il, cur):
        p = f"{PREFIX}{il}.mlp."
        gate = self._matmul_fp8(p + "gate_proj", cur)
        up = self._matmul_fp8(p + "up_proj", cur)
        activated = gate / (1.0 + np.exp(-gate)) * up
        return self._matmul_fp8(p + "down_proj", activated)

    def _routed_expert(self, il, expert, cur):
        gu = self._expert(il, "gate_up_proj", expert, cur)
        gate = gu[:self.moe_inter]
        up = gu[self.moe_inter:]
        np.clip(up, -self.swiglu_limit, self.swiglu_limit, out=up)
        activated = gate / (1.0 + np.exp(-gate))
        np.clip(activated, None, self.swiglu_limit, out=activated)
        return self._expert(il, "down_proj", expert, activated * up)

    def _shared_expert(self, il, cur):
        p = f"{PREFIX}{il}.mlp.shared_experts."
        gate = self._matmul_fp8(p + "gate_proj", cur)
        up = self._matmul_fp8(p + "up_proj", cur)
        activated = gate / (1.0 + np.exp(-gate)) * up
        return self._matmul_fp8(p + "down_proj", activated)

    def _moe(self, il, position, cur, capture):
        p = f"{PREFIX}{il}.mlp."
        probs = sigmoid(self._plane_bf16(p + "gate.weight") @ cur)
        key = probs + self._plane_f32(p + "gate.e_score_correction_bias",
                                      (self.experts,))
        order = np.argsort(-key, kind="stable")
        selected = order[:self.top_k]
        weights = probs[selected]
        if self.norm_topk:
            total = float(weights.sum())
            if total < WSUM_FLOOR:
                total = WSUM_FLOOR
            weights = weights / total * self.route_scale
        ffn = np.zeros(self.hidden, dtype=np.float32)
        for k in range(self.top_k):
            ffn += weights[k] * self._routed_expert(il, int(selected[k]), cur)
        ffn += self._shared_expert(il, cur)
        capture[(position, il)] = (selected.astype(np.int32),
                                   weights.astype(np.float32))
        return ffn

    def decode_step(self, token_id, position, states, caches, capture):
        if token_id < 0 or token_id >= self.vocab:
            raise ValueError(f"token {token_id} outside vocabulary "
                             f"{self.vocab}")
        if position + 1 > MAX_INDEXER_CONTEXT:
            raise ValueError(
                f"position {position} exceeds the {MAX_INDEXER_CONTEXT}-token "
                f"bound where the DSA indexer selection is the identity")
        embedding = bf16_to_f32(
            self._weight("model.embed_tokens.weight")[token_id])
        if position == 0:
            states["streams"] = np.tile(embedding, (self.hc, 1))
            caches.clear()
        streams = states["streams"]
        for il in range(self.layers):
            base = f"{PREFIX}{il}."
            pre, post = self._hc_pre(
                streams,
                self._plane_f32(base + "hc_attn_layer.hc_pre.hc_fn",
                                (2 * self.hc, self.hc * self.hidden)),
                self._plane_f32(base + "hc_attn_layer.hc_pre.hc_scale", (2,)),
                self._plane_f32(base + "hc_attn_layer.hc_pre.hc_base",
                                (2 * self.hc,)))
            cur = (streams * pre[:, None]).sum(axis=0)
            cur = rmsnorm(cur,
                          self._plane_bf16(base + "input_layernorm.weight"),
                          self.eps)
            attn = self._attention(il, position, cur, caches)
            streams = self._hc_post(streams, attn, post)
            pre, post = self._hc_pre(
                streams,
                self._plane_f32(base + "hc_mlp_layer.hc_pre.hc_fn",
                                (2 * self.hc, self.hc * self.hidden)),
                self._plane_f32(base + "hc_mlp_layer.hc_pre.hc_scale", (2,)),
                self._plane_f32(base + "hc_mlp_layer.hc_pre.hc_base",
                                (2 * self.hc,)))
            cur = (streams * pre[:, None]).sum(axis=0)
            cur = rmsnorm(
                cur, self._plane_bf16(base + "post_attention_layernorm.weight"),
                self.eps)
            ffn = self._dense_ffn(il, cur) if il == 0 else self._moe(
                il, position, cur, capture)
            streams = self._hc_post(streams, ffn, post)
        states["streams"] = streams
        return streams

    def logits(self, streams, chunk=4096):
        flat = streams.reshape(-1).astype(np.float32)
        flat_norm = flat / np.sqrt(np.mean(flat * flat) + self.eps)
        head_fn = self._plane_f32("model.hc_head.hc_head_fn",
                                  (self.hc, self.hc * self.hidden))
        head_scale = self._plane_f32("model.hc_head.hc_head_scale", (1,))
        head_base = self._plane_f32("model.hc_head.hc_head_base",
                                    (self.hc,))
        mixes = head_fn @ flat_norm
        pre = sigmoid(mixes * float(head_scale.reshape(-1)[0]) +
                      head_base) + self.hc_eps
        collapsed = (streams * pre[:, None]).sum(axis=0)
        normed = rmsnorm(collapsed, self._plane_bf16("model.norm.weight"),
                         self.eps)
        lm = self._weight("lm_head.weight")
        if lm.dtype != np.float32:
            raise ValueError("reference lm_head must be F32")
        best = -np.inf
        best_token = -1
        for start in range(0, lm.shape[0], chunk):
            stop = min(start + chunk, lm.shape[0])
            scores = lm[start:stop] @ normed
            i = int(np.argmax(scores))
            if float(scores[i]) > best:
                best = float(scores[i])
                best_token = start + i
        return best_token, best


ENGINE_CLASS = Hy4Engine
