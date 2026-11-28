import numpy as np

from t1_reference_common import (Safetensors, bf16_round_f32, bf16_to_f32,
                                 define_float, define_uint, f32_to_bf16_u16,
                                 fp8_block_to_bf16, rmsnorm, sigmoid)

PREFIX = "model.language_model.layers."
SLIDING_WINDOW = 1024
DEFINES_VS_CONFIG = [
    ("HIDDEN_DIMENSION", "hidden_size", "uint"),
    ("LAYER_COUNT", "num_hidden_layers", "uint"),
    ("OUTPUT_VOCAB_COUNT", "vocab_size", "uint"),
    ("RMS_NORM_EPSILON", "rms_norm_eps", "float"),
    ("END_OF_TEXT_TOKEN_ID", "eos_token_id", "uint"),
    ("SLIDING_KV_HEAD_COUNT", "num_key_value_heads", "uint"),
    ("SLIDING_HEAD_DIMENSION", "head_dim", "uint"),
    ("DENSE_INTERMEDIATE_DIMENSION", "intermediate_size", "uint"),
]
MOE_DEFINES_VS_CONFIG = [
    ("ROUTED_EXPERT_COUNT", "num_experts", "uint"),
    ("EXPERTS_PER_TOKEN", "top_k_experts", "uint"),
    ("EXPERT_INTERMEDIATE_DIMENSION", "moe_intermediate_size", "uint"),
]
class Gemma4ConfigError(ValueError):
    pass


def cross_check(defines, config):
    checks = list(DEFINES_VS_CONFIG)
    if config.get("enable_moe_block"):
        checks += MOE_DEFINES_VS_CONFIG
    for dname, cname, kind in checks:
        if cname not in config:
            raise Gemma4ConfigError(f"config key {cname} missing for SPARK_LLM_{dname}")
        want = define_uint(defines, dname) if kind == "uint" \
            else define_float(defines, dname)
        got = config[cname]
        if isinstance(got, list):
            got = got[0]
        if abs(float(want) - float(got)) > 0:
            raise Gemma4ConfigError(
                f"SPARK_LLM_{dname}={want} disagrees with config {cname}={got}")
    layer_types = config.get("layer_types")
    if not isinstance(layer_types, list) or \
            len(layer_types) != define_uint(defines, "LAYER_COUNT"):
        raise Gemma4ConfigError("config layer_types missing or wrong length")
    full = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    period = define_uint(defines, "FULL_LAYER_PERIOD")
    phase = define_uint(defines, "FULL_LAYER_PHASE")
    expected = [i for i in range(len(layer_types)) if i % period == phase]
    if full != expected:
        raise Gemma4ConfigError(
            f"config full_attention layers {full} disagree with phase pattern "
            f"{expected}")
    return []


class Gemma4Engine:
    def __init__(self, checkpoint_dir, defines, config):
        self.mismatches = cross_check(defines, config)
        self.st = Safetensors(checkpoint_dir)
        self.hidden = int(config["hidden_size"])
        self.layers = int(config["num_hidden_layers"])
        self.vocab = int(config["vocab_size"])
        self.eps = float(config["rms_norm_eps"])
        self.layer_types = list(config["layer_types"])
        self.s_q_heads = define_uint(defines, "SLIDING_QUERY_HEAD_COUNT")
        self.s_kv_heads = define_uint(defines, "SLIDING_KV_HEAD_COUNT")
        self.s_dim = define_uint(defines, "SLIDING_HEAD_DIMENSION")
        self.f_q_heads = define_uint(defines, "FULL_QUERY_HEAD_COUNT")
        self.f_kv_heads = define_uint(defines, "FULL_KV_HEAD_COUNT")
        self.f_dim = define_uint(defines, "FULL_HEAD_DIMENSION")
        self.rotated = define_uint(defines, "FULL_ROTATED_PAIR_COUNT")
        self.qk_scale = define_float(defines, "QK_SCALE")
        self.embed_scale = np.float32(define_float(defines, "EMBED_SCALE"))
        self.inter = define_uint(defines, "DENSE_INTERMEDIATE_DIMENSION")
        self.eot = define_uint(defines, "END_OF_TEXT_TOKEN_ID")
        self.moe = bool(config.get("enable_moe_block"))
        self.expert_count = 0
        self.top_k = 0
        self.expert_inter = 0
        self.router = {}
        self.tables = {}
        self.caches = {}
        self.layer_trace = {}
        for kind, base, dim, pairs in (
            ("sliding", define_float(defines, "SLIDING_ROPE_THETA"),
             self.s_dim, self.s_dim // 2),
            ("full", define_float(defines, "FULL_ROPE_BASE"),
             self.f_dim, self.rotated),
        ):
            exponent = 2.0 * np.arange(dim // 2, dtype=np.float32) \
                / np.float32(dim)
            inv_freq = np.where(np.arange(dim // 2) < pairs,
                                np.power(np.float32(base), -exponent),
                                np.float32(0.0)).astype(np.float32)
            self.tables[kind] = inv_freq
        q = self.st.entry(f"{PREFIX}0.self_attn.q_proj.weight")
        if q["shape"][0] != self.s_q_heads * self.s_dim:
            raise Gemma4ConfigError(
                f"layer 0 q_proj rows {q['shape'][0]} disagree with sliding "
                f"q heads x dim {self.s_q_heads * self.s_dim}")
        k = self.st.entry(f"{PREFIX}0.self_attn.k_proj.weight")
        v = self.st.entry(f"{PREFIX}0.self_attn.v_proj.weight")
        if k["shape"] != v["shape"]:
            raise Gemma4ConfigError("k_proj and v_proj shapes disagree")
        scalar = self.st.entry(f"{PREFIX}0.layer_scalar")
        if len(scalar["shape"]) != 1 or scalar["shape"][0] != 1:
            raise Gemma4ConfigError("layer_scalar must be a single element")
        if self.moe:
            self.expert_count = define_uint(defines, "ROUTED_EXPERT_COUNT")
            self.top_k = define_uint(defines, "EXPERTS_PER_TOKEN")
            self.expert_inter = define_uint(
                defines, "EXPERT_INTERMEDIATE_DIMENSION")
            for i in range(self.layers):
                base = f"{PREFIX}{i}."
                gate_up = self.st.entry(base + "experts.gate_up_proj")
                if gate_up["shape"] != [self.expert_count,
                                        2 * self.expert_inter, self.hidden]:
                    raise Gemma4ConfigError(
                        f"layer {i} experts.gate_up_proj shape "
                        f"{gate_up['shape']} disagrees with routed_experts x "
                        f"2 x expert_inter x hidden")
                down = self.st.entry(base + "experts.down_proj")
                if down["shape"] != [self.expert_count, self.hidden,
                                     self.expert_inter]:
                    raise Gemma4ConfigError(
                        f"layer {i} experts.down_proj shape "
                        f"{down['shape']} disagrees with routed_experts x "
                        f"hidden x expert_inter")
                proj = self.st.entry(base + "router.proj.weight")
                if proj["shape"] != [self.expert_count, self.hidden]:
                    raise Gemma4ConfigError(
                        f"layer {i} router.proj.weight shape "
                        f"{proj['shape']} disagrees with routed_experts x "
                        f"hidden")
                for norm in ("post_feedforward_layernorm_1.weight",
                             "pre_feedforward_layernorm_2.weight",
                             "post_feedforward_layernorm_2.weight"):
                    entry = self.st.entry(base + norm)
                    if entry["shape"] != [self.hidden]:
                        raise Gemma4ConfigError(
                            f"layer {i} {norm} shape {entry['shape']} "
                            f"disagrees with hidden")
                scale = self.st.pread(base + "router.scale")
                if scale.dtype != np.uint16 or scale.shape != (self.hidden,):
                    raise Gemma4ConfigError(
                        f"layer {i} router.scale must be BF16 hidden vector")
                per_expert = self.st.pread(base + "router.per_expert_scale")
                if per_expert.dtype != np.uint16 \
                        or per_expert.shape != (self.expert_count,):
                    raise Gemma4ConfigError(
                        f"layer {i} router.per_expert_scale must be BF16 "
                        f"routed_experts vector")
                self.router[i] = (
                    bf16_to_f32(self.st.pread(base + "router.proj.weight"))
                    .astype(np.float32),
                    bf16_to_f32(scale).astype(np.float32)
                    * np.float32(self.hidden) ** np.float32(-0.5),
                    bf16_to_f32(per_expert).astype(np.float32))

    def tensor(self, name):
        raw = self.st.pread(name)
        if raw.dtype == np.uint16:
            return bf16_to_f32(raw)
        if raw.dtype == np.uint8:
            scale = self.st.pread(name + "_scale_inv").astype(np.float32)
            rows, cols = raw.shape
            return bf16_to_f32(fp8_block_to_bf16(raw, scale, rows, cols))
        return raw.astype(np.float32)

    def linear(self, x, name):
        return bf16_round_f32(self.tensor(name + ".weight") @ x)

    def gelu_mul(self, gate, up):
        cube = gate * gate * gate
        return bf16_round_f32((0.5 * gate
                               * (1.0 + np.tanh(0.7978845608028654
                                                * (gate + 0.044715 * cube))))
                              * up)

    def expert_gate_up(self, index, expert):
        raw = self.st.raw_slab(f"{PREFIX}{index}.experts.gate_up_proj",
                               expert, 1)
        if raw.dtype != np.uint16:
            raise ValueError("reference expert gate_up must be BF16")
        return bf16_to_f32(raw[0])

    def expert_down(self, index, expert):
        raw = self.st.raw_slab(f"{PREFIX}{index}.experts.down_proj",
                               expert, 1)
        if raw.dtype != np.uint16:
            raise ValueError("reference expert down must be BF16")
        return bf16_to_f32(raw[0])

    def embed(self, token_id):
        raw = self.st.raw_rows("model.language_model.embed_tokens.weight",
                               token_id, 1)
        if raw.dtype != np.uint16:
            raise ValueError("reference embedding must be BF16")
        return bf16_round_f32(bf16_to_f32(raw[0]) * self.embed_scale)

    def cos_sin(self, kind, position):
        inv_freq = self.tables[kind]
        angles = np.float32(position) * inv_freq
        cos = bf16_round_f32(np.cos(angles))
        sin = bf16_round_f32(np.sin(angles))
        return np.tile(cos, 2).astype(np.float32), np.tile(sin, 2).astype(np.float32)

    def rope(self, rows, cos, sin):
        dim = rows.shape[-1]
        half = dim // 2
        rotated = np.concatenate([-rows[..., half:], rows[..., :half]], axis=-1)
        out = bf16_round_f32(bf16_round_f32(rows * cos)
                             + bf16_round_f32(rotated * sin))
        return out

    def head_rms(self, rows, weight, heads, dim, scaled=True):
        m = rows.reshape(-1, heads, dim)
        rms = np.sqrt((m * m).sum(axis=-1, keepdims=True) / dim + self.eps)
        normed = m / rms
        if scaled:
            normed = normed * weight.reshape(1, 1, dim)
        return bf16_round_f32(normed.reshape(rows.shape))

    def attention(self, index, x, position, cache):
        p = f"{PREFIX}{index}.self_attn."
        sliding = self.layer_types[index] == "sliding_attention"
        q_heads, kv_heads, dim = ((self.s_q_heads, self.s_kv_heads, self.s_dim)
                                  if sliding else
                                  (self.f_q_heads, self.f_kv_heads, self.f_dim))
        kind = "sliding" if sliding else "full"
        cos, sin = self.cos_sin(kind, position)
        q = self.head_rms(self.linear(x, p + "q_proj"), self.tensor(p + "q_norm.weight"),
                          q_heads, dim)
        k_raw = self.linear(x, p + "k_proj")
        if sliding:
            v_raw = self.linear(x, p + "v_proj")
        else:
            v_raw = k_raw
        v = self.head_rms(v_raw, None, kv_heads, dim, scaled=False)
        k = self.head_rms(k_raw, self.tensor(p + "k_norm.weight"), kv_heads, dim)
        q = self.rope(q.reshape(q_heads, dim), cos, sin)
        k = self.rope(k.reshape(kv_heads, dim), cos, sin)
        cache.append((f32_to_bf16_u16(k.reshape(-1)),
                      f32_to_bf16_u16(v.reshape(-1))))
        keys = bf16_to_f32(np.stack([row[0] for row in cache])) \
            .reshape(len(cache), kv_heads, dim)
        values = bf16_to_f32(np.stack([row[1] for row in cache])) \
            .reshape(len(cache), kv_heads, dim)
        window_lo = max(0, len(cache) - SLIDING_WINDOW) if sliding else 0
        group = q_heads // kv_heads
        out = np.empty((q_heads, dim), dtype=np.float32)
        for h in range(q_heads):
            kvh = h // group
            scores = bf16_round_f32((keys[:, kvh, :] @ q[h])
                                    * np.float32(self.qk_scale))
            if window_lo > 0:
                scores[:window_lo] = -np.inf
            weights = np.exp(scores - scores.max())
            weights = weights / weights.sum()
            probs = bf16_round_f32(weights)
            out[h] = bf16_round_f32(probs @ values[:, kvh, :])
        return self.linear(out.reshape(-1), p + "o_proj")

    def mlp(self, index, x):
        p = f"{PREFIX}{index}.mlp."
        gate = self.linear(x, p + "gate_proj")
        up = self.linear(x, p + "up_proj")
        return self.linear(self.gelu_mul(gate, up), p + "down_proj")

    def moe_branch(self, index, streams, routed_input):
        proj, gain, per_expert_scale = self.router[index]
        probe = self.head_rms(streams, None, 1, self.hidden, scaled=False)
        scores = proj @ bf16_round_f32(probe * gain)
        probs = np.exp(scores - scores.max())
        probs = probs / probs.sum()
        top = np.argsort(-probs, kind="stable")[:self.top_k]
        weights = probs[top]
        weights = weights / weights.sum()
        weights = weights * per_expert_scale[top]
        pairs = sorted(zip((int(e) for e in top),
                           (float(w) for w in weights)))
        branch = np.zeros(self.hidden, dtype=np.float32)
        for expert, weight in pairs:
            gate_up = self.expert_gate_up(index, expert) @ routed_input
            half = self.expert_inter
            activated = self.gelu_mul(gate_up[:half], gate_up[half:])
            out = self.expert_down(index, expert) @ activated
            branch = bf16_round_f32(branch + bf16_round_f32(
                np.float32(weight) * out))
        return branch

    def forward_layer(self, index, streams, position):
        p = f"{PREFIX}{index}."
        x = bf16_round_f32(rmsnorm(streams,
                                   self.tensor(p + "input_layernorm.weight"),
                                   self.eps))
        attention = self.attention(index, x, position, self.caches[index])
        attended = bf16_round_f32(rmsnorm(
            attention, self.tensor(p + "post_attention_layernorm.weight"),
            self.eps))
        streams = bf16_round_f32(streams + attended)
        x = bf16_round_f32(rmsnorm(streams,
                                   self.tensor(p + "pre_feedforward_layernorm.weight"),
                                   self.eps))
        if self.moe:
            dense = self.mlp(index, x)
            dense = bf16_round_f32(rmsnorm(
                dense, self.tensor(p + "post_feedforward_layernorm_1.weight"),
                self.eps))
            routed = bf16_round_f32(rmsnorm(
                streams, self.tensor(p + "pre_feedforward_layernorm_2.weight"),
                self.eps))
            branch = self.moe_branch(index, streams, routed)
            branch = bf16_round_f32(rmsnorm(
                branch, self.tensor(p + "post_feedforward_layernorm_2.weight"),
                self.eps))
            mlp = bf16_round_f32(dense + branch)
        else:
            mlp = self.mlp(index, x)
        delta = bf16_round_f32(rmsnorm(
            mlp, self.tensor(p + "post_feedforward_layernorm.weight"), self.eps))
        streams = bf16_round_f32(streams + delta)
        scalar = self.tensor(p + "layer_scalar")[0]
        streams = bf16_round_f32(streams * scalar)
        self.layer_trace[index] = streams
        return streams

    def decode_step(self, token_id, position, states, caches, capture):
        if token_id < 0 or token_id >= self.vocab:
            raise ValueError(f"token {token_id} outside vocabulary {self.vocab}")
        if position == 0:
            self.caches = {}
            for i in range(self.layers):
                self.caches[i] = []
        streams = self.embed(token_id)
        for i in range(self.layers):
            streams = self.forward_layer(i, streams, position)
            if not np.isfinite(streams).all():
                raise ValueError(f"nonfinite reference state at layer {i}")
        return streams

    def logits(self, streams, chunk=4096):
        norm = bf16_round_f32(rmsnorm(
            streams, self.tensor("model.language_model.norm.weight"), self.eps))
        entry = self.st.entry("model.language_model.embed_tokens.weight")
        if entry["dtype"] != "BF16":
            raise ValueError("reference embedding must be BF16")
        rows_total = entry["shape"][0]
        best = -np.inf
        best_token = -1
        for start in range(0, rows_total, chunk):
            count = min(chunk, rows_total - start)
            lm = self.st.raw_rows("model.language_model.embed_tokens.weight", start, count)
            scores = bf16_to_f32(lm) @ norm
            i = int(np.argmax(scores))
            if float(scores[i]) > best:
                best = float(scores[i])
                best_token = start + i
        return best_token, best


ENGINE_CLASS = Gemma4Engine
