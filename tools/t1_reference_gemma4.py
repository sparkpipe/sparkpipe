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
class Gemma4ConfigError(ValueError):
    pass


def cross_check(defines, config):
    for dname, cname, kind in DEFINES_VS_CONFIG:
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

    def tensor(self, name):
        raw = self.st.raw(name)
        if raw.dtype == np.uint16:
            return bf16_to_f32(raw)
        if raw.dtype == np.uint8:
            scale = self.st.raw(name + "_scale_inv").astype(np.float32)
            rows, cols = raw.shape
            return bf16_to_f32(fp8_block_to_bf16(raw, scale, rows, cols))
        return raw.astype(np.float32)

    def linear(self, x, name):
        return bf16_round_f32(self.tensor(name + ".weight") @ x)

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
        cube = gate * gate * gate
        activated = bf16_round_f32((0.5 * gate
                                    * (1.0 + np.tanh(0.7978845608028654
                                                     * (gate + 0.044715 * cube))))
                                   * up)
        return self.linear(activated, p + "down_proj")

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
        lm = self.st.raw("model.language_model.embed_tokens.weight")
        if lm.dtype != np.uint16:
            raise ValueError("reference embedding must be BF16")
        best = -np.inf
        best_token = -1
        for start in range(0, lm.shape[0], chunk):
            scores = bf16_to_f32(lm[start:start + chunk]) @ norm
            i = int(np.argmax(scores))
            if float(scores[i]) > best:
                best = float(scores[i])
                best_token = start + i
        return best_token, best


ENGINE_CLASS = Gemma4Engine
