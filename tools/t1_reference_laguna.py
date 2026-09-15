import numpy as np

from t1_reference_common import (Safetensors, bf16_round_f32, bf16_to_f32,
                                 define_float, define_uint, f32_to_bf16_u16,
                                 fp8_block_to_bf16, rmsnorm, sigmoid)

PREFIX = "model.layers."
ATTENTION_SCALE = 0.08838834764831845
SLIDING_ROPE_THETA = 10000.0
DEFINES_VS_CONFIG = [
    ("HIDDEN_DIMENSION", "hidden_size", "uint"),
    ("LAYER_COUNT", "num_hidden_layers", "uint"),
    ("OUTPUT_VOCAB_COUNT", "vocab_size", "uint"),
    ("RMS_NORM_EPSILON", "rms_norm_eps", "float"),
    ("END_OF_TEXT_TOKEN_ID", "eos_token_id", "uint"),
    ("ATTENTION_HEAD_DIMENSION", "head_dim", "uint"),
    ("ATTENTION_KV_HEAD_COUNT", "num_key_value_heads", "uint"),
    ("SLIDING_WINDOW", "sliding_window", "uint"),
    ("MOE_EXPERT_COUNT", "num_experts", "uint"),
    ("MOE_TOP_K", "num_experts_per_tok", "uint"),
    ("MOE_INTERMEDIATE_DIMENSION", "moe_intermediate_size", "uint"),
    ("MOE_ROUTED_SCALING_FACTOR", "moe_routed_scaling_factor", "float"),
    ("DENSE_INTERMEDIATE_DIMENSION", "intermediate_size", "uint"),
]


class LagunaConfigError(ValueError):
    pass


def softplus(x):
    return np.logaddexp(np.float32(0), x)


def cross_check(defines, config):
    for dname, cname, kind in DEFINES_VS_CONFIG:
        if cname not in config:
            raise LagunaConfigError(f"config key {cname} missing for SPARK_LLM_{dname}")
        want = define_uint(defines, dname) if kind == "uint" \
            else define_float(defines, dname)
        got = config[cname]
        if isinstance(got, list):
            got = got[0]
        if abs(float(want) - float(got)) > 0:
            raise LagunaConfigError(
                f"SPARK_LLM_{dname}={want} disagrees with config {cname}={got}")
    layer_types = config.get("layer_types")
    if not isinstance(layer_types, list) or \
            len(layer_types) != define_uint(defines, "LAYER_COUNT"):
        raise LagunaConfigError("config layer_types missing or wrong length")
    rope = config.get("rope_parameters") or {}
    full = rope.get("full_attention") or {}
    checks = [
        ("ROPE_FULL_THETA", "rope_theta", "float"),
        ("ROPE_FULL_FACTOR", "factor", "float"),
        ("ROPE_FULL_ORIGINAL_POSITIONS", "original_max_position_embeddings",
         "float"),
        ("ROPE_FULL_BETA_FAST", "beta_fast", "float"),
        ("ROPE_FULL_BETA_SLOW", "beta_slow", "float"),
        ("ROPE_FULL_ATTENTION_FACTOR", "attention_factor", "float"),
    ]
    for dname, cname, kind in checks:
        if cname not in full:
            raise LagunaConfigError(
                f"config key full_attention.{cname} missing for SPARK_LLM_{dname}")
        want = define_uint(defines, dname) if kind == "uint" \
            else define_float(defines, dname)
        if abs(float(want) - float(full[cname])) > 0:
            raise LagunaConfigError(
                f"SPARK_LLM_{dname}={want} disagrees with config "
                f"full_attention.{cname}={full[cname]}")
    return []


class LagunaEngine:
    def __init__(self, checkpoint_dir, defines, config):
        self.mismatches = cross_check(defines, config)
        self.st = Safetensors(checkpoint_dir)
        self.hidden = int(config["hidden_size"])
        self.layers = int(config["num_hidden_layers"])
        self.vocab = int(config["vocab_size"])
        self.eps = float(config["rms_norm_eps"])
        self.layer_types = list(config["layer_types"])
        self.head_dim = int(config["head_dim"])
        self.kv_heads = int(config["num_key_value_heads"])
        self.q_heads_full = define_uint(defines, "Q_HEAD_COUNT_FULL")
        self.q_heads_sliding = define_uint(defines, "Q_HEAD_COUNT_SLIDING")
        self.window = int(config["sliding_window"])
        self.scale = np.float32(ATTENTION_SCALE)
        self.experts = int(config["num_experts"])
        self.topk = int(config["num_experts_per_tok"])
        self.scaling = float(config["moe_routed_scaling_factor"])
        self.mlp_only = set(int(i) for i in config.get("mlp_only_layers", []))
        mlp_types = config.get("mlp_layer_types")
        if mlp_types is not None:
            if len(mlp_types) != self.layers:
                raise LagunaConfigError("config mlp_layer_types wrong length")
            dense = set(i for i, t in enumerate(mlp_types) if t == "dense")
            if dense != self.mlp_only:
                raise LagunaConfigError(
                    "config mlp_layer_types disagrees with mlp_only_layers")
        self.eot = int(config["eos_token_id"][0] if isinstance(
            config["eos_token_id"], list) else config["eos_token_id"])
        self.caches = {}
        self.rotary = define_uint(defines, "ROPE_FULL_ROTARY_DIMENSION")
        self.yarn_theta = define_float(defines, "ROPE_FULL_THETA")
        self.yarn_factor = define_float(defines, "ROPE_FULL_FACTOR")
        self.yarn_positions = define_float(defines, "ROPE_FULL_ORIGINAL_POSITIONS")
        self.yarn_beta_fast = define_float(defines, "ROPE_FULL_BETA_FAST")
        self.yarn_beta_slow = define_float(defines, "ROPE_FULL_BETA_SLOW")
        self.yarn_attention_factor = \
            np.float32(define_float(defines, "ROPE_FULL_ATTENTION_FACTOR"))
        self.yarn = self.yarn_table().astype(np.float32)
        self.sliding_freq = np.exp2(-(2.0 * np.arange(self.head_dim // 2,
                                                      dtype=np.float32)
                                     / np.float32(self.head_dim))
                                    * np.log2(np.float32(SLIDING_ROPE_THETA)))
        layer0_sliding = self.layer_types[0] == "sliding_attention"
        layer0_heads = self.q_heads_sliding if layer0_sliding             else self.q_heads_full
        q = self.st.entry(f"{PREFIX}0.self_attn.q_proj.weight")
        if q["shape"][0] != layer0_heads * self.head_dim:
            raise LagunaConfigError(
                f"layer 0 q_proj rows {q['shape'][0]} disagree with "
                f"q heads x dim {layer0_heads * self.head_dim}")
        gate = self.st.entry(f"{PREFIX}0.self_attn.g_proj.weight")
        if gate["shape"][0] != layer0_heads:
            raise LagunaConfigError(
                f"layer 0 g_proj rows {gate['shape'][0]} disagree with per-head "
                f"scalar gating {layer0_heads}")

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
        raw = self.st.raw_rows("model.embed_tokens.weight", token_id, 1)
        if raw.dtype != np.uint16:
            raise ValueError("reference embedding must be BF16")
        return bf16_to_f32(raw[0])

    def yarn_table(self):
        dim = self.rotary
        half = dim // 2
        low_exact = dim * np.log(self.yarn_positions
                                 / (self.yarn_beta_fast * 2.0 * np.pi)) \
            / (2.0 * np.log(self.yarn_theta))
        high_exact = dim * np.log(self.yarn_positions
                                  / (self.yarn_beta_slow * 2.0 * np.pi)) \
            / (2.0 * np.log(self.yarn_theta))
        low = np.floor(np.clip(low_exact, 0.0, dim - 1.0))
        high = np.ceil(np.clip(high_exact, 0.0, dim - 1.0))
        index = np.arange(half, dtype=np.float64)
        base = np.power(self.yarn_theta, -2.0 * index / dim)
        ramp = (index - low) / max(high - low, 1e-6)
        blend = np.clip(ramp, 0.0, 1.0)
        return base * (1.0 - blend) + (base / self.yarn_factor) * blend

    def cos_sin(self, sliding, position):
        if sliding:
            angles = np.float32(position) * self.sliding_freq
            factor = np.float32(1.0)
        else:
            angles = np.float32(position) * self.yarn
            factor = self.yarn_attention_factor
        return np.cos(angles) * factor, np.sin(angles) * factor

    def rope(self, rows, cos, sin):
        pairs = len(cos)
        low = rows[..., :pairs].copy()
        high = rows[..., pairs:2 * pairs].copy()
        rows[..., :pairs] = low * cos - high * sin
        rows[..., pairs:2 * pairs] = high * cos + low * sin
        return rows

    def head_rms(self, rows, weight, heads, dim):
        m = rows.reshape(-1, heads, dim)
        rms = np.sqrt((m * m).sum(axis=-1, keepdims=True) / dim + self.eps)
        normed = m / rms * weight.reshape(1, 1, dim)
        return bf16_round_f32(normed.reshape(rows.shape))

    def attention(self, index, x, position, cache):
        p = f"{PREFIX}{index}.self_attn."
        sliding = self.layer_types[index] == "sliding_attention"
        q_heads = self.q_heads_sliding if sliding else self.q_heads_full
        cos, sin = self.cos_sin(sliding, position)
        q = self.linear(x, p + "q_proj").reshape(q_heads, self.head_dim)
        k = self.linear(x, p + "k_proj").reshape(self.kv_heads, self.head_dim)
        v = self.linear(x, p + "v_proj").reshape(self.kv_heads, self.head_dim)
        q = self.rope(self.head_rms(q, self.tensor(p + "q_norm.weight"),
                                    q_heads, self.head_dim), cos, sin)
        k = self.rope(self.head_rms(k, self.tensor(p + "k_norm.weight"),
                                    self.kv_heads, self.head_dim), cos, sin)
        q = bf16_round_f32(q)
        k = bf16_round_f32(k)
        cache.append((f32_to_bf16_u16(k.reshape(-1)),
                      f32_to_bf16_u16(v.reshape(-1))))
        keys = bf16_to_f32(np.stack([row[0] for row in cache])) \
            .reshape(len(cache), self.kv_heads, self.head_dim)
        values = bf16_to_f32(np.stack([row[1] for row in cache])) \
            .reshape(len(cache), self.kv_heads, self.head_dim)
        window_lo = max(0, len(cache) - self.window) if sliding else 0
        group = q_heads // self.kv_heads
        out = np.empty((q_heads, self.head_dim), dtype=np.float32)
        for h in range(q_heads):
            kvh = h // group
            scores = (keys[:, kvh, :] @ q[h]) * self.scale
            if window_lo > 0:
                scores[:window_lo] = -np.inf
            weights = np.exp(scores - scores.max())
            weights = weights / weights.sum()
            out[h] = weights @ values[:, kvh, :]
        gate = softplus(self.linear(x, p + "g_proj").reshape(q_heads, 1))
        gated = bf16_round_f32((out * gate).reshape(-1))
        return self.linear(gated, p + "o_proj")

    def expert_mlp(self, prefix, x):
        gate = self.linear(x, prefix + "gate_proj")
        up = self.linear(x, prefix + "up_proj")
        activated = bf16_round_f32(bf16_round_f32(gate * sigmoid(gate)) * up)
        return self.linear(activated, prefix + "down_proj")

    def sparse_mlp(self, index, x, sink):
        p = f"{PREFIX}{index}.mlp."
        scores = sigmoid(self.tensor(p + "gate.weight") @ x)
        bias_name = f"{PREFIX}{index}.mlp.gate.e_score_correction_bias"
        if bias_name in self.st.map:
            bias = self.st.raw(bias_name).astype(np.float32).reshape(-1)
        else:
            bias = np.zeros(self.experts, dtype=np.float32)
        choice = scores + bias
        order = np.argsort(-choice, kind="stable")
        selected = np.sort(order[:self.topk])
        picked = scores[selected]
        weights = picked / picked.sum() * np.float32(self.scaling)
        routed = np.zeros_like(x)
        for i in range(self.topk):
            output = self.expert_mlp(p + f"experts.{int(selected[i])}.", x)
            routed = routed + output * np.float32(weights[i])
        shared = self.expert_mlp(p + "shared_expert.", x)
        sink.append(selected.astype(np.int32))
        sink.append(weights.astype(np.float32))
        return bf16_round_f32(routed + shared)

    def forward_layer(self, index, streams, position, sink):
        p = f"{PREFIX}{index}."
        x = bf16_round_f32(rmsnorm(streams,
                                   self.tensor(p + "input_layernorm.weight"),
                                   self.eps))
        attention = self.attention(index, x, position, self.caches[index])
        streams = bf16_round_f32(streams + attention)
        x = bf16_round_f32(rmsnorm(streams,
                                   self.tensor(p + "post_attention_layernorm.weight"),
                                   self.eps))
        if index in self.mlp_only:
            mlp = self.expert_mlp(p + "mlp.", x)
        else:
            mlp = self.sparse_mlp(index, x, sink)
        return bf16_round_f32(streams + mlp)

    def decode_step(self, token_id, position, states, caches, capture):
        if token_id < 0 or token_id >= self.vocab:
            raise ValueError(f"token {token_id} outside vocabulary {self.vocab}")
        if position == 0:
            self.caches = {}
            for i in range(self.layers):
                self.caches[i] = []
        streams = self.embed(token_id)
        for i in range(self.layers):
            sink = []
            streams = self.forward_layer(i, streams, position, sink)
            if sink:
                capture[(position, i)] = sink
            if not np.isfinite(streams).all():
                raise ValueError(f"nonfinite reference state at layer {i}")
        return streams

    def logits(self, streams, chunk=4096):
        norm = bf16_round_f32(rmsnorm(streams,
                                      self.tensor("model.norm.weight"), self.eps))
        lm = self.st.raw("lm_head.weight")
        if lm.dtype != np.uint16:
            raise ValueError("reference lm_head must be BF16")
        best = -np.inf
        best_token = -1
        for start in range(0, lm.shape[0], chunk):
            scores = bf16_to_f32(lm[start:start + chunk]) @ norm
            i = int(np.argmax(scores))
            if float(scores[i]) > best:
                best = float(scores[i])
                best_token = start + i
        return best_token, best


ENGINE_CLASS = LagunaEngine
