import numpy as np

from t1_reference_common import (Safetensors, bf16_round_f32, bf16_to_f32,
                                 define_float, define_uint, f32_to_bf16_u16,
                                 fp8_block_to_bf16, nvfp4_to_f32, rmsnorm,
                                 sigmoid)

PREFIX = "model.layers."
L2_EPS = 1e-6


class Qwen38MaxConfigError(ValueError):
    pass


def softplus(x):
    return np.logaddexp(np.float32(0), x)


def swish(x):
    return x * sigmoid(x)


def cross_check(defines, config):
    checks = [
        ("HIDDEN_DIMENSION", "hidden_size", "uint"),
        ("LAYER_COUNT", "num_hidden_layers", "uint"),
        ("OUTPUT_VOCAB_COUNT", "vocab_size", "uint"),
        ("RMS_NORM_EPSILON", "rms_norm_eps", "float"),
        ("MOE_EXPERT_COUNT", "num_experts", "uint"),
        ("MOE_TOP_K", "num_experts_per_tok", "uint"),
        ("END_OF_TEXT_TOKEN_ID", "eos_token_id", "uint"),
        ("ATTENTION_PERIOD", "full_attention_interval", "uint"),
        ("KDA_HEAD_KEY_DIMENSION", "linear_key_head_dim", "uint"),
        ("KDA_HEAD_VALUE_DIMENSION", "linear_value_head_dim", "uint"),
    ]
    for dname, cname, kind in checks:
        if cname not in config:
            raise Qwen38MaxConfigError(f"config key {cname} missing for SPARK_LLM_{dname}")
        want = define_uint(defines, dname) if kind == "uint" \
            else define_float(defines, dname)
        got = config[cname]
        if isinstance(got, list):
            got = got[0]
        if abs(float(want) - float(got)) > 0:
            raise Qwen38MaxConfigError(
                f"SPARK_LLM_{dname}={want} disagrees with config {cname}={got}")
    return []


class Qwen38MaxEngine:
    def __init__(self, checkpoint_dir, defines, config):
        self.mismatches = cross_check(defines, config)
        self.st = Safetensors(checkpoint_dir)
        self.hidden = int(config["hidden_size"])
        self.layers = int(config["num_hidden_layers"])
        self.vocab = int(config["vocab_size"])
        self.eps = float(config["rms_norm_eps"])
        self.heads = int(config["num_attention_heads"])
        self.kv_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config["head_dim"])
        self.rope_dim = int(self.head_dim * float(config.get(
            "partial_rotary_factor", 0.25)))
        self.rope_theta = float(config["rope_parameters"]["rope_theta"])
        self.k_heads = int(config["linear_num_key_heads"])
        self.v_heads = int(config["linear_num_value_heads"])
        self.kd = int(config["linear_key_head_dim"])
        self.conv = int(config["linear_conv_kernel_dim"])
        self.limit = 10.0
        self.experts = int(config["num_experts"])
        self.topk = int(config["num_experts_per_tok"])
        self.layer_types = list(config["layer_types"])
        self.eot = int(config["eos_token_id"] if not isinstance(
            config["eos_token_id"], list) else config["eos_token_id"][0])
        if len(self.layer_types) != self.layers:
            raise Qwen38MaxConfigError("layer_types disagree with layer count")
        self.gva = self.v_heads // self.k_heads
        self.attn_group = self.heads // self.kv_heads

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
        if token_id < 0 or token_id >= self.vocab:
            raise ValueError(f"token {token_id} outside vocabulary {self.vocab}")
        raw = self.st.raw_rows("model.embed_tokens.weight", token_id, 1)
        if raw.dtype != np.uint16:
            raise ValueError("reference embedding must be BF16")
        return bf16_to_f32(raw[0])

    def partial_rope(self, rows, position):
        half = self.rope_dim // 2
        pairs = np.arange(half, dtype=np.float32)
        freq = np.exp2(-(2.0 * pairs / self.rope_dim)
                       * np.log2(np.float32(self.rope_theta)))
        angle = np.float32(position) * freq
        cos = np.cos(angle)
        sin = np.sin(angle)
        real = rows[:, 0:half].copy()
        imag = rows[:, half:self.rope_dim].copy()
        rows[:, 0:half] = real * cos - imag * sin
        rows[:, half:self.rope_dim] = imag * cos + real * sin
        return rows

    def gdn_attention(self, prefix, x, layer_state):
        qkv = bf16_round_f32(self.tensor(prefix + "linear_attn.in_proj_qkv.weight") @ x)
        channels = qkv.shape[0]
        if layer_state is None:
            layer_state = {"window": np.zeros((channels, self.conv),
                                              dtype=np.uint16),
                           "state": np.zeros((self.v_heads, self.kd, self.kd),
                                             dtype=np.float32)}
        window = layer_state["window"]
        taps = np.concatenate([window[:, 1:], f32_to_bf16_u16(qkv).reshape(-1, 1)],
                              axis=1)
        weights = self.tensor(prefix + "linear_attn.conv1d.weight").reshape(channels,
                                                                self.conv)
        acc = (bf16_to_f32(taps) * weights).sum(axis=1)
        conv_out = bf16_round_f32(swish(acc))
        layer_state["window"] = taps
        qk = self.k_heads * self.kd
        a_pre = bf16_round_f32(self.tensor(prefix + "linear_attn.in_proj_a.weight") @ x)
        b_pre = bf16_round_f32(self.tensor(prefix + "linear_attn.in_proj_b.weight") @ x)
        a_log = bf16_to_f32(self.st.raw(prefix + "linear_attn.A_log").reshape(-1))
        dt_bias = bf16_to_f32(self.st.raw(prefix + "linear_attn.dt_bias").reshape(-1))
        log_decay = -np.exp(a_log) * softplus(a_pre + dt_bias)
        beta = sigmoid(b_pre)
        q = conv_out[0:qk].reshape(self.k_heads, self.kd)
        k = conv_out[qk:2 * qk].reshape(self.k_heads, self.kd)
        v = conv_out[2 * qk:].reshape(self.v_heads, self.kd)
        q = q / np.sqrt((q * q).sum(axis=1, keepdims=True) + L2_EPS) \
            / np.sqrt(np.float32(self.kd))
        k = k / np.sqrt((k * k).sum(axis=1, keepdims=True) + L2_EPS)
        state = layer_state["state"]
        out = np.empty((self.v_heads, self.kd), dtype=np.float32)
        for h in range(self.v_heads):
            key = h // self.gva
            decay = np.exp(log_decay[h])
            decayed = state[h] * decay
            kv_memory = (decayed * k[key][:, None]).sum(axis=0)
            delta = (v[h] - kv_memory) * beta[h]
            state[h] = decayed + k[key][:, None] * delta[None, :]
            out[h] = (state[h] * q[key][:, None]).sum(axis=0)
        core = bf16_round_f32(out.reshape(-1))
        z = bf16_round_f32(self.tensor(prefix + "linear_attn.in_proj_z.weight") @ x)
        norm_w = bf16_to_f32(self.st.raw(prefix + "linear_attn.norm.weight").reshape(-1))
        zc = core.reshape(self.v_heads, self.kd)
        variance = (zc * zc).sum(axis=1) / self.kd
        normed = zc / np.sqrt(variance + self.eps)[:, None] * norm_w[None, :]
        gated = bf16_round_f32(normed.reshape(-1) * swish(z))
        return self.linear(gated, prefix + "linear_attn.out_proj"), layer_state

    def full_attention(self, prefix, x, cache, position):
        qf = bf16_round_f32(self.tensor(prefix + "self_attn.q_proj.weight") @ x)
        qf = qf.reshape(self.heads, 2 * self.head_dim)
        value = qf[:, 0:self.head_dim].copy()
        gate = sigmoid(qf[:, self.head_dim:])
        qn = bf16_to_f32(self.st.raw(prefix + "self_attn.q_norm.weight").reshape(-1))
        kn = bf16_to_f32(self.st.raw(prefix + "self_attn.k_norm.weight").reshape(-1))
        value = value / np.sqrt((value * value).sum(axis=1, keepdims=True)
                                / self.head_dim + self.eps) * qn[None, :]
        value = self.partial_rope(value, position)
        k_raw = bf16_round_f32(self.tensor(prefix + "self_attn.k_proj.weight") @ x)
        v_raw = bf16_round_f32(self.tensor(prefix + "self_attn.v_proj.weight") @ x)
        k = k_raw.reshape(self.kv_heads, self.head_dim)
        k = k / np.sqrt((k * k).sum(axis=1, keepdims=True) / self.head_dim
                        + self.eps) * kn[None, :]
        k = self.partial_rope(k, position)
        cache.append((bf16_round_f32(k.reshape(-1)),
                      bf16_round_f32(v_raw.reshape(-1))))
        keys = np.stack([row[0] for row in cache])
        values = np.stack([row[1] for row in cache])
        keys = keys.reshape(len(cache), self.kv_heads, self.head_dim)
        values = values.reshape(len(cache), self.kv_heads, self.head_dim)
        out = np.empty((self.heads, self.head_dim), dtype=np.float32)
        scale = np.float32(1.0 / np.sqrt(self.head_dim))
        for h in range(self.heads):
            kvh = h // self.attn_group
            scores = (keys[:, kvh, :] @ value[h]) * scale
            weights = np.exp(scores - scores.max())
            weights = weights / weights.sum()
            out[h] = weights @ values[:, kvh, :]
        gated = bf16_round_f32(out.reshape(-1) * gate.reshape(-1))
        return self.linear(gated, prefix + "self_attn.o_proj")

    def dequant_expert(self, prefix, name):
        payload = self.st.raw(prefix + name + ".weight")
        scale_e4m3 = self.st.raw(prefix + name + ".weight_scale")
        scale_2 = self.st.raw(prefix + name + ".weight_scale_2")
        rows = payload.shape[0]
        cols = payload.shape[1] * 2
        scalar = np.float32(scale_2.reshape(-1)[0])
        return nvfp4_to_f32(payload, scale_e4m3.reshape(rows, -1), rows,
                            cols) * scalar

    def routed_expert(self, prefix, expert, x):
        gate_w = self.dequant_expert(prefix + f"experts.{expert}.", "gate_proj")
        up_w = self.dequant_expert(prefix + f"experts.{expert}.", "up_proj")
        down_w = self.dequant_expert(prefix + f"experts.{expert}.", "down_proj")
        gate = np.minimum(bf16_round_f32(gate_w @ x), self.limit)
        up = bf16_round_f32(up_w @ x)
        activated = bf16_round_f32(swish(gate) * up)
        return bf16_round_f32(down_w @ activated)

    def shared_expert(self, prefix, x):
        gate = self.linear(x, prefix + "mlp.shared_expert.gate_proj")
        up = self.linear(x, prefix + "mlp.shared_expert.up_proj")
        activated = bf16_round_f32(swish(gate) * up)
        return self.linear(activated, prefix + "mlp.shared_expert.down_proj")

    def moe(self, prefix, x, sink):
        scores = self.tensor(prefix + "mlp.gate.weight") @ x
        order = np.argsort(-scores, kind="stable")
        selected = np.sort(order[:self.topk])
        picked = scores[selected]
        shifted = picked - picked.max()
        weights = np.exp(shifted) / np.exp(shifted).sum()
        routed = np.zeros_like(x)
        for i in range(self.topk):
            output = self.routed_expert(prefix + "mlp.", int(selected[i]), x)
            routed = bf16_round_f32(routed + bf16_round_f32(output * weights[i]))
        shared = self.shared_expert(prefix, x)
        gate_logit = bf16_round_f32(
            self.tensor(prefix + "mlp.shared_expert_gate.weight") @ x)
        coeff = sigmoid(gate_logit[0])
        sink.append(selected.astype(np.int32))
        sink.append(weights.astype(np.float32))
        return bf16_round_f32(routed + bf16_round_f32(coeff * shared))

    def forward_layer(self, index, streams, states, caches, sink):
        prefix = PREFIX + str(index) + "."
        x = bf16_round_f32(rmsnorm(
            streams, self.tensor(prefix + "input_layernorm.weight"), self.eps))
        if self.layer_types[index] == "linear_attention":
            attention, states[index] = self.gdn_attention(prefix, x,
                                                          states.get(index))
        elif self.layer_types[index] == "full_attention":
            attention = self.full_attention(prefix, x, caches[index], index)
        else:
            raise ValueError(f"unsupported layer type {self.layer_types[index]}")
        streams = bf16_round_f32(streams + attention)
        x = bf16_round_f32(rmsnorm(
            streams, self.tensor(prefix + "post_attention_layernorm.weight"),
            self.eps))
        return bf16_round_f32(streams + self.moe(prefix, x, sink))

    def decode_step(self, token_id, position, states, caches, capture):
        if token_id < 0 or token_id >= self.vocab:
            raise ValueError(f"token {token_id} outside vocabulary {self.vocab}")
        raw = self.st.raw_rows("model.embed_tokens.weight", token_id, 1)
        if raw.dtype != np.uint16:
            raise ValueError("reference embedding must be BF16")
        if position == 0:
            for i in range(self.layers):
                if self.layer_types[i] == "full_attention":
                    caches[i] = []
        streams = bf16_to_f32(raw[0])
        for i in range(self.layers):
            sink = []
            streams = self.forward_layer(i, streams, states, caches, sink)
            if sink:
                capture[(position, i)] = sink
            if not np.isfinite(streams).all():
                raise ValueError(f"nonfinite reference state at layer {i}")
        return streams

    def logits(self, streams, chunk=4096):
        norm = bf16_round_f32(rmsnorm(
            streams, self.tensor("model.norm.weight"), self.eps))
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


ENGINE_CLASS = Qwen38MaxEngine
