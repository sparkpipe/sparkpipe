import os

import numpy as np

from t1_reference_common import (Safetensors, bf16_round_f32, bf16_to_f32,
                                 define_float, define_uint, f32_to_bf16_u16,
                                 fp8_block_to_bf16, nvfp4_to_f32, rmsnorm,
                                 sigmoid)

PREFIX = "model.language_model.layers."
L2_EPS = 1e-6


class Qwen38_27bConfigError(ValueError):
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
        ("END_OF_TEXT_TOKEN_ID", "eos_token_id", "uint"),
        ("ATTENTION_PERIOD", "full_attention_interval", "uint"),
        ("QUERY_HEAD_COUNT", "num_attention_heads", "uint"),
        ("KV_HEAD_COUNT", "num_key_value_heads", "uint"),
        ("HEAD_DIMENSION", "head_dim", "uint"),
        ("GDN_KEY_HEAD_COUNT", "linear_num_key_heads", "uint"),
        ("GDN_VALUE_HEAD_COUNT", "linear_num_value_heads", "uint"),
        ("GDN_HEAD_KEY_DIMENSION", "linear_key_head_dim", "uint"),
        ("GDN_HEAD_VALUE_DIMENSION", "linear_value_head_dim", "uint"),
        ("GDN_CONV_KERNEL", "linear_conv_kernel_dim", "uint"),
        ("DENSE_INTERMEDIATE_DIMENSION", "intermediate_size", "uint"),
        ("MTP_LAYER_COUNT", "mtp_num_hidden_layers", "uint"),
    ]
    mismatches = []
    for dname, cname, kind in checks:
        if cname not in config:
            raise Qwen38_27bConfigError(
                f"config key {cname} missing for SPARK_LLM_{dname}")
        want = define_uint(defines, dname) if kind == "uint" \
            else define_float(defines, dname)
        got = config[cname]
        if isinstance(got, list):
            got = got[0]
        if abs(float(want) - float(got)) > 0:
            mismatches.append({"define": f"SPARK_LLM_{dname}", "value": want,
                               "config_key": cname, "config_value": got})
    rope_dim = int(define_uint(defines, "HEAD_DIMENSION")
                   * float(config.get("partial_rotary_factor", 0.25)))
    if rope_dim != define_uint(defines, "ROPE_DIMENSION"):
        mismatches.append({"define": "SPARK_LLM_ROPE_DIMENSION",
                           "value": define_uint(defines, "ROPE_DIMENSION"),
                           "config_key": "head_dim*partial_rotary_factor",
                           "config_value": rope_dim})
    if "rope_parameters" not in config:
        raise Qwen38_27bConfigError("config key rope_parameters missing")
    theta = float(config["rope_parameters"]["rope_theta"])
    if abs(float(define_float(defines, "ROPE_THETA")) - theta) > 0:
        mismatches.append({"define": "SPARK_LLM_ROPE_THETA",
                           "value": define_float(defines, "ROPE_THETA"),
                           "config_key": "rope_theta", "config_value": theta})
    layer_types = config.get("layer_types")
    if not isinstance(layer_types, list):
        raise Qwen38_27bConfigError("config key layer_types missing")
    layers = int(config["num_hidden_layers"])
    if len(layer_types) != layers:
        raise Qwen38_27bConfigError("layer_types disagree with layer count")
    period = define_uint(defines, "ATTENTION_PERIOD")
    phase = define_uint(defines, "FULL_ATTENTION_PHASE")
    if phase >= period:
        mismatches.append({"define": "SPARK_LLM_FULL_ATTENTION_PHASE",
                           "value": phase, "config_key": "attention_period",
                           "config_value": period})
    else:
        for index, kind in enumerate(layer_types):
            want = "full_attention" if index % period == phase \
                else "linear_attention"
            if kind != want:
                mismatches.append(
                    {"define": "SPARK_LLM_FULL_ATTENTION_PHASE", "value": phase,
                     "config_key": f"layer_types[{index}]", "config_value": kind})
    if config.get("hidden_act") != "silu":
        mismatches.append({"define": "SPARK_LLM_HIDDEN_ACT", "value": "silu",
                           "config_key": "hidden_act",
                           "config_value": config.get("hidden_act")})
    if mismatches:
        row = mismatches[0]
        raise Qwen38_27bConfigError(
            f"{row['define']}={row['value']} disagrees with "
            f"{row['config_key']}={row['config_value']}")
    return mismatches


class Qwen38_27bEngine:
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
        self.intermediate = int(config["intermediate_size"])
        self.layer_types = list(config["layer_types"])
        self.eot = int(config["eos_token_id"] if not isinstance(
            config["eos_token_id"], list) else config["eos_token_id"][0])
        if len(self.layer_types) != self.layers:
            raise Qwen38_27bConfigError("layer_types disagree with layer count")
        self.gva = self.v_heads // self.k_heads
        self.attn_group = self.heads // self.kv_heads
        if self.v_heads % self.k_heads != 0:
            raise Qwen38_27bConfigError(
                f"GDN value heads {self.v_heads} not divisible by key heads "
                f"{self.k_heads}")
        self._weights = {}
        self._memo_bytes = 0
        self._memo_limit = int(os.environ.get("T1_REF_WEIGHT_CACHE_BYTES", "0"))

    def has(self, name):
        try:
            self.st.entry(name)
        except (KeyError, FileNotFoundError):
            return False
        return True

    def weight_u16(self, name):
        if name in self._weights:
            return self._weights[name]
        fresh = False
        if self.has(name + "_packed"):
            payload = self.st.pread(name + "_packed")
            if payload.dtype != np.uint8:
                raise ValueError(f"nvfp4 payload {name}_packed must be U8")
            scale = self.st.pread(name + "_scale")
            if scale.dtype != np.uint8:
                raise ValueError(
                    f"nvfp4 group scales for {name} must be F8_E4M3")
            rows, packed = payload.shape
            cols = packed * 2
            if cols % 16 != 0:
                raise ValueError(f"nvfp4 input dim {cols} not a multiple of 16")
            if scale.size != rows * (cols // 16):
                raise ValueError(f"nvfp4 scale extent disagrees with {name}")
            scalar = np.float32(self.st.pread(name + "_global_scale")
                                .reshape(-1)[0])
            stored = np.empty((rows, cols), dtype=np.uint16)
            for r0 in range(0, rows, 512):
                r1 = min(r0 + 512, rows)
                w = nvfp4_to_f32(payload[r0:r1],
                                 scale.reshape(rows, -1)[r0:r1], r1 - r0,
                                 cols)
                stored[r0:r1] = f32_to_bf16_u16(w * (np.float32(0.5) / scalar))
            fresh = True
        else:
            raw = self.st.pread(name)
            if raw.dtype == np.uint16:
                stored = raw
            elif raw.dtype == np.uint8:
                scale = self.st.pread(name + "_scale_inv")
                if scale.dtype == np.uint16:
                    scale = bf16_to_f32(scale)
                elif scale.dtype != np.float32:
                    raise ValueError(f"unsupported scale_inv dtype for "
                                     f"{name}: {scale.dtype}")
                rows, cols = raw.shape
                stored = fp8_block_to_bf16(raw, scale, rows, cols)
                fresh = True
            else:
                raise ValueError(f"unsupported weight dtype for {name}: "
                                 f"{raw.dtype}")
        if fresh and self._memo_bytes + stored.nbytes <= self._memo_limit:
            self._weights[name] = stored
            self._memo_bytes += stored.nbytes
        return stored

    def tensor(self, name):
        return bf16_to_f32(self.weight_u16(name))

    def linear(self, x, name):
        w16 = self.weight_u16(name + ".weight")
        out = np.empty(w16.shape[0], dtype=np.float32)
        for r0 in range(0, w16.shape[0], 512):
            r1 = min(r0 + 512, w16.shape[0])
            out[r0:r1] = bf16_to_f32(w16[r0:r1]) @ x
        return bf16_round_f32(out)

    def embed(self, token_id):
        if token_id < 0 or token_id >= self.vocab:
            raise ValueError(f"token {token_id} outside vocabulary {self.vocab}")
        raw = self.st.raw_rows("model.language_model.embed_tokens.weight",
                               token_id, 1)
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
        a_log = bf16_to_f32(self.st.pread(prefix + "linear_attn.A_log").reshape(-1))
        dt_bias = bf16_to_f32(self.st.pread(prefix + "linear_attn.dt_bias").reshape(-1))
        if a_log.shape[0] != self.v_heads or dt_bias.shape[0] != self.v_heads:
            raise Qwen38_27bConfigError(
                "A_log/dt_bias disagree with GDN value head count")
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
        norm_w = bf16_to_f32(self.st.pread(prefix + "linear_attn.norm.weight").reshape(-1))
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
        qn = bf16_to_f32(self.st.pread(prefix + "self_attn.q_norm.weight").reshape(-1))
        kn = bf16_to_f32(self.st.pread(prefix + "self_attn.k_norm.weight").reshape(-1))
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

    def mlp(self, prefix, x):
        gate = self.linear(x, prefix + "mlp.gate_proj")
        up = self.linear(x, prefix + "mlp.up_proj")
        activated = bf16_round_f32(swish(gate) * up)
        return self.linear(activated, prefix + "mlp.down_proj")

    def forward_layer(self, index, streams, states, caches):
        prefix = PREFIX + str(index) + "."
        x = bf16_round_f32(rmsnorm(
            streams, self.tensor(prefix + "input_layernorm.weight"), self.eps))
        if self.layer_types[index] == "linear_attention":
            attention, states[index] = self.gdn_attention(prefix, x,
                                                          states.get(index))
        elif self.layer_types[index] == "full_attention":
            attention = self.full_attention(prefix, x, caches[index],
                                             len(caches[index]))
        else:
            raise ValueError(f"unsupported layer type {self.layer_types[index]}")
        streams = bf16_round_f32(streams + attention)
        x = bf16_round_f32(rmsnorm(
            streams, self.tensor(prefix + "post_attention_layernorm.weight"),
            self.eps))
        return bf16_round_f32(streams + self.mlp(prefix, x))

    def decode_step(self, token_id, position, states, caches, capture):
        del capture
        if token_id < 0 or token_id >= self.vocab:
            raise ValueError(f"token {token_id} outside vocabulary {self.vocab}")
        raw = self.st.raw_rows("model.language_model.embed_tokens.weight",
                               token_id, 1)
        if raw.dtype != np.uint16:
            raise ValueError("reference embedding must be BF16")
        if position == 0:
            for i in range(self.layers):
                if self.layer_types[i] == "full_attention":
                    caches[i] = []
        streams = bf16_to_f32(raw[0])
        for i in range(self.layers):
            streams = self.forward_layer(i, streams, states, caches)
            if not np.isfinite(streams).all():
                raise ValueError(f"nonfinite reference state at layer {i}")
        return streams

    def logits(self, streams, chunk=4096):
        norm = bf16_round_f32(rmsnorm(
            streams, self.tensor("model.language_model.norm.weight"), self.eps))
        entry = self.st.entry("lm_head.weight")
        if entry["dtype"] != "BF16":
            raise ValueError("reference lm_head must be BF16")
        rows_total = entry["shape"][0]
        best = -np.inf
        best_token = -1
        for start in range(0, rows_total, chunk):
            count = min(chunk, rows_total - start)
            lm = self.st.raw_rows("lm_head.weight", start, count)
            scores = bf16_to_f32(lm) @ norm
            i = int(np.argmax(scores))
            if float(scores[i]) > best:
                best = float(scores[i])
                best_token = start + i
        return best_token, best


ENGINE_CLASS = Qwen38_27bEngine
