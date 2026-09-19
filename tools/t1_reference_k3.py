import os

import numpy as np

from t1_reference_common import (Safetensors, bf16_round_f32, bf16_to_f32,
                                 define_float, define_uint, f32_to_bf16_u16,
                                 rmsnorm, sigmoid)

PREFIX = "language_model.model.layers."
QK_L2_NORM_EPS = 1e-5
DELTA_RULE_L2_EPS = 1e-6
LORA_RMS_EPS = 1e-6
ATTNRES_BLOCK_LAYERS = 12
ATTNRES_MAX_SOURCES = 9
A_LOG_SOURCE_HEADS = 128
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0
MLA_QK_SCALE = 0.07216878365
WEIGHT_CACHE_BYTES = 16 << 20
DEFINES_VS_CONFIG = [
    ("HIDDEN_DIMENSION", "hidden_size", "uint"),
    ("LAYER_COUNT", "num_hidden_layers", "uint"),
    ("OUTPUT_VOCAB_COUNT", "vocab_size", "uint"),
    ("RMS_NORM_EPSILON", "rms_norm_eps", "float"),
    ("MOE_EXPERT_COUNT", "num_experts", "uint"),
    ("MOE_TOP_K", "num_experts_per_token", "uint"),
    ("MOE_INTERMEDIATE_DIMENSION", "moe_intermediate_size", "uint"),
    ("MOE_ROUTED_EXPERT_HIDDEN_DIMENSION", "routed_expert_hidden_size", "uint"),
    ("MOE_ROUTED_SCALING_FACTOR", "routed_scaling_factor", "float"),
    ("MOE_SHARED_EXPERT_COUNT", "num_shared_experts", "uint"),
    ("FIRST_ROUTED_LAYER", "first_k_dense_replace", "uint"),
    ("MLA_HEAD_COUNT", "num_attention_heads", "uint"),
    ("MLA_QUERY_A_DIMENSION", "q_lora_rank", "uint"),
    ("MLA_LATENT_DIMENSION", "kv_lora_rank", "uint"),
    ("MLA_QK_NOPE_HEAD_DIMENSION", "qk_nope_head_dim", "uint"),
    ("MLA_QK_ROPE_HEAD_DIMENSION", "qk_rope_head_dim", "uint"),
    ("MLA_V_HEAD_DIMENSION", "v_head_dim", "uint"),
]
DEFINES_RECORDED_ONLY = []

_E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                  -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                 dtype=np.float32)


class K3ConfigError(ValueError):
    pass


def define_of(kind, defines, dname):
    return define_uint(defines, dname) if kind == "uint" \
        else define_float(defines, dname)


def scalar_of(kind, value):
    got = value[0] if isinstance(value, list) else value
    if isinstance(got, bool):
        return int(got)
    return float(got)


def cross_check(defines, config):
    mismatches = []
    for dname, cname, kind in DEFINES_VS_CONFIG:
        if cname not in config:
            raise K3ConfigError(f"config key {cname} missing for SPARK_LLM_{dname}")
        want = define_of(kind, defines, dname)
        got = scalar_of(kind, config[cname])
        if abs(want - got) > 0:
            mismatches.append({"define": f"SPARK_LLM_{dname}", "value": want,
                               "config_key": cname, "config_value": got})
    for row in mismatches:
        raise K3ConfigError(
            f"{row['define']}={row['value']} disagrees with "
            f"{row['config_key']}={row['config_value']}")
    for dname, _, _ in DEFINES_RECORDED_ONLY:
        if f"SPARK_LLM_{dname}" not in defines:
            raise K3ConfigError(f"SPARK_LLM_{dname} missing from header")
    return mismatches


def require(condition, message):
    if not condition:
        raise K3ConfigError(message)


def situ(gate, up):
    gate = gate.astype(np.float32)
    up = up.astype(np.float32)
    activated = SITU_BETA * np.tanh(gate / SITU_BETA) * sigmoid(gate)
    linear = SITU_LINEAR_BETA * np.tanh(up / SITU_LINEAR_BETA)
    return bf16_round_f32(activated * linear)


def kda_delta_step(state, q, k, v, retention, beta):
    """One KDA delta-rule token step for one head, f32 throughout.

    The predict/write/read order is the LmDeltaRuleKernel contract; the
    caller normalises q and k exactly as the kernel does before calling.
    """
    predicted = (state * (k * retention)[:, None]).sum(axis=0)
    state[:] = retention[:, None] * state \
        + beta * (v - predicted)[None, :] * k[:, None]
    return (state * q[:, None]).sum(axis=0)


def l2_per_head(x, heads, dim, eps):
    m = x.reshape(heads, dim)
    return m / np.sqrt((m * m).sum(axis=1, keepdims=True) + np.float32(eps))


def attnres_mix(bank, partial, score_weight, sources, eps):
    """LmAttnResKernel: softmax over rmsnorm-scored sources, weighted sum.

    The bank contributes sources-1 slots and the partial row is the last
    source, exactly as the kernel lays them out.
    """
    values_list = list(bank[:sources - 1]) + [partial]
    require(len(values_list) == sources and sources <= ATTNRES_MAX_SOURCES,
            f"attnres source count {len(values_list)} disagrees with {sources}")
    scores = np.empty(sources, dtype=np.float32)
    for i, values in enumerate(values_list):
        inverse = 1.0 / np.sqrt(np.mean(values * values, axis=-1,
                                        keepdims=True) + eps)
        scores[i] = float((values * inverse * score_weight).sum())
    shifted = np.exp(scores - scores.max())
    weights = shifted / shifted.sum()
    total = np.zeros_like(partial)
    for i, values in enumerate(values_list):
        total += weights[i] * values
    return bf16_round_f32(total)


def dequant_mxfp4_f32(payload_u8, scale_u8, out_dim, in_dim):
    """MXFP4 weight dequantisation to f32.

    Bit-identical to mxfp4_to_bf16 then bf16_to_f32: E2M1 carries one
    mantissa bit and the E8M0 group scale is a power of two, so the product
    is exactly representable (E8M0 0xff is NaN and rejected; the packer's
    overflow advisory above code 253 applies to bf16 storage, not f32 math).
    """
    require(payload_u8.dtype == np.uint8 and scale_u8.dtype == np.uint8,
            "mxfp4 payload and scales must be U8")
    require(payload_u8.shape == (out_dim, in_dim // 2),
            f"mxfp4 payload shape {payload_u8.shape} disagrees with "
            f"({out_dim}, {in_dim // 2})")
    require(scale_u8.shape == (out_dim, in_dim // 32),
            f"mxfp4 scale shape {scale_u8.shape} disagrees with "
            f"({out_dim}, {in_dim // 32})")
    if scale_u8.size and int(scale_u8.max()) == 0xFF:
        raise ValueError("E8M0 code 0xff (NaN) in mxfp4 scale plane")
    groups = in_dim // 32
    nib = np.empty((out_dim, in_dim), dtype=np.int32)
    packed = payload_u8.astype(np.int32)
    nib[:, 0::2] = packed & 0xF
    nib[:, 1::2] = packed >> 4
    weights = _E2M1[nib].reshape(out_dim, groups, 32)
    scale = np.ldexp(np.float32(1),
                     scale_u8.astype(np.int32).reshape(out_dim, groups) - 127)
    return (weights * scale[:, :, None]).reshape(out_dim, in_dim)


class K3Engine:
    def __init__(self, checkpoint_dir, defines, config):
        self.mismatches = cross_check(defines, config)
        self.st = Safetensors(checkpoint_dir)
        self.hidden = int(config["hidden_size"])
        self.layers = int(config["num_hidden_layers"])
        self.vocab = int(config["vocab_size"])
        self.eps = float(config["rms_norm_eps"])
        require(self.eps == 1e-5,
                f"rms_norm_eps {self.eps} disagrees with the kernel contract "
                f"1e-5")
        self.heads = int(config["num_attention_heads"])
        self.q_lora = int(config["q_lora_rank"])
        self.kv_lora = int(config["kv_lora_rank"])
        self.nope = int(config["qk_nope_head_dim"])
        self.rope = int(config["qk_rope_head_dim"])
        self.v_head = int(config["v_head_dim"])
        self.block = int(config["attn_res_block_size"])
        require(self.block == ATTNRES_BLOCK_LAYERS,
                f"attn_res_block_size {self.block} disagrees with the pinned "
                f"{ATTNRES_BLOCK_LAYERS}")
        require(config["hidden_act"] == "situ",
                f"hidden_act {config['hidden_act']} is not situ")
        require(float(config["activation_situ_beta"]) == SITU_BETA
                and float(config["activation_situ_linear_beta"])
                == SITU_LINEAR_BETA,
                "situ betas disagree with the pinned 4.0/25.0")
        require(config.get("latent_moe_use_norm") is True,
                "latent_moe_use_norm must be true")
        require(config.get("moe_renormalize") is True,
                "moe_renormalize must be true")
        require(config["moe_router_activation_func"] == "sigmoid",
                "moe_router_activation_func must be sigmoid")
        require(int(config.get("num_expert_group", 1)) == 1,
                "num_expert_group must be 1; the production top-k is not "
                "grouped")
        self.qk_scale = np.float32((self.nope + self.rope) ** -0.5)
        if (self.nope, self.rope) == (128, 64):
            require(abs(MLA_QK_SCALE - float(self.qk_scale)) < 1e-9,
                    "the pinned MLA qk scale disagrees with "
                    "1/sqrt(nope+rope) at the production geometry")
        lac = config["linear_attn_config"]
        self.kda_heads = int(lac["num_heads"])
        self.kd = int(lac["head_dim"])
        self.conv = int(lac["short_conv_kernel_size"])
        self.gate_lb = float(lac["gate_lower_bound"])
        require(lac.get("use_full_rank_gate") is True,
                "linear_attn_config.use_full_rank_gate must be true")
        require(config.get("mla_use_nope") is not False
                and config.get("mla_use_output_gate") is not False,
                "mla_use_nope and mla_use_output_gate must be true")
        self.kda_dim = self.kda_heads * self.kd
        self.experts = int(config["num_experts"])
        self.topk = int(config["num_experts_per_token"])
        self.shared = int(config["num_shared_experts"])
        self.inter = int(config["moe_intermediate_size"])
        self.routed_hidden = int(config["routed_expert_hidden_size"])
        self.scaling = float(config["routed_scaling_factor"])
        self.first_routed = int(config["first_k_dense_replace"])
        self.eot = int(config["eos_token_id"][0] if isinstance(
            config["eos_token_id"], list) else config["eos_token_id"])
        full_from_config = {int(i) - 1 for i in lac["full_attn_layers"]}
        kda_from_config = {int(i) - 1 for i in lac["kda_layers"]}
        require(len(full_from_config) + len(kda_from_config) == self.layers
                and not (full_from_config & kda_from_config),
                "full_attn_layers and kda_layers must partition the layers")
        period = define_uint(defines, "ATTENTION_PERIOD")
        phase = define_uint(defines, "GLOBAL_ATTENTION_PHASE")
        formula = {i for i in range(self.layers)
                   if i % period == phase or i == self.layers - 1}
        require(formula == full_from_config,
                f"the header attention period formula {sorted(formula)} "
                f"disagrees with config full_attn_layers "
                f"{sorted(full_from_config)}")
        require({i for i in range(self.layers)} - formula == kda_from_config,
                "config kda_layers disagrees with the header attention period")
        self.is_kda = [i not in full_from_config for i in range(self.layers)]
        require(sum(self.is_kda) == define_uint(defines, "KDA_LAYER_COUNT"),
                "KDA layer count disagrees with SPARK_LLM_KDA_LAYER_COUNT")
        self._small = {}
        self._expert_cache = {}
        self._expert_bytes = 0
        self._expert_budget = int(os.environ.get(
            "T1_K3_EXPERT_CACHE_GB", "10")) << 30
        self._probe_shapes()

    def _shape_of(self, name):
        return tuple(self.st.entry(name)["shape"])

    def _dtype_of(self, name):
        return self.st.entry(name)["dtype"]

    def _expect(self, name, shape, dtype=None):
        got = self._shape_of(name)
        require(got == tuple(shape),
                f"{name} shape {got} disagrees with expected {tuple(shape)}")
        if dtype is not None:
            require(self._dtype_of(name) == dtype,
                    f"{name} dtype {self._dtype_of(name)} is not {dtype}")

    def _probe_shapes(self):
        k = PREFIX + "0."
        self._expect(k + "self_attn.q_proj.weight",
                     (self.kda_dim, self.hidden), "BF16")
        self._expect(k + "self_attn.f_a_proj.weight", (self.kd, self.hidden),
                     "BF16")
        self._expect(k + "self_attn.f_b_proj.weight",
                     (self.kda_dim, self.kd), "BF16")
        self._expect(k + "self_attn.A_log", (A_LOG_SOURCE_HEADS,), "F32")
        self._expect(k + "self_attn.dt_bias", (self.kda_dim,), "F32")
        self._expect(k + "self_attn.o_norm.weight", (self.kd,), "F32")
        self._expect(k + "self_attn.g_proj.weight",
                     (self.kda_dim, self.hidden), "BF16")
        self._expect(k + "self_attn.o_proj.weight",
                     (self.hidden, self.kda_dim), "BF16")
        conv = self._shape_of(k + "self_attn.q_conv1d.weight")
        require(conv in [(self.kda_dim, 1, self.conv),
                         (self.kda_dim, self.conv)],
                f"kda conv shape {conv} disagrees with the kernel geometry")
        if not self.is_kda[3]:
            m = PREFIX + "3."
            self._expect(m + "self_attn.q_a_proj.weight",
                         (self.q_lora, self.hidden), "BF16")
            self._expect(m + "self_attn.q_b_proj.weight",
                         (self.heads * (self.nope + self.rope), self.q_lora),
                         "BF16")
            self._expect(m + "self_attn.kv_a_proj_with_mqa.weight",
                         (self.kv_lora + self.rope, self.hidden), "BF16")
            self._expect(m + "self_attn.kv_b_proj.weight",
                         (self.heads * (self.nope + self.v_head),
                          self.kv_lora), "BF16")
            self._expect(m + "self_attn.g_proj.weight",
                         (self.heads * self.v_head, self.hidden), "BF16")
            self._expect(m + "self_attn.o_proj.weight",
                         (self.hidden, self.heads * self.v_head), "BF16")
        e = PREFIX + "1."
        self._expect(e + "block_sparse_moe.gate.weight",
                     (self.experts, self.hidden), "BF16")
        self._expect(e + "block_sparse_moe.routed_expert_down_proj.weight",
                     (self.routed_hidden, self.hidden), "BF16")
        self._expect(e + "block_sparse_moe.routed_expert_up_proj.weight",
                     (self.hidden, self.routed_hidden), "BF16")
        self._expect(e + "block_sparse_moe.routed_expert_norm.weight",
                     (self.routed_hidden,), "BF16")
        self._expect(e + "block_sparse_moe.shared_experts.gate_proj.weight",
                     (self.inter * self.shared, self.hidden), "BF16")
        self._expect(e + "block_sparse_moe.shared_experts.down_proj.weight",
                     (self.hidden, self.inter * self.shared), "BF16")
        self._expect(e + "block_sparse_moe.experts.0.w1.weight_packed",
                     (self.inter, self.routed_hidden // 2), "U8")
        self._expect(e + "block_sparse_moe.experts.0.w1.weight_scale",
                     (self.inter, self.routed_hidden // 32), "U8")
        self._expect(e + "block_sparse_moe.experts.0.w3.weight_packed",
                     (self.inter, self.routed_hidden // 2), "U8")
        self._expect(e + "block_sparse_moe.experts.0.w3.weight_scale",
                     (self.inter, self.routed_hidden // 32), "U8")
        self._expect(e + "block_sparse_moe.experts.0.w2.weight_packed",
                     (self.routed_hidden, self.inter // 2), "U8")
        self._expect(e + "block_sparse_moe.experts.0.w2.weight_scale",
                     (self.routed_hidden, self.inter // 32), "U8")
        self._expect("language_model.model.embed_tokens.weight",
                     (self.vocab, self.hidden), "BF16")
        self._expect("language_model.lm_head.weight",
                     (self.vocab, self.hidden), "BF16")

    def tensor(self, name):
        """f32 view of a checkpoint tensor. Small tensors ride the safetensors
        cache; anything at or above WEIGHT_CACHE_BYTES is read row-wise
        without caching, because the K3 hot set is far larger than host
        memory and an unbounded cache is an OOM."""
        key = ("t", name)
        if key in self._small:
            return self._small[key]
        entry = self.st.entry(name)
        extent = entry["data_offsets"][1] - entry["data_offsets"][0]
        if extent >= WEIGHT_CACHE_BYTES:
            raw = self.st.raw_rows(name, 0, entry["shape"][0])
        else:
            raw = self.st.raw(name)
        value = self._to_f32(name, raw)
        if extent < WEIGHT_CACHE_BYTES:
            self._small[key] = value
        return value

    @staticmethod
    def _to_f32(name, raw):
        if raw.dtype == np.uint16:
            return bf16_to_f32(raw)
        if raw.dtype == np.uint8:
            raise ValueError(f"{name}: raw U8 on a non-mxfp4 path")
        return raw.astype(np.float32)

    def linear(self, x, name):
        return bf16_round_f32(self.tensor(name + ".weight") @ x)

    def conv_weight(self, name):
        raw = self.st.raw(name)
        require(raw.dtype == np.float32, f"{name} must be F32")
        return raw.reshape(-1, self.conv)

    def embed(self, token_id):
        if token_id < 0 or token_id >= self.vocab:
            raise ValueError(f"token {token_id} outside vocabulary {self.vocab}")
        raw = self.st.raw_rows("language_model.model.embed_tokens.weight",
                               token_id, 1)
        if raw.dtype != np.uint16:
            raise ValueError("reference embedding must be BF16")
        return bf16_to_f32(raw[0])

    def short_conv(self, raw, weights, window):
        taps = np.concatenate([window[:, 1:], f32_to_bf16_u16(raw).reshape(-1, 1)],
                              axis=1)
        acc = (bf16_to_f32(taps) * weights).sum(axis=1)
        return bf16_round_f32(acc * sigmoid(acc)), taps

    def kda_attention(self, prefix, x, layer_state):
        heads, kd = self.kda_heads, self.kd
        q_raw = self.linear(x, prefix + "self_attn.q_proj")
        k_raw = self.linear(x, prefix + "self_attn.k_proj")
        v_raw = self.linear(x, prefix + "self_attn.v_proj")
        b_raw = self.linear(x, prefix + "self_attn.b_proj")
        latent = self.linear(x, prefix + "self_attn.f_a_proj")
        gate = self.linear(x, prefix + "self_attn.g_proj")
        q_conv, layer_state["wq"] = self.short_conv(
            q_raw, self.conv_weight(prefix + "self_attn.q_conv1d.weight"),
            layer_state["wq"])
        k_conv, layer_state["wk"] = self.short_conv(
            k_raw, self.conv_weight(prefix + "self_attn.k_conv1d.weight"),
            layer_state["wk"])
        v_conv, layer_state["wv"] = self.short_conv(
            v_raw, self.conv_weight(prefix + "self_attn.v_conv1d.weight"),
            layer_state["wv"])
        q_stage1 = bf16_round_f32(l2_per_head(q_conv, heads, kd,
                                              QK_L2_NORM_EPS).reshape(-1))
        k_stage1 = bf16_round_f32(l2_per_head(k_conv, heads, kd,
                                              QK_L2_NORM_EPS).reshape(-1))
        q2 = l2_per_head(q_stage1, heads, kd, DELTA_RULE_L2_EPS) \
            / np.sqrt(np.float32(kd))
        k2 = l2_per_head(k_stage1, heads, kd, DELTA_RULE_L2_EPS)
        v2 = v_conv.reshape(heads, kd)
        beta = sigmoid(b_raw).reshape(heads)
        decay_logit = bf16_round_f32(
            self.tensor(prefix + "self_attn.f_b_proj.weight") @ latent
        ).reshape(heads, kd)
        dt_bias = self.st.raw(prefix + "self_attn.dt_bias").astype(np.float32)
        a_log = self.st.raw(prefix + "self_attn.A_log").astype(np.float32)
        require(a_log.shape[0] == A_LOG_SOURCE_HEADS,
                f"A_log has {a_log.shape[0]} entries, the contract pins "
                f"{A_LOG_SOURCE_HEADS} narrowed to {heads}")
        head_scale = np.exp(a_log[:heads]).reshape(heads, 1)
        retention = np.exp(self.gate_lb * sigmoid(
            head_scale * (decay_logit + dt_bias.reshape(heads, kd))))
        state = layer_state["state"]
        out = np.empty((heads, kd), dtype=np.float32)
        for h in range(heads):
            out[h] = kda_delta_step(state[h], q2[h], k2[h], v2[h],
                                    retention[h], beta[h])
        o32 = bf16_round_f32(out.reshape(-1)).reshape(heads, kd)
        o_norm = self.st.raw(prefix + "self_attn.o_norm.weight") \
            .astype(np.float32)
        rms = np.sqrt((o32 * o32).sum(axis=1) / kd + self.eps)
        normed = bf16_round_f32(o32 / rms[:, None] * o_norm[None, :])
        gated = bf16_round_f32(normed * sigmoid(gate).reshape(heads, kd))
        return self.linear(gated.reshape(-1), prefix + "self_attn.o_proj")

    def mla_attention(self, prefix, x, cache):
        normed_q = bf16_round_f32(rmsnorm(
            self.linear(x, prefix + "self_attn.q_a_proj"),
            self.tensor(prefix + "self_attn.q_a_layernorm.weight"),
            LORA_RMS_EPS))
        q = bf16_round_f32(
            normed_q @ self.tensor(prefix + "self_attn.q_b_proj.weight").T)
        kv_a = self.linear(x, prefix + "self_attn.kv_a_proj_with_mqa")
        slot = np.concatenate([
            bf16_round_f32(rmsnorm(kv_a[:self.kv_lora],
                                   self.tensor(
                                       prefix +
                                       "self_attn.kv_a_layernorm.weight"),
                                   LORA_RMS_EPS)),
            bf16_round_f32(kv_a[self.kv_lora:])])
        cache.append(slot)
        slots = np.stack(cache)
        qh = q.reshape(self.heads, self.nope + self.rope)
        kvb = bf16_to_f32(
            self.st.raw_rows(prefix + "self_attn.kv_b_proj.weight", 0,
                             self.heads * (self.nope + self.v_head))
        ).reshape(self.heads, self.nope + self.v_head, self.kv_lora)
        normed = slots[:, :self.kv_lora]
        rope = slots[:, self.kv_lora:]
        values = np.empty((self.heads, self.v_head), dtype=np.float32)
        for h in range(self.heads):
            keys = normed @ kvb[h, :self.nope].T
            scores = (qh[h, :self.nope] @ keys.T
                      + rope @ qh[h, self.nope:]) * self.qk_scale
            weights = np.exp(scores - scores.max())
            weights = weights / weights.sum()
            attended = bf16_round_f32(weights @ normed)
            values[h] = bf16_round_f32(kvb[h, self.nope:] @ attended)
        gate = self.linear(x, prefix + "self_attn.g_proj")
        gated = bf16_round_f32(values.reshape(-1) * sigmoid(gate))
        return self.linear(gated, prefix + "self_attn.o_proj")

    def expert_mlp(self, w_up, w_down, x):
        gate_up = bf16_round_f32(w_up @ x)
        intermediate = situ(gate_up[:gate_up.shape[0] // 2],
                            gate_up[gate_up.shape[0] // 2:])
        return bf16_round_f32(w_down @ intermediate)

    def _expert_pair(self, prefix, expert):
        key = (prefix, expert)
        hit = self._expert_cache.get(key)
        if hit is not None:
            return hit
        base = prefix + f"block_sparse_moe.experts.{expert}."
        w_gate = dequant_mxfp4_f32(
            self.st.raw_rows(base + "w1.weight_packed", 0, self.inter),
            self.st.raw_rows(base + "w1.weight_scale", 0, self.inter),
            self.inter, self.routed_hidden)
        w_up = dequant_mxfp4_f32(
            self.st.raw_rows(base + "w3.weight_packed", 0, self.inter),
            self.st.raw_rows(base + "w3.weight_scale", 0, self.inter),
            self.inter, self.routed_hidden)
        w1 = np.concatenate([w_gate, w_up])
        del w_gate, w_up
        w2 = dequant_mxfp4_f32(
            self.st.raw_rows(base + "w2.weight_packed", 0, self.routed_hidden),
            self.st.raw_rows(base + "w2.weight_scale", 0, self.routed_hidden),
            self.routed_hidden, self.inter)
        while self._expert_bytes + w1.nbytes + w2.nbytes > self._expert_budget \
                and self._expert_cache:
            victim = next(iter(self._expert_cache))
            self._expert_bytes -= self._expert_cache[victim][0].nbytes \
                + self._expert_cache[victim][1].nbytes
            del self._expert_cache[victim]
        self._expert_cache[key] = (w1, w2)
        self._expert_bytes += w1.nbytes + w2.nbytes
        return w1, w2

    def routed_expert(self, prefix, expert, x):
        w1, w2 = self._expert_pair(prefix, expert)
        gate_up = bf16_round_f32(w1 @ x)
        intermediate = situ(gate_up[:self.inter], gate_up[self.inter:])
        return bf16_round_f32(w2 @ intermediate)

    def dense_mlp(self, prefix, x):
        gate = self.linear(x, prefix + "mlp.gate_proj")
        up = self.linear(x, prefix + "mlp.up_proj")
        intermediate = situ(gate, up)
        return self.linear(intermediate, prefix + "mlp.down_proj")

    def sparse_mlp(self, prefix, x, sink):
        logits = bf16_to_f32(
            self.st.raw(prefix + "block_sparse_moe.gate.weight")) @ x
        scores = sigmoid(logits)
        choice = scores + self.tensor(
            prefix + "block_sparse_moe.gate.e_score_correction_bias")
        order = np.argsort(-choice, kind="stable")
        selected = np.sort(order[:self.topk])
        picked = scores[selected]
        weights = picked / (picked.sum() + 1e-20) * np.float32(self.scaling)
        lat_in = bf16_round_f32(
            self.tensor(prefix
                        + "block_sparse_moe.routed_expert_down_proj.weight")
            @ x)
        routed = np.zeros(self.routed_hidden, dtype=np.float32)
        for i in range(self.topk):
            output = self.routed_expert(prefix, int(selected[i]), lat_in)
            routed += output * weights[i]
        latent = bf16_round_f32(rmsnorm(
            bf16_round_f32(routed),
            self.tensor(prefix + "block_sparse_moe.routed_expert_norm.weight"),
            self.eps))
        routed_out = bf16_round_f32(
            self.tensor(prefix + "block_sparse_moe.routed_expert_up_proj.weight")
            @ latent)
        shared_up = np.concatenate([
            self.tensor(prefix +
                        "block_sparse_moe.shared_experts.gate_proj.weight"),
            self.tensor(prefix +
                        "block_sparse_moe.shared_experts.up_proj.weight")])
        shared = self.expert_mlp(
            shared_up,
            self.tensor(prefix +
                        "block_sparse_moe.shared_experts.down_proj.weight"),
            x)
        sink.append(selected.astype(np.int32))
        sink.append(weights.astype(np.float32))
        return routed_out, shared

    def res_weight(self, prefix, kind):
        gamma = self.tensor(prefix + kind + "res_norm.weight")
        proj = self.tensor(prefix + kind + "res_proj.weight").reshape(-1)
        return bf16_round_f32(gamma * proj)

    def forward_layer(self, index, hidden, partial, bank, states, caches, sink,
                      emit):
        prefix = PREFIX + str(index) + "."
        boundary = index % self.block == 0
        if index > 0:
            hidden = attnres_mix(bank, partial,
                                 self.res_weight(prefix, "self_attention_"),
                                 (index - 1) // self.block + 2, self.eps)
        if boundary:
            if index == 0:
                partial = bf16_round_f32(hidden)
            bank[index // self.block] = bf16_round_f32(partial)
        normed = bf16_round_f32(rmsnorm(
            hidden, self.tensor(prefix + "input_layernorm.weight"), self.eps))
        if self.is_kda[index]:
            attention = self.kda_attention(prefix, normed, states[index])
        else:
            attention = self.mla_attention(prefix, normed, caches[index])
        if boundary:
            partial = bf16_round_f32(attention)
        else:
            partial = bf16_round_f32(partial + attention)
        hidden = attnres_mix(bank, partial, self.res_weight(prefix, "mlp_"),
                             index // self.block + 2, self.eps)
        normed = bf16_round_f32(rmsnorm(
            hidden, self.tensor(prefix + "post_attention_layernorm.weight"),
            self.eps))
        if index < self.first_routed:
            mlp = self.dense_mlp(prefix, normed)
            partial = bf16_round_f32(partial + mlp)
        else:
            routed, shared = self.sparse_mlp(prefix, normed, sink)
            partial = bf16_round_f32(partial + routed)
            partial = bf16_round_f32(partial + shared)
        emit(partial)
        return hidden, partial

    def decode_step(self, token_id, position, states, caches, capture,
                    capture_streams=None):
        if token_id < 0 or token_id >= self.vocab:
            raise ValueError(f"token {token_id} outside vocabulary {self.vocab}")
        embedding = self.embed(token_id)
        if position == 0:
            for i in range(self.layers):
                if self.is_kda[i]:
                    states[i] = {
                        "state": np.zeros((self.kda_heads, self.kd, self.kd),
                                          dtype=np.float32),
                        "wq": np.zeros((self.kda_dim, self.conv),
                                       dtype=np.uint16),
                        "wk": np.zeros((self.kda_dim, self.conv),
                                       dtype=np.uint16),
                        "wv": np.zeros((self.kda_dim, self.conv),
                                       dtype=np.uint16),
                    }
                else:
                    caches[i] = []

        def emit(streams):
            if capture_streams is not None:
                capture_streams(i, streams)

        hidden = embedding
        partial = embedding
        bank = [None] * (self.layers // self.block + 1)
        for i in range(self.layers):
            sink = []
            hidden, partial = self.forward_layer(
                i, hidden, partial, bank, states, caches, sink, emit)
            if sink:
                capture[(position, i)] = sink
            if not np.isfinite(partial).all() or not np.isfinite(hidden).all():
                raise ValueError(f"nonfinite reference state at layer {i}")
        head_prefix = "language_model.model."
        hidden = attnres_mix(
            bank, partial,
            bf16_round_f32(
                self.tensor(head_prefix + "output_attn_res_norm.weight")
                * self.tensor(head_prefix + "output_attn_res_proj.weight")
                .reshape(-1)),
            (self.layers - 1) // self.block + 2, self.eps)
        if not np.isfinite(hidden).all():
            raise ValueError("nonfinite reference output")
        return hidden

    def logits(self, streams, chunk=4096):
        norm = bf16_round_f32(rmsnorm(
            streams, self.tensor("language_model.model.norm.weight"),
            self.eps))
        lm = self.st.raw("language_model.lm_head.weight")
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


ENGINE_CLASS = K3Engine
