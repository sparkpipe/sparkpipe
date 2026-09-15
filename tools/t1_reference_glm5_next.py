import numpy as np

from t1_reference_common import (Safetensors, bf16_round_f32, bf16_to_f32,
                                 define_float, define_uint, f32_to_bf16_u16,
                                 fp8_block_to_bf16, rmsnorm, sigmoid)

PREFIX = "model.language_model.layers."
QK_L2_EPS = 1e-6
MAX_INDEXER_CONTEXT = 2048
DEFINES_VS_CONFIG = [
    ("HIDDEN_DIMENSION", "hidden_size", "uint"),
    ("LAYER_COUNT", "num_hidden_layers", "uint"),
    ("OUTPUT_VOCAB_COUNT", "vocab_size", "uint"),
    ("RMS_NORM_EPSILON", "rms_norm_eps", "float"),
        ("MOE_EXPERT_COUNT", "n_routed_experts", "uint"),
    ("MOE_TOP_K", "num_experts_per_tok", "uint"),
    ("MOE_ROUTED_SCALING_FACTOR", "routed_scaling_factor", "float"),
    ("MOE_NORM_TOPK_PROB", "norm_topk_prob", "uint"),
    ("FIRST_ROUTED_LAYER", "first_k_dense_replace", "uint"),
    ("MLA_HEAD_COUNT", "num_attention_heads", "uint"),
    ("MLA_LATENT_DIMENSION", "kv_lora_rank", "uint"),
    ("MLA_QK_NOPE_HEAD_DIMENSION", "qk_nope_head_dim", "uint"),
    ("END_OF_TEXT_TOKEN_ID", "eos_token_id", "uint"),
]
DEFINES_RECORDED_ONLY = [("MLA_V_HEAD_DIMENSION", "v_head_dim", "uint")]


class Glm5NextConfigError(ValueError):
    pass


def define_of(kind, defines, dname):
    return define_uint(defines, dname) if kind == "uint" \
        else define_float(defines, dname)


def scalar_of(kind, value):
    got = value[0] if isinstance(value, list) else value
    return float(got)


def cross_check(defines, config):
    mismatches = []
    for dname, cname, kind in DEFINES_VS_CONFIG + DEFINES_RECORDED_ONLY:
        if cname not in config:
            raise Glm5NextConfigError(f"config key {cname} missing for SPARK_LLM_{dname}")
        want = define_of(kind, defines, dname)
        got = scalar_of(kind, config[cname])
        if abs(want - got) > 0:
            mismatches.append({"define": f"SPARK_LLM_{dname}", "value": want,
                               "config_key": cname, "config_value": got})
    for dname, _, _ in DEFINES_VS_CONFIG:
        for row in mismatches:
            if row["define"] == f"SPARK_LLM_{dname}":
                raise Glm5NextConfigError(
                    f"{row['define']}={row['value']} disagrees with "
                    f"{row['config_key']}={row['config_value']}")
    return mismatches


class Glm5NextEngine:
    def __init__(self, checkpoint_dir, defines, config):
        self.mismatches = cross_check(defines, config)
        self.st = Safetensors(checkpoint_dir)
        self.hidden = int(config["hidden_size"])
        self.layers = int(config["num_hidden_layers"])
        self.vocab = int(config["vocab_size"])
        self.eps = float(config["rms_norm_eps"])
        self.hc = int(config["hc_mult"])
        self.hc_eps = float(config["hc_eps"])
        self.sinkhorn = int(config["hc_sinkhorn_iters"])
        self.heads = int(config["num_attention_heads"])
        self.latent = int(config["kv_lora_rank"])
        self.nope = int(config["qk_nope_head_dim"])
        self.vdim = int(config["v_head_dim"])
        self.kda = config["linear_attn_config"]
        self.kd = int(self.kda["head_dim"])
        self.kheads = int(self.kda["num_heads"])
        self.conv = int(self.kda["short_conv_kernel_size"])
        self.gate_lb = float(self.kda["gate_lower_bound"])
        self.limit = float(config["swiglu_limit"])
        self.experts = int(config["n_routed_experts"])
        self.shared = int(config["n_shared_experts"])
        self.topk = int(config["num_experts_per_tok"])
        self.scaling = float(config["routed_scaling_factor"])
        self.layer_types = list(config["layer_types"])
        self.mlp_types = list(config["mlp_layer_types"])
        self.eot = int(config["eos_token_id"][0] if isinstance(
            config["eos_token_id"], list) else config["eos_token_id"])
        if len(self.layer_types) != self.layers or len(self.mlp_types) != self.layers:
            raise Glm5NextConfigError("layer type lists disagree with layer count")
        dsa_layers = [i for i, t in enumerate(self.layer_types)
                      if t == "deepseek_sparse_attention"]
        if not dsa_layers:
            raise Glm5NextConfigError("no deepseek_sparse_attention layer")
        entry = self.st.entry(
            f"{PREFIX}{dsa_layers[0]}.self_attn.kv_b_proj.weight")
        per_head = entry["shape"][0] // self.heads
        if per_head != self.nope + self.vdim:
            raise Glm5NextConfigError(
                f"kv_b_proj per-head stride {per_head} disagrees with "
                f"nope {self.nope} + v_head_dim {self.vdim}")

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
        raw = self.st.raw_rows("model.language_model.embed_tokens.weight", token_id, 1)
        if raw.dtype != np.uint16:
            raise ValueError("reference embedding must be BF16")
        return bf16_to_f32(raw[0])

    def hc_site(self, prefix, streams, site):
        flat = streams.reshape(-1)
        inv = 1.0 / np.sqrt((flat * flat).sum() / (self.hc * self.hidden) + self.eps)
        mixes = (self.tensor(prefix + site + "_fn") @ flat) * inv
        base = self.st.raw(prefix + site + "_base").astype(np.float32)
        scale = self.st.raw(prefix + site + "_scale").astype(np.float32)
        hc = self.hc
        pre = sigmoid(mixes[0:hc] * scale[0] + base[0:hc]) + self.hc_eps
        post = 2.0 * sigmoid(mixes[hc:2 * hc] * scale[1] + base[hc:2 * hc])
        rows = (mixes[2 * hc:].reshape(hc, hc) * scale[2]
                + base[2 * hc:].reshape(hc, hc))
        comb = np.exp(rows - rows.max())
        comb = comb / comb.sum(axis=1, keepdims=True) + self.hc_eps
        for it in range(self.sinkhorn):
            if it != 0:
                comb = comb / (comb.sum(axis=1, keepdims=True) + self.hc_eps)
            comb = comb / (comb.sum(axis=0, keepdims=True) + self.hc_eps)
        collapsed = bf16_round_f32(np.sum(pre[:, None] * streams, axis=0))
        return collapsed, post, comb

    def hc_post(self, streams, output, post, comb):
        contribution = bf16_round_f32(bf16_round_f32(post)[:, None] * output)
        residual = bf16_round_f32(bf16_round_f32(comb).T @ streams)
        return bf16_round_f32(contribution + residual)

    def short_conv(self, raw, weights, window):
        taps = np.concatenate([window[:, 1:], f32_to_bf16_u16(raw).reshape(-1, 1)],
                              axis=1)
        acc = (bf16_to_f32(taps) * weights).sum(axis=1)
        return bf16_round_f32(acc * sigmoid(acc)), taps

    def l2_heads(self, flat, heads, dim):
        m = flat.reshape(heads, dim)
        return m / np.sqrt((m * m).sum(axis=1, keepdims=True) + QK_L2_EPS)

    def kda_attention(self, prefix, x, layer_state):
        heads, kd = self.kheads, self.kd
        q_raw = self.linear(x, prefix + "self_attn.q_proj")
        k_raw = self.linear(x, prefix + "self_attn.k_proj")
        v_raw = self.linear(x, prefix + "self_attn.v_proj")
        b_raw = self.linear(x, prefix + "self_attn.b_proj")
        q_conv, layer_state["wq"] = self.short_conv(
            q_raw, self.tensor(prefix + "self_attn.q_conv1d.weight").reshape(
                heads * kd, self.conv), layer_state["wq"])
        k_conv, layer_state["wk"] = self.short_conv(
            k_raw, self.tensor(prefix + "self_attn.k_conv1d.weight").reshape(
                heads * kd, self.conv), layer_state["wk"])
        v_conv, layer_state["wv"] = self.short_conv(
            v_raw, self.tensor(prefix + "self_attn.v_conv1d.weight").reshape(
                heads * kd, self.conv), layer_state["wv"])
        latent = self.linear(x, prefix + "self_attn.f_a_proj")
        dl = bf16_round_f32(
            latent @ self.tensor(prefix + "self_attn.f_b_proj.weight").T)
        a_log = self.st.raw(prefix + "self_attn.A_log").astype(np.float32)
        dt_bias = self.st.raw(prefix + "self_attn.dt_bias").astype(np.float32)
        scaled = (np.exp(a_log).reshape(heads, 1)
                  * (dl.reshape(heads, kd) + dt_bias.reshape(heads, kd)))
        retention = np.exp(self.gate_lb * sigmoid(scaled))
        beta = sigmoid(b_raw).reshape(heads)
        q2 = self.l2_heads(q_conv, heads, kd) / np.sqrt(np.float32(kd))
        k2 = self.l2_heads(k_conv, heads, kd)
        v2 = v_conv.reshape(heads, kd)
        state = layer_state["state"]
        out = np.empty((heads, kd), dtype=np.float32)
        for h in range(heads):
            pred = (state[h] * (k2[h] * retention[h])[:, None]).sum(axis=0)
            state[h] = (retention[h][:, None] * state[h]
                        + beta[h] * (v2[h] - pred)[None, :] * k2[h][:, None])
            out[h] = (state[h] * q2[h][:, None]).sum(axis=0)
        return self.kda_out(prefix, x, out.reshape(-1))

    def kda_out(self, prefix, x, delta_out):
        o32 = delta_out.reshape(self.kheads, self.kd)
        rms = np.sqrt((o32 * o32).sum(axis=1) / self.kd + self.eps)
        o_norm = bf16_to_f32(
            self.st.raw(prefix + "self_attn.o_norm.weight").reshape(-1))
        gated_o = o32 / rms[:, None] * o_norm[None, :]
        gate = bf16_round_f32(bf16_round_f32(
            self.linear(x, prefix + "self_attn.g_a_proj"))
            @ self.tensor(prefix + "self_attn.g_b_proj.weight").T)
        gs = sigmoid(gate.reshape(self.kheads, self.kd))
        return self.linear((gated_o * gs).reshape(-1), prefix + "self_attn.o_proj")

    def dsa_attention(self, prefix, x, cache):
        q_norm = bf16_round_f32(rmsnorm(
            self.linear(x, prefix + "self_attn.q_a_proj"),
            self.tensor(prefix + "self_attn.q_a_layernorm.weight"), self.eps))
        q = bf16_round_f32(
            q_norm @ self.tensor(prefix + "self_attn.q_b_proj.weight").T)
        kv_raw = self.linear(x, prefix + "self_attn.kv_a_proj_with_mqa")
        slot = bf16_round_f32(rmsnorm(
            kv_raw, self.tensor(prefix + "self_attn.kv_a_layernorm.weight"),
            self.eps))
        cache.append(slot)
        kvb = self.st.raw(prefix + "self_attn.kv_b_proj.weight")
        slots = np.stack([bf16_to_f32(row) for row in cache])
        qh = q.reshape(self.heads, self.nope)
        attn = np.empty((self.heads, self.vdim), dtype=np.float32)
        for h in range(self.heads):
            base = h * (self.nope + self.vdim)
            wk = bf16_to_f32(kvb[base:base + self.nope])
            wv = bf16_to_f32(kvb[base + self.nope:base + self.nope + self.vdim])
            ql = bf16_round_f32(qh[h] @ wk)
            scores = (slots @ ql) * np.float32(self.nope ** -0.5)
            weights = np.exp(scores - scores.max())
            weights = weights / weights.sum()
            al = bf16_round_f32(weights @ slots)
            attn[h] = bf16_round_f32(al @ wv.T)
        return self.linear(attn.reshape(-1), prefix + "self_attn.o_proj")

    def expert_mlp(self, prefix, x):
        gate = np.minimum(self.linear(x, prefix + "gate_proj"), self.limit)
        up = np.clip(self.linear(x, prefix + "up_proj"), -self.limit, self.limit)
        activated = bf16_round_f32(bf16_round_f32(gate * sigmoid(gate)) * up)
        return self.linear(activated, prefix + "down_proj")

    def sparse_mlp(self, prefix, x, sink):
        scores = sigmoid(self.tensor(prefix + "mlp.gate.weight") @ x)
        choice = scores + self.tensor(prefix + "mlp.gate.e_score_correction_bias")
        order = np.argsort(-choice, kind="stable")
        selected = np.sort(order[:self.topk])
        picked = scores[selected]
        weights = picked / (picked.sum() + 1e-20) * self.scaling
        routed = np.zeros_like(x)
        for i in range(self.topk):
            expert = prefix + f"mlp.experts.{selected[i]}."
            output = self.expert_mlp(expert, x)
            routed = bf16_round_f32(routed + bf16_round_f32(output * weights[i]))
        shared = self.expert_mlp(prefix + "mlp.shared_experts.", x)
        sink.append(selected.astype(np.int32))
        sink.append(weights.astype(np.float32))
        return bf16_round_f32(routed + shared)

    def forward_layer(self, index, streams, states, caches, sink):
        prefix = PREFIX + str(index) + "."
        collapsed, post, comb = self.hc_site(prefix, streams, "hc_attn")
        x = bf16_round_f32(rmsnorm(
            collapsed, self.tensor(prefix + "input_layernorm.weight"), self.eps))
        if self.layer_types[index] == "linear_attention":
            attention = self.kda_attention(prefix, x, states[index])
        elif self.layer_types[index] == "deepseek_sparse_attention":
            if len(caches[index]) >= MAX_INDEXER_CONTEXT:
                raise ValueError("context exceeds indexer topk; reference undefined")
            attention = self.dsa_attention(prefix, x, caches[index])
        else:
            raise ValueError(f"unsupported layer type {self.layer_types[index]}")
        streams = self.hc_post(streams, attention, post, comb)
        collapsed, post, comb = self.hc_site(prefix, streams, "hc_ffn")
        x = bf16_round_f32(rmsnorm(
            collapsed, self.tensor(prefix + "post_attention_layernorm.weight"),
            self.eps))
        if self.mlp_types[index] == "dense":
            mlp = self.expert_mlp(prefix + "mlp.", x)
        elif self.mlp_types[index] == "sparse":
            mlp = self.sparse_mlp(prefix, x, sink)
        else:
            raise ValueError(f"unsupported mlp type {self.mlp_types[index]}")
        return self.hc_post(streams, mlp, post, comb)

    def decode_step(self, token_id, position, states, caches, capture):
        if token_id < 0 or token_id >= self.vocab:
            raise ValueError(f"token {token_id} outside vocabulary {self.vocab}")
        raw = self.st.raw_rows("model.language_model.embed_tokens.weight", token_id, 1)
        if raw.dtype != np.uint16:
            raise ValueError("reference embedding must be BF16")
        embedding = bf16_to_f32(raw[0])
        if position == 0:
            for i in range(self.layers):
                if self.layer_types[i] == "linear_attention":
                    states[i] = {"state": np.zeros((self.kheads, self.kd, self.kd),
                                                   dtype=np.float32),
                                 "wq": np.zeros((self.kheads * self.kd, self.conv),
                                                dtype=np.uint16),
                                 "wk": np.zeros((self.kheads * self.kd, self.conv),
                                                dtype=np.uint16),
                                 "wv": np.zeros((self.kheads * self.kd, self.conv),
                                                dtype=np.uint16)}
                else:
                    caches[i] = []
        streams = np.tile(embedding, (self.hc, 1))
        for i in range(self.layers):
            sink = []
            streams = self.forward_layer(i, streams, states, caches, sink)
            if sink:
                capture[(position, i)] = sink
        return streams

    def logits(self, streams, chunk=4096):
        collapsed = bf16_round_f32(streams.mean(axis=0))
        norm = bf16_round_f32(rmsnorm(
            collapsed, self.tensor("model.language_model.norm.weight"), self.eps))
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


def kvb_head_stride(config, st):
    entry = st.entry("model.language_model.layers.3.self_attn.kv_b_proj.weight")
    per_head = entry["shape"][0] // int(config["num_attention_heads"])
    if per_head != int(config["qk_nope_head_dim"]) + 256:
        raise Glm5NextConfigError(
            f"kv_b_proj per-head stride {per_head} disagrees with nope+256")
    return per_head
ENGINE_CLASS = Glm5NextEngine
