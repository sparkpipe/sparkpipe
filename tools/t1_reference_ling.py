import numpy as np

from t1_reference_common import (Safetensors, bf16_round_f32, bf16_to_f32,
                                 define_float, define_uint, f32_to_bf16_u16,
                                 fp8_block_to_bf16, rmsnorm, sigmoid)

PREFIX = "model.layers."
L2_EPSILON = 1e-6
DEFINES_VS_CONFIG = [
    ("HIDDEN_DIMENSION", "hidden_size", "uint"),
    ("LAYER_COUNT", "num_hidden_layers", "uint"),
    ("OUTPUT_VOCAB_COUNT", "vocab_size", "uint"),
    ("RMS_NORM_EPSILON", "rms_norm_eps", "float"),
    ("END_OF_TEXT_TOKEN_ID", "eos_token_id", "uint"),
    ("ATTENTION_PERIOD", "layer_group_size", "uint"),
    ("MLA_HEAD_COUNT", "num_attention_heads", "uint"),
    ("MLA_LATENT_DIMENSION", "kv_lora_rank", "uint"),
    ("MLA_QK_NOPE_HEAD_DIMENSION", "qk_nope_head_dim", "uint"),
    ("MLA_QK_ROPE_HEAD_DIMENSION", "qk_rope_head_dim", "uint"),
    ("MLA_VALUE_HEAD_DIMENSION", "v_head_dim", "uint"),
    ("MLA_ROPE_THETA", "rope_theta", "float"),
    ("KDA_CONV_KERNEL", "short_conv_kernel_size", "uint"),
    ("KDA_GATE_LOWER_BOUND", "kda_lower_bound", "float"),
    ("MOE_EXPERT_COUNT", "num_experts", "uint"),
    ("MOE_TOP_K", "num_experts_per_tok", "uint"),
    ("MOE_ROUTER_GROUP_COUNT", "n_group", "uint"),
    ("MOE_ROUTER_TOP_GROUPS", "topk_group", "uint"),
    ("MOE_ROUTED_SCALING_FACTOR", "routed_scaling_factor", "float"),
    ("MOE_INTERMEDIATE_DIMENSION", "moe_intermediate_size", "uint"),
    ("DENSE_INTERMEDIATE_DIMENSION", "intermediate_size", "uint"),
    ("FIRST_ROUTED_LAYER", "first_k_dense_replace", "uint"),
]


class LingConfigError(ValueError):
    pass


def define_of(kind, defines, dname):
    return define_uint(defines, dname) if kind == "uint" \
        else define_float(defines, dname)


def cross_check(defines, config):
    for dname, cname, kind in DEFINES_VS_CONFIG:
        if cname not in config:
            raise LingConfigError(f"config key {cname} missing for SPARK_LLM_{dname}")
        want = define_of(kind, defines, dname)
        got = config[cname]
        if isinstance(got, list):
            got = got[0]
        if abs(float(want) - float(got)) > 0:
            raise LingConfigError(
                f"SPARK_LLM_{dname}={want} disagrees with config {cname}={got}")
    head_dim = define_uint(defines, "MLA_QK_NOPE_HEAD_DIMENSION") \
        + define_uint(defines, "MLA_QK_ROPE_HEAD_DIMENSION")
    derived_scale = float(np.float32(head_dim ** -0.5))
    pinned_scale = float(np.float32(define_float(defines, "MLA_QK_SCALE")))
    if abs(derived_scale - pinned_scale) > 0.0:
        raise LingConfigError(
            f"SPARK_LLM_MLA_QK_SCALE={pinned_scale} disagrees with "
            f"qk_head_dim^-0.5={derived_scale}")
    phase = define_uint(defines, "GLOBAL_ATTENTION_PHASE")
    period = define_uint(defines, "ATTENTION_PERIOD")
    layers = define_uint(defines, "LAYER_COUNT")
    mla = [i for i in range(layers) if i % period == phase]
    if len(mla) != define_uint(defines, "MLA_LAYER_COUNT"):
        raise LingConfigError(
            f"SPARK_LLM_MLA_LAYER_COUNT={define_uint(defines, 'MLA_LAYER_COUNT')} "
            f"disagrees with phase {phase} pattern {mla}")
    return []


class LingEngine:
    def __init__(self, checkpoint_dir, defines, config):
        self.mismatches = cross_check(defines, config)
        self.st = Safetensors(checkpoint_dir)
        self.hidden = int(config["hidden_size"])
        self.layers = int(config["num_hidden_layers"])
        self.vocab = int(config["vocab_size"])
        self.eps = float(config["rms_norm_eps"])
        self.period = int(config["layer_group_size"])
        self.phase = define_uint(defines, "GLOBAL_ATTENTION_PHASE")
        self.mla_layers = set(i for i in range(self.layers)
                              if i % self.period == self.phase)
        self.heads = int(config["num_attention_heads"])
        self.nope = int(config["qk_nope_head_dim"])
        self.rope_dim = int(config["qk_rope_head_dim"])
        self.vdim = int(config["v_head_dim"])
        self.head_dim = self.nope + self.rope_dim
        self.latent = int(config["kv_lora_rank"])
        self.qk_scale = float(np.float32(self.head_dim ** -0.5))
        self.rope_theta = float(config["rope_theta"])
        self.kda_heads = self.heads
        self.kd = int(config["head_dim"])
        self.conv = int(config["short_conv_kernel_size"])
        self.lower = float(config["kda_lower_bound"])
        self.experts = int(config["num_experts"])
        self.topk = int(config["num_experts_per_tok"])
        self.groups = int(config["n_group"])
        self.top_groups = int(config["topk_group"])
        self.scaling = float(config["routed_scaling_factor"])
        self.expert_inter = int(config["moe_intermediate_size"])
        self.first_routed = int(config["first_k_dense_replace"])
        self.eot = int(config["eos_token_id"])
        self.states = {}
        self.caches = {}
        self.position = 0
        self.rope_freq = np.exp2(-(2.0 * np.arange(self.rope_dim // 2,
                                                   dtype=np.float32)
                                  / np.float32(self.rope_dim))
                                 * np.log2(np.float32(self.rope_theta)))
        self.embed_name = "model.word_embeddings.weight" \
            if "model.word_embeddings.weight" in self.st.map \
            else "model.embed_tokens.weight"
        kvb = self.st.entry(f"{PREFIX}{min(self.mla_layers)}.attention.kv_b_proj.weight")
        if kvb["shape"][0] != self.heads * (self.nope + self.vdim):
            raise LingConfigError(
                f"kv_b_proj rows {kvb['shape'][0]} disagree with "
                f"heads*(nope+vdim) {self.heads * (self.nope + self.vdim)}")
        gate = self.st.entry(f"{PREFIX}{min(self.mla_layers)}.attention.g_proj.weight")
        if gate["shape"][0] != self.heads:
            raise LingConfigError("MLA g_proj rows disagree with head count")

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
        raw = self.st.raw_rows(self.embed_name, token_id, 1)
        if raw.dtype != np.uint16:
            raise ValueError("reference embedding must be BF16")
        return bf16_to_f32(raw[0])

    def rope_interleaved(self, rows, offset, dim, position):
        half = dim // 2
        angles = np.float32(position) * self.rope_freq
        cos = np.cos(angles)
        sin = np.sin(angles)
        section = rows[..., offset:offset + dim]
        a = section[..., 0::2].copy()
        b = section[..., 1::2].copy()
        section[..., 0::2] = a * cos - b * sin
        section[..., 1::2] = a * sin + b * cos
        return rows

    def short_conv(self, index, name, raw):
        p = f"{PREFIX}{index}.attention."
        window = self.states[index][name]
        taps = np.concatenate([window[:, 1:],
                               f32_to_bf16_u16(raw).reshape(-1, 1)], axis=1)
        weights = self.tensor(p + name + "_conv1d.weight").reshape(-1, self.conv)
        total = (bf16_to_f32(taps) * weights).sum(axis=1)
        self.states[index][name] = taps
        return bf16_round_f32(total * sigmoid(total))

    def kda_attention(self, index, x):
        p = f"{PREFIX}{index}.attention."
        heads, kd = self.kda_heads, self.kd
        q = self.short_conv(index, "q", self.linear(x, p + "q_proj"))
        k = self.short_conv(index, "k", self.linear(x, p + "k_proj"))
        v = self.short_conv(index, "v", self.linear(x, p + "v_proj"))
        beta = sigmoid(self.linear(x, p + "b_proj"))
        retention = bf16_round_f32(self.linear(x, p + "f_proj"))
        a_log = self.st.raw(p + "A_log").astype(np.float32)
        dt_bias = self.st.raw(p + "dt_bias").astype(np.float32)
        scaled = np.exp(a_log).reshape(heads, 1) \
            * (retention.reshape(heads, kd) + dt_bias.reshape(heads, kd))
        retention = np.exp(np.float32(self.lower) * sigmoid(scaled))
        gate = bf16_round_f32(self.linear(x, p + "g_proj"))
        q = q.reshape(heads, kd)
        k = k.reshape(heads, kd)
        v = v.reshape(heads, kd)
        q = bf16_round_f32(q / np.sqrt((q * q).sum(axis=1, keepdims=True)
                                       + L2_EPSILON))
        q = q / np.float32(np.sqrt(kd))
        k = bf16_round_f32(k / np.sqrt((k * k).sum(axis=1, keepdims=True)
                                       + L2_EPSILON))
        k = k / np.sqrt((k * k).sum(axis=1, keepdims=True) + L2_EPSILON)
        state = self.states[index]["state"]
        out = np.empty((heads, kd), dtype=np.float32)
        for h in range(heads):
            predicted = (state[h] * (k[h] * retention[h])[:, None]).sum(axis=0)
            delta = beta[h] * (v[h] - predicted)
            state[h] = retention[h][:, None] * state[h] + delta * k[h][:, None]
            out[h] = bf16_round_f32((state[h] * q[h][:, None]).sum(axis=0))
        norm_w = bf16_to_f32(self.st.raw(p + "o_norm.weight").reshape(-1))
        rms = np.sqrt((out * out).sum(axis=1) / kd + self.eps)
        core = bf16_round_f32(out / rms[:, None] * norm_w[None, :])
        core = bf16_round_f32(core * sigmoid(gate.reshape(heads, kd)))
        return self.linear(core.reshape(-1), p + "o_proj")

    def mla_attention(self, index, x, position):
        p = f"{PREFIX}{index}.attention."
        heads, nope, rope, latent, vdim = (self.heads, self.nope, self.rope_dim,
                                           self.latent, self.vdim)
        q = self.linear(x, p + "q_proj").reshape(heads, self.head_dim)
        self.rope_interleaved(q, nope, rope, position)
        q = bf16_round_f32(q)
        slot = self.linear(x, p + "kv_a_proj_with_mqa")
        slot[:latent] = rmsnorm(slot[:latent],
                                self.tensor(p + "kv_a_layernorm.weight"), self.eps)
        self.rope_interleaved(slot, latent, rope, position)
        slot = f32_to_bf16_u16(slot)
        cache = self.caches[index]
        cache.append(slot)
        slots = bf16_to_f32(np.stack(cache))
        kvb = bf16_to_f32(self.st.raw(p + "kv_b_proj.weight")) \
            .reshape(heads, nope + vdim, latent)
        gates = sigmoid(self.tensor(p + "g_proj.weight") @ x).reshape(heads, 1)
        attn = np.empty((heads, vdim), dtype=np.float32)
        for h in range(heads):
            q_latent = kvb[h, :nope, :].T @ q[h, :nope]
            scores = (slots[:, :latent] @ q_latent
                      + slots[:, latent:] @ q[h, nope:]) * np.float32(self.qk_scale)
            weights = np.exp(scores - scores.max())
            weights = weights / weights.sum()
            attention_latent = weights @ slots[:, :latent]
            attn[h] = (kvb[h, nope:, :] @ attention_latent) * gates[h]
        return self.linear(attn.reshape(-1), p + "dense")

    def expert_mlp(self, prefix, x):
        gate = self.linear(x, prefix + "gate_proj")
        up = self.linear(x, prefix + "up_proj")
        activated = bf16_round_f32(bf16_round_f32(gate * sigmoid(gate)) * up)
        return self.linear(activated, prefix + "down_proj")

    def sparse_mlp(self, index, x, sink):
        p = f"{PREFIX}{index}.mlp."
        scores = sigmoid(self.tensor(p + "gate.weight") @ x)
        choice = scores + self.tensor(p + "gate.expert_bias").reshape(-1)
        per_group = self.experts // self.groups
        top2 = np.sort(choice.reshape(self.groups, per_group), axis=1)[:, -2:]
        group_key = top2[:, 0] + top2[:, 1]
        chosen_groups = np.argsort(-group_key, kind="stable")[:self.top_groups]
        mask = np.zeros(self.experts, dtype=bool)
        for g in chosen_groups:
            mask[g * per_group:(g + 1) * per_group] = True
        masked = np.where(mask, choice, -np.inf)
        selected = np.argsort(-masked, kind="stable")[:self.topk]
        picked = scores[selected]
        weights = picked / (picked.sum() + 1e-20) * np.float32(self.scaling)
        routed = np.zeros_like(x)
        for i in range(self.topk):
            output = self.expert_mlp(p + f"experts.{int(selected[i])}.", x)
            routed = routed + output * np.float32(weights[i])
        shared = self.expert_mlp(p + "shared_experts.", x)
        sink.append(selected.astype(np.int32))
        sink.append(weights.astype(np.float32))
        return bf16_round_f32(routed + shared)

    def forward_layer(self, index, streams, sink):
        p = f"{PREFIX}{index}."
        x = bf16_round_f32(rmsnorm(streams,
                                   self.tensor(p + "input_layernorm.weight"),
                                   self.eps))
        if index in self.mla_layers:
            attention = self.mla_attention(index, x, self.position)
        else:
            attention = self.kda_attention(index, x)
        streams = bf16_round_f32(streams + attention)
        x = bf16_round_f32(rmsnorm(streams,
                                   self.tensor(p + "post_attention_layernorm.weight"),
                                   self.eps))
        if index < self.first_routed:
            mlp = self.expert_mlp(p + "mlp.", x)
        else:
            mlp = self.sparse_mlp(index, x, sink)
        return bf16_round_f32(streams + mlp)

    def decode_step(self, token_id, position, states, caches, capture):
        if token_id < 0 or token_id >= self.vocab:
            raise ValueError(f"token {token_id} outside vocabulary {self.vocab}")
        self.position = position
        if position == 0:
            self.states = {}
            self.caches = {}
            for i in range(self.layers):
                if i in self.mla_layers:
                    self.caches[i] = []
                else:
                    self.states[i] = {
                        "state": np.zeros((self.kda_heads, self.kd, self.kd),
                                          dtype=np.float32),
                        "q": np.zeros((self.kda_heads * self.kd, self.conv),
                                      dtype=np.uint16),
                        "k": np.zeros((self.kda_heads * self.kd, self.conv),
                                      dtype=np.uint16),
                        "v": np.zeros((self.kda_heads * self.kd, self.conv),
                                      dtype=np.uint16),
                    }
        streams = self.embed(token_id)
        for i in range(self.layers):
            sink = []
            streams = self.forward_layer(i, streams, sink)
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


ENGINE_CLASS = LingEngine
