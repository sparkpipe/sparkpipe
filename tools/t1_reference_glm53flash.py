import os
from collections import OrderedDict

import numpy as np

from t1_reference_common import (Safetensors, _E4M3_LUT, bf16_round_f32,
                                 bf16_to_f32, define_float, define_uint,
                                 f32_to_bf16_u16, fp8_block_to_bf16,
                                 nvfp4_to_f32, rmsnorm, sigmoid)

PREFIX = "model.language_model.layers."
QK_L2_EPS = 1e-6
DEFINES_VS_CONFIG = [
    ("HIDDEN_DIMENSION", "hidden_size", "uint"),
    ("LAYER_COUNT", "num_hidden_layers", "uint"),
    ("OUTPUT_VOCAB_COUNT", "vocab_size", "uint"),
    ("RMS_NORM_EPSILON", "rms_norm_eps", "float"),
    ("SWIGLU_LIMIT", "swiglu_limit", "float"),
    ("MAXIMUM_CONTEXT_TOKENS", "max_position_embeddings", "uint"),
    ("END_OF_TEXT_TOKEN_ID", "eos_token_id", "uint"),
    ("MOE_EXPERT_COUNT", "n_routed_experts", "uint"),
    ("MOE_TOP_K", "num_experts_per_tok", "uint"),
    ("MOE_SHARED_EXPERT_COUNT", "n_shared_experts", "uint"),
    ("MOE_INTERMEDIATE_DIMENSION", "moe_intermediate_size", "uint"),
    ("MOE_ROUTED_SCALING_FACTOR", "routed_scaling_factor", "float"),
    ("MOE_NORM_TOPK_PROB", "norm_topk_prob", "uint"),
    ("FIRST_ROUTED_LAYER", "first_k_dense_replace", "uint"),
    ("DENSE_INTERMEDIATE_DIMENSION", "intermediate_size", "uint"),
    ("MLA_HEAD_COUNT", "num_attention_heads", "uint"),
    ("MLA_QUERY_A_DIMENSION", "q_lora_rank", "uint"),
    ("MLA_LATENT_DIMENSION", "kv_lora_rank", "uint"),
    ("MLA_QK_NOPE_HEAD_DIMENSION", "qk_nope_head_dim", "uint"),
    ("MLA_QK_ROPE_HEAD_DIMENSION", "qk_rope_head_dim", "uint"),
    ("KDA_HEAD_COUNT", ("linear_attn_config", "num_heads"), "uint"),
    ("KDA_HEAD_KEY_DIMENSION", ("linear_attn_config", "head_dim"), "uint"),
    ("KDA_CONV_KERNEL", ("linear_attn_config", "short_conv_kernel_size"),
     "uint"),
    ("KDA_GATE_LOWER_BOUND", ("linear_attn_config", "gate_lower_bound"),
     "float"),
]
DEFINES_RECORDED_ONLY = [
    ("MLA_V_HEAD_DIMENSION", "v_head_dim", "uint"),
]
LINEAR_ATTENTION = "linear_attention"
SPARSE_ATTENTION = "deepseek_sparse_attention"


class Glm53FlashConfigError(ValueError):
    pass


def define_of(kind, defines, dname):
    return define_uint(defines, dname) if kind == "uint" \
        else define_float(defines, dname)


def config_value(config, cname):
    if isinstance(cname, tuple):
        value = config
        for key in cname:
            if key not in value:
                raise Glm53FlashConfigError(
                    f"config key {'.'.join(cname)} missing for the flash "
                    f"family closure")
            value = value[key]
        return value
    return config[cname]


def scalar_of(value):
    got = value[0] if isinstance(value, list) else value
    return float(got)


def cross_check(defines, config):
    mismatches = []
    for dname, cname, kind in DEFINES_VS_CONFIG + DEFINES_RECORDED_ONLY:
        want = define_of(kind, defines, dname)
        got = scalar_of(config_value(config, cname))
        if abs(want - got) > 0:
            mismatches.append({"define": f"SPARK_LLM_{dname}", "value": want,
                               "config_key": ".".join(cname) if
                               isinstance(cname, tuple) else cname,
                               "config_value": got})
    for dname, _, _ in DEFINES_VS_CONFIG:
        for row in mismatches:
            if row["define"] == f"SPARK_LLM_{dname}":
                raise Glm53FlashConfigError(
                    f"{row['define']}={row['value']} disagrees with "
                    f"{row['config_key']}={row['config_value']}")
    return mismatches


class RawCache:
    def __init__(self, cap):
        self.cap = cap
        self.bytes = 0
        self.entries = OrderedDict()

    def get(self, name):
        if name not in self.entries:
            return None
        self.entries.move_to_end(name)
        return self.entries[name]

    def put(self, name, array):
        if name in self.entries:
            return
        self.entries[name] = array
        self.bytes += array.nbytes
        while self.bytes > self.cap and len(self.entries) > 1:
            _, victim = self.entries.popitem(last=False)
            self.bytes -= victim.nbytes


class Glm53FlashEngine:
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
        self.topk = int(config["num_experts_per_tok"])
        self.norm_topk = int(config["norm_topk_prob"])
        self.scaling = float(config["routed_scaling_factor"])
        self.index_topk = int(config["index_topk"])
        self.eot = int(config["eos_token_id"][0] if isinstance(
            config["eos_token_id"], list) else config["eos_token_id"])
        self.layer_types = list(config["layer_types"])
        self.mlp_types = list(config["mlp_types"] if "mlp_types" in config
                              else config["mlp_layer_types"])
        if len(self.layer_types) != self.layers or \
                len(self.mlp_types) != self.layers:
            raise Glm53FlashConfigError(
                "layer type lists disagree with layer count")
        kda_layers = [i for i, t in enumerate(self.layer_types)
                      if t == LINEAR_ATTENTION]
        full_layers = [i for i, t in enumerate(self.layer_types)
                       if t == SPARSE_ATTENTION]
        if kda_layers != list(self.kda["kda_layers"]) or \
                full_layers != list(self.kda["full_attn_layers"]):
            raise Glm53FlashConfigError(
                "linear_attn_config layer lists disagree with layer_types")
        if not kda_layers or not full_layers:
            raise Glm53FlashConfigError(
                "flash family requires both linear and sparse attention "
                "layers")
        self.first_routed = int(config["first_k_dense_replace"])
        entry = self.st.entry(f"{PREFIX}{full_layers[0]}.self_attn."
                              "kv_b_proj.weight")
        if entry["dtype"] != "BF16":
            raise Glm53FlashConfigError(
                f"kv_b_proj payload {entry['dtype']} is not BF16; the "
                f"reference kv_b spine is BF16 in every release")
        per_head = entry["shape"][0] // self.heads
        if per_head != self.nope + self.vdim:
            raise Glm53FlashConfigError(
                f"kv_b_proj per-head stride {per_head} disagrees with "
                f"nope {self.nope} + v_head_dim {self.vdim}")
        self.cache = RawCache(int(os.environ.get("T1_REF_CACHE_BYTES",
                                                 768 * (1 << 20))))

    def raw(self, name):
        held = self.cache.get(name)
        if held is not None:
            return held
        array = self.st.pread(name)
        self.cache.put(name, array)
        return array

    def scale_tensor(self, name, scale_name, global_name):
        raw = self.raw(name)
        rows, packed_cols = raw.shape
        real_cols = packed_cols * 2
        scales = self.raw(scale_name)
        if scales.shape != (rows, real_cols // 16):
            raise ValueError(f"{scale_name}: shape {scales.shape}, expected "
                             f"{(rows, real_cols // 16)}")
        global_f32 = float(self.raw(global_name).reshape(()))
        values = nvfp4_to_f32(raw, scales, rows, real_cols)
        return bf16_to_f32(f32_to_bf16_u16(values * global_f32))

    def tensor(self, name):
        raw = self.raw(name)
        if raw.dtype == np.uint16:
            return bf16_to_f32(raw)
        if raw.dtype == np.float32:
            return raw.astype(np.float32)
        entry = self.st.entry(name)
        if entry["dtype"] == "F8_E4M3":
            scale_name = name + "_scale_inv"
            scale = self.raw(scale_name).astype(np.float32)
            if scale.shape != ((raw.shape[0] + 127) // 128,
                               (raw.shape[1] + 127) // 128):
                raise ValueError(f"{scale_name}: shape {scale.shape} does "
                                 f"not block {raw.shape}")
            return bf16_to_f32(fp8_block_to_bf16(raw, scale, raw.shape[0],
                                                 raw.shape[1]))
        if entry["dtype"] == "U8":
            return self.scale_tensor(name, name + "_scale",
                                     name + "_scale_2")
        raise ValueError(f"{name}: unsupported spine payload "
                         f"{entry['dtype']}")

    def expert_weight(self, name):
        raw = self.raw(name)
        entry = self.st.entry(name)
        if raw.dtype == np.uint16:
            return bf16_to_f32(raw)
        if entry["dtype"] == "F8_E4M3":
            scale_name = name + "_scale_inv"
            scale = self.raw(scale_name).astype(np.float32)
            rows, cols = raw.shape
            codes = _E4M3_LUT[raw].astype(np.float32)
            tiled = np.repeat(np.repeat(scale, 128, axis=0), 128, axis=1)
            return codes * tiled[:rows, :cols]
        if entry["dtype"] == "U8":
            rows, packed_cols = raw.shape
            real_cols = packed_cols * 2
            scale_name = name + "_scale"
            scales = self.raw(scale_name)
            if scales.shape != (rows, real_cols // 16):
                raise ValueError(f"{scale_name}: shape {scales.shape}, "
                                 f"expected {(rows, real_cols // 16)}")
            global_name = name + "_scale_2"
            global_f32 = float(self.raw(global_name).reshape(()))
            values = nvfp4_to_f32(raw, scales, rows, real_cols)
            return values * global_f32
        raise ValueError(f"{name}: unsupported expert payload "
                         f"{entry['dtype']}")

    def linear(self, x, name):
        return bf16_round_f32(self.tensor(name + ".weight") @ x)

    def embed(self, token_id):
        if token_id < 0 or token_id >= self.vocab:
            raise ValueError(f"token {token_id} outside vocabulary "
                             f"{self.vocab}")
        raw = self.st.raw_rows("model.language_model.embed_tokens.weight",
                               token_id, 1)
        if raw.dtype != np.uint16:
            raise ValueError("reference embedding must be BF16")
        return bf16_to_f32(raw[0])

    def hc_site(self, prefix, streams, site):
        flat = streams.reshape(-1)
        inv = 1.0 / np.sqrt((flat * flat).sum() / (self.hc * self.hidden)
                            + self.eps)
        mixes = (self.tensor(prefix + site + "_fn") @ flat) * inv
        base = self.tensor(prefix + site + "_base")
        scale = self.tensor(prefix + site + "_scale")
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
        contribution = bf16_round_f32(
            bf16_round_f32(post)[:, None] * output)
        residual = bf16_round_f32(bf16_round_f32(comb).T @ streams)
        return bf16_round_f32(contribution + residual)

    def short_conv(self, raw_f32, weights, window):
        taps = np.concatenate(
            [window[:, 1:], f32_to_bf16_u16(raw_f32).reshape(-1, 1)], axis=1)
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
        q_conv_w = bf16_round_f32(self.tensor(
            prefix + "self_attn.q_conv1d.weight").reshape(heads * kd,
                                                          self.conv))
        k_conv_w = bf16_round_f32(self.tensor(
            prefix + "self_attn.k_conv1d.weight").reshape(heads * kd,
                                                          self.conv))
        v_conv_w = bf16_round_f32(self.tensor(
            prefix + "self_attn.v_conv1d.weight").reshape(heads * kd,
                                                          self.conv))
        q_conv, layer_state["wq"] = self.short_conv(q_raw, q_conv_w,
                                                    layer_state["wq"])
        k_conv, layer_state["wk"] = self.short_conv(k_raw, k_conv_w,
                                                    layer_state["wk"])
        v_conv, layer_state["wv"] = self.short_conv(v_raw, v_conv_w,
                                                    layer_state["wv"])
        latent = self.linear(x, prefix + "self_attn.f_a_proj")
        dl = bf16_round_f32(
            latent @ self.tensor(prefix + "self_attn.f_b_proj.weight").T)
        a_log = self.raw(prefix + "self_attn.A_log").astype(np.float32)
        dt_bias = self.raw(prefix + "self_attn.dt_bias").astype(np.float32)
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
            self.raw(prefix + "self_attn.o_norm.weight").reshape(-1))
        gated_o = o32 / rms[:, None] * o_norm[None, :]
        gate = bf16_round_f32(bf16_round_f32(
            self.linear(x, prefix + "self_attn.g_a_proj"))
            @ self.tensor(prefix + "self_attn.g_b_proj.weight").T)
        gs = sigmoid(gate.reshape(self.kheads, self.kd))
        return self.linear((gated_o * gs).reshape(-1),
                           prefix + "self_attn.o_proj")

    def dsa_attention(self, prefix, x, cache):
        q_norm = bf16_round_f32(rmsnorm(
            self.linear(x, prefix + "self_attn.q_a_proj"),
            self.tensor(prefix + "self_attn.q_a_layernorm.weight"), self.eps))
        q = bf16_round_f32(
            q_norm @ self.tensor(prefix + "self_attn.q_b_proj.weight").T)
        kv_raw = self.linear(x, prefix + "self_attn.kv_a_proj_with_mqa")
        slot = bf16_round_f32(rmsnorm(
            kv_raw[:self.latent],
            self.tensor(prefix + "self_attn.kv_a_layernorm.weight"),
            self.eps))
        cache.append(f32_to_bf16_u16(slot))
        kvb = self.raw(prefix + "self_attn.kv_b_proj.weight")
        if kvb.dtype != np.uint16:
            raise ValueError("reference kv_b_proj must be BF16")
        slots = bf16_to_f32(np.stack(cache))
        qh = q.reshape(self.heads, self.nope)
        attn = np.empty((self.heads, self.vdim), dtype=np.float32)
        for h in range(self.heads):
            base = h * (self.nope + self.vdim)
            wk = bf16_to_f32(kvb[base:base + self.nope])
            wv = bf16_to_f32(kvb[base + self.nope:base + self.nope
                                 + self.vdim])
            ql = bf16_round_f32(qh[h] @ wk)
            scores = (slots @ ql) * np.float32(self.nope ** -0.5)
            weights = np.exp(scores - scores.max())
            weights = weights / weights.sum()
            al = bf16_round_f32(weights @ slots)
            attn[h] = bf16_round_f32(al @ wv.T)
        return self.linear(attn.reshape(-1), prefix + "self_attn.o_proj")

    def routed_expert_mlp(self, expert_prefix, x):
        gate = np.minimum(bf16_round_f32(
            self.expert_weight(expert_prefix + "gate_proj.weight") @ x),
            self.limit)
        up = np.clip(bf16_round_f32(
            self.expert_weight(expert_prefix + "up_proj.weight") @ x),
            -self.limit, self.limit)
        activated = bf16_round_f32(bf16_round_f32(gate * sigmoid(gate)) * up)
        return bf16_round_f32(
            self.expert_weight(expert_prefix + "down_proj.weight")
            @ activated)

    def spine_mlp(self, mlp_prefix, x):
        gate = np.minimum(self.linear(x, mlp_prefix + "gate_proj"),
                          self.limit)
        up = np.clip(self.linear(x, mlp_prefix + "up_proj"), -self.limit,
                     self.limit)
        activated = bf16_round_f32(bf16_round_f32(gate * sigmoid(gate)) * up)
        return self.linear(activated, mlp_prefix + "down_proj")

    def sparse_mlp(self, prefix, x, sink):
        scores = sigmoid(self.tensor(prefix + "mlp.gate.weight") @ x)
        choice = scores + self.tensor(
            prefix + "mlp.gate.e_score_correction_bias")
        order = np.argsort(-choice, kind="stable")
        selected = np.sort(order[:self.topk])
        picked = scores[selected]
        if self.norm_topk:
            weights_out = picked / (picked.sum() + 1e-20) \
                * np.float32(self.scaling)
        else:
            weights_out = picked * np.float32(self.scaling)
        routed = np.zeros_like(x)
        for i in range(self.topk):
            expert = prefix + f"mlp.experts.{selected[i]}."
            output = self.routed_expert_mlp(expert, x)
            routed = bf16_round_f32(
                routed + bf16_round_f32(output * weights_out[i]))
        shared = self.spine_mlp(prefix + "mlp.shared_experts.", x)
        sink.append(selected.astype(np.int32))
        sink.append(weights_out.astype(np.float32))
        return bf16_round_f32(routed + shared)

    def forward_layer(self, index, streams, states, caches, sink):
        prefix = PREFIX + str(index) + "."
        collapsed, post, comb = self.hc_site(prefix, streams, "hc_attn")
        x = bf16_round_f32(rmsnorm(
            collapsed, self.tensor(prefix + "input_layernorm.weight"),
            self.eps))
        if self.layer_types[index] == LINEAR_ATTENTION:
            attention = self.kda_attention(prefix, x, states[index])
        elif self.layer_types[index] == SPARSE_ATTENTION:
            if len(caches[index]) >= self.index_topk:
                raise ValueError(
                    "context exceeds indexer topk; reference undefined")
            attention = self.dsa_attention(prefix, x, caches[index])
        else:
            raise ValueError(f"unsupported layer type "
                             f"{self.layer_types[index]}")
        streams = self.hc_post(streams, attention, post, comb)
        collapsed, post, comb = self.hc_site(prefix, streams, "hc_ffn")
        x = bf16_round_f32(rmsnorm(
            collapsed, self.tensor(prefix + "post_attention_layernorm.weight"),
            self.eps))
        if self.mlp_types[index] == "dense":
            mlp = self.spine_mlp(prefix + "mlp.", x)
        elif self.mlp_types[index] == "sparse":
            mlp = self.sparse_mlp(prefix, x, sink)
        else:
            raise ValueError(f"unsupported mlp type {self.mlp_types[index]}")
        return self.hc_post(streams, mlp, post, comb)

    def decode_step(self, token_id, position, states, caches, capture,
                    capture_streams=None):
        if token_id < 0 or token_id >= self.vocab:
            raise ValueError(f"token {token_id} outside vocabulary "
                             f"{self.vocab}")
        if position == 0:
            for i in range(self.layers):
                if self.layer_types[i] == LINEAR_ATTENTION:
                    states[i] = {
                        "state": np.zeros((self.kheads, self.kd, self.kd),
                                          dtype=np.float32),
                        "wq": np.zeros((self.kheads * self.kd, self.conv),
                                       dtype=np.uint16),
                        "wk": np.zeros((self.kheads * self.kd, self.conv),
                                       dtype=np.uint16),
                        "wv": np.zeros((self.kheads * self.kd, self.conv),
                                       dtype=np.uint16)}
                else:
                    caches[i] = []
        embedding = self.embed(token_id)
        streams = np.tile(embedding, (self.hc, 1))
        for i in range(self.layers):
            sink = []
            streams = self.forward_layer(i, streams, states, caches, sink)
            if sink:
                capture[(position, i)] = sink
            if capture_streams is not None:
                capture_streams(i, streams)
        return streams

    def logits(self, streams, chunk=4096):
        collapsed = bf16_round_f32(streams.mean(axis=0))
        norm = bf16_round_f32(rmsnorm(
            collapsed, self.tensor("model.language_model.norm.weight"),
            self.eps))
        lm = self.st.entry("lm_head.weight")
        if lm["dtype"] != "BF16":
            raise ValueError("reference lm_head must be BF16")
        best = -np.inf
        best_token = -1
        for start in range(0, lm["shape"][0], chunk):
            rows = self.st.raw_rows("lm_head.weight", start,
                                    min(chunk, lm["shape"][0] - start))
            scores = bf16_to_f32(rows) @ norm
            i = int(np.argmax(scores))
            if float(scores[i]) > best:
                best = float(scores[i])
                best_token = start + i
        return best_token, best


ENGINE_CLASS = Glm53FlashEngine
