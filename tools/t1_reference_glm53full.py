import numpy as np

from t1_reference_common import (Safetensors, _E4M3_LUT, bf16_round_f32,
                                 bf16_to_f32, define_float, define_uint,
                                 f32_to_bf16_u16, fp8_block_to_bf16, rmsnorm,
                                 sigmoid)

PREFIX = "model.layers."
HIDDEN = 6144
LAYERS = 78
HEADS = 64
LATENT = 512
ROPE_DIM = 64
QK_NOPE = 192
V_DIM = 256
LATENT_ROW = LATENT + ROPE_DIM
QUERY_A_DIM = 2048
QK_SCALE = 0.0625
ROPE_THETA = 8000000.0
EPSILON = 1e-5
FIRST_ROUTED = 3
EXPERTS = 256
TOP_K = 8
ROUTED_SCALE = 2.5
INTERMEDIATE = 2048
DENSE_INTERMEDIATE = 12288
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
    ("DENSE_INTERMEDIATE_DIMENSION", "intermediate_size", "uint"),
    ("MOE_INTERMEDIATE_DIMENSION", "moe_intermediate_size", "uint"),
    ("MLA_HEAD_COUNT", "num_attention_heads", "uint"),
    ("MLA_QUERY_A_DIMENSION", "q_lora_rank", "uint"),
    ("MLA_LATENT_DIMENSION", "kv_lora_rank", "uint"),
    ("MLA_QK_NOPE_HEAD_DIMENSION", "qk_nope_head_dim", "uint"),
    ("MLA_QK_ROPE_HEAD_DIMENSION", "qk_rope_head_dim", "uint"),
    ("MLA_V_HEAD_DIMENSION", "v_head_dim", "uint"),
    ("DSA_SELECTED_TOKEN_COUNT", "index_topk", "uint"),
    ("DSA_INDEX_HEAD_COUNT", "index_n_heads", "uint"),
    ("DSA_INDEX_HEAD_DIMENSION", "index_head_dim", "uint"),
    ("DSA_INDEX_SHARE_GROUP_LAYER_COUNT", "index_topk_freq", "uint"),
    ("DSA_INDEX_SKIP_TOPK_OFFSET", "index_skip_topk_offset", "uint"),
    ("END_OF_TEXT_TOKEN_ID", "eos_token_id", "uint"),
]
DEFINES_RECORDED_ONLY = []


class Glm53FullConfigError(ValueError):
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
            raise Glm53FullConfigError(
                f"config key {cname} missing for SPARK_LLM_{dname}")
        want = define_of(kind, defines, dname)
        got = scalar_of(kind, config[cname])
        if abs(want - got) > 0:
            mismatches.append({"define": f"SPARK_LLM_{dname}", "value": want,
                               "config_key": cname, "config_value": got})
    for dname, _, _ in DEFINES_VS_CONFIG:
        for row in mismatches:
            if row["define"] == f"SPARK_LLM_{dname}":
                raise Glm53FullConfigError(
                    f"{row['define']}={row['value']} disagrees with "
                    f"{row['config_key']}={row['config_value']}")
    return mismatches


def rope_apply(values, offset, position):
    half = ROPE_DIM // 2
    for index in range(half):
        angle = float(position) * np.float32(np.float32(ROPE_THETA) **
                                             np.float32(-2.0 * index / ROPE_DIM))
        c = np.float32(np.cos(np.float32(angle)))
        s = np.float32(np.sin(np.float32(angle)))
        low_at = offset + index * 2
        high_at = low_at + 1
        low = bf16_to_f32(values[low_at:low_at + 1])[0]
        high = bf16_to_f32(values[high_at:high_at + 1])[0]
        values[low_at] = f32_to_bf16_u16(
            np.array([low * c - high * s]))[0]
        values[high_at] = f32_to_bf16_u16(
            np.array([low * s + high * c]))[0]


class Glm53FullEngine:
    def __init__(self, checkpoint_dir, defines, config):
        self.mismatches = cross_check(defines, config)
        self.st = Safetensors(checkpoint_dir)
        self.layers = int(config["num_hidden_layers"])
        self.vocab = int(config["vocab_size"])
        self.eot = int(config["eos_token_id"][0] if isinstance(
            config["eos_token_id"], list) else config["eos_token_id"])
        entry = self.st.entry(f"{PREFIX}{FIRST_ROUTED}.mlp.experts.0.up_proj.weight")
        if entry["dtype"] != "F8_E4M3" or tuple(entry["shape"]) != \
                (INTERMEDIATE, HIDDEN):
            raise Glm53FullConfigError(
                f"expert payload {entry['dtype']} {entry['shape']} is not the "
                f"blockwise-FP8 release shape ({INTERMEDIATE}, {HIDDEN})")

    def tensor(self, name):
        raw = self.st.raw(name)
        if raw.dtype == np.uint16:
            return bf16_to_f32(raw)
        if raw.dtype == np.uint8:
            scale = self.st.raw(name + "_scale_inv").astype(np.float32)
            rows, cols = raw.shape
            return bf16_to_f32(fp8_block_to_bf16(raw, scale, rows, cols))
        return raw.astype(np.float32)

    def vector(self, name):
        raw = self.st.raw(name)
        if raw.dtype == np.uint16:
            return bf16_to_f32(raw.reshape(-1))
        if raw.dtype == np.uint8:
            cols = raw.shape[-1]
            scale = self.st.raw(name + "_scale_inv").astype(np.float32)
            return bf16_to_f32(fp8_block_to_bf16(
                raw.reshape(1, cols), scale.reshape(1, (cols + 127) // 128),
                1, cols).reshape(-1))
        return raw.astype(np.float32).reshape(-1)

    def expert_weight(self, name):
        raw = self.st.raw(name)
        if raw.dtype != np.uint8:
            raise ValueError(f"expert {name} is not blockwise F8_E4M3")
        scale = self.st.raw(name + "_scale_inv").astype(np.float32)
        rows, cols = raw.shape
        codes = _E4M3_LUT[raw.reshape(rows, cols)].astype(np.float32)
        tiled = np.repeat(np.repeat(scale, 128, axis=0), 128, axis=1)
        return codes * tiled[:rows, :cols]

    def linear(self, x, name):
        return f32_to_bf16_u16(self.tensor(name + ".weight") @ x)

    def linear_fused_gate_up(self, x, up_name, gate_name):
        up = self.tensor(up_name) @ x
        gate = self.tensor(gate_name) @ x
        return np.concatenate(
            [f32_to_bf16_u16(up), f32_to_bf16_u16(gate)])

    def embed(self, token_id):
        if token_id < 0 or token_id >= self.vocab:
            raise ValueError(f"token {token_id} outside vocabulary {self.vocab}")
        raw = self.st.raw_rows("model.embed_tokens.weight", token_id, 1)
        if raw.dtype != np.uint16:
            raise ValueError("reference embedding must be BF16")
        return raw[0]

    def fused_residual_norm(self, value, weight_name):
        normed = bf16_round_f32(rmsnorm(value, self.vector(weight_name),
                                        EPSILON))
        return f32_to_bf16_u16(value), normed

    def attention(self, layer, normed, position, cache):
        prefix = f"{PREFIX}{layer}.self_attn."
        q_a = self.linear(normed, prefix + "q_a_proj")
        q_a_normed = bf16_round_f32(rmsnorm(
            bf16_to_f32(q_a), self.vector(prefix + "q_a_layernorm.weight"),
            EPSILON))
        q = self.linear(q_a_normed, prefix + "q_b_proj")
        kv_slot = self.linear(normed, prefix + "kv_a_proj_with_mqa")
        kv_normed = bf16_round_f32(rmsnorm(
            bf16_to_f32(kv_slot[:LATENT]),
            self.vector(prefix + "kv_a_layernorm.weight"), EPSILON))
        kv_slot[:LATENT] = f32_to_bf16_u16(kv_normed)
        rope_apply(kv_slot, LATENT, position)
        q_heads = q.reshape(HEADS, QK_NOPE + ROPE_DIM)
        q_rope = np.empty((HEADS, ROPE_DIM), dtype=np.uint16)
        for head in range(HEADS):
            rope_apply(q_heads[head], QK_NOPE, position)
            q_rope[head] = q_heads[head][QK_NOPE:]
        kv_b = self.tensor(prefix + "kv_b_proj.weight")
        per_head = QK_NOPE + V_DIM
        query_latent = np.empty((HEADS, LATENT), dtype=np.uint16)
        for head in range(HEADS):
            block = kv_b[head * per_head:head * per_head + QK_NOPE, :]
            query_latent[head] = f32_to_bf16_u16(
                bf16_to_f32(q_heads[head][:QK_NOPE]) @ block)
        cache.append(kv_slot.copy())
        slots = bf16_to_f32(np.stack(cache))
        attention_latent = np.empty((HEADS, LATENT), dtype=np.uint16)
        for head in range(HEADS):
            qcat = np.concatenate(
                [bf16_to_f32(query_latent[head]), bf16_to_f32(q_rope[head])])
            scores = (slots @ qcat) * np.float32(QK_SCALE)
            weights = np.exp(scores - scores.max())
            weights = weights / weights.sum()
            mixed = slots[:, :LATENT].T @ weights
            attention_latent[head] = f32_to_bf16_u16(mixed)
        attention_value = np.empty(HEADS * V_DIM, dtype=np.uint16)
        for head in range(HEADS):
            block = kv_b[head * per_head + QK_NOPE:(head + 1) * per_head, :]
            attention_value[head * V_DIM:(head + 1) * V_DIM] = \
                f32_to_bf16_u16(block @ bf16_to_f32(attention_latent[head]))
        return self.linear(attention_value, prefix + "o_proj")

    def silu_mul(self, fused):
        up = bf16_to_f32(fused[:fused.shape[0] // 2])
        gate = bf16_to_f32(fused[fused.shape[0] // 2:])
        return f32_to_bf16_u16((gate / (1.0 + np.exp(-gate))) * up)

    def down_linear(self, x, name):
        return f32_to_bf16_u16(self.tensor(name) @ x)

    def dense_mlp(self, layer, normed):
        prefix = f"{PREFIX}{layer}.mlp."
        fused = self.linear_fused_gate_up(
            normed, prefix + "up_proj.weight",
            prefix + "gate_proj.weight")
        return self.down_linear(self.silu_mul(fused), prefix + "down_proj.weight")

    def expert_mlp(self, layer, expert, normed_f32):
        base = f"{PREFIX}{layer}.mlp.experts.{expert}."
        up = f32_to_bf16_u16(self.expert_weight(base + "up_proj.weight") @
                             normed_f32)
        gate = f32_to_bf16_u16(self.expert_weight(base + "gate_proj.weight") @
                               normed_f32)
        intermediate = self.silu_mul(np.concatenate([up, gate]))
        return f32_to_bf16_u16(
            self.expert_weight(base + "down_proj.weight") @
            bf16_to_f32(intermediate))

    def routed_mlp(self, layer, normed, sink):
        prefix = f"{PREFIX}{layer}.mlp."
        normed_f32 = normed
        logits = self.tensor(prefix + "gate.weight") @ normed_f32
        scores = sigmoid(logits.astype(np.float32))
        choice = scores + self.vector(prefix + "gate.e_score_correction_bias")
        order = np.argsort(-choice, kind="stable")
        selected = order[:TOP_K]
        picked = scores[selected]
        weights = picked / (picked.sum() + 1e-20) * np.float32(ROUTED_SCALE)
        routed = np.zeros(HIDDEN, dtype=np.float32)
        for index in range(TOP_K):
            output = self.expert_mlp(layer, int(selected[index]), normed_f32)
            routed += bf16_to_f32(output) * weights[index]
        shared = prefix + "shared_experts."
        shared_fused = self.linear_fused_gate_up(
            normed_f32, shared + "up_proj.weight", shared + "gate_proj.weight")
        shared_out = self.down_linear(
            self.silu_mul(shared_fused), shared + "down_proj.weight")
        sink.append(selected.astype(np.int32))
        sink.append(weights.astype(np.float32))
        return f32_to_bf16_u16(routed + bf16_to_f32(shared_out))

    def forward_layer(self, index, residual, delta, position, cache, sink):
        prefix = f"{PREFIX}{index}."
        value = bf16_to_f32(delta) + bf16_to_f32(residual)
        residual, normed = self.fused_residual_norm(
            value, prefix + "input_layernorm.weight")
        attention = self.attention(index, normed, position, cache)
        value = bf16_to_f32(attention) + bf16_to_f32(residual)
        residual, normed = self.fused_residual_norm(
            value, prefix + "post_attention_layernorm.weight")
        if index < FIRST_ROUTED:
            return residual, self.dense_mlp(index, normed)
        return residual, self.routed_mlp(index, normed, sink)

    def decode_step(self, token_id, position, states, caches, capture):
        if token_id < 0 or token_id >= self.vocab:
            raise ValueError(f"token {token_id} outside vocabulary {self.vocab}")
        if position == 0:
            for i in range(self.layers):
                caches[i] = []
        residual = np.zeros(HIDDEN, dtype=np.uint16)
        delta = self.embed(token_id)
        for i in range(self.layers):
            sink = []
            residual, delta = self.forward_layer(i, residual, delta, position,
                                                 caches[i], sink)
            if sink:
                capture[(position, i)] = sink
        return bf16_to_f32(residual) + bf16_to_f32(delta)

    def logits(self, streams, chunk=8192):
        value = streams if streams.dtype == np.float32 \
            else bf16_to_f32(streams)
        normed = bf16_round_f32(rmsnorm(value, self.tensor("model.norm.weight"),
                                        EPSILON))
        entry = self.st.entry("lm_head.weight")
        rows_total = entry["shape"][0]
        if entry["dtype"] != "BF16":
            raise ValueError("reference lm_head must be BF16")
        best = -np.inf
        best_token = -1
        for start in range(0, rows_total, chunk):
            count = min(chunk, rows_total - start)
            rows = self.st.raw_rows("lm_head.weight", start, count)
            scores = bf16_to_f32(rows) @ normed
            i = int(np.argmax(scores))
            if float(scores[i]) > best:
                best = float(scores[i])
                best_token = start + i
        return best_token, best


ENGINE_CLASS = Glm53FullEngine
