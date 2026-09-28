from concurrent.futures import ThreadPoolExecutor

import numpy as np

from t1_reference_common import (_E2M1_LUT, _E4M3_LUT, Safetensors,
                                 bf16_round_f32, bf16_to_f32, define_float,
                                 define_uint, rmsnorm, sigmoid)

EMBED_NAME = "model.embed_tokens.weight"
FINAL_NORM_NAME = "model.norm.weight"
LM_HEAD_NAME = "lm_head.weight"
FP8_BLOCK = 128
MXFP4_BLOCK = 32
E4M3_MAX_CODE = 0x7E
EXPERT_THREADS = 8


class Mimo26ConfigError(ValueError):
    pass


def _pair_lut():
    table = np.empty((256, 2), dtype=np.float32)
    codes = np.arange(256)
    table[:, 0] = _E2M1_LUT[codes & 0xF]
    table[:, 1] = _E2M1_LUT[codes >> 4]
    return table


PAIR_LUT = _pair_lut()


E8M0_LUT = np.append(np.ldexp(np.float64(1.0), np.arange(255) - 127), np.nan).astype(np.float32)


def e8m0_scale(codes):
    return E8M0_LUT[codes]


def mxfp4_matvec(payload, scales, x):
    rows, pair_cols = payload.shape
    groups = scales.shape[1]
    if pair_cols * 2 != x.shape[0] or groups * MXFP4_BLOCK != x.shape[0]:
        raise Mimo26ConfigError(f"mxfp4 operand {payload.shape}/{scales.shape} "
                                f"disagrees with input {x.shape[0]}")
    pairs = x.reshape(pair_cols, 2)
    table = (pairs[:, None, 0] * PAIR_LUT[None, :, 0]
             + pairs[:, None, 1] * PAIR_LUT[None, :, 1]).reshape(-1)
    index = np.add((np.arange(pair_cols, dtype=np.int32) * 256)[None, :], payload, dtype=np.int32)
    partial = np.take(table, index).reshape(rows, groups, MXFP4_BLOCK // 2)
    return (partial.sum(axis=2, dtype=np.float32) * e8m0_scale(scales)).sum(
        axis=1, dtype=np.float32)


def fp8_dequant_rows(payload, scale_grid, block_of_row):
    rows, cols = payload.shape
    if scale_grid.shape[1] * FP8_BLOCK != cols:
        raise Mimo26ConfigError(f"fp8 grid {scale_grid.shape} does not tile "
                                f"{cols} columns")
    out = _E4M3_LUT[payload].astype(np.float32)
    col_scale = np.repeat(scale_grid, FP8_BLOCK, axis=1)
    out *= col_scale[block_of_row]
    return out


def check_block_amax(payload, block_of_row, name):
    blocks = int(block_of_row.max()) + 1
    cols = payload.shape[1] // FP8_BLOCK
    peak = np.zeros((blocks, cols), dtype=np.uint8)
    magnitude = (payload & 0x7F).reshape(payload.shape[0], cols, FP8_BLOCK).max(axis=2)
    np.maximum.at(peak, block_of_row, magnitude)
    missing = int((peak != E4M3_MAX_CODE).sum())
    if missing:
        raise Mimo26ConfigError(
            f"{name}: {missing} of {peak.size} fp8 blocks lack an e4m3 max code "
            "under the declared row-to-block map; the qkv source layout is not "
            "the declared rank interleave")


def define_number(defines, name):
    try:
        return float(define_uint(defines, name))
    except ValueError:
        return define_float(defines, name)


def cross_check(defines, config):
    checks = [
        ("HIDDEN_DIMENSION", config["hidden_size"]),
        ("LAYER_COUNT", config["num_hidden_layers"]),
        ("OUTPUT_VOCAB_COUNT", config["vocab_size"]),
        ("MAXIMUM_CONTEXT_TOKENS", config["max_position_embeddings"]),
        ("RMS_NORM_EPSILON", config["layernorm_epsilon"]),
        ("END_OF_TEXT_TOKEN_ID", config["eos_token_id"]),
        ("ATTENTION_HEAD_COUNT", config["num_attention_heads"]),
        ("SWA_ATTENTION_HEAD_COUNT", config["swa_num_attention_heads"]),
        ("HEAD_DIMENSION", config["head_dim"]),
        ("SWA_HEAD_DIMENSION", config["swa_head_dim"]),
        ("VALUE_HEAD_DIMENSION", config["v_head_dim"]),
        ("SWA_VALUE_HEAD_DIMENSION", config["swa_v_head_dim"]),
        ("FULL_KV_HEAD_COUNT", config["num_key_value_heads"]),
        ("SWA_KV_HEAD_COUNT", config["swa_num_key_value_heads"]),
        ("ROPE_DIMENSION", int(config["head_dim"] * config["partial_rotary_factor"])),
        ("FULL_ROPE_THETA", config["rope_theta"]),
        ("SWA_ROPE_THETA", config["swa_rope_theta"]),
        ("SLIDING_WINDOW_TOKENS", config["sliding_window"]),
        ("ATTENTION_VALUE_SCALE", config["attention_value_scale"]),
        ("FULL_SINK_BIAS", int(bool(config["add_full_attention_sink_bias"]))),
        ("SWA_SINK_BIAS", int(bool(config["add_swa_attention_sink_bias"]))),
        ("DENSE_INTERMEDIATE_DIMENSION", config["intermediate_size"]),
        ("EXPERT_INTERMEDIATE_DIMENSION", config["moe_intermediate_size"]),
        ("ROUTED_EXPERT_COUNT", config["n_routed_experts"]),
        ("EXPERTS_PER_TOKEN", config["num_experts_per_tok"]),
        ("ROUTER_GROUP_COUNT", config["n_group"]),
        ("ROUTER_TOP_GROUP_COUNT", config["topk_group"]),
        ("ROUTER_NORM_TOPK", int(bool(config["norm_topk_prob"]))),
    ]
    for name, got in checks:
        if isinstance(got, list):
            raise Mimo26ConfigError(f"config value for SPARK_LLM_{name} is a list")
        want = define_number(defines, name)
        if want != float(got):
            raise Mimo26ConfigError(f"SPARK_LLM_{name}={want} disagrees with config value {got}")
    scaling = config.get("routed_scaling_factor")
    want_scaling = define_float(defines, "ROUTED_SCALING_FACTOR")
    if (1.0 if scaling is None else float(scaling)) != want_scaling:
        raise Mimo26ConfigError(f"SPARK_LLM_ROUTED_SCALING_FACTOR={want_scaling} "
                                f"disagrees with config routed_scaling_factor={scaling}")
    for key, value in (("scoring_func", "sigmoid"), ("topk_method", "noaux_tc"),
                       ("hidden_act", "silu"), ("attention_projection_layout", "fused_qkv"),
                       ("tie_word_embeddings", False), ("attention_bias", False)):
        if config.get(key) != value:
            raise Mimo26ConfigError(f"config {key}={config.get(key)!r}; the reference implements {value!r}")
    quant = config.get("quantization_config") or {}
    if quant.get("fmt") != "e4m3" or quant.get("weight_block_size") != [FP8_BLOCK, FP8_BLOCK] \
            or quant.get("mxfp4_block_size") != MXFP4_BLOCK:
        raise Mimo26ConfigError(f"quantization_config {quant} is not fp8 e4m3 [128,128] + mxfp4 g32")
    full_mask = define_uint(defines, "FULL_ATTENTION_LAYER_MASK")
    pattern = config["hybrid_layer_pattern"]
    moe = config["moe_layer_freq"]
    if len(pattern) != config["num_hidden_layers"] or len(moe) != len(pattern):
        raise Mimo26ConfigError("hybrid_layer_pattern / moe_layer_freq length disagrees with the layer count")
    for layer, kind in enumerate(pattern):
        if (kind == 0) != bool((full_mask >> layer) & 1):
            raise Mimo26ConfigError(f"layer {layer}: config kind {kind} disagrees with SPARK_LLM_FULL_ATTENTION_LAYER_MASK")
    first_routed = define_uint(defines, "FIRST_ROUTED_LAYER")
    for layer, flag in enumerate(moe):
        if bool(flag) != (layer >= first_routed):
            raise Mimo26ConfigError(f"layer {layer}: moe_layer_freq {flag} disagrees with SPARK_LLM_FIRST_ROUTED_LAYER")
    return []


class Mimo26Engine:
    def __init__(self, checkpoint_dir, defines, config):
        self.mismatches = cross_check(defines, config)
        self.st = Safetensors(checkpoint_dir)
        self.hidden = int(config["hidden_size"])
        self.layers = int(config["num_hidden_layers"])
        self.vocab = int(config["vocab_size"])
        self.eps = float(config["layernorm_epsilon"])
        self.heads = int(config["num_attention_heads"])
        self.head_dim = int(config["head_dim"])
        self.v_dim = int(config["v_head_dim"])
        self.rope_dim = define_uint(defines, "ROPE_DIMENSION")
        self.kv_heads = {0: int(config["num_key_value_heads"]), 1: int(config["swa_num_key_value_heads"])}
        self.theta = {0: float(config["rope_theta"]), 1: float(config["swa_rope_theta"])}
        self.window = int(config["sliding_window"])
        self.value_scale = float(config["attention_value_scale"])
        self.sink = {0: bool(config["add_full_attention_sink_bias"]), 1: bool(config["add_swa_attention_sink_bias"])}
        self.pattern = [int(v) for v in config["hybrid_layer_pattern"]]
        self.first_routed = define_uint(defines, "FIRST_ROUTED_LAYER")
        self.experts = int(config["n_routed_experts"])
        self.top_k = int(config["num_experts_per_tok"])
        self.expert_inter = int(config["moe_intermediate_size"])
        self.dense_inter = int(config["intermediate_size"])
        self.scaling = 1.0 if config.get("routed_scaling_factor") is None else float(config["routed_scaling_factor"])
        self.router_eps = define_float(defines, "ROUTER_NORM_EPSILON")
        self.interleave = define_uint(defines, "QKV_SOURCE_INTERLEAVE_RANKS")
        self.max_context = int(config["max_position_embeddings"])
        self.eot = int(config["eos_token_id"])
        self.pool = ThreadPoolExecutor(max_workers=EXPERT_THREADS)
        half = self.rope_dim // 2
        self.inv_freq = {kind: (1.0 / (np.float32(theta) ** (np.arange(0, self.rope_dim, 2, dtype=np.float32) / np.float32(self.rope_dim)))).astype(np.float32)
                         for kind, theta in self.theta.items()}
        if half * 2 != self.rope_dim or self.rope_dim > self.head_dim:
            raise Mimo26ConfigError(f"rope dimension {self.rope_dim} invalid for head {self.head_dim}")
        self._spine = {}
        self._lm_head = None
        self._check_vocab_tensors()

    def _shape(self, name):
        entry = self.st.entry(name)
        return tuple(entry["shape"]), entry["dtype"]

    def _expect(self, name, shape, dtype):
        got, got_dtype = self._shape(name)
        if got != tuple(shape) or got_dtype != dtype:
            raise Mimo26ConfigError(f"tensor {name} is {got_dtype}{list(got)}, expected {dtype}{list(shape)}")

    def _check_vocab_tensors(self):
        self._expect(EMBED_NAME, (self.vocab, self.hidden), "BF16")
        self._expect(LM_HEAD_NAME, (self.vocab, self.hidden), "BF16")
        self._expect(FINAL_NORM_NAME, (self.hidden,), "BF16")

    def vector(self, name):
        raw = self.st.raw(name)
        if raw.dtype == np.uint16:
            return bf16_to_f32(raw).reshape(-1)
        if raw.dtype == np.float32:
            return raw.reshape(-1)
        raise Mimo26ConfigError(f"tensor {name} dtype {raw.dtype} is not BF16/F32")

    def qkv_layout(self, kind):
        kv = self.kv_heads[kind]
        ranks = self.interleave
        if self.heads % ranks or kv % ranks:
            raise Mimo26ConfigError(f"interleave {ranks} does not divide heads {self.heads}/{kv}")
        q_local = self.heads // ranks * self.head_dim
        k_local = kv // ranks * self.head_dim
        v_local = kv // ranks * self.v_dim
        per_rank = q_local + k_local + v_local
        order = []
        for rank in range(ranks):
            order.append(np.arange(rank * per_rank, rank * per_rank + q_local))
        for rank in range(ranks):
            base = rank * per_rank + q_local
            order.append(np.arange(base, base + k_local))
        for rank in range(ranks):
            base = rank * per_rank + q_local + k_local
            order.append(np.arange(base, base + v_local))
        order = np.concatenate(order)
        blocks_per_rank = (per_rank + FP8_BLOCK - 1) // FP8_BLOCK
        rows = np.arange(per_rank * ranks)
        block_of_row = (rows // per_rank) * blocks_per_rank + (rows % per_rank) // FP8_BLOCK
        return order, block_of_row, per_rank * ranks, blocks_per_rank * ranks

    def _qkv(self, layer):
        key = ("qkv", layer)
        if key in self._spine:
            return self._spine[key]
        kind = self.pattern[layer]
        order, block_of_row, rows, grid_rows = self.qkv_layout(kind)
        name = f"model.layers.{layer}.self_attn.qkv_proj.weight"
        self._expect(name, (rows, self.hidden), "F8_E4M3")
        self._expect(name + "_scale_inv", (grid_rows, self.hidden // FP8_BLOCK), "F32")
        payload = self.st.pread(name)
        check_block_amax(payload, block_of_row, name)
        weight = fp8_dequant_rows(payload, self.st.pread(name + "_scale_inv"), block_of_row)[order]
        self._spine[key] = weight
        return weight

    def _dense(self, layer, proj, rows, cols):
        key = (proj, layer)
        if key in self._spine:
            return self._spine[key]
        name = f"model.layers.{layer}.mlp.{proj}.weight"
        self._expect(name, (rows, cols), "F8_E4M3")
        self._expect(name + "_scale_inv", (rows // FP8_BLOCK, cols // FP8_BLOCK), "F32")
        block_of_row = np.arange(rows) // FP8_BLOCK
        weight = fp8_dequant_rows(self.st.pread(name), self.st.pread(name + "_scale_inv"), block_of_row)
        self._spine[key] = weight
        return weight

    def _bf16_matrix(self, name, shape):
        key = ("bf16", name)
        if key not in self._spine:
            self._expect(name, shape, "BF16")
            self._spine[key] = self.st.pread(name)
        return self._spine[key]

    def _bf16_matvec(self, name, shape, x):
        raw = self._bf16_matrix(name, shape)
        out = np.empty(shape[0], dtype=np.float32)
        step = 1024
        for first in range(0, shape[0], step):
            out[first:first + step] = bf16_to_f32(raw[first:first + step]) @ x
        return out

    def embed(self, token_id):
        if token_id < 0 or token_id >= self.vocab:
            raise ValueError(f"token {token_id} outside vocabulary {self.vocab}")
        return bf16_to_f32(self.st.raw_rows(EMBED_NAME, token_id, 1)[0])

    def rope(self, rows, position, kind):
        freqs = np.float32(position) * self.inv_freq[kind]
        cos = np.cos(np.concatenate([freqs, freqs])).astype(np.float32)
        sin = np.sin(np.concatenate([freqs, freqs])).astype(np.float32)
        half = self.rope_dim // 2
        part = rows[:, :self.rope_dim]
        rotated = np.concatenate([-part[:, half:], part[:, :half]], axis=1)
        out = rows.copy()
        out[:, :self.rope_dim] = part * cos + rotated * sin
        return out

    def attention(self, layer, x, cache, position):
        kind = self.pattern[layer]
        kv = self.kv_heads[kind]
        q_rows = self.heads * self.head_dim
        k_rows = kv * self.head_dim
        fused = bf16_round_f32(self._qkv(layer) @ x)
        q = fused[:q_rows].reshape(self.heads, self.head_dim)
        k = fused[q_rows:q_rows + k_rows].reshape(kv, self.head_dim)
        v = bf16_round_f32(fused[q_rows + k_rows:].reshape(kv, self.v_dim) * np.float32(self.value_scale))
        q = bf16_round_f32(self.rope(q, position, kind))
        k = bf16_round_f32(self.rope(k, position, kind))
        cache.append((position, k, v))
        visible = [entry for entry in cache if kind == 0 or entry[0] > position - self.window]
        keys = np.stack([entry[1] for entry in visible], axis=1)
        values = np.stack([entry[2] for entry in visible], axis=1)
        group = self.heads // kv
        scores = np.einsum("kgd,ktd->kgt", q.reshape(kv, group, self.head_dim), keys).reshape(self.heads, -1)
        scores *= np.float32(self.head_dim ** -0.5)
        if self.sink[kind]:
            sink = self.vector(f"model.layers.{layer}.self_attn.attention_sink_bias")
            if sink.shape[0] != self.heads:
                raise Mimo26ConfigError(f"layer {layer} sink has {sink.shape[0]} heads")
            row_max = np.maximum(scores.max(axis=1), sink)
            weights = np.exp(scores - row_max[:, None])
            total = weights.sum(axis=1) + np.exp(sink - row_max)
        else:
            weights = np.exp(scores - scores.max(axis=1)[:, None])
            total = weights.sum(axis=1)
        weights = (weights / total[:, None]).astype(np.float32)
        out = np.einsum("kgt,ktd->kgd", weights.reshape(kv, group, -1), values).reshape(-1)
        attended = bf16_round_f32(out)
        return bf16_round_f32(self._bf16_matvec(f"model.layers.{layer}.self_attn.o_proj.weight",
                                                (self.hidden, self.heads * self.v_dim), attended))

    def dense_mlp(self, layer, x):
        gate = bf16_round_f32(self._dense(layer, "gate_proj", self.dense_inter, self.hidden) @ x)
        up = bf16_round_f32(self._dense(layer, "up_proj", self.dense_inter, self.hidden) @ x)
        act = bf16_round_f32(bf16_round_f32(gate * sigmoid(gate)) * up)
        return bf16_round_f32(self._dense(layer, "down_proj", self.hidden, self.dense_inter) @ act)

    def route(self, layer, x):
        prefix = f"model.layers.{layer}.mlp.gate."
        router = self._bf16_matrix(prefix + "weight", (self.experts, self.hidden))
        logits = bf16_to_f32(router) @ x
        scores = sigmoid(logits.astype(np.float32)).astype(np.float32)
        bias = self.vector(prefix + "e_score_correction_bias")
        choice = scores + bias
        ids = np.argsort(-choice, kind="stable")[:self.top_k].astype(np.int32)
        weights = scores[ids]
        weights = (weights / (weights.sum(dtype=np.float32) + np.float32(self.router_eps))).astype(np.float32)
        return ids, (weights * np.float32(self.scaling)).astype(np.float32)

    def check_expert(self, layer, expert):
        prefix = f"model.layers.{layer}.mlp.experts.{expert}."
        for proj, rows, cols in (("gate_proj", self.expert_inter, self.hidden),
                                 ("up_proj", self.expert_inter, self.hidden),
                                 ("down_proj", self.hidden, self.expert_inter)):
            self._expect(prefix + proj + ".weight", (rows, cols // 2), "U8")
            self._expect(prefix + proj + ".weight_scale", (rows, cols // MXFP4_BLOCK), "U8")

    def expert(self, layer, expert, x):
        prefix = f"model.layers.{layer}.mlp.experts.{expert}."
        gate = bf16_round_f32(mxfp4_matvec(self.st.pread(prefix + "gate_proj.weight"),
                                           self.st.pread(prefix + "gate_proj.weight_scale"), x))
        up = bf16_round_f32(mxfp4_matvec(self.st.pread(prefix + "up_proj.weight"),
                                         self.st.pread(prefix + "up_proj.weight_scale"), x))
        act = bf16_round_f32(bf16_round_f32(gate * sigmoid(gate)) * up)
        return bf16_round_f32(mxfp4_matvec(self.st.pread(prefix + "down_proj.weight"),
                                           self.st.pread(prefix + "down_proj.weight_scale"), act))

    def moe(self, layer, x, position, capture):
        ids, weights = self.route(layer, x)
        capture[(position, layer)] = (ids.copy(), weights.copy())
        for expert in ids:
            self.check_expert(layer, int(expert))
        outputs = list(self.pool.map(lambda e: self.expert(layer, int(e), x), ids))
        total = np.zeros(self.hidden, dtype=np.float32)
        for weight, out in zip(weights, outputs):
            total += weight * out
        return bf16_round_f32(total)

    def forward_layer(self, layer, streams, caches, position, capture):
        prefix = f"model.layers.{layer}."
        x = bf16_round_f32(rmsnorm(streams, self.vector(prefix + "input_layernorm.weight"), self.eps))
        streams = bf16_round_f32(streams + self.attention(layer, x, caches[layer], position))
        x = bf16_round_f32(rmsnorm(streams, self.vector(prefix + "post_attention_layernorm.weight"), self.eps))
        if layer >= self.first_routed:
            mlp = self.moe(layer, x, position, capture)
        else:
            mlp = self.dense_mlp(layer, x)
        return bf16_round_f32(streams + mlp)

    def decode_step(self, token_id, position, states, caches, capture, capture_streams=None):
        del states
        if position < 0 or position >= self.max_context:
            raise ValueError(f"position {position} outside modeled context {self.max_context}")
        if position == 0:
            for layer in range(self.layers):
                caches[layer] = []
        streams = self.embed(token_id)
        for layer in range(self.layers):
            streams = self.forward_layer(layer, streams, caches, position, capture)
            if capture_streams is not None:
                capture_streams(layer, streams)
            if not np.isfinite(streams).all():
                raise ValueError(f"nonfinite reference state at layer {layer}")
        return streams

    def logits(self, streams):
        norm = bf16_round_f32(rmsnorm(streams, self.vector(FINAL_NORM_NAME), self.eps))
        scores = self._bf16_matvec(LM_HEAD_NAME, (self.vocab, self.hidden), norm)
        token = int(np.argmax(scores))
        return token, float(scores[token])


ENGINE_CLASS = Mimo26Engine
