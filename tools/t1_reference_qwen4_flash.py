import numpy as np

from t1_reference_common import (Safetensors, bf16_round_f32, bf16_to_f32,
                                 f32_to_bf16_u16, define_float, define_uint,
                                 sigmoid)

PREFIX = "model.language_model.layers."
HC_STREAMS = 4
HIDDEN = 2560
HC_WIDTH = HC_STREAMS * HIDDEN
GDN_QK = 2048
GDN_VALUE_DIM = 6144
GDN_HEADS = 48
GDN_HKV = 3
GDN_HD = 128
ATTN_HEADS = 24
ATTN_KV_HEADS = 2
ATTN_HEAD_DIM = 256
ROPE_DIM = 64
ROPE_THETA = 10000000.0
ATTN_PERIOD = 4
FULL_PHASE = 3
PLE_LAYER = 1
PLE_HEADS = 16
PLE_HEAD_DIM = 160
PLE_ROWS = 320001536
PLE_SHARDS = 128
EXPERTS = 512
TOPK = 10
EXPERT_DIM = 640
EOS_INVALID = 248044
EPS = 1e-6
INDEXER_ALL_VISIBLE_MAX_CONTEXT = 2040

_E4M3 = np.zeros(256, dtype=np.float32)
for _i in range(256):
    _s = -1.0 if _i & 0x80 else 1.0
    _e = (_i >> 3) & 0xF
    _m = _i & 0x7
    if _e == 0:
        _v = _m / 8.0 * 2.0 ** -6
    elif _e == 15 and _m == 7:
        _v = np.nan
    else:
        _v = (1.0 + _m / 8.0) * 2.0 ** (_e - 7)
    _E4M3[_i] = _s * _v


class Qwen4FlashConfigError(ValueError):
    pass


def swish(x):
    return x * sigmoid(x)


def softplus(x):
    return np.logaddexp(np.float32(0), x)


def cross_check(defines, config):
    checks = [
        ("HIDDEN_DIMENSION", "hidden_size", "uint"),
        ("LAYER_COUNT", "num_hidden_layers", "uint"),
        ("OUTPUT_VOCAB_COUNT", "vocab_size", "uint"),
        ("RMS_NORM_EPSILON", "rms_norm_eps", "float"),
        ("ATTN_HEAD_COUNT", "num_attention_heads", "uint"),
        ("KV_HEAD_COUNT", "num_key_value_heads", "uint"),
        ("HEAD_DIMENSION", "head_dim", "uint"),
        ("GDN_KEY_HEAD_COUNT", "linear_num_key_heads", "uint"),
        ("GDN_VALUE_HEAD_COUNT", "linear_num_value_heads", "uint"),
        ("GDN_HEAD_KEY_DIMENSION", "linear_key_head_dim", "uint"),
        ("ROUTED_EXPERT_COUNT", "num_experts", "uint"),
        ("EXPERTS_PER_TOKEN", "num_experts_per_tok", "uint"),
        ("ATTN_PERIOD", "full_attention_interval", "uint"),
        ("EOS_TOKEN_ID", "eos_token_id", "uint"),
    ]
    mismatches = []
    for dname, cname, kind in checks:
        if cname not in config:
            mismatches.append(f"config key {cname} missing for SPARK_LLM_{dname}")
            continue
        want = define_uint(defines, dname) if kind == "uint" \
            else define_float(defines, dname)
        got = config[cname]
        if isinstance(got, list):
            got = got[0]
        if abs(float(want) - float(got)) > 0:
            mismatches.append(
                f"SPARK_LLM_{dname}={want} disagrees with config {cname}={got}")
    return mismatches


def group_norm(streams, gain, eps=EPS):
    streams = streams.reshape(HC_STREAMS, HIDDEN)
    gain = gain.reshape(HC_STREAMS, HIDDEN)
    variance = (streams * streams).sum(axis=1) / HIDDEN
    out = streams / np.sqrt(variance + eps)[:, None] * gain
    return bf16_round_f32(out.reshape(-1))


class Qwen4FlashEngine:
    def __init__(self, checkpoint_dir, defines, config):
        self.mismatches = cross_check(defines, config)
        self.st = Safetensors(checkpoint_dir)
        self.layers = int(config["num_hidden_layers"])
        self.vocab = int(config["vocab_size"])
        self.eps = float(config["rms_norm_eps"])
        self.eot = int(config["eos_token_id"][0] if isinstance(
            config["eos_token_id"], list) else config["eos_token_id"])
        self.layer_types = list(config["layer_types"])
        if len(self.layer_types) != self.layers:
            raise Qwen4FlashConfigError("layer_types disagree with layer count")
        try:
            self.st.entry(PREFIX + "0.mlp.experts.0.gate_proj.weight")
            self.experts_fp8 = True
        except KeyError:
            self.experts_fp8 = False
        self.ple_multipliers = self.st.raw(
            PREFIX + f"{PLE_LAYER}.ple.ple_embedding.layer_multipliers")
        self.ple_vocabs = self.st.raw(
            PREFIX + f"{PLE_LAYER}.ple.ple_embedding.ngram_heads_vocab_sizes")
        self.ple_offsets = self.st.raw(
            PREFIX + f"{PLE_LAYER}.ple.ple_embedding.ngram_heads_offsets")
        if int(self.ple_vocabs.sum()) > PLE_ROWS:
            raise Qwen4FlashConfigError("ple head vocabs exceed the ngram table")
        self.ple_rows_per_shard = PLE_ROWS // PLE_SHARDS
        self.ple_history = [0, 0]
        self.ple_tail = None
        self._lm_head_u16 = None

    def tensor(self, name):
        raw = self.st.raw(name)
        if raw.dtype == np.uint16:
            return bf16_to_f32(raw)
        if raw.dtype == np.uint8:
            scale = self.st.raw(name + "_scale_inv")
            if scale.dtype == np.uint16:
                scale = bf16_to_f32(scale)
            scale = scale.astype(np.float32)
            rows, cols = raw.shape
            codes = _E4M3[raw.reshape(rows, cols)]
            expanded = np.repeat(np.repeat(scale, 128, axis=0), 128, axis=1)
            return codes * expanded[:rows, :cols]
        return raw.astype(np.float32)

    def linear(self, x, name):
        return bf16_round_f32(self.tensor(name + ".weight") @ x)

    def _raw_f32(self, name):
        raw = self.st.raw(name).reshape(-1)
        if raw.dtype == np.uint16:
            return bf16_to_f32(raw)
        return raw.astype(np.float32)

    @staticmethod
    def _wrap64(value):
        return np.int64((value + (1 << 63)) % (1 << 64) - (1 << 63))

    def embed(self, token_id):
        raw = self.st.raw_rows("model.language_model.embed_tokens.weight",
                               token_id, 1)
        if raw.dtype != np.uint16:
            raise ValueError("reference embedding must be BF16")
        return bf16_to_f32(raw[0])

    def hc_prep(self, hc_prefix, streams):
        normed = group_norm(streams, bf16_to_f32(self.st.raw(
            hc_prefix + "hc_norm.weight").reshape(-1)), self.eps)
        lowrank = self.linear(normed, hc_prefix + "input_mix_weight_down")
        lowrank = bf16_round_f32(swish(lowrank / np.float32(HC_STREAMS)))
        up = self.linear(lowrank, hc_prefix + "input_mix_weight_up")
        up_s = up.reshape(HC_STREAMS, HIDDEN)
        normed_s = normed.reshape(HC_STREAMS, HIDDEN)
        mixed = bf16_round_f32(
            (sigmoid(up_s) * normed_s).sum(axis=0) / np.float32(HC_STREAMS))
        inject = self.linear(normed, hc_prefix + "block_inject_weight")
        return mixed, inject

    def hc_inject(self, streams, inject_pre, delta):
        scale = np.float32(2.0) * sigmoid(
            inject_pre.reshape(HC_STREAMS, 1) / np.float32(HC_STREAMS))
        updated = streams.reshape(HC_STREAMS, HIDDEN) + scale * delta.reshape(1, HIDDEN)
        return bf16_round_f32(updated.reshape(-1))

    def rope(self, rows, position):
        half = ROPE_DIM // 2
        pairs = np.arange(half, dtype=np.float32)
        freq = np.exp2(-(2.0 * pairs / ROPE_DIM)
                       * np.log2(np.float32(ROPE_THETA)))
        angle = np.float32(position) * freq
        cos = np.cos(angle)
        sin = np.sin(angle)
        real = rows[:, 0:half].copy()
        imag = rows[:, half:ROPE_DIM].copy()
        rows[:, 0:half] = real * cos - imag * sin
        rows[:, half:ROPE_DIM] = imag * cos + real * sin
        return rows

    def head_rmsnorm(self, rows, gain, eps):
        variance = (rows * rows).sum(axis=1) / ATTN_HEAD_DIM
        return rows / np.sqrt(variance + eps)[:, None] * gain[None, :]

    def gdn_attention(self, layer, x, layer_state):
        p = PREFIX + f"{layer}.linear_attn."
        qkv = self.linear(x, p + "in_proj_qkv")
        z = self.linear(x, p + "in_proj_z")
        beta_pre = self.linear(x, p + "in_proj_b")
        decay_pre = self.linear(x, p + "in_proj_a")
        channels = 2 * GDN_QK + GDN_VALUE_DIM
        if layer_state is None:
            layer_state = {"tail": np.zeros((channels, 3), dtype=np.uint16),
                           "state": np.zeros((GDN_HEADS, GDN_HD, GDN_HD),
                                             dtype=np.float32)}
        taps = np.concatenate([layer_state["tail"],
                               f32_to_bf16_u16(qkv).reshape(-1, 1)], axis=1)
        weights = self.tensor(p + "conv1d.weight").reshape(channels, 4)
        acc = (bf16_to_f32(taps) * weights).sum(axis=1)
        conv_out = bf16_round_f32(swish(acc))
        layer_state["tail"] = taps[:, 1:]
        a_log = self._raw_f32(p + "A_log")
        dt_bias = self._raw_f32(p + "dt_bias")
        log_decay = -np.exp(a_log) * softplus(decay_pre + dt_bias)
        beta = sigmoid(beta_pre)
        out = np.empty(GDN_VALUE_DIM, dtype=np.float32)
        state = layer_state["state"]
        for h in range(GDN_HEADS):
            key_head = h // GDN_HKV
            q = conv_out[key_head * GDN_HD:(key_head + 1) * GDN_HD]
            k = conv_out[GDN_QK + key_head * GDN_HD:
                         GDN_QK + (key_head + 1) * GDN_HD]
            v = conv_out[2 * GDN_QK + h * GDN_HD:2 * GDN_QK + (h + 1) * GDN_HD]
            q = q / np.sqrt((q * q).sum() + EPS) / np.sqrt(np.float32(GDN_HD))
            k = k / np.sqrt((k * k).sum() + EPS)
            decay = np.exp(log_decay[h])
            decayed = state[h] * decay
            kv_memory = (decayed * k[:, None]).sum(axis=0)
            delta = (v - kv_memory) * beta[h]
            state[h] = decayed + k[:, None] * delta[None, :]
            out[h * GDN_HD:(h + 1) * GDN_HD] = (state[h] * q[:, None]).sum(axis=0)
        core = bf16_round_f32(out)
        norm_w = bf16_to_f32(self.st.raw(p + "norm.weight").reshape(-1))
        core_heads = core.reshape(GDN_HEADS, GDN_HD)
        z_heads = z.reshape(GDN_HEADS, GDN_HD)
        variance = (core_heads * core_heads).sum(axis=1) / GDN_HD
        normed = core_heads / np.sqrt(variance + EPS)[:, None] * norm_w[None, :]
        gated = bf16_round_f32((normed * swish(z_heads)).reshape(-1))
        return self.linear(gated, p + "out_proj"), layer_state

    def full_attention(self, layer, x, cache, position):
        p = PREFIX + f"{layer}.self_attn."
        q_fused = self.linear(x, p + "q_proj").reshape(ATTN_HEADS,
                                                       2 * ATTN_HEAD_DIM)
        gate = sigmoid(q_fused[:, ATTN_HEAD_DIM:])
        q_norm = bf16_to_f32(self.st.raw(p + "q_norm.weight").reshape(-1))
        k_norm = bf16_to_f32(self.st.raw(p + "k_norm.weight").reshape(-1))
        value = self.head_rmsnorm(q_fused[:, :ATTN_HEAD_DIM], q_norm, self.eps)
        value = self.rope(value, position)
        k_raw = self.linear(x, p + "k_proj").reshape(ATTN_KV_HEADS, ATTN_HEAD_DIM)
        v_raw = self.linear(x, p + "v_proj")
        k = self.head_rmsnorm(k_raw, k_norm, self.eps)
        k = self.rope(k, position)
        cache.append((f32_to_bf16_u16(k.reshape(-1)), f32_to_bf16_u16(v_raw)))
        keys = bf16_to_f32(np.stack([row[0] for row in cache]).reshape(
            len(cache), ATTN_KV_HEADS, ATTN_HEAD_DIM))
        values = bf16_to_f32(np.stack([row[1] for row in cache]).reshape(
            len(cache), ATTN_KV_HEADS, ATTN_HEAD_DIM))
        out = np.empty((ATTN_HEADS, ATTN_HEAD_DIM), dtype=np.float32)
        scale = np.float32(1.0 / 16.0)
        for h in range(ATTN_HEADS):
            kvh = h // (ATTN_HEADS // ATTN_KV_HEADS)
            scores = (keys[:, kvh, :] @ value[h]) * scale
            weights = np.exp(scores - scores.max())
            weights = weights / weights.sum()
            out[h] = weights @ values[:, kvh, :]
        gated = bf16_round_f32((out * gate).reshape(-1))
        return self.linear(gated, p + "o_proj")

    def routed_expert(self, layer, expert, x):
        base = PREFIX + f"{layer}.mlp.experts.{expert}."
        gate_w = self.tensor(base + "gate_proj.weight")
        up_w = self.tensor(base + "up_proj.weight")
        down_w = self.tensor(base + "down_proj.weight")
        gate = bf16_round_f32(gate_w @ x)
        up = bf16_round_f32(up_w @ x)
        activated = bf16_round_f32(swish(gate) * up)
        return bf16_round_f32(down_w @ activated)

    def moe(self, layer, x, sink):
        scores = self.tensor(PREFIX + f"{layer}.mlp.gate.weight") @ x
        order = np.argsort(-scores, kind="stable")
        selected = np.sort(order[:TOPK])
        picked = scores[selected]
        shifted = picked - picked.max()
        weights = np.exp(shifted) / np.exp(shifted).sum()
        routed = np.zeros(HIDDEN, dtype=np.float32)
        for i in range(TOPK):
            output = self.routed_expert(layer, int(selected[i]), x)
            routed += weights[i] * output
        routed = bf16_round_f32(routed)
        gate = self.linear(x, PREFIX + f"{layer}.mlp.shared_expert.gate_proj")
        up = self.linear(x, PREFIX + f"{layer}.mlp.shared_expert.up_proj")
        activated = bf16_round_f32(swish(gate) * up)
        down = self.linear(activated,
                           PREFIX + f"{layer}.mlp.shared_expert.down_proj")
        gate_weight = bf16_to_f32(self.st.raw(
            PREFIX + f"{layer}.mlp.shared_expert_gate.weight").reshape(-1))
        coeff = sigmoid(np.float32(x @ gate_weight))
        shared = bf16_round_f32(coeff * down)
        combined = bf16_round_f32(routed + shared)
        sink.append(selected.astype(np.int32))
        sink.append(weights.astype(np.float32))
        return combined

    def ple(self, streams, history):
        embedding = np.zeros(PLE_HEADS * PLE_HEAD_DIM, dtype=np.uint16)
        for head in range(PLE_HEADS):
            ngram = (head // 8) + 2
            shifted = []
            for s in range(ngram):
                source = 2 - s
                valid = True
                for p in range(max(source, 0), 2):
                    if history[p] == EOS_INVALID:
                        valid = False
                shifted.append(history[source] if source >= 0 and valid
                               else EOS_INVALID)
            mixed = self._wrap64(int(shifted[0]) * int(self.ple_multipliers[0]))
            for s in range(1, ngram):
                mixed = np.int64(mixed) ^ self._wrap64(
                    int(shifted[s]) * int(self.ple_multipliers[s]))
            head_vocab = int(self.ple_vocabs[head])
            m = mixed % head_vocab
            if m < 0:
                m += head_vocab
            global_id = int(self.ple_offsets[head]) + int(m)
            shard = global_id // self.ple_rows_per_shard
            in_shard = global_id % self.ple_rows_per_shard
            row = self.st.raw_rows(
                PREFIX + f"{PLE_LAYER}.ple.ple_embedding.ngram_embedding."
                f"shard_{shard}.weight", in_shard, 1)
            embedding[head * PLE_HEAD_DIM:(head + 1) * PLE_HEAD_DIM] = row[0]
        flat = bf16_to_f32(embedding)
        key = self.linear(flat, PREFIX + f"{PLE_LAYER}.ple.key_proj")
        value = self.linear(flat, PREFIX + f"{PLE_LAYER}.ple.value_proj")
        norm_query = group_norm(streams, bf16_to_f32(self.st.raw(
            PREFIX + f"{PLE_LAYER}.ple.norm_query.weight").reshape(-1)), self.eps)
        norm_key = group_norm(key, bf16_to_f32(self.st.raw(
            PREFIX + f"{PLE_LAYER}.ple.norm_key.weight").reshape(-1)), self.eps)
        q_s = norm_query.reshape(HC_STREAMS, HIDDEN)
        k_s = norm_key.reshape(HC_STREAMS, HIDDEN)
        dot = (k_s * q_s).sum(axis=1) / np.sqrt(np.float32(HIDDEN))
        magnitude = np.sqrt(np.maximum(np.abs(dot), np.float32(1e-6)))
        gate = sigmoid(np.copysign(magnitude, dot))
        gated = bf16_round_f32((gate[:, None] * value.reshape(1, HIDDEN)).reshape(-1))
        norm_conv = group_norm(gated, bf16_to_f32(self.st.raw(
            PREFIX + f"{PLE_LAYER}.ple.norm_conv.weight").reshape(-1)), self.eps)
        conv_weight = self.tensor(
            PREFIX + f"{PLE_LAYER}.ple.conv1d.weight").reshape(HC_WIDTH, 4)
        if self.ple_tail is None:
            self.ple_tail = np.zeros((HC_WIDTH, 9), dtype=np.uint16)
        window = np.zeros((HC_WIDTH, 4), dtype=np.float32)
        window[:, 0] = bf16_to_f32(f32_to_bf16_u16(norm_conv))
        for tap in range(1, 4):
            window[:, tap] = bf16_to_f32(self.ple_tail[:, tap * 3 - 1])
        acc = (window * conv_weight).sum(axis=1)
        conv_out = bf16_round_f32(swish(acc))
        self.ple_tail = np.concatenate(
            [f32_to_bf16_u16(norm_conv).reshape(-1, 1),
             self.ple_tail[:, :8]], axis=1)
        updated = bf16_round_f32(streams + gated)
        return bf16_round_f32(updated + conv_out)

    def forward_layer(self, index, streams, states, caches, sink, history):
        if index == PLE_LAYER:
            streams = self.ple(streams, history)
        mixed, inject_pre = self.hc_prep(
            PREFIX + f"{index}.attn_hyper_connection.", streams)
        if self.layer_types[index] == "linear_attention":
            delta, states[index] = self.gdn_attention(index, mixed,
                                                      states.get(index))
        elif self.layer_types[index] == "full_attention":
            delta = self.full_attention(index, mixed, caches[index],
                                        len(caches[index]))
        else:
            raise ValueError(f"unsupported layer type {self.layer_types[index]}")
        streams = self.hc_inject(streams, inject_pre, delta)
        mixed, inject_pre = self.hc_prep(
            PREFIX + f"{index}.mlp_hyper_connection.", streams)
        delta = self.moe(index, mixed, sink)
        return self.hc_inject(streams, inject_pre, delta)

    def decode_step(self, token_id, position, states, caches, capture):
        if token_id < 0 or token_id >= self.vocab:
            raise ValueError(f"token {token_id} outside vocabulary {self.vocab}")
        if position > INDEXER_ALL_VISIBLE_MAX_CONTEXT:
            raise ValueError("reference models the indexer-all-visible regime; "
                             f"context {position} exceeds {INDEXER_ALL_VISIBLE_MAX_CONTEXT}")
        if position == 0:
            for i in range(self.layers):
                if self.layer_types[i] == "full_attention":
                    caches[i] = []
            self.ple_history = [0, 0]
            self.ple_tail = None
        streams = bf16_round_f32(np.tile(self.embed(token_id), HC_STREAMS))
        history = self.ple_history + [token_id]
        for i in range(self.layers):
            sink = []
            streams = self.forward_layer(i, streams, states, caches, sink, history)
            if sink:
                capture[(position, i)] = sink
            if not np.isfinite(streams).all():
                raise ValueError(f"nonfinite reference state at layer {i}")
        self.ple_history = history[-2:]
        return streams

    def logits(self, streams, chunk=4096):
        normed = group_norm(streams, bf16_to_f32(self.st.raw(
            "model.language_model.hyper_connection_mixer.hc_norm.weight"
        ).reshape(-1)), self.eps)
        lowrank = self.linear(
            normed,
            "model.language_model.hyper_connection_mixer.input_mix_weight_down")
        lowrank = bf16_round_f32(swish(lowrank / np.float32(HC_STREAMS)))
        up = self.linear(
            lowrank,
            "model.language_model.hyper_connection_mixer.input_mix_weight_up")
        up_s = up.reshape(HC_STREAMS, HIDDEN)
        normed_s = normed.reshape(HC_STREAMS, HIDDEN)
        mixed = bf16_round_f32(
            (sigmoid(up_s) * normed_s).sum(axis=0) / np.float32(HC_STREAMS))
        if self._lm_head_u16 is None:
            self._lm_head_u16 = self.st.raw("lm_head.weight")
            if self._lm_head_u16.dtype != np.uint16:
                raise ValueError("reference lm_head must be BF16")
        lm = self._lm_head_u16
        best = -np.inf
        best_token = -1
        for start in range(0, lm.shape[0], chunk):
            scores = bf16_to_f32(lm[start:start + chunk]) @ mixed
            i = int(np.argmax(scores))
            if float(scores[i]) > best:
                best = float(scores[i])
                best_token = start + i
        return best_token, best


ENGINE_CLASS = Qwen4FlashEngine
