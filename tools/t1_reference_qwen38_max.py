import json
import os
import time

import numpy as np

from t1_reference_common import (_E2M1_LUT, _E4M3_LUT, Safetensors,
                                 bf16_round_f32, bf16_to_f32, define_float,
                                 define_uint, f32_to_bf16_u16,
                                 fp8_block_to_bf16, rmsnorm, sigmoid)

PREFIX = "model.layers."
L2_EPS = 1e-6
LINEAR_TILE_ELEMENTS = 1 << 24
DEQUANT_TILE_ROWS = 512
SMALL_WEIGHT_BYTES = 1 << 20


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
        ("ROUTED_EXPERT_COUNT", "num_experts", "uint"),
        ("EXPERTS_PER_TOKEN", "num_experts_per_tok", "uint"),
        ("END_OF_TEXT_TOKEN_ID", "eos_token_id", "uint"),
        ("ATTN_PERIOD", "full_attention_interval", "uint"),
        ("KDA_HEAD_KEY_DIMENSION", "linear_key_head_dim", "uint"),
        ("KDA_HEAD_VALUE_DIMENSION", "linear_value_head_dim", "uint"),
    ]
    for dname, cname, kind in checks:
        if dname not in defines:
            raise Qwen38MaxConfigError(
                f"SPARK_LLM_{dname} missing from llm defines header")
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


class Nvfp4Tile:
    def __init__(self, rows, packed, groups):
        self.rows = rows
        self.packed = packed
        self.groups = groups
        self.payload = np.empty((rows, packed), dtype=np.uint8)
        self.scales = np.empty((rows, groups), dtype=np.uint8)
        self.weights = np.empty((rows, groups, 16), dtype=np.float32)

    def dequant(self, count, scalar):
        payload = self.payload[:count]
        weights = self.weights[:count]
        weights[:, :, 0::2] = _E2M1_LUT[(payload & 0xF).reshape(
            count, self.groups, 8)]
        weights[:, :, 1::2] = _E2M1_LUT[(payload >> 4).reshape(
            count, self.groups, 8)]
        weights *= _E4M3_LUT[self.scales[:count]][:, :, None]
        weights *= scalar
        return weights.reshape(count, self.groups * 16)


class Qwen38MaxEngine:
    def __init__(self, checkpoint_dir, defines, config):
        self.started = time.monotonic()
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
        self._u8_tile = np.empty(LINEAR_TILE_ELEMENTS * 2, dtype=np.uint8)
        self._u32_tile = np.empty(LINEAR_TILE_ELEMENTS, dtype=np.uint32)
        self._scalar_buf = bytearray(4)
        self._small = {}
        self._views = {}
        self._tiles = {}
        self._entries = {}
        self._bases = {}

    def progress(self, **fields):
        fields["elapsed_s"] = round(time.monotonic() - self.started, 3)
        print(json.dumps({"family": "qwen38_max", **fields}), flush=True)

    def _entry_of(self, name):
        entry = self._entries.get(name)
        if entry is None:
            entry = self.st.entry(name)
            self._entries[name] = entry
        return entry

    def _release_header(self, name):
        fname = self.st.map.get(name, "model.safetensors")
        if ".mlp.experts." in name and fname in self.st.headers:
            del self.st.headers[fname]
            handle = self.st.fds.pop(fname, None)
            if handle is not None:
                handle.close()

    def _open(self, name):
        fname = self.st.map.get(name, "model.safetensors")
        if fname not in self.st.fds:
            self.st._open(fname)
        base = self._bases.get(fname)
        if base is None:
            base = self.st.headers[fname][1]
            self._bases[fname] = base
        return self.st.fds[fname], base

    def _read_rows(self, name, buf, first_row, row_bytes, row_count):
        entry = self._entry_of(name)
        start, end = entry["data_offsets"]
        extent = end - start
        if extent % row_bytes != 0 \
                or extent // row_bytes < first_row + row_count:
            raise ValueError(f"row slab {first_row}+{row_count} outside "
                             f"{name} extent {extent} row_bytes {row_bytes}")
        need = row_bytes * row_count
        if len(buf) != need:
            raise ValueError(f"buffer size {len(buf)} disagrees with {name} "
                             f"row slab {need}")
        fh, base = self._open(name)
        got = os.preadv(fh.fileno(), [buf], base + start + first_row * row_bytes)
        if got != need:
            raise ValueError(f"short read for {name}: {got} of {need}")
        if ".mlp.experts." not in name and hasattr(os, "posix_fadvise") \
                and hasattr(os, "POSIX_FADV_DONTNEED"):
            try:
                os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass

    def _view(self, name):
        if name in self._views:
            return self._views[name]
        entry = self._entry_of(name)
        extent = entry["data_offsets"][1] - entry["data_offsets"][0]
        buf = bytearray(extent)
        self._read_rows(name, memoryview(buf), 0, extent, 1)
        dtype = {"BF16": np.uint16, "F32": np.float32, "F16": np.float16,
                 "U8": np.uint8, "F8_E4M3": np.uint8, "I64": np.int64}[
            entry["dtype"]]
        array = np.frombuffer(buf, dtype=dtype).reshape(entry["shape"])
        self._views[name] = array
        return array

    def _to_f32(self, name, raw):
        if raw.dtype == np.uint16:
            return bf16_to_f32(raw)
        if raw.dtype == np.uint8:
            scale = self.st.pread(name + "_scale_inv").astype(np.float32)
            rows, cols = raw.shape
            return bf16_to_f32(fp8_block_to_bf16(raw, scale, rows, cols))
        return raw.astype(np.float32)

    def tensor(self, name):
        entry = self._entry_of(name)
        extent = entry["data_offsets"][1] - entry["data_offsets"][0]
        if extent >= SMALL_WEIGHT_BYTES:
            return self._to_f32(name, self._view(name))
        if name not in self._small:
            self._small[name] = self._to_f32(name, self._view(name))
        return self._small[name]

    def _shape_of(self, name):
        return [int(s) for s in self._entry_of(name)["shape"]]

    def _gemv(self, x, wname):
        shape = self._shape_of(wname)
        rows, cols = shape[0], shape[-1]
        if cols != x.shape[0]:
            raise ValueError(f"weight {wname} shape {shape} disagrees with "
                             f"activation width {x.shape[0]}")
        out = np.empty(rows, dtype=np.float32)
        done = 0
        while done < rows:
            count = min(LINEAR_TILE_ELEMENTS // cols, rows - done)
            buf = memoryview(self._u8_tile)[:count * cols * 2]
            self._read_rows(wname, buf, done, cols * 2, count)
            flat = self._u32_tile[:count * cols]
            np.copyto(flat, np.frombuffer(buf, dtype=np.uint16))
            np.left_shift(flat, np.uint32(16), out=flat)
            np.dot(flat.view(np.float32).reshape(count, cols), x,
                   out=out[done:done + count])
            done += count
        return out

    def linear(self, x, name):
        return bf16_round_f32(self._gemv(x, name + ".weight"))

    def _scalar_of(self, name):
        self._read_rows(name, memoryview(self._scalar_buf), 0, 4, 1)
        return np.frombuffer(self._scalar_buf, dtype=np.float32)[0]

    def _nvfp4_gemv(self, base, name, x):
        payload_name = base + name + ".weight"
        scale_name = base + name + ".weight_scale"
        scalar_name = base + name + ".weight_scale_2"
        rows, packed = self._shape_of(payload_name)
        cols = packed * 2
        if cols % 16 != 0:
            raise ValueError(f"nvfp4 input dim {cols} not a multiple of 16")
        if cols != x.shape[0]:
            raise ValueError(f"weight {payload_name} input dim {cols} "
                             f"disagrees with activation width {x.shape[0]}")
        if self._shape_of(scale_name)[-1] != cols // 16:
            raise ValueError(f"nvfp4 scale extent disagrees with {payload_name}")
        scalar = np.float32(self._scalar_of(scalar_name))
        key = (rows, cols)
        if key not in self._tiles:
            self._tiles[key] = Nvfp4Tile(DEQUANT_TILE_ROWS, packed, cols // 16)
        tile = self._tiles[key]
        out = np.empty(rows, dtype=np.float32)
        done = 0
        while done < rows:
            count = min(tile.rows, rows - done)
            self._read_rows(payload_name,
                            memoryview(tile.payload[:count].reshape(-1)),
                            done, packed, count)
            self._read_rows(scale_name,
                            memoryview(tile.scales[:count].reshape(-1)),
                            done, cols // 16, count)
            np.dot(tile.dequant(count, scalar), x, out=out[done:done + count])
            done += count
        return out

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
        qkv = bf16_round_f32(self._gemv(
            x, prefix + "linear_attn.in_proj_qkv.weight"))
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
        a_pre = bf16_round_f32(self._gemv(
            x, prefix + "linear_attn.in_proj_a.weight"))
        b_pre = bf16_round_f32(self._gemv(
            x, prefix + "linear_attn.in_proj_b.weight"))
        a_log = bf16_to_f32(self.st.pread(
            prefix + "linear_attn.A_log").reshape(-1))
        dt_bias = bf16_to_f32(self.st.pread(
            prefix + "linear_attn.dt_bias").reshape(-1))
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
        z = bf16_round_f32(self._gemv(x, prefix + "linear_attn.in_proj_z.weight"))
        norm_w = bf16_to_f32(self.st.pread(
            prefix + "linear_attn.norm.weight").reshape(-1))
        zc = core.reshape(self.v_heads, self.kd)
        variance = (zc * zc).sum(axis=1) / self.kd
        normed = zc / np.sqrt(variance + self.eps)[:, None] * norm_w[None, :]
        gated = bf16_round_f32(normed.reshape(-1) * swish(z))
        return self.linear(gated, prefix + "linear_attn.out_proj"), layer_state

    def full_attention(self, prefix, x, cache, position):
        qf = bf16_round_f32(self._gemv(x, prefix + "self_attn.q_proj.weight"))
        qf = qf.reshape(self.heads, 2 * self.head_dim)
        value = qf[:, 0:self.head_dim].copy()
        gate = sigmoid(qf[:, self.head_dim:])
        qn = bf16_to_f32(self.st.pread(
            prefix + "self_attn.q_norm.weight").reshape(-1))
        kn = bf16_to_f32(self.st.pread(
            prefix + "self_attn.k_norm.weight").reshape(-1))
        value = value / np.sqrt((value * value).sum(axis=1, keepdims=True)
                                / self.head_dim + self.eps) * qn[None, :]
        value = self.partial_rope(value, position)
        k_raw = bf16_round_f32(self._gemv(x, prefix + "self_attn.k_proj.weight"))
        v_raw = bf16_round_f32(self._gemv(x, prefix + "self_attn.v_proj.weight"))
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

    def routed_expert(self, base, expert, x):
        expert_base = base + f"experts.{expert}."
        gate = np.minimum(bf16_round_f32(self._nvfp4_gemv(
            expert_base, "gate_proj", x)), self.limit)
        up = bf16_round_f32(self._nvfp4_gemv(expert_base, "up_proj", x))
        activated = bf16_round_f32(swish(gate) * up)
        out = bf16_round_f32(self._nvfp4_gemv(expert_base, "down_proj",
                                              activated))
        self._release_header(expert_base + "gate_proj.weight")
        return out

    def shared_expert(self, prefix, x):
        gate = self.linear(x, prefix + "mlp.shared_expert.gate_proj")
        up = self.linear(x, prefix + "mlp.shared_expert.up_proj")
        activated = bf16_round_f32(swish(gate) * up)
        return self.linear(activated, prefix + "mlp.shared_expert.down_proj")

    def moe(self, prefix, x, sink):
        scores = self._gemv(x, prefix + "mlp.gate.weight")
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

    def decode_step(self, token_id, position, states, caches, capture,
                    capture_streams=None):
        mark = time.monotonic()
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
            if capture_streams is not None:
                capture_streams(i, streams)
            if not np.isfinite(streams).all():
                raise ValueError(f"nonfinite reference state at layer {i}")
            self.progress(position=position, layer=i,
                          layer_ms=round((time.monotonic() - mark) * 1000, 1))
            mark = time.monotonic()
        return streams

    def logits(self, streams):
        norm = bf16_round_f32(rmsnorm(
            streams, self.tensor("model.norm.weight"), self.eps))
        lm = self._entry_of("lm_head.weight")
        if lm["dtype"] != "BF16":
            raise ValueError("reference lm_head must be BF16")
        rows_total = int(lm["shape"][0])
        cols = int(lm["shape"][1])
        if cols != norm.shape[0]:
            raise ValueError(f"lm_head width {cols} disagrees with hidden "
                             f"{norm.shape[0]}")
        best = -np.inf
        best_token = -1
        tile_rows = max(1, LINEAR_TILE_ELEMENTS // cols)
        done = 0
        while done < rows_total:
            count = min(tile_rows, rows_total - done)
            buf = memoryview(self._u8_tile)[:count * cols * 2]
            self._read_rows("lm_head.weight", buf, done, cols * 2, count)
            flat = self._u32_tile[:count * cols]
            np.copyto(flat, np.frombuffer(buf, dtype=np.uint16))
            np.left_shift(flat, np.uint32(16), out=flat)
            scores = flat.view(np.float32).reshape(count, cols) @ norm
            i = int(np.argmax(scores))
            if float(scores[i]) > best:
                best = float(scores[i])
                best_token = done + i
            done += count
        return best_token, best


ENGINE_CLASS = Qwen38MaxEngine
