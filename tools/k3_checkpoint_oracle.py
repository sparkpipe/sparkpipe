#!/usr/bin/env python3
"""Verify placed K3 pack bytes against the warm-storage checkpoint.

Offline oracle: no model execution, no daemon contact. The checkpoint is the
only ground truth and the pack must equal what the pack law (pack==packer==
checkpoint) derives from it. Byte-exact expectations are recomputed for a
deterministic sample of every tensor class:

  direct moves      norms, router, KDA convs/dt_bias/A_log/o_norm, routed,
                    shared, dense, vocab rows - rank slices are row/col
                    ranges of the checkpoint tensors
  gamma folds       attnres_* = res_proj * res_norm, f32 semantics, BF16 RNE
  KDA fusion        kda_qkv_beta sections q|k|v|beta against the four
                    checkpoint projections with per-head rank row mapping
  MLA q-fold        mla_q_up = kv_b k_nope^T q_b per head, rope rows copied;
                    mla_kv_b_value = kv_b value half
  expert planes     mxfp4 payload + E8M0 scale bytes under the interleave
                    grid: w1 k-tile slice [t0, t0+take), w2 cell slice
                    [c0, c0+take), gate=w1 up=w3 concat order

Exit 0 = every sampled byte matches, 1 = mismatch, 2 = oracle error. Each
mismatch names tensor, byte offset, expected/got hex. --digest adds a
streaming sha256 of the whole pack so verdicts anchor to placed bytes.

usage:
  k3_checkpoint_oracle.py --pack PACK --checkpoint DIR [--seed N]
      [--experts N] [--tiles N] [--digest] [--json]
"""
import argparse
import hashlib
import json
import mmap
import os
import random
import struct
import sys

MAGIC = 0x4B33504B
ALIGN = 128
GROUP = 32
TILE_K_DEFAULT = 128
CELL_ROWS = 16
A_LOG_SOURCE_HEADS = 128

REPLICATED = {
    "attn_norm_weight", "mlp_norm_weight", "router_weight", "router_bias",
    "kda_decay_down_weight", "kda_decay_bias",
    "kda_head_log_scale", "kda_out_norm_weight",
    "mla_q_down_weight", "mla_q_norm_weight", "mla_kv_a_weight",
    "mla_kv_a_norm_weight", "routed_norm_weight",
}
KDA_OUT_HEADS = ("kda_qkv_beta_weight", "kda_q_conv_weight",
                 "kda_k_conv_weight", "kda_v_conv_weight",
                 "kda_decay_up_weight", "kda_gate_weight")
MLA_OUT_HEADS = ("mla_q_up_weight", "mla_kv_b_value_weight",
                 "mla_gate_weight")
MLA_IN_HEADS = ("mla_out_weight",)
KDA_IN_HEADS = ("kda_out_weight",)
OUTPUT_DIM = ("routed_down_weight",)
INPUT_DIM = ("routed_up_weight",)
CONCAT_OUTPUT = ("shared_w1_weight", "dense_gate_up_weight")
INPUT_DIM_PLAIN = ("shared_w2_weight", "dense_down_weight")


class OracleFailure(RuntimeError):
    pass


class Checkpoint:
    def __init__(self, model_dir):
        self.model_dir = os.path.abspath(model_dir)
        index = os.path.join(self.model_dir, "model.safetensors.index.json")
        if os.path.isfile(index):
            with open(index, "rb") as handle:
                self.weight_map = json.loads(handle.read())["weight_map"]
        else:
            single = os.path.join(self.model_dir, "model.safetensors")
            if not os.path.isfile(single):
                raise OracleFailure(f"no safetensors index or file in {model_dir}")
            self.weight_map = None
            self.single = os.path.basename(single)
        self.headers = {}
        self.handles = {}

    def names(self):
        if self.weight_map is not None:
            return set(self.weight_map)
        return set(self._header(self.single)[0])

    def _header(self, shard):
        if shard not in self.headers:
            path = os.path.join(self.model_dir, shard)
            handle = open(path, "rb")
            length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(length))
            header.pop("__metadata__", None)
            self.headers[shard] = (header, 8 + length)
            self.handles[shard] = handle
        return self.headers[shard]

    def _locate(self, name):
        shard = self.weight_map.get(name) if self.weight_map is not None \
            else (self.single if name in self._header(self.single)[0] else None)
        if shard is None:
            raise OracleFailure(f"missing checkpoint tensor: {name}")
        header, base = self._header(shard)
        entry = header[name]
        begin, end = entry["data_offsets"]
        return shard, base + begin, end - begin, entry["dtype"], entry["shape"]

    def raw(self, name, offset=0, length=None):
        shard, start, size, _, _ = self._locate(name)
        if length is None:
            length = size - offset
        if offset < 0 or offset + length > size:
            raise OracleFailure(
                f"{name}: read [{offset}, {offset + length}) past {size} B")
        return os.pread(self.handles[shard].fileno(), length, start + offset)

    def info(self, name):
        _, _, size, dtype, shape = self._locate(name)
        return dtype, tuple(shape), size


class PackView:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.handle = open(path, "rb")
        self.raw = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)
        magic, version, length = struct.unpack_from("<IIQ", self.raw, 0)
        if magic != MAGIC:
            raise OracleFailure(f"{path}: magic {magic:#x} is not a K3 pack")
        if version != 2:
            raise OracleFailure(f"{path}: format version {version}, expected 2")
        self.manifest = self._manifest(self.raw, length)
        self.base = 16 + length
        self.base += (-self.base) % ALIGN
        self.config = self.manifest["config"]
        self.file_bytes = len(self.raw)

    @staticmethod
    def _manifest(raw, length):
        blob = bytes(raw[16:16 + length])
        try:
            return json.loads(blob.decode("utf-8", errors="strict"))
        except UnicodeDecodeError as failure:
            raise OracleFailure(f"pack manifest undecodable: {failure}")

    def entry(self, name):
        if name not in self.manifest["tensors"]:
            raise OracleFailure(f"pack manifest lacks tensor {name}")
        return self.manifest["tensors"][name]

    def span(self, name, offset=0, length=None):
        entry = self.entry(name)
        if length is None:
            length = entry["bytes"] - offset
        start = self.base + entry["offset"] + offset
        if start + length > self.file_bytes:
            raise OracleFailure(f"{name}: payload runs past EOF")
        return self.raw[start:start + length]

    def close(self):
        self.raw.close()
        self.handle.close()


def f32_round(value):
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return float("inf") if value > 0 else float("-inf")


def bf16_elements(raw):
    return [struct.unpack_from("<f", struct.pack("<I", u << 16))[0]
            for (u,) in struct.iter_unpack("<H", raw)]


def bf16_raw_of_f32(values):
    out = bytearray()
    for value in values:
        u = struct.unpack("<I", struct.pack("<f", f32_round(value)))[0]
        out += struct.pack("<H", (u + 0x7FFF + ((u >> 16) & 1)) >> 16)
    return bytes(out)


def gamma_fold(proj_raw, gamma_raw, count):
    proj = bf16_elements(proj_raw)
    gamma = bf16_elements(gamma_raw)
    if len(proj) != count or len(gamma) != count:
        raise OracleFailure("gamma fold operand size mismatch")
    return bf16_raw_of_f32([proj[i] * gamma[i] for i in range(count)])


def q_fold_head(kv_b_head_raw, q_b_head_raw, nope, rope, v_head, kv_lora,
                q_lora):
    try:
        import numpy as np
    except ImportError as failure:
        raise OracleFailure(
            f"numpy is required for the MLA q-fold check: {failure}")
    kv_b = (np.frombuffer(kv_b_head_raw, dtype=np.uint16)
            .reshape(nope + v_head, kv_lora).astype(np.uint32) << 16) \
        .view(np.float32)
    q_b = (np.frombuffer(q_b_head_raw, dtype=np.uint16)
           .reshape(nope + rope, q_lora).astype(np.uint32) << 16) \
        .view(np.float32)
    absorbed = np.einsum("nl,nq->lq", kv_b[:nope, :], q_b[:nope, :])
    folded = np.concatenate([absorbed, q_b[nope:, :]], axis=0)
    value = kv_b[nope:, :]

    def to_bf16(planef):
        u = np.ascontiguousarray(planef, dtype=np.float32).view(np.uint32)
        rounded = u + 0x7FFF + ((u >> 16) & 1)
        return (rounded >> 16).astype("<u2").tobytes()

    return to_bf16(folded), to_bf16(value)


class Oracle:
    def __init__(self, pack, cp, seed, expert_samples, tile_samples):
        self.pack = pack
        self.cp = cp
        self.rng = random.Random(seed)
        self.expert_samples = expert_samples
        self.tile_samples = tile_samples
        self.cfg = pack.config
        self.mismatches = []
        self.checks = 0
        self.bytes_checked = 0
        self.per_class = {}

    def fail(self, name, offset, expected, got):
        if len(self.mismatches) < 40:
            self.mismatches.append({
                "tensor": name, "offset": offset,
                "expected": expected[:12].hex(), "got": got[:12].hex()})

    def compare(self, name, expected, got, cls):
        self.checks += 1
        self.per_class[cls] = self.per_class.get(cls, 0) + 1
        self.bytes_checked += len(expected)
        if len(expected) != len(got):
            self.fail(f"{name} length {len(expected)} vs {len(got)}", 0,
                      b"", b"")
        elif expected != got:
            first = next(i for i in range(len(expected))
                         if expected[i] != got[i])
            self.fail(name, first, expected[first:first + 12],
                      got[first:first + 12])

    def head_bounds(self, kind, degree, rank):
        heads = self.cfg["kda_heads"] if kind.startswith("kda") \
            else self.cfg["heads"]
        per = heads // degree
        return per * rank, per * (rank + 1)

    def pick_rows(self, lo, hi, count):
        count = min(count, hi - lo)
        return sorted(self.rng.sample(range(lo, hi), count))

    def row_picks(self, name, rows_total, lo, hi, count):
        entry = self.pack.entry(name)
        if rows_total <= 0 or entry["bytes"] % rows_total:
            raise OracleFailure(f"{name}: {entry['bytes']} B does not tile "
                                f"{rows_total} rows")
        row_bytes = entry["bytes"] // rows_total
        picks = self.pick_rows(lo, hi, count)
        got = bytearray()
        for row in picks:
            start = self.pack.base + entry["offset"] + row * row_bytes
            got += self.pack.raw[start:start + row_bytes]
        return bytes(got), row_bytes, picks

    def cp_rows(self, cp_name, row_bytes, picks, base_row=0):
        out = bytearray()
        for row in picks:
            out += self.cp.raw(cp_name, (base_row + row) * row_bytes,
                               row_bytes)
        return bytes(out)

    def rank_block_rows(self, rows_full, degree, rank):
        if rows_full % degree:
            raise OracleFailure(f"{rows_full} rows do not split {degree}")
        per = rows_full // degree
        return per * rank, per * (rank + 1)

    def check_row_split(self, name, cp_name, rows_full, degree, rank, picks_n,
                        cls):
        lo, hi = self.rank_block_rows(rows_full, degree, rank)
        span = hi - lo
        got, row_bytes, picks = self.row_picks(name, span, 0, span, picks_n)
        expected = self.cp_rows(cp_name, row_bytes, [lo + p for p in picks])
        self.compare(f"{name} pack rows {picks} = cp rows "
                     f"{[lo + p for p in picks]}", expected, got, cls)

    def check_col_split(self, name, cp_name, degree, rank, rows, in_full,
                        picks_n, cls):
        per = in_full // degree
        got, _, picks = self.row_picks(name, rows, 0, rows, picks_n)
        expected = bytearray()
        for row in picks:
            expected += self.cp.raw(cp_name, row * in_full + per * rank, per)
        self.compare(f"{name} rows {picks} cols [{per * rank},"
                     f"{per * (rank + 1)})", bytes(expected), got, cls)


def check_common(oracle, layer, degree, rank, hidden):
    p = f"model.layers.{layer}."
    sp = f"language_model.model.layers.{layer}."
    got = oracle.pack.span(p + "attn_norm_weight")
    oracle.compare(p + "attn_norm_weight", oracle.cp.raw(sp +
                   "input_layernorm.weight"), got, "norm")
    got = oracle.pack.span(p + "mlp_norm_weight")
    oracle.compare(p + "mlp_norm_weight", oracle.cp.raw(sp +
                   "post_attention_layernorm.weight"), got, "norm")
    for dst, res in (("attnres_attn_weight", "self_attention_res"),
                     ("attnres_mlp_weight", "mlp_res")):
        proj = oracle.cp.raw(sp + res + "_proj.weight")
        gamma = oracle.cp.raw(sp + res + "_norm.weight")
        oracle.compare(p + dst, gamma_fold(proj, gamma, hidden),
                       oracle.pack.span(p + dst), "gamma_fold")


def check_kda(oracle, layer, degree, rank, deep):
    cfg = oracle.cfg
    hidden, kda_dim = cfg["hidden"], cfg["kda_heads"] * cfg["kda_head"]
    kda_head = cfg["kda_head"]
    p = f"model.layers.{layer}."
    a = f"language_model.model.layers.{layer}.self_attn."
    entry = oracle.pack.entry(p + "kda_qkv_beta_weight")
    h0, h1 = oracle.head_bounds("kda", degree, rank)
    sections = entry.get("sections")
    if not sections:
        raise OracleFailure(f"{p}kda_qkv_beta_weight lacks sections")
    expected_rows = {"q": kda_dim, "k": kda_dim, "v": kda_dim,
                     "beta": cfg["kda_heads"]}
    cursor = 0
    row_bytes = hidden * 2
    for section in sections:
        name, rph = section["name"], section["rows_per_head"]
        if section["row_offset"] != cursor:
            raise OracleFailure(f"{p}kda_qkv_beta section {name} offset "
                                f"{section['row_offset']} != {cursor}")
        if expected_rows[name] // degree != section["rows"]:
            raise OracleFailure(f"{p}kda_qkv_beta section {name} rows "
                                f"{section['rows']} != "
                                f"{expected_rows[name] // degree}")
        lo = h0 * rph
        hi = h1 * rph
        span = hi - lo
        picks = oracle.pick_rows(0, span, 6)
        got = bytearray()
        for i in picks:
            start = (oracle.pack.base + entry["offset"]
                     + (section["row_offset"] + i) * row_bytes)
            got += oracle.pack.raw[start:start + row_bytes]
        cp_field = "b_proj" if name == "beta" else f"{name}_proj"
        expected = oracle.cp_rows(a + f"{cp_field}.weight", row_bytes,
                                  [lo + i for i in picks])
        oracle.compare(f"{p}kda_qkv_beta[{name}] cp rows "
                       f"{[lo + i for i in picks]}",
                       bytes(expected), bytes(got), "kda_fused")
        cursor += section["rows"]
    if cursor != sum(expected_rows.values()) // degree:
        raise OracleFailure(f"{p}kda_qkv_beta sections cover {cursor}")
    for conv in "qkv":
        oracle.check_row_split(p + f"kda_{conv}_conv_weight",
                               a + f"{conv}_conv1d.weight", kda_dim, degree,
                               rank, kda_dim // degree, "kda_conv")
    got = oracle.pack.span(p + "kda_decay_bias")
    oracle.compare(p + "kda_decay_bias", oracle.cp.raw(a + "dt_bias"),
                   got, "kda_f32")
    expected = oracle.cp.raw(a + "A_log", 0, cfg["kda_heads"] * 4)
    oracle.compare(p + "kda_head_log_scale", expected,
                   oracle.pack.span(p + "kda_head_log_scale"), "kda_f32")
    got = oracle.pack.span(p + "kda_out_norm_weight")
    oracle.compare(p + "kda_out_norm_weight", oracle.cp.raw(a + "o_norm.weight"),
                   got, "kda_f32")
    oracle.check_row_split(p + "kda_decay_up_weight", a + "f_b_proj.weight",
                           kda_dim, degree, rank, 4, "kda_proj")
    oracle.check_row_split(p + "kda_gate_weight", a + "g_proj.weight",
                           kda_dim, degree, rank, 4, "kda_proj")
    in_full = kda_dim * 2
    oracle.check_col_split(p + "kda_out_weight", a + "o_proj.weight",
                           degree, rank, hidden, in_full, 4, "kda_proj")
    picks_n = kda_head if deep else 4
    oracle.check_row_split(p + "kda_decay_down_weight",
                           a + "f_a_proj.weight", kda_head, 1, 0,
                           picks_n, "kda_proj")


def check_mla(oracle, layer, degree, rank, deep):
    cfg = oracle.cfg
    hidden = cfg["hidden"]
    heads, nope, rope = cfg["heads"], cfg["nope"], cfg["rope"]
    kv_lora, v_head, q_lora = cfg["kv_lora"], cfg["v_head"], cfg["q_lora"]
    p = f"model.layers.{layer}."
    a = f"language_model.model.layers.{layer}.self_attn."
    oracle.check_row_split(p + "mla_q_down_weight", a + "q_a_proj.weight",
                           q_lora, 1, 0, q_lora if deep else 4, "mla_proj")
    got = oracle.pack.span(p + "mla_q_norm_weight")
    oracle.compare(p + "mla_q_norm_weight",
                   oracle.cp.raw(a + "q_a_layernorm.weight"), got, "norm")
    oracle.check_row_split(p + "mla_kv_a_weight",
                           a + "kv_a_proj_with_mqa.weight", kv_lora + rope,
                           1, 0, kv_lora + rope if deep else 4, "mla_proj")
    got = oracle.pack.span(p + "mla_kv_a_norm_weight")
    oracle.compare(p + "mla_kv_a_norm_weight",
                   oracle.cp.raw(a + "kv_a_layernorm.weight"), got, "norm")
    h0, h1 = oracle.head_bounds("mla", degree, rank)
    heads_fold = range(h0, h1) if deep else oracle.pick_rows(h0, h1, 1)
    q_b_row = (nope + rope) * q_lora * 2
    kv_b_row = (nope + v_head) * kv_lora * 2
    for h in heads_fold:
        q_fold, _ = q_fold_head(
            oracle.cp.raw(a + "kv_b_proj.weight", h * kv_b_row, kv_b_row),
            oracle.cp.raw(a + "q_b_proj.weight", h * q_b_row, q_b_row),
            nope, rope, v_head, kv_lora, q_lora)
        got = oracle.pack.span(
            p + "mla_q_up_weight", (h - h0) * (kv_lora + rope) * q_lora * 2,
            (kv_lora + rope) * q_lora * 2)
        oracle.compare(f"{p}mla_q_up_weight head {h} (q-fold)", q_fold, got,
                       "mla_qfold")
        value = oracle.cp.raw(a + "kv_b_proj.weight",
                              h * kv_b_row + nope * kv_lora * 2,
                              v_head * kv_lora * 2)
        got = oracle.pack.span(
            p + "mla_kv_b_value_weight",
            (h - h0) * v_head * kv_lora * 2, v_head * kv_lora * 2)
        oracle.compare(f"{p}mla_kv_b_value_weight head {h}", value, got,
                       "mla_proj")
    oracle.check_row_split(p + "mla_gate_weight", a + "g_proj.weight",
                           heads * v_head, degree, rank, 4, "mla_proj")
    oracle.check_col_split(p + "mla_out_weight", a + "o_proj.weight",
                           degree, rank, hidden, heads * v_head * 2, 4,
                           "mla_proj")


def interleave_expect(oracle, cp_prefix, expert, global_tile, neuron,
                      k_dim, plane_row_bytes, scale_stride):
    payload = oracle.cp.raw(f"{cp_prefix}.weight_packed",
                            neuron * plane_row_bytes + global_tile * 64, 64)
    scales = oracle.cp.raw(f"{cp_prefix}.weight_scale",
                           neuron * scale_stride + global_tile * 4, 4)
    return payload, scales


def check_experts(oracle, layer, degree, rank, moe, deep):
    cfg = oracle.cfg
    latent, inter = cfg["latent"], cfg["intermediate"]
    p = f"model.layers.{layer}."
    m = f"language_model.model.layers.{layer}.block_sparse_moe."
    n_experts = cfg["experts"]
    exp_count = min(6 if deep else oracle.expert_samples, n_experts)
    experts = sorted(oracle.rng.sample(range(n_experts), exp_count))
    entry = oracle.pack.entry(p + "expert_w1_weight")
    geom = entry["interleave"]
    tile_k = geom["tile_k"]
    k_tiles_full, cells_full = latent // tile_k, (2 * inter) // CELL_ROWS
    if k_tiles_full % degree:
        raise OracleFailure(f"{p}w1 {k_tiles_full} tiles split {degree}")
    take_k = k_tiles_full // degree
    t0 = take_k * rank
    rpe_rank = take_k * cells_full * (CELL_ROWS + 1)
    if entry["bytes"] != rpe_rank * 64 * n_experts:
        raise OracleFailure(f"{p}expert_w1_weight {entry['bytes']} B, "
                            f"rank grid prices {rpe_rank * 64 * n_experts}")
    tiles = sorted(set(min(t, take_k - 1) for t in
                       (0, take_k // 2, take_k - 1)))[:oracle.tile_samples]
    for e in experts:
        for t_loc in tiles:
            for c in oracle.rng.sample(range(cells_full),
                                       2 if deep else 1):
                base = e * rpe_rank + (t_loc * cells_full + c) \
                    * (CELL_ROWS + 1)
                got = oracle.pack.span(p + "expert_w1_weight",
                                       base * 64, (CELL_ROWS + 1) * 64)
                expected = bytearray()
                for sub in range(CELL_ROWS):
                    neuron = c * CELL_ROWS + sub
                    cp = m + (f"experts.{e}.w1" if neuron < inter
                              else f"experts.{e}.w3")
                    n_loc = neuron if neuron < inter else neuron - inter
                    pay, _ = interleave_expect(
                        oracle, cp, e, t0 + t_loc, n_loc, latent,
                        latent // 2, latent // GROUP)
                    expected += pay
                for sub in range(CELL_ROWS):
                    neuron = c * CELL_ROWS + sub
                    cp = m + (f"experts.{e}.w1" if neuron < inter
                              else f"experts.{e}.w3")
                    n_loc = neuron if neuron < inter else neuron - inter
                    _, sc = interleave_expect(
                        oracle, cp, e, t0 + t_loc, n_loc, latent,
                        latent // 2, latent // GROUP)
                    expected += sc
                oracle.compare(f"{p}expert_w1[e={e} t={t0 + t_loc} "
                               f"c={c}]", bytes(expected), got,
                               "expert_w1")
    entry = oracle.pack.entry(p + "expert_w2_weight")
    geom = entry["interleave"]
    k_tiles = inter // tile_k
    cells_full = latent // CELL_ROWS
    if cells_full % degree:
        raise OracleFailure(f"{p}w2 {cells_full} cells split {degree}")
    take_c = cells_full // degree
    c0 = take_c * rank
    rpe_rank = k_tiles * take_c * (CELL_ROWS + 1)
    if entry["bytes"] != rpe_rank * 64 * n_experts:
        raise OracleFailure(f"{p}expert_w2_weight {entry['bytes']} B, "
                            f"rank grid prices {rpe_rank * 64 * n_experts}")
    tiles = sorted(set(min(t, k_tiles - 1) for t in
                       (0, k_tiles // 2, k_tiles - 1)))[:oracle.tile_samples]
    for e in experts:
        for t in tiles:
            for c_loc in oracle.rng.sample(range(take_c),
                                           2 if deep else 1):
                base = e * rpe_rank + (t * take_c + c_loc) \
                    * (CELL_ROWS + 1)
                got = oracle.pack.span(p + "expert_w2_weight",
                                       base * 64, (CELL_ROWS + 1) * 64)
                expected = bytearray()
                neuron0 = (c0 + c_loc) * CELL_ROWS
                for sub in range(CELL_ROWS):
                    pay, _ = interleave_expect(
                        oracle, m + f"experts.{e}.w2", e, t, neuron0 + sub,
                        inter, inter // 2, inter // GROUP)
                    expected += pay
                for sub in range(CELL_ROWS):
                    _, sc = interleave_expect(
                        oracle, m + f"experts.{e}.w2", e, t, neuron0 + sub,
                        inter, inter // 2, inter // GROUP)
                    expected += sc
                oracle.compare(f"{p}expert_w2[e={e} t={t} "
                               f"c={c0 + c_loc}]", bytes(expected), got,
                               "expert_w2")


def check_moe(oracle, layer, degree, rank, deep):
    cfg = oracle.cfg
    hidden, latent, inter = cfg["hidden"], cfg["latent"], cfg["intermediate"]
    shared = cfg["shared"]
    n_experts = cfg["experts"]
    p = f"model.layers.{layer}."
    m = f"language_model.model.layers.{layer}.block_sparse_moe."
    oracle.check_row_split(p + "router_weight", m + "gate.weight",
                           n_experts, 1, 0, 6 if deep else 2, "router")
    dtype, _, _ = oracle.cp.info(m + "gate.e_score_correction_bias")
    got = oracle.pack.span(p + "router_bias")
    expected = oracle.cp.raw(m + "gate.e_score_correction_bias")
    if dtype not in ("F32", "BF16"):
        raise OracleFailure(f"{p}router bias dtype {dtype}")
    oracle.compare(p + "router_bias", expected, got, "router")
    half_rows = shared
    lo, hi = oracle.rank_block_rows(half_rows, degree, rank)
    gate = oracle.cp.raw(m + "shared_experts.gate_proj.weight")
    up = oracle.cp.raw(m + "shared_experts.up_proj.weight")
    row_bytes = hidden * 2
    rank_rows = half_rows // degree
    got, _, picks = oracle.row_picks(p + "shared_w1_weight", rank_rows * 2,
                                     0, rank_rows, 4)
    expected = bytearray()
    for row in picks:
        expected += gate[(lo + row) * row_bytes:(lo + row + 1) * row_bytes]
    got2, _, picks2 = oracle.row_picks(p + "shared_w1_weight", rank_rows * 2,
                                       rank_rows, rank_rows * 2, 4)
    for row in picks2:
        src = row - rank_rows
        expected += up[(lo + src) * row_bytes:(lo + src + 1) * row_bytes]
    oracle.compare(p + "shared_w1_weight", bytes(expected),
                   got + got2, "shared")
    oracle.check_col_split(p + "shared_w2_weight",
                           m + "shared_experts.down_proj.weight",
                           degree, rank, hidden, shared * 2, 4, "shared")
    oracle.check_row_split(p + "routed_down_weight",
                           m + "routed_expert_down_proj.weight", latent,
                           degree, rank, 4, "routed")
    oracle.check_col_split(p + "routed_up_weight",
                           m + "routed_expert_up_proj.weight",
                           degree, rank, hidden, latent * 2, 4, "routed")
    got = oracle.pack.span(p + "routed_norm_weight")
    oracle.compare(p + "routed_norm_weight",
                   oracle.cp.raw(m + "routed_expert_norm.weight"), got,
                   "norm")


def check_dense(oracle, layer, degree, rank):
    cfg = oracle.cfg
    hidden = cfg["hidden"]
    p = f"model.layers.{layer}."
    m = f"language_model.model.layers.{layer}.mlp."
    _, _, gate_bytes = oracle.cp.info(m + "gate_proj.weight")
    half_rows = gate_bytes // (hidden * 2)
    lo, hi = oracle.rank_block_rows(half_rows, degree, rank)
    gate = oracle.cp.raw(m + "gate_proj.weight")
    up = oracle.cp.raw(m + "up_proj.weight")
    row_bytes = hidden * 2
    rank_rows = half_rows // degree
    got, _, picks = oracle.row_picks(p + "dense_gate_up_weight",
                                     rank_rows * 2, 0, rank_rows, 4)
    expected = bytearray()
    for row in picks:
        expected += gate[(lo + row) * row_bytes:(lo + row + 1) * row_bytes]
    got2, _, picks2 = oracle.row_picks(p + "dense_gate_up_weight",
                                       rank_rows * 2, rank_rows,
                                       rank_rows * 2, 4)
    for row in picks2:
        src = row - rank_rows
        expected += up[(lo + src) * row_bytes:(lo + src + 1) * row_bytes]
    oracle.compare(p + "dense_gate_up_weight", bytes(expected),
                   got + got2, "dense")
    _, _, down_bytes = oracle.cp.info(m + "down_proj.weight")
    in_cols = down_bytes // (hidden * 2)
    oracle.check_col_split(p + "dense_down_weight", m + "down_proj.weight",
                           degree, rank, hidden, in_cols * 2, 4, "dense")


def check_vocab(oracle, name, cp_name, vocab, hidden, degree, rank):
    lo, hi = oracle.rank_block_rows(vocab, degree, rank)
    span = hi - lo
    got, _, picks = oracle.row_picks(name, span, 0, span, 8)
    expected = oracle.cp_rows(cp_name, hidden * 2, [lo + p for p in picks])
    oracle.compare(name, expected, got, "vocab")


def check_globals(oracle, cp_names, degree, rank, hidden, first, count,
                  total):
    cfg = oracle.cfg
    vocab = cfg["vocab"]
    if first == 0:
        check_vocab(oracle, "model.embed_tokens.weight",
                    "language_model.model.embed_tokens.weight", vocab,
                    hidden, degree, rank)
    if first + count == total:
        got = oracle.pack.span("model.norm.weight")
        oracle.compare("model.norm.weight",
                       oracle.cp.raw("language_model.model.norm.weight"),
                       got, "norm")
        proj = oracle.cp.raw("language_model.model.output_attn_res_proj.weight")
        gamma = oracle.cp.raw("language_model.model.output_attn_res_norm.weight")
        oracle.compare("model.attnres_out_weight",
                       gamma_fold(proj, gamma, hidden),
                       oracle.pack.span("model.attnres_out_weight"),
                       "gamma_fold")
        check_vocab(oracle, "lm_head.weight",
                    "language_model.lm_head.weight", vocab, hidden, degree,
                    rank)


def check_config_echo(oracle, cp):
    path = os.path.join(cp.model_dir, "config.json")
    with open(path, "rb") as handle:
        config = json.loads(handle.read())
    if isinstance(config.get("text_config"), dict):
        config = config["text_config"]
    cfg = oracle.cfg
    lac = config.get("linear_attn_config", {})
    top_k = config.get("num_experts_per_tok")
    if top_k is None:
        top_k = config.get("num_experts_per_token")
    want = {
        "hidden": config["hidden_size"],
        "total_layers": config["num_hidden_layers"],
        "experts": config["num_experts"],
        "top_k": top_k,
        "latent": config["routed_expert_hidden_size"],
        "intermediate": config["moe_intermediate_size"],
        "shared": config.get("num_shared_experts", 1)
        * config["moe_intermediate_size"],
        "vocab": config["vocab_size"],
        "kda_heads": lac.get("num_heads", config["num_attention_heads"]),
        "kda_head": lac.get("head_dim", config.get("head_dim")),
        "heads": config["num_attention_heads"],
        "kv_lora": config["kv_lora_rank"],
        "rope": config["qk_rope_head_dim"],
        "v_head": config["v_head_dim"],
        "nope": config["qk_nope_head_dim"],
        "q_lora": config["q_lora_rank"],
    }
    problems = [f"{k}: pack {cfg.get(k)}, checkpoint {v}"
                for k, v in want.items() if cfg.get(k) != v]
    if problems:
        raise OracleFailure("config echo drift: " + "; ".join(problems))


def run_oracle(args):
    pack = PackView(args.pack)
    cp = Checkpoint(args.checkpoint)
    oracle = Oracle(pack, cp, args.seed, args.experts, args.tiles)
    check_config_echo(oracle, cp)
    cfg = pack.config
    hidden = cfg["hidden"]
    degree = cfg.get("tp_degree", 1)
    rank = cfg.get("tp_rank", 0)
    first, count, total = cfg["first_layer"], cfg["layers"], cfg["total_layers"]
    if first + count > total or count <= 0:
        raise OracleFailure(f"pack slice {first}+{count} of {total}")
    lac_path = os.path.join(cp.model_dir, "config.json")
    with open(lac_path, "rb") as handle:
        raw_cfg = json.loads(handle.read())
    if isinstance(raw_cfg.get("text_config"), dict):
        raw_cfg = raw_cfg["text_config"]
    kda_layers = set(raw_cfg.get("linear_attn_config", {}).get(
        "kda_layers", []))
    first_dense = raw_cfg.get("first_k_dense_replace", 1)
    cp_names = cp.names()
    layers = list(range(first, first + count))
    deep = {layers[0]}
    deep_kda = deep_mla = None
    for layer in layers:
        attn = "kda" if (layer + 1) in kda_layers else "mla"
        if deep_kda is None and attn == "kda":
            deep_kda = layer
        if deep_mla is None and attn == "mla":
            deep_mla = layer
    deep.update(x for x in (deep_kda, deep_mla) if x is not None)
    for layer in layers:
        attn = "kda" if (layer + 1) in kda_layers else "mla"
        is_deep = layer in deep
        sp = f"language_model.model.layers.{layer}."
        moe = f"{sp}block_sparse_moe.gate.weight" in cp_names \
            and layer >= first_dense
        check_common(oracle, layer, degree, rank, hidden)
        if attn == "kda":
            check_kda(oracle, layer, degree, rank, is_deep)
        else:
            check_mla(oracle, layer, degree, rank, is_deep)
        if moe:
            check_experts(oracle, layer, degree, rank, True, is_deep)
            check_moe(oracle, layer, degree, rank, is_deep)
        else:
            check_dense(oracle, layer, degree, rank)
    check_globals(oracle, cp_names, degree, rank, hidden, first, count,
                  total)
    result = {
        "pack": pack.path, "checkpoint": cp.model_dir,
        "tp_degree": degree, "tp_rank": rank, "first_layer": first,
        "layers": count, "seed": args.seed,
        "checks": oracle.checks,
        "bytes_checked": oracle.bytes_checked,
        "mismatches": oracle.mismatches,
        "per_class": oracle.per_class,
        "lines": [],
    }
    for mismatch in oracle.mismatches[:10]:
        result["lines"].append(
            f"  MISMATCH {mismatch['tensor']} @ {mismatch['offset']}: "
            f"expected {mismatch['expected']} got {mismatch['got']}")
    result["verdict"] = "pass" if not oracle.mismatches else "fail"
    if args.digest:
        digest = hashlib.sha256()
        pack.handle.seek(0)
        while True:
            chunk = pack.handle.read(1 << 24)
            if not chunk:
                break
            digest.update(chunk)
        result["pack_sha256"] = digest.hexdigest()
    result["pack_bytes"] = pack.file_bytes
    pack.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--experts", type=int, default=3)
    parser.add_argument("--tiles", type=int, default=3)
    parser.add_argument("--digest", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        verdict = run_oracle(args)
    except OracleFailure as failure:
        print(f"K3 ORACLE ERROR: {failure}")
        return 2
    if args.json:
        print(json.dumps(verdict, indent=1))
    else:
        for line in verdict["lines"]:
            print(line)
        print(f"K3 ORACLE {verdict['verdict'].upper()}: {verdict['pack']} "
              f"rank={verdict['tp_rank']}/{verdict['tp_degree']} "
              f"layers {verdict['first_layer']}+{verdict['layers']}")
        print(f"  checks={verdict['checks']} "
              f"bytes={verdict['bytes_checked']} "
              f"mismatches={len(verdict['mismatches'])} "
              f"sha256={verdict.get('pack_sha256', '-')} "
              f"{verdict['pack_bytes']} B")
    return 0 if verdict["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
