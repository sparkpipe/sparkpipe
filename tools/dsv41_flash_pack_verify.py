#!/usr/bin/env python3
"""Independent real-source pack verifier for DeepSeek-V4.1-Flash stagepacks.

Re-derives the expected per-rank pack layout and byte content straight from the
warm checkpoint headers and the pinned geometry facts (own tables, no converter
imports), then proves every pack byte against the checkpoint:

  pack header fields vs literals           pack == format
  directory records vs derived layout      pack == packer
  per-layer kind sets vs literal bits      pack == kind map
  every payload/scale byte vs checkpoint   pack == checkpoint
  layout agreement vs plan.json            converter == verifier

usage:
  verify HEADERS_JSON OUT_DIR WARM_DIR PLAN_JSON [LO HI]
Exit 0 = all ranks verified (optionally restricted to source shards [LO,HI]).
"""

from __future__ import annotations

import json
import os
import struct
import sys

SHARD_NAMING = "model-%05d-of-%05d.safetensors"
SHARD_COUNT = 48
LAYER_COUNT = 40
HIDDEN = 5120
VOCAB = 129280
ROUTED_EXPERTS = 384
EXPERT_WIDTH = 2304
KV_SOURCE_LAYERS = (2, 8, 14, 20)
GATE_LAYERS = (2, 8, 14)
INDEX_SOURCE_LAYERS = (2, 8, 14, 20, 24, 28, 32, 36)
SWA_ONLY_LAYERS = (0, 1, 38, 39)
GLOBAL_LAYER = 0xFFFFFFFF
HEADER_BYTES = 257
ENTRY_BYTES = 64
ALIGN = 256
WINDOW = 16 << 20
MAGIC = 0x31413444

DTYPE_BYTES = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1}

K_EMBEDDING = 0
K_FINAL_NORM = 1
K_LM_HEAD = 2
K_ATTN_NORM = 3
K_FFN_NORM = 4
K_Q_A = 5
K_Q_B = 6
K_KV_A = 7
K_Q_NORM = 8
K_KV_NORM = 9
K_ATTN_SINK = 10
K_O_A = 11
K_O_B = 12
K_IDX_Q_B = 13
K_IDX_WK = 14
K_IDX_WP = 15
K_IDX_KN = 16
K_C_WKV = 17
K_C_WGATE = 18
K_C_NORM = 19
K_HC_A_FN = 20
K_HC_A_BASE = 21
K_HC_A_SCALE = 22
K_HC_F_FN = 23
K_HC_F_BASE = 24
K_HC_F_SCALE = 25
K_ROUTER = 26
K_ROUTER_BIAS = 27
K_ROUTER_BIAS_VL = 28
K_EXP_W1 = 29
K_EXP_W2 = 30
K_EXP_W3 = 31
K_SH_W1 = 32
K_SH_W2 = 33
K_SH_W3 = 34
KIND_COUNT = 35
EXPERT_KINDS = (K_EXP_W1, K_EXP_W2, K_EXP_W3)
FP8_KINDS = (K_Q_A, K_Q_B, K_KV_A, K_O_A, K_O_B, K_IDX_Q_B, K_SH_W1, K_SH_W2, K_SH_W3)
F32_KINDS = (K_ATTN_SINK, K_HC_A_FN, K_HC_A_BASE, K_HC_A_SCALE,
             K_HC_F_FN, K_HC_F_BASE, K_HC_F_SCALE, K_ROUTER_BIAS, K_ROUTER_BIAS_VL)
ATTENTION_KINDS = tuple(range(K_ATTN_NORM, K_O_B + 1))
HC_KINDS = tuple(range(K_HC_A_FN, K_HC_F_SCALE + 1))
ROUTER_KINDS = (K_ROUTER, K_ROUTER_BIAS, K_ROUTER_BIAS_VL)
SHARED_KINDS = (K_SH_W1, K_SH_W2, K_SH_W3)


class Fail(Exception):
    pass


def fail(message: str):
    raise Fail(message)


def normalize(shape: list) -> list:
    return shape if len(shape) == 2 else [1, shape[0]]


def in_layer(kind: int, layer: int) -> bool:
    if kind <= K_LM_HEAD:
        return False
    if kind in (K_IDX_Q_B, K_IDX_WK, K_IDX_WP, K_IDX_KN):
        if layer in SWA_ONLY_LAYERS:
            return False
        if kind in (K_IDX_WK, K_IDX_KN):
            return layer in KV_SOURCE_LAYERS
        return layer in INDEX_SOURCE_LAYERS
    if kind in (K_C_WKV, K_C_WGATE, K_C_NORM):
        return layer in GATE_LAYERS if kind == K_C_WGATE else layer in KV_SOURCE_LAYERS
    return True


def layer_kind_bits(layer: int) -> int:
    bits = 0
    for kind in ATTENTION_KINDS + HC_KINDS + ROUTER_KINDS + SHARED_KINDS + EXPERT_KINDS:
        bits |= 1 << kind
    if layer in INDEX_SOURCE_LAYERS and layer not in SWA_ONLY_LAYERS:
        bits |= (1 << K_IDX_Q_B) | (1 << K_IDX_WP)
    if layer in KV_SOURCE_LAYERS:
        bits |= (1 << K_IDX_WK) | (1 << K_IDX_KN) | (1 << K_C_WKV) | (1 << K_C_NORM)
    if layer in GATE_LAYERS:
        bits |= 1 << K_C_WGATE
    return bits


def entry_shape(kind: int, tp: int):
    if kind in (K_EMBEDDING, K_LM_HEAD):
        return (VOCAB // tp, HIDDEN)
    if kind in (K_ATTN_NORM, K_FFN_NORM, K_FINAL_NORM):
        return (1, HIDDEN)
    if kind == K_Q_A:
        return (1280, HIDDEN)
    if kind == K_Q_B:
        return (32768 // tp, 1280)
    if kind == K_KV_A:
        return (512, HIDDEN)
    if kind == K_Q_NORM:
        return (1, 1280)
    if kind == K_KV_NORM:
        return (1, 512)
    if kind == K_ATTN_SINK:
        return (1, 64 // tp)
    if kind == K_O_A:
        return (8192 // tp, 4096 // tp)
    if kind == K_O_B:
        return (5120, 8192 // tp)
    if kind == K_IDX_Q_B:
        return (4096 // tp, 1280)
    if kind == K_IDX_WK:
        return (128, 512)
    if kind == K_IDX_WP:
        return (32 // tp, 5120)
    if kind == K_IDX_KN:
        return (1, 128)
    if kind in (K_C_WKV, K_C_WGATE):
        return (512, HIDDEN)
    if kind == K_C_NORM:
        return (1, 512)
    if kind in (K_HC_A_FN, K_HC_F_FN):
        return (24, 20480)
    if kind in (K_HC_A_BASE, K_HC_F_BASE):
        return (1, 24)
    if kind in (K_HC_A_SCALE, K_HC_F_SCALE):
        return (1, 3)
    if kind == K_ROUTER:
        return (384, HIDDEN)
    if kind in (K_ROUTER_BIAS, K_ROUTER_BIAS_VL):
        return (1, 384)
    if kind in EXPERT_KINDS:
        return (5120, EXPERT_WIDTH // 2) if kind == K_EXP_W2 else (EXPERT_WIDTH, HIDDEN // 2)
    if kind == K_SH_W2:
        return (HIDDEN, EXPERT_WIDTH)
    return (EXPERT_WIDTH, HIDDEN)


def expected_entry(kind: int, layer: int, tp: int) -> dict:
    rows, cols = entry_shape(kind, tp)
    if kind in EXPERT_KINDS:
        groups = ROUTED_EXPERTS // tp
        return {"kind": kind, "layer": layer, "payload_type": 4, "codec": 7, "scale_encoding": 3,
                "groups": groups, "rows": rows, "cols": cols,
                "payload_bytes": rows * cols * groups,
                "scale_bytes": rows * (cols * 2 // 32) * groups}
    if kind in FP8_KINDS:
        return {"kind": kind, "layer": layer, "payload_type": 4, "codec": 5, "scale_encoding": 3,
                "groups": 1, "rows": rows, "cols": cols,
                "payload_bytes": rows * cols, "scale_bytes": (rows // 32) * (cols // 32)}
    if kind in F32_KINDS:
        return {"kind": kind, "layer": layer, "payload_type": 2, "codec": 0, "scale_encoding": 0,
                "groups": 1, "rows": rows, "cols": cols,
                "payload_bytes": rows * cols * 4, "scale_bytes": 0}
    return {"kind": kind, "layer": layer, "payload_type": 1, "codec": 1, "scale_encoding": 0,
            "groups": 1, "rows": rows, "cols": cols,
            "payload_bytes": rows * cols * 2, "scale_bytes": 0}


def source_ref(kind: int, layer: int, plane: int):
    a = f"layers.{layer}.attn."
    f = f"layers.{layer}.ffn."
    suffix = ("weight", "scale")[plane]
    if kind == K_EMBEDDING:
        return ("embed.weight", "rows")
    if kind == K_FINAL_NORM:
        return ("norm.weight", "repl")
    if kind == K_LM_HEAD:
        return ("head.weight", "rows")
    if kind == K_ATTN_NORM:
        return (f"layers.{layer}.attn_norm.weight", "repl")
    if kind == K_FFN_NORM:
        return (f"layers.{layer}.ffn_norm.weight", "repl")
    if kind == K_Q_A:
        return (a + "wq_a." + suffix, "repl")
    if kind == K_Q_B:
        return (a + "wq_b." + suffix, "rows")
    if kind == K_KV_A:
        return (a + "wkv." + suffix, "repl")
    if kind == K_Q_NORM:
        return (a + "q_norm.weight", "repl")
    if kind == K_KV_NORM:
        return (a + "kv_norm.weight", "repl")
    if kind == K_ATTN_SINK:
        return (a + "attn_sink", "cols")
    if kind == K_O_A:
        return (a + "wo_a." + suffix, "diag")
    if kind == K_O_B:
        return (a + "wo_b." + suffix, "cols")
    if kind == K_IDX_Q_B:
        return (a + "indexer.wq_b." + suffix, "rows")
    if kind == K_IDX_WK:
        return (a + "indexer.wk.weight", "repl")
    if kind == K_IDX_WP:
        return (a + "indexer.weights_proj.weight", "rows")
    if kind == K_IDX_KN:
        return (a + "indexer.k_norm.weight", "repl")
    if kind == K_C_WKV:
        return (a + "compressor.wkv.weight", "repl")
    if kind == K_C_WGATE:
        return (a + "compressor.wgate.weight", "repl")
    if kind == K_C_NORM:
        return (a + "compressor.norm.weight", "repl")
    if kind == K_HC_A_FN:
        return (f"layers.{layer}.hc_attn_fn", "repl")
    if kind == K_HC_A_BASE:
        return (f"layers.{layer}.hc_attn_base", "repl")
    if kind == K_HC_A_SCALE:
        return (f"layers.{layer}.hc_attn_scale", "repl")
    if kind == K_HC_F_FN:
        return (f"layers.{layer}.hc_ffn_fn", "repl")
    if kind == K_HC_F_BASE:
        return (f"layers.{layer}.hc_ffn_base", "repl")
    if kind == K_HC_F_SCALE:
        return (f"layers.{layer}.hc_ffn_scale", "repl")
    if kind == K_ROUTER:
        return (f + "gate.weight", "repl")
    if kind == K_ROUTER_BIAS:
        return (f + "gate.bias", "repl")
    if kind == K_ROUTER_BIAS_VL:
        return (f + "gate.bias_vl", "repl")
    if kind in EXPERT_KINDS:
        w = "w2" if kind == K_EXP_W2 else ("w1" if kind == K_EXP_W1 else "w3")
        return (f + "experts.{e}." + w + "." + suffix, "expert")
    if kind == K_SH_W1:
        return (f + "shared_experts.w1." + suffix, "repl")
    if kind == K_SH_W2:
        return (f + "shared_experts.w2." + suffix, "repl")
    if kind == K_SH_W3:
        return (f + "shared_experts.w3." + suffix, "repl")
    fail(f"unknown kind {kind}")


def derive_layout(tp: int):
    order = [(K_EMBEDDING, GLOBAL_LAYER), (K_FINAL_NORM, GLOBAL_LAYER), (K_LM_HEAD, GLOBAL_LAYER)]
    for layer in range(LAYER_COUNT):
        for kind in range(K_ATTN_NORM, KIND_COUNT):
            if in_layer(kind, layer):
                order.append((kind, layer))
    directory_offset = (HEADER_BYTES + ALIGN - 1) & ~(ALIGN - 1)
    cursor = (directory_offset + len(order) * ENTRY_BYTES + ALIGN - 1) & ~(ALIGN - 1)
    entries = []
    for kind, layer in order:
        entry = expected_entry(kind, layer, tp)
        cursor = (cursor + ALIGN - 1) & ~(ALIGN - 1)
        entry["payload_offset"] = cursor
        cursor += entry["payload_bytes"]
        if entry["scale_bytes"]:
            cursor = (cursor + ALIGN - 1) & ~(ALIGN - 1)
            entry["scale_offset"] = cursor
            cursor += entry["scale_bytes"]
        else:
            entry["scale_offset"] = 0
        entries.append(entry)
    return order, entries, cursor


def pieces_for(flat: dict, tp: int, order: list, entries: list, lo: int, hi: int) -> list:
    pieces = []
    for entry, (kind, layer) in zip(entries, order):
        for plane, offset, total in ((0, entry["payload_offset"], entry["payload_bytes"]),
                                     (1, entry["scale_offset"], entry["scale_bytes"])):
            if total == 0:
                continue
            name, mode = source_ref(kind, layer, plane)
            if mode == "expert":
                groups = ROUTED_EXPERTS // tp
                per = total // groups
                for rank in range(tp):
                    for g in range(groups):
                        src = flat[name.format(e=rank * groups + g)]
                        pieces.append((rank, kind, layer, plane, src, 0, src["shape"][0],
                                       0, src["shape"][1], offset + g * per, per))
                continue
            src0 = flat[name]
            rows, cols = src0["shape"]
            for rank in range(tp):
                if mode == "repl":
                    win = (0, rows, 0, cols)
                elif mode == "rows":
                    step = rows // tp
                    win = (rank * step, (rank + 1) * step, 0, cols)
                elif mode == "cols":
                    step = cols // tp
                    win = (0, rows, rank * step, (rank + 1) * step)
                else:
                    win = (rank * (rows // tp), (rank + 1) * (rows // tp),
                           rank * (cols // tp), (rank + 1) * (cols // tp))
                pieces.append((rank, kind, layer, plane, src0, win[0], win[1], win[2], win[3],
                               offset, total))
    return [p for p in pieces if lo <= p[4]["shard"] <= hi]


def compare_piece(src_fd: int, dst_fd: int, piece: tuple) -> int:
    _, kind, layer, plane, src, r0, r1, c0, c1, dst_offset, total = piece
    esz = DTYPE_BYTES[src["dtype"]]
    row_bytes = src["shape"][1] * esz
    span = (c1 - c0) * esz
    done = 0
    row = r0
    while row < r1:
        block = min(max(WINDOW // row_bytes, 1), r1 - row)
        want = block * row_bytes
        raw = os.pread(src_fd, want, src["begin"] + row * row_bytes)
        if len(raw) != want:
            fail(f"short source read {src['name']} row {row}")
        pack = os.pread(dst_fd, block * span, dst_offset + done)
        if len(pack) != block * span:
            fail(f"short pack read kind {kind} layer {layer} rank offset {dst_offset + done}")
        for index in range(block):
            base = index * row_bytes
            if raw[base + c0 * esz:base + c1 * esz] != pack[index * span:(index + 1) * span]:
                fail(f"BYTE MISMATCH {src['name']} row {row + index} kind {kind} "
                     f"layer {layer} plane {plane}")
        done += block * span
        row += block
    if done != total:
        fail(f"piece span {done} != entry bytes {total} for {src['name']}")
    return done


def verify_header(fd: int, rank: int, tp: int, plan: dict, entries: list,
                  directory_offset: int, file_bytes: int) -> None:
    raw = os.pread(fd, HEADER_BYTES, 0)
    if len(raw) != HEADER_BYTES:
        fail(f"rank{rank}: short header")
    fields = struct.unpack_from("<20I2Q", raw, 0)
    literal = (MAGIC, 1, HEADER_BYTES, ENTRY_BYTES, 1, 0, len(entries), 1, 0, 0,
               LAYER_COUNT, LAYER_COUNT, HIDDEN, VOCAB, ROUTED_EXPERTS, 5, 7, tp, rank, 0)
    if fields[:20] != literal:
        fail(f"rank{rank}: header fields {fields[:20]} != literals {literal}")
    if fields[20] != directory_offset or fields[21] != file_bytes:
        fail(f"rank{rank}: directory/file bytes {fields[20:22]} != "
             f"{(directory_offset, file_bytes)}")
    if raw[96:161].rstrip(b"\0").decode() != plan["revision"]:
        fail(f"rank{rank}: model revision mismatch")
    for slot, key in ((161, "contract_sha256"), (193, "config_sha256"), (225, "recipe_sha256")):
        if raw[slot:slot + 32] != bytes.fromhex(plan[key]):
            fail(f"rank{rank}: digest {key} mismatch")
    if os.fstat(fd).st_size != file_bytes:
        fail(f"rank{rank}: size {os.fstat(fd).st_size} != header file_bytes {file_bytes}")


def verify_directory(fd: int, rank: int, entries: list, directory_offset: int) -> None:
    for index, entry in enumerate(entries):
        raw = os.pread(fd, ENTRY_BYTES, directory_offset + index * ENTRY_BYTES)
        got = struct.unpack("<8I4Q", raw)
        want = (entry["kind"], entry["layer"], entry["payload_type"], entry["codec"],
                entry["scale_encoding"], entry["groups"], entry["rows"], entry["cols"],
                entry["payload_offset"], entry["payload_bytes"], entry["scale_offset"],
                entry["scale_bytes"])
        if got != want:
            fail(f"rank{rank}: directory record {index} kind {entry['kind']} "
                 f"layer {entry['layer']}: got {got} want {want}")


def cmd_verify(headers_json: str, out_dir: str, warm: str, plan_path: str,
               lo: int, hi: int) -> int:
    with open(plan_path, encoding="utf-8") as handle:
        plan = json.load(handle)
    tp = plan["tp"]
    with open(headers_json, encoding="utf-8") as handle:
        document = json.load(handle)
    flat = {}
    for shard, shard_entries in document["shards"].items():
        for name, meta in shard_entries.items():
            meta = dict(meta)
            meta["shard"] = int(shard)
            meta["shape"] = normalize(meta["shape"])
            meta["name"] = name
            flat[name] = meta
    order, entries, file_bytes = derive_layout(tp)
    if len(entries) != plan["tensor_count"] or file_bytes != plan["file_bytes"]:
        fail(f"derived layout {len(entries)}/{file_bytes} != plan "
             f"{plan['tensor_count']}/{plan['file_bytes']}")
    for derived, recorded in zip(entries, plan["entries"]):
        for key in ("kind", "layer", "payload_type", "codec", "scale_encoding", "groups",
                    "rows", "cols", "payload_bytes", "scale_bytes", "payload_offset",
                    "scale_offset"):
            if derived[key] != recorded[key]:
                fail(f"layout divergence kind {derived['kind']} layer {derived['layer']} "
                     f"field {key}: {derived[key]} != {recorded[key]}")
    spine_total = sum(e["payload_bytes"] + e["scale_bytes"] for e in entries
                      if e["kind"] not in EXPERT_KINDS)
    expert_total = sum(e["payload_bytes"] + e["scale_bytes"] for e in entries
                       if e["kind"] in EXPERT_KINDS)
    if (spine_total, expert_total) != (plan["spine_bytes"], plan["expert_bytes"]):
        fail("spine/expert accounting diverges from plan")
    bits = {}
    for entry in entries:
        if entry["layer"] == GLOBAL_LAYER:
            continue
        if bits.get(entry["layer"], 0) & (1 << entry["kind"]):
            fail(f"duplicate kind {entry['kind']} layer {entry['layer']}")
        bits[entry["layer"]] = bits.get(entry["layer"], 0) | (1 << entry["kind"])
    for layer in range(LAYER_COUNT):
        if bits.get(layer, 0) != layer_kind_bits(layer):
            fail(f"layer {layer} kind bits {bits.get(layer, 0):x} != literal "
                 f"{layer_kind_bits(layer):x}")
    directory_offset = (HEADER_BYTES + ALIGN - 1) & ~(ALIGN - 1)
    dst_fds = [os.open(os.path.join(out_dir, f"rank{rank}.spstage"), os.O_RDONLY)
               for rank in range(tp)]
    src_fds = {}
    compared = 0
    try:
        for rank in range(tp):
            verify_header(dst_fds[rank], rank, tp, plan, entries, directory_offset, file_bytes)
        verify_directory(dst_fds[0], 0, entries, directory_offset)
        verify_directory(dst_fds[tp - 1], tp - 1, entries, directory_offset)
        for piece in pieces_for(flat, tp, order, entries, lo, hi):
            shard = piece[4]["shard"]
            if shard not in src_fds:
                src_fds[shard] = os.open(
                    os.path.join(warm, SHARD_NAMING % (shard, SHARD_COUNT)), os.O_RDONLY)
            compared += compare_piece(src_fds[shard], dst_fds[piece[0]], piece)
    finally:
        for fd in dst_fds:
            os.close(fd)
        for fd in src_fds.values():
            os.close(fd)
    with open(os.path.join(out_dir, f"verify_{lo:02d}_{hi:02d}.done"), "w",
              encoding="utf-8") as handle:
        handle.write(f"{compared}\n")
    print(f"dsv41 pack verify ok: shards [{lo},{hi}] tp{tp}: header+directory literals, "
          f"kind census, {compared} bytes == checkpoint")
    return 0


def main(argv: list) -> int:
    if len(argv) not in (6, 8):
        print(__doc__)
        return 2
    try:
        return cmd_verify(argv[2], argv[3], argv[4], argv[5],
                          int(argv[6]) if len(argv) == 8 else 1,
                          int(argv[7]) if len(argv) == 8 else SHARD_COUNT)
    except Fail as error:
        print(f"dsv41_flash_pack_verify FAIL: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"dsv41_flash_pack_verify OS FAIL: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
