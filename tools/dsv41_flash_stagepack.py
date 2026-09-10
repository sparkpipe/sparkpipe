#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash warm-checkpoint-to-stagepack converter (TP-parameterized).

subcommands:
  headers WARM_DIR INDEX_JSON OUT_JSON
      read all safetensors shard headers, cross-check against the index weight_map
  plan HEADERS_JSON OUT_DIR TP REVISION CONTRACT_SHA CONFIG_SHA RECIPE_SHA
      validate the checkpoint against the kind map, write pack skeletons + plan.json
  copy HEADERS_JSON PLAN_JSON OUT_DIR WARM_DIR LO HI
      copy payload contributed by source shards [LO,HI] into every rank pack
Exit 0 = step complete; nonzero = the failure is named on stderr.
"""

from __future__ import annotations

import json
import os
import struct
import sys

SHARD_COUNT = 48
SHARD_NAMING = "model-%05d-of-%05d.safetensors"
LAYER_COUNT = 40
HIDDEN = 5120
VOCAB = 129280
ROUTED_EXPERTS = 384
EXPERT_WIDTH = 2304
KV_SOURCE_LAYERS = (2, 8, 14, 20)
GATE_LAYERS = (2, 8, 14)
INDEX_SOURCE_LAYERS = (2, 8, 14, 20, 24, 28, 32, 36)
SWA_ONLY_LAYERS = (0, 1, 38, 39)
ENGRAM_TENSOR_COUNT = 12
GLOBAL_LAYER = 0xFFFFFFFF
HEADER_BYTES = 257
ENTRY_BYTES = 64
ALIGN = 256
COPY_BUDGET = 16 << 20
MODEL_REVISION_BYTES = 65
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

PAYLOAD_BF16 = 1
PAYLOAD_F32 = 2
PAYLOAD_PACKED = 4
CODEC_FP8 = 5
CODEC_MXFP4 = 7
SCALE_NONE = 0
SCALE_E8M0 = 3


class Fail(Exception):
    pass


def fail(message: str):
    raise Fail(message)


def is_global(kind: int) -> bool:
    return kind <= K_LM_HEAD


def kind_in_layer(kind: int, layer: int) -> bool:
    if is_global(kind):
        return False
    if kind in (K_IDX_Q_B, K_IDX_WK, K_IDX_WP, K_IDX_KN):
        if layer in SWA_ONLY_LAYERS:
            return False
        if kind in (K_IDX_WK, K_IDX_KN):
            return layer in KV_SOURCE_LAYERS
        return layer in INDEX_SOURCE_LAYERS
    if kind in (K_C_WKV, K_C_WGATE, K_C_NORM):
        if kind == K_C_WGATE:
            return layer in GATE_LAYERS
        return layer in KV_SOURCE_LAYERS
    return True


def entry_shape(kind: int, tp: int):
    if kind in (K_EMBEDDING, K_LM_HEAD):
        return (VOCAB // tp, HIDDEN, 1)
    if kind in (K_ATTN_NORM, K_FFN_NORM, K_FINAL_NORM):
        return (1, HIDDEN, 1)
    if kind == K_Q_A:
        return (1280, HIDDEN, 1)
    if kind == K_Q_B:
        return (32768 // tp, 1280, 1)
    if kind == K_KV_A:
        return (512, HIDDEN, 1)
    if kind == K_Q_NORM:
        return (1, 1280, 1)
    if kind == K_KV_NORM:
        return (1, 512, 1)
    if kind == K_ATTN_SINK:
        return (1, 64 // tp, 1)
    if kind == K_O_A:
        return (8192 // tp, 4096 // tp, 1)
    if kind == K_O_B:
        return (5120, 8192 // tp, 1)
    if kind == K_IDX_Q_B:
        return (4096 // tp, 1280, 1)
    if kind == K_IDX_WK:
        return (128, 512, 1)
    if kind == K_IDX_WP:
        return (32 // tp, 5120, 1)
    if kind == K_IDX_KN:
        return (1, 128, 1)
    if kind in (K_C_WKV, K_C_WGATE):
        return (512, HIDDEN, 1)
    if kind == K_C_NORM:
        return (1, 512, 1)
    if kind in (K_HC_A_FN, K_HC_F_FN):
        return (24, 20480, 1)
    if kind in (K_HC_A_BASE, K_HC_F_BASE):
        return (1, 24, 1)
    if kind in (K_HC_A_SCALE, K_HC_F_SCALE):
        return (1, 3, 1)
    if kind == K_ROUTER:
        return (384, HIDDEN, 1)
    if kind in (K_ROUTER_BIAS, K_ROUTER_BIAS_VL):
        return (1, 384, 1)
    if kind in EXPERT_KINDS:
        groups = ROUTED_EXPERTS // tp
        if kind == K_EXP_W2:
            return (5120, EXPERT_WIDTH // 2, groups)
        return (EXPERT_WIDTH, HIDDEN // 2, groups)
    if kind in (K_SH_W1, K_SH_W3):
        return (EXPERT_WIDTH, HIDDEN, 1)
    if kind == K_SH_W2:
        return (HIDDEN, EXPERT_WIDTH, 1)
    fail(f"unknown kind {kind}")


def entry_layout(kind: int, tp: int):
    rows, cols, groups = entry_shape(kind, tp)
    if kind in EXPERT_KINDS:
        return (PAYLOAD_PACKED, CODEC_MXFP4, SCALE_E8M0,
                rows * cols * groups, rows * (cols * 2 // 32) * groups)
    if kind in (K_Q_A, K_Q_B, K_KV_A, K_O_A, K_O_B, K_IDX_Q_B, K_SH_W1, K_SH_W2, K_SH_W3):
        return PAYLOAD_PACKED, CODEC_FP8, SCALE_E8M0, rows * cols, (rows // 32) * (cols // 32)
    if kind in (K_ATTN_SINK, K_HC_A_FN, K_HC_A_BASE, K_HC_A_SCALE,
                K_HC_F_FN, K_HC_F_BASE, K_HC_F_SCALE, K_ROUTER_BIAS, K_ROUTER_BIAS_VL):
        return PAYLOAD_F32, 0, SCALE_NONE, rows * cols * 4, 0
    return PAYLOAD_BF16, 1, SCALE_NONE, rows * cols * 2, 0


def checkpoint_spec(kind: int, layer: int):
    a = f"layers.{layer}.attn."
    f = f"layers.{layer}.ffn."
    if kind == K_ATTN_NORM:
        return (f"layers.{layer}.attn_norm.weight", None, "BF16", [HIDDEN], None, "repl")
    if kind == K_FFN_NORM:
        return (f"layers.{layer}.ffn_norm.weight", None, "BF16", [HIDDEN], None, "repl")
    if kind == K_Q_A:
        shape = [1280, HIDDEN]
        return (a + "wq_a.weight", a + "wq_a.scale", "F8_E4M3", shape,
                fp8_scale_shape(shape), "repl")
    if kind == K_Q_B:
        shape = [32768, 1280]
        return (a + "wq_b.weight", a + "wq_b.scale", "F8_E4M3", shape,
                fp8_scale_shape(shape), "rows")
    if kind == K_KV_A:
        shape = [512, HIDDEN]
        return (a + "wkv.weight", a + "wkv.scale", "F8_E4M3", shape,
                fp8_scale_shape(shape), "repl")
    if kind == K_Q_NORM:
        return (a + "q_norm.weight", None, "BF16", [1280], None, "repl")
    if kind == K_KV_NORM:
        return (a + "kv_norm.weight", None, "BF16", [512], None, "repl")
    if kind == K_ATTN_SINK:
        return (a + "attn_sink", None, "F32", [64], None, "sink")
    if kind == K_O_A:
        shape = [8192, 4096]
        return (a + "wo_a.weight", a + "wo_a.scale", "F8_E4M3", shape,
                fp8_scale_shape(shape), "diag")
    if kind == K_O_B:
        shape = [5120, 8192]
        return (a + "wo_b.weight", a + "wo_b.scale", "F8_E4M3", shape,
                fp8_scale_shape(shape), "ob")
    if kind == K_IDX_Q_B:
        shape = [4096, 1280]
        return (a + "indexer.wq_b.weight", a + "indexer.wq_b.scale", "F8_E4M3", shape,
                fp8_scale_shape(shape), "rows")
    if kind == K_IDX_WK:
        return (a + "indexer.wk.weight", None, "BF16", [128, 512], None, "repl")
    if kind == K_IDX_WP:
        return (a + "indexer.weights_proj.weight", None, "BF16", [32, 5120], None, "rows")
    if kind == K_IDX_KN:
        return (a + "indexer.k_norm.weight", None, "BF16", [128], None, "repl")
    if kind == K_C_WKV:
        return (a + "compressor.wkv.weight", None, "BF16", [512, HIDDEN], None, "repl")
    if kind == K_C_WGATE:
        return (a + "compressor.wgate.weight", None, "BF16", [512, HIDDEN], None, "repl")
    if kind == K_C_NORM:
        return (a + "compressor.norm.weight", None, "BF16", [512], None, "repl")
    if kind == K_HC_A_FN:
        return (f"layers.{layer}.hc_attn_fn", None, "F32", [24, 20480], None, "repl")
    if kind == K_HC_A_BASE:
        return (f"layers.{layer}.hc_attn_base", None, "F32", [24], None, "repl")
    if kind == K_HC_A_SCALE:
        return (f"layers.{layer}.hc_attn_scale", None, "F32", [3], None, "repl")
    if kind == K_HC_F_FN:
        return (f"layers.{layer}.hc_ffn_fn", None, "F32", [24, 20480], None, "repl")
    if kind == K_HC_F_BASE:
        return (f"layers.{layer}.hc_ffn_base", None, "F32", [24], None, "repl")
    if kind == K_HC_F_SCALE:
        return (f"layers.{layer}.hc_ffn_scale", None, "F32", [3], None, "repl")
    if kind == K_ROUTER:
        return (f + "gate.weight", None, "BF16", [384, HIDDEN], None, "repl")
    if kind == K_ROUTER_BIAS:
        return (f + "gate.bias", None, "F32", [384], None, "repl")
    if kind == K_ROUTER_BIAS_VL:
        return (f + "gate.bias_vl", None, "F32", [384], None, "repl")
    if kind in EXPERT_KINDS:
        w = "w2" if kind == K_EXP_W2 else "w1" if kind == K_EXP_W1 else "w3"
        shape = [5120, EXPERT_WIDTH // 2] if kind == K_EXP_W2 else [EXPERT_WIDTH, HIDDEN // 2]
        return (f + "experts.{e}." + w + ".weight", f + "experts.{e}." + w + ".scale",
                "I8", shape, expert_scale_shape(shape), "expert")
    if kind == K_SH_W1:
        shape = [EXPERT_WIDTH, HIDDEN]
        return (f + "shared_experts.w1.weight", f + "shared_experts.w1.scale", "F8_E4M3",
                shape, fp8_scale_shape(shape), "repl")
    if kind == K_SH_W2:
        shape = [HIDDEN, EXPERT_WIDTH]
        return (f + "shared_experts.w2.weight", f + "shared_experts.w2.scale", "F8_E4M3",
                shape, fp8_scale_shape(shape), "repl")
    if kind == K_SH_W3:
        shape = [EXPERT_WIDTH, HIDDEN]
        return (f + "shared_experts.w3.weight", f + "shared_experts.w3.scale", "F8_E4M3",
                shape, fp8_scale_shape(shape), "repl")
    fail(f"unknown kind {kind}")


def global_spec(kind: int):
    if kind == K_EMBEDDING:
        return ("embed.weight", None, "BF16", [VOCAB, HIDDEN], None, "rows")
    if kind == K_FINAL_NORM:
        return ("norm.weight", None, "BF16", [HIDDEN], None, "repl")
    return ("head.weight", None, "BF16", [VOCAB, HIDDEN], None, "rows")


def normalize(shape: list) -> list:
    if len(shape) == 2:
        return shape
    if len(shape) == 1:
        return [1, shape[0]]
    fail(f"tensor rank {len(shape)} unsupported: {shape}")


def expected_bytes(dtype: str, shape: list) -> int:
    if dtype not in DTYPE_BYTES:
        fail(f"unknown safetensors dtype {dtype}")
    count = 1
    for dim in shape:
        count *= dim
    return count * DTYPE_BYTES[dtype]


def check_shape(name: str, flat: dict, dtype: str, shape: list):
    got = flat.get(name)
    if got is None:
        fail(f"checkpoint tensor missing: {name}")
    if got["dtype"] != dtype or normalize(got["shape"]) != normalize(shape):
        fail(f"shape/dtype mismatch {name}: checkpoint {got['dtype']} {got['shape']} "
             f"!= expected {dtype} {shape}")


def entry_specs() -> list:
    specs = []
    for kind in range(KIND_COUNT):
        if is_global(kind):
            specs.append((kind, GLOBAL_LAYER, global_spec(kind)))
            continue
        for layer in range(LAYER_COUNT):
            if kind_in_layer(kind, layer):
                specs.append((kind, layer, checkpoint_spec(kind, layer)))
    return specs


def consumed_names() -> set:
    names = set()
    for kind, _, spec in entry_specs():
        if "{e}" in spec[0]:
            for expert in range(ROUTED_EXPERTS):
                names.add(spec[0].format(e=expert))
                names.add(spec[1].format(e=expert))
        else:
            names.add(spec[0])
            if spec[1]:
                names.add(spec[1])
    return names


def load_flat(headers_json: str) -> dict:
    with open(headers_json, encoding="utf-8") as handle:
        document = json.load(handle)
    flat = {}
    for shard, entries in document["shards"].items():
        for name, meta in entries.items():
            if name in flat:
                fail(f"duplicate tensor name across shards: {name}")
            meta = dict(meta)
            meta["shard"] = int(shard)
            meta["shape"] = normalize(meta["shape"])
            flat[name] = meta
    return document, flat


def cmd_headers(warm: str, index_path: str, out_json: str):
    with open(index_path, encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    table = {}
    seen = set()
    for shard in range(1, SHARD_COUNT + 1):
        path = os.path.join(warm, SHARD_NAMING % (shard, SHARD_COUNT))
        fd = os.open(path, os.O_RDONLY)
        try:
            raw_len = os.pread(fd, 8, 0)
            if len(raw_len) != 8:
                fail(f"short header length field shard {shard}")
            header_len = struct.unpack("<Q", raw_len)[0]
            raw = os.pread(fd, header_len, 8)
            if len(raw) != header_len:
                fail(f"short header shard {shard}")
            header = json.loads(raw)
            base = 8 + header_len
            entries = {}
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                begin, end = meta["data_offsets"]
                if end - begin != expected_bytes(meta["dtype"], meta["shape"]):
                    fail(f"byte span mismatch {name} shard {shard}")
                entries[name] = {"dtype": meta["dtype"], "shape": meta["shape"],
                                 "begin": base + begin, "end": base + end}
                seen.add(name)
            table[str(shard)] = entries
        finally:
            os.close(fd)
    wanted = set(weight_map)
    if seen != wanted:
        fail(f"shard headers vs index weight_map disagree: "
             f"missing {sorted(wanted - seen)[:8]} extra {sorted(seen - wanted)[:8]}")
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump({"warm": warm, "shards": table}, handle)
    print(f"headers ok: {len(seen)} tensors across {SHARD_COUNT} shards")


def fp8_scale_shape(shape: list) -> list:
    return [shape[0] // 32, shape[1] // 32]


def expert_scale_shape(shape: list) -> list:
    return [shape[0], shape[1] * 2 // 32]


def rank_window(mode: str, rank: int, tp: int, rows: int, cols: int):
    if mode == "repl":
        return 0, rows, 0, cols
    if mode == "rows":
        r = rows // tp
        return rank * r, (rank + 1) * r, 0, cols
    if mode in ("sink", "ob"):
        c = cols // tp
        return 0, rows, rank * c, (rank + 1) * c
    if mode == "diag":
        return rank * (rows // tp), (rank + 1) * (rows // tp), \
            rank * (cols // tp), (rank + 1) * (cols // tp)
    fail(f"unknown mode {mode}")


def build_pieces(flat: dict, tp: int, entries: list, order: list) -> list:
    pieces = []
    for entry, (kind, _) in zip(entries, order):
        layer = entry["layer"]
        groups = entry["groups"]
        spec = global_spec(kind) if is_global(kind) else checkpoint_spec(kind, layer)
        wname, sname, _, wshape, sshape, mode = spec
        for plane, offset_field, bytes_field, name, shape in (
                ("payload", "payload_offset", "payload_bytes", wname, wshape),
                ("scale", "scale_offset", "scale_bytes", sname, sshape)):
            if not name:
                continue
            if mode == "expert":
                per_group = entry[bytes_field] // groups
                for rank in range(tp):
                    for g in range(groups):
                        src_name = name.format(e=rank * groups + g)
                        src = flat[src_name]
                        pieces.append({"rank": rank, "kind": kind, "layer": layer,
                                       "plane": plane, "shard": src["shard"],
                                       "name": src_name, "row0": 0, "row1": shape[0],
                                       "col0": 0, "col1": shape[1],
                                       "dst_offset": entry[offset_field] + g * per_group,
                                       "dst_bytes": per_group})
                continue
            src0 = flat[name]
            for rank in range(tp):
                r0, r1, c0, c1 = rank_window(mode, rank, tp, src0["shape"][0], src0["shape"][1])
                pieces.append({"rank": rank, "kind": kind, "layer": layer, "plane": plane,
                               "shard": src0["shard"], "name": name,
                               "row0": r0, "row1": r1, "col0": c0, "col1": c1,
                               "dst_offset": entry[offset_field],
                               "dst_bytes": entry[bytes_field]})
    return pieces


def cmd_plan(headers_json: str, out_dir: str, tp: int, revision: str,
             contract_hex: str, config_hex: str, recipe_hex: str):
    if tp <= 0 or ROUTED_EXPERTS % tp or VOCAB % tp or 64 % tp or 4096 % tp or 32 % tp:
        fail(f"tp {tp} does not divide the sharded dimensions")
    document, flat = load_flat(headers_json)
    consumed = consumed_names()
    missing = sorted(consumed - set(flat))
    if missing:
        fail(f"kind map consumes tensors absent from the checkpoint: {missing[:8]}")
    engram, sidecar, scope = {}, [], []
    for name, meta in flat.items():
        if name in consumed:
            continue
        if name.startswith("layers.") and ".engram." in name:
            engram[name] = meta
        elif name.startswith("mtp."):
            sidecar.append(name)
        elif name.startswith(("vision.", "aligner.", "image_")):
            scope.append(name)
        else:
            fail(f"unaccounted checkpoint tensor: {name}")
    if len(engram) != ENGRAM_TENSOR_COUNT:
        fail(f"engram module inventory {len(engram)} != {ENGRAM_TENSOR_COUNT}")
    for kind, _, spec in entry_specs():
        if "{e}" in spec[0]:
            for expert in range(ROUTED_EXPERTS):
                check_shape(spec[0].format(e=expert), flat, spec[2], spec[3])
                check_shape(spec[1].format(e=expert), flat, "F8_E8M0", spec[4])
        else:
            check_shape(spec[0], flat, spec[2], spec[3])
            if spec[1]:
                check_shape(spec[1], flat, "F8_E8M0", spec[4])
    order = [(kind, GLOBAL_LAYER) for kind in range(KIND_COUNT) if is_global(kind)]
    for layer in range(LAYER_COUNT):
        for kind in range(K_ATTN_NORM, KIND_COUNT):
            if kind_in_layer(kind, layer):
                order.append((kind, layer))
    entries = []
    cursor = 0
    for kind, layer in order:
        rows, cols, groups = entry_shape(kind, tp)
        payload_type, codec, scale_enc, payload_bytes, scale_bytes = entry_layout(kind, tp)
        cursor = (cursor + ALIGN - 1) & ~(ALIGN - 1)
        entry = {"kind": kind, "layer": layer, "payload_type": payload_type, "codec": codec,
                 "scale_encoding": scale_enc, "groups": groups, "rows": rows, "cols": cols,
                 "payload_bytes": payload_bytes, "scale_bytes": scale_bytes,
                 "payload_offset": cursor}
        cursor += payload_bytes
        if scale_bytes:
            cursor = (cursor + ALIGN - 1) & ~(ALIGN - 1)
            entry["scale_offset"] = cursor
            cursor += scale_bytes
        else:
            entry["scale_offset"] = 0
        entries.append(entry)
    directory_offset = (HEADER_BYTES + ALIGN - 1) & ~(ALIGN - 1)
    payload_base = (directory_offset + len(entries) * ENTRY_BYTES + ALIGN - 1) & ~(ALIGN - 1)
    for entry in entries:
        entry["payload_offset"] += payload_base
        if entry["scale_offset"]:
            entry["scale_offset"] += payload_base
    file_bytes = cursor + payload_base
    spine_bytes = 0
    expert_bytes = 0
    for entry in entries:
        total = entry["payload_bytes"] + entry["scale_bytes"]
        if entry["kind"] in EXPERT_KINDS:
            expert_bytes += total
        else:
            spine_bytes += total
    header = bytearray(HEADER_BYTES)
    struct.pack_into("<20I2Q", header, 0, MAGIC, 1, HEADER_BYTES, ENTRY_BYTES, 1, 0,
                     len(entries), 1, 0, 0, LAYER_COUNT, LAYER_COUNT, HIDDEN, VOCAB,
                     ROUTED_EXPERTS, CODEC_FP8, CODEC_MXFP4, tp, 0, 0,
                     directory_offset, file_bytes)
    rev = revision.encode()
    if len(rev) >= MODEL_REVISION_BYTES:
        fail("revision does not fit the header field")
    header[96:96 + len(rev)] = rev
    for slot, hextext in ((0, contract_hex), (1, config_hex), (2, recipe_hex)):
        if len(hextext) != 64:
            fail(f"digest slot {slot} must be 64 hex chars")
        header[161 + slot * 32:193 + slot * 32] = bytes.fromhex(hextext)
    os.makedirs(out_dir, exist_ok=True)
    for rank in range(tp):
        struct.pack_into("<I", header, 72, rank)
        path = os.path.join(out_dir, f"rank{rank}.spstage")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.ftruncate(fd, file_bytes)
            os.pwrite(fd, bytes(header), 0)
            for index, entry in enumerate(entries):
                record = struct.pack("<8I4Q", entry["kind"], entry["layer"],
                                     entry["payload_type"], entry["codec"],
                                     entry["scale_encoding"], entry["groups"],
                                     entry["rows"], entry["cols"], entry["payload_offset"],
                                     entry["payload_bytes"], entry["scale_offset"],
                                     entry["scale_bytes"])
                os.pwrite(fd, record, directory_offset + index * ENTRY_BYTES)
        finally:
            os.close(fd)
    pieces = build_pieces(flat, tp, entries, order)
    plan = {"tp": tp, "expert_codec": "mxfp4", "revision": revision,
            "contract_sha256": contract_hex, "config_sha256": config_hex,
            "recipe_sha256": recipe_hex, "directory_offset": directory_offset,
            "file_bytes": file_bytes, "tensor_count": len(entries),
            "spine_bytes": spine_bytes, "expert_bytes": expert_bytes,
            "entries": entries, "pieces": pieces}
    with open(os.path.join(out_dir, "plan.json"), "w", encoding="utf-8") as handle:
        json.dump(plan, handle)
    engram_report = {name: {"shard": meta["shard"], "dtype": meta["dtype"],
                            "shape": meta["shape"], "bytes": meta["end"] - meta["begin"],
                            "shard_path": os.path.join(document["warm"], SHARD_NAMING % (meta["shard"], SHARD_COUNT))}
                     for name, meta in sorted(engram.items())}
    accounting = {"checkpoint_tensors": len(flat), "pack_consumed": len(consumed),
                  "engram_excluded": len(engram), "dspark_sidecar": len(sidecar),
                  "vision_aligner_out_of_scope": len(scope)}
    if sum(accounting[k] for k in ("pack_consumed", "engram_excluded", "dspark_sidecar",
                                   "vision_aligner_out_of_scope")) != len(flat):
        fail("tensor accounting does not close")
    with open(os.path.join(out_dir, "engram_report.json"), "w", encoding="utf-8") as handle:
        json.dump({"engram_tensors": engram_report,
                   "engram_bytes_total": sum(m["end"] - m["begin"] for m in engram.values()),
                   "dspark_sidecar_tensors": len(sidecar),
                   "vision_aligner_tensors": len(scope),
                   "accounting": accounting}, handle, indent=1)
    print(f"plan ok: {len(entries)} entries/rank x {tp} ranks, file_bytes {file_bytes}, "
          f"spine {spine_bytes}, expert {expert_bytes}, pieces {len(pieces)}; "
          f"engram excluded {sum(m['end'] - m['begin'] for m in engram.values())} B, "
          f"sidecar {len(sidecar)}, vision {len(scope)}")


def copy_region(src_fd: int, dst_fd: int, src: dict, piece: dict, dst_offset: int) -> int:
    esz = DTYPE_BYTES[src["dtype"]]
    row_bytes = src["shape"][1] * esz
    span = (piece["col1"] - piece["col0"]) * esz
    r0, r1, c0, c1 = piece["row0"], piece["row1"], piece["col0"], piece["col1"]
    done = 0
    row = r0
    while row < r1:
        block = min(max(COPY_BUDGET // row_bytes, 1), r1 - row)
        want = block * row_bytes
        raw = os.pread(src_fd, want, src["begin"] + row * row_bytes)
        if len(raw) != want:
            fail(f"short read {src['name']} rows {row}..{row + block}")
        for index in range(block):
            base = index * row_bytes
            os.pwrite(dst_fd, raw[base + c0 * esz:base + c1 * esz], dst_offset + done)
            done += span
        row += block
    return done


def cmd_copy(headers_json: str, plan_path: str, out_dir: str, warm: str, lo: int, hi: int):
    with open(plan_path, encoding="utf-8") as handle:
        plan = json.load(handle)
    _, flat = load_flat(headers_json)
    fds = [os.open(os.path.join(out_dir, f"rank{rank}.spstage"), os.O_WRONLY)
           for rank in range(plan["tp"])]
    by_shard = {}
    for piece in plan["pieces"]:
        if lo <= piece["shard"] <= hi:
            by_shard.setdefault(piece["shard"], []).append(piece)
    copied = 0
    try:
        for shard in sorted(by_shard):
            path = os.path.join(warm, SHARD_NAMING % (shard, SHARD_COUNT))
            fd = os.open(path, os.O_RDONLY)
            try:
                for piece in by_shard[shard]:
                    src = flat[piece["name"]]
                    wrote = copy_region(fd, fds[piece["rank"]], src, piece, piece["dst_offset"])
                    if wrote != piece["dst_bytes"]:
                        fail(f"piece size mismatch {piece['name']} rank {piece['rank']} "
                             f"{wrote} != {piece['dst_bytes']}")
                    copied += wrote
                if hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
    finally:
        for fd in fds:
            os.close(fd)
    with open(os.path.join(out_dir, f"copy_{lo:02d}_{hi:02d}.done"), "w", encoding="utf-8") as handle:
        handle.write(f"{copied}\n")
    print(f"copy shards [{lo},{hi}] ok: {sum(len(v) for v in by_shard.values())} pieces, "
          f"{copied} bytes")


def main(argv: list) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    try:
        if argv[1] == "headers" and len(argv) == 5:
            cmd_headers(argv[2], argv[3], argv[4])
        elif argv[1] == "plan" and len(argv) == 9:
            cmd_plan(argv[2], argv[3], int(argv[4]), argv[5], argv[6], argv[7], argv[8])
        elif argv[1] == "copy" and len(argv) == 8:
            cmd_copy(argv[2], argv[3], argv[4], argv[5], int(argv[6]), int(argv[7]))
        else:
            print(__doc__)
            return 2
    except Fail as error:
        print(f"dsv41_flash_stagepack FAIL: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"dsv41_flash_stagepack OS FAIL: {error}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
