#!/usr/bin/env python3
"""Engram row-shard cutter + verifier for DeepSeek-V4.1-Flash (operator row-shard ruling).

Cuts the two host-resident engram hash tables (layers 1/14) plus their replicated
small tensors out of the warm checkpoint into 16 per-mesh-rank shard files
(TP8xPP2 = 16 sparks, one rank per spark). Wire format: stagepack header/entry
structs with an engram tensor-kind space; embed entries carry the fp8 payload
plane plus the E8M0 scale plane sliced by rank; wkv/q_weight/k_weight ride
replicated and are served from the weightd spine. The .experts manifest keys
(layer, row-block) groups with weight+scale ranges so the driver leases rows
through the expert-pool seam.

Partition is the reference ParallelEngramEmbedding layout: rows_per_rank =
ceil(entries / 16), rank r owns rows [r*part, min(entries, (r+1)*part)),
owner(id) = id // part.

  selftest                     fabricate a tiny checkpoint in a temp dir, cut + verify round-trip
  cut WARM_DIR OUT_DIR [SPEC]  cut shards + .experts + engram_receipt.json; SPEC = "all" (default),
                               a rank ("3"), a range ("2-5"), or comma mix ("0,7-9"); each rank is
                               checkpointed into the receipt on completion, so an interrupted run
                               resumes by re-running with the next SPEC
  verify WARM_DIR OUT_DIR      re-derive layout, boundary rows/planes vs checkpoint, manifest ck128, sha256

Exit 0 = step complete; nonzero = the failure is named on stderr.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile

RANKS = 16
LAYER_IDS = (1, 14)
ENTRIES = {1: 384006168, 14: 384016682}
HEAD_DIM = 256
SCALE_COLS = 8
BLOCK_ROWS = 32768
WKV_ROWS = 25600
WKV_COLS = 6144
WKV_SCALE_ROWS = 800
WKV_SCALE_COLS = 192
GK_ROWS = 4
GK_COLS = 5120
WARM_REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
WARM_CONFIG_SHA256 = "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"
REF_ENGRAM_SHA256 = "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897"
REF_MODEL_SHA256 = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
REF_KERNEL_SHA256 = "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
SHARD_NAMING = "model-%05d-of-%05d.safetensors"
HEADER_BYTES = 257
ENTRY_BYTES = 64
ALIGN = 256
MAGIC = 0x31474544
FORMAT_VERSION = 1
CODEC_ABI = 1
PT_BF16, PT_PACKED = 1, 4
COD_BF16, COD_FP8 = 1, 5
SE_NONE, SE_E8M0 = 0, 3
MANIFEST_MAGIC = 0x58504557
MANIFEST_VERSION = 2
EXPERT_BYTES_MAX = 64 * 1024 * 1024
COPY_CHUNK = 64 * 1024 * 1024

K_EMBED_W1, K_EMBED_W14, K_WKV1, K_WKV14, K_QW1, K_QW14, K_KW1, K_KW14 = range(8)
KIND_COUNT = 8
EMBED_KINDS = (K_EMBED_W1, K_EMBED_W14)


class ShardFailure(Exception):
    pass


def fail(message: str):
    raise ShardFailure(message)


def rows_per_rank(entries: int, ranks: int = RANKS) -> int:
    return (entries + ranks - 1) // ranks


def rank_rows(entries: int, rank: int, ranks: int = RANKS) -> int:
    part = rows_per_rank(entries, ranks)
    return max(0, min(part, entries - rank * part))


def kind_spec(kind: int, entries_map: dict[int, int]):
    if kind == K_EMBED_W1:
        return (1, entries_map[1], HEAD_DIM, 1)
    if kind == K_EMBED_W14:
        return (14, entries_map[14], HEAD_DIM, 1)
    if kind == K_WKV1:
        return (1, WKV_ROWS, WKV_COLS, 1)
    if kind == K_WKV14:
        return (14, WKV_ROWS, WKV_COLS, 1)
    if kind == K_QW1:
        return (1, GK_ROWS, GK_COLS, 2)
    if kind == K_QW14:
        return (14, GK_ROWS, GK_COLS, 2)
    if kind == K_KW1:
        return (1, GK_ROWS, GK_COLS, 2)
    if kind == K_KW14:
        return (14, GK_ROWS, GK_COLS, 2)
    fail(f"unknown kind {kind}")


def entry_kind_order() -> list[tuple[int, int]]:
    return [(K_EMBED_W1, 1), (K_WKV1, 1), (K_QW1, 1), (K_KW1, 1),
            (K_EMBED_W14, 14), (K_WKV14, 14), (K_QW14, 14), (K_KW14, 14)]


def shard_tensor_name(layer: int, suffix: str) -> str:
    return f"layers.{layer}.engram.{suffix}"


def entry_source_name(kind: int) -> str:
    if kind in (K_EMBED_W1, K_EMBED_W14):
        return "embed.weight"
    if kind in (K_WKV1, K_WKV14):
        return "wkv.weight"
    if kind in (K_QW1, K_QW14):
        return "q_weight"
    return "k_weight"


def contract_sha(entries_map: dict[int, int]) -> str:
    recipe = json.dumps({
        "magic": MAGIC, "format_version": FORMAT_VERSION, "ranks": RANKS,
        "layer_ids": list(LAYER_IDS), "entries": entries_map,
        "head_dim": HEAD_DIM, "scale_cols": SCALE_COLS, "block_rows": BLOCK_ROWS,
        "wkv": [WKV_ROWS, WKV_COLS], "gate_weights": [GK_ROWS, GK_COLS],
        "partition": "ceil(rows/ranks), owner=id//part",
        "ref_engram": REF_ENGRAM_SHA256, "ref_model": REF_MODEL_SHA256,
        "ref_kernel": REF_KERNEL_SHA256, "revision": WARM_REVISION,
    }, sort_keys=True).encode()
    return hashlib.sha256(recipe).hexdigest()


def derive_layout(entries_map: dict[int, int], rank: int) -> tuple[list[dict], int]:
    directory_offset = (HEADER_BYTES + ALIGN - 1) & ~(ALIGN - 1)
    payload_base = (directory_offset + KIND_COUNT * ENTRY_BYTES + ALIGN - 1) & ~(ALIGN - 1)
    entries = []
    cursor = 0
    for kind, layer in entry_kind_order():
        _, rows, cols, element_bytes = kind_spec(kind, entries_map)
        if kind == K_EMBED_W1:
            rows = rank_rows(entries_map[1], rank)
        if kind == K_EMBED_W14:
            rows = rank_rows(entries_map[14], rank)
        if kind in EMBED_KINDS:
            scale_rows, scale_cols = rows, SCALE_COLS
        elif kind in (K_WKV1, K_WKV14):
            scale_rows, scale_cols = WKV_SCALE_ROWS, WKV_SCALE_COLS
        else:
            scale_rows, scale_cols = 0, 0
        payload_type = PT_BF16 if element_bytes == 2 else PT_PACKED
        codec = COD_BF16 if element_bytes == 2 else COD_FP8
        entry = {"kind": kind, "layer": layer, "payload_type": payload_type,
                 "codec": codec,
                 "scale_encoding": SE_E8M0 if scale_cols else SE_NONE,
                 "groups": 1, "rows": rows, "cols": cols,
                 "payload_bytes": rows * cols * element_bytes,
                 "scale_bytes": scale_rows * scale_cols * 1}
        cursor = (cursor + ALIGN - 1) & ~(ALIGN - 1)
        entry["payload_offset"] = cursor + payload_base
        cursor += entry["payload_bytes"]
        if entry["scale_bytes"]:
            cursor = (cursor + ALIGN - 1) & ~(ALIGN - 1)
            entry["scale_offset"] = cursor + payload_base
            cursor += entry["scale_bytes"]
        else:
            entry["scale_offset"] = 0
        entries.append(entry)
    file_bytes = cursor + payload_base
    return entries, file_bytes


def safetensors_index(path: str) -> tuple[dict, int]:
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            fail(f"{path}: truncated safetensors prefix")
        header_len = struct.unpack("<Q", prefix)[0]
        raw = handle.read(header_len)
        if len(raw) != header_len:
            fail(f"{path}: truncated safetensors header")
    header = json.loads(raw)
    header.pop("__metadata__", None)
    return header, 8 + header_len


def check_shapes(wanted: dict, entries_map: dict[int, int]):
    for layer in LAYER_IDS:
        for suffix, rows, cols, dtype in (
                ("embed.weight", entries_map[layer], HEAD_DIM, "F8_E4M3"),
                ("embed.scale", entries_map[layer], SCALE_COLS, "F8_E8M0"),
                ("wkv.weight", WKV_ROWS, WKV_COLS, "F8_E4M3"),
                ("wkv.scale", WKV_SCALE_ROWS, WKV_SCALE_COLS, "F8_E8M0"),
                ("q_weight", GK_ROWS, GK_COLS, "BF16"),
                ("k_weight", GK_ROWS, GK_COLS, "BF16")):
            meta = wanted[shard_tensor_name(layer, suffix)]
            if meta["dtype"] != dtype or meta["shape"] != [rows, cols]:
                fail(f"layers.{layer}.engram.{suffix}: dtype/shape "
                     f"{meta['dtype']} {meta['shape']} != {dtype} {[rows, cols]}")
            dtype_bytes = 2 if dtype == "BF16" else 1
            if meta["end"] - meta["begin"] != rows * cols * dtype_bytes:
                fail(f"layers.{layer}.engram.{suffix}: byte span mismatch")


def resolve_tensors(warm_dir: str, entries_map: dict[int, int]) -> dict[str, dict]:
    index_path = os.path.join(warm_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        fail(f"{index_path}: missing index")
    weight_map = json.load(open(index_path)).get("weight_map")
    if not isinstance(weight_map, dict):
        fail("index weight_map missing")
    wanted = {}
    for layer in LAYER_IDS:
        for suffix in ("embed.weight", "embed.scale", "wkv.weight", "wkv.scale",
                       "q_weight", "k_weight"):
            name = shard_tensor_name(layer, suffix)
            shard = weight_map.get(name)
            if shard is None:
                fail(f"{name}: absent from weight_map")
            wanted[name] = {"shard": shard}
    parsed = {}
    for name, meta in wanted.items():
        if meta["shard"] not in parsed:
            parsed[meta["shard"]] = safetensors_index(os.path.join(warm_dir, meta["shard"]))
        header, data_start = parsed[meta["shard"]]
        flat = header.get(name)
        if flat is None:
            fail(f"{name}: absent from shard header {meta['shard']}")
        begin, end = flat["data_offsets"]
        meta.update({"path": os.path.join(warm_dir, meta["shard"]),
                     "begin": begin, "end": end, "dtype": flat["dtype"],
                     "shape": flat["shape"], "data_start": data_start})
    check_shapes(wanted, entries_map)
    return wanted


def load_checkpoint_config(warm_dir: str, entries_map: dict[int, int]):
    config_path = os.path.join(warm_dir, "config.json")
    config_raw = open(config_path, "rb").read()
    strict = entries_map == dict(ENTRIES)
    if strict:
        digest = hashlib.sha256(config_raw).hexdigest()
        if digest != WARM_CONFIG_SHA256:
            fail(f"config.json sha256 {digest} != pinned {WARM_CONFIG_SHA256}")
    config = json.loads(config_raw)
    holder = config if "engram_layer_ids" in config else config.get("text_config", {})
    observed = (holder.get("engram_layer_ids"), holder.get("engram_num_embeddings"),
                holder.get("engram_head_dim"), holder.get("engram_compressed_vocab_size"))
    if list(observed[0] or ()) != list(LAYER_IDS) or list(observed[1] or ()) != [entries_map[1], entries_map[14]]:
        fail(f"config engram geometry mismatch: {observed[:2]}")
    if observed[2] not in (None, HEAD_DIM) or observed[3] not in (None, 99092):
        fail(f"config engram head_dim/compressed vocab mismatch: {observed[2:]}")


def copy_range(dst_fd: int, src_path: str, src_offset: int, dst_offset: int, length: int):
    with open(src_path, "rb") as src:
        src.seek(src_offset)
        done = 0
        while done != length:
            data = src.read(min(COPY_CHUNK, length - done))
            if not data:
                fail(f"{src_path}: short read at {src_offset + done}")
            if os.pwrite(dst_fd, data, dst_offset + done) != len(data):
                fail(f"pwrite short at {dst_offset + done}")
            done += len(data)


def read_range(path: str, offset: int, length: int) -> bytes:
    with open(path, "rb") as handle:
        handle.seek(offset)
        data = handle.read(length)
    if len(data) != length:
        fail(f"{path}: short read {len(data)} != {length} at {offset}")
    return data


MASK64 = (1 << 64) - 1
CK_C1 = 0x87c37b91114253d5
CK_C2 = 0x4cf5ad432745937f


def _rotl64(x: int, r: int) -> int:
    return ((x << r) | (x >> (64 - r))) & MASK64


def _fmix64(k: int) -> int:
    k ^= k >> 33
    k = (k * 0xff51afd7ed558ccd) & MASK64
    k ^= k >> 33
    k = (k * 0xc4ceb9fe1a85ec53) & MASK64
    k ^= k >> 33
    return k


def ck128(data: bytes) -> bytes:
    h1 = h2 = 0
    total = len(data)
    aligned = total - (total % 16)
    for k1, k2 in struct.iter_unpack("<QQ", memoryview(data)[:aligned]):
        k1 = (k1 * CK_C1) & MASK64
        k1 = _rotl64(k1, 31)
        k1 = (k1 * CK_C2) & MASK64
        h1 ^= k1
        h1 = (_rotl64(h1, 27) + h2) & MASK64
        h1 = (h1 * 5 + 0x52dce729) & MASK64
        k2 = (k2 * CK_C2) & MASK64
        k2 = _rotl64(k2, 33)
        k2 = (k2 * CK_C1) & MASK64
        h2 ^= k2
        h2 = (_rotl64(h2, 31) + h1) & MASK64
        h2 = (h2 * 5 + 0x38495ab5) & MASK64
    tail = data[aligned:]
    if len(tail) > 8:
        k2 = int.from_bytes(tail[8:16].ljust(8, b"\0"), "little")
        k2 = (k2 * CK_C2) & MASK64
        k2 = _rotl64(k2, 33)
        k2 = (k2 * CK_C1) & MASK64
        h2 ^= k2
    k1 = int.from_bytes(tail[:8].ljust(8, b"\0"), "little")
    k1 = (k1 * CK_C1) & MASK64
    k1 = _rotl64(k1, 31)
    k1 = (k1 * CK_C2) & MASK64
    h1 ^= k1
    h1 ^= total
    h2 ^= total
    h1 = (h1 + h2) & MASK64
    h2 = (h2 + h1) & MASK64
    h1 = _fmix64(h1)
    h2 = _fmix64(h2)
    h1 = (h1 + h2) & MASK64
    h2 = (h2 + h1) & MASK64
    return struct.pack("<QQ", h1, h2)


def build_header(rank: int, tensor_count: int, file_bytes: int,
                 contract_hex: str, config_hex: str, recipe_hex: str) -> bytes:
    header = bytearray(HEADER_BYTES)
    struct.pack_into("<20I2Q", header, 0, MAGIC, FORMAT_VERSION, HEADER_BYTES,
                     ENTRY_BYTES, CODEC_ABI, 0, tensor_count, 1, 0, 0, 40, 40,
                     5120, 129280, 384, COD_FP8, 7, RANKS, rank, 0,
                     512, file_bytes)
    revision = WARM_REVISION.encode()
    header[96:96 + len(revision)] = revision
    for slot, hextext in ((0, contract_hex), (1, config_hex), (2, recipe_hex)):
        if len(hextext) != 64:
            fail(f"digest slot {slot} must be 64 hex chars")
        header[161 + slot * 32:193 + slot * 32] = bytes.fromhex(hextext)
    return bytes(header)


def write_shard(out_dir: str, rank: int, header: bytes, entries: list[dict],
                wanted: dict, entries_map: dict[int, int]) -> dict:
    path = os.path.join(out_dir, f"engram.rank{rank}.spengram")
    file_bytes = max(entry["payload_offset"] + entry["payload_bytes"] for entry in entries)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(fd, file_bytes)
        os.pwrite(fd, header, 0)
        for index, entry in enumerate(entries):
            record = struct.pack("<8I4Q", entry["kind"], entry["layer"],
                                 entry["payload_type"], entry["codec"],
                                 entry["scale_encoding"], entry["groups"],
                                 entry["rows"], entry["cols"], entry["payload_offset"],
                                 entry["payload_bytes"], entry["scale_offset"],
                                 entry["scale_bytes"])
            os.pwrite(fd, record, 512 + index * ENTRY_BYTES)
        for kind in EMBED_KINDS:
            entry = next(e for e in entries if e["kind"] == kind)
            layer = entry["layer"]
            source = wanted[shard_tensor_name(layer, "embed.weight")]
            row0 = rank * rows_per_rank(entries_map[layer])
            copy_range(fd, source["path"],
                       source["data_start"] + source["begin"] + row0 * entry["cols"],
                       entry["payload_offset"], entry["payload_bytes"])
            scale = wanted[shard_tensor_name(layer, "embed.scale")]
            copy_range(fd, scale["path"],
                       scale["data_start"] + scale["begin"] + row0 * SCALE_COLS,
                       entry["scale_offset"], entry["scale_bytes"])
        for kind in (K_WKV1, K_QW1, K_KW1, K_WKV14, K_QW14, K_KW14):
            entry = next(e for e in entries if e["kind"] == kind)
            source = wanted[shard_tensor_name(entry["layer"], entry_source_name(kind))]
            copy_range(fd, source["path"], source["data_start"] + source["begin"],
                       entry["payload_offset"], entry["payload_bytes"])
            if entry["scale_bytes"]:
                scale = wanted[shard_tensor_name(entry["layer"], "wkv.scale")]
                copy_range(fd, scale["path"], scale["data_start"] + scale["begin"],
                           entry["scale_offset"], entry["scale_bytes"])
    finally:
        os.close(fd)
    return {"path": path, "file_bytes": file_bytes}


def write_manifest(shard: dict, entries: list[dict]) -> dict:
    body = bytearray()
    for kind in EMBED_KINDS:
        entry = next(e for e in entries if e["kind"] == kind)
        blocks = (entry["rows"] + BLOCK_ROWS - 1) // BLOCK_ROWS
        for block in range(blocks):
            row0 = block * BLOCK_ROWS
            nrows = min(BLOCK_ROWS, entry["rows"] - row0)
            planes = ((kind * 2, entry["payload_offset"] + row0 * entry["cols"],
                       nrows * entry["cols"]),
                      (kind * 2 + 1, entry["scale_offset"] + row0 * SCALE_COLS,
                       nrows * SCALE_COLS))
            for kind_id, offset, length in planes:
                data = read_range(shard["path"], offset, length)
                body += struct.pack("<IIIIQQ", entry["layer"], block, kind_id, 0,
                                    offset, length)
                body += ck128(data)
    manifest_path = shard["path"] + ".experts"
    with open(manifest_path, "wb") as handle:
        handle.write(struct.pack("<IIII", MANIFEST_MAGIC, MANIFEST_VERSION,
                                 len(body) // 48, 0))
        handle.write(body)
    return {"path": manifest_path, "records": len(body) // 48}


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            piece = handle.read(COPY_CHUNK)
            if not piece:
                break
            digest.update(piece)
    return digest.hexdigest()


def parse_rank_spec(spec: str) -> list[int]:
    if spec in ("all", ""):
        return list(range(RANKS))
    ranks: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            span = list(range(int(lo), int(hi) + 1))
        else:
            span = [int(part)]
        for rank in span:
            if rank < 0 or rank >= RANKS:
                fail(f"rank {rank} out of range 0..{RANKS - 1}")
            if rank not in ranks:
                ranks.append(rank)
    return sorted(ranks)


def cmd_cut(warm_dir: str, out_dir: str, entries_map: dict[int, int],
            config_hex: str, ranks: list[int]) -> dict:
    load_checkpoint_config(warm_dir, entries_map)
    wanted = resolve_tensors(warm_dir, entries_map)
    contract = contract_sha(entries_map)
    os.makedirs(out_dir, exist_ok=True)
    receipt_path = os.path.join(out_dir, "engram_receipt.json")
    if os.path.isfile(receipt_path):
        receipt = json.load(open(receipt_path, encoding="utf-8"))
        if receipt.get("contract_sha256") != contract or receipt.get("ranks") != RANKS:
            fail("existing engram_receipt.json geometry/contract mismatch")
    else:
        receipt = {"contract_sha256": contract, "config_sha256": config_hex,
                   "revision": WARM_REVISION, "ranks": RANKS,
                   "entries": entries_map, "shards": []}
    by_rank = {record["rank"]: record for record in receipt["shards"]}
    for rank in ranks:
        entries, file_bytes = derive_layout(entries_map, rank)
        header = build_header(rank, len(entries), file_bytes, contract,
                              config_hex, "0" * 64)
        shard = write_shard(out_dir, rank, header, entries, wanted, entries_map)
        manifest = write_manifest(shard, entries)
        by_rank[rank] = {
            "rank": rank, "path": os.path.basename(shard["path"]),
            "file_bytes": shard["file_bytes"],
            "manifest_records": manifest["records"],
            "sha256": sha256_file(shard["path"]),
            "rows": {str(layer): rank_rows(entries_map[layer], rank)
                     for layer in LAYER_IDS},
        }
        receipt["shards"] = [by_rank[r] for r in sorted(by_rank)]
        with open(receipt_path, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=1, sort_keys=True)
        print(f"rank{rank}: {shard['file_bytes']} bytes, "
              f"{manifest['records']} manifest records", flush=True)
    print(f"wrote {receipt_path}")
    return receipt


def load_manifest(path: str) -> list[dict]:
    data = read_range(path, 0, os.path.getsize(path))
    if len(data) < 16:
        fail(f"{path}: manifest truncated")
    magic, version, count, zero = struct.unpack_from("<IIII", data, 0)
    if magic != MANIFEST_MAGIC or version != MANIFEST_VERSION or zero != 0:
        fail(f"{path}: manifest header mismatch")
    if len(data) != 16 + count * 48:
        fail(f"{path}: manifest length mismatch")
    records = []
    for index in range(count):
        base = 16 + index * 48
        layer, expert, kind, reserved, offset, length = struct.unpack_from("<IIIIQQ", data, base)
        if reserved != 0:
            fail(f"{path}: record {index} reserved nonzero")
        records.append({"layer": layer, "expert": expert, "kind": kind,
                        "offset": offset, "bytes": length,
                        "ck128": data[base + 32:base + 48]})
    return records


def cmd_verify(warm_dir: str, out_dir: str, entries_map: dict[int, int]) -> dict:
    load_checkpoint_config(warm_dir, entries_map)
    wanted = resolve_tensors(warm_dir, entries_map)
    receipt = json.load(open(os.path.join(out_dir, "engram_receipt.json")))
    if receipt["ranks"] != RANKS or receipt["entries"] != {str(k): v for k, v in entries_map.items()}:
        fail("receipt geometry mismatch")
    contract = contract_sha(entries_map)
    if receipt["contract_sha256"] != contract:
        fail("receipt contract sha mismatch")
    report = {"ranks": [], "rows_checked": 0, "blocks_checked": 0}
    partition = {layer: 0 for layer in LAYER_IDS}
    for rank_record in receipt["shards"]:
        rank = rank_record["rank"]
        path = os.path.join(out_dir, rank_record["path"])
        if sha256_file(path) != rank_record["sha256"]:
            fail(f"rank{rank}: sha256 mismatch vs receipt")
        raw = read_range(path, 0, HEADER_BYTES)
        fields = struct.unpack_from("<20I2Q", raw, 0)
        (magic, version, header_bytes, entry_bytes, abi, flags, tensor_count,
         _, _, _, _, _, _, _, _, _, _, tp_degree, stored_rank, _) = fields[:20]
        file_bytes = fields[20]
        if magic != MAGIC or version != FORMAT_VERSION or header_bytes != HEADER_BYTES \
                or entry_bytes != ENTRY_BYTES or abi != CODEC_ABI or flags != 0:
            fail(f"rank{rank}: header literals mismatch")
        if tp_degree != RANKS or stored_rank != rank or tensor_count != KIND_COUNT:
            fail(f"rank{rank}: topology/tensor-count mismatch")
        if raw[161:193].hex() != contract:
            fail(f"rank{rank}: header contract sha mismatch")
        entries, file_bytes = derive_layout(entries_map, rank)
        if file_bytes != rank_record["file_bytes"] or file_bytes != fields[21]:
            fail(f"rank{rank}: file_bytes mismatch")
        directory = read_range(path, 512, KIND_COUNT * ENTRY_BYTES)
        for index, entry in enumerate(entries):
            record = struct.unpack_from("<8I4Q", directory, index * ENTRY_BYTES)
            if record != (entry["kind"], entry["layer"], entry["payload_type"],
                          entry["codec"], entry["scale_encoding"], entry["groups"],
                          entry["rows"], entry["cols"], entry["payload_offset"],
                          entry["payload_bytes"], entry["scale_offset"],
                          entry["scale_bytes"]):
                fail(f"rank{rank}: entry {index} layout mismatch")
        checked_rows = 0
        for kind in EMBED_KINDS:
            entry = next(e for e in entries if e["kind"] == kind)
            layer = entry["layer"]
            row0 = rank * rows_per_rank(entries_map[layer])
            partition[layer] += entry["rows"]
            source = wanted[shard_tensor_name(layer, "embed.weight")]
            scale = wanted[shard_tensor_name(layer, "embed.scale")]
            probes = sorted({0, 1, entry["rows"] // 2, entry["rows"] - 2,
                             entry["rows"] - 1})
            for probe in probes:
                local = read_range(path, entry["payload_offset"] + probe * entry["cols"],
                                   entry["cols"])
                remote = read_range(source["path"], source["data_start"]
                                    + source["begin"] + (row0 + probe) * entry["cols"],
                                    entry["cols"])
                if local != remote:
                    fail(f"rank{rank}: weight row {probe} kind {kind} != checkpoint")
                local = read_range(path, entry["scale_offset"] + probe * SCALE_COLS,
                                   SCALE_COLS)
                remote = read_range(scale["path"], scale["data_start"]
                                    + scale["begin"] + (row0 + probe) * SCALE_COLS,
                                    SCALE_COLS)
                if local != remote:
                    fail(f"rank{rank}: scale row {probe} kind {kind} != checkpoint")
                checked_rows += 1
        for kind in (K_WKV1, K_QW1, K_KW1, K_WKV14, K_QW14, K_KW14):
            entry = next(e for e in entries if e["kind"] == kind)
            source = wanted[shard_tensor_name(entry["layer"], entry_source_name(kind))]
            head = read_range(path, entry["payload_offset"], min(1048576, entry["payload_bytes"]))
            remote = read_range(source["path"], source["data_start"] + source["begin"], len(head))
            if head != remote:
                fail(f"rank{rank}: spine head mismatch kind {kind}")
            tail_len = min(1048576, entry["payload_bytes"])
            tail = read_range(path, entry["payload_offset"] + entry["payload_bytes"] - tail_len, tail_len)
            remote = read_range(source["path"], source["data_start"] + source["end"] - tail_len, tail_len)
            if tail != remote:
                fail(f"rank{rank}: spine tail mismatch kind {kind}")
        records = load_manifest(path + ".experts")
        expect_records = 0
        for kind in EMBED_KINDS:
            entry = next(e for e in entries if e["kind"] == kind)
            blocks = (entry["rows"] + BLOCK_ROWS - 1) // BLOCK_ROWS
            expect_records += blocks * 2
        if len(records) != expect_records:
            fail(f"rank{rank}: manifest records {len(records)} != {expect_records}")
        blocks_checked = 0
        for record in records:
            if record["layer"] not in LAYER_IDS:
                fail(f"rank{rank}: manifest layer {record['layer']} unexpected")
            kind = record["kind"] // 2
            plane = record["kind"] % 2
            if kind not in EMBED_KINDS:
                fail(f"rank{rank}: manifest kind {record['kind']} unexpected")
            entry = next(e for e in entries if e["kind"] == kind)
            block = record["expert"]
            row0 = block * BLOCK_ROWS
            if row0 >= entry["rows"]:
                fail(f"rank{rank}: manifest block {block} out of range")
            nrows = min(BLOCK_ROWS, entry["rows"] - row0)
            base = entry["payload_offset"] if plane == 0 else entry["scale_offset"]
            stride = entry["cols"] if plane == 0 else SCALE_COLS
            if record["offset"] != base + row0 * stride:
                fail(f"rank{rank}: manifest block {block} plane {plane} offset mismatch")
            if record["bytes"] != nrows * stride:
                fail(f"rank{rank}: manifest block {block} plane {plane} bytes mismatch")
            if record["bytes"] > EXPERT_BYTES_MAX:
                fail(f"rank{rank}: manifest block {block} exceeds expert bytes max")
            if ck128(read_range(path, record["offset"], record["bytes"])) != record["ck128"]:
                fail(f"rank{rank}: manifest block {block} plane {plane} ck128 mismatch")
            if plane == 1:
                blocks_checked += 1
        report["ranks"].append({"rank": rank, "rows_checked": checked_rows,
                                "blocks_checked": blocks_checked})
        report["rows_checked"] += checked_rows
        report["blocks_checked"] += blocks_checked
        print(f"rank{rank}: verified ({checked_rows} rows, {blocks_checked} blocks)",
              flush=True)
    for layer in LAYER_IDS:
        if partition[layer] != entries_map[layer]:
            fail(f"layer {layer} partition sum {partition[layer]} != {entries_map[layer]}")
    return report


def selftest_entries() -> dict[int, int]:
    return {1: 100003, 14: 100037}


def selftest_write_checkpoint(warm: str, entries_map: dict[int, int]):
    config = {"text_config": {"engram_layer_ids": list(LAYER_IDS),
                              "engram_num_embeddings": [entries_map[1], entries_map[14]],
                              "engram_head_dim": HEAD_DIM,
                              "engram_compressed_vocab_size": 99092}}
    with open(os.path.join(warm, "config.json"), "wb") as handle:
        handle.write(json.dumps(config).encode())
    index = {"weight_map": {}}
    header = {}
    blob = bytearray()
    for layer in LAYER_IDS:
        for suffix, rows, cols, dtype in (
                ("embed.weight", entries_map[layer], HEAD_DIM, "F8_E4M3"),
                ("embed.scale", entries_map[layer], SCALE_COLS, "F8_E8M0"),
                ("wkv.weight", WKV_ROWS, WKV_COLS, "F8_E4M3"),
                ("wkv.scale", WKV_SCALE_ROWS, WKV_SCALE_COLS, "F8_E8M0"),
                ("q_weight", GK_ROWS, GK_COLS, "BF16"),
                ("k_weight", GK_ROWS, GK_COLS, "BF16")):
            name = shard_tensor_name(layer, suffix)
            index["weight_map"][name] = SHARD_NAMING % (1, 1)
            begin = len(blob)
            seed = (layer * 1315423911 + hash(name)) & 0xFFFFFFFF
            values = bytearray()
            while len(values) < rows * cols * (2 if dtype == "BF16" else 1):
                seed = (seed * 1103515245 + 12345) & 0xFFFFFFFF
                values += seed.to_bytes(4, "little")
            blob += values[:rows * cols * (2 if dtype == "BF16" else 1)]
            header[name] = {"dtype": dtype, "shape": [rows, cols],
                            "data_offsets": [begin, len(blob)]}
    header_bytes = json.dumps(header).encode()
    with open(os.path.join(warm, "model.safetensors.index.json"), "w") as handle:
        json.dump(index, handle)
    with open(os.path.join(warm, SHARD_NAMING % (1, 1)), "wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(bytes(blob))


def cmd_selftest() -> int:
    entries_map = selftest_entries()
    tmp = tempfile.mkdtemp(prefix="dsv41-engram-selftest-")
    try:
        warm = os.path.join(tmp, "warm")
        os.makedirs(warm)
        selftest_write_checkpoint(warm, entries_map)
        out = os.path.join(tmp, "shards")
        receipt = cmd_cut(warm, out, entries_map,
                          hashlib.sha256(b"selftest").hexdigest(),
                          list(range(RANKS)))
        if len(receipt["shards"]) != RANKS:
            fail("selftest shard count mismatch")
        report = cmd_verify(warm, out, entries_map)
        if len(report["ranks"]) != RANKS:
            fail("selftest verify rank count mismatch")
        for layer in LAYER_IDS:
            part = rows_per_rank(entries_map[layer])
            if rank_rows(entries_map[layer], RANKS - 1) != entries_map[layer] - (RANKS - 1) * part:
                fail("selftest last-rank partition mismatch")
        print(f"selftest PASS rows={report['rows_checked']} "
              f"blocks={report['blocks_checked']}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str]) -> int:
    if len(argv) == 1 and argv[0] == "selftest":
        return cmd_selftest()
    if len(argv) in (3, 4) and argv[0] == "cut":
        ranks = parse_rank_spec(argv[3]) if len(argv) == 4 else list(range(RANKS))
        receipt = cmd_cut(argv[1], argv[2], dict(ENTRIES), WARM_CONFIG_SHA256,
                          ranks)
        print(f"contract {receipt['contract_sha256']}")
        return 0
    if len(argv) == 3 and argv[0] == "verify":
        cmd_verify(argv[1], argv[2], dict(ENTRIES))
        print("verify PASS")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except ShardFailure as error:
        print(f"engram shard failure: {error}", file=sys.stderr)
        sys.exit(1)
