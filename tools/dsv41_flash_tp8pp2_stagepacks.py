#!/usr/bin/env python3
"""Cut the dsv41flash mxfp4 TP8xPP2 mesh stage packs from the placed TP8 rank set.

The 2026-09-11 operator ruling serves dsv41 flash as a TP8xPP2 mesh over the
16 sparks (stage_count 2, 20/20 layer split, world rank w = stage_index*8 +
tp_rank -> spark w). The fleet holds the verified full-model TP8 rank packs
(39.2 GB/node, rank r -> spark r + replica r+8). This tool cuts each verified
rank pack into its two stage packs byte-verbatim:

  cut SOURCE_PACK SOURCE_SHA256 OUT_DIR STAGE_INDEX
      fail-closed unless the source pack matches its placed sha256 sidecar;
      writes stage<index>.rank<r>.spstage + .experts-less pack + .receipt.json
      + .sha256 (the weightd manifest is produced separately by the module's
      compiled dsv41_flash_experts_manifest tool against the cut pack).

  verify STAGE_PACK SOURCE_SHA256
      independent re-verify of a cut pack: header fields, directory subset
      (every entry byte-identical to its source entry apart from offsets),
      payload/scale regions byte-equal to the source pack regions.

Stage semantics follow modules/dsv41_flash_resident_decode_stage
(spark_dsv41_flash_stagepack_format.h): stage 0 owns K_EMBEDDING and layers
[first, first+20), stage 1 owns K_FINAL_NORM + K_LM_HEAD and layers
[20, 40); entry shapes/codecs never change (same TP shards as the source).
Exit 0 = the demanded step completed and proved; nonzero names the failure.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys

MAGIC = 0x31413444
FORMAT_VERSION = 1
HEADER_BYTES = 257
ENTRY_BYTES = 64
ALIGN = 256
LAYER_COUNT = 40
STAGE_COUNT = 2
GLOBAL_LAYER = 0xFFFFFFFF
K_EMBEDDING = 0
K_FINAL_NORM = 1
K_LM_HEAD = 2
KIND_COUNT = 35
HEADER_STRUCT = struct.Struct("<20I2Q")
ENTRY_STRUCT = struct.Struct("<8I4Q")
CHUNK = 16 << 20

RECEIPT_KIND = "sparkpipe.dsv41flash.tp8pp2.stagepack-receipt.v1"


class Fail(Exception):
    pass


def is_global(kind: int) -> bool:
    return kind <= K_LM_HEAD


def align_up(value: int) -> int:
    return (value + ALIGN - 1) & ~(ALIGN - 1)


def stage_window(stage_index: int) -> tuple[int, int]:
    if stage_index not in (0, 1):
        raise Fail(f"stage index {stage_index} outside the (8,2) mesh")
    first = stage_index * (LAYER_COUNT // STAGE_COUNT)
    return first, LAYER_COUNT // STAGE_COUNT


def stage_entries(entries: list[dict], stage_index: int) -> list[dict]:
    first, count = stage_window(stage_index)
    picked = []
    if stage_index == 0:
        picked = [e for e in entries if e["layer"] == GLOBAL_LAYER and e["kind"] == K_EMBEDDING]
        if len(picked) != 1:
            raise Fail(f"source pack carries {len(picked)} embedding globals, expected 1")
    picked += [e for e in entries
               if e["layer"] != GLOBAL_LAYER and first <= e["layer"] < first + count]
    if stage_index == 1:
        head = [e for e in entries
                if e["layer"] == GLOBAL_LAYER and e["kind"] in (K_FINAL_NORM, K_LM_HEAD)]
        if len(head) != 2:
            raise Fail(f"source pack carries {len(head)} final-head globals, expected 2")
        picked = head + picked
    for entry in picked:
        if entry["layer"] != GLOBAL_LAYER and not (first <= entry["layer"] < first + count):
            raise Fail(f"stage selector picked out-of-window layer {entry['layer']}")
        if is_global(entry["kind"]) and entry["layer"] != GLOBAL_LAYER:
            raise Fail(f"global kind {entry['kind']} carries a layer index")
    return picked


def read_header(pack_path: str, full_model: bool = True) -> tuple[dict, list[dict]]:
    with open(pack_path, "rb") as handle:
        raw = handle.read(HEADER_BYTES)
    if len(raw) != HEADER_BYTES:
        raise Fail(f"source header truncated: {len(raw)} bytes")
    fields = HEADER_STRUCT.unpack_from(raw)
    if fields[0] != MAGIC or fields[1] != FORMAT_VERSION:
        raise Fail("source pack magic/version mismatch")
    header = dict(magic=fields[0], format_version=fields[1], header_bytes=fields[2],
                  entry_bytes=fields[3], codec_abi=fields[4], flags=fields[5],
                  tensor_count=fields[6], stage_count=fields[7], stage_index=fields[8],
                  first_layer_index=fields[9], layer_count=fields[10],
                  total_layer_count=fields[11], hidden=fields[12], vocab=fields[13],
                  experts=fields[14], linear_codec=fields[15], expert_codec=fields[16],
                  tp_degree=fields[17], tp_rank=fields[18], reserved=fields[19],
                  directory_offset=fields[20], file_bytes=fields[21],
                  revision=raw[96:161].split(b"\0", 1)[0].decode(),
                  contract_sha=raw[161:193].hex(), config_sha=raw[193:225].hex(),
                  recipe_sha=raw[225:257].hex())
    if full_model:
        if header["stage_count"] != 1 or header["stage_index"] != 0:
            raise Fail("source pack is not a full-model (stage_count 1) pack")
        if header["layer_count"] != LAYER_COUNT or header["first_layer_index"] != 0:
            raise Fail("source pack does not span the 40-layer stack")
    else:
        if header["stage_count"] != STAGE_COUNT or header["stage_index"] not in (0, 1):
            raise Fail("stage pack header does not declare the (8,2) mesh")
        first, count = stage_window(header["stage_index"])
        if header["first_layer_index"] != first or header["layer_count"] != count:
            raise Fail("stage pack window does not match the 20/20 split")
        if header["total_layer_count"] != LAYER_COUNT:
            raise Fail("stage pack total layer count drift")
    with open(pack_path, "rb") as handle:
        handle.seek(header["directory_offset"])
        raw_directory = handle.read(header["tensor_count"] * ENTRY_BYTES)
    if len(raw_directory) != header["tensor_count"] * ENTRY_BYTES:
        raise Fail("source directory truncated")
    entries = []
    for index in range(header["tensor_count"]):
        fields = ENTRY_STRUCT.unpack_from(raw_directory, index * ENTRY_BYTES)
        entries.append(dict(kind=fields[0], layer=fields[1], payload_type=fields[2],
                            codec=fields[3], scale_encoding=fields[4], groups=fields[5],
                            rows=fields[6], cols=fields[7], payload_offset=fields[8],
                            payload_bytes=fields[9], scale_offset=fields[10],
                            scale_bytes=fields[11]))
    return header, entries


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb", buffering=0) as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def check_source(source_pack: str, source_sha: str) -> str:
    actual = sha256_file(source_pack)
    if actual != source_sha:
        raise Fail(f"source pack sha {actual[:16]}... != placed sidecar {source_sha[:16]}...")
    return actual


def output_name(tp_rank: int, stage_index: int) -> str:
    return f"stage{stage_index}.rank{tp_rank}.spstage"


def copy_region(source_fd: int, dest_fd: int, src_offset: int, dst_offset: int, length: int,
                label: str) -> None:
    remaining = length
    while remaining > 0:
        block = os.pread(source_fd, min(CHUNK, remaining), src_offset + (length - remaining))
        if not block:
            raise Fail(f"source short read in {label} at {src_offset}")
        written = os.pwrite(dest_fd, block, dst_offset + (length - remaining))
        if written != len(block):
            raise Fail(f"short write in {label}")
        remaining -= len(block)


def verify_regions(source_pack: str, stage_pack: str, entries: list[dict],
                   source_entries_by_key: dict) -> None:
    source_fd = os.open(source_pack, os.O_RDONLY)
    try:
        stage_fd = os.open(stage_pack, os.O_RDONLY)
        try:
            for entry in entries:
                key = (entry["kind"], entry["layer"])
                origin = source_entries_by_key.get(key)
                if origin is None:
                    raise Fail(f"stage entry {key} absent from the source pack")
                for plane in ("payload", "scale"):
                    length = entry[f"{plane}_bytes"]
                    if not length:
                        continue
                    src = entry[f"{plane}_offset"]
                    dst = origin[f"{plane}_offset"]
                    remaining = length
                    while remaining > 0:
                        step = min(CHUNK, remaining)
                        done = length - remaining
                        want = os.pread(source_fd, step, dst + done)
                        have = os.pread(stage_fd, step, src + done)
                        if want != have:
                            raise Fail(
                                f"byte mismatch kind={entry['kind']} layer={entry['layer']} "
                                f"{plane} at +{done}")
                        remaining -= step
        finally:
            os.close(stage_fd)
    finally:
        os.close(source_fd)


def cut(source_pack: str, source_sha: str, out_dir: str, stage_index: int) -> dict:
    actual_sha = check_source(source_pack, source_sha)
    header, entries = read_header(source_pack)
    if header["tp_degree"] != 8:
        raise Fail(f"source pack tp_degree {header['tp_degree']} != 8")
    picked = stage_entries(entries, stage_index)
    first, count = stage_window(stage_index)
    directory_offset = align_up(HEADER_BYTES)
    payload_base = align_up(directory_offset + len(picked) * ENTRY_BYTES)
    cursor = payload_base
    laid = []
    for entry in picked:
        payload_offset = align_up(cursor)
        cursor = payload_offset + entry["payload_bytes"]
        scale_offset = 0
        if entry["scale_bytes"]:
            scale_offset = align_up(cursor)
            cursor = scale_offset + entry["scale_bytes"]
        laid.append(dict(entry, payload_offset=payload_offset, scale_offset=scale_offset,
                         src_payload_offset=entry["payload_offset"],
                         src_scale_offset=entry["scale_offset"]))
    file_bytes = cursor
    tp_rank = header["tp_rank"]
    name = output_name(tp_rank, stage_index)
    stage_pack = os.path.join(out_dir, name)
    if os.path.exists(stage_pack):
        raise Fail(f"output already exists (never overwrite): {stage_pack}")
    source_fd = os.open(source_pack, os.O_RDONLY)
    try:
        fd = os.open(stage_pack, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            os.ftruncate(fd, file_bytes)
            head = bytearray(HEADER_BYTES)
            HEADER_STRUCT.pack_into(head, 0, MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES,
                                    header["codec_abi"], header["flags"], len(laid),
                                    STAGE_COUNT, stage_index, first, count,
                                    header["total_layer_count"], header["hidden"],
                                    header["vocab"], header["experts"],
                                    header["linear_codec"], header["expert_codec"],
                                    header["tp_degree"], tp_rank, header["reserved"],
                                    directory_offset, file_bytes)
            head[96:161] = header["revision"].encode()
            head[161:193] = bytes.fromhex(header["contract_sha"])
            head[193:225] = bytes.fromhex(header["config_sha"])
            head[225:257] = bytes.fromhex(header["recipe_sha"])
            os.pwrite(fd, bytes(head), 0)
            directory = bytearray()
            for entry in laid:
                directory += ENTRY_STRUCT.pack(
                    entry["kind"], entry["layer"], entry["payload_type"], entry["codec"],
                    entry["scale_encoding"], entry["groups"], entry["rows"], entry["cols"],
                    entry["payload_offset"], entry["payload_bytes"],
                    entry["scale_offset"], entry["scale_bytes"])
            os.pwrite(fd, bytes(directory), directory_offset)
            for entry in laid:
                for plane in ("payload", "scale"):
                    length = entry[f"{plane}_bytes"]
                    if length:
                        copy_region(source_fd, fd, entry[f"src_{plane}_offset"],
                                    entry[f"{plane}_offset"], length,
                                    f"kind={entry['kind']} layer={entry['layer']} {plane}")
        finally:
            os.close(fd)
    finally:
        os.close(source_fd)
    _, verify_entries = read_header(stage_pack, full_model=False)
    source_entries_by_key = {(e["kind"], e["layer"]): e for e in entries}
    verify_regions(source_pack, stage_pack, verify_entries, source_entries_by_key)
    stage_header, _ = read_header(stage_pack, full_model=False)
    if stage_header["file_bytes"] != os.path.getsize(stage_pack):
        raise Fail("stage pack size drift")
    output_sha = sha256_file(stage_pack)
    receipt = {
        "kind": RECEIPT_KIND,
        "tool": "tools/dsv41_flash_tp8pp2_stagepacks.py",
        "source_pack": os.path.abspath(source_pack),
        "source_sha256": actual_sha,
        "source_receipt_sha": source_sha,
        "stage_index": stage_index,
        "stage_count": STAGE_COUNT,
        "first_layer_index": first,
        "layer_count": count,
        "tp_degree": 8,
        "tp_rank": tp_rank,
        "world_rank": stage_index * 8 + tp_rank,
        "tensor_count": len(verify_entries),
        "file_bytes": file_bytes,
        "output_sha256": output_sha,
        "revision": header["revision"],
        "verify": "byte-verbatim region identity against the placed tp8 rank pack",
        "verdict": "PASS",
    }
    with open(stage_pack + ".receipt.json", "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=1)
        handle.write("\n")
    with open(stage_pack + ".sha256", "w", encoding="utf-8") as handle:
        handle.write(f"{output_sha}  {name}\n")
    print(f"cut {name}: tensors={len(verify_entries)} bytes={file_bytes} sha={output_sha[:16]}... "
          f"verify=PASS")
    return receipt


def verify(source_pack: str, source_sha: str, stage_pack: str) -> dict:
    header, entries = read_header(source_pack)
    with open(stage_pack + ".sha256", encoding="utf-8") as handle:
        placed_sha = handle.read().split()[0]
    actual = check_source(source_pack, source_sha)
    stage_header, stage_entries_list = read_header(stage_pack, full_model=False)
    first, count = stage_window(stage_header["stage_index"])
    expected = stage_entries(entries, stage_header["stage_index"])
    if len(expected) != len(stage_entries_list):
        raise Fail(f"stage tensor count {len(stage_entries_list)} != expected {len(expected)}")
    by_key = {(e["kind"], e["layer"]): e for e in entries}
    verify_regions(source_pack, stage_pack, stage_entries_list, by_key)
    if stage_header["file_bytes"] != os.path.getsize(stage_pack):
        raise Fail("stage pack size drift")
    output_sha = sha256_file(stage_pack)
    if output_sha != placed_sha:
        raise Fail(f"stage pack sha {output_sha[:16]}... != placed sidecar {placed_sha[:16]}...")
    receipt = {
        "kind": RECEIPT_KIND,
        "tool": "tools/dsv41_flash_tp8pp2_stagepacks.py",
        "mode": "verify",
        "stage_pack": os.path.abspath(stage_pack),
        "source_pack": os.path.abspath(source_pack),
        "source_sha256": actual,
        "stage_index": stage_header["stage_index"],
        "first_layer_index": first,
        "layer_count": count,
        "tp_degree": stage_header["tp_degree"],
        "tp_rank": stage_header["tp_rank"],
        "tensor_count": len(stage_entries_list),
        "file_bytes": stage_header["file_bytes"],
        "output_sha256": output_sha,
        "verdict": "PASS",
    }
    print(f"verify {os.path.basename(stage_pack)}: tensors={len(stage_entries_list)} "
          f"sha={output_sha[:16]}... PASS")
    return receipt


def main(argv: list) -> int:
    if len(argv) == 6 and argv[1] == "cut":
        cut(argv[2], argv[3], argv[4], int(argv[5]))
        return 0
    if len(argv) == 5 and argv[1] == "verify":
        verify(argv[2], argv[3], argv[4])
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except Fail as error:
        print(f"dsv41_flash_tp8pp2_stagepacks FAIL: {error}", file=sys.stderr)
        sys.exit(1)
    except OSError as error:
        print(f"dsv41_flash_tp8pp2_stagepacks OS FAIL: {error}", file=sys.stderr)
        sys.exit(3)
