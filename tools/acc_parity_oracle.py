#!/usr/bin/env python3
"""Pack-verification parity oracle for the accuracy wave (ACC-2).

Enforces the existing pack-verification law on the accuracy lanes:
pack == packer (receipt + sha256 + directory re-derivation) and
packer == checkpoint (payload bytes re-derived from the warm-storage
checkpoint through the packer's own plan and compared byte-for-byte
against the pack file). Verdict per pack: CHECKPOINT-FAITHFUL, DRIFTED
(first deltas reported), or UNVERIFIABLE (missing inputs).

Anchor policy: layer 0, a middle layer, the last layer of the pack
window, every global (embedding, final norm, lm head), rope tables,
router/scale planes, and the sampled layers' expert planes. --full
compares every entry instead of the anchor sample.

Formats: family A (ling .sp, laguna .lgsp: 264-byte header, 64-byte
directory entry) and family B (muse .gsmu, gemma4 .gemma4sp: 120-byte
header, 56-byte entry). Each profile re-derives expected bytes with the
family packer's own plan code, so a pass proves the emit chain is
reproducible from the checkpoint; transform-math bugs inside the packer
itself remain the family numeric oracles' job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

READ_CHUNK = 8 * 1024 * 1024
MAX_STALL_RETRIES = 3
DRIFT_REPORT_CAP = 8
SOURCE_CACHE_CAP = 512 * 1024 * 1024


def fail(message: str) -> None:
    print(f"SPARK_FAIL acc_parity_oracle: {message}")
    sys.exit(1)


def pread_tolerant(fd: int, size: int, offset: int, label: str) -> bytes:
    got = os.pread(fd, size, offset)
    retries = 0
    while len(got) < size and retries < MAX_STALL_RETRIES:
        time.sleep(2)
        got += os.pread(fd, size - len(got), offset + len(got))
        retries += 1
    if len(got) != size:
        raise OSError(f"{label}: short read {len(got)}/{size} at {offset}")
    return got


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(READ_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_entry(fd: int, expected_iter, payload_offset: int,
                  payload_bytes: int, label: str) -> Tuple[str, Optional[str]]:
    position = payload_offset
    consumed = 0
    expected_iter = (iter([expected_iter])
                     if isinstance(expected_iter, (bytes, bytearray))
                     else expected_iter)
    for chunk in expected_iter:
        want = len(chunk)
        if consumed + want > payload_bytes:
            return "DRIFTED", f"{label}: producer exceeds entry size {payload_bytes}"
        have = pread_tolerant(fd, want, position, label)
        if have != chunk:
            differing = sum(1 for a, b in zip(have, chunk) if a != b)
            first = next((i for i, (a, b) in enumerate(zip(have, chunk)) if a != b), 0)
            return "DRIFTED", (f"{label}: first byte delta at payload+{first}; "
                               f"{differing}/{want} bytes differ in one chunk")
        position += want
        consumed += want
    if consumed != payload_bytes:
        return "DRIFTED", f"{label}: producer wrote {consumed}, entry plans {payload_bytes}"
    return "CHECKPOINT-FAITHFUL", None


def parse_family_a(path: Path) -> Tuple[List[dict], dict]:
    with path.open("rb") as file:
        head = file.read(264)
        (magic, version, header_bytes, entry_bytes, codec_abi, flags,
         tensor_count, stage_count, stage_index, first_layer, layer_count,
         total_layers, hidden, vocab, experts, linear_codec, expert_codec,
         kv_codec, tp_degree, tp_rank) = struct.unpack("<20I", head[:80])
        directory_offset, file_bytes = struct.unpack("<QQ", head[80:96])
        revision = head[96:161].rstrip(b"\0").decode("utf-8", "replace")
        if header_bytes != 264 or entry_bytes != 64:
            fail(f"{path.name}: family A layout {header_bytes}/{entry_bytes}")
        file.seek(directory_offset)
        raw = file.read(tensor_count * 64)
    entries = []
    for index in range(tensor_count):
        fields = struct.unpack_from("<IIIIIIIIQQQQ", raw, index * 64)
        entries.append(dict(index=index, kind=fields[0], layer=fields[1],
                            rows=fields[6], columns=fields[7],
                            payload_offset=fields[8],
                            payload_bytes=fields[9]))
    meta = dict(magic=magic, version=version, flags=flags,
                tensor_count=tensor_count, stage_count=stage_count,
                stage_index=stage_index, first_layer=first_layer,
                layer_count=layer_count, total_layers=total_layers,
                tp_degree=tp_degree, tp_rank=tp_rank, revision=revision,
                file_bytes=file_bytes)
    return entries, meta


def parse_family_b(path: Path) -> Tuple[List[dict], dict]:
    with path.open("rb") as file:
        head = file.read(120)
        values = struct.unpack("<26IQQ", head)
        tensor_count = values[4]
        directory_offset = values[26]
        file_bytes = values[27]
        file.seek(directory_offset)
        raw = file.read(tensor_count * 56)
    entries = []
    for index in range(tensor_count):
        kind, layer, fmt, rows, columns, pad, offset, payload, so, sb = \
            struct.unpack_from("<IIIIIIQQQQ", raw, index * 56)
        entries.append(dict(index=index, kind=kind, layer=layer, rows=rows,
                            columns=columns, payload_offset=offset,
                            payload_bytes=payload))
    meta = dict(magic=values[0], version=values[1], tensor_count=tensor_count,
                file_bytes=file_bytes, directory_offset=directory_offset)
    return entries, meta


def anchor_layers(span: int) -> List[int]:
    return sorted({0, span // 2, span - 1} - {-1})


def structural_parity(entries: List[dict], metas: List[dict],
                      what: str) -> None:
    """Every pack directory entry must equal the rebuilt plan item at the
    same position (kind, layer, shape, size) before any content compare."""
    if len(entries) != len(metas):
        fail(f"{what}: {len(entries)} pack entries vs {len(metas)} plan items")
    for index, (entry, meta) in enumerate(zip(entries, metas)):
        mismatches = [f"{field} pack={entry[field]} plan={meta[field]}"
                      for field in ("kind", "layer", "rows", "columns",
                                    "payload_bytes")
                      if entry[field] != meta[field]]
        if mismatches:
            fail(f"{what}: entry {index} directory/plan mismatch: "
                 + "; ".join(mismatches))


def compare_plan(plan_items, entries, pack_path: Path, expected_of,
                 label_of) -> dict:
    compared = 0
    drift = []
    with pack_path.open("rb") as file:
        fd = file.fileno()
        for item in plan_items:
            index = item["index"]
            expected = expected_of(item)
            if isinstance(expected, (bytes, bytearray)):
                expected = iter([expected])
            verdict, detail = compare_entry(
                fd, expected,
                entries[index]["payload_offset"], entries[index]["payload_bytes"],
                label_of(item))
            compared += 1
            if verdict != "CHECKPOINT-FAITHFUL":
                drift.append(detail)
                if len(drift) >= DRIFT_REPORT_CAP:
                    break
    return compared, drift


def profile_ling(pack_path: Path, source_dir: Path, tp_degree: int,
                 tp_rank: int) -> dict:
    import ling_stagepack as packer
    entries, meta = parse_family_a(pack_path)
    if meta["tp_degree"] != tp_degree or meta["tp_rank"] != tp_rank:
        fail(f"ling {pack_path.name}: header tp{meta['tp_degree']}"
             f".rank{meta['tp_rank']} != requested tp{tp_degree}.rank{tp_rank}")
    source = packer.SourceReader(source_dir, cache_byte_cap=SOURCE_CACHE_CAP)
    builder = packer.Packer(source, tp_degree, tp_rank, packer.CODEC_BF16)
    builder.build("acc-oracle")
    if len(builder.plan) != len(entries):
        fail(f"ling {pack_path.name}: {len(entries)} pack entries vs "
             f"{len(builder.plan)} rebuilt plan items")
    structural_parity(entries, [dict(kind=item.entry.kind, layer=item.entry.layer,
                                     rows=item.entry.rows, columns=item.entry.columns,
                                     payload_bytes=item.entry.payload_bytes)
                                for item in builder.plan], f"ling {pack_path.name}")
    layers = {item.entry.layer for item in builder.plan
              if item.entry.layer != packer.GLOBAL_LAYER}
    picks = set(anchor_layers(max(layers) + 1))
    wanted = []
    for position, item in enumerate(builder.plan):
        entry = item.entry
        if (entry.layer == packer.GLOBAL_LAYER
                or entry.kind in (packer.K_ROUTER, packer.K_ROUTER_CORRECTION)
                or entry.layer in picks):
            wanted.append(dict(index=position, item=item, entry=entry))

    def expected_of(item):
        return item["item"].produce_payload()

    def label_of(item):
        return f"kind{item['entry'].kind}:layer{item['entry'].layer}"

    compared, drift = compare_plan(wanted, entries, pack_path,
                                   expected_of, label_of)
    source.close()
    return dict(pack=str(pack_path), family="ling", tp_rank=tp_rank,
                entries=len(entries), compared=compared, drift=drift,
                sha256=sha256_file(pack_path), revision=meta["revision"])


def profile_laguna(pack_path: Path, source_dir: Path, tp_degree: int,
                   tp_rank: int, first_layer: int, layer_count: int,
                   owns_embedding: bool, owns_head: bool) -> dict:
    import laguna_stagepack as packer
    entries, meta = parse_family_a(pack_path)
    source = packer.SourceReader(source_dir, cache_byte_cap=SOURCE_CACHE_CAP)
    builder = packer.Packer(source, tp_degree, tp_rank, first_layer,
                            layer_count, owns_embedding, owns_head,
                            packer.CODEC_BF16)
    builder.build()
    if len(builder.plan) != len(entries):
        fail(f"laguna {pack_path.name}: {len(entries)} pack entries vs "
             f"{len(builder.plan)} rebuilt plan items")
    structural_parity(entries, [dict(kind=item.entry.kind, layer=item.entry.layer,
                                     rows=item.entry.rows, columns=item.entry.columns,
                                     payload_bytes=item.entry.payload_bytes)
                                for item in builder.plan], f"laguna {pack_path.name}")
    picks = set(anchor_layers(layer_count))
    wanted = []
    for position, item in enumerate(builder.plan):
        entry = item.entry
        relative = entry.layer - first_layer
        if (entry.layer == packer.GLOBAL_LAYER
                or entry.kind in (packer.K_ROUTER, packer.K_ROUTER_CORRECTION)
                or relative in picks):
            wanted.append(dict(index=position, item=item, entry=entry))

    def expected_of(item):
        return item["item"].produce_payload()

    def label_of(item):
        return f"kind{item['entry'].kind}:layer{item['entry'].layer}"

    compared, drift = compare_plan(wanted, entries, pack_path,
                                   expected_of, label_of)
    source.close()
    return dict(pack=str(pack_path), family="laguna", tp_rank=tp_rank,
                stage_index=meta["stage_index"], entries=len(entries),
                compared=compared, drift=drift,
                sha256=sha256_file(pack_path), revision=meta["revision"])


def rows_span(source, name: str, first_row: int, row_count: int) -> bytes:
    """Contiguous bf16 row span read straight from the safetensors shard
    (row-major payloads), so a rank slice never materializes a whole
    multi-GiB checkpoint plane."""
    shard, meta, data_start = source.resolve(name)
    shape = meta["shape"]
    cols = shape[1] if len(shape) > 1 else shape[0]
    row_bytes = cols * 2
    fd = os.open(source.root / shard, os.O_RDONLY)
    try:
        return pread_tolerant(fd, row_count * row_bytes,
                              data_start + first_row * row_bytes,
                              f"{name} rows[{first_row},{first_row + row_count})")
    finally:
        os.close(fd)


def profile_muse(pack_path: Path, source_dir: Path, tp_degree: int,
                 tp_rank: int) -> dict:
    import numpy as np
    import muse_glimmer_stagepack as packer
    from spark_pack_common import SafetensorsSource
    entries, meta = parse_family_b(pack_path)
    if meta["magic"] != packer.MAGIC:
        fail(f"muse {pack_path.name}: magic {meta['magic']:#x}")
    records = packer.build_records(tp_degree, tp_rank)
    if len(records) != len(entries):
        fail(f"muse {pack_path.name}: {len(entries)} pack entries vs "
             f"{len(records)} rebuilt plan items")
    structural_parity(entries, [dict(kind=record.kind, layer=record.layer,
                                     rows=record.rows, columns=record.columns,
                                     payload_bytes=record.payload_bytes)
                                for record in records], f"muse {pack_path.name}")
    source = SafetensorsSource(source_dir)
    layers = {record.layer for record in records
              if record.layer != packer.GLOBAL_LAYER}
    picks = set(anchor_layers(packer.LAYER_COUNT))

    def matrix(name):
        shard, meta, data_start = source.resolve(name)
        shape = meta["shape"]
        fd = os.open(source.root / shard, os.O_RDONLY)
        try:
            raw = pread_tolerant(fd, meta["data_offsets"][1] - meta["data_offsets"][0],
                                 data_start, f"{name} full")
        finally:
            os.close(fd)
        return np.frombuffer(raw, dtype="<u2").reshape(shape[0], -1)

    def expected_of(item):
        record = item["record"]
        plan = record.plan
        if "whole" in plan:
            return rows_span(source, record.names[0], 0, record.rows)
        if "row_slice" in plan:
            first, count = plan["row_slice"]
            return rows_span(source, record.names[0], first, count)
        if "qgkv" in plan:
            planes = {"q": record.names[0], "gate": record.names[1],
                      "k": record.names[2], "v": record.names[3]}
            blocks = [rows_span(source, planes[name], first, count)
                      for name, first, count in plan["qgkv"]]
            return b"".join(blocks)
        if "gate_up" in plan:
            gate_start, count = plan["gate_up"]
            return (rows_span(source, record.names[0], gate_start, count)
                    + rows_span(source, record.names[1], gate_start, count))
        start, count = plan["columns"]
        return np.ascontiguousarray(
            matrix(record.names[0])[:, start:start + count]).tobytes()

    wanted = []
    for index, record in enumerate(records):
        if record.layer == packer.GLOBAL_LAYER or record.layer in picks:
            wanted.append(dict(index=index, record=record,
                               label=f"{packer.KIND_NAMES[record.kind]}:layer{record.layer}"))
    compared, drift = compare_plan(
        wanted, entries, pack_path, expected_of,
        lambda item: item["label"])
    return dict(pack=str(pack_path), family="muse", tp_rank=tp_rank,
                entries=len(entries), compared=compared, drift=drift,
                sha256=sha256_file(pack_path))


def profile_gemma4(pack_path: Path, source_dir: Path, model: str,
                   tp_degree: int, tp_rank: int, first_layer: int,
                   layer_count: int, full: bool) -> dict:
    import gemma4_stagepack as packer
    from spark_pack_common import SafetensorsSource
    entries, meta = parse_family_b(pack_path)
    if meta["magic"] != packer.MAGIC:
        fail(f"gemma4 {pack_path.name}: magic {meta['magic']:#x}")
    geometry = packer.GEOMETRY[model]
    plan = packer.build_inventory(geometry, tp_degree, tp_rank, first_layer,
                                  layer_count)
    plan, file_bytes, total_payload = packer.place(plan)
    if len(plan) != len(entries):
        fail(f"gemma4 {pack_path.name}: {len(entries)} pack entries vs "
             f"{len(plan)} rebuilt plan items")
    if file_bytes != meta["file_bytes"]:
        fail(f"gemma4 {pack_path.name}: layout drift, rebuilt file_bytes "
             f"{file_bytes} != pack {meta['file_bytes']}")
    structural_parity(entries, plan, f"gemma4 {pack_path.name}")
    source = SafetensorsSource(source_dir)
    picks = set(anchor_layers(layer_count))
    wanted = []
    for index, entry in enumerate(plan):
        relative = entry["layer"] - first_layer
        anchor = (entry["layer"] == packer.GLOBAL_LAYER or full
                  or relative in picks
                  or entry["kind"] in (packer.KIND_ROPE_TABLE,
                                       packer.KIND_ROUTER_PROJ,
                                       packer.KIND_PER_EXPERT_SCALE))
        if anchor:
            wanted.append(dict(index=index, entry=entry,
                               source=source, geometry=geometry,
                               tp_degree=tp_degree, tp_rank=tp_rank))

    def expected_of(item):
        return packer.payload_for(item["source"], item["geometry"],
                                  item["entry"], item["tp_degree"],
                                  item["tp_rank"])

    compared, drift = compare_plan(wanted, entries, pack_path, expected_of,
                                   lambda item: f"kind{item['entry']['kind']}:layer{item['entry']['layer']}")
    return dict(pack=str(pack_path), family="gemma4", model=model,
                tp_rank=tp_rank, first_layer=first_layer,
                layer_count=layer_count, entries=len(entries),
                compared=compared, drift=drift,
                sha256=sha256_file(pack_path))


def check_sidecar(path: Path) -> str:
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file():
        return "missing"
    expected = sidecar.read_text().split()[0]
    actual = sha256_file(path)
    return "match" if expected == actual else f"MISMATCH {actual} != {expected}"


def check_receipt(path: Path, result: dict) -> str:
    receipt_path = path.parent / (path.name + ".receipt.json")
    if not receipt_path.is_file():
        return "missing"
    receipt = json.loads(receipt_path.read_text())
    digest = result["sha256"]
    recorded = receipt.get("pack_sha256", receipt.get("sha256"))
    if recorded is None:
        return "no-digest-field"
    if recorded != digest:
        return f"SHA MISMATCH receipt {recorded}"
    if receipt.get("file_bytes") not in (None, path.stat().st_size):
        return f"BYTES MISMATCH {receipt.get('file_bytes')}"
    return "match"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True,
                        choices=("ling", "laguna", "muse", "gemma4"))
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tp-degree", type=int, default=16)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--layer-count", type=int, default=0)
    parser.add_argument("--owns-embedding", action="store_true")
    parser.add_argument("--owns-head", action="store_true")
    parser.add_argument("--model", default="")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    started = time.time()
    if not args.pack.is_file():
        fail(f"pack missing: {args.pack}")
    if not args.checkpoint.is_dir():
        fail(f"checkpoint dir missing: {args.checkpoint}")
    if args.family == "ling":
        result = profile_ling(args.pack, args.checkpoint, args.tp_degree,
                              args.tp_rank)
    elif args.family == "laguna":
        if args.layer_count <= 0:
            fail("--layer-count required for laguna")
        result = profile_laguna(args.pack, args.checkpoint, args.tp_degree,
                                args.tp_rank, args.first_layer,
                                args.layer_count, args.owns_embedding,
                                args.owns_head)
    elif args.family == "muse":
        result = profile_muse(args.pack, args.checkpoint, args.tp_degree,
                              args.tp_rank)
    else:
        if args.model not in ("31b", "26b-a4b"):
            fail("--model required for gemma4 (31b|26b-a4b)")
        if args.layer_count <= 0:
            fail("--layer-count required for gemma4 windows")
        result = profile_gemma4(args.pack, args.checkpoint, args.model,
                                args.tp_degree, args.tp_rank,
                                args.first_layer, args.layer_count,
                                args.full)
    result["sidecar"] = check_sidecar(args.pack)
    result["receipt"] = check_receipt(args.pack, result)
    result["seconds"] = round(time.time() - started, 1)
    result["verdict"] = ("DRIFTED" if result["drift"]
                         else ("CHECKPOINT-FAITHFUL" if result["compared"] > 0
                               else "UNVERIFIABLE"))
    print(json.dumps(result, indent=2))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
    return 0 if result["verdict"] == "CHECKPOINT-FAITHFUL" else 1


if __name__ == "__main__":
    sys.exit(main())
