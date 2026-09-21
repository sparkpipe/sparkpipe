#!/usr/bin/env python3
"""Mechanically verify placed DeepSeek-V4 family stage packs without the warm
checkpoint (the local-NVMe sweep form), and emit the placed receipt pair.

Two families, keyed on --family:

  dsv41flash (default) - DSV4.1-Flash packs (tools/dsv41_flash_stagepack.py,
      257-byte header, 64-byte directory records, mxfp4 experts + fp8
      spine). Verifies header identity (magic, geometry, codecs, tp/rank,
      revision), the full per-entry contract (kind/layer closure via the
      kind map, payload type/codec/scale encoding, groups, rows/columns,
      byte math), the byte-exact directory layout (256-aligned cursor walk),
      and the file extent.

  dsv4flash - DSV4-Flash TP16 rank packs (tools/dsv4_tp16_stagepack.py
      sharding of a tools/dsv4_stagepack.py full pack, 80-byte header,
      40-byte records). Rebuilds the expected per-rank directory from the
      contract records and the sharding rules (row indices, column slices,
      MTP replication, global filtering) and proves every entry field, the
      exact offsets, and the extent.

On PASS with --emit-receipt, writes the placed receipt pair:
<pack>.receipt.json (kind sparkpipe.dsv4.pack-verify-receipt.v1, carrying
sha256 and output_sha256) plus the <pack>.sha256 sidecar in sha256sum
format. A pre-existing receipt is never overwritten; --receipt
cross-checks an existing one instead.

Exit 0 only on PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import re

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from spark_pack_common import sha256_file, write_receipt  # noqa: E402

REPO_ROOT = Path(_TOOLS_DIR).resolve().parent
DEFAULT_CONTRACT = REPO_ROOT / "model_contracts" / "dsv4_flash.json"


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, str(Path(_TOOLS_DIR) / path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Fail(Exception):
    pass


def fail(check: str, detail: str) -> None:
    raise Fail(f"FAIL {check}: {detail}")


def parse_rank(path: Path) -> int:
    match = re.search(r"rank(\d+)", path.name)
    if not match:
        fail("rank", f"cannot derive the rank from {path.name}; pass --rank")
    return int(match.group(1))


def verify_dsv41(pack: Path) -> dict:
    """The DSV4.1-Flash wire contract, from tools/dsv41_flash_stagepack.py."""
    dsv41 = load_module("dsv41_flash_stagepack_verify_tables", "dsv41_flash_stagepack.py")
    header_bytes = dsv41.HEADER_BYTES
    entry_bytes = dsv41.ENTRY_BYTES
    align = dsv41.ALIGN
    size = pack.stat().st_size
    with pack.open("rb") as f:
        head = f.read(header_bytes)
        if len(head) != header_bytes:
            fail("header", f"{pack.name}: short header {len(head)}")
        vals = struct.unpack_from("<20I2Q", head, 0)
        revision = head[96:161].rstrip(b"\0").decode("utf-8", "replace")
        digests = [head[161 + slot * 32:193 + slot * 32].hex() for slot in range(3)]
        (magic, version, hb, eb, abi, flags, tensor_count, stage_count,
         stage_index, first_layer, layer_count, total_layers, hidden, vocab,
         experts, linear_codec, expert_codec, tp_degree, tp_rank) = vals[:19]
        directory_offset, file_bytes = vals[20], vals[21]
        if magic != dsv41.MAGIC:
            fail("magic", f"{pack.name}: {magic:#x}")
        if version != 1 or hb != header_bytes or eb != entry_bytes:
            fail("layout", f"{pack.name}: v{version} header {hb} entry {eb}")
        if abi != 1 or flags != 0:
            fail("flags", f"{pack.name}: abi {abi} flags {flags:#x}")
        if (hidden, vocab, experts) != (dsv41.HIDDEN, dsv41.VOCAB, dsv41.ROUTED_EXPERTS):
            fail("geometry", f"{pack.name}: {hidden}x{vocab}x{experts}")
        if linear_codec != dsv41.CODEC_FP8 or expert_codec not in (
                dsv41.CODEC_MXFP4, dsv41.CODEC_NVFP4):
            fail("codecs", f"{pack.name}: linear {linear_codec} expert {expert_codec}")
        expert_codec_name = "nvfp4" if expert_codec == dsv41.CODEC_NVFP4 else "mxfp4"
        if (stage_count, stage_index) != (1, 0) or first_layer != 0 or \
                layer_count != dsv41.LAYER_COUNT or total_layers != dsv41.LAYER_COUNT:
            fail("stages", f"{pack.name}: stage {stage_index}/{stage_count} "
                           f"layers {first_layer}+{layer_count}/{total_layers}")
        if not revision:
            fail("revision", f"{pack.name}: empty model revision field")
        dir_expect = (header_bytes + align - 1) & ~(align - 1)
        if directory_offset != dir_expect:
            fail("directory_offset", f"{pack.name}: {directory_offset}")
        if file_bytes != size:
            fail("extent", f"{pack.name}: header {file_bytes}, stat {size}")
        if directory_offset + tensor_count * entry_bytes > size:
            fail("directory", f"{pack.name}: directory exceeds file")
        f.seek(directory_offset)
        raw = f.read(tensor_count * entry_bytes)
        if len(raw) != tensor_count * entry_bytes:
            fail("directory", f"{pack.name}: truncated directory")

    expected = []
    order = [(kind, dsv41.GLOBAL_LAYER) for kind in range(dsv41.KIND_COUNT)
             if dsv41.is_global(kind)]
    for layer in range(dsv41.LAYER_COUNT):
        for kind in range(dsv41.K_ATTN_NORM, dsv41.KIND_COUNT):
            if dsv41.kind_in_layer(kind, layer):
                order.append((kind, layer))
    for kind, layer in order:
        rows, cols, groups = dsv41.entry_shape(kind, tp_degree)
        payload_type, codec, scale_enc, pbytes, sbytes = dsv41.entry_layout(
            kind, tp_degree, expert_codec_name)
        expected.append(dict(kind=kind, layer=layer, payload_type=payload_type,
                             codec=codec, scale_encoding=scale_enc, groups=groups,
                             rows=rows, cols=cols, payload_bytes=pbytes,
                             scale_bytes=sbytes))
    if tensor_count != len(expected):
        fail("tensor_count", f"{pack.name}: {tensor_count} on the wire, "
                             f"kind map expects {len(expected)}")
    cursor = 0
    payload_base = (directory_offset + tensor_count * entry_bytes + align - 1) & ~(align - 1)
    for index, entry in enumerate(expected):
        cursor = (cursor + align - 1) & ~(align - 1)
        entry["payload_offset"] = cursor + payload_base
        cursor += entry["payload_bytes"]
        if entry["scale_bytes"]:
            cursor = (cursor + align - 1) & ~(align - 1)
            entry["scale_offset"] = cursor + payload_base
            cursor += entry["scale_bytes"]
        else:
            entry["scale_offset"] = 0
    if cursor + payload_base != size:
        fail("extent", f"{pack.name}: layout ends at {cursor + payload_base}, "
                       f"file is {size}")
    problems = []
    for index, (want, got) in enumerate(zip(expected, [struct.unpack_from(
            "<8I4Q", raw, i * entry_bytes) for i in range(tensor_count)])):
        got_fields = dict(kind=got[0], layer=got[1], payload_type=got[2],
                          codec=got[3], scale_encoding=got[4], groups=got[5],
                          rows=got[6], cols=got[7], payload_offset=got[8],
                          payload_bytes=got[9], scale_offset=got[10],
                          scale_bytes=got[11])
        for field, value in want.items():
            if got_fields[field] != value:
                problems.append(
                    f"entry[{index}] kind={want['kind']} layer={hex(want['layer'])}: "
                    f"{field} {got_fields[field]} != {value}")
                break
    if problems:
        fail("directory", problems[0] + (f" (+{len(problems) - 1} more)" if len(problems) > 1 else ""))
    return dict(family="dsv41flash", pack=pack.name, file_bytes=size,
                tensor_count=tensor_count, tp_degree=tp_degree, tp_rank=tp_rank,
                revision=revision, contract_sha256=digests[0],
                config_sha256=digests[1], recipe_sha256=digests[2])


def verify_dsv4flash(pack: Path, contract_path: Path, rank: int | None) -> dict:
    """The DSV4-Flash TP16 rank contract: contract records sharded by
    tools/dsv4_tp16_stagepack.py's rules."""
    stagepack = load_module("dsv4_stagepack_verify_tables", "dsv4_stagepack.py")
    tp16 = load_module("dsv4_tp16_stagepack_verify_tables", "dsv4_tp16_stagepack.py")
    if rank is None:
        rank = parse_rank(pack)
    header_struct = tp16.HEADER
    entry_struct = tp16.ENTRY
    size = pack.stat().st_size
    with pack.open("rb") as f:
        head = f.read(header_struct.size)
        if len(head) != header_struct.size:
            fail("header", f"{pack.name}: short header {len(head)}")
        header = header_struct.unpack(head)
        (magic, version, hb, eb, abi, linear_codec, expert_codec, kv_codec,
         tensor_count, first_layer, layer_count, total_layers, hidden, vocab,
         experts, packed_mtp, directory_offset, file_bytes) = header
        if magic != tp16.MAGIC:
            fail("magic", f"{pack.name}: {magic:#x}")
        if version != tp16.VERSION:
            fail("version", f"{pack.name}: {version}")
        if hb != header_struct.size or eb != entry_struct.size:
            fail("layout", f"{pack.name}: header {hb} entry {eb}")
        if abi != 1 or (linear_codec, expert_codec, kv_codec) != (5, 7, 1):
            fail("codecs", f"{pack.name}: abi {abi} codecs "
                           f"{(linear_codec, expert_codec, kv_codec)}")
        if (hidden, vocab, experts) != (tp16.HIDDEN, tp16.VOCAB, tp16.EXPERTS):
            fail("geometry", f"{pack.name}: {hidden}x{vocab}x{experts}")
        if (first_layer, layer_count) != tp16.layer_slice(1, 0) or \
                total_layers != tp16.LAYERS:
            fail("stages", f"{pack.name}: layers {first_layer}+{layer_count}/{total_layers}")
        if packed_mtp != stagepack.MTP_LAYER_COUNT_MAX:
            fail("mtp", f"{pack.name}: packed_mtp_layer_count {packed_mtp}")
        if directory_offset != header_struct.size:
            fail("directory_offset", f"{pack.name}: {directory_offset}")
        if file_bytes != size:
            fail("extent", f"{pack.name}: header {file_bytes}, stat {size}")
        if directory_offset + tensor_count * entry_struct.size > size:
            fail("directory", f"{pack.name}: directory exceeds file")
        f.seek(directory_offset)
        raw = f.read(tensor_count * entry_struct.size)
        if len(raw) != tensor_count * entry_struct.size:
            fail("directory", f"{pack.name}: truncated directory")

    contract = json.loads(contract_path.read_text())
    records = stagepack.build_records(contract, 0, tp16.LAYERS)
    expected = []
    for record in records:
        entry = (record.kind, record.layer, record.weight_format,
                 record.rows, record.columns, 0, 0, 0)
        try:
            planned = tp16.plan_entry(entry, rank, 1, 0)
        except stagepack.PackFailure as error:
            if str(error) == "filtered":
                continue
            raise
        planned_kind, planned_layer, weight, rows, cols = planned[0][:5]
        pbytes = tp16.payload_bytes(weight, rows, cols)
        sbytes = tp16.scale_bytes(weight, rows, cols)
        expected.append(dict(kind=planned_kind, layer=planned_layer,
                             weight=weight, rows=rows, cols=cols,
                             payload_bytes=pbytes, scale_bytes=sbytes))
    if tensor_count != len(expected):
        fail("tensor_count", f"{pack.name}: {tensor_count} on the wire, "
                             f"rank {rank} sharding expects {len(expected)}")
    cursor = header_struct.size + entry_struct.size * len(expected)
    for entry in expected:
        entry["payload_offset"] = cursor
        cursor += entry["payload_bytes"]
        if entry["scale_bytes"]:
            entry["scale_offset"] = cursor
            cursor += entry["scale_bytes"]
        else:
            entry["scale_offset"] = 0
    if cursor != size:
        fail("extent", f"{pack.name}: layout ends at {cursor}, file is {size}")
    problems = []
    for index in range(tensor_count):
        got = struct.unpack_from("<6I2Q", raw, index * entry_struct.size)
        want = expected[index]
        got_fields = dict(kind=got[0], layer=got[1], weight=got[2], rows=got[3],
                          cols=got[4], reserved=got[5], payload_offset=got[6],
                          scale_offset=got[7])
        want["reserved"] = 0
        for field in ("kind", "layer", "weight", "rows", "cols", "reserved",
                      "payload_offset", "scale_offset"):
            if got_fields[field] != want[field]:
                problems.append(
                    f"entry[{index}] kind={want['kind']} layer={hex(want['layer'])}: "
                    f"{field} {got_fields[field]} != {want[field]}")
                break
    if problems:
        fail("directory", problems[0] + (f" (+{len(problems) - 1} more)" if len(problems) > 1 else ""))
    return dict(family="dsv4flash", pack=pack.name, file_bytes=size,
                tensor_count=tensor_count, tp_degree=tp16.TP_DEGREE,
                tp_rank=rank, first_layer=first_layer, layer_count=layer_count)


def check_receipt(path: Path, summary: dict) -> None:
    receipt = json.loads(path.read_text())
    for field in ("sha256", "output_sha256", "file_bytes", "tensor_count"):
        if field in receipt and receipt[field] not in (None, summary[field]):
            fail("receipt", f"{path.name}: {field} {receipt[field]} != "
                            f"{summary[field]}")


def emit(summary: dict, pack: Path) -> None:
    receipt_path = Path(str(pack) + ".receipt.json")
    if receipt_path.is_file():
        print(f"receipt {receipt_path.name} already present; left untouched")
        return
    receipt = {
        "kind": "sparkpipe.dsv4.pack-verify-receipt.v1",
        "tool": "tools/dsv41_verify_pack.py",
        "family": summary["family"],
        "pack": pack.name,
        "tp_degree": summary["tp_degree"],
        "tp_rank": summary["tp_rank"],
        "tensor_count": summary["tensor_count"],
        "file_bytes": summary["file_bytes"],
        "verify_mode": "structure-only (no warm checkpoint)",
        "sha256": summary["sha256"],
        "output_sha256": summary["sha256"],
    }
    if summary.get("revision"):
        receipt["revision"] = summary["revision"]
    write_receipt(receipt, receipt_path, suffix=None)
    sidecar = Path(str(pack) + ".sha256")
    sidecar.write_text(f"{summary['sha256']}  {pack.name}\n")
    print(f"receipt {receipt_path.name} + {sidecar.name} written")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--family", choices=("dsv41flash", "dsv4flash"),
                        default="dsv41flash")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT,
        help="dsv4_flash contract for --family dsv4flash (default: "
             "model_contracts/dsv4_flash.json)")
    parser.add_argument("--rank", type=int, default=None,
        help="the rank pack's tp rank (default: parsed from the filename)")
    parser.add_argument("--receipt", type=Path,
        help="cross-check this existing receipt instead of emitting")
    parser.add_argument("--emit-receipt", action="store_true",
        help="write <pack>.receipt.json + <pack>.sha256 on PASS")
    args = parser.parse_args()

    try:
        if args.family == "dsv41flash":
            summary = verify_dsv41(args.pack)
        else:
            summary = verify_dsv4flash(args.pack, args.contract, args.rank)
        digest = sha256_file(args.pack)
        summary["sha256"] = digest
    except Fail as failure:
        print(failure, file=sys.stderr)
        return 1
    if args.receipt is not None:
        check_receipt(args.receipt, summary)
    print(f"PASS {args.pack.name}: {summary['family']} header identity, "
          f"{summary['tensor_count']} directory entries (tp "
          f"{summary['tp_degree']}/{summary['tp_rank']}), layout+extent, "
          f"sha256 {digest[:16]}...")
    if args.emit_receipt:
        emit(summary, args.pack)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Fail as failure:
        print(failure, file=sys.stderr)
        sys.exit(1)
