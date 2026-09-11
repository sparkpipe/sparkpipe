#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

_spec = importlib.util.spec_from_file_location(
    "qwen38_max_stagepack_tables", str(Path(_TOOLS_DIR) / "qwen38_stagepack.py"))
_tables = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tables)
_tables.EXPERT_CODEC = "nvfp4"
_tables.STRIP_MTP = True

ENTRY_STRUCT = struct.Struct("<6I4Q")
HEADER2_STRUCT = struct.Struct("<28I2Q")


def split_summary(pack: Path) -> dict:
    size = pack.stat().st_size
    with pack.open("rb") as f:
        header = HEADER2_STRUCT.unpack(f.read(128))
        tp_degree, tp_rank = header[26], header[27]
        tensor_count = header[4]
        raw = f.read(tensor_count * ENTRY_STRUCT.size)
    refs = {
        (ref.kind, ref.layer): ref
        for ref in _tables.build_inventory(header[7], header[6])
    }
    buckets = ("expert", "spine_sharded", "spine_replicated")
    splits = {b: {"tensors": 0, "payload_bytes": 0, "scale_bytes": 0} for b in buckets}
    for index in range(tensor_count):
        kind, layer, fmt, rows, cols, group, off, p_bytes, s_off, s_bytes = \
            ENTRY_STRUCT.unpack_from(raw, index * ENTRY_STRUCT.size)
        ref = refs[(kind, layer)]
        plan = _tables.build_tp_plan(ref, tp_degree, tp_rank)
        if ref.kind in (6, 7, 8):
            bucket = "expert"
        elif plan is None:
            bucket = "spine_replicated"
        else:
            bucket = "spine_sharded"
        splits[bucket]["tensors"] += 1
        splits[bucket]["payload_bytes"] += p_bytes
        splits[bucket]["scale_bytes"] += s_bytes
    return {
        "pack": str(pack),
        "file_bytes": size,
        "tensor_count": tensor_count,
        "tp_degree": tp_degree,
        "tp_rank": tp_rank,
        "mtp_layer_count": header[25],
        "splits": splits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="emit the SPINE/EXPERT split "
                                     "summary for one qwen38_max TP16 rank pack")
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    summary = split_summary(args.pack)
    digest = hashlib.sha256()
    with args.pack.open("rb") as f:
        while True:
            chunk = f.read(16 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    summary["sha256"] = digest.hexdigest()
    text = json.dumps(summary, indent=2) + "\n"
    out = args.out or Path(str(args.pack) + ".splits.json")
    out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
