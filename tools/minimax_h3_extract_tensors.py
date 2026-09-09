#!/usr/bin/env python3
"""Extract named tensors from a warm-copy component as raw little-endian files.

Gate tooling for the minimax lane: streams each requested tensor with pread
(never holds the shard), writes <name with dots -> underscores>.bin plus a
manifest of `name rows columns file` lines for the C validators. Payloads are
copied verbatim. Refuses to run if any requested name is missing (fail-closed
census).
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

DTYPE_BYTES = {"BF16": 2, "F32": 4, "F64": 8, "I64": 8, "U8": 1, "I8": 1, "BOOL": 1}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-dir", type=Path, required=True)
    parser.add_argument("--index", default="diffusion_pytorch_model.safetensors.index.json")
    parser.add_argument("--names", required=True, help="comma-separated tensor names")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    wanted = [name.strip() for name in args.names.split(",") if name.strip()]
    index_path = args.component_dir / args.index
    weight_map = json.loads(index_path.read_text())["weight_map"]
    missing = [name for name in wanted if name not in weight_map]
    if missing:
        raise SystemExit(f"missing tensors in {index_path}: {missing}")
    handles = {}
    manifest_lines = []
    for name in wanted:
        shard = weight_map[name]
        if shard not in handles:
            handles[shard] = (args.component_dir / shard).open("rb")
        file = handles[shard]
        (header_len,) = struct.unpack("<Q", file.read(8))
        header = json.loads(file.read(header_len))
        file.seek(0)
        info = header[name]
        shape = info["shape"]
        dtype = info["dtype"]
        if len(shape) >= 2:
            rows = 1
            for dim in shape[:-1]:
                rows *= dim
            columns = shape[-1]
        elif len(shape) == 1:
            rows = 1
            columns = shape[0]
        else:
            rows = 1
            columns = 1
        base, end = info["data_offsets"]
        length = end - base
        if rows * columns * DTYPE_BYTES[dtype] != length:
            raise SystemExit(f"shape/dtype mismatch for {name}")
        target = name.replace(".", "_") + ".bin"
        with (args.out / target).open("wb") as out:
            file.seek(base)
            remaining = length
            while remaining > 0:
                chunk = file.read(min(remaining, 1 << 24))
                if not chunk:
                    raise SystemExit(f"short read on {name}")
                out.write(chunk)
                remaining -= len(chunk)
        manifest_lines.append(f"{name} {rows} {columns} {target} {dtype}")
    for file in handles.values():
        file.close()
    (args.out / "manifest.txt").write_text("\n".join(manifest_lines) + "\n")
    print(f"extracted {len(wanted)} tensors into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
