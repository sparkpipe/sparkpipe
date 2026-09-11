#!/usr/bin/env python3
"""Convert .npz fixtures to raw little-endian binaries with shape-encoded names.

Setup-time tool for the minimax lane gates: the C validators read flat
binaries only, no zip/npy parsing. Each member <name> with shape (a,b,...)
and dtype <u2/<f4/<f8/<i8/<i4 becomes <name>__axbx....<ext> in the output
directory. Payload bytes are copied verbatim (never requantized).
"""

from __future__ import annotations

import argparse
import ast
import json
import struct
import sys
import zipfile
from pathlib import Path

DESCR = {"<u2": ("u16", 2), "<f4": ("f32", 4), "<f8": ("f64", 8), "<i8": ("i64", 8), "<i4": ("i32", 4)}


def npy_payload(member: bytes) -> tuple[str, tuple[int, ...], bytes]:
    if member[:6] != b"\x93NUMPY":
        raise SystemExit("not an npy member")
    major = member[6]
    (header_len,) = struct.unpack_from("<H" if major == 1 else "<I", member, 8)
    offset = 10 if major == 1 else 12
    header = ast.literal_eval(member[offset:offset + header_len].decode("latin1"))
    descr = header["descr"]
    if header["fortran_order"]:
        raise SystemExit("fortran-order arrays unsupported")
    return descr, tuple(header["shape"]), member[offset + header_len:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    with zipfile.ZipFile(args.npz) as archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue
            name = member[:-4]
            descr, shape, payload = npy_payload(archive.read(member))
            if descr not in DESCR:
                raise SystemExit(f"unsupported descr {descr} for {name}")
            ext, esize = DESCR[descr]
            expected = esize * (1 if not shape else 1)
            for dim in shape:
                expected *= dim
            if len(payload) != expected:
                raise SystemExit(f"payload size mismatch for {name}")
            target = f"{name}__{'x'.join(str(d) for d in shape) or '1'}.{ext}"
            (args.out / target).write_bytes(payload)
            manifest.append({"name": name, "file": target, "shape": shape, "dtype": ext})
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(f"converted {len(manifest)} arrays into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
