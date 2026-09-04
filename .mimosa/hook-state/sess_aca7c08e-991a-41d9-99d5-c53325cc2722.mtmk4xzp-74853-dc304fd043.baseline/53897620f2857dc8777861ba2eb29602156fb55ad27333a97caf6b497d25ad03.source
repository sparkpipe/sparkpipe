#!/usr/bin/env python3
"""One-line JSON audit of a deployed K3 pack: sha256 + manifest config.

CPU-only (mmap + hash); kimi-k3 lane's deployed-pack audit.
"""
import hashlib
import json
import mmap
import struct
import sys
from pathlib import Path

MAGIC = 0x4B33504B


def main():
    path = Path(sys.argv[1])
    with open(path, "rb") as handle:
        raw = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        magic, version, length = struct.unpack_from("<IIQ", raw, 0)
        info = {
            "file": str(path),
            "bytes": len(raw),
            "magic_ok": magic == MAGIC,
            "version": version,
        }
        if magic != MAGIC or version != 2:
            print(json.dumps(info))
            return 1
        manifest = json.loads(raw[16:16 + length])
        cfg = manifest["config"]
        info.update({
            "sha256": hashlib.sha256(raw).hexdigest(),
            "tensors": len(manifest["tensors"]),
            "first_layer": cfg.get("first_layer"),
            "layers": cfg.get("layers"),
            "tp_degree": cfg.get("tp_degree"),
            "tp_rank": cfg.get("tp_rank"),
        })
        raw.close()
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
