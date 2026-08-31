#!/usr/bin/env python3
"""In-place provenance patch for glm5_next TP16 rank packs.

The packs built by the original packer carry zeroed source_config_sha256 /
pack_recipe_sha256 header fields; the module header validation (wide-prefill
era) rejects zeroed provenance. This patcher writes the TRUE values into the
existing 21.7GB pack in place (header bytes only):

  source_config_sha256 = sha256(source model config.json)
  pack_recipe_sha256   = sha256(pack directory region -- the serialized
                               tensor plan, self-verifying from the file)

Derives the pack path from the hostname when --pack is not given
(spark<hex> -> rank int(hex,16)).
"""
import argparse
import hashlib
import json
import socket
import struct
import sys
from pathlib import Path

HEADER_BYTES = 264
ENTRY_BYTES = 64
CONFIG_OFFSET = 193
RECIPE_OFFSET = 225


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", help="pack path (default: derive from hostname)")
    ap.add_argument("--source-config", default="/mnt/model-warm/glm-5.3-flash/config.json")
    args = ap.parse_args()

    if args.pack:
        pack = Path(args.pack)
    else:
        host = socket.gethostname().strip()
        idx = int(host[5:], 16)
        pack = Path.home() / "sparkdata" / "glm5_next.tp16" / "packs" / f"glm5_next_stage.tp16.rank{idx}.g5nsp"
    if not pack.exists():
        print(f"FAIL: pack {pack} missing")
        return 1

    config_sha = hashlib.sha256(Path(args.source_config).read_bytes()).hexdigest()
    with open(pack, "r+b") as f:
        header = f.read(HEADER_BYTES)
        tensor_count, = struct.unpack_from("<I", header, 24)
        directory_offset, = struct.unpack_from("<Q", header, 80)
        f.seek(directory_offset)
        recipe_sha = hashlib.sha256(f.read(tensor_count * ENTRY_BYTES)).hexdigest()

        existing_cfg = header[CONFIG_OFFSET:CONFIG_OFFSET + 32]
        existing_rec = header[RECIPE_OFFSET:RECIPE_OFFSET + 32]
        if existing_cfg != bytes(32) or existing_rec != bytes(32):
            print(f"SKIP: provenance already present (cfg={existing_cfg.hex()[:8]}...)")
            return 0

        f.seek(CONFIG_OFFSET)
        f.write(bytes.fromhex(config_sha))
        f.seek(RECIPE_OFFSET)
        f.write(bytes.fromhex(recipe_sha))

    with open(pack, "rb") as f:
        header = f.read(HEADER_BYTES)
    assert header[CONFIG_OFFSET:CONFIG_OFFSET + 32] == bytes.fromhex(config_sha)
    assert header[RECIPE_OFFSET:RECIPE_OFFSET + 32] == bytes.fromhex(recipe_sha)
    print(f"PATCHED {pack.name}: config_sha={config_sha[:12]}... recipe_sha={recipe_sha[:12]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
