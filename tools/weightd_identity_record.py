#!/usr/bin/env python3
"""Emit a weightd identity record to STDOUT — consumed by test consumers so
that no identity string ever flows from argv into an IPC payload.

Layout (little-endian, matches SparkWeightdIdentity + path suffix):
  u64 geometry_fingerprint
  u64 arena_bytes
  u32 abi_version        (SPARK_WEIGHTD_IPC_ABI_VERSION)
  u32 topology
  u32 reserved0 = 0
  u32 reserved1 = 0
  64B model, 128B revision, 65B pack_sha256 (hex, NUL-padded), 7B reserved
  u32 path_bytes, path_bytes path (the placed rank pack)

usage: MODEL=dsv4flash REVISION=7872f01b... PACK_SHA=hex PACK=/path
       GEOMETRY=fnvhex weightd_identity_record.py > identity.bin
"""
import os
import struct
import sys

ID_BYTES = 64
REVISION_BYTES = 128
SHA_BYTES = 65
ABI_VERSION = 2


def fnv1a(data):
    h = 1469598103934665603
    for b in data:
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def main():
    pack = os.environ.get("PACK")
    model = os.environ.get("MODEL", "dsv4flash")
    revision = os.environ.get("REVISION", "")
    sha_hex = os.environ.get("PACK_SHA", "")
    assert pack, "PACK env required"
    assert revision, "REVISION env required"
    assert len(sha_hex) == 64, "PACK_SHA must be a 64-hex digest"
    pack_path = os.path.realpath(pack)
    assert os.path.isfile(pack_path), pack_path
    arena_bytes = os.path.getsize(pack_path)

    with open(pack_path, "rb") as f:
        header = f.read(80)
    geometry = fnv1a(header) ^ fnv1a(model)

    identity = struct.pack("<QQIIII",
        geometry, arena_bytes, ABI_VERSION, 16, 0, 0)
    identity += model.encode().ljust(ID_BYTES, b"\0")
    identity += revision.encode().ljust(REVISION_BYTES, b"\0")
    identity += sha_hex.encode().ljust(SHA_BYTES, b"\0")
    identity += b"\0" * 7
    path_bytes = pack_path.encode()
    out = identity + struct.pack("<I", len(path_bytes)) + path_bytes
    sys.stdout.buffer.write(out)
    sys.stdout.buffer.flush()
    print(f"identity record: {len(out)} bytes for {model}/{revision[:12]} "
        f"arena={arena_bytes}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
