#!/usr/bin/env python3
"""Structural check for a placed ling TP16 rank pack and its experts manifest."""
import struct
import sys

EXPECTED_REVISION = "e0dfe7cd0f6e3b572bbbc0a8a84947469e428cc3"
EXPECTED_BYTES = 15725069824
EXPECTED_TENSORS = 730
EXPECTED_MANIFEST_RANGES = 40960


def main() -> int:
    pack_path, experts_path, rank = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with open(pack_path, "rb") as f:
        header = f.read(264)
        f.seek(0, 2)
        size = f.tell()
    fields = struct.unpack_from("<20I", header, 0)
    directory_offset, file_bytes = struct.unpack_from("<QQ", header, 80)
    revision = header[96:161].rstrip(b"\0").decode()
    checks = {
        "magic": fields[0] == 0x33474E4C,
        "format": fields[1] == 1,
        "tensors": fields[6] == EXPECTED_TENSORS,
        "tp_degree": fields[18] == 16,
        "tp_rank": fields[19] == rank,
        "revision": revision == EXPECTED_REVISION,
        "size": size == EXPECTED_BYTES and file_bytes == EXPECTED_BYTES,
        "directory_offset": directory_offset == 512,
    }
    with open(experts_path, "rb") as f:
        magic, version, ranges, reserved = struct.unpack("<IIII", f.read(16))
        f.seek(0, 2)
        experts_bytes = f.tell()
    checks["experts_magic"] = magic == 0x58504557
    checks["experts_version"] = version == 2
    checks["experts_ranges"] = ranges == EXPECTED_MANIFEST_RANGES
    checks["experts_size"] = experts_bytes == 16 + EXPECTED_MANIFEST_RANGES * 48
    checks["experts_reserved"] = reserved == 0
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        print("FAIL " + ",".join(failed))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
