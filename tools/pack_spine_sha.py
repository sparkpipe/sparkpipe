#!/usr/bin/env python3
"""Per-rank spine digest of a resident stage pack (design §2.2).

The spine is every directory entry whose tensor kind is not a routed-expert
kind. For each spine entry the record is

  [tensor_kind, layer, group_count, rows, columns, payload_type, weight_codec,
   scale_encoding, payload_sha256, scale_sha256]

(scale_sha256 is the empty-input SHA when the entry has no scale plane). The
records are sorted, written one canonical JSON array per line, and
spine_digest = SHA-256 of those lines. Two packs have equal spine digests iff
every spine tensor has the same identity, geometry, codec and bytes, wherever
it sits in the file. An E, K or D comparison is refused unless all per-rank
digests are equal (tools/ab_receipt.py).

The directory layout (264-byte header "<20I2Q..." with tensor_count at word 6
and the directory offset at byte 80; 64-byte "<8I4Q" entries) is shared by the
resident stage packs; --layout selects the magic and the expert kinds.

usage:
  pack_spine_sha.py --layout NAME PACK [PACK ...] [--records OUT.jsonl] [--json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import struct
import sys
from pathlib import Path

FORMAT = "sparkpipe-spine-digest-v1"
HEADER_BYTES = 264
ENTRY_BYTES = 64
LAYOUTS = {
    "glm5_next": {"magic": 0x33584C47, "expert_kinds": (22, 23)},
    "glm52": {"magic": 0x32534C47, "expert_kinds": (22, 23)},
}
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
CHUNK = 64 << 20


class PackError(ValueError):
    pass


def region_sha256(view, offset: int, size: int, file_bytes: int) -> str:
    if size == 0:
        return EMPTY_SHA256
    if offset + size > file_bytes:
        raise PackError(f"region {offset}+{size} runs past the end of the pack ({file_bytes} bytes)")
    digest = hashlib.sha256()
    position = offset
    end = offset + size
    while position < end:
        step = min(CHUNK, end - position)
        digest.update(view[position:position + step])
        position += step
    return digest.hexdigest()


def spine_records(path: Path, layout: dict) -> tuple:
    size = path.stat().st_size
    if size < HEADER_BYTES:
        raise PackError(f"{path}: {size} bytes is shorter than the header")
    with open(path, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as view:
        words = struct.unpack_from("<20I", view, 0)
        directory_offset, file_bytes = struct.unpack_from("<QQ", view, 80)
        if words[0] != layout["magic"]:
            raise PackError(f"{path}: magic {words[0]:#x} is not {layout['magic']:#x}")
        if words[3] != ENTRY_BYTES or words[2] != HEADER_BYTES:
            raise PackError(f"{path}: header/entry sizes {words[2]}/{words[3]} are not {HEADER_BYTES}/{ENTRY_BYTES}")
        if file_bytes != size:
            raise PackError(f"{path}: header file_bytes {file_bytes} differs from the file size {size}")
        count = words[6]
        if directory_offset + count * ENTRY_BYTES > size:
            raise PackError(f"{path}: directory runs past the end of the pack")
        records, experts = [], 0
        for index in range(count):
            e = struct.unpack_from("<8I4Q", view, directory_offset + index * ENTRY_BYTES)
            kind, layer, payload_type, codec, scale_encoding, groups, rows, columns = e[:8]
            payload_offset, payload_bytes, scale_offset, scale_bytes = e[8:]
            if kind in layout["expert_kinds"]:
                experts += 1
                continue
            records.append([kind, layer, groups, rows, columns, payload_type, codec, scale_encoding,
                            region_sha256(view, payload_offset, payload_bytes, size),
                            region_sha256(view, scale_offset, scale_bytes, size) if scale_bytes else EMPTY_SHA256])
    records.sort()
    return records, experts, count, words


def digest(records: list) -> str:
    lines = "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)
    return hashlib.sha256(lines.encode("ascii")).hexdigest()


def spine_digest(path, layout_name: str) -> dict:
    layout = LAYOUTS[layout_name]
    records, experts, count, words = spine_records(Path(path), layout)
    return {"format": FORMAT, "pack": str(path), "layout": layout_name, "tensor_count": count,
            "spine_entries": len(records), "expert_entries": experts,
            "stage_index": words[8], "tp_rank": words[19], "tp_degree": words[18],
            "spine_digest": digest(records), "records": records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", required=True, choices=sorted(LAYOUTS))
    parser.add_argument("--records")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("packs", nargs="+")
    args = parser.parse_args()
    results = []
    try:
        for pack in args.packs:
            results.append(spine_digest(pack, args.layout))
    except (PackError, OSError) as error:
        print(f"pack_spine_sha: REFUSED: {error}", file=sys.stderr)
        return 1
    if args.records:
        with open(args.records, "w") as handle:
            for result in results:
                for record in result["records"]:
                    handle.write(json.dumps({"pack": result["pack"], "record": record}) + "\n")
    for result in results:
        result.pop("records")
        if args.json:
            print(json.dumps(result, sort_keys=True))
        else:
            print(f"{result['spine_digest']}  {result['pack']}  spine={result['spine_entries']} experts={result['expert_entries']} tp_rank={result['tp_rank']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
