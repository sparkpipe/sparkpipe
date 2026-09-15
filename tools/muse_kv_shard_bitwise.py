import argparse
import hashlib
import os
import re
import struct
import sys

MAGIC = 0x47534D55
HEADER_BYTES = 120
ENTRY_BYTES = 56
HEADER_STRUCT = struct.Struct("<26I2Q")
ENTRY_STRUCT = struct.Struct("<6I4Q")
KIND_QGKV = 7
KIND_FINAL_NORM = 1
NORM_KINDS = (1, 3, 4, 5, 6)


class PackError(ValueError):
    pass


class Pack:
    def __init__(self, path, rank):
        self.path = path
        self.rank = rank
        self.fd = open(path, "rb")
        header = self.fd.read(HEADER_BYTES)
        if len(header) != HEADER_BYTES:
            raise PackError(f"{path}: truncated header")
        fields = HEADER_STRUCT.unpack(header)
        if fields[0] != MAGIC:
            raise PackError(f"{path}: bad magic {fields[0]:#x}")
        self.entry_bytes = fields[3]
        self.entry_count = fields[4]
        self.entries = []
        self.fd.seek(HEADER_BYTES)
        for _ in range(self.entry_count):
            raw = self.fd.read(self.entry_bytes)
            if len(raw) != self.entry_bytes:
                raise PackError(f"{path}: truncated entry table")
            kind, layer, weight_format, rows, columns, _reserved, \
                payload_offset, payload_bytes, _pad, scale_bytes = \
                ENTRY_STRUCT.unpack(raw)
            self.entries.append({"kind": kind, "layer": layer, "rows": rows,
                                 "columns": columns,
                                 "offset": payload_offset,
                                 "bytes": payload_bytes,
                                 "scale_bytes": scale_bytes})

    def kv_span(self, layer, kv_heads, head_dim, local_heads):
        rows = local_heads * 2 * head_dim + 2 * head_dim
        for entry in self.entries:
            if entry["kind"] == KIND_QGKV and entry["layer"] == layer:
                if entry["rows"] != rows:
                    raise PackError(
                        f"{self.path}: qgkv layer {layer} rows {entry['rows']} "
                        f"!= expected {rows}")
                columns = entry["columns"]
                base = entry["offset"] + local_heads * 2 * head_dim * columns * 2
                length = 2 * head_dim * columns * 2
                return base, length
        raise PackError(f"{self.path}: no qgkv entry for layer {layer}")

    def read(self, offset, length):
        self.fd.seek(offset)
        data = self.fd.read(length)
        if len(data) != length:
            raise PackError(f"{self.path}: short read at {offset}")
        return data

    def whole_entries(self):
        for entry in self.entries:
            if entry["kind"] in NORM_KINDS:
                yield entry


def sha256_span(pack, offset, length):
    digest = hashlib.sha256()
    cursor = offset
    remaining = length
    while remaining > 0:
        chunk = pack.read(cursor, min(1 << 20, remaining))
        digest.update(chunk)
        cursor += len(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def rank_of(path, explicit):
    if explicit is not None:
        return explicit
    match = re.search(r"rank0*(\d+)", os.path.basename(path))
    if match is None:
        raise PackError(f"{path}: cannot infer rank from filename")
    return int(match.group(1))


def geometry(packs):
    counts = {p.entry_count for p in packs}
    if len(counts) != 1:
        raise PackError(f"entry counts disagree across packs: {counts}")
    return None


def compare(arguments):
    packs = []
    for index, path in enumerate(arguments.pack):
        rank = rank_of(path, arguments.rank[index]
                       if index < len(arguments.rank) else None)
        packs.append(Pack(path, rank))
    geometry(packs)
    groups = {}
    for pack in packs:
        groups.setdefault(pack.rank * arguments.kv_heads
                          // arguments.tp_degree, []).append(pack)
    failures = 0
    checked = 0
    for group, members in sorted(groups.items()):
        if len(members) < 2:
            print(f"group {group}: rank(s) "
                  f"{[m.rank for m in members]} single member, skipped")
            continue
        reference = members[0]
        for layer in range(arguments.layers):
            base, length = reference.kv_span(layer, arguments.kv_heads,
                                             arguments.head_dim,
                                             arguments.local_heads)
            want = sha256_span(reference, base, length)
            for other in members[1:]:
                base2, length2 = other.kv_span(layer, arguments.kv_heads,
                                               arguments.head_dim,
                                               arguments.local_heads)
                got = sha256_span(other, base2, length2)
                checked += 1
                if got != want:
                    failures += 1
                    print(f"MISMATCH kv group {group} layer {layer}: "
                          f"rank {reference.rank} {want[:16]} != "
                          f"rank {other.rank} {got[:16]}")
    for entry in packs[0].whole_entries():
        want = sha256_span(packs[0], entry["offset"], entry["bytes"])
        for other in packs[1:]:
            matches = [e for e in other.whole_entries()
                       if e["kind"] == entry["kind"]
                       and e["layer"] == entry["layer"]]
            if len(matches) != 1:
                raise PackError(f"{other.path}: replicated entry "
                                f"{entry['kind']}/{entry['layer']} not found")
            got = sha256_span(other, matches[0]["offset"], matches[0]["bytes"])
            checked += 1
            if got != want:
                failures += 1
                print(f"MISMATCH replicated kind {entry['kind']} layer "
                      f"{entry['layer']}: rank {packs[0].rank} != "
                      f"rank {other.rank}")
    print(f"checked {checked} spans across {len(packs)} packs, "
          f"{failures} mismatches")
    if failures:
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", action="append", required=True)
    parser.add_argument("--rank", action="append", type=int, default=[])
    parser.add_argument("--tp-degree", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--local-heads", type=int, default=2)
    parser.add_argument("--layers", type=int, default=52)
    arguments = parser.parse_args()
    if len(arguments.pack) < 2:
        print("need at least two --pack paths", file=sys.stderr)
        return 2
    return compare(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
