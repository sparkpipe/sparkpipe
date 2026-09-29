import argparse
import json
import struct
import sys

MAGIC = 0x44525053
VERSION = 1
HEADER = struct.Struct("<IIIIQQ")
ENTRY_FIXED = struct.Struct("<QQII")


def write_table(path, entries, depth, vocab):
    if depth < 1 or depth > 32:
        raise ValueError("depth must be 1..32")
    rows = []
    for (sequence, position), tokens in sorted(entries.items()):
        tokens = [int(t) for t in tokens][:depth]
        if not tokens:
            continue
        if any(t < 0 or t >= vocab for t in tokens):
            raise ValueError(f"sequence {sequence} position {position}: token outside vocabulary {vocab}")
        rows.append(ENTRY_FIXED.pack(int(sequence), int(position), len(tokens), 0) + struct.pack(f"<{depth}I", *(tokens + [0] * (depth - len(tokens)))))
    if not rows:
        raise ValueError("a draft table needs at least one entry")
    with open(path, "wb") as handle:
        handle.write(HEADER.pack(MAGIC, VERSION, depth, vocab, len(rows), 0))
        for row in rows:
            handle.write(row)
    return len(rows)


def read_table(path):
    with open(path, "rb") as handle:
        raw = handle.read()
    magic, version, depth, vocab, count, reserved = HEADER.unpack_from(raw, 0)
    if magic != MAGIC or version != VERSION or reserved != 0 or not 1 <= depth <= 32:
        raise ValueError(f"{path}: not a version {VERSION} draft table")
    entry_bytes = ENTRY_FIXED.size + 4 * depth
    if len(raw) != HEADER.size + count * entry_bytes:
        raise ValueError(f"{path}: {len(raw)} bytes for {count} entries of {entry_bytes}")
    entries = {}
    previous = None
    for index in range(count):
        offset = HEADER.size + index * entry_bytes
        sequence, position, length, pad = ENTRY_FIXED.unpack_from(raw, offset)
        tokens = list(struct.unpack_from(f"<{depth}I", raw, offset + ENTRY_FIXED.size))
        if pad != 0 or not 1 <= length <= depth or any(tokens[length:]) or any(t >= vocab for t in tokens[:length]):
            raise ValueError(f"{path}: entry {index} is malformed")
        if previous is not None and (sequence, position) <= previous:
            raise ValueError(f"{path}: entries are not strictly sorted at {index}")
        previous = (sequence, position)
        entries[(sequence, position)] = tokens[:length]
    return {"depth": depth, "vocab": vocab, "entries": entries}


def main():
    parser = argparse.ArgumentParser(description="inspect a recorded draft table (key: sequence id and the position of the last committed token; value: the draft chain)")
    parser.add_argument("table")
    args = parser.parse_args()
    table = read_table(args.table)
    lengths = [len(tokens) for tokens in table["entries"].values()]
    print(json.dumps({"depth": table["depth"], "vocab": table["vocab"], "entries": len(lengths),
                      "sequences": len({key[0] for key in table["entries"]}), "mean_chain": sum(lengths) / len(lengths)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
