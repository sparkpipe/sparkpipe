#!/usr/bin/env python3
"""Copy the replicated fast-path regions between two minimax.text.bf16.tp4
packs that share one directory layout. Used to propagate the repaired
EMBEDDING / FINAL_NORM / INPUT_NORM / POST_ATTENTION_NORM / Q_NORM /
K_NORM payloads from a warm-repaired rank pack to the other ranks whose
corrupted bytes are identical (the packer bug read every rank's replicated
tensors from the same shard position zero)."""
import argparse
import os
import struct
import sys
import tempfile
from pathlib import Path

HEADER_BYTES = 120
ENTRY_BYTES = 56
ENTRY_STRUCT = struct.Struct("<6I4Q")
CHUNK_BYTES = 8 * 1024 * 1024
REPLICATED_KINDS = [0, 1, 3, 4, 5, 6]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_bytes = args.source.stat().st_size
    dest_bytes = args.dest.stat().st_size
    if source_bytes != dest_bytes:
        raise SystemExit("pack sizes differ; directories may not match")
    with args.source.open("rb") as source, args.dest.open("rb") as dest:
        source.seek(16)
        count = struct.unpack("<I", source.read(4))[0]
        dest.seek(16)
        if struct.unpack("<I", dest.read(4))[0] != count:
            raise SystemExit("tensor counts differ")
        source.seek(HEADER_BYTES)
        dest.seek(HEADER_BYTES)
        regions = []
        for _ in range(count):
            entry = ENTRY_STRUCT.unpack(source.read(ENTRY_BYTES))
            other = ENTRY_STRUCT.unpack(dest.read(ENTRY_BYTES))
            if entry[:6] != other[:6]:
                raise SystemExit("directory mismatch between packs")
            kind, layer, _, rows, cols, _ = entry[:6]
            off, pb = entry[6], entry[7]
            if kind in REPLICATED_KINDS:
                if pb != rows * cols * 2:
                    raise SystemExit(f"region {kind}/{layer}: unexpected byte count")
                regions.append((off, pb))
        print(f"region-copy: regions={len(regions)} bytes={sum(p for _, p in regions)}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        with tempfile.NamedTemporaryFile(prefix=f".{args.output.name}.", suffix=".tmp",
                                         dir=args.output.parent, delete=False) as temp:
            temp_path = temp.name
            dest.seek(0)
            remaining = dest_bytes
            while remaining > 0:
                step = min(remaining, CHUNK_BYTES)
                chunk = dest.read(step)
                if len(chunk) != step:
                    raise SystemExit("short dest read")
                temp.write(chunk)
                remaining -= step
            for off, pb in regions:
                source.seek(off)
                temp.seek(off)
                remaining = pb
                while remaining > 0:
                    step = min(remaining, CHUNK_BYTES)
                    chunk = source.read(step)
                    if len(chunk) != step:
                        raise SystemExit("short source region read")
                    temp.write(chunk)
                    remaining -= step
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp_path, args.output)
    print(f"region-copy: wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
