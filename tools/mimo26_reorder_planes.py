#!/usr/bin/env python3
"""Reorder file-major staged mimo26 planes into plan span order (local repair).

The 2026-09-23 file-major emission ruling appends each plane's spans in
shard-visit order. A multi-span record whose tensors live in several source
shards gets its plane bytes permuted relative to the plan span order that
the pack directory, the assembler and the verifier all use. This tool
recomputes the deterministic append layout (the same shard grouping the
emitter used) and rewrites only the planes whose append order differs,
touching local staging only - no checkpoint bytes are read."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS))
import mimo26_stagepack as m  # noqa: E402

CHUNK = 1 << 20


def payload_span_bytes(span) -> int:
    if span.mode == m.SPAN_MX:
        return span.rows * (span.full_columns // 2)
    if span.mode == m.SPAN_DENSE:
        return span.rows * span.full_columns * m.element_bytes(span.dtype)
    if span.mode == m.SPAN_RECT:
        return span.rows * span.columns * m.element_bytes(span.dtype)
    raise SystemExit(f"unknown payload span mode {span.mode}")


def scale_span_bytes(span, source) -> int:
    if span.mode == m.SPAN_MX:
        return span.rows * (span.full_columns // m.MXFP4_GROUP)
    grid_columns = source.shard_header(
        source.weight_map[span.scale_name])[span.scale_name]["shape"][1]
    columns = span.scale_columns or grid_columns
    return span.scale_rows * columns * m.element_bytes("F32")


def span_bytes(span, plane, source) -> int:
    return payload_span_bytes(span) if plane else scale_span_bytes(span, source)


def append_layout(source, records):
    layout = {}
    by_shard = {}
    for record in records:
        for plane in (True, False):
            for span in record.spans:
                if not plane and not span.scale_name:
                    continue
                name = m.payload_name(span, plane)
                shard = source.weight_map.get(name)
                if shard is None:
                    raise SystemExit(f"span source not in index: {name}")
                by_shard.setdefault(shard, []).append((record, span, plane))
    for shard in sorted(by_shard):
        for record, span, plane in by_shard[shard]:
            length = span_bytes(span, plane, source)
            name = m.stage_name(record, "payload" if plane else "scale")
            layout.setdefault(name, []).append((record, span, plane, length))
    return layout


def reorder(stage_dir: Path, layout, source) -> tuple:
    fixed = skipped = 0
    for name, items in layout.items():
        path = stage_dir / name
        if not path.exists():
            continue
        record, plane = items[0][0], items[0][2]
        total = record.payload_bytes if plane else record.scale_bytes
        size = path.stat().st_size
        if size != total:
            raise SystemExit(
                f"{path}: staged {size} bytes, plan expects {total}; "
                "not a completed file-major plane - re-emit this rank")
        want = [(span, span_bytes(span, plane, source))
                for span in record.spans]
        if sum(length for _, length in want) != total:
            raise SystemExit(f"{path}: span lengths do not close on {total}")
        if [id(span) for span, _, _, _ in items] == [id(span) for span, _ in want]:
            skipped += 1
            continue
        offsets = {}
        cursor = 0
        for _, span, _, length in items:
            offsets[id(span)] = cursor
            cursor += length
        if cursor != total:
            raise SystemExit(f"{path}: append layout {cursor} != plan {total}")
        temporary = path.with_name(path.name + ".planorder")
        with path.open("rb") as src, temporary.open("wb") as dst:
            for span, length in want:
                src.seek(offsets[id(span)])
                remaining = length
                while remaining > 0:
                    chunk = src.read(min(remaining, CHUNK))
                    if not chunk:
                        raise SystemExit(f"{path}: short read during reorder")
                    dst.write(chunk)
                    remaining -= len(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        if temporary.stat().st_size != total:
            temporary.unlink()
            raise SystemExit(f"{path}: reorder wrote {temporary.stat().st_size}")
        os.replace(temporary, path)
        fixed += 1
    return fixed, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(m.ARMS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--stage-dir", required=True)
    args = parser.parse_args()
    source = m.SafetensorsSource(Path(args.checkpoint))
    m.check_source(args.arm, source)
    records = m.build_plan(args.arm, source.config, args.tp, args.rank)
    layout = append_layout(source, records)
    fixed, skipped = reorder(Path(args.stage_dir), layout, source)
    print(f"reorder: {fixed} planes rewritten into plan order, "
          f"{skipped} already plan-ordered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
