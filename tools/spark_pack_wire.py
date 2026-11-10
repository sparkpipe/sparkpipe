from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

from spark_pack_common import PackFailure, align_up

STAGEPACK_HEADER = struct.Struct("<26I2Q")
STAGEPACK_ENTRY = struct.Struct("<6I4Q")
PAYLOAD_ALIGNMENT = 256

CODEC_HEADER = struct.Struct("<20I2Q65s32s32s32s7x")
CODEC_HEADER_BYTES = CODEC_HEADER.size
CODEC_ENTRY_BYTES = 64
CODEC_HEADER_FIELDS = struct.Struct("<20I2Q")
CODEC_ENTRY = struct.Struct("<8I4Q")
MODEL_REVISION_BYTES = 65
DIGEST_BYTES = 32

EXPERT_MANIFEST_MAGIC = 0x58504557
EXPERT_MANIFEST_VERSION = 2
EXPERT_RANGE_BYTES_MAX = 64 * 1024 * 1024
EXPERT_MANIFEST_HEADER = struct.Struct("<IIII")
EXPERT_RECORD = struct.Struct("<IIIIQQ")

MASK64 = (1 << 64) - 1
CK128_C1 = 0x87c37b91114253d5
CK128_C2 = 0x4cf5ad432745937f


class Entry:
    def __init__(self, kind: int, layer: int, payload_type: int, weight_codec: int,
                 scale_encoding: int, group_count: int, rows: int, columns: int):
        self.kind = kind
        self.layer = layer
        self.payload_type = payload_type
        self.weight_codec = weight_codec
        self.scale_encoding = scale_encoding
        self.group_count = group_count
        self.rows = rows
        self.columns = columns
        self.payload_offset = 0
        self.payload_bytes = 0
        self.scale_offset = 0
        self.scale_bytes = 0


def serialize_entry(entry: Entry) -> bytes:
    return CODEC_ENTRY.pack(
        entry.kind, entry.layer, entry.payload_type, entry.weight_codec,
        entry.scale_encoding, entry.group_count, entry.rows, entry.columns,
        entry.payload_offset, entry.payload_bytes, entry.scale_offset, entry.scale_bytes)


def codec_header(fields: Sequence[int], directory_offset: int, file_bytes: int, revision: bytes,
                 digests: Sequence[bytes] = (), header_bytes: int = CODEC_HEADER_BYTES) -> bytes:
    if len(fields) != 20:
        raise PackFailure(f"header field count {len(fields)}, expected 20")
    if len(revision) > MODEL_REVISION_BYTES:
        raise PackFailure("revision does not fit the header field")
    slots = list(digests) + [bytes(DIGEST_BYTES)] * (3 - len(digests))
    if len(slots) != 3 or any(len(slot) != DIGEST_BYTES for slot in slots):
        raise PackFailure("the header carries three 32-byte digests")
    header = (CODEC_HEADER_FIELDS.pack(*fields, directory_offset, file_bytes)
              + revision.ljust(MODEL_REVISION_BYTES, b"\0") + b"".join(slots))
    if len(header) > header_bytes:
        raise PackFailure(f"header assembled {len(header)} bytes, expected {header_bytes}")
    return header.ljust(header_bytes, b"\0")


def truncated_revision(revision: str) -> bytes:
    return revision.encode()[:MODEL_REVISION_BYTES - 1]


def region_offsets(sizes: Iterable[Tuple[int, int]], cursor: int, *, alignment: int = PAYLOAD_ALIGNMENT,
                   align_payloads: bool = True, align_scales: bool = True,
                   scales_last: bool = False) -> Tuple[List[Tuple[int, int]], int]:
    sizes = list(sizes)
    offsets = [[0, 0] for _ in sizes]
    for index, (payload_bytes, scale_bytes) in enumerate(sizes):
        offsets[index][0] = align_up(cursor, alignment) if align_payloads else cursor
        cursor = offsets[index][0] + payload_bytes
        if scale_bytes and not scales_last:
            offsets[index][1] = align_up(cursor, alignment) if align_scales else cursor
            cursor = offsets[index][1] + scale_bytes
    if scales_last:
        for index, (_, scale_bytes) in enumerate(sizes):
            if scale_bytes:
                offsets[index][1] = align_up(cursor, alignment) if align_scales else cursor
                cursor = offsets[index][1] + scale_bytes
    return [tuple(pair) for pair in offsets], cursor


def place_entries(entries: Sequence[Entry], cursor: int, **policy) -> int:
    offsets, end = region_offsets(((entry.payload_bytes, entry.scale_bytes) for entry in entries), cursor, **policy)
    for entry, (payload_offset, scale_offset) in zip(entries, offsets):
        entry.payload_offset = payload_offset
        entry.scale_offset = scale_offset
    return end


def codec_directory_offset(header_bytes: int = CODEC_HEADER_BYTES) -> int:
    return align_up(header_bytes, PAYLOAD_ALIGNMENT)


def write_prefix(out, header: bytes, directory_offset: int, entries: Iterable[Entry]) -> None:
    out.write(header)
    out.seek(directory_offset)
    for entry in entries:
        out.write(serialize_entry(entry))


class SlabDigestSink:
    def __init__(self, slab_bytes: int):
        self.slab_bytes = slab_bytes
        self.buffer = b""
        self.digests: List[bytes] = []

    def feed(self, chunk: bytes) -> None:
        while chunk:
            take = min(self.slab_bytes - len(self.buffer), len(chunk))
            self.buffer += chunk[:take]
            chunk = chunk[take:]
            if len(self.buffer) == self.slab_bytes:
                self.digests.append(ck128(self.buffer))
                self.buffer = b""


def emit_region(out, offset: int, expected: int, chunks: Iterator[bytes],
                sink: Optional[SlabDigestSink] = None) -> None:
    out.seek(offset)
    written = 0
    for chunk in chunks:
        if len(chunk) > expected - written:
            raise PackFailure(f"region at {offset}: producer exceeds {expected} bytes")
        out.write(chunk)
        if sink is not None:
            sink.feed(chunk)
        written += len(chunk)
    if written != expected:
        raise PackFailure(f"region at {offset}: producer wrote {written}, expected {expected}")


def emit_item(out, item, payload_sink: Optional[SlabDigestSink] = None,
              scale_sink: Optional[SlabDigestSink] = None) -> None:
    entry = item.entry
    emit_region(out, entry.payload_offset, entry.payload_bytes, item.produce_payload(), payload_sink)
    if entry.scale_bytes and item.produce_scale is None:
        raise PackFailure(f"{entry.kind}: scale segment has no producer")
    emit_region(out, entry.scale_offset, entry.scale_bytes,
                item.produce_scale() if item.produce_scale else iter(()), scale_sink)


def item_region(item) -> Tuple[int, int]:
    entry = item.entry
    if entry.scale_bytes:
        return entry.payload_offset, entry.scale_offset + entry.scale_bytes
    return entry.payload_offset, entry.payload_offset + entry.payload_bytes


def _rotl64(value: int, bits: int) -> int:
    return ((value << bits) | (value >> (64 - bits))) & MASK64


def _fmix64(value: int) -> int:
    value ^= value >> 33
    value = (value * 0xff51afd7ed558ccd) & MASK64
    value ^= value >> 33
    value = (value * 0xc4ceb9fe1a85ec53) & MASK64
    value ^= value >> 33
    return value


def ck128(data: bytes) -> bytes:
    h1 = h2 = 0
    whole = len(data) // 16
    for k1, k2 in struct.iter_unpack("<QQ", memoryview(data)[:whole * 16]):
        k1 = _rotl64((k1 * CK128_C1) & MASK64, 31)
        h1 ^= (k1 * CK128_C2) & MASK64
        h1 = (_rotl64(h1, 27) + h2) & MASK64
        h1 = (h1 * 5 + 0x52dce729) & MASK64
        k2 = _rotl64((k2 * CK128_C2) & MASK64, 33)
        h2 ^= (k2 * CK128_C1) & MASK64
        h2 = (_rotl64(h2, 31) + h1) & MASK64
        h2 = (h2 * 5 + 0x38495ab5) & MASK64
    tail = bytes(data[whole * 16:])
    if len(tail) > 8:
        k2 = int.from_bytes(tail[8:16].ljust(8, b"\0"), "little")
        k2 = _rotl64((k2 * CK128_C2) & MASK64, 33)
        h2 ^= (k2 * CK128_C1) & MASK64
    k1 = int.from_bytes(tail[:8].ljust(8, b"\0"), "little")
    k1 = _rotl64((k1 * CK128_C1) & MASK64, 31)
    h1 ^= (k1 * CK128_C2) & MASK64
    h1 ^= len(data) & MASK64
    h2 ^= len(data) & MASK64
    h1 = (h1 + h2) & MASK64
    h2 = (h2 + h1) & MASK64
    h1 = _fmix64(h1)
    h2 = _fmix64(h2)
    h1 = (h1 + h2) & MASK64
    h2 = (h2 + h1) & MASK64
    return struct.pack("<QQ", h1, h2)


def expert_manifest_bytes(records: Sequence[Tuple[int, int, int, int, int, bytes]], count: int) -> bytes:
    body = b"".join(EXPERT_RECORD.pack(layer, expert, kind, 0, offset, length) + digest
                    for layer, expert, kind, offset, length, digest in records)
    return EXPERT_MANIFEST_HEADER.pack(EXPERT_MANIFEST_MAGIC, EXPERT_MANIFEST_VERSION, count, 0) + body


def journal_record(index: int, item, digest: bytes) -> str:
    entry = item.entry
    begin, end = item_region(item)
    return json.dumps({"index": index, "kind": entry.kind, "begin": begin, "end": end, "ck128": digest.hex(),
                       "payload_bytes": entry.payload_bytes, "scale_bytes": entry.scale_bytes}) + "\n"


def append_journal(journal, out, index: int, item) -> None:
    out.flush()
    begin, end = item_region(item)
    out.seek(begin)
    journal.write(journal_record(index, item, ck128(out.read(end - begin))))
    journal.flush()
    os.fsync(journal.fileno())


def trusted_journal_prefix(journal_file: Path, partial: Path, plan: Sequence) -> int:
    records = []
    for line in journal_file.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:
            break
    trusted = 0
    for index, record in enumerate(records):
        if record.get("index") != index:
            break
        item = plan[index]
        begin, end = item_region(item)
        if (record.get("kind") != item.entry.kind or record.get("begin") != begin
                or record.get("end") != end
                or record.get("payload_bytes") != item.entry.payload_bytes
                or record.get("scale_bytes") != item.entry.scale_bytes):
            raise PackFailure("resume journal disagrees with the plan; remove "
                              f"the stale {journal_file.name} and partial")
        if not partial.exists() or partial.stat().st_size < end:
            break
        with partial.open("rb") as handle:
            handle.seek(begin)
            if ck128(handle.read(end - begin)).hex() != record.get("ck128"):
                break
        trusted += 1
    if trusted != len(records):
        journal_file.write_text("".join(json.dumps(record) + "\n" for record in records[:trusted]))
    return trusted
