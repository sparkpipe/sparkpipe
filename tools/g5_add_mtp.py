#!/usr/bin/env python3
"""g5_add_mtp.py — append the MTP block to an existing .g5nsp pack.

Operator ruling 2026-08-31: MTP weights must be IN the packs (speculation
needs the data). The MTP entries sit at the END of the packer's plan, so
the upgrade is a pure append: old bytes verbatim, MTP payloads appended,
the full (n+24)-entry directory re-emitted at a new trailing offset, and
header fields patched (entry_count, flags, directory_offset, file_bytes).
The verifier (--mtp --expected-bytes) validates the result end-to-end
against the source checkpoint.
"""
import argparse
import mmap
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm5_next_resident_stagepack as P  # noqa: E402

HEADER_BYTES = P.HEADER_BYTES
ENTRY_BYTES = P.ENTRY_BYTES
ALIGNMENT = P.ALIGNMENT
MTP_LAYER = P.MTP_LAYER
ENTRY_FMT = "<IIIIIIIIQQQQ"  # serialize_entry's format: 8U32 + 4U64 = 64B


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", required=True, help="existing (no-MTP) pack")
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True, help="upgraded output path")
    ap.add_argument("--tp-rank", type=int, required=True)
    ap.add_argument("--tp-degree", type=int, default=16)
    args = ap.parse_args()

    src_path = Path(args.pack)
    old_size = src_path.stat().st_size
    f = open(src_path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    fields = struct.unpack_from("<20I", mm, 0)
    magic, ver, flags, entry_count = fields[0], fields[1], fields[5], fields[6]
    dir_off, fb = struct.unpack_from("<QQ", mm, 80)
    assert magic == P.MAGIC and ver == 1, "not a v1 g5nsp pack"
    assert flags == 0, f"pack already carries flags={flags}"
    assert fb == old_size, f"header file_bytes {fb} != {old_size}"

    source = P.SourceReader(Path(args.source))
    packer = P.Packer(source, args.tp_degree, args.tp_rank, 0, fields[10],
                      True, True, True)
    packer.build()
    mtp_items = [it for it in packer.plan if it.entry.layer == MTP_LAYER]
    non_mtp = [it for it in packer.plan if it.entry.layer != MTP_LAYER]
    assert len(non_mtp) == entry_count, \
        f"plan mismatch: {len(non_mtp)} vs pack {entry_count}"

    # old directory, parsed; must byte-match the no-MTP plan (drift = abort)
    old_raw = []
    for i in range(entry_count):
        raw = struct.unpack_from(ENTRY_FMT, mm, dir_off + i * ENTRY_BYTES)
        e = non_mtp[i].entry
        want = (e.kind, e.layer, e.payload_type, e.weight_codec,
                e.scale_encoding, e.group_count, e.rows, e.columns)
        assert raw[:8] == want, f"entry {i} drift: {raw[:8]} != {want}"
        assert raw[9] == e.payload_bytes and raw[11] == e.scale_bytes, \
            f"entry {i} byte drift kind={e.kind}"
        old_raw.append(raw)

    # assign offsets for the appended MTP payloads
    cursor = (old_size + ALIGNMENT - 1) & ~(ALIGNMENT - 1)
    new_entries = []
    for it in mtp_items:
        e = it.entry
        e.payload_offset = (cursor + ALIGNMENT - 1) & ~(ALIGNMENT - 1)
        cursor = e.payload_offset + e.payload_bytes
        if e.scale_bytes:
            e.scale_offset = (cursor + ALIGNMENT - 1) & ~(ALIGNMENT - 1)
            cursor = e.scale_offset + e.scale_bytes
        new_entries.append((e, it))

    new_dir_off = (cursor + ALIGNMENT - 1) & ~(ALIGNMENT - 1)
    new_count = entry_count + len(mtp_items)
    new_file_bytes = new_dir_off + new_count * ENTRY_BYTES

    with open(args.output, "wb") as out:
        out.write(mm[:old_size])          # old header+directory+payloads
        for e, it in new_entries:         # appended MTP payloads
            out.seek(e.payload_offset)
            for chunk in it.produce_payload():
                out.write(chunk)
            if e.scale_bytes and it.produce_scale:
                out.seek(e.scale_offset)
                for chunk in it.produce_scale():
                    out.write(chunk)
        out.seek(new_dir_off)             # superseding full directory,
        oi = 0                            # emitted in full mtp-plan order
        for it in packer.plan:
            if it.entry.layer == MTP_LAYER:
                out.write(P.serialize_entry(it.entry))
            else:
                out.write(struct.pack(ENTRY_FMT, *old_raw[oi]))
                oi += 1
        assert oi == entry_count, f"old entries left: {entry_count - oi}"
        header = bytearray(mm[:HEADER_BYTES])
        struct.pack_into("<I", header, 20, 1)             # flags: MTP
        struct.pack_into("<I", header, 24, new_count)     # entry_count
        struct.pack_into("<QQ", header, 80, new_dir_off, new_file_bytes)
        out.seek(0)
        out.write(header)
    print(f"{Path(args.output).name}: {old_size} -> {new_file_bytes} bytes, "
          f"{new_count} tensors (+{len(mtp_items)} MTP), flags=1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
