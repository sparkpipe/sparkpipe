#!/usr/bin/env python3
"""Strip the MTP tail from placed stagepacks, in place.

Operator directive 2026-09-04: speculation is consolidated on the
speculator system, so serving stagepacks carry no MTP data. The MTP
entries ride at the tail of the file, so the strip is:

  1. parse the header + directory (family header layout),
  2. classify MTP entries (sentinel layer / family MTP kinds),
  3. FAIL CLOSED unless the dropped region is exactly the file tail
     (max kept end <= min dropped start),
  4. chattr -i, truncate, patch tensor_count / mtp_layer_count /
     file_bytes, chattr +i,
  5. re-sha in small chunks (fadvise DONTNEED; the memory law) and
     rewrite the receipt recording the strip.

Never strips a pack whose module still expects MTP: the no-MTP module
variant must be built+published BEFORE any restart loads a stripped
pack. Run per file; the caller owns the fan-out width.

  python3 tools/stagepack_mtp_strip.py --pack <file> --family qwen4_flash [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

GLOBAL_LAYER = 0xFFFFFFFF
MTP_LAYER = 0xFFFFFFFE

# family -> header/directory layout facts. All known families share the
# 26-u32 + 2-u64 120-byte header and the 56-byte directory entry; the
# family entry records where mtp_layer_count lives and which GLOBAL-
# layer tensor kinds are MTP-scoped (plus whether the family tags MTP
# entries with the sentinel layer).
FAMILIES = {
    "qwen4_flash": {
        "u32_count": 26, "count_index": 4, "mtp_index": 25,
        "u64_count_index": 0,   # u64[0] = directory_offset
        "entry_bytes": 56,      # <6I4Q: kind,layer,fmt,group + off/bytes x2
        "payload_off_at": 24, "scale_off_at": 40,
        "mtp_global_kinds": {28, 29, 30, 31, 43, 44},
        "mtp_layer_range": None, "sentinel_mtp": True,   # layer == 0xFFFFFFFE
    },
    # dsv4 (DSpark draft blocks): 80-byte header (<16I2Q), 40-byte
    # entries (<6I2Q) WITHOUT size fields - offsets are monotonic, so
    # the tail check runs on offsets alone. MTP = the three draft
    # layers at 0xFFFFFFFB.. and the mtp.* globals (kinds 41..49).
    "dsv4": {
        "u32_count": 16, "count_index": 8, "mtp_index": 15,
        "u64_count_index": 0,   # u64[0] = directory offset (header size)
        "entry_bytes": 40,
        "payload_off_at": 24, "scale_off_at": 32,
        "mtp_global_kinds": set(range(41, 50)),
        "mtp_layer_range": (0xFFFFFFFB, 3), "sentinel_mtp": False,
    },
}


def is_mtp(entry: dict, family: dict) -> bool:
    if entry["layer"] == GLOBAL_LAYER and entry["kind"] in family["mtp_global_kinds"]:
        return True
    if family["sentinel_mtp"] and entry["layer"] == MTP_LAYER:
        return True
    rng = family["mtp_layer_range"]
    if rng is not None and rng[0] <= entry["layer"] < rng[0] + rng[1]:
        return True
    return False


def sha256_chunked(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        fd = file.fileno()
        while True:
            chunk = file.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except (AttributeError, OSError):
                pass
    return digest.hexdigest()


def chattr(path: Path, flag: str) -> bool:
    """Set the immutable flag; plain first, sudo as the fallback. The
    ring needs CAP_LINUX_IMMUTABLE, so unprivileged runs no-op on -i of
    an unprotected file but fail +i — callers must treat False as "the
    relock did not happen"."""
    # Argument-list exec (shell=False, the default): nothing is parsed
    # by a shell, and "--" ends option parsing so a --pack path that
    # starts with "-" can never be read as a chattr flag.
    for prefix in ([], ["sudo"]):
        result = subprocess.run(prefix + ["chattr", flag, "--", str(path)],
                                shell=False,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            return True
    return False



def read_receipt(receipt: Path) -> dict:
    if receipt.exists():
        try:
            return json.loads(receipt.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def old_marker_not_in_receipt(receipt: Path) -> bool:
    return read_receipt(receipt).get("mtp") != "stripped"


def update_receipt(receipt: Path, digest: str, kept_end, dropped, reclaim, prior, locked) -> None:
    receipt_data = read_receipt(receipt)
    receipt_data.update({
        "output_sha256": digest,
        "mtp": "stripped",
        "mtp_entries_dropped": dropped,
        "locked": locked,
    })
    if kept_end is not None:
        receipt_data["file_bytes"] = kept_end
    if reclaim is not None:
        receipt_data["bytes_reclaimed"] = reclaim
    if prior is not None:
        receipt_data["prior_sha256"] = prior
    receipt.write_text(json.dumps(receipt_data, indent=2, sort_keys=True) + "\n")


def is_immutable(pack: Path) -> bool:
    result = subprocess.run(["lsattr", str(pack)], stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    return "i" in (result.stdout.split()[0] if result.stdout else "")



def compact_pack(pack: Path, family: dict, entries, dropped, kept, kept_end,
                 u32s, u64s, receipt: Path, args) -> int:
    """Rewrite the file keeping only non-MTP byte regions. Regions tile
    the payload area by sorted start offsets (offset-only layouts carry
    no sizes; the next start bounds each region). Kept regions are
    re-emitted 256-aligned with the directory rewritten to match."""
    import os as _os
    entry_bytes = family["entry_bytes"]
    original_size = pack.stat().st_size

    def start_of(e):
        present = [c for c in (e["payload_off"], e["scale_off"]) if c]
        return min(present) if present else 0

    # Tile the payload area at EVERY entry boundary (payload start,
    # scale start) so interleaved layouts split cleanly; each region is
    # owned by the entry that references its start offset.
    by_offset = {}
    for e in entries:
        if e["payload_off"]:
            by_offset[e["payload_off"]] = (e, "payload_off")
        if e["scale_off"]:
            by_offset[e["scale_off"]] = (e, "scale_off")
    edges = sorted(by_offset) + [original_size]
    kept_regions = []
    for i in range(len(edges) - 1):
        start, end = edges[i], edges[i + 1]
        owner = by_offset.get(start)
        if owner is None:
            continue
        e, key = owner
        if is_mtp(e, family):
            continue
        kept_regions.append((start, end, e, key))
    keep_bytes = sum(t - s for (s, t, _, _) in kept_regions if t > s)
    if args.dry_run:
        print(f"COMPACT {pack}: {len(kept_regions)} kept regions, {keep_bytes} bytes ({keep_bytes / 2**30:.2f} GiB) [dry]")
        return 0
    tmp = Path(str(pack) + ".compact.tmp")
    if not chattr(pack, "-i") and is_immutable(pack):
        print(f"FAIL {pack}: immutable and cannot clear the flag (need root)", file=sys.stderr)
        return 2
    header_len = family["u32_count"] * 4 + 16
    dir_len = len(kept) * entry_bytes
    payload_base = ((header_len + dir_len + 255) // 256) * 256
    cursor = payload_base
    new_records = []
    with pack.open("rb") as src, tmp.open("wb") as dst:
        dst.truncate(payload_base + keep_bytes)
        records_by_entry = {}
        for (start, end, e, key) in kept_regions:
            if end <= start:
                continue
            length = end - start
            rec = records_by_entry.setdefault(id(e), dict(e))
            rec[key] = cursor + (rec[key] - start)
            # sendfile may transfer fewer bytes than asked (the 2GB
            # short-send class); drive it to completion.
            sent_total = 0
            pos = start
            while sent_total < length:
                sent = _os.sendfile(dst.fileno(), src.fileno(), pos,
                                    length - sent_total)
                if sent == 0:
                    print(f"FAIL {pack}: sendfile stalled at {pos} (+{sent_total} of {length})", file=sys.stderr)
                    return 1
                pos += sent
                sent_total += sent
            cursor += length
        new_records = list(records_by_entry.values())
        u32s = list(u32s)
        u32s[family["count_index"]] = len(new_records)
        u32s[family["mtp_index"]] = 0
        dst.seek(0)
        dst.write(struct.pack(f"<{family['u32_count']}I", *u32s))
        u64s = list(u64s)
        u64s[0] = header_len
        u64s[1] = cursor
        dst.seek(family["u32_count"] * 4)
        dst.write(struct.pack("<2Q", *u64s))
        dst.seek(header_len)
        for rec in new_records:
            if entry_bytes == 56:
                dst.write(struct.pack("<6I", rec["kind"], rec["layer"],
                                      rec["format"], rec["rows"],
                                      rec["columns"], 0))
                dst.write(struct.pack("<4Q", rec["payload_off"],
                                      rec["payload_bytes"],
                                      rec["scale_off"], rec["scale_bytes"]))
            else:
                dst.write(struct.pack("<4I", rec["kind"], rec["layer"],
                                      rec["format"], rec["rows"]))
                dst.write(struct.pack("<2I", rec["columns"], 0))
                dst.write(struct.pack("<2Q", rec["payload_off"],
                                      rec["scale_off"]))
        dst.flush()
        _os.fsync(dst.fileno())
    digest = sha256_chunked(tmp)
    prior = read_receipt(receipt).get("output_sha256")
    _os.replace(tmp, pack)
    reclaim = original_size - cursor
    update_receipt(receipt, digest, kept_end=cursor, dropped=len(dropped),
                   reclaim=reclaim, prior=prior, locked=None)
    locked = chattr(pack, "+i")
    update_receipt(receipt, digest, kept_end=cursor, dropped=len(dropped),
                   reclaim=reclaim, prior=prior, locked=locked)
    print(f"DONE {pack}: compacted {original_size} -> {cursor} bytes "
          f"({reclaim / 2**30:.2f} GiB reclaimed), sha {digest[:16]}... locked={locked}")
    return 0 if locked else 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--family", choices=sorted(FAMILIES), default="qwen4_flash")
    parser.add_argument("--compact", action="store_true",
                        help="rewrite the pack keeping only non-MTP regions (for layouts where MTP is not the file tail)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    family = FAMILIES[args.family]
    pack: Path = args.pack
    receipt: Path = Path(str(pack) + ".receipt.json")

    actual_bytes = pack.stat().st_size
    with pack.open("rb") as file:
        raw = file.read(120)
    if len(raw) != 120:
        print(f"FAIL {pack}: short header", file=sys.stderr)
        return 1
    u32s = list(struct.unpack_from(f"<{family['u32_count']}I", raw, 0))
    tensor_count = u32s[family["count_index"]]
    u64s = list(struct.unpack_from("<2Q", raw, family["u32_count"] * 4))
    directory_offset = u64s[family["u64_count_index"]]
    file_bytes = actual_bytes
    entry_bytes = family["entry_bytes"]
    with pack.open("rb") as file:
        file.seek(directory_offset)
        raw_dir = file.read(tensor_count * entry_bytes)
    if len(raw_dir) != tensor_count * entry_bytes:
        print(f"FAIL {pack}: short directory", file=sys.stderr)
        return 1
    entries = []
    for i in range(tensor_count):
        block = raw_dir[i * entry_bytes:(i + 1) * entry_bytes]
        kind, layer, fmt, rows, columns = struct.unpack_from("<5I", block, 0)
        payload_off, scale_off = struct.unpack_from("<2Q", block, family["payload_off_at"])
        entries.append({"kind": kind, "layer": layer, "format": fmt,
                        "rows": rows, "columns": columns,
                        "payload_off": payload_off, "payload_bytes": 0,
                        "scale_off": scale_off, "scale_bytes": 0})
    if family["entry_bytes"] == 56:
        # the flash layout carries explicit sizes; use them for the
        # tail check (the dsv4 layout relies on monotonic offsets).
        # <6I4Q: payload_off@24 payload_bytes@32 scale_off@40
        # scale_bytes@48.
        for i, e in enumerate(entries):
            block = raw_dir[i * entry_bytes:(i + 1) * entry_bytes]
            e["payload_bytes"] = struct.unpack_from("<Q", block, 32)[0]
            e["scale_bytes"] = struct.unpack_from("<Q", block, 48)[0]
    dropped = [e for e in entries if is_mtp(e, family)]
    kept = [e for e in entries if not is_mtp(e, family)]
    if not dropped:
        needs_field_zero = u32s[family["mtp_index"]] != 0
        if needs_field_zero or (u32s[family["mtp_index"]] == 0 and old_marker_not_in_receipt(receipt)):
            # Two no-truncate cases: a vintage pack whose header says
            # mtp=1 but which carries no MTP entries (zero the field),
            # or a pack a prior run mutated but whose receipt/relock
            # never completed. Both re-sha (the header bytes change)
            # and relock.
            print(f"NORMALIZE {pack}: no MTP entries ({tensor_count} tensors); "
                  + ("zeroing the mtp_layer_count header field" if needs_field_zero else "completing receipt + relock"))
            if not chattr(pack, "-i") and is_immutable(pack):
                print(f"FAIL {pack}: immutable and cannot clear the flag (need root)", file=sys.stderr)
                return 2
            prior = read_receipt(receipt).get("output_sha256")
            if needs_field_zero:
                with pack.open("r+b") as file:
                    u32s[family["mtp_index"]] = 0
                    file.seek(0)
                    file.write(struct.pack(f"<{family['u32_count']}I", *u32s))
                    file.flush()
                    os.fsync(file.fileno())
            digest = sha256_chunked(pack)
            update_receipt(receipt, digest, kept_end=file_bytes,
                           dropped=0, reclaim=0, prior=prior, locked=None)
            locked = chattr(pack, "+i")
            update_receipt(receipt, digest, kept_end=file_bytes,
                           dropped=0, reclaim=0, prior=prior, locked=locked)
            print(f"DONE {pack}: sha {digest[:16]}... locked={locked}")
            return 0 if locked else 3
        print(f"PASS {pack}: no MTP entries ({tensor_count} tensors) - nothing to strip")
        return 0
    # Fail-closed tail check: the dropped region must BE the tail.
    kept_end = 0
    for e in kept:
        kept_end = max(kept_end, e["payload_off"] + e["payload_bytes"],
                       e["scale_off"] + e["scale_bytes"])
    dropped_starts = []
    for e in dropped:
        if family["entry_bytes"] == 56:
            # size-carrying: an offset only counts if bytes follow it
            if e["payload_bytes"]:
                dropped_starts.append(e["payload_off"])
            if e["scale_bytes"]:
                dropped_starts.append(e["scale_off"])
        else:
            # offset-only layout: any non-zero offset is a real region
            if e["payload_off"]:
                dropped_starts.append(e["payload_off"])
            if e["scale_off"]:
                dropped_starts.append(e["scale_off"])
    dropped_start = min(dropped_starts)
    if kept_end > dropped_start:
        if not args.compact:
            print(f"FAIL {pack}: MTP region is not the tail (kept end {kept_end} > dropped start {dropped_start}); use --compact to rewrite kept regions, or rebuild MTP-less", file=sys.stderr)
            return 1
        return compact_pack(pack, family, entries, dropped, kept, kept_end,
                            u32s, u64s, receipt, args)
    new_count = tensor_count - len(dropped)
    # Cut at the first dropped byte: on size-carrying layouts that only
    # leaves the tail's alignment pad behind; on offset-only layouts
    # (dsv4) it is the only safe cut, since kept ends are proven
    # <= dropped_start by monotonicity.
    cut_point = dropped_start
    reclaim = file_bytes - cut_point
    print(f"STRIP {pack}: {tensor_count} -> {new_count} tensors, {reclaim} bytes ({reclaim / 2**30:.2f} GiB)")
    if args.dry_run:
        return 0

    if not chattr(pack, "-i") and is_immutable(pack):
        print(f"FAIL {pack}: immutable and cannot clear the flag (need root)", file=sys.stderr)
        return 2
    with pack.open("r+b") as file:
        file.truncate(cut_point)
        u32s[family["count_index"]] = new_count
        u32s[family["mtp_index"]] = 0
        file.seek(0)
        file.write(struct.pack(f"<{family['u32_count']}I", *u32s))
        u64s[1] = cut_point  # u64[1] is file_bytes in both layouts
        file.seek(family["u32_count"] * 4)
        file.write(struct.pack("<2Q", *u64s))
        file.flush()
        os.fsync(file.fileno())

    digest = sha256_chunked(pack)
    prior = read_receipt(receipt).get("output_sha256")
    update_receipt(receipt, digest, kept_end=cut_point, dropped=len(dropped),
                   reclaim=reclaim, prior=prior, locked=None)
    locked = chattr(pack, "+i")
    update_receipt(receipt, digest, kept_end=cut_point, dropped=len(dropped),
                   reclaim=reclaim, prior=prior, locked=locked)
    print(f"DONE {pack}: sha {digest[:16]}... locked={locked}")
    return 0 if locked else 3


if __name__ == "__main__":
    sys.exit(main())
