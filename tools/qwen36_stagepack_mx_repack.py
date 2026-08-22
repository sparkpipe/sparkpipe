#!/usr/bin/env python3
"""Repack a qwen36 fp8 stage pack from FP8_E4M3_F32B128 (per-128x128-tile
FP32 scales) to FP8_E4M3_E8M0B128 (per-row e8m0 scales, one per 128-K group)
- the scale layout the SM121 native block-scaled fp8 MMA path reads
  (SparkLmSm121ScaleB: neuron * (K/128) + k/128).

The payload stays E4M3; each tile's FP32 scale is rounded UP to a power of
two (ratio in (0.5, 1], payloads shrink, no overflow) and every payload
element is re-rounded to nearest E4M3. Usage-approved quantization trade:
small per-element loss, tensor-core GEMMs in return.

Usage: qwen36_stagepack_mx_repack.py <in.qwen36sp> <out.qwen36sp>
"""
import struct
import sys

import numpy as np

MAGIC = 0x50533651
HEADER_BYTES = 120
ENTRY_BYTES = 56
FMT_F32B128 = 5
FMT_E8M0B128 = 6
CHUNK_ROWS = 2048  # multiple of the 128-row tile span

_E4M3_LUT = np.zeros(256, dtype=np.float32)
for _b in range(256):
    _s = -1.0 if _b & 0x80 else 1.0
    _e = (_b >> 3) & 0xF
    _m = _b & 7
    if _e == 0:
        _v = _m / 8.0
    elif _e == 15 and _m == 7:
        _v = float("nan")
    else:
        _v = (1.0 + _m / 8.0) * (2.0 ** (_e - 7))
    _E4M3_LUT[_b] = _s * _v
_FINITE_MAGS = np.array(sorted({abs(v) for v in _E4M3_LUT if np.isfinite(v)}), dtype=np.float32)
_MAG_TO_BYTE = {}
for _b in range(256):
    _v = _E4M3_LUT[_b]
    if np.isfinite(_v):
        _MAG_TO_BYTE.setdefault(abs(_v), _b & 0x7F)
_MAG_BYTES = np.array([_MAG_TO_BYTE[m] for m in _FINITE_MAGS], dtype=np.uint8)


def e4m3_encode_mag(mag):
    idx = np.clip(np.searchsorted(_FINITE_MAGS, mag), 1, len(_FINITE_MAGS))
    lo = _FINITE_MAGS[idx - 1]
    hi = _FINITE_MAGS[idx]
    pick_hi = (mag - lo) > (hi - mag)
    enc = np.where(pick_hi, hi, lo)
    enc = np.minimum(enc, 448.0)
    bidx = np.clip(np.searchsorted(_FINITE_MAGS, enc), 0, len(_MAG_BYTES) - 1)
    return _MAG_BYTES[bidx]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    raw = open(src, "rb").read()
    magic, version, hbytes, ebytes, count = struct.unpack_from("<5I", raw, 0)
    assert magic == MAGIC, f"bad magic {magic:#x}"
    assert hbytes == HEADER_BYTES and ebytes == ENTRY_BYTES
    dir_off = struct.unpack_from("<Q", raw, 104)[0]  # after 26 uint32 header fields
    entries = []
    for i in range(count):
        off = dir_off + i * ENTRY_BYTES
        kind, layer, fmt, rows, cols, group = struct.unpack_from("<6I", raw, off)
        poff, pbytes, soff, sbytes = struct.unpack_from("<4Q", raw, off + 24)
        entries.append(dict(kind=kind, layer=layer, fmt=fmt, rows=rows, cols=cols,
                            group=group, poff=poff, pbytes=pbytes, soff=soff, sbytes=sbytes))

    stats = []
    with open(dst, "wb") as f:
        f.write(raw[:dir_off])
        f.write(b"\x00" * (count * ENTRY_BYTES))  # directory placeholder
        cursor = dir_off + count * ENTRY_BYTES
        for e in entries:
            pad = (-cursor) % 256
            f.write(b"\x00" * pad)
            cursor += pad
            poff_new, pbytes_new = cursor, e["pbytes"]
            # Only 128x128-divisible matrices convert: the native MMA needs
            # output%128 and input%128; small projections (GDN beta/decay,
            # 48-row outputs) stay F32B128 on the scalar path.
            # NOTE: the original version read the STALE parsing-loop rows/cols
            # here - the gate silently failed for every entry and the tool
            # degenerated to a byte-copy passthrough (the vacuous 0.0 error).
            convertible = (e["fmt"] == FMT_F32B128 and
                e["rows"] % 128 == 0 and e["cols"] % 128 == 0)
            if not convertible:
                blob = raw[e["poff"] : e["poff"] + e["pbytes"]]
                scale = raw[e["soff"] : e["soff"] + e["sbytes"]] if e["sbytes"] else b""
                f.write(blob)
                f.write(scale)
                cursor += len(blob) + len(scale)
                soff_new, sbytes_new = poff_new + len(blob), len(scale)
                e["new_fmt"], e["new_group"] = e["fmt"], e["group"]
                stats.append((e["kind"], e["layer"], e["rows"], e["cols"], 0.0, 0.0))
            else:
                rows, cols = e["rows"], e["cols"]
                tile = np.frombuffer(raw, dtype=np.float32, count=e["sbytes"] // 4,
                                     offset=e["soff"]).reshape(rows // 128, cols // 128)
                new_scale = np.zeros((rows, cols // 128), dtype=np.uint8)
                max_rel, sum_rel, n_rel = 0.0, 0.0, 0
                for r0 in range(0, rows, CHUNK_ROWS):
                    r1 = min(r0 + CHUNK_ROWS, rows)
                    block = np.frombuffer(raw, dtype=np.uint8, count=(r1 - r0) * cols,
                                          offset=e["poff"] + r0 * cols).reshape(r1 - r0, cols)
                    trow0, trow1 = r0 // 128, (r1 - 1) // 128 + 1
                    # per-row, per-128-K-group tile scales: rows expanded from
                    # the tile grid; the column axis stays per-group [rb, groups]
                    tile_cols = np.repeat(tile[trow0:trow1], 128, axis=0)[: r1 - r0]
                    tile_g = tile_cols[:, :, None]
                    with np.errstate(divide="ignore"):
                        e_log = np.ceil(np.log2(np.maximum(tile_cols, 1e-38)))
                    code = np.clip((e_log + 127).astype(np.int32), 0, 254).astype(np.uint8)
                    new_scale[r0:r1] = code
                    pow2 = 2.0 ** (code.astype(np.float32) - 127.0)
                    ratio = tile_g / pow2[:, :, None]  # (0.5, 1]
                    old_vals = _E4M3_LUT[block.reshape(-1)].reshape(r1 - r0, cols // 128, 128)
                    scaled = np.abs(old_vals) * ratio
                    nbytes = e4m3_encode_mag(scaled)
                    sign = (block.reshape(r1 - r0, cols // 128, 128) & 0x80) & (
                        (scaled != 0).astype(np.uint8) << 7)
                    out_block = (nbytes | sign).reshape(-1).tobytes()
                    f.write(out_block)
                    cursor += len(out_block)
                    new_vals = _E4M3_LUT[np.frombuffer(out_block, dtype=np.uint8)].reshape(
                        r1 - r0, cols // 128, 128)
                    denom = np.maximum(np.abs(old_vals * tile_g), 1e-30)
                    rel = np.abs(new_vals * pow2[:, :, None] - old_vals * tile_g) / denom
                    finite = np.isfinite(rel)
                    if finite.any():
                        max_rel = max(max_rel, float(rel[finite].max()))
                        sum_rel += float(rel[finite].sum())
                        n_rel += int(finite.sum())
                scale_blob = new_scale.reshape(-1).tobytes()
                f.write(scale_blob)
                cursor += len(scale_blob)
                soff_new, sbytes_new = poff_new + pbytes_new, len(scale_blob)
                e["new_fmt"], e["new_group"] = FMT_E8M0B128, 128
                stats.append((e["kind"], e["layer"], rows, cols, max_rel, sum_rel / max(n_rel, 1)))
            e["new_poff"], e["new_pbytes"] = poff_new, pbytes_new
            e["new_soff"], e["new_sbytes"] = soff_new, sbytes_new
        # rewrite the header's file_bytes (the loader bounds-checks offsets
        # against it; the first repack left the ORIGINAL size and every tail
        # entry past it failed validation)
        f.seek(112)
        f.write(struct.pack("<Q", cursor))
        # rewrite the directory in place with updated format/scale fields
        f.seek(dir_off)
        for e in entries:
            f.write(struct.pack("<6I", e["kind"], e["layer"], e["new_fmt"], e["rows"],
                                e["cols"], e["new_group"]))
            f.write(struct.pack("<4Q", e["new_poff"], e["new_pbytes"], e["new_soff"], e["new_sbytes"]))

    mx = [s for s in stats if s[4] > 0.0 or s[5] > 0.0]
    worst = max((s[4] for s in stats), default=0.0)
    mean = max((s[5] for s in stats), default=0.0)
    print(f"entries={len(entries)} converted={sum(1 for e in entries if e['fmt']==FMT_F32B128)} "
          f"worst_max_rel={worst:.5f} worst_mean_rel={mean:.6f}")
    for s in [s for s in stats if s[4] > 0][:6]:
        print(f"  kind={s[0]} layer={s[1]} {s[2]}x{s[3]} max_rel={s[4]:.5f} mean_rel={s[5]:.6f}")


if __name__ == "__main__":
    main()
