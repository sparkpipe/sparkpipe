#!/usr/bin/env python3
"""Independent CPU check: every o_proj (ATTN_OUTPUT / KDA_OUT) entry in a
glm5_next TP16 rank pack vs the checkpoint's column slice.

Deliberately does NOT import the packer's produce closures (the built-in
pack verifier already proves pack==packer; this proves packer==checkpoint
for the tensor family the fixed2 repack re-oriented). Own safetensors
reader, own e4m3 decode, own bf16 RNE cast, own column slice:
    expected = bf16( dequant(checkpoint o_proj)[:, c0:c1] )   (F8 source)
    expected = checkpoint o_proj[:, c0:c1]                     (BF16 source)

A shard-axis/offset/mapping defect mismatches wholesale (whole blocks
displaced); a rounding-mode difference is sparse 0-1 ulp. Decision rule:
exact match = PASS; <=0.01% elements off by <=1 ulp = PASS(rounding-note);
anything else = FAIL with the mismatch fingerprint.

usage: glm5_next_oproj_verify.py --pack <rank r>.g5nsp \
    --source /mnt/model-warm/glm-5.3-flash --tp-rank r [--layers 0-44]
"""
from __future__ import annotations

import argparse
import json
import mmap
import struct
import sys
from pathlib import Path

import numpy as np

ENTRY_BYTES = 64
K_ATTN_OUTPUT, K_KDA_OUT = 11, 36
K_EMBEDDING, K_LM_HEAD = 0, 2
K_KDA_QKV_BETA, K_KDA_DECAY_BIAS, K_KDA_HEAD_LOG_SCALE = 26, 33, 34
K_KV_B_KEY_T, K_KV_B_VALUE = 9, 10
TOTAL_LAYERS = 45          # 0..44 (44 = last KDA layer; MTP 45 not in packs)
HIDDEN = 4096
KDA_DIM, KDA_HEADS = 64 * 128, 64
QKV_BETA_SECTIONS = (KDA_DIM, KDA_DIM, KDA_DIM, KDA_HEADS)


def is_dsa(layer: int) -> bool:
    return layer >= 3 and layer < TOTAL_LAYERS and (layer - 3) % 4 == 0


class Safetensors:
    """Minimal mmap reader: header json -> (dtype, shape, offset)."""

    DT = {"BF16": (np.dtype("<u2"), 2), "F32": (np.dtype("<f4"), 4),
          "F8_E4M3": (np.dtype("u1"), 1)}

    def __init__(self, model_dir: Path):
        idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
        self.weight_map = idx["weight_map"]
        self.model_dir = model_dir
        self._open: dict = {}

    def meta(self, name: str):
        f = self.weight_map[name]
        if f not in self._open:
            fh = open(self.model_dir / f, "rb")
            n = struct.unpack("<Q", fh.read(8))[0]
            self._open[f] = (fh, json.loads(fh.read(n)), 8 + n)
        _fh, hdr, base = self._open[f]
        t = hdr[name]
        return t["dtype"], t["shape"], base + t["data_offsets"][0]

    def raw(self, name: str) -> np.ndarray:
        dtype, shape, off = self.meta(name)
        dt, _sz = self.DT[dtype]
        count = 1
        for d in shape:
            count *= d
        return np.memmap(self.model_dir / self.weight_map[name], dtype=dt,
                         mode="r", offset=off, shape=(count,)).reshape(shape)


def e4m3_decode(codes: np.ndarray) -> np.ndarray:
    """e4m3 (bias 7, subnormal, no inf; 0x7F = nan) -> f32, vectorized."""
    c = codes.astype(np.uint32)
    bits = c & 0x7F
    exp = (bits >> 3) & 0xF
    man = bits & 0x7
    vals = np.where(
        exp == 0,
        (man.astype(np.float32) / 8.0) * (2.0 ** -6),           # subnormal
        (1.0 + man.astype(np.float32) / 8.0) * (2.0 ** (exp.astype(np.float32) - 7.0)),
    )
    vals = np.where(bits == 0x7F, np.float32("nan"), vals)
    out = vals.copy()
    out[c & 0x80 != 0] = -out[c & 0x80 != 0]
    return out


def bf16_rne_u16(f32: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even f32->bf16 bit cast (independent write-up)."""
    bits = np.ascontiguousarray(f32, dtype=np.float32).view(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    return ((bits + np.uint32(0x7FFF) + lsb) >> np.uint32(16)).astype(np.uint16)


def expand_blocks(scale: np.ndarray, rows: int, cols: int) -> np.ndarray:
    return np.repeat(np.repeat(scale, 128, axis=0)[:rows], 128, axis=1)[:, :cols]


def parse_pack(path: Path):
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    dir_off = struct.unpack_from("<Q", mm, 80)[0]
    count = struct.unpack_from("<I", mm, 24)[0]
    entries = {}
    for i in range(count):
        e = struct.unpack_from("<8I4Q", mm, dir_off + i * ENTRY_BYTES)
        entries[(e[0], e[1])] = {"rows": e[6], "columns": e[7],
                                 "payload_offset": e[8], "payload_bytes": e[9]}
    return mm, entries


def bf16_matrix(st: "Safetensors", name: str, rows: int, cols: int) -> np.ndarray:
    """Full-width bf16 u16 view (F8 sources dequant through my own decode)."""
    dtype, shape, _ = st.meta(name)
    if dtype == "BF16":
        return st.raw(name).reshape(rows, cols)
    if dtype == "F8_E4M3":
        _dt, ssh, _ = st.meta(name + "_scale_inv")
        scale = st.raw(name + "_scale_inv").view("<f4")
        return bf16_rne_u16(e4m3_decode(st.raw(name).reshape(rows, cols))
                            * expand_blocks(scale.reshape(ssh), rows, cols))
    raise ValueError(f"{name}: {dtype}")


def f32_vector(st: "Safetensors", name: str) -> np.ndarray:
    dtype, shape, _ = st.meta(name)
    if dtype == "F32":
        return st.raw(name).view("<f4").reshape(-1)
    if dtype == "BF16":   # upcast exact
        return (st.raw(name).view("<u2").astype(np.uint32)
                << np.uint32(16)).view("<f4").reshape(-1)
    raise ValueError(f"{name}: {dtype}")


def check_entry(entries, kind, layer, rows, cols):
    ent = entries.get((kind, layer))
    if ent is None:
        return None
    if (ent["rows"], ent["columns"]) != (rows, cols):
        print(f"FAIL kind={kind} L{layer}: geometry {ent['rows']}x{ent['columns']}"
              f" != {rows}x{cols}")
        return "fail"
    return ent


def read_payload(mm, ent, rows, cols, item_bytes, dtype_str="<u2"):
    return np.frombuffer(mm, dtype=dtype_str, count=rows * cols,
                         offset=ent["payload_offset"]).reshape(rows, cols)


def compare(got, want, label, failures, notes):
    eq = got == want
    if eq.all():
        print(f"PASS {label} exact")
        return failures, notes
    bad = int((~eq).sum())
    frac = bad / eq.size
    diff = got.astype(np.int32) - want.astype(np.int32)
    max_ulp = int(np.abs(diff[~eq]).max())
    if frac <= 1e-4 and max_ulp <= 1:
        print(f"PASS {label} rounding-note: {bad}/{eq.size} off by <=1 ulp")
        return failures, notes + 1
    first = np.argwhere(~eq)[0]
    print(f"FAIL {label}: {bad}/{eq.size} mismatched (frac {frac:.4f},"
          f" max_ulp {max_ulp}, first at {tuple(first)})")
    return failures + 1, notes


def families(st, mm, entries, tp_rank: int, tp_degree: int):
    """Independent checks beyond o_proj: the fused q|k|v|beta sections,
    the f32 decay vectors, and the owner-rank embedding/head shards."""
    failures = notes = 0
    n = tp_degree
    for layer in (0, 17):
        if is_dsa(layer):
            continue
        a = f"model.language_model.layers.{layer}.self_attn."
        # fused q|k|v|beta: per-section row slices, vstacked in q,k,v,b order
        parts, total_rows = [], 0
        for name, sec_rows in zip(("q_proj", "k_proj", "v_proj", "b_proj"),
                                  QKV_BETA_SECTIONS):
            m = bf16_matrix(st, a + name + ".weight", sec_rows, HIDDEN)
            c = sec_rows // n
            parts.append(m[tp_rank * c:(tp_rank + 1) * c, :])
            total_rows += c
        want = np.vstack(parts)
        r = check_entry(entries, K_KDA_QKV_BETA, layer, total_rows, HIDDEN)
        if r == "fail":
            failures += 1
        elif r is not None:
            got = read_payload(mm, r, total_rows, HIDDEN, 2)
            failures, notes = compare(
                got, want, f"L{layer} qkv_beta fused [{total_rows}x{HIDDEN}]",
                failures, notes)
        # f32 vectors, cols-sharded
        for kind, cname in ((K_KDA_DECAY_BIAS, "dt_bias"),
                            (K_KDA_HEAD_LOG_SCALE, "A_log")):
            v = f32_vector(st, a + cname)
            c = v.shape[0] // n
            want_v = np.ascontiguousarray(v[tp_rank * c:(tp_rank + 1) * c])
            r = check_entry(entries, kind, layer, 1, c)
            if r == "fail":
                failures += 1
            elif r is not None:
                got_v = read_payload(mm, r, 1, c, 4, "<f4")
                failures, notes = compare(
                    got_v, want_v, f"L{layer} {cname} f32 [{c}]", failures, notes)
    # kv_b per-head split on DSA layers: key TRANSPOSED [h, latent, nope],
    # value [h, vdim, latent]; both replicated (not tp-sliced).
    for layer in (3, 43):
        name = f"model.language_model.layers.{layer}.self_attn.kv_b_proj.weight"
        _dt, shape, _ = st.meta(name)
        assert tuple(shape) == (64 * 512, 512), shape
        m = st.raw(name).reshape(64, 512, 512)
        r = check_entry(entries, K_KV_B_KEY_T, layer, 512, 256)
        if r == "fail":
            failures += 1
        elif r is not None:
            want_k = np.ascontiguousarray(m[:, :256, :].transpose(0, 2, 1))
            got_k = np.frombuffer(mm, dtype="<u2", count=want_k.size,
                                  offset=r["payload_offset"]).reshape(want_k.shape)
            failures, notes = compare(got_k, want_k, f"L{layer} kv_b key_t"
                                      f" [64x512x256]", failures, notes)
        r = check_entry(entries, K_KV_B_VALUE, layer, 256, 512)
        if r == "fail":
            failures += 1
        elif r is not None:
            want_v = np.ascontiguousarray(m[:, 256:, :])
            got_v = np.frombuffer(mm, dtype="<u2", count=want_v.size,
                                  offset=r["payload_offset"]).reshape(want_v.shape)
            failures, notes = compare(got_v, want_v, f"L{layer} kv_b value"
                                      f" [64x256x512]", failures, notes)
    # embedding + lm_head (owner rank packs only)
    for kind, name in ((K_EMBEDDING, "model.language_model.embed_tokens.weight"),
                       (K_LM_HEAD, "lm_head.weight")):
        _dt, shape, _ = st.meta(name)
        rows, cols = shape
        c = rows // n
        want_e = bf16_matrix(st, name, rows, cols)[tp_rank * c:(tp_rank + 1) * c, :]
        r = check_entry(entries, kind, 0xFFFFFFFF, c, cols)
        if r == "fail":
            failures += 1
        elif r is None:
            print(f"SKIP kind={kind} {name.split('.')[-2]} (not this pack's ownership)")
        else:
            got_e = read_payload(mm, r, c, cols, 2)
            failures, notes = compare(
                got_e, want_e, f"{name.split('.')[-2]} rows [{c}x{cols}]",
                failures, notes)
    return failures, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--tp-rank", type=int, required=True)
    ap.add_argument("--tp-degree", type=int, default=16)
    ap.add_argument("--layers", default=f"0-{TOTAL_LAYERS - 1}")
    ap.add_argument("--families", action="store_true",
                    help="also check qkv_beta fusion, f32 decay vectors,"
                         " embedding/head shards (independent expectations)")
    args = ap.parse_args()

    if "-" in args.layers:
        lo, hi = (int(x) for x in args.layers.split("-"))
        layers = list(range(lo, hi + 1))
    else:
        layers = [int(x) for x in args.layers.split(",")]

    st = Safetensors(Path(args.source))
    mm, entries = parse_pack(Path(args.pack))
    failures = notes = checked = 0

    for layer in layers:
        name = f"model.language_model.layers.{layer}.self_attn.o_proj.weight"
        kind = K_ATTN_OUTPUT if is_dsa(layer) else K_KDA_OUT
        ent = entries.get((kind, layer))
        dtype, shape, _ = st.meta(name)
        rows, cols = shape
        count = cols // args.tp_degree
        c0 = args.tp_rank * count
        if ent is None:
            print(f"FAIL L{layer}: no entry kind={kind} in pack directory")
            failures += 1
            continue
        if (ent["rows"], ent["columns"]) != (rows, count):
            print(f"FAIL L{layer}: entry geometry {ent['rows']}x{ent['columns']}"
                  f" != {rows}x{count}")
            failures += 1
            continue
        got = np.frombuffer(mm, dtype="<u2", count=rows * count,
                            offset=ent["payload_offset"]).reshape(rows, count)
        if dtype == "BF16":
            want = st.raw(name).reshape(rows, cols)[:, c0:c0 + count]
        else:
            codes = st.raw(name)
            _dt, ssh, _ = st.meta(name + "_scale_inv")
            scale = st.raw(name + "_scale_inv").view("<f4")
            want = bf16_rne_u16(e4m3_decode(codes)
                                * expand_blocks(scale.reshape(ssh), rows, cols)
                                )[:, c0:c0 + count]
        eq = got == want
        checked += 1
        if eq.all():
            print(f"PASS L{layer} {'DSA' if is_dsa(layer) else 'KDA'} o_proj"
                  f" [{rows}x{count}] exact ({dtype})")
            continue
        bad = int((~eq).sum())
        frac = bad / eq.size
        diff = got.astype(np.int32) - want.astype(np.int32)
        max_ulp = int(np.abs(diff[~eq]).max())
        if frac <= 1e-4 and max_ulp <= 1:
            print(f"PASS L{layer} ({dtype}) rounding-note: {bad}/{eq.size}"
                  f" off by <=1 ulp")
            notes += 1
        else:
            first = np.argwhere(~eq)[0]
            print(f"FAIL L{layer} ({dtype}): {bad}/{eq.size} mismatched"
                  f" (frac {frac:.4f}, max_ulp {max_ulp}, first at {tuple(first)})")
            failures += 1

    print(f"oproj-verify rank {args.tp_rank}: {checked} layers,"
          f" failures={failures} rounding-notes={notes}")
    if args.families:
        st2 = st  # reader already open
        f_fail, f_notes = families(st2, mm, entries, args.tp_rank, args.tp_degree)
        failures += f_fail
        notes += f_notes
        print(f"families-verify rank {args.tp_rank}: failures={f_fail}"
              f" rounding-notes={f_notes}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
