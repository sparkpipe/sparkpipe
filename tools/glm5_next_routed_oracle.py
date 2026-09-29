#!/usr/bin/env python3
"""Float64 oracle for the glm5_next routed-expert parity dumps (docs/ROUTED_EXPERT_PARITY.md)."""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

HEADER = struct.Struct("<20I2Q")
ENTRY = struct.Struct("<8I4Q")
EXPERT_UP_GATE = 22
EXPERT_DOWN = 23
HIDDEN = 4096
TOP_K = 8
EXPERTS = 288
SWIGLU_LIMIT = 10.0
BAND_ABS = 1e-3
BAND_REL = 0.02
BAND_OUTLIER_FRACTION = 1e-3
BAND_OUTLIER_FLOOR = 8
EMULATED_REL_L2 = 2.0 ** -8
EXACT_REL_L2 = 0.01
CODEC_BF16, CODEC_FP8, CODEC_NVFP4 = 1, 5, 6


def e4m3_table():
    table = np.zeros(256, dtype=np.float64)
    for b in range(256):
        sign = -1.0 if b & 0x80 else 1.0
        exponent = (b >> 3) & 15
        mantissa = b & 7
        if exponent == 15 and mantissa == 7:
            table[b] = np.nan
        elif exponent == 0:
            table[b] = sign * (mantissa / 8.0) * 2.0 ** -6
        else:
            table[b] = sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
    return table


E4M3 = e4m3_table()
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float64)


def bf16_to_f64(raw):
    return (raw.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


class Pack:
    def __init__(self, path):
        self.file = open(path, "rb")
        header = HEADER.unpack(self.file.read(HEADER.size))
        self.header = header
        tensor_count, directory = header[6], header[20]
        self.entries = {}
        self.file.seek(directory)
        for _ in range(tensor_count):
            e = ENTRY.unpack(self.file.read(ENTRY.size))
            if e[0] in (EXPERT_UP_GATE, EXPERT_DOWN):
                self.entries[(e[1], e[0])] = e

    def read(self, offset, size):
        self.file.seek(offset)
        data = self.file.read(size)
        if len(data) != size:
            raise SystemExit(f"short read at {offset}")
        return data

    def weight(self, layer, kind, expert):
        _, _, _, codec, encoding, groups, rows, cols, p_off, p_bytes, s_off, s_bytes = self.entries[(layer, kind)]
        per = p_bytes // groups
        payload = np.frombuffer(self.read(p_off + expert * per, per), dtype=np.uint8)
        if codec == CODEC_BF16:
            return bf16_to_f64(payload.view(np.uint16)).reshape(rows, cols), codec
        if codec == CODEC_FP8:
            per_scale = s_bytes // groups
            scale = np.frombuffer(self.read(s_off + expert * per_scale, per_scale), dtype=np.float32).astype(np.float64)
            scale = scale.reshape(rows, cols // 128)
            return E4M3[payload].reshape(rows, cols) * np.repeat(scale, 128, axis=1), codec
        if codec == CODEC_NVFP4:
            globals_bytes = groups * 4
            block = (s_bytes - globals_bytes) // groups
            global_scale = struct.unpack("<f", self.read(s_off + expert * 4, 4))[0]
            blocks = np.frombuffer(self.read(s_off + globals_bytes + expert * block, block), dtype=np.uint8)
            blocks = E4M3[blocks].reshape(rows, cols // 16)
            codes = np.empty(payload.size * 2, dtype=np.uint8)
            codes[0::2] = payload & 15
            codes[1::2] = payload >> 4
            values = E2M1[codes].reshape(rows, cols)
            return values * np.repeat(blocks, 16, axis=1) * float(global_scale), codec
        raise SystemExit(f"codec {codec} has no oracle (O1 codecs are code paths only)")


def bf16_round(values):
    as32 = np.asarray(values, dtype=np.float64).astype(np.float32)
    bits = as32.view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return bf16_to_f64(bits.astype(np.uint16))


def reference(pack, layer, x, experts, weights, emulate=False, weight_bf16=False):
    rows = x.shape[0]
    out = np.zeros((rows, HIDDEN), dtype=np.float64)
    keep = bf16_round if emulate else (lambda v: v)
    for expert in np.unique(experts):
        w1, _ = pack.weight(layer, EXPERT_UP_GATE, int(expert))
        w2, _ = pack.weight(layer, EXPERT_DOWN, int(expert))
        if weight_bf16:
            w1, w2 = bf16_round(w1), bf16_round(w2)
        width = w2.shape[1]
        for row, k in zip(*np.nonzero(experts == expert)):
            up_gate = keep(w1 @ x[row])
            up = np.clip(up_gate[:width], -SWIGLU_LIMIT, SWIGLU_LIMIT)
            gate = np.minimum(up_gate[width:], SWIGLU_LIMIT)
            hidden = keep(gate / (1.0 + np.exp(-gate)) * up)
            out[row] += weights[row, k] * keep(w2 @ hidden)
    return keep(out)


def score(candidate, ref):
    d = np.abs(candidate - ref)
    violations = int(np.count_nonzero((d > BAND_ABS) & (d > BAND_REL * np.abs(ref))))
    rel_l2 = float(np.linalg.norm(candidate - ref) / max(np.linalg.norm(ref), 1e-30))
    return {"max_abs": float(d.max()), "rel_l2": rel_l2, "band_violations": violations,
            "elements": int(d.size), "ref_rms": float(np.sqrt(np.mean(ref * ref)))}


CEILING_INPUTS = ("x.bf16", "route_expert.u32", "route_weight.f32")


def load_ceiling(ceiling_root, directory, rows):
    base = ceiling_root / directory.name
    for name in CEILING_INPUTS:
        if not (base / name).is_file() or (base / name).read_bytes() != (directory / name).read_bytes():
            raise SystemExit(f"ceiling {base} does not match {directory}: {name} is missing or differs")
    reference_path = base / "ref.f64"
    if not reference_path.is_file() or reference_path.stat().st_size != rows * HIDDEN * 8:
        raise SystemExit(f"ceiling {base} has no complete ref.f64; score the ceiling dump with this oracle first")
    return np.fromfile(reference_path, dtype=np.float64).reshape(rows, HIDDEN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path)
    parser.add_argument("--ceiling", type=Path, help="dump directory of the BF16-expert arm, already scored")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    meta = json.loads((args.dump / "meta.json").read_text())
    pack = Pack(meta["pack"])
    receipt = {"codec": meta["codec_name"], "pack": meta["pack"], "header_expert_codec": meta["header_expert_codec"],
               "band": {"abs": BAND_ABS, "rel": BAND_REL, "outlier_fraction": BAND_OUTLIER_FRACTION, "outlier_floor": BAND_OUTLIER_FLOOR,
                        "emulated_rel_l2": EMULATED_REL_L2, "exact_rel_l2": EXACT_REL_L2}, "runs": []}
    failed = False
    for run in meta["runs"]:
        directory = Path(run["dir"])
        rows = run["rows"]
        x = bf16_to_f64(np.fromfile(directory / "x.bf16", dtype=np.uint16)).reshape(rows, HIDDEN)
        experts = np.fromfile(directory / "route_expert.u32", dtype=np.uint32).reshape(rows, TOP_K)
        weights = np.fromfile(directory / "route_weight.f32", dtype=np.float32).astype(np.float64).reshape(rows, TOP_K)
        ref = reference(pack, run["layer"], x, experts, weights)
        ref.tofile(directory / "ref.f64")
        emulated = {False: reference(pack, run["layer"], x, experts, weights, emulate=True)}
        ceiling = load_ceiling(args.ceiling, directory, rows) if args.ceiling is not None else None
        entry = {"layer": run["layer"], "rows": rows, "natural_path": run["natural_path"],
                 "natural_equals_path": run["natural_equals_path"], "status": run["status"], "paths": {}}
        for path, status in run["status"].items():
            if status != 0:
                entry["paths"][path] = {"status": status, "ran": False}
                continue
            out = bf16_to_f64(np.fromfile(directory / f"out_{path}.bf16", dtype=np.uint16)).reshape(rows, HIDDEN)
            gemm = path == "gemm" or (path == "natural" and run["natural_path"] == "gemm")
            if gemm not in emulated:
                emulated[gemm] = reference(pack, run["layer"], x, experts, weights, emulate=True, weight_bf16=True)
            result = {"status": 0, "ran": True, "weight_rounded_to_bf16": gemm,
                      "vs_emulated_f64": score(out, emulated[gemm]), "vs_exact_codec_f64": score(out, ref)}
            if ceiling is not None:
                result["vs_bf16_ceiling_f64"] = score(out, ceiling)
            own = result["vs_emulated_f64"]
            result["pass"] = (own["band_violations"] <= max(BAND_OUTLIER_FLOOR, BAND_OUTLIER_FRACTION * own["elements"])
                              and own["rel_l2"] <= EMULATED_REL_L2
                              and result["vs_exact_codec_f64"]["rel_l2"] <= EXACT_REL_L2)
            failed |= not result["pass"]
            entry["paths"][path] = result
            own = result["vs_emulated_f64"]
            exact = result["vs_exact_codec_f64"]
            line = (f"layer={run['layer']:2d} rows={rows:3d} path={path:8s} emulated: rel_l2={own['rel_l2']:.2e} max_abs={own['max_abs']:.2e} "
                    f"band_viol={own['band_violations']}/{own['elements']} | exact: rel_l2={exact['rel_l2']:.3e} rms={exact['ref_rms']:.3f}")
            if ceiling is not None:
                line += f" | vs_bf16_ceiling rel_l2={result['vs_bf16_ceiling_f64']['rel_l2']:.3e}"
            print(line + (" PASS" if result["pass"] else " FAIL"))
        failed |= not run["natural_equals_path"]
        receipt["runs"].append(entry)
    receipt["verdict"] = "FAIL" if failed else "PASS"
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt, indent=1) + "\n")
    print(f"ROUTED-ORACLE codec={meta['codec_name']} {receipt['verdict']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
