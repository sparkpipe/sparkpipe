#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("EXL3_INT8_GEMV", "0")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exl3_expert_dequant import (
    CODEBOOK_NAMES, CODEBOOKS, HAD, TILE, Checkpoint, Decoder, bf16_to_f64, codebook_lut, load_linear,
    slice_linear,
)

INT8_GEMV_MODE = os.environ["EXL3_INT8_GEMV"]
GATE4_RELATIVE = 2.0 ** -10
GATE5_RELATIVE = 4e-3


def torch_modules():
    import torch
    import exllamav3
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.quant.exl3 import LinearEXL3
    return torch, exllamav3, ext, LinearEXL3


def gpu_inner(torch, ext, trellis: np.ndarray, codebook: int) -> np.ndarray:
    k, n = trellis.shape[0] * TILE, trellis.shape[1] * TILE
    w = torch.empty((k, n), dtype=torch.half, device="cuda")
    t = torch.from_numpy(np.ascontiguousarray(trellis)).cuda()
    ext.reconstruct(w, t, trellis.shape[2] // TILE, codebook == 1, codebook == 2)
    torch.cuda.synchronize()
    return w.cpu().numpy()


def synthetic_gate(decoder: Decoder, seed: int) -> dict:
    torch, _, ext, _ = torch_modules()
    rng = np.random.default_rng(seed)
    results = {}
    ok = True
    for codebook in (0, 1, 2):
        for bits in range(1, 9):
            trellis = rng.integers(-32768, 32768, size=(64, 64, TILE * bits), dtype=np.int64).astype(np.int16)
            mine = decoder.inner(trellis, codebook)
            theirs = gpu_inner(torch, ext, trellis, codebook)
            equal = bool(np.array_equal(mine.view(np.uint16), theirs.view(np.uint16)))
            coverage = int(np.unique(decoder.states(trellis)).size)
            results[f"{CODEBOOK_NAMES[codebook]}.K{bits}"] = {"bit_exact": equal, "states_covered": coverage}
            ok = ok and equal
    luts = {}
    for name, codebook in CODEBOOKS.items():
        luts[name] = hashlib.sha256(codebook_lut(codebook).tobytes()).hexdigest()
    results["lut_sha256"] = luts
    results["pass"] = ok
    return results


def full_state_gate(decoder: Decoder) -> dict:
    torch, _, ext, _ = torch_modules()
    bits = 8
    results = {}
    ok = True
    for codebook in (0, 1, 2):
        lut = codebook_lut(codebook)
        seen = np.zeros(1 << 16, dtype=bool)
        mismatches = 0
        for batch_start in range(0, 1 << 16, 1 << 12):
            tiles = []
            for state_block in range(batch_start, batch_start + (1 << 12), 256):
                high = state_block >> 8
                seq = np.empty(256, dtype=np.uint8)
                seq[0::2] = high
                seq[1::2] = np.arange(128, dtype=np.uint8) * 2
                tiles.append(seq.copy())
                seq2 = seq.copy()
                seq2[1::2] = np.arange(128, dtype=np.uint8) * 2 + 1
                tiles.append(seq2)
            data = np.stack(tiles)
            count = data.shape[0]
            pad = (-count) % 8
            if pad:
                data = np.concatenate([data, np.zeros((pad, 256), dtype=np.uint8)])
            packed = pack_bytes_msb(data)
            trellis = packed.reshape(-1, 8, TILE * bits).view(np.int16)
            mine_states = decoder.states(trellis)
            theirs = gpu_inner(torch, ext, trellis, codebook)
            theirs_tiles = theirs.reshape(trellis.shape[0], TILE, trellis.shape[1], TILE).transpose(0, 2, 1, 3)
            theirs_flat = theirs_tiles.reshape(-1, 256).view(np.uint16)
            expected = lut[mine_states].view(np.uint16)
            mismatches += int(np.count_nonzero(expected != theirs_flat))
            seen[mine_states.reshape(-1)] = True
        results[CODEBOOK_NAMES[codebook]] = {"states_covered": int(seen.sum()), "lut_mismatches": mismatches}
        ok = ok and mismatches == 0 and int(seen.sum()) == (1 << 16)
    results["pass"] = ok
    return results


def pack_bytes_msb(data: np.ndarray) -> np.ndarray:
    words = data.reshape(data.shape[0], -1, 4).astype(np.uint32)
    value = (words[:, :, 0] << 24) | (words[:, :, 1] << 16) | (words[:, :, 2] << 8) | words[:, :, 3]
    return value.astype(np.uint32).view(np.uint16).reshape(data.shape[0], -1)


def real_gate(decoder: Decoder, source: Checkpoint, prefix: str, reference: Checkpoint | None,
              reference_name: str | None, slice_axis: str | None, slice_start: int, slice_count: int,
              seed: int) -> dict:
    torch, _, ext, LinearEXL3 = torch_modules()
    trellis, suh, svh, codebook = load_linear(source, prefix)
    full_codes = None
    if slice_axis:
        full_codes, _ = decoder.weight_bf16(trellis, suh, svh, codebook)
        trellis, suh, svh = slice_linear(trellis, suh, svh, slice_axis, slice_start, slice_count)
    k, n = trellis.shape[0] * TILE, trellis.shape[1] * TILE
    record = {"prefix": prefix, "codebook": CODEBOOK_NAMES[codebook], "bits": trellis.shape[2] // TILE,
              "k_in": k, "n_out": n, "slice": [slice_axis, slice_start, slice_count] if slice_axis else None,
              "trellis_sha256": hashlib.sha256(trellis.tobytes()).hexdigest()}
    mine_inner = decoder.inner(trellis, codebook)
    theirs_inner = gpu_inner(torch, ext, trellis, codebook)
    record["gate2_inner_bit_exact"] = bool(np.array_equal(mine_inner.view(np.uint16), theirs_inner.view(np.uint16)))
    w64 = decoder.weight_f64(trellis, suh, svh, codebook)
    codes, corrected = decoder.weight_bf16(trellis, suh, svh, codebook)
    wb = bf16_to_f64(codes)
    ulp = np.ldexp(1.0, np.frexp(np.abs(wb))[1] - 8)
    record["gate3_max_err_over_half_ulp"] = float(np.max(np.abs(wb - w64) / (0.5 * ulp)))
    record["gate3_tie_corrections"] = corrected
    if full_codes is not None:
        part = full_codes[:, slice_start:slice_start + slice_count] if slice_axis == "out" \
            else full_codes[slice_start:slice_start + slice_count, :]
        record["slice_equals_full_decode"] = bool(np.array_equal(part, codes))
    t = torch.from_numpy(np.ascontiguousarray(trellis)).cuda()
    su = torch.from_numpy(suh.copy()).cuda()
    sv = torch.from_numpy(svh.copy()).cuda()
    marker = torch.zeros((), dtype=torch.int32, device="cuda")
    linear = LinearEXL3(None, k, n, suh=su, svh=sv, trellis=t,
                        mcg=marker if codebook == 1 else None, mul1=marker if codebook == 2 else None,
                        key=prefix)
    gwt = linear.get_weight_tensor().float().cpu().numpy().astype(np.float64)
    rms = float(np.sqrt(np.mean(w64 * w64)))
    floor = rms * 2.0 ** -6
    diff = gwt - w64
    record["gate4_rel_fro_vs_get_weight_tensor"] = float(np.linalg.norm(diff) / np.linalg.norm(w64))
    record["gate4_max_abs_over_max_w"] = float(np.max(np.abs(diff)) / np.max(np.abs(w64)))
    record["gate4_max_elementwise_rel_floor_rms_2m6"] = float(np.max(np.abs(diff) / np.maximum(np.abs(w64), floor)))
    rng = np.random.default_rng(seed)
    forward = {}
    for rows in (1, 8, 256):
        x = rng.standard_normal((rows, k)).astype(np.float16)
        xt = torch.from_numpy(x).cuda()
        y = linear.forward(xt, {}).float().cpu().numpy().astype(np.float64)
        y_ref = x.astype(np.float64) @ w64
        y_bf16 = x.astype(np.float64) @ wb
        denom = float(np.linalg.norm(y_ref))
        forward[str(rows)] = {"rel_fro_exllamav3_vs_fp64": float(np.linalg.norm(y - y_ref) / denom),
                              "rel_fro_bf16_vs_fp64": float(np.linalg.norm(y_bf16 - y_ref) / denom)}
    record["gate5_forward"] = forward
    if reference is not None and reference_name:
        ref = reference.array(reference_name)
        if ref.dtype == np.uint16:
            ref64 = bf16_to_f64(ref)
        else:
            ref64 = ref.astype(np.float64)
        ref64 = ref64.T
        if slice_axis == "out":
            ref64 = ref64[:, slice_start:slice_start + slice_count]
        elif slice_axis == "in":
            ref64 = ref64[slice_start:slice_start + slice_count, :]
        if ref64.shape != w64.shape:
            record["audit_error"] = f"reference {reference_name} shape {ref64.shape} != {w64.shape}"
        else:
            err = w64 - ref64
            record["audit_rel_rmse_vs_reference"] = float(np.sqrt(np.mean(err * err) / np.mean(ref64 * ref64)))
            record["audit_reference"] = reference_name
    record["pass"] = (record.get("slice_equals_full_decode", True) and record["gate2_inner_bit_exact"] and record["gate3_max_err_over_half_ulp"] <= 1.0
                      and record["gate4_rel_fro_vs_get_weight_tensor"] <= GATE4_RELATIVE
                      and record["gate4_max_abs_over_max_w"] <= GATE4_RELATIVE
                      and all(v["rel_fro_exllamav3_vs_fp64"] <= GATE5_RELATIVE for v in forward.values()))
    return record


def environment() -> dict:
    torch, exllamav3, _, _ = torch_modules()
    from exllamav3.version import __version__ as exllamav3_version
    return {"exllamav3": exllamav3_version, "exllamav3_path": str(Path(exllamav3.__file__).parent),
            "torch": torch.__version__,
            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)), "numpy": np.__version__,
            "python": platform.python_version(), "EXL3_INT8_GEMV": INT8_GEMV_MODE}


def main() -> int:
    parser = argparse.ArgumentParser(description="Parity gates of exl3_expert_dequant against exllamav3.")
    parser.add_argument("--plan", required=True, help="JSON list of {source, prefix, reference?, reference_name?, slice?}")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    decoder = Decoder()
    started = time.time()
    report = {"environment": environment(), "synthetic": synthetic_gate(decoder, args.seed),
              "all_states": full_state_gate(decoder), "tensors": []}
    checkpoints = {}
    for item in json.loads(Path(args.plan).read_text()):
        source = checkpoints.setdefault(item["source"], Checkpoint(Path(item["source"])))
        reference = None
        if item.get("reference"):
            reference = checkpoints.setdefault(item["reference"], Checkpoint(Path(item["reference"])))
        sl = item.get("slice") or [None, 0, 0]
        record = real_gate(decoder, source, item["prefix"], reference, item.get("reference_name"),
                           sl[0], sl[1], sl[2], args.seed)
        record["source"] = item["source"]
        report["tensors"].append(record)
        print(json.dumps({k: record[k] for k in ("prefix", "codebook", "bits", "gate2_inner_bit_exact",
                                                  "gate3_max_err_over_half_ulp",
                                                  "gate4_rel_fro_vs_get_weight_tensor", "gate4_max_abs_over_max_w", "pass")}), flush=True)
    report["seconds"] = round(time.time() - started, 1)
    report["pass"] = (report["synthetic"]["pass"] and report["all_states"]["pass"]
                      and all(r["pass"] for r in report["tensors"]))
    Path(args.output).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"PARITY-{'PASS' if report['pass'] else 'FAIL'} tensors={len(report['tensors'])} "
          f"synthetic={report['synthetic']['pass']} all_states={report['all_states']['pass']}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
