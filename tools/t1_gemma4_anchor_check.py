import argparse
import importlib
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_reference_common import bf16_to_f32, f32_to_bf16_u16  # noqa: E402


def load_bins(directory):
    arrays = {}
    with open(os.path.join(directory, "manifest.txt")) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"bad manifest line: {line}")
            name, rows, cols, code = parts[0], int(parts[1]), int(parts[2]), \
                int(parts[3])
            dtype = {0: np.uint16, 2: np.float32, 3: np.float64}[code]
            count = rows * cols
            raw = open(os.path.join(directory, name), "rb").read()
            if len(raw) != count * dtype().itemsize:
                raise ValueError(f"{name}: extent mismatch")
            arrays[name[:-4] if name.endswith(".bin") else name] = \
                np.frombuffer(raw, dtype=dtype).reshape(rows, cols)
    return arrays


def ulp_distance(want_u16, got_f32):
    got_u16 = f32_to_bf16_u16(np.asarray(got_f32, dtype=np.float32).reshape(-1))
    want = want_u16.reshape(-1).astype(np.int32)
    got = got_u16.astype(np.int32)
    return int(np.max(np.abs(want - got))) if want.size else 0


def build_engine(arguments):
    from t1_reference_common import parse_llm_defines
    module = importlib.import_module("t1_reference_gemma4")
    config = json.load(open(os.path.join(arguments.checkpoint, "config.json")))
    if "text_config" in config:
        config = config["text_config"]
    defines = parse_llm_defines(arguments.header)
    return module.ENGINE_CLASS(arguments.checkpoint, defines, config)


def check_rope_tables(engine, bins, report):
    positions = [int(v) for v in bins["31B__rope__positions"].reshape(-1)]
    worst = 0
    for position in positions:
        for kind in ("sliding", "full"):
            cos, sin = engine.cos_sin(kind, position)
            worst = max(worst, ulp_distance(bins[f"31B__rope__cos_{kind}_{position}"],
                                            cos))
            worst = max(worst, ulp_distance(bins[f"31B__rope__sin_{kind}_{position}"],
                                            sin))
    report.append(("rope_tables_bitwise", worst, 0))
    return worst == 0


def check_keqv(engine, bins, report):
    q_raw = bins["31B__keqv__q_raw"].astype(np.uint16)
    k_raw = bins["31B__keqv__k_raw"].astype(np.uint16)
    q_norm_w = bf16_to_f32(bins["31B__keqv__q_norm_w"].astype(np.uint16).reshape(-1))
    k_norm_w = bf16_to_f32(bins["31B__keqv__k_norm_w"].astype(np.uint16).reshape(-1))
    cos = bf16_to_f32(bins["31B__keqv__cos"].astype(np.uint16))
    sin = bf16_to_f32(bins["31B__keqv__sin"].astype(np.uint16))
    dim = q_raw.shape[1] // 32
    worst = 0
    for row in range(q_raw.shape[0]):
        q = engine.head_rms(bf16_to_f32(q_raw[row]), q_norm_w, 32, dim)
        v = engine.head_rms(bf16_to_f32(k_raw[row]), None, 4, dim, scaled=False)
        k = engine.head_rms(bf16_to_f32(k_raw[row]), k_norm_w, 4, dim)
        k_rope = engine.rope(k.reshape(4, dim), cos[row], sin[row])
        q_worst = max(ulp_distance(bins["31B__keqv__q_norm"][row], q),
                      ulp_distance(bins["31B__keqv__v_norm"][row], v),
                      ulp_distance(bins["31B__keqv__k_norm"][row], k))
        rope_exact = ulp_distance(bins["31B__keqv__k_rope"][row], k_rope)
        worst = max(worst, q_worst, rope_exact)
    report.append(("keqv_norm_v_eq_k_ulp", worst, 1))
    report.append(("keqv_k_rope_ulp", worst, 0))
    return worst == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--header", required=True)
    arguments = parser.parse_args()
    bins = load_bins(arguments.anchors)
    engine = build_engine(arguments)
    report = []
    results = [
        check_rope_tables(engine, bins, report),
        check_keqv(engine, bins, report),
    ]
    for name, worst, limit in report:
        print(f"{'PASS' if worst <= limit else 'FAIL'} {name}: worst {worst} ulp "
              f"(limit {limit})")
    if all(results):
        print("RESULT: PASS (gemma4 warm-checkpoint anchor oracle)")
        return 0
    print("RESULT: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
