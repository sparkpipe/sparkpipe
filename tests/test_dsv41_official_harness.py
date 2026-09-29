#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

try:
    import torch
except ImportError:
    torch = None


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    if torch is None:
        print("SKIP test_dsv41_official_harness (torch unavailable on this host)")
        return 0
    import dsv41_official_harness as harness
    from t1_reference_common import bf16_to_f32, f32_to_bf16_u16
    from t1_reference_dsv41 import _fp4_qdq_e4m3_16, _fp4_qdq_e8m0, _fp8_qdq_rows

    ties = harness.round_e2m1(torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -2.5]))
    expect(ties.tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, -2.0], f"e2m1 ties must round to even: {ties.tolist()}")

    rng = np.random.default_rng(11)
    values = rng.standard_normal((6, 512)).astype(np.float32) * np.float32(2.0)
    values[1] *= np.float32(1e-3)
    values[2, :32] = np.float32(1.5)
    words = f32_to_bf16_u16(values)
    rounded = bf16_to_f32(words)
    as_torch = torch.from_numpy(rounded.copy()).to(torch.bfloat16)

    fp8 = harness.act_quant(as_torch.clone(), 32, "ue8m0", torch.float8_e8m0fnu, True).float().numpy()
    expect(np.array_equal(fp8, bf16_to_f32(f32_to_bf16_u16(_fp8_qdq_rows(rounded)))),
           "act_quant (fp8, ue8m0, block 32) differs from the reference _fp8_qdq_rows")

    q, s = harness.act_quant(as_torch.clone(), 32, "ue8m0", torch.float8_e8m0fnu, False)
    expect(np.array_equal((q.reshape(6, 16, 32) * s.reshape(6, 16, 1)).reshape(6, 512).numpy(), _fp8_qdq_rows(rounded)),
           "act_quant (non in-place) payload times scale differs from the reference")

    fp4 = harness.fp4_act_quant(as_torch.clone(), 32, True, torch.float8_e8m0fnu).float().numpy()
    expect(np.array_equal(fp4, bf16_to_f32(f32_to_bf16_u16(_fp4_qdq_e8m0(rounded)))),
           "fp4_act_quant (ue8m0, block 32) differs from the reference _fp4_qdq_e8m0")

    kv = harness.fp4_act_quant(as_torch.clone(), 16, True, torch.float8_e4m3fn).float().numpy()
    expect(np.array_equal(kv, bf16_to_f32(f32_to_bf16_u16(_fp4_qdq_e4m3_16(rounded)))),
           "fp4_act_quant (e4m3, block 16) differs from the reference _fp4_qdq_e4m3_16")

    print("PASS dsv41 official harness kernels: e2m1 ties, fp8 ue8m0/32, fp4 ue8m0/32 and fp4 e4m3/16 "
          "quantization equal the reference quantizers bit for bit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
