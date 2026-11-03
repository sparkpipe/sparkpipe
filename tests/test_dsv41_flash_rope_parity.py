#!/usr/bin/env python3
"""GB10 bitwise parity of the DSV4.1 driver RoPE with DeepSeek's construction.

Builds build/test_dsv41_flash_rope_parity and runs it on an sm_121 device. The driver's two
frequency tables (theta 10000 plain, compress theta 160000 with YaRN), its sine and cosine at
positions up to 1,048,575, and its rotated bf16 rows (forward and inverse) must be the same bits
as the publisher inference/model.py construction run by torch on the same GPU: fp32
1 / base ** (2j / d), the YaRN blend, fp32 angle = position * frequency, polar. The rotation is
DeepSeek's production form fmaf(x0, cos, -(x1 * sin)), fmaf(x0, sin, x1 * cos) with a
round-to-nearest-even bf16 store; the publisher's complex multiply must agree on at least 99.9%
of the values.
"""

import math
import subprocess
import sys
import tempfile
from pathlib import Path

import gpu_test_runner

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_dsv41_flash_rope_parity"
PAIRS = 32
HEADS = 3


def reference_table(torch, base, original):
    freqs = 1.0 / (base ** (torch.arange(0, 2 * PAIRS, 2, dtype=torch.float32, device="cuda") / (2 * PAIRS)))
    if original > 0:
        def corrected(rotations):
            return 2 * PAIRS * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected(32)), 0)
        high = min(math.ceil(corrected(1)), 2 * PAIRS - 1)
        ramp = ((torch.arange(PAIRS, dtype=torch.float32, device="cuda") - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / 16 * (1 - smooth) + freqs * smooth
    return freqs


def bits(torch, tensor):
    return tensor.contiguous().view(torch.int32 if tensor.dtype == torch.float32 else torch.int16).cpu()


def main():
    if gpu_test_runner.built_binary("test_dsv41_flash_rope_parity", BINARY) is None:
        return 0
    try:
        import numpy as np
        import torch
    except ImportError:
        print("SKIP test_dsv41_flash_rope_parity (torch is not installed on this host)")
        return 0
    if not torch.cuda.is_available():
        print("SKIP test_dsv41_flash_rope_parity (torch has no CUDA device on this host)")
        return 0
    failures = []

    def check(condition, name):
        if not condition:
            failures.append(name)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "rope.bin"
        result = subprocess.run([str(BINARY), str(out)], capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout + result.stderr, end="")
            return 1
        rows = int(result.stdout.split()[1])
        raw = out.read_bytes()
    cursor = 0

    def take(dtype, count):
        nonlocal cursor
        array = np.frombuffer(raw, dtype=dtype, count=count, offset=cursor)
        cursor += array.nbytes
        return torch.from_numpy(array.copy())

    positions = take(np.uint32, rows).to(torch.int64)
    values = rows * HEADS * 2 * PAIRS
    data = take(np.uint16, values).view(torch.int16).view(torch.bfloat16).reshape(rows, HEADS, 2 * PAIRS)
    tables = take(np.float32, 2 * PAIRS).reshape(2, PAIRS)
    publisher_agree = publisher_total = 0
    for which, (base, original) in enumerate(((10000.0, 0), (160000.0, 65536))):
        name = "compress" if original else "full"
        sine = take(np.float32, rows * PAIRS).reshape(rows, PAIRS)
        cosine = take(np.float32, rows * PAIRS).reshape(rows, PAIRS)
        rotated = [take(np.uint16, values).view(torch.int16).reshape(rows, HEADS, 2 * PAIRS) for _ in range(2)]
        table = reference_table(torch, base, original)
        check(torch.equal(bits(torch, table), tables[which].view(torch.int32)), f"the {name} frequency table is the publisher fp32 table bit for bit")
        cis = torch.polar(torch.ones(rows, PAIRS, device="cuda"), torch.outer(positions.cuda(), table))
        check(torch.equal(bits(torch, cis.imag), sine.view(torch.int32)) and torch.equal(bits(torch, cis.real), cosine.view(torch.int32)),
              f"the {name} sine and cosine at every position are the publisher polar values bit for bit")
        x = data.cuda().float()
        real, imaginary = x[..., 0::2], x[..., 1::2]
        for inverse in (0, 1):
            c = cis.real[:, None, :]
            s = (-cis.imag if inverse else cis.imag)[:, None, :]
            first = (real.double() * c.double() - (imaginary * s).double()).float()
            second = (real.double() * s.double() + (imaginary * c).double()).float()
            expected = torch.stack((first, second), dim=-1).flatten(-2).to(torch.bfloat16)
            check(torch.equal(bits(torch, expected), rotated[inverse]),
                  f"the {name} {'inverse ' if inverse else ''}rotation is the fmaf form with a bf16 store bit for bit")
            turn = cis.conj() if inverse else cis
            publisher = torch.view_as_real(torch.view_as_complex(x.unflatten(-1, (-1, 2))) * turn[:, None, :]).flatten(-2).to(torch.bfloat16)
            publisher_agree += int((bits(torch, publisher) == rotated[inverse]).sum())
            publisher_total += values
    check(cursor == len(raw), "the binary wrote exactly the expected record")
    check(publisher_agree >= 0.999 * publisher_total, "the publisher complex multiply agrees on at least 99.9% of the rotated values")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print(f"PASS dsv41 rope parity: tables, sine/cosine at {rows} positions up to {int(positions.max())} and forward/inverse rotations "
          f"are DeepSeek's fp32 construction bit for bit; publisher complex multiply agrees on {publisher_agree}/{publisher_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
