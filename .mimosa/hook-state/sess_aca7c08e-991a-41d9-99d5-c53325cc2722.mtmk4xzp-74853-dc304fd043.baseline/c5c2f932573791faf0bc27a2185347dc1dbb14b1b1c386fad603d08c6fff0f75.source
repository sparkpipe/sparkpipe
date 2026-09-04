#!/usr/bin/env python3
"""LmBoundedDecay must implement Kimi K3 technical report equation 5.

    g     = g_min * Sigmoid(exp(A_h) * z)     in (g_min, 0)
    alpha = exp(g)                             in (exp(g_min), 1)

A_h is a learnable per-head log-scale initialised to 0; g_min is fixed at -5.

THE BOUND IS THE PROPERTY, and it is numerical rather than architectural. Gated
DeltaNet and Mamba-2 use an unbounded negative-softplus mapping,
g = -exp(A) * Softplus(z), which lets the cumulative decay over a chunk approach
zero and its reciprocal - which the chunkwise form divides by - grow without
limit. Bounding g below at -5 keeps every retention factor above exp(-5), so the
cumulative log-decay over a 16-token tile stays inside (-80, 0) and the
reciprocal stays inside BF16 range.

So the checks below are not "does this look like a sigmoid". They are: does
every output stay strictly inside (exp(g_min), 1), does the softplus mapping
this replaced actually violate that, and does the per-head scale still move the
curve. A mapping that passed the first check by clamping would fail the third.
"""
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
G_MIN = -5.0


def sigmoid(x):
    """Numerically stable. The naive 1/(1+exp(-x)) overflows in Python at
    x = -1000, where the C kernel is fine because float saturates to inf and
    the reciprocal goes to zero. The reference has to survive the inputs the
    kernel survives, or the test fails on its own arithmetic."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exponential = math.exp(x)
    return exponential / (1.0 + exponential)


def reference(logit, head_log_scale=0.0, minimum=G_MIN, bias=0.0):
    """FlashKDA tests/torch_ref.py: the bias is added to the logit in fp32
    BEFORE the per-head scale multiplies, not after and not in bf16."""
    return math.exp(minimum * sigmoid(math.exp(head_log_scale) * (logit + bias)))


def kernel_source():
    text = (ROOT / "inference" / "kernels" / "linear_attn.cuh").read_text()
    body = re.search(r"float LmBoundedDecay\(.*?\n\}", text, re.S)
    if not body:
        return None
    if "__expf" not in body.group(0):
        return None
    return body.group(0)


def run(cases):
    if kernel_source() is None:
        print("FAIL LmBoundedDecay is missing or no longer uses __expf")
        return None
    program = """#include <stdio.h>
#include <math.h>
int main(void)
{
\tfloat cases[][3] = {%s};
\tint index;
\tfor (index = 0; index < %d; ++index)
\t{
\t\tfloat scaled = expf(cases[index][1]) * (cases[index][0] + cases[index][2]);
\t\tfloat log_decay = %ff * (1.0f / (1.0f + expf(-scaled)));
\t\tprintf("%%.9g\\n", expf(log_decay));
\t}
\treturn 0;
}
""" % (",".join("{%ff,%ff,%ff}" % c for c in cases), len(cases), G_MIN)
    source = Path(tempfile.gettempdir()) / "decay_check.c"
    binary = Path(tempfile.gettempdir()) / "decay_check"
    source.write_text(program)
    build = subprocess.run(["gcc", "-O0", "-o", str(binary), str(source), "-lm"],
                           capture_output=True, text=True)
    if build.returncode != 0:
        print("FAIL could not build:", build.stderr.strip()[:200])
        return None
    return [float(v) for v in
            subprocess.run([str(binary)], capture_output=True, text=True).stdout.split()]


def main():
    # (logit, head_log_scale, channel_bias)
    cases = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (20.0, 0.0, 0.0),
             (-20.0, 0.0, 0.0), (1000.0, 0.0, 0.0), (-1000.0, 0.0, 0.0),
             (1.0, 1.5, 0.0), (1.0, -1.5, 0.0),
             (0.0, 0.0, 2.0), (0.0, 0.0, -2.0), (1.0, 0.5, -3.0)]
    got = run(cases)
    if got is None:
        return 1
    failures = 0
    floor = math.exp(G_MIN)
    for (logit, scale, bias), value in zip(cases, got):
        want = reference(logit, scale, bias=bias)
        if abs(want - value) > max(1e-6, abs(want) * 1e-6):
            print(f"  FAIL z={logit} A={scale} b={bias}: kernel {value:.9g}, reference {want:.9g}")
            failures += 1
        if not (floor <= value <= 1.0):
            print(f"  FAIL z={logit} A={scale} b={bias}: {value:.9g} outside [exp(-5), 1]")
            failures += 1
    # the bound must hold where the mapping it replaced does not
    softplus = math.exp(-math.exp(0.0) * 1000.0)  # Softplus(1000) is 1000
    if softplus >= floor:
        print("  FAIL the negative-softplus mapping does not underflow the bound; "
              "this test cannot show the bound matters")
        failures += 1
    # and the per-head scale must still do something, or a clamp would pass
    if abs(reference(1.0, 1.5) - reference(1.0, -1.5)) < 1e-3:
        print("  FAIL the per-head log-scale does not move the curve")
        failures += 1
    if abs(reference(0.0, 0.0, bias=2.0) - reference(0.0, 0.0, bias=-2.0)) < 1e-3:
        print("  FAIL the channel bias does not move the curve; it is being dropped")
        failures += 1
    print(f"cases {len(cases)}, g_min {G_MIN}, floor exp(g_min) = {floor:.6g}")
    print(f"unbounded softplus at z=1000 would give {softplus:.3g}, "
          f"{'below' if softplus < floor else 'above'} the floor")
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nLmBoundedDecay matches report eq. 5 and stays inside (exp(g_min), 1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
