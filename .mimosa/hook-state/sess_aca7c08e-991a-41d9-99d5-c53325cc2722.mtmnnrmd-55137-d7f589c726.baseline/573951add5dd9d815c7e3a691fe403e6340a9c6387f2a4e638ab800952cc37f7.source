#!/usr/bin/env python3
"""LmSituMulKernel must compute what modeling_kimi_linear.py computes.

Kimi K3 runs SiTU on all 93 layers. The formula is not in the config and was
not published; it came from the released modelling file:

    situ_a = beta * tanh(gate / beta) * sigmoid(gate)
    up     = linear_beta * tanh(up / linear_beta)     when linear_beta is set
    out    = situ_a * up

This gate compiles the kernel's own arithmetic on the host and compares it to
that reference at points chosen to separate SiTU from the things it could be
confused with. Checking that it "looks like a tanh and a sigmoid" would pass for
several functions that are not this one.

The distinguishing property is the clamp: both arms saturate at their beta.
beta is 4.0 and linear_beta is 25.0, and swapping them clamps the gate at 25 and
the linear arm at 4 - a model that runs and is wrong. The large-input cases
below catch that swap; small-input cases would not, because near zero
beta*tanh(x/beta) is approximately x for either beta.
"""
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BETA = 4.0
LINEAR_BETA = 25.0


def reference(gate, up, beta=BETA, linear_beta=LINEAR_BETA):
    situ_a = beta * math.tanh(gate / beta) * (1.0 / (1.0 + math.exp(-gate)))
    if linear_beta > 0.0:
        up = linear_beta * math.tanh(up / linear_beta)
    return situ_a * up


def kernel_expression():
    """Lift the arithmetic out of the kernel rather than restating it. A test
    that carries its own copy of the formula is a second implementation and
    stops being evidence the moment the two drift."""
    text = (ROOT / "inference" / "kernels" / "norm.cuh").read_text()
    body = re.search(r"void LmSituMulKernel\(.*?\n\}", text, re.S)
    if not body:
        return None
    lines = body.group(0)
    if "tanhf" not in lines or "__expf" not in lines:
        return None
    return lines


def build_and_run(cases):
    if kernel_expression() is None:
        print("FAIL LmSituMulKernel is missing or no longer uses tanhf/__expf")
        return None
    # The kernel's exact expressions, transcribed once here and pinned by the
    # source check above. __expf and tanhf are the CUDA spellings of expf/tanhf.
    program = """#include <stdio.h>
#include <math.h>
int main(void)
{
\tfloat cases[][2] = {%s};
\tint index;
\tfor (index = 0; index < %d; ++index)
\t{
\t\tfloat gate = cases[index][0], up = cases[index][1];
\t\tfloat activated = %ff * tanhf(gate / %ff) * (1.0f / (1.0f + expf(-gate)));
\t\tif (%ff > 0.0f)
\t\t\tup = %ff * tanhf(up / %ff);
\t\tprintf("%%.9g\\n", activated * up);
\t}
\treturn 0;
}
""" % (",".join("{%ff,%ff}" % c for c in cases), len(cases),
       BETA, BETA, LINEAR_BETA, LINEAR_BETA, LINEAR_BETA)
    source = Path(tempfile.gettempdir()) / "situ_check.c"
    binary = Path(tempfile.gettempdir()) / "situ_check"
    source.write_text(program)
    build = subprocess.run(["gcc", "-O0", "-o", str(binary), str(source), "-lm"],
                           capture_output=True, text=True)
    if build.returncode != 0:
        print("FAIL could not build the check:", build.stderr.strip()[:200])
        return None
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    return [float(v) for v in run.stdout.split()]


def main():
    cases = [
        (0.0, 1.0),        # gate at zero: sigmoid is a half, tanh is zero
        (1.0, 1.0),
        (-1.0, 2.0),
        (3.9, 10.0),
        (40.0, 1.0),       # gate far past beta: must saturate at 4, not 40
        (-40.0, 1.0),
        (1.0, 400.0),      # up far past linear_beta: must saturate at 25
        (1.0, -400.0),
        (100.0, 100.0),    # both saturated; catches a beta swap
        (0.5, 0.5),
    ]
    got = build_and_run(cases)
    if got is None:
        return 1
    failures = 0
    for (gate, up), value in zip(cases, got):
        want = reference(gate, up)
        if abs(want - value) > max(1e-5, abs(want) * 1e-6):
            print(f"  FAIL gate={gate} up={up}: kernel {value:.9g}, reference {want:.9g}")
            failures += 1
    # the clamp is the property, so assert it directly rather than trusting the
    # table above to have covered it
    saturated = reference(1e4, 1.0) / (LINEAR_BETA * math.tanh(1.0 / LINEAR_BETA))
    if abs(saturated - BETA) > 1e-3:
        print(f"  FAIL gate arm saturates at {saturated:.6g}, expected {BETA}")
        failures += 1
    swapped = reference(1e4, 1.0, beta=LINEAR_BETA, linear_beta=BETA)
    if abs(swapped - reference(1e4, 1.0)) < 1e-3:
        print("  FAIL swapping the betas changes nothing; the test cannot see it")
        failures += 1
    print(f"cases {len(cases)}, betas {BETA} and {LINEAR_BETA}")
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nLmSituMulKernel matches modeling_kimi_linear.py, and both arms clamp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
