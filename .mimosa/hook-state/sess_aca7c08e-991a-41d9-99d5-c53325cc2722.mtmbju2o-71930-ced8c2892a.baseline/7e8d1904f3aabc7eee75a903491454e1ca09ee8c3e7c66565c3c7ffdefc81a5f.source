#!/usr/bin/env python3
"""Run the rest of K3's layer kernels on a CPU and score each against a reference.

Six kernels the KDA and router harnesses do not reach: the short convolution
with its Swish, per-head L2 normalisation, SiTU, the fused residual RMS norm,
the attention output gate, and the MoE finalize.

The finalize is the reason this exists in the shape it does. Its launch was
wrong four ways and compiled - tokens and dimension swapped, the expert id
where the packed row belongs, a 1D grid for a kernel indexing blockIdx.y.
test_kernel_launches.py catches that by reading the call site. This catches it
by running the kernel and comparing numbers, which is the half that survives
someone rewriting the call in a form the parser does not recognise.

Every reference here is written from the source the kernel was written from:
FlashKDA's ShortConvolution(activation='silu') and l2_normalize, the K3 report's
eq. 6 and eq. 12, and KimiSparseMoeBlock's weighted sum.
"""
import math
import subprocess
import sys
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tests" / "host_cuda" / "layer_host.cu"
BINARY = Path("/tmp") / "lm_layer_host"
TOLERANCE = 6e-3   # bf16 storage on both ends of every comparison


def build():
    result = subprocess.run(
        [host_cuda_cxx(), "-std=c++17", "-O0", f"-I{ROOT}", f"-I{ROOT}/tests/host_cuda",
         "-x", "c++", str(SOURCE), "-o", str(BINARY)],
        capture_output=True, text=True)
    if result.returncode != 0:
        errors = [l for l in result.stderr.split("\n") if "error" in l]
        print("FAIL host build:", (errors or [result.stderr])[0][:200])
        return None
    run = subprocess.run([str(BINARY)], capture_output=True, text=True)
    if run.returncode != 0:
        print("FAIL host run:", run.stderr.strip()[:200])
        return None
    return run.stdout


def silu(x):
    return x / (1.0 + math.exp(-x)) if x > -60 else 0.0


def compare(name, want, got, failures):
    if len(want) != len(got):
        print(f"  FAIL {name}: {len(got)} outputs, reference {len(want)}")
        return failures + 1, 0.0
    magnitude = max((abs(v) for v in want), default=0.0) or 1.0
    worst = max((abs(a - b) for a, b in zip(want, got)), default=0.0)
    relative = worst / magnitude
    if relative > TOLERANCE:
        print(f"  FAIL {name}: relative {relative:.3e}")
        return failures + 1, relative
    print(f"  ok   {name:12s} relative {relative:.2e}  over {len(got)} values")
    return failures, relative


def main():
    text = build()
    if text is None:
        return 1
    lines = text.strip().split("\n")
    header = lines[0].split()
    rows, heads, head_dim = int(header[1]), int(header[3]), int(header[5])
    channels, conv, inter, top_k = (int(header[7]), int(header[9]),
                                    int(header[11]), int(header[13]))
    v = {}
    for line in lines[1:]:
        tag, _, number = line.partition(" ")
        v.setdefault(tag, []).append(float(number))

    failures = 0

    # short convolution: shift the window, admit the token, dot the taps, Swish
    want = []
    window = list(v["window"])
    for row in range(rows):
        for channel in range(channels):
            taps = window[channel * conv:(channel + 1) * conv]
            taps = taps[1:] + [v["convin"][row * channels + channel]]
            window[channel * conv:(channel + 1) * conv] = taps
            total = sum(t * w for t, w in
                        zip(taps, v["convw"][channel * conv:(channel + 1) * conv]))
            want.append(silu(total))
    failures, _ = compare("conv+swish", want, v["convout"], failures)

    # per-head L2
    want = []
    for row in range(rows):
        for head in range(heads):
            base = row * channels + head * head_dim
            block = v["l2in"][base:base + head_dim]
            inverse = 1.0 / math.sqrt(sum(x * x for x in block) + 1e-6)
            want.extend(x * inverse for x in block)
    failures, _ = compare("l2 per head", want, v["l2out"], failures)

    # SiTU: beta*tanh(gate/beta)*sigmoid(gate) * linear_beta*tanh(up/linear_beta)
    want = []
    for row in range(rows):
        base = row * inter * 2
        for index in range(inter):
            gate = v["gateup"][base + index]
            up = v["gateup"][base + inter + index]
            activated = 4.0 * math.tanh(gate / 4.0) / (1.0 + math.exp(-gate))
            want.append(activated * (25.0 * math.tanh(up / 25.0)))
    failures, _ = compare("situ", want, v["situout"], failures)

    # fused residual RMS norm: (in + residual) normalised, times the weight
    want, want_residual = [], []
    for row in range(rows):
        base = row * channels
        summed = [v["normin"][base + i] + v["normres"][base + i]
                  for i in range(channels)]
        inverse = 1.0 / math.sqrt(sum(x * x for x in summed) / channels + 1e-5)
        want.extend(x * inverse * v["normw"][i] for i, x in enumerate(summed))
        want_residual.extend(summed)
    failures, _ = compare("rms norm", want, v["normout"], failures)
    failures, _ = compare("rms residual", want_residual, v["normresout"], failures)

    # output gate: value * sigmoid(gate)
    want = [a / (1.0 + math.exp(-b)) for a, b in zip(v["gated"], v["gateval"])]
    failures, _ = compare("output gate", want, v["gateout"], failures)

    # MoE finalize: weighted sum over the token's routes
    want = []
    for row in range(rows):
        for element in range(channels):
            total = 0.0
            for slot in range(top_k):
                packed = row * top_k + slot
                total += (v["expert"][packed * channels + element]
                          * v["rweight"][row * top_k + slot])
            want.append(total)
    failures, _ = compare("moe finalize", want, v["finalout"], failures)

    # AttnRes: normalise each candidate to score it, then mix the UNNORMALISED
    # candidates. Report eq. 9 and _apply_attn_res both do this, and normalising
    # the values too would flatten exactly the layers the mechanism weighs.
    sources = 4
    want = []
    for row in range(rows):
        candidates = []
        for source in range(sources - 1):
            # [source][row][channels]: the layout the driver's bank store writes
            # and the only one stable while the source count grows.
            base = (source * rows + row) * channels
            candidates.append(v["bank"][base:base + channels])
        candidates.append(v["partial"][row * channels:(row + 1) * channels])
        scores = []
        for candidate in candidates:
            inverse = 1.0 / math.sqrt(
                sum(x * x for x in candidate) / channels + 1e-5)
            scores.append(sum(x * inverse * w
                              for x, w in zip(candidate, v["attnresw"])))
        top = max(scores)
        weights = [math.exp(s - top) for s in scores]
        total = sum(weights)
        want.extend(sum(w * c[e] for w, c in zip(weights, candidates)) / total
                    for e in range(channels))
    failures, _ = compare("attn res", want, v["attnresout"], failures)

    print(f"\nrows {rows}  heads {heads}  channels {channels}  "
          f"tolerance {TOLERANCE:.0e}")
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("seven more kernels run on a CPU and match their references")
    return 0


if __name__ == "__main__":
    sys.exit(main())
