#!/usr/bin/env python3
"""Run the real routing kernels on a CPU and score them against KimiMoEGate.

This path had three defects, all found by reading and none by running: the bias
folded into the mixture weight, the bias a scalar where 896 are needed, and no
renormalisation. Reading found them; nothing had executed them.

tests/host_cuda/router_host.cu includes norm.cuh and topk.cuh unmodified.

The reference, from KimiMoEGate.forward:

    scores            = sigmoid(logits)
    scores_for_choice = scores + e_score_correction_bias
    topk_idx          = topk(scores_for_choice, k)
    topk_weight       = scores.gather(topk_idx)        <- unbiased
    topk_weight      /= topk_weight.sum()              <- moe_renormalize

Three properties are checked, not one comparison. The chosen set must match.
The weights must come from the unbiased scores. And they must sum to one - a
kernel that skipped the renormalisation would still pick the right experts.
"""
import math
import subprocess
import sys
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tests" / "host_cuda" / "router_host.cu"
BINARY = Path("/tmp") / "lm_router_host"


def build():
    result = subprocess.run(
        [host_cuda_cxx(), "-std=c++17", "-O0", f"-I{ROOT}", f"-I{ROOT}/tests/host_cuda",
         "-x", "c++", str(SOURCE), "-o", str(BINARY)],
        capture_output=True, text=True)
    if result.returncode != 0:
        errors = [l for l in result.stderr.split("\n") if "error" in l]
        print("FAIL host build:", (errors or [result.stderr])[0][:200])
        return False
    return True


def main():
    if not build():
        return 1
    run = subprocess.run([str(BINARY)], capture_output=True, text=True)
    if run.returncode != 0:
        print("FAIL host run:", run.stderr.strip()[:200])
        return 1
    lines = run.stdout.strip().split("\n")
    header = lines[0].split()
    experts, top_k, rows = int(header[1]), int(header[3]), int(header[5])
    values = {"bias": [], "logit": [], "pick": [], "weight": []}
    for line in lines[1:]:
        tag, _, number = line.partition(" ")
        values[tag].append(float(number) if tag != "pick" else int(number))

    failures = 0
    bias_would_have_differed = 0
    for row in range(rows):
        logits = values["logit"][row * experts:(row + 1) * experts]
        scores = [1.0 / (1.0 + math.exp(-x)) for x in logits]
        biased = [s + b for s, b in zip(scores, values["bias"])]
        want = sorted(range(experts), key=lambda i: biased[i], reverse=True)[:top_k]
        got = values["pick"][row * top_k:(row + 1) * top_k]
        if set(got) != set(want):
            print(f"  FAIL row {row}: chose {sorted(got)}, reference {sorted(want)}")
            failures += 1
        # weights from the UNBIASED scores, renormalised
        raw = [scores[i] for i in got]
        total = sum(raw) + 1e-20
        want_weights = [r / total for r in raw]
        got_weights = values["weight"][row * top_k:(row + 1) * top_k]
        for expected, actual in zip(want_weights, got_weights):
            if abs(expected - actual) > 1e-5:
                print(f"  FAIL row {row}: weight {actual:.9g}, reference {expected:.9g}")
                failures += 1
                break
        if abs(sum(got_weights) - 1.0) > 1e-5:
            print(f"  FAIL row {row}: weights sum to {sum(got_weights):.9g}, not 1")
            failures += 1
        # would using the biased score as the weight have been visible here?
        biased_weights = [biased[i] for i in got]
        biased_total = sum(biased_weights)
        if any(abs(b / biased_total - w) > 1e-4
               for b, w in zip(biased_weights, want_weights)):
            bias_would_have_differed += 1

    print(f"experts {experts}  top_k {top_k}  rows {rows}")
    print(f"rows where weighing by the biased score would have differed: "
          f"{bias_would_have_differed}")
    if bias_would_have_differed == 0:
        print("\nFAIL the bias is too small to distinguish select-from-weigh; "
              "the test proves nothing about the defect it exists for")
        return 1
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nthe real routing kernels, run on a CPU, match KimiMoEGate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
