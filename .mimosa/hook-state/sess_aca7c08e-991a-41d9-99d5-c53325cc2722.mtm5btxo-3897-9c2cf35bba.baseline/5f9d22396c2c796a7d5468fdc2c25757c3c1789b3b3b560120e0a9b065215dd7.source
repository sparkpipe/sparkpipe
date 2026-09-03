#!/usr/bin/env python3
"""Run the real KDA decode kernels on a CPU and score them against the reference.

We have no hardware, so review and emulation are the whole verification budget.
Emulation is the stronger half: review says the arithmetic looks right, a run
says it produced these numbers and the reference produced those.

tests/host_cuda/kda_host.cu includes inference/kernels/linear_attn.cuh
UNMODIFIED and gives it a grid. The kernels compiled there are the kernels
compiled for sm_121a. A reimplementation inside the test would only prove the
test agrees with itself.

The reference below is the recurrence from the K3 report and FlashKDA's
torch_ref, written out here in float:

    alpha     = exp(g_min * sigmoid(exp(A_h) * (z + bias)))
    predicted = (Diag(alpha) S)^T k
    S         = Diag(alpha) S + beta (v - predicted) k^T
    o         = S^T q                       with the UPDATED state

What this cannot catch is stated in the shim: races, warp assumptions, anything
that needs more than one thread. It catches arithmetic, indexing and argument
order, which is where every defect found in this tree has actually lived.
"""
import math
import subprocess
import sys
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tests" / "host_cuda" / "kda_host.cu"
BINARY = Path("/tmp") / "lm_kda_host"


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


def parse(text):
    lines = text.strip().split("\n")
    header = lines[0].split()
    shape = dict(heads=int(header[1]), key=int(header[3]),
                 value=int(header[5]), steps=int(header[7]),
                 gmin=float(header[9]))
    values = {"bias": [], "scale": [], "q": [], "k": [], "z": [], "v": [],
              "beta": [], "out": [], "ret": [], "replay_mismatch": [],
              "run_mismatch": [], "verify_mismatch": []}
    for line in lines[1:]:
        tag, _, number = line.partition(" ")
        values[tag].append(float(number.split()[0]))
    return shape, values


def reference(shape, values):
    heads, key_dim, value_dim = shape["heads"], shape["key"], shape["value"]
    steps, gmin = shape["steps"], shape["gmin"]
    state = [[[0.0] * value_dim for _ in range(key_dim)] for _ in range(heads)]
    out = []
    per_step_k = heads * key_dim
    per_step_v = heads * value_dim
    for step in range(steps):
        q = list(values["q"][step * per_step_k:(step + 1) * per_step_k])
        k = list(values["k"][step * per_step_k:(step + 1) * per_step_k])
        z = values["z"][step * per_step_k:(step + 1) * per_step_k]
        v = values["v"][step * per_step_v:(step + 1) * per_step_v]
        beta = values["beta"][step * heads:(step + 1) * heads]
        for head in range(heads):
            # the kernel L2-normalizes q and k per head before the recurrence
            # (reference: use_qk_l2norm_in_kernel=True), so the model does too
            base = head * key_dim
            kk = sum(k[base + i] * k[base + i] for i in range(key_dim))
            qq = sum(q[base + i] * q[base + i] for i in range(key_dim))
            ks = (kk + 1e-6) ** -0.5
            qs = (qq + 1e-6) ** -0.5
            for i in range(key_dim):
                k[base + i] *= ks
                q[base + i] *= qs
            alpha = []
            for channel in range(key_dim):
                index = head * key_dim + channel
                logit = z[index] + values["bias"][index]
                scaled = math.exp(values["scale"][head]) * logit
                sigmoid = 1.0 / (1.0 + math.exp(-scaled)) if scaled > -60 else 0.0
                alpha.append(math.exp(gmin * sigmoid))
            # the prediction reads the DECAYED state
            predicted = [
                sum(state[head][c][e] * alpha[c] * k[head * key_dim + c]
                    for c in range(key_dim))
                for e in range(value_dim)]
            for c in range(key_dim):
                for e in range(value_dim):
                    state[head][c][e] = (alpha[c] * state[head][c][e]
                                         + beta[head]
                                         * (v[head * value_dim + e] - predicted[e])
                                         * k[head * key_dim + c])
            for e in range(value_dim):
                out.append(sum(state[head][c][e] * q[head * key_dim + c]
                               for c in range(key_dim)))
    return out


def main():
    if not build():
        return 1
    run = subprocess.run([str(BINARY)], capture_output=True, text=True)
    if run.returncode != 0:
        print("FAIL host run:", run.stderr.strip()[:200])
        return 1
    shape, values = parse(run.stdout)
    want = reference(shape, values)
    got = values["out"]
    if len(want) != len(got):
        print(f"FAIL reference produced {len(want)} outputs, kernels {len(got)}")
        return 1

    worst = 0.0
    magnitude = max(abs(x) for x in want) or 1.0
    for expected, actual in zip(want, got):
        worst = max(worst, abs(expected - actual))
    relative = worst / magnitude

    print(f"heads {shape['heads']}  key {shape['key']}  value {shape['value']}  "
          f"steps {shape['steps']}")
    print(f"outputs compared {len(got)}")
    print(f"max |kernel - reference| = {worst:.3e}  relative {relative:.3e}")

    # bf16 carries about three decimal digits, and the state accumulates across
    # six steps, so the bar is 1e-2 relative. Tighter would fail on rounding;
    # looser would pass with the decay dropped, which is the bug this path had.
    if relative > 1e-2:
        print("\nFAIL the kernels do not compute the reference recurrence")
        return 1

    # a run that agrees because everything is zero proves nothing
    if magnitude < 1e-4:
        print("\nFAIL the outputs are all near zero; the test cannot see anything")
        return 1

    # ReplaySSM: folding the accepted prefix from a checkpoint must land on the
    # state the recurrent path committed. Here the checkpoint is zero and the
    # prefix is everything, so the two states must be identical - not close.
    # Anything else means the fold and the decode disagree, which is the drift
    # SGLang describes as leaving "every output looking correct".
    mismatch = values["replay_mismatch"][0] if values["replay_mismatch"] else -1
    if mismatch != 0:
        print(f"\nFAIL the replay fold differs from the decode state in "
              f"{mismatch:.0f} bytes")
        return 1
    print("replay fold reproduces the committed state exactly, 0 bytes differ")
    # THE RUN IS ITS STEPS. One kernel call over the whole sequence must equal
    # the six decode calls byte for byte - outputs and final state - because
    # the state crosses tokens in shared memory at the slot's own BF16 width.
    # Prefill and DSpark verify both stand on this equivalence.
    run = values["run_mismatch"][0] if values["run_mismatch"] else -1
    if run != 0:
        print(f"\nFAIL the sequence run differs from its decode steps "
              f"({run:.0f} values)")
        return 1
    print("a six-token run equals six decode calls, bit for bit")
    # AND AN UNCOMMITTED RUN TOUCHES NOTHING. commit == 0 is the verify mode:
    # every output produced, the slot exactly as it was found.
    verify = values["verify_mismatch"][0] if values["verify_mismatch"] else -1
    if verify != 0:
        print(f"\nFAIL the uncommitted run leaked state or changed outputs "
              f"({verify:.0f} values)")
        return 1
    print("an uncommitted run produces the outputs and abandons the state")
    # SENSITIVITY, MEASURED RATHER THAN ASSUMED. Perturbing the fold's retention
    # factor by 0.1% moves no bf16 byte at all; 1% moves five. So this catches a
    # fold that reads a different gate, and does not catch one that differs
    # below a bf16 ULP - which is the right floor, because a drift smaller than
    # the state's own quantisation is not a drift in the state.

    print("\nthe real KDA kernels, run on a CPU, match the reference recurrence")
    return 0


if __name__ == "__main__":
    sys.exit(main())
