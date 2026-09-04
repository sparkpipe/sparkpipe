#!/usr/bin/env python3
"""Gate the bf16-state delta-rule variant against the fp32 kernel's contracts.

tests/host_cuda/kda_bf16_state_host.cu includes inference/kernels/
linear_attn.cuh UNMODIFIED and runs both state-element instantiations of the
real kernels - float for the contract slot, uint16_t for the admission-time
bf16 option of kimi_k3's K3_KDA_STATE_SLOT_BYTES_BF16. The numbers below came
out of the code that will run on hardware; a reimplementation would only
prove the test agrees with itself.

The three properties gated are the ones the option signs, and each maps to a
line the harness prints:

  1. BIT-IDENTICAL GIVEN THE SAME FP32 INPUT STATE. exact_* covers a
     multi-step run whose states round-trip losslessly; step_* covers 64
     random single-step trials where the bf16 pool must hold exactly
     LmFloatToBf16 of the fp32 pool - the commit rounding and nothing else.
  2. BOUNDED DIVERGENCE FROM ARBITRARY FP32 STATE. Each commit adds at most
     half a bf16 ulp and the recurrence contracts the error by at most the
     retention factor, so the divergence follows a geometric series bounded
     by 0.5 / (1 - alpha_max) ulps at the state's operating magnitude (the
     absolute error the next step's fp32 math sees; per-element exponents
     are the wrong yardstick near cancellation). The harness computes the
     envelope from the retention factors it dealt; this gate measures the
     divergence after every one of 64 commits and holds it to the envelope.
  3. THE REPLAY FOLD IS BYTE-EXACT under bf16 state, both from the initial
     state and from a mid-run checkpoint over the accepted tail - the
     ReplaySSM property across a rounding boundary.
"""
import subprocess
import sys
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tests" / "host_cuda" / "kda_bf16_state_host.cu"
BINARY = Path("/tmp") / "lm_kda_bf16_state_host"


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
    values = {}
    for line in text.strip().split("\n"):
        tag, _, number = line.partition(" ")
        values[tag] = float(number.split()[0])
    return values


def main():
    if not build():
        return 1
    run = subprocess.run([str(BINARY)], capture_output=True, text=True)
    if run.returncode != 0:
        print("FAIL host run:", run.stderr.strip()[:200])
        return 1
    values = parse(run.stdout)

    failures = 0

    def demand(condition, message):
        nonlocal failures
        if not condition:
            print(f"  FAIL {message}")
            failures += 1

    # 1. the same fp32 input state, the same bits out
    demand(values.get("exact_output_mismatch", -1) == 0,
           f"lossless run: {values.get('exact_output_mismatch'):.0f} output "
           f"values differ between the fp32 and bf16 kernels")
    demand(values.get("exact_state_mismatch", -1) == 0,
           f"lossless run: {values.get('exact_state_mismatch'):.0f} state "
           f"elements are not the exact bf16 of the fp32 kernel's")
    demand(values.get("step_trials", 0) == 64,
           "the single-step trials did not run")
    demand(values.get("step_output_mismatch", -1) == 0,
           f"{values.get('step_output_mismatch'):.0f} single-step outputs "
           f"differ; given the same fp32 input state the per-step math must "
           f"be bit-identical")
    demand(values.get("step_store_mismatch", -1) == 0,
           f"{values.get('step_store_mismatch'):.0f} committed elements are "
           f"not exactly LmFloatToBf16 of the fp32 kernel's state; the "
           f"commit store is the only rounding the option may add")
    print("bit-identical per step, and the commit store is exactly "
          "round-to-nearest-even (64 trials + an 8-step lossless run)")

    # 2. divergence from arbitrary fp32 state stays inside the envelope
    worst = values.get("ulp_max", -1)
    envelope = values.get("ulp_envelope", -1)
    demand(0.0 < worst <= envelope,
           f"divergence {worst:.3g} bf16 ulps against envelope "
           f"{envelope:.3g} (0.5/(1-alpha_max)+1); the per-commit rounding "
           f"must stay inside the geometric bound over "
           f"{values.get('ulp_steps', 0):.0f} commits")
    demand(values.get("ulp_second_half", -1) >= 0
           and values.get("ulp_second_half", 0) <= envelope,
           "the divergence keeps growing in the second half of the run; "
           "the rounding is compounding instead of contracting")
    print(f"divergence over 64 commits: worst {worst:.3g} bf16 ulps, "
          f"envelope {envelope:.3g} at alpha_max "
          f"{values.get('ulp_alpha_max', 0):.4f} "
          f"(first half {values.get('ulp_first_half', 0):.3g}, "
          f"second half {values.get('ulp_second_half', 0):.3g})")

    # 3. the replay fold under bf16 state
    demand(values.get("fold_mismatch", -1) == 0,
           f"the bf16 fold differs from serial decode in "
           f"{values.get('fold_mismatch'):.0f} bytes; the fold must convert "
           f"at the same two points per committed step")
    demand(values.get("fold_prefix_mismatch", -1) == 0,
           f"the checkpoint fold over the accepted tail differs in "
           f"{values.get('fold_prefix_mismatch'):.0f} bytes")
    print("the replay fold lands on the serial-decode state byte for byte, "
          "from zero and from a mid-run checkpoint")

    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nthe bf16-state kernels keep the fp32 kernel's contracts: "
          "bit-identical math, one RNE rounding per commit, byte-exact fold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
