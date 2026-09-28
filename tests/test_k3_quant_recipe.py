#!/usr/bin/env python3
"""K3's quantisation must follow the checkpoint's recipe, not a global choice.

A second external audit found the expert path quantising its ACTIVATIONS to
MXFP4. The checkpoint sets input_activations null: it quantises weights and
says nothing about activations, so an inference stack runs BF16 activations
against streamed MXFP4 weights. The expert GEMM now keeps A at 16 stored bits
with no scale against B at 4 with an E8M0 scale every 32; reintroducing
activation quantisation fails this gate. docs/archive/K3_WEIGHT_ONLY_MXFP4.md
has the full requirement.

config.json's quantization_config quantises weights to 4 bits at group 32 and
carries an ignore list: self_attn, shared_experts, the dense mlp projections,
lm_head and the vision tower. The report's deployment section says the
quantisation-aware training ran from SFT onward - so the routed experts were
trained INTO that grid and nothing else was.

That makes the ignore list a correctness constraint rather than a preference. A
tensor that never saw QAT has no protection in a 4-bit grid, which is the same
reason a derived factorisation of those experts could not be stored at MXFP4
either: the grid is not the protection, the training into the grid is.

This gate checks that only the two routed-expert GEMMs take the quantised
Format and every other projection in the layer names a high-precision one.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAYER = ROOT / "inference" / "llms" / "kimi_k3" / "layer.cuh"
BIND = ROOT / "inference" / "llms" / "kimi_k3" / "bind.cu"
# the only tensors the checkpoint quantises
QUANTISED = ("expert_w1", "expert_w2")


def main():
    text = re.sub(r"//[^\n]*", "", LAYER.read_text())
    failures = 0
    generic = []
    for match in re.finditer(r"K3Project<(\w+)>\(b,\s*[\w>\-\.]+,\s*b->(\w+)", text):
        fmt, tensor = match.group(1), match.group(2)
        if fmt == "Format" and not any(q in tensor for q in QUANTISED):
            generic.append(tensor)
            failures += 1
    for tensor in generic:
        print(f"  FAIL {tensor} takes the quantised Format; the checkpoint's "
              f"ignore list excludes it from QAT")
    # ACTIVATIONS ARE NOT QUANTISED, AND THIS GATE DID NOT CHECK.
    #
    # The checkpoint sets input_activations null. It quantises weights and says
    # nothing about activations, so an inference stack runs BF16 activations
    # against streamed MXFP4 weights. This file passed while the expert path
    # quantised its activations to MXFP4, because it only ever asked which
    # WEIGHT projections took the quantised Format.
    #
    # A second audit found it. The check is one line and its absence was the
    # whole of the defect.
    for match in re.finditer(r"K3Quantise<(\w+)>", text):
        if match.group(1) == "Format":
            print("  FAIL an activation is quantised with the weight Format; "
                  "the checkpoint sets input_activations null")
            failures += 1

    helper = re.search(r"static int32_t K3Project\b.*?\n\}", text, re.S)
    outside = text.replace(helper.group(0), "") if helper else text
    launches = re.findall(r"\b(Lm\w*Launch\w*)<\s*Format\b", outside)
    quantising = sorted({name for name in launches
                         if not name.startswith("LmGemmWeightOnly")})
    if quantising:
        print(f"  FAIL {quantising} take Format; only a weight-only launch "
              f"keeps the activations BF16, anything else quantises them")
        failures += 1
    w1 = [name for name in launches if name.startswith("LmGemmWeightOnlyIndirect")]
    w2 = [name for name in launches if name.startswith("LmGemmWeightOnly")
          and not name.startswith("LmGemmWeightOnlyIndirect")]
    expert_gemms = len(w1) + len(w2)
    for name in ("LmGemmWeightOnlyIndirectInterleavedLaunch",
                 "LmGemmWeightOnlyInterleavedLaunch"):
        if name not in launches:
            print(f"  FAIL no {name} takes Format; the pack V2 interleaved "
                  f"stream is the production expert layout")
            failures += 1
    if len(w1) != len(w2):
        print(f"  FAIL {len(w1)} indirect w1 launches against {len(w2)} packed "
              f"w2 launches; every expert layout needs both halves, w1 "
              f"reading A through the route map")
        failures += 1
    if "K3Quantise" in text:
        print("  FAIL an activation quantiser still exists in the layer; the "
              "recipe has no place for one")
        failures += 1
    # and the route expansion gather must be GONE: the first expert GEMM
    # stages its A rows through route_source_token directly, so the packed
    # copy is a double-touch the indirect launch retired. The deletion
    # comment narrates the old dataflow, so this reads the code only.
    code = "\n".join(line.split("//")[0] for line in text.split("\n"))
    if "LmGatherRowsKernel" in code:
        print("  FAIL the route expansion gather survived; the indirect-A "
              "w1 launch reads the un-gathered latent through the route map")
        failures += 1
    # and Format must be the checkpoint's, not something else 4-bit
    bind = BIND.read_text()
    if "K3LaunchSlice<LmMxfp4" not in bind:
        print("  FAIL the slice does not instantiate LmMxfp4; the checkpoint is "
              "MXFP4 group 32 and a different 4-bit grid is a different model")
        failures += 1
    print(f"projections checked {len(re.findall(r'K3Project<', text))}, "
          f"expert GEMMs {expert_gemms}")
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nonly the routed experts are quantised, and at the grid they were "
          "trained in")
    return 0


if __name__ == "__main__":
    sys.exit(main())
