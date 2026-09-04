#!/usr/bin/env python3
"""Each model must rotate with the convention its checkpoint was trained under.

Both conventions are called "rope". Half-split pairs element i with
i + rope_dim/2; interleaved pairs 2i with 2i+1, the view_as_complex layout. They
are different rotations of the same vector, so serving a checkpoint under the
wrong one produces text that is fluent and positionally wrong - no crash, no
NaN, nothing a shape check would catch.

kernels/attn.cuh carried a comment warning about exactly this and a single
hardcoded convention, which was correct until a model needed the other one.
DeepSeek V4 encodes interleaved pairs to match its released checkpoint.

This gate pins the instantiation to the documented convention per model. The
expectations below cite where each comes from, because a table of expectations
with no provenance is just a second place to be wrong.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# model -> (pairing, source of the claim)
EXPECTED = {
    "dsv4_resident_decode_stage": ("LM_ROPE_INTERLEAVED",
                    "DeepSeek-V4 paper and reference: RoPE encoded as interleaved "
                    "pairs, view_as_complex style, for both the main attention and "
                    "the compressor sub-module"),
    "glm5_2": ("LM_ROPE_INTERLEAVED",
                "zai-org/GLM-5.2 config: rope_interleave and "
                "indexer_rope_interleave are true"),
    "qwen_3_6": ("LM_ROPE_HALF_SPLIT", "Qwen3.6 partial rotary over the head suffix"),
    "mimo_2_5": ("LM_ROPE_HALF_SPLIT", "MiMo 2.5"),
    # K3 applies NO rope. modeling_kimi_linear.py sets rotary_emb = None, asserts
    # use_nope, and splits q_rot/k_rot out only to concatenate them back
    # unrotated. Position is carried by KDA's decay instead, which is how it
    # reaches 1M tokens without RoPE rescaling. A pairing expectation here
    # would assert a convention for a rotation that never happens.
    "kimi_k3": (None, "NoPE - no rotation on any layer"),
}

ROPE_CALL = re.compile(r"LmRope(?:PerHead|Yarn)?Kernel\s*<([^>]*)>")


def pairings_used(model):
    """Every pairing the model's sources instantiate, by reading the template
    arguments. A default argument means half-split, which is what the kernel
    declares."""
    found = set()
    if model == "dsv4_resident_decode_stage":
        source = ROOT / "modules" / model / "source" / "spark_dsv4_resident_decode_stage_cuda.cu"
        text = source.read_text()
        if "(head_dim - rope_dim) + 2u * pair" in text:
            found.add("LM_ROPE_INTERLEAVED")
        return found
    for name in ("unity.cu", "layer.cuh"):
        path = ROOT / "inference" / "llms" / model / name
        if not path.exists():
            continue
        for arguments in ROPE_CALL.findall(path.read_text()):
            if "LM_ROPE_INTERLEAVED" in arguments:
                found.add("LM_ROPE_INTERLEAVED")
            elif "LM_ROPE_HALF_SPLIT" in arguments:
                found.add("LM_ROPE_HALF_SPLIT")
            else:
                found.add("LM_ROPE_HALF_SPLIT")
    return found


def kernel_offers_both():
    """The kernel must actually implement two distinct pairings. If someone
    collapses them back to one, every expectation below passes vacuously."""
    text = (ROOT / "inference" / "kernels" / "attn.cuh").read_text()
    if "LM_ROPE_INTERLEAVED" not in text or "LM_ROPE_HALF_SPLIT" not in text:
        return "attn.cuh does not declare both pairings"
    body = re.search(r"void LmRopeRotate\(.*?\n\}", text, re.S)
    if not body:
        return "LmRopeRotate is gone; the pairing is hardcoded again"
    if "index * 2u" not in body.group(0) or "half + index" not in body.group(0):
        return "LmRopeRotate no longer computes two distinct offsets"
    return None


def main():
    problem = kernel_offers_both()
    if problem:
        print(f"FAIL {problem}")
        return 1
    failures = 0
    for model in sorted(EXPECTED):
        want, source = EXPECTED[model]
        used = pairings_used(model)
        if want is None:
            if used:
                failures += 1
                print(f"  FAIL {model}: rotates with {sorted(used)}, but this model is NoPE")
                print(f"         {source}")
            else:
                print(f"  ok   {model}: no rope, as the reference has it")
            continue
        if not used:
            print(f"  --   {model}: no rope call sites yet")
            continue
        if used == {want}:
            print(f"  ok   {model}: {want.replace('LM_ROPE_', '').lower()}")
            continue
        failures += 1
        print(f"  FAIL {model}: instantiates {sorted(used)}, checkpoint wants {want}")
        print(f"         {source}")
    print()
    if failures:
        print(f"FAIL ({failures} model(s) rotating against their checkpoint)")
        return 1
    print("every model rotates with the convention its checkpoint uses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
