#!/usr/bin/env python3
"""K3's driver-side performance contracts, gated as source.

Two contracts live in inference/llms/kimi_k3 and nowhere else, and both are
the kind that compiles while it rots:

  1. THE BF16 KDA STATE OPTION FAILS CLOSED. The slot is 6 MiB of fp32 per
     sequence per layer, ~40% of a B64 step's bytes (the roadmap's K3 state
     correction), so the half-width option is the biggest batch lever in the
     model - and LmDeltaRuleKernel still addresses the pool as float. A flag
     that launches against a half-width pool does not crash; it mis-strides
     every head and sequence and decodes fluently. So the consumer flag must
     exist, the bind must propagate it, the pool arithmetic must be exactly
     half, and every launch path must refuse it until the kernel grows the
     bf16-store variant the flag's comment specifies. Default off, and
     nothing in the tree may set it.

  2. THE LAYER PATH STAYS GRAPH-CAPTURABLE. Roadmap D10/D1 put CUDA graphs
     first on the attack list (~3,300 launches per K3 token). Capture breaks
     on host-device traffic between launches, so the layer, slice, engine
     and bind sources must never name a synchronising or copying CUDA call
     outside a comment.

The gather/indirect-A contract (roadmap D9) is LANDED: the w1 expert GEMM
reads A rows through route_source_token, so the gather launch, its buffer and
its recipe-gate check are gone together and the gate now holds the deletion.

  3. THE PACK V2 BIND IS THE ONLY BIND. The pack emits fused
     kda_qkv_beta_weight and kda_decay_gate_down_weight per KDA layer and
     interleaved expert weight+scale streams; the V1 per-projection tensors
     do not exist. So the stale fields must be gone from both structs, the
     bind must propagate the two fused tensors and the interleave flag, the
     KDA projection block must be exactly two wide GEMMs on the normed input
     (six became two: the launch count drops by four per KDA layer) plus the
     one section split, and the MoE must refuse the interleaved stream until
     the grouped GEMM learns the 17-row cell - LmScaleTensor cannot address
     scales co-tiled with payload, so no scale plane call may survive.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
K3 = ROOT / "inference" / "llms" / "kimi_k3"


def defines(path):
    text = path.read_text().replace("\\\n", " ")
    values = {}
    for name, value in re.findall(r"#define (K3_\w+)[ \t]+([^\n]+)", text):
        expression = re.sub(r"\(u?int\d+_t\)", "", value.strip())
        expression = re.sub(r"(?<=\d)ull|(?<=\d)u", "", expression)
        expression = re.sub(r"K3_\w+",
                            lambda m: str(values.get(m.group(0), 0)),
                            expression)
        try:
            values[name] = int(eval(expression))
        except Exception:
            pass
    return values


def function_body(text, name):
    """One top-level function body, braces balanced."""
    match = re.search(r"static int32_t " + name + r"\(", text)
    if match is None:
        return ""
    index = text.find("{", match.end())
    depth, cursor = 1, index + 1
    while cursor < len(text) and depth:
        if text[cursor] == "{":
            depth += 1
        elif text[cursor] == "}":
            depth -= 1
        cursor += 1
    return text[index:cursor]


def main():
    config = defines(K3 / "config.h")
    layer = (K3 / "layer.cuh").read_text()
    slice_ = (K3 / "slice.cuh").read_text()
    failures = 0

    # -- the slot arithmetic: fp32 slot, and the option is exactly half -------
    heads = config.get("K3_KDA_HEADS", 0)
    slot = config.get("K3_KDA_STATE_SLOT_BYTES", 0)
    slot_bf16 = config.get("K3_KDA_STATE_SLOT_BYTES_BF16", 0)
    expect = heads * config.get("K3_KDA_KEY_DIM", 0) \
        * config.get("K3_KDA_VALUE_DIM", 0) * 4
    if slot != expect or slot_bf16 * 2 != slot or slot_bf16 == 0:
        print(f"  FAIL state slot {slot}, bf16 {slot_bf16}, expected "
              f"{expect} and exactly half")
        failures += 1

    # -- the flag exists on both structs and the bind propagates it -----------
    if "uint32_t kda_state_bf16;" not in layer:
        print("  FAIL K3LayerBuffers lost the bf16-state flag")
        failures += 1
    if "uint32_t kda_state_bf16;" not in slice_:
        print("  FAIL K3SliceState lost the bf16-state flag")
        failures += 1
    if "buffers->kda_state_bf16 = state->kda_state_bf16;" not in slice_:
        print("  FAIL the bind no longer propagates the state dtype flag")
        failures += 1
    if "K3_KDA_STATE_SLOT_BYTES_BF16" not in slice_:
        print("  FAIL the bind does not stride the pool by the flag; a pool "
              "bound at one width and launched at another aliases sequences")
        failures += 1

    # -- every launch path REFUSES the flag while the kernel is fp32-only -----
    for name, text, where in (
            ("layer.cuh", layer, "K3LayerKda"),
            ("slice.cuh", slice_, "K3FoldAccepted")):
        if not re.search(
                r"kda_state_bf16 != 0u \)\s*\n\s*return\(LM_LAUNCH_ERR_SHAPE\)",
                text):
            print(f"  FAIL {name}: {where} does not fail closed on the "
                  f"bf16-state flag; the delta kernel addresses the pool as "
                  f"float and a launched flag mis-strides every head")
            failures += 1

    # -- default off: the only assignment is the bind's propagation -----------
    for name in ("layer.cuh", "slice.cuh", "bind.cu", "unity.cu",
                 "engine.h", "dspark.h", "pipeline_sideband.h"):
        text = re.sub(r"//[^\n]*", "", (K3 / name).read_text())
        for match in re.finditer(r"kda_state_bf16\s*=\s*([^;\n]+);", text):
            value = match.group(1).strip()
            if value not in ("state->kda_state_bf16",):
                print(f"  FAIL {name} sets kda_state_bf16 to {value!r}; the "
                      f"option is admission-time and off by default, and no "
                      f"driver source may turn it on while launches refuse it")
                failures += 1

    # -- the layer path stays capturable: no host-device traffic --------------
    for name in ("layer.cuh", "slice.cuh", "bind.cu", "unity.cu", "engine.h"):
        text = re.sub(r"//[^\n]*", "", (K3 / name).read_text())
        for call in ("cudaMemcpy", "cudaMalloc", "cudaFree",
                     "cudaStreamSynchronize", "cudaDeviceSynchronize",
                     "cudaMemcpyAsync"):
            if call in text:
                print(f"  FAIL {name} names {call}; host-device traffic "
                      f"between launches is what breaks CUDA graph capture "
                      f"(roadmap D10)")
                failures += 1

    # -- the indirect-A contract, LANDED: the gather is gone, the map feeds ---
    #    the w1 GEMM directly -------------------------------------------------
    if re.search(r"LmGatherRowsKernel|route_gather_bf16",
                 re.sub(r"//[^\n]*", "", layer)):
        print("  FAIL the route gather survived; the grouped GEMM stages A "
              "rows through route_source_token (route.cuh's contract), so the "
              "packed copy is a double-touch the kernel made dead (D9)")
        failures += 1
    moe = function_body(re.sub(r"//[^\n]*", "", layer), "K3LayerLatentMoe")
    if "gemm.source_row_map = b->route_source_token;" not in moe:
        print("  FAIL the w1 launch does not read A rows through "
              "route_source_token; the map the route build writes is exactly "
              "the indirect-A consumer contract")
        failures += 1
    if "LmGemmWeightOnlyIndirectLaunch<" not in moe:
        print("  FAIL the w1 expert GEMM is not the indirect launch; without "
              "INDIRECT_A the source_row_map word is refused by the launcher")
        failures += 1

    # -- pack V2: the V1 projection fields are gone, the fused ones bind ------
    for stale in ("kda_q_weight", "kda_q_scale", "kda_k_weight", "kda_k_scale",
                  "kda_v_weight", "kda_v_scale", "kda_beta_weight",
                  "kda_gate_down_weight",
                  "expert_w1_scale", "expert_w2_scale"):
        for name, text in (("layer.cuh", layer), ("slice.cuh", slice_)):
            if re.search(r"\b" + stale + r"\b", text):
                print(f"  FAIL {name} still names {stale}; pack V2 does not "
                      f"emit it - the fused tensors are the only bind")
                failures += 1
    # Released checkpoint (docs/K3_GATE_RECONCILIATION.md): q|k|v|beta is
    # the fused wide tensor; decay_down stays a standalone 128-wide
    # bottleneck and the gate is the checkpoint's full-rank g_proj - the
    # low-rank decay|gate fusion does not exist in this checkpoint.
    for field in ("kda_qkv_beta_weight", "kda_decay_down_weight",
                  "kda_gate_weight"):
        if f"const void *{field};" not in layer:
            print(f"  FAIL K3LayerBuffers lost {field}")
            failures += 1
        if f"buffers->{field} = weights->{field};" not in slice_:
            print(f"  FAIL the bind no longer propagates {field}; a layer "
                  f"would project through a stale pointer")
            failures += 1
    if "buffers->expert_interleave = weights->expert_interleave;" not in slice_:
        print("  FAIL the bind no longer propagates the interleave flag; an "
              "interleaved pack would launch instead of refusing")
        failures += 1

    # -- the KDA projection block: two wide GEMMs plus one section split ------
    kda = function_body(re.sub(r"//[^\n]*", "", layer), "K3LayerKda")
    wide = len(re.findall(r"K3Project<\w+>\(b,b->normed_bf16", kda))
    if wide != 3:
        print(f"  FAIL {wide} projection GEMMs read normed_bf16 in K3LayerKda; "
              f"the released checkpoint keeps qkv_beta fused with "
              f"decay_down standalone and the full-rank gate (docs/"
              f"K3_GATE_RECONCILIATION.md), so exactly three wide GEMMs run")
        failures += 1
    if kda.count("LM_LAUNCH((K3SplitFusedProjectionsKernel<K3_LAYER_THREADS>)") != 1:
        print("  FAIL the fused projections lost their section split; every "
              "consumer reads dense rows, so the wide GEMM needs the split")
        failures += 1
    # Released checkpoint (docs/K3_GATE_RECONCILIATION.md): q|k|v|beta is the
    # only fused wide tensor - decay_down projects standalone into latent and
    # the gate is full rank, so no decay|gate scratch exists.
    if "uint16_t *fused_qkvb_bf16;" not in layer:
        print("  FAIL K3LayerBuffers lost the fused_qkvb_bf16 wide scratch")
        failures += 1
    for gone_scratch in ("fused_decay_gate_bf16", "gate_latent_bf16"):
        if f"uint16_t *{gone_scratch};" in layer or gone_scratch in slice_:
            print(f"  FAIL the dead {gone_scratch} decay|gate scratch survives")
            failures += 1

    # -- the interleaved expert stream fails closed ---------------------------
    if "expert_interleave != 0u" not in moe:
        print("  FAIL K3LayerLatentMoe does not refuse the interleave flag; "
              "the grouped GEMM cannot read the 17-row cell, so launching "
              "would read scale bytes as payload")
        failures += 1
    if "LmScaleTensorBlockUe8m0" in re.sub(r"//[^\n]*", "", layer):
        print("  FAIL a far-plane scale descriptor survives; pack V2 co-tiles "
              "the scales with the payload and no LmScaleTensor can address "
              "them - the descriptor is None until the kernels wave lands")
        failures += 1

    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nthe bf16 state option is wired, refused, and off; the layer "
          "path captures; the gather is deleted and the map feeds the GEMM")
    return 0


if __name__ == "__main__":
    sys.exit(main())
