"""Every constant a model declares must be used by that model.

A constant carried into config.h and never referenced is a piece of the
architecture nobody implemented. That is not a style point - it found two real
defects in llms/glm5_2 within an hour of each other:

  GLM52_QUERY_A_DIM      sat unused while the attention projection did one GEMM
                         instead of the two-stage low-rank pair the rank is for
  GLM52_FIRST_ROUTED_LAYER
  GLM52_DENSE_INTERMEDIATE
                         sat unused while every layer was routed, including the
                         three that have no experts

Both were invisible to the compiler, to every other gate, and to a numerical
comparison - which would have said "wrong output" without saying that a whole
projection stage was missing.

Constants that are deliberately unused belong in a config with a comment saying
why, not silently. The exemption list below is that comment, and it is short on
purpose: a long one means the check has been turned off rather than satisfied.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Constants a model declares that nothing in the tree is expected to reference.
# Each needs a reason. "Not implemented yet" is a valid reason and should be
# stated, because then this list is the list of what is missing.
EXEMPT = {
    "K3_ATTNRES_BANK_BYTES": "the per-request cost of the bank, for whoever "
                             "sizes the pool and the stage payload. Ten hidden "
                             "states a token against one, and the layer does "
                             "not allocate - it is handed buffers",
    "K3_MAX_CONTEXT": "a pool-sizing number for the host, not a layer input",
    "K3_ROUTED_SCALE": "1.0 in this checkpoint, so the multiply is omitted "
                       "rather than emitted as a no-op. It stops being safe to "
                       "omit the moment a sibling checkpoint sets it otherwise",
    "K3_MLA_USE_NOPE": "a fact about the model that the code expresses by "
                       "calling no rope kernel; test_rope_pairing.py is what "
                       "enforces it",
    "K3_MLA_OUTPUT_GATE": "same - expressed by LmOutputGateKernel being called "
                          "on the MLA path, not by reading the flag",
    "K3_KDA_FULL_RANK_GATE": "same. K3 replaced Kimi Linear's low-rank output "
                             "gate with a full-rank projection, which is what "
                             "kda_gate is low-rank g_a/g_b, like the decay",
    "K3_ATTNRES_BLOCK_SIZE": "NOT IMPLEMENTED. AttnRes needs 9 hidden states "
                             "per token across the stage boundary; see "
                             "docs/MODEL_SUPPORT.md item 7",
    "K3_MXFP4_GROUP": "the routed experts are MXFP4 at group 32; the format "
                      "trait carries the group and no checkpoint is loaded yet",
    "K3_MTP_LAYERS": "0 in config.json, 1 in the report's Table 1. The "
                     "checkpoint and the paper disagree and nothing here "
                     "drives speculation, so neither value is acted on",
    "K3_LAYER_KIND": "no layer.cuh to dispatch to yet",
    "GLM52_LAYER_KIND": "uniform model - the selector returns one kind, so "
                        "there is nothing for a dispatcher to choose",
    "DSV4_LAYER_KIND": "the three kinds it returns have no entry points; see "
                       "tests/test_layer_kinds.py KNOWN_INCOMPLETE",
    "QWEN38_27B_ATTENTION_PERIOD": "read by QWEN38_27B_LAYER_IS_LINEAR, which the host "
                               "evaluates to pick the entry point - the kernel "
                               "side never sees the period",
    "QWEN38_27B_FULL_PHASE": "same",
    "QWEN38_27B_ATTN_OUTPUT_GATE": "NOT IMPLEMENTED. The checkpoint sets "
                               "attn_output_gate true and Qwen38_27bLayerAttention "
                               "does not apply it, so the full-attention path is "
                               "missing a sigmoid gate on the attention output "
                               "before the output projection. Three layers in "
                               "four do not reach this path; the fourth is wrong "
                               "until a gate kernel exists",
    "GLM52_MTP_DRAFT_TOKENS": "speculation is wired in kernels/speculate.cuh but no model drives it yet",
    "GLM52_MTP_LAYER_INDEX": "same",
    "GLM52_WEIGHT_LAYERS": "used by the host packer, not by kernels",
    "QWEN38_27B_VOCAB": "same",
    "K3_VOCAB": "same",
    "MIMO25_ROPE_HALF": "derived; LmRopePerHeadKernel computes the half internally",
    "DSV4_LAYERS": "the layer loop is the host's; layer.cuh is one layer",
    "DSV4_KV_HEADS": "one KV head is what makes the cache a latent; the geometry "
                     "carries it, not the sequence",
    "DSV4_SHARED_EXPERTS": "derived into DSV4_SHARED_INTERMEDIATE, which is used",
    "DSV4_QUERY_LORA_RANK": "the rank the host puts in LmLowRankWeights",
    "DSV4_KV_QUANT_BLOCK": "the KV cache is stored BF16; quantising it at block "
                           "64 on the nope dimensions is not implemented, and "
                           "would halve the cache read that dominates decode",
    "GLM52_MXFP4_GROUP": "MXFP4 is a supported format with no checkpoint using it",
    "GLM52_FP8_SCALE_BLOCK": "the format trait carries its own group size",
    "GLM52_NVFP4_GROUP": "same",
    "MIMO25_RMS_EPSILON": "passed by the host, not named in unity.cu",
    "QWEN38_27B_MTP_LAYERS": "speculation not driven yet",
    # glm5_2, with the reason each is not referenced by layer.cuh
    "GLM52_LAYERS": "the layer loop is the host's; layer.cuh is one layer",
    "GLM52_ROUTED_LAYERS": "derived, used by the host packer",
    "GLM52_W1_COMPONENTS": "layer.cuh writes the factor of two directly",
    "GLM52_QK_NOPE_DIM": "raw-path width, set into LmLowRankWeights by the host",
    "GLM52_VALUE_DIM": "same",
    "GLM52_QUERY_A_DIM": "same - the rank the host puts in LmLowRankWeights",
    "GLM52_DSA_INDEX_EPSILON": "the index-path norm is not implemented; the "
                               "scoring kernel takes raw index queries",
    "GLM52_ROUTED_SCALE": "the host pre-scales route_weight; see the comment in "
                          "Glm52LayerMoe about why scaling the gates is not the "
                          "same as scaling the result",
}


def main() -> int:
    failures = 0
    for model in sorted(p for p in (ROOT / "inference/llms").iterdir() if p.is_dir()):
        config = model / "config.h"
        if not config.exists():
            continue
        declared = re.findall(r"^#define ([A-Z][A-Z0-9_]*)", config.read_text(encoding="utf-8"), re.M)
        # everything the model and the library could reference it from
        corpus = ""
        for path in list(model.glob("*")) + list((ROOT / "inference/kernels").rglob("*.cuh")) + list((ROOT / "runtime").rglob("*")):
            if path.is_file() and path.name != "config.h":
                corpus += path.read_text(encoding="utf-8", errors="ignore")
        # A constant consumed by a sibling macro in the same header IS used, once
        # something calls that macro. MIMO25_ATTENTION_PERIOD is read only by
        # MIMO25_LAYER_KIND, which bind.cu evaluates; excluding config.h from the
        # corpus made it look dead and would have bought an exemption for a
        # constant that is genuinely load-bearing. Only the bodies count - the
        # left-hand side of a #define is a declaration, not a use.
        joined = config.read_text(encoding="utf-8").replace("\\\n", " ")
        for body in re.findall(r"^#define\s+[A-Z][A-Z0-9_]*(?:\([^)]*\))?\s+(.*)$",
                               joined, re.M):
            corpus += body + "\n"
        unused = [d for d in declared
                  if corpus.count(d) == 0 and d not in EXEMPT and not d.endswith("_H")]
        # A model with no layer sequence has every constant unused, which is one
        # fact and not thirteen. Reporting it as thirteen failures would drown
        # the models that DO have a sequence and skip something in it.
        has_sequence = (model / "layer.cuh").exists()
        if not has_sequence:
            print(f"  --   {model.name}: no layer.cuh; {len(declared)} constants "
                  f"unused because nothing sequences them yet")
            continue
        if unused:
            print(f"  FAIL {model.name}: declared and never used by its sequence")
            for name in unused:
                print(f"         {name}")
            failures += len(unused)
        else:
            exempt_here = [d for d in declared if d in EXEMPT]
            print(f"  ok   {model.name}: {len(declared)} constants, "
                  f"{len(exempt_here)} exempt with a stated reason")
    print(f"\n{'FAIL' if failures else 'PASS'} ({failures} unused)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
