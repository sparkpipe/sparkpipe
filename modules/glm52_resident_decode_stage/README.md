# GLM52 resident decode-stage firmware

This directory is an exact model-specific CUDA firmware link unit. It is not a generic serving backend and it is not a reusable graph interpreter.

The fixed stage program is specialized for SM121 GLM 5.2 BF16 decode with 6144 hidden elements, 64 MLA heads, 512 latent elements, 64 adjacent-pair RoPE elements, 2048 selected context tokens, 64-token KV blocks, a 256-token restricted vocabulary head, and depth-2 MXFP4 MTP draft verification.

One submission executes this stream-ordered sequence:

```text
embedding row into hidden, zero residual
per layer: attention chunk
  fused residual RMSNorm, Q/KV latent projections, q_a norm
  DSA indexer on full-indexer layers when context > 2048 selected tokens
  rope + current KV latent write, resident (sparse or dense) MLA attention
  attention output projection
per layer: MLP chunk - dense below layer 3, routed MoE above it
  router GEMM, sigmoid top-k selection, renormalised mixture,
  routed-expert grouped GEMMs in the package codec, weighted finalize,
  shared expert summed ungated
optional final head on the stage that owns it
external completion
```

The stage's gates are the retained-receipt chain in
`../resident_decode_stage_rules.mk`; the retired `validate_layer0_*` /
`package_layer0_*` targets no longer exist. Every publish-shaped target
requires nvcc for `sm_121a`, a readable stage pack at `STAGE_PACK_PATH`, and
the executable validator wrapper; the validator source digest rides
`SPARK_GLM52_CUDA_VALIDATOR_SHA256` inside `RUNTIME_CONFIGURATION`, so the
configuration hash names the exact validator text that approved an artifact.

Host-side checks (no GPU needed):

```sh
make -C modules/glm52_resident_decode_stage contract \
    EXPERT_CODEC=<int6|int7|int8|fp8|nvfp4|mxfp4> \
    MODEL_REVISION=$(git rev-parse HEAD) CONTRACT_SHA256=<package sha>
python3 tests/test_glm52_cuda_validator_tier2_oracle.py   # tier-2 oracle math, all six codecs
```

Hardware validation of one codec configuration:

```sh
GLM52_MODEL_REVISION=$(git rev-parse HEAD) \
GLM52_CONTRACT_SHA256=<sha from tools/glm52_model_contract.py --print-build-identity> \
make -C modules/glm52_resident_decode_stage validate \
    EXPERT_CODEC=fp8 MODEL_REVISION=$GLM52_MODEL_REVISION CONTRACT_SHA256=$GLM52_CONTRACT_SHA256
```

That compiles the module archive, runs the wrapper
(`validation/validate_glm52_resident_decode_stage_cuda.sh`), which rebuilds
core/runtime libraries, compiles the validator against the same defines the
archive used, and executes three tiers against an fp32 CPU oracle: one dense
layer's forward plus a bit-exact determinism re-walk (tier 1), the first
routed-expert layer including router/top-k determinism, one expert forward
element-wise, and a bit-exact leg re-run (tier 2a), and the DSA indexer at
context 2065 > 2048 selected tokens with forced-selection-set verification
(tier 2b). `publish` additionally retains the recipe, configuration hash and
validator digest under the module library;
`variants` / `publish_variants` repeat the archive (and validation) per batch
bucket.

The serving adapter builds separately:

```sh
make -C modules/glm52_resident_decode_stage adapter EXPERT_CODEC=<c> ...
```

The node context binds resident weight pointers, paged KV cache, streams, workspaces, RoPE tables, token maps, and output buffers once when the driver instance is created. Per-submission inputs are only dynamic decode facts such as active sequence count, requested token count, sequence identity, deadline, priority, and residency token. The firmware admission function chooses the opaque pipeline slot; SparkPipe does not assign or interpret CUDA stream/KV ownership.

The module also publishes direct admission and snapshot symbols. They expose only neutral scheduling data: accepted/rejected, dispatch slot, dispatch generation/cookies, private queue pressure, resident token capacity, active submissions, stale-admission count, and zero memcpy/host-staging counters.

Sparse-token selection runs in-stage: on every full-indexer layer whose context exceeds `SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT` the indexer projects, norms, ropes, caches and scores index keys, then the shared radix top-k fixes the 2048 attended positions; shorter contexts and non-leader layers skip it (group-shared index state rides the wave's index ordinals, and pipeline-parallel stages ship selections through the sideband).

Normal publication validates a new archive exactly once per codec configuration:

```sh
make -C modules/glm52_resident_decode_stage publish \
    EXPERT_CODEC=<c> MODEL_REVISION=<rev> CONTRACT_SHA256=<sha> \
    CUDA_ARCH=sm_121a
```

The source is correctness-first until hardware profiling says which fused pieces should be replaced by tensor-core or persistent-kernel implementations. It must not be published unless the hardware validator passes the numerical checks of all three tiers.

## DFlash2 speculative decoding (DSpark drafter)

The stage carries the second DFlash2 adoption (after qwen36/qwen38): the
drafter engine is the `glm52_dspark_draft_backend` archive, ar'd into this
module's link unit (`MODULE_EXTRA_ARCHIVE_OBJECTS` in
`../resident_decode_stage_rules.mk`), so the published link-unit is
self-contained. The round shape is the proven one-frame verify:

- every aux capture layer {7,22,38,54,69} copies the wave's post-layer
  hidden into the drafter's device tap arena (one ring row per walked token,
  `(resident slot, position mod block)`-addressed);
- a frame with `DSPARK_DRAFT_AFTER` ends at the head: the module computes
  each lane's accept depth from its own emissions (verify frames), stages
  each lane's anchor token into the drafter context, runs the block forward,
  and hands the next block's ids to the adapter through the frame's draft
  view;
- the following submission walks `[anchor emission, d_1..d_6]` as a
  `SPECULATIVE_VERIFY` prefill-kind frame; the adapter credits accepted+1
  tokens and stamps the accept depth back on the next frame so EVERY rank
  derives identical position books without a cross-rank reduction.

Gates: `build/test_glm52_spec_verify_contract` pins the accept rule against
the neutral policy's `ResolveVerifierTokens`, the stamp reconciliation space,
and the shared tap geometry. Speculation arms only on single-rank builds
(`SPARK_GLM52_SERVING_TP_DEGREE==1`, `SPARK_GLM52_SERVING_STAGE_COUNT==1`,
module geometry stage_count==1/tp_degree==1) — a fanout deployment would
need a draft transport that does not exist, and is refused loudly instead.
Drafter numerics gate on GB10 via the backend's epoch-3 validator before any
draft-path number is trusted.
