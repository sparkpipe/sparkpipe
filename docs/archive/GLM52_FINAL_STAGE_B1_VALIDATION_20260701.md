# GLM52 Final Stage B1 Validation - 2026-07-01

This records the first successful C/CUDA final-from-hidden validation on
`spark2` after adding the explicit final-stage mode.

It is not a full prompt-to-token end-to-end run. It proves this narrower segment:

```text
layer 74 hidden BF16
    -> routed layers 75,76,77
    -> final RMSNorm
    -> restricted logits
    -> restricted argmax
    -> MTP draft/verify
    -> final token evidence
```

The runtime path is C/CUDA. Python was used only by the setup-time pack reuse
check:

```text
glm52_b12x_resident_pack.py --reuse-valid --require-reuse --verify-reused-sha256
```

No `.spb12x` pack was rebuilt during this run.

## Command

Run from:

```text
spark2:/tmp/sparkpipe_glm52_live_main_20260630
```

Commit:

```text
adbeebd Prepare GLM52 validation output directories
```

Command:

```sh
make glm52_resident_decode_stage_firmware_package \
  NVCC=/usr/local/cuda/bin/nvcc \
  GLM52_MODEL_DIR=/mnt/mac/16tb0/models/hf/nvidia/GLM-5.2-NVFP4 \
  MAX_STAGE_MICROSECONDS=1000000 \
  GLM52_VALIDATION_MODE=routed_from_hidden_final \
  GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT=1 \
  GLM52_VALIDATION_FIRST_ROUTED_LAYER_INDEX=75 \
  GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT=3 \
  GLM52_PIPELINE_INPUT_HIDDEN_BF16=/tmp/glm52_pipeline_sim/layer0074_direct.bf16 \
  GLM52_PIPELINE_OUTPUT_HIDDEN_BF16=build/glm52_pipeline_validation/final_75_77_b1_hidden.bf16 \
  B12X_MOE_PACK_OUTPUT_DIR=build/glm52_b12x_resident_moe_0075_0077_v3 \
  B12X_MOE_PACK_LAYERS=75,76,77 \
  B12X_MOE_PACK_REQUIRE_REUSE=1 \
  B12X_MOE_PACK_VERIFY_REUSED_SHA256=1 \
  GLM52_ENABLE_CUDA_GRAPH_REPLAY=0
```

## Raw Module Validation

The static-archive validator passed:

```text
routed_pipeline_from_hidden_final=1
final_stage=1
first_routed_layer=75
routed_chain_layers=3
total_submissions=3
total_us=10849.824
maximum_us=3736.960
restricted_token=1037
mtp_draft=1011
mtp_reject=1003
layer3_selected_expert=221
layer3_bound_experts=256
real_lm_head=1
real_lm_head_max_logit_error=0.00000095
```

## Packaged Driver Validation

The packaged model driver validation also passed:

```text
routed_pipeline_from_hidden_final=1
final_stage=1
first_routed_layer=75
routed_chain_layers=3
total_submissions=3
total_us=13849.056
maximum_us=6319.488
restricted_token=1037
mtp_draft=1011
mtp_reject=1003
layer3_selected_expert=221
layer3_bound_experts=256
real_lm_head=1
real_lm_head_max_logit_error=0.00000083
```

Output artifacts:

```text
final hidden:
  /tmp/sparkpipe_glm52_live_main_20260630/modules/glm52_resident_decode_stage/build/glm52_pipeline_validation/final_75_77_b1_hidden.bf16
  sha256=89f17647bef72072c0e66c158221a79fb9a4bb68a00314b0bb477a39785d661a

package manifest:
  /tmp/sparkpipe_glm52_live_main_20260630/build/packages/glm52_resident_decode_stage/model_package.json
  sha256=2ef5989741b02396f3d37fdae43f8d4b0f1210eeb9d2a36ed5f619ba43c2c5fb
```

## What This Fixed

The first final-from-hidden attempt failed correctly because per-routed-layer
KV caches were zeroed. The validator now seeds each bound routed layer cache
with the deterministic previous-token reference window before the layer submit.

That seed represents the resident KV state the final stage must own for previous
tokens. It is validation setup state, not a serve-time Python or file fallback.

The second attempt failed because the package Makefile did not create the
validation output directory. The Makefile now prepares
`GLM52_PIPELINE_OUTPUT_HIDDEN_BF16` parent directories before running the
validator.

## Remaining Before Full End-To-End

Still required for a true prompt-to-token PASS:

```text
1. Produce the layer-10 hidden from token/dense prefix through C/CUDA.
2. Chain all routed hidden stages 11-74 using C/CUDA stage execution.
3. Feed layer-74 hidden into routed_from_hidden_final.
4. Replace file/manual hidden handoff with the required persistent hidden
   transport module for production.
5. Repeat B8/B16/B32/B64 runs and record slowest-stage throughput.
```

Do not call this full GLM52 inference yet. It is the final-stage token gate
passing from a known layer-74 hidden vector.
