# Migrating glm5_next onto the common GLM modules

glm5_next (glm53flash) was left out of the mod-infra wave that moved glm52
onto the common GLM modules. This note is the recipe for its adoption. Every
step preserves the capability-alignment law: the family copy is deleted in the
same commit that flips the include.

Status on main (2026-09-28): step 6 is done apart from two wiring changes.
Steps 1-5, 7 and 8 are still open. glm5_next still has its private
`source/cuda/{config.h,launch_shape.h,layer.cuh,unity.cu,api.h}`, its
`llm_defines.h` has 52 `SPARK_LLM_*` keys and none of `TILE_K`, `TILE_WARPS`
or `HEAD_TILE`, and its Makefile includes only
`../resident_decode_stage_rules.mk`.

## What exists on main

- `common/common_glm_cuda_tree/` — `spark_glm_cuda_config.h`,
  `spark_glm_cuda_launch_shape.h`, `spark_glm_cuda_layer.cuh`,
  `spark_glm_cuda_unity.cu`, `spark_glm_cuda_api.h`,
  `spark_glm_batch_tuning.h`. Everything is written against `SPARK_LLM_*`
  keys; nothing in the tree mentions glm52.
- `common/common_glm_stage_module/spark_glm_stage_module.h` — the 16
  stage-module functions that are byte-identical between the glm52 and
  glm5_next module.c files, parameterized by `SPARK_GLM_STAGE_*` bindings.
- `common/common_kv_geometry.h` — `SparkGlmKvFillCapacityRequest` plus
  `_Static_assert` invariants over `SPARK_GLM_KV_*` constants.
- `common/glm_resident_stage_wrapper.mk` — the codec table, compile flags,
  include flags, archive paths and the adapter shared-library stanza shared
  by the glm pair, layered under `modules/resident_decode_stage_rules.mk`.
- `model-families/glm52/include/sparkpipe/llm_defines.h` — the glm52
  parameter file, `SPARK_LLM_*` keys congruent with the glm5_next seed and
  the `llm_specifics.h` specimen.
- `model-families/common/include/sparkpipe/spark_tp_mesh_kernels.cuh` +
  `spark_tp_mesh_register.h` (M-0) — the shared TP mesh kernels, seeded from
  glm5_next's own FP32-accumulate combine. glm5_next's `cuda.cu` already
  includes them.
- `include/sparkpipe/family/module/spark_module_combine.h` — the combine
  wrappers and `SPARK_FAMILY(ModuleRegisterCombines)`.

## Steps for glm5_next

1. Parameter file: glm5_next already has
   `model-families/glm5_next/include/sparkpipe/llm_defines.h` (the 52-key
   seed). Extend it with the keys the common tree needs that the seed lacks:
   `SPARK_LLM_TILE_K`, `SPARK_LLM_TILE_WARPS`, `SPARK_LLM_HEAD_TILE`,
   `SPARK_LLM_ATTN_THREADS`, the `SPARK_LLM_KV_*` block, the
   `SPARK_LLM_STAGE_*` ABI values, and `SPARK_LLM_BATCH_MODULE_ID_*`. The
   seed's `SPARK_LLM_TILE_N`/`SPARK_LLM_TILE_STAGES`/`SPARK_LLM_LAYER_THREADS`
   already match the common key names by design.
2. Family headers become shims: `spark_glm5_next_model.h` aliases every
   `SPARK_GLM5_NEXT_MODEL_*` onto `SPARK_LLM_*` (glm52's model.h is the
   worked example). Keep the glm5_next-local KDA/HC/indexer keys in the
   family file; they are family extras, not common keys.
3. kv geometry: `spark_glm5_next_kv_geometry.h` shrinks to the seven
   `SPARK_GLM_KV_*` constants (layer count from `..._DSA_LAYER_COUNT`,
   compressed from `..._MLA_LATENT_DIMENSION`, position from
   `..._MLA_QK_ROPE_HEAD_DIMENSION`, arena from `..._MLA_KV_A_DIMENSION`)
   followed by `#include "common/common_kv_geometry.h"`. Delete the family
   `SparkGlm5NextKvFillCapacityRequest` and the `#ifndef` guards — the
   asserts in the common header replace them.
4. CUDA tree: delete `source/cuda/config.h`, `launch_shape.h`, `layer.cuh`,
   `unity.cu`, `api.h`. In `spark_glm5_next_resident_decode_stage_cuda.cu`
   include `common/common_glm_cuda_tree/spark_glm_cuda_unity.cu` instead.
   The KDA + HC sinkhorn + indexer-kv arms (~1,200 LOC that only exist in
   glm5_next's layer.cuh) move to a family-local
   `spark_glm5_next_cuda_layer_ext.cuh` included right after the common
   layer, keyed off `GLM_LAYER_KIND`/`LM_LAYER_LINEAR`. The per-layer
   graph-capture wrappers are gone, because the whole-step graph engine
   captures the step; glm5_next's extra `SparkGraph*` entry points stay
   family-side until the stage-module step.
   Caution: glm5_next's config asserts `SPARK_LLM_MLA_QK_ROPE_HEAD_DIMENSION
   == 0`; keep that assert in the family shim, not the common tree.
5. Stage module: define the `SPARK_GLM_STAGE_*` bindings against the
   glm5_next types and include
   `common/common_glm_stage_module/spark_glm_stage_module.h` (glm52's
   module.c binding block is the template). Delete the 16 identical
   function bodies. glm5_next keeps its superset functions (graph capture,
   MTP, KDA recurrent, page copy, worker completion).
6. M-0, done except for wiring. glm5_next defines no private launchers:
   `SparkTpLaunchAccumAdd` and `SparkTpLaunchAccumU64Max` come from the
   shared `spark_tp_mesh_kernels.cuh`, and `4b0bcc2` moved its FP32 combine
   wrappers into `spark_module_combine.h`. Two changes remain. The module
   assigns the combine wrappers field by field in
   `SparkGlm5NextModuleInitializeTpCollective` instead of calling
   `SPARK_FAMILY(ModuleRegisterCombines)`. Its `internal.h` also re-declares
   the `SparkTpLaunch*` prototypes that `spark_tp_mesh_register.h` provides.
7. Makefile: set `GLM_FAMILY`, `GLM_EXPERT_CODECS` (no bf16),
   `MODULE_IDENTIFIER_PREFIX` (shape tag `h4096.l45.kda34.e288.k8`) and the
   family sources, then `include ../../common/glm_resident_stage_wrapper.mk`
   followed by `../resident_decode_stage_rules.mk`. The extra validator
   (mtp_parity) and the extra host sources stay in the family Makefile.
8. Receipts: host `make contract` across the codec set, the sm_121a compile
   gate, the module tests in `tests/test_common_glm_modules.py` (they parse
   glm52's llm_defines.h; add the glm5_next path), plus behavior-identity
   tokens for one decode frame before/after per the #977 pattern.

## Ordered by risk

Steps 1-3 are mechanical (header shims, no behavior). Step 4 is the largest
diff — land it alone with the compile gate. Steps 5-7 change module wiring —
land with the behavior-identity receipt. The rest of step 6 is independent
and can land at any time.
