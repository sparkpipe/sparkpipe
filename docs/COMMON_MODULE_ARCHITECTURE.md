# Common Module Architecture — parameterized shared modules for every driver

Status: mgr2 architecture deliverable, 2026-09-13. Inputs: SEAM-1 + SEAM-2 fleet seam
surveys (branches `lane/wave-seam1-survey`, `lane/wave-seam2-survey` — full per-file
inventories, divergence measurements, and `seams.json` machine files), the DRY-1
packbuilder study (#976), the constant audit generative set (#979, mechanisms G1-G6),
and the operator's parameterization directive.

## 1. The measured problem

Across the 13 driver families the surveys measured **~26,000 wasted LOC of near-copy
code** (SEAM-1: ~15-17k over 53,781 driver-side LOC in six families; SEAM-2: ~10.5k in
seven). The decisive finding is not the volume — it is this:

**The common seams for the worst clusters already exist. The duplication is seam
bypass, not missing infrastructure.**

- `spark_serving_adapter_template.h` exists; k3/dsv4/gemma4/muse adopt it (293-656 LOC
  each); ling and laguna hand-roll 1,200+ LOC adapters that are 89-95% role-identical
  to each other and to the template.
- `spark_pack_load_common.h` exists; only gemma4 adopted it; five families hand-roll.
- glm5_next re-implements a TP-collective JSON parser that glm52 — in the same tree —
  takes from the shared template.
- Two pack-acquisition paths coexist for identical formats: weightd lazy-pack attach
  (qwen38max, glm pair) vs direct pack read (qwen4_flash, qwen38-27b). The lazy path is
  the law; the second path is a bypass.
- k3 is an entire earlier generation: it builds on `inference/llms/kimi_k3/` and
  bypasses 5 of 8 shared seams. Its kv_geometry filler omits the `abi_version` field
  ling sets — live contract drift, today.

Consequence: every fix lands N times or not at all; every driver debug session re-learns
the same modules; a new driver starts by copying 5,000 LOC it does not understand.

## 2. Principles (the operator's mechanism, made precise)

1. **Common parameterized modules in shared code.** A module owns one coherent job
   (KDA state algebra, rope tables, stagepack validation, serving TP config, ...). Its
   code lives once, in `common/` (new top-level) or beside its existing shared seam.
2. **`llm_defines.h` — one parameter file per driver.** Each driver's include path
   carries its own `llm_defines.h` holding EVERY value the common modules need: model
   geometry, kernel tile choices, state-machine sizes, batch-tuning ladder, family
   table for its stagepack kinds. One place to verify a driver's entire configuration.
   The per-family headers (`model.h`, `firmware.h`, `config.h`, `batch_tuning.h`)
   become thin shims that include it, so nothing else changes at adoption time.
   Measured union today: ~200 keys (SEAM-1), with SEAM-2's conditional blocks
   (MLA/KDA/GQA/rope) ~65 generic keys cover the common surface; the rest are family
   extras that stay in the family file.
3. **`llm_specifics.h` — the reference with intentionally uncompilable values.** A
   checked-in specimen whose sentinel identifiers ARE the documentation:

```c
#define SPARK_HY4_HIDDEN_SIZE            SET_ME_HIDDEN_SIZE
#define SPARK_HY4_O_GROUPS               SET_ME_O_GROUPS
#define SPARK_HY4_KDA_STATE_SLOTS        SET_ME_KDA_STATE_SLOTS
```

   A new driver copies it, fills every `SET_ME_*`, and **when the compiler errors are
   gone, the configuration is done** — the error text is the checklist, no code
   comments required, no auditing of modules the driver does not use.
4. **No universality requirement.** A common module may serve one driver if its math is
   generic — it is then simply infrastructure a future derived LLM inherits.
5. **Every common module ships a fully-validating test.** The validation-harness
   cluster (the surveys' single largest waste class after kernels) becomes ONE harness
   with per-family stage tables — and it doubles as the accuracy-law instrument:
   module tests prove the module, the T1 decode receipt proves the driver.
6. **Adoption deletes.** Capability alignment: a module is adopted only when its copies
   are deleted in the same PR. `#ifndef` guards and restated constants are findings,
   not style — they are how drifted contracts happen (k3's missing `abi_version`).
7. **Identity-proofed conversion.** Every migration PR proves byte-identity or
   behavior-identity against the old copy (the #977 pattern: refactored packer rebuilds
   the pack byte-identical). The compile gate plus the module's own test plus the
   identity receipt = green.

## 3. The module catalog

Merged and deduplicated from both surveys. "Params" = llm_defines keys the module
consumes. "Seed" = the family whose copy becomes the common body. "Proof" = the
identity gate for the first adoption.

| # | Common module | Owns | Params | Seed | First adopters (delete their copies) |
|---|---|---|---|---|---|
| 1 | `common_gdn_stage_kernels.cu` | the 22-kernel qwen decode suite (AttnDecode, AttnPrepare, ChunkStep, MoE gather/scatter, router) | 45 | qwen38_max cuda.cu | qwen38_max, qwen4_flash, qwen38_27b (22/22 kernels identical after name normalization) |
| 2 | `common_glm_cuda_tree` (config+unity+layer) | GLM kernel tree + tile config (tiles already identical 128/64/2/8) | 60 | glm52 | glm52, glm5_next (87.4% rename-only) |
| 3 | `common_serving_tp_config` | TP rail hosts, session ports, algorithms, member validation — the template's missing loader | 34 (22 TP-collective) | glm52 via `SparkServingAdapterTemplateLoadTpCollective` | ling, laguna (89-95% role-identical), glm5_next (kills its 370-LOC hand-rolled parser) |
| 4 | `common_serving_frame` | the deployment-config handler skeleton | 12 | glm52 adapter | glm52, glm5_next, qwen38-27b (its 2,322-LOC third-generation frame server) |
| 5 | `common_stagepack_format_ext.h` | stagepack structs + validator + per-family tensor-kind table (the 64-byte entry is already proven cross-family identical by DRY-1) | 25 | qwen4_flash format.h | all 13 families (the quintuplets: 1,879 LOC) |
| 6 | `common_pack_load_bind` | pack load/bind over `spark_pack_load_common.h` | 12 (existing macro keys) | gemma4's adoption | ling, laguna, muse, k3, dsv5 |
| 7 | `common_kv_frame` (`LmKvFramePrepareFrame` et al.) | the JIT KV tier: frame preparation, eviction, restore batches | 10 | qwen38_max module.c | qwen38_max, qwen4_flash, muse_glimmer (all three use it) |
| 8 | `common_kv_geometry.h` | capacity fillers + geometry asserts | 9+12 | glm52/glm5_next twin (differs by exactly 7 constants) | glm pair, k3 (kills its `abi_version` drift), ling |
| 9 | `spark_hybrid_state.h` | KDA/GDN/sliding state: ordinal builder, slot-bytes algebra, pool sizing (kernels stay in inference/kernels) | 11 | k3 KDA | k3, ling, gemma4, laguna |
| 10 | `spark_rope_plan.h` | rope table builder (theta/yarn/table) + upload; launches ride the shared LmRopePerHeadKernel | 8 | ling | ling, gemma4, qwen trio, glm pair |
| 11 | `common_glm_stage_module` | the shared GLM module functions (26 shared, 17 at 100%; glm52 is a strict subset) | 15 | glm52 module.c | glm52, glm5_next |
| 12 | `spark_resident_decode_stage_firmware_common.h` | firmware ABI constants + per-family value table | 10 | ling/laguna pair | all families (kills the ABI drift class) |
| 13 | `common_validation_oracle` + harness | the GDN oracle twin (507=507 lines), the GPU gates, the rigs | 8-12 | qwen reference.c | ling, gemma4, dsv5, muse, qwen trio (keeps laguna's layer7 realpack rig distinct) |
| 14 | `common_pack_synthesizer` | in-module pack_synthesize.c copies (DRY-1's corpus missed these) | 10 | ling vs laguna (20.6% normalized divergence) | ling, laguna, muse, k3, dsv5 |
| 15 | `common_deployment_generator` | gen_deployment with per-family key table | 4 | any (50% pairwise) | all families |
| 16 | batch-tuning ladder shim | bucket ladder shape | 5 | ling/laguna shim | k3, dsv5 |
| 17 | Makefile wrappers | rules.mk-only thin wrappers (glm Makefiles are 83.5% identical despite rules.mk existing) | 7 | rules.mk | glm pair, qwen trio (60-71%) |

Small-constants fold (router sort capacity, caps, the five rope-formula restatements)
folds into keys 1/5/10 — including the live bug class where qwen4_flash ships qwen38's
`SPARK_QWEN38_ROUTER_SORT_CAPACITY` name.

## 4. Test contract (per module, non-negotiable)

Each module ships, in the same PR as its first adoption:

1. **Module test** — exercises the interface against synthetic vectors generated from
   `llm_defines.h` values (the derived-constants law: the test parses the same single
   source). Runs in CI, host-side; no GPU required for algebra modules; kernel modules
   get the existing sm_121a gate plus one correctness vector.
2. **Identity receipt** for the adoption — byte-identity where the artifact is a pack
   (the #977 pattern), behavior-identity (same tokens/hidden dump) where it is a
   runtime path.
3. **Negative control** — the test fails when a value is flipped (the K3A oracle
   pattern: flipped byte must FAIL with the exact tensor/offset).

This is what makes "the only issues are integration problems" true: a bug found inside
a common module is fixed once, and every driver's test suite proves the fix.

## 5. Already-common: the bypass ledger (migrate, do not rebuild)

| Shared seam | Used by | Bypassed by | Action |
|---|---|---|---|
| `spark_serving_adapter_template.h` | k3, dsv5, gemma4, muse | ling, laguna, glm5_next (parser), qwen38-27b | modules 3+4 |
| `spark_pack_load_common.h` | gemma4 | ling, laguna, muse, k3, dsv5 | module 6 |
| weightd lazy-pack attach | qwen38max, glm pair | qwen4_flash, qwen38-27b (direct pack read for identical formats) | unify on the lazy path per the weights law — the direct-read bypasses get deleted, not wrapped |
| `rules.mk` | some | glm pair, qwen trio | module 17 |
| tp_device_collective (shared transport) | all via the seam | k3's TP4xPP4 nccl pin (deployment leftover) | flip at next deployment regen |

## 6. Adoption order — cheapest identity proof first

1. **Byte-identical kills** (days, not weeks): `SparkModuleKvPrepareFrame` (byte-equal
   today), the qwen kernel suite (identical after rename), glm52→glm5_next subset
   functions, ling/laguna api.h/launch_shape/unity.cu rename-only trio, kv_geometry
   twins.
2. **Template completions**: `common_serving_tp_config` (the template's missing
   loader) then migrate ling/laguna/glm5_next/qwen38-27b adapters; pack-load migration
   for the five hand-rollers (mechanical — the template exists).
3. **Format + firmware headers**: stagepack format table (extends DRY-1's
   stagepack_core wave 1, #977) and the firmware ABI header.
4. **Algebra modules**: hybrid state, rope plan, kv geometry filler (kills the k3
   `abi_version` drift).
5. **k3's generational migration** last and separately: it is a different generation
   (`inference/llms/kimi_k3/`) bypassing 5 of 8 seams; migrate after the modules it
   needs are stable, one seam per PR with identity receipts.
6. **Tools** continue per the DRY-1 order (qwen4_flash → qwen38max → 27b → glm pair →
   ling → dsv5 → k3 → hy4), now fed by module 14's synthesizer and the common format
   table.

Each step lands as its own PR per the 1:1 law; a family is "on common modules" when
its deleted-copy LOC is ledgered in the PR description (capability alignment: DONE =
replaced deleted).

## 7. Relation to existing programs

- **DRY-1 (#976) + stagepack_core (#977)**: the tooling layer of the same idea. This
  doc extends it to driver-side modules; the conversion orders interleave.
- **Constant audit generative set (#979, G1-G6)**: `llm_defines.h` is G2/G3's parent
  promoted to a file: `gen_geometry_header.py` emits the derivable rows FROM it, the
  `_Static_assert` chains guard the cross-domain invariants, G4's wire-code registry
  and G6's `tools/port_ledger.json` hang off the same single source. One pyramid:
  `llm_defines.h` (values) → generators (derived rows) → `_Static_assert` (cross-domain)
  → CI identity checks (drift alarms).
- **Accuracy law (09-13)**: module 13's harness is the instrument; the T1 decode
  receipt remains the driver-level gate. Determinism rigs become common code with
  per-family tables instead of five private copies.
- **Weights law**: the direct-read bypass unification (section 5) is required
  compliance, scheduled with wave 2 of the adoption order.
- **Fast inference**: kernel modules 1/2 consolidate the exact kernels the roofline
  program tunes; hill-climb gains then land once for every adopting driver instead of
  being re-ported three times.

## 8. New-driver recipe (the payoff)

1. Copy `llm_specifics.h` to the new family's include path.
2. Fill every `SET_ME_*` from the model card. Compiler errors gone = config done.
3. Write only what no common module covers (genuinely new math; propose it as module
   candidates — generic math gets promoted even at one user).
4. Integration-test against the common validation harness, then the T1 receipt.

hy4 today (188-LOC honest lifecycle shell, zero copies) is exactly this starting shape.

## 9. Note on minimax-h3

`lane/minimax-driver` carries a diffusion MEDIA stage (adaLN, audio/video VAE,
flow-matching sigma scheduler — `lane/minimax-driver-rebase6` does not exist on
origin). It is not a decode-stage driver; modules 5, 15, 12 and the serving template
apply to it, and its flow scheduler is unique-math candidate for a future common
module. Its merge to main is a separate lane decision.

## 10. Annexes

- SEAM-1: branch `lane/wave-seam1-survey` — report + `seams.json` (six families,
  9 clusters, per-family define counts 81-211, union estimate ~200 keys).
- SEAM-2: branch `lane/wave-seam2-survey` — report + `seams.json` (seven families,
  13 clusters, role-level clone method, minimax branch survey).
- DRY-1: #976 (packbuilder/verifier consolidation, 13,748 LOC corpus).
- Constant audit: #979, `docs/CONSTANT_AUDIT.md`, generative set G1-G6.

## 11. Interface details (the no-audit contract)

Every module's complete public surface. A dev reads this section and `llm_specifics.h`;
nothing else. Signatures use the transport/module C conventions (compact Allman,
stdint, `SparkStatus` returns). Modules marked *(shim)* wrap existing shared seams.

### M-0 `spark_tp_mesh_kernels.cuh` + `spark_tp_mesh_register.h` — TP combine kernels *(LIVE)*
The fused FP32 allreduce (100us path, piece 1) + publish/wait/guard + u64 maxloc.
```c
/* kernels.cuh (CUDACC only): launchers take (cudaStream_t, ...) */
cudaError_t SparkGlm5NextLaunchSumRanksF32(stream, void *dest,
    const void *const *sources, uint32_t source_count, uint32_t element_count);
cudaError_t SparkGlm5NextLaunchSeedF32(stream, float *dest, const void *a,
    const void *b, uint32_t element_count);          /* fallback path */
cudaError_t SparkGlm5NextLaunchAddF32(stream, float *dest, const void *b, uint32_t n);
cudaError_t SparkGlm5NextLaunchRoundF32(stream, void *dest, const float *src, uint32_t n);
cudaError_t SparkGlm5NextLaunchAccumU64Max(stream, uint64_t *dest, const uint64_t *src, uint32_t n);
cudaError_t SparkGlm5NextLaunchMeshPublish/Wait/Guard(stream, ...);  /* transport-internal */
/* family/module/spark_module_combine.h: one call fills the transport config */
static inline void SPARK_FAMILY(ModuleRegisterCombines)(SparkTpDeviceCollectiveConfig *configuration);
```
Params (llm_defines): none beyond `SPARK_LLM_TILE_THREADS`-class constants. Adoption:
delete the private kernel copies in cuda.cu. Done for every TP driver except k3: the
private copies summed rank by rank in BF16, and the common combines sum all ranks in
FP32 and round once.

### M-1 `common_gdn_stage_kernels.cu` — qwen decode kernel suite

`common/common_gdn_stage_kernels.cu` holds the GDN, attention, MoE and head kernels with their `LmGdnStageLaunch*` launchers, declared in `common_gdn_stage_kernels.h`. qwen38_max and qwen4_flash include it into their CUDA unit, and each family's `llm_defines.h` supplies the `SPARK_LLM_*` geometry. qwen38_27b still carries its own copies.

**Grouped expert views are rank-local.** `LmGdnStageLaunchGroupedExpertLinear` and `LmGdnStageLaunchGroupedExpertTileLinear` take the stage pack's view of this rank's expert shard. With `SPARK_LLM_ROUTED_EXPERT_COUNT` experts over `tp_degree` ranks, the view's `output_dimension` is `experts_per_rank × rows_per_expert`, and its payload and scales start at this rank's first expert. Only the route tables (`group_row_offset`, `group_tile_prefix`) are global, so the launchers offset them by `tp_rank × experts_per_rank`.

The launchers validate `tp_degree` and `tp_rank` before dividing by the degree. An FP8 block-128 view needs `rows_per_expert` and `input_dimension` to be multiples of 128, because its scales are stored per 128×128 block. An NVFP4 view needs `input_dimension` to be a multiple of 16. `tests/test_gdn_stage_launch_checks.cu` checks these refusals; it needs nvcc but no GPU.

### M-2 `common_glm_cuda_tree` — GLM kernel tree
`config.h` + `unity.cu` + `layer.cuh` with the family prefix parameterized by
`SPARK_LLM_FAMILY_TAG` include-path resolution (glm52/glm5_next include trees already
87.4% rename-only). Interface = the existing `Glm5<Lm>Launch*` surface, unchanged
names after adoption. Params: 60 keys (tile config, KDA/MLA geometry, HC).

### M-3 `common_serving_tp_config` *(shim over the adapter template)*
```c
SparkStatus SparkServingAdapterTemplateLoadTpCollective(
    const SparkServingAdapterTemplateConfiguration *configuration,
    SparkServingAdapterTemplateRuntime *runtime);   /* the missing loader; seed: glm52 */
```
Deletes the 370-LOC hand-rolled TP JSON parsers (ling/laguna/glm5_next/qwen38-27b).
Status: every adapter with a TP collective (dsv4, glm52, glm5_next, laguna, ling,
qwen38-27b) loads it through `SparkServingAdapterTemplateLoadTpCollective`. glm5_next
moved last; its loader produced byte-identical state for the sixteen committed stage
configs and rejected the same 27 malformed ones before and after.

### M-4 `common_serving_frame` *(shim)* — deployment-config handler skeleton.
12 params; seed glm52 adapter. qwen38-27b's 2,322-LOC frame server collapses onto it.

### M-5 `common_stagepack_format_ext.h`
```c
typedef struct { char magic[8]; uint32_t abi_version, kind_table_id, entry_count;
                 uint64_t entry_bytes; } SparkStagepackHeader;   /* 64B entry proven */
SparkStatus SparkStagepackValidate(const void *mapping, size_t bytes,
    uint32_t family_kind_table_id, SparkStagepackView *out);
```
25 params incl. the per-family tensor-kind table (generated from llm_defines).

### M-6 `common_pack_load_bind` *(shim over spark_pack_load_common.h)*
```c
SparkStatus SparkCommonPackLoadBind(const SparkLlmPackPlan *plan,   /* 12 macro keys */
    SparkLlmPackBinding *out);
```

### M-7 `common_kv_frame`

`common/common_kv_frame.h` runs the JIT KV tier for qwen38_max, qwen4_flash and muse_glimmer. Each module reaches it through `family/module/spark_module_open_kv_tier.h`, `spark_module_kv_frame_ops.h` and `spark_module_kv_prepare_frame.h`, and its `llm_defines.h` names the frame's constants (`SPARK_LLM_KV_BLOCK_TOKENS`, `SPARK_LLM_KV_STAGING_RECORDS` and the rest).

`LmKvFramePrepareFrame` prepares one decode frame in six steps, each its own function:

1. `LmKvFrameCheckTable` validates the block table and grows the logical-to-slot map.
2. `LmKvFrameCollectLanes` lists the frame's lanes in the order their first row appears, each with the blocks its longest row needs.
3. `LmKvFrameSetPins` pins the blocks that are already resident.
4. `LmKvFrameRestoreMissing` claims a slot for each missing block (`LmKvFrameClaimSlot`, which evicts when the pool is full) and restores the blocks in batches of `SPARK_LLM_KV_STAGING_RECORDS` (`LmKvFrameFlushRestore`).
5. `LmKvFrameUploadTables` and `LmKvFrameMapRows` upload the frame's block table and slot mapping.
6. The pins are released. On failure, `LmKvFrameUnwind` also returns the slots claimed for blocks that were not restored.

**Eviction.** A slot the frame claims stays pinned until the frame ends, so a later eviction in the same frame cannot take it back. A frame that needs more blocks than the pool holds fails with `SPARK_STATUS_CAPACITY_EXCEEDED` and returns every slot it claimed. The eviction cursor moves past each slot it evicts, so the pool evicts in slot order instead of evicting again the block the previous frame restored.

`tests/test_llm_module_contract.c` covers restores that span several batches, lane order, write-back on eviction, two evictions in one frame, a frame larger than the pool, eviction order, and the unwind.

### M-8 `common_kv_geometry.h` — capacity fillers + asserts (glm twins differ by 7 keys).

### M-9 `spark_hybrid_state.h` — KDA/GDN/sliding ordinal builder + slot algebra.
```c
uint64_t SparkHybridStateOrdinal(const SparkLlmHybridPlan *plan, uint32_t layer);
uint64_t SparkHybridSlotBytes(const SparkLlmHybridPlan *plan);
```

### M-10 `spark_rope_plan.h` — theta/yarn table builder + upload (8 params; kernels stay shared).

### M-11 `common_glm_stage_module` — the 26 shared GLM module functions (glm52 strict subset).

### M-12 `spark_resident_decode_stage_firmware_common.h` — ABI constants + per-family value table.

### M-13 `common_validation_oracle` — oracle twins + GPU gates + rigs, per-family tables.

### M-14 `common_pack_synthesizer` — in-module pack_synthesize (ling/laguna 20.6% divergence).

### M-15 `common_deployment_generator` — gen_deployment + per-family key table (4 params).

### M-16 batch-tuning ladder shim (5 params) — k3, dsv5.

### M-17 Makefile wrappers over rules.mk (7 params).

## 12. llm_specifics.h / llm_defines.h — the mechanism (LIVE)

- `model-families/common/include/sparkpipe/llm_specifics.h` — the specimen, 52
  sentinel keys, every value `SET_ME_*` (intentionally uncompilable). A new driver
  copies it to its include path as `llm_defines.h`, fills values from the model card,
  and the compiler errors ARE the remaining-work checklist.
- `model-families/glm5_next/include/sparkpipe/llm_defines.h` — the first real seed
  (verified: covers 52/52 specimen keys).
- Family headers (`spark_<family>_model.h` etc.) become thin shims including it —
  adoption is include-path only, no call-site changes.
- The generators (gen_geometry_header.py), `_Static_assert` chains, and CI identity
  checks all key off this file per section 7's pyramid.

## 13. Mesh-kernel adoption status (the pilot)

M-0 is the pilot of this system: extracted from glm5_next (fused FP32 sum by-value
16-source kernel, seed/add/round fallback, u64 max, mesh publish/wait/guard), with
glm5_next converted (private copies deleted).

Every module whose driver runs `tp_device_collective.c` compiles the header, because
the collective calls its mesh launchers and a driver without them does not link.
`spark_tp_mesh_register.h` declares the six launchers the combines call.

**One set of combines.** `include/sparkpipe/family/module/spark_module_combine.h` holds
the six combine wrappers (fused FP32 sum, FP32 seed, add and round, BF16 sum, u64 max)
and `SPARK_FAMILY(ModuleRegisterCombines)`. Each wrapper reports a failed launch
through `SparkStageModuleCudaStatus` with the family's module tag: it logs the site and
returns `SPARK_STATUS_CAPACITY_EXCEEDED` for an out-of-memory error and
`SPARK_STATUS_INTERNAL_ERROR` for any other. The TP-open templates register them for
nine drivers, qwen38_27b's `spark_qwen38_27b_tp.c` for its own, and glm5_next assigns
them itself because its HC collective takes five of the six. k3 still runs the hidden
transport. `tests/test_tp_collective_open.py` checks, for every mesh driver, that each
collective registers its family's combines and that they classify a failed launch that
way.
