# Common Module Architecture — parameterized shared modules for every driver

Status: maintained by the lead dev and last verified against the code on
2026-09-28.

Where live adoption status lives:

- §3 (module surfaces) and §5 (mesh kernels) in this doc.
- The allow-lists in `tests/test_template_adoption.py`, which the gate
  enforces.

Headers under `include/sparkpipe/family/` are covered in
[`FAMILY_TEMPLATES.md`](FAMILY_TEMPLATES.md).

## 1. History

This design dates from 2026-09-13 (commit e889ccb). It drew on four inputs:

- the SEAM-1 and SEAM-2 surveys, on branches `lane/wave-seam1-survey` and
  `lane/wave-seam2-survey`;
- the DRY-1 packbuilder study (#976);
- the constant audit (#979), now
  [`archive/CONSTANT_AUDIT.md`](archive/CONSTANT_AUDIT.md);
- the operator's parameterization directive.

The surveys estimated about 26,000 lines of near-copy driver code. Most of it
bypassed shared seams that already existed rather than filling gaps.

Several bypasses the surveys named have since been removed:

- Every TP adapter loads through the shared loader (§3 M-3).
- qwen4_flash attaches its pack through `spark_weightd_lazy_pack.h`.
- No deployment pins NCCL.

The survey's per-family line counts and bypass table are history, not status.

## 2. Principles

1. **Shared code lives once.** A common module owns one coherent job, such
   as KDA state algebra, rope tables, stagepack format or serving TP config.
   Its code lives in `common/`, in `model-families/common/include/sparkpipe/`,
   or as a family template under `include/sparkpipe/family/`.
2. **`llm_defines.h` is the one parameter file per family.** Each family's
   include path carries an `llm_defines.h` with every value the common
   modules need: geometry, tile choices, state sizes and batch ladder. The
   family model header should include it instead of restating it. Six
   families do so today; §4 lists the rest.
3. **`llm_specifics.h` is the specimen.**
   `model-families/common/include/sparkpipe/llm_specifics.h` defines 52
   `SPARK_LLM_*` keys, each set to an intentionally uncompilable `SET_ME_*`
   value. To start a new driver:
   1. Copy the file to the family as `llm_defines.h`.
   2. Fill in every value.
   3. Build. When the compiler errors are gone, the configuration is done.
4. **No universality requirement.** A common module may serve a single
   driver if its math is generic.
5. **Every common module ships a validating test.** Each module has its own
   test, so a fix inside it is proved once for every driver.
6. **Adoption deletes.** A module is adopted only when the copies it
   replaces are deleted in the same PR. Restated constants and `#ifndef`
   defaults are findings, not style.
7. **Conversions prove identity.** Each migration PR shows either byte
   identity (for a pack) or behavior identity (same tokens or hidden dump)
   against the copy it replaces. Example: #977 rebuilt the pack
   byte-identically.

### Test contract (per module)

Each module ships three tests, in the same PR as its first adoption:

1. **Module test.** It uses synthetic vectors generated from `llm_defines.h`
   values. Algebra modules run host-side. Kernel modules get the sm_121a
   gate plus one correctness vector.
2. **Identity receipt** for the adoption.
3. **Negative control.** Flipping a value must fail, and the failure must
   name the tensor or key.

## 3. Module surfaces

This section lists symbols that exist in the tree. A planned module gets
no signature until it has code.

### Implemented

#### M-0 `spark_tp_mesh_kernels.cuh` + `spark_tp_mesh_register.h`: TP combine kernels

Both files are in `model-families/common/include/sparkpipe/`.

`spark_tp_mesh_register.h` declares the six launchers the combines call:

- `SparkTpLaunchSumRanksF32`
- `SparkTpLaunchSeedF32`
- `SparkTpLaunchAddF32`
- `SparkTpLaunchRoundF32`
- `SparkTpLaunchAccumAdd`
- `SparkTpLaunchAccumU64Max`

`spark_tp_mesh_kernels.cuh` also defines the mesh launchers that
`ring/transport/tp_device_collective.c` calls:

- `SparkTpLaunchMeshPublish`, `SparkTpLaunchMeshWait` and
  `SparkTpLaunchMeshGuard`
- `SparkTpLaunchMeshSeqPad` and `SparkTpLaunchMeshCopyDown`
- `SparkTpLaunchMeshTree` and `SparkTpLaunchMeshRoundLoop`
- `SparkTpLaunchMeshHardware`

`include/sparkpipe/family/module/spark_module_combine.h` wraps the six
launchers and fills the transport config through
`SPARK_FAMILY(ModuleRegisterCombines)(SparkTpDeviceCollectiveConfig *)`.

These kernels take no `llm_defines.h` keys.

Every TP driver except k3 uses them. The private copies they replaced summed
rank by rank in BF16. The common combines sum all ranks in FP32 and round
once (82217a1).

##### Build marker

In CUDA units, `spark_tp_mesh_kernels.cuh` defines
`SPARK_TP_MESH_KERNELS_MARKER`, a version string, and emits it twice:

- as the `__constant__` array `SparkTpMeshKernelsBuildMarker`;
- as the host array `SparkTpMeshKernelsBuildMarkerHost`. This copy is marked
  `__attribute__((used))`, so the host compiler keeps it even though nothing
  references it.

`modules/resident_decode_stage_rules.mk` reads the marker value from the
header with `sed` into `MODULE_MESH_KERNELS_MARKER`. Its `publish` target
fails with `MESH-KERNELS-MARKER-MISSING` when that value is empty or when
`strings` does not find it as a whole line in the module archive.

The host copy guarantees that a plain copy of the string sits in the
archive's host object, so the check does not rely on the `__constant__`
copy. Both copies come from the same macro in the common header, so the
check passes only when the archive carries the current header's marker. An
archive built from a stale copy with an older marker, or without the
header, fails.

#### M-1 `common_gdn_stage_kernels.cu`: qwen decode kernel suite

`common/common_gdn_stage_kernels.cu` holds the GDN, attention, MoE and head kernels with their `LmGdnStageLaunch*` launchers, declared in `common_gdn_stage_kernels.h`. qwen38_max and qwen4_flash include it into their CUDA unit, and each family's `llm_defines.h` supplies the `SPARK_LLM_*` geometry. qwen38_27b still carries its own copies.

**Grouped expert views are rank-local.** `LmGdnStageLaunchGroupedExpertLinear` and `LmGdnStageLaunchGroupedExpertTileLinear` take the stage pack's view of this rank's expert shard. With `SPARK_LLM_ROUTED_EXPERT_COUNT` experts over `tp_degree` ranks, the view's `output_dimension` is `experts_per_rank × rows_per_expert`, and its payload and scales start at this rank's first expert. Only the route tables (`group_row_offset`, `group_tile_prefix`) are global, so the launchers offset them by `tp_rank × experts_per_rank`.

The launchers validate `tp_degree` and `tp_rank` before dividing by the degree. An FP8 block-128 view needs `rows_per_expert` and `input_dimension` to be multiples of 128, because its scales are stored per 128×128 block. An NVFP4 view needs `input_dimension` to be a multiple of 16. `tests/test_gdn_stage_launch_checks.cu` checks these refusals; it needs nvcc but no GPU.

#### M-2 `model-families/glm52/cuda_tree/`: GLM kernel tree

- **Files.** `spark_glm_cuda_api.h`, `spark_glm_cuda_config.h`,
  `spark_glm_cuda_launch_shape.h`, `spark_glm_cuda_layer.cuh`,
  `spark_glm_cuda_unity.cu` and `spark_glm_batch_tuning.h`.
- **Users.** Only glm52 compiles it, through `spark_glm_cuda_unity.cu`.
  `tests/test_common_glm_cuda_tree.c` covers it.
- **glm5_next.** It keeps its own tree under
  `modules/glm5_next_resident_decode_stage/source/cuda/`, and shares pieces
  with glm52 through `include/sparkpipe/family/glm/`.

#### M-3 `common_serving_tp_config`: the adapter template's TP loader

```c
SparkStatus SparkServingAdapterTemplateLoadTpCollective(
    const SparkServingAdapterTemplateConfiguration *configuration,
    SparkServingAdapterTemplateRuntime *runtime);
```

Every adapter with a TP collective loads its config through this function:
dsv4, glm52, glm5_next, laguna, ling and qwen38_27b.

glm5_next moved last. Its loader produced byte-identical state for the
sixteen committed stage configs, and rejected the same 27 malformed configs
before and after the move.

#### M-5 `common/common_stagepack_format_ext.h`: stagepack format

- **Directory entry.** `SparkStagePackEntry` is 56 wire bytes, fixed by
  `SPARK_STAGEPACK_ENTRY_BYTES` and a `static_assert`.
- **Family tables.** `SparkStagePackFamilySpec` holds geometry, and
  `SparkStagePackKindTable` maps tensor kinds to roles.
- **Helpers.** The `SparkStagePackFamily*` helpers derive tensor shapes,
  for example `SparkStagePackFamilyTensorShapeOf` and
  `SparkStagePackFamilyNarrowShape`.
- **Users.** qwen4_flash only: its `llm_defines.h`, module, pack
  synthesizer and `tools/qwen4_flash_stagepack.py`.
- **Tests.** `tests/test_llm_stagepack_format.c` and its `_negative` twin.

#### M-7 `common_kv_frame`

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

#### M-8 `common/common_kv_geometry.h`: KV capacity filler

`SparkGlmKvFillCapacityRequest`, plus `_Static_assert`s on the
`SPARK_GLM_KV_*` geometry. Users: glm52 only, through
`spark_glm52_kv_geometry.h`.

#### M-9 `model-families/common/include/sparkpipe/spark_hybrid_state.h`: hybrid state algebra

- **Macros.** `SPARK_HYBRID_*` covers phase and window layer classes, KV
  slot bytes and KDA state bytes.
- **Function.** One static inline function:
  ```c
  SparkHybridStateBuildOrdinals(phase_ordinal_by_layer, window_ordinal_by_layer,
      phase_count, window_count, total_layer_count, first_layer_index,
      layer_count, period, phase)
  ```
  There is no plan struct.
- **Users.** gemma4.

#### M-10 `model-families/common/include/sparkpipe/spark_rope_plan.h`: rope tables

- **Domains.** `SparkRopeDomainInitTheta` and `SparkRopeDomainInitTable`.
- **YaRN.** `SparkRopePlanBuildYarnInvFrequency` and
  `SparkRopePlanYarnTableIsValid`.
- **Users.** gemma4 and laguna. `tests/test_rope_plan.c` covers it.

#### M-11 `model-families/glm52/stage_module/spark_glm_stage_module.h`: shared GLM module functions

Users: glm52 only. `tests/test_common_glm_stage_module.c` covers it.

#### M-12 `spark_resident_decode_stage_firmware_common.h`: firmware ABI constants

This header is in `model-families/common/include/sparkpipe/`. The firmware
headers of laguna and ling include it.

#### M-14 `spark_pack_synthesize_common.h`: shared pack synthesis core

Six pack synthesizers include it: glm5_next, laguna, ling, qwen38_27b,
qwen38_max and qwen4_flash.

`tests/test_template_adoption.py` lists dsv41_flash, gemma4 and
muse_glimmer as known offenders, because each has its own generator.

#### M-16 Batch-tuning ladder

Each module's `spark_*_batch_tuning.h` includes one of two shared headers:

- `include/sparkpipe/spark_batch_variant_tuning_common.h`
- `spark_glm_batch_tuning.h`

glm52, glm5_next, laguna and ling do so. dsv4 and k3 are allow-listed
offenders ("own bucket ladder").

#### M-17 Makefile rules

Every `*_resident_decode_stage` Makefile includes
`modules/resident_decode_stage_rules.mk`. That is 13 modules. k3 and
mimo26 have no module Makefile.

#### `spark_pack_load_common.h`: region hook

`spark_pack_load_common.h` is in `model-families/common/include/sparkpipe/`.
For each directory entry, its `LoadEntry` validates the entry through the
family's `ValidateEntry`, rejects a duplicate and marks coverage. Unless the
family marks the entry validate-only (`SPARK_PACK_LOAD_ENTRY_IS_VALIDATE_ONLY`),
it then obtains device pointers for the entry's payload and scale planes in
two steps.

1. If the family defines `SPARK_PACK_LOAD_REGION_HOOK`, `LoadEntry` calls
   the hook with the state, the entry, the pack file and two pointer
   outputs.
   - A return of 1 means the hook supplied the pointers, which may be null.
   - Any other value means the hook did not consume the entry.
   - The hook returns an `int`, not a `SparkStatus`, so it cannot report an
     error.
2. If there is no hook, or the hook did not consume the entry, `LoadEntry`
   calls `SparkStageModuleLoadDeviceRegion` for the payload. It calls it for
   the scale too when `scale_bytes != 0`. That function does not copy the
   region:
   - With weightd configured, it attaches each pack file once per ledger
     through `SparkWeightdAttachMappedPack` and returns a pointer into that
     whole-pack mapping.
   - Without weightd, `SparkWeightdAttachRequested` returns `BUSY`, and the
     function logs a refusal and fails with `UNSUPPORTED`.

Three families define the hook: laguna (`SparkLagunaModuleRegionHook`),
qwen38_max (`SparkQwen38MaxModuleRegionHook`) and qwen4_flash
(`SparkQwen4FlashModuleRegionHook`). Each hook serves entries from the
module's `SparkWeightdLazyPack`:

- **Routed expert slabs** (`MOE_W1`, `MOE_W3` and `MOE_DOWN`; laguna's
  `EXPERT_GATE_UP` and `EXPERT_DOWN`): the hook returns 1 with null
  pointers, so nothing is mapped at load time. At execute time the module
  leases the routed experts through the lazy pack's `SparkWeightdMap`
  (`SparkWeightdMapAcquire`).
- **Every other entry**: the hook calls `SparkWeightdLazyPackSlice` into the
  resident spine for the payload, and for the scale when `scale_bytes != 0`,
  and returns 1 if every slice succeeds.
- **Return 0**: when the module has no lazy pack, when the lazy pack is not
  `ready` (checked by qwen38_max and qwen4_flash), or when a slice fails.

**Open debt.** A hook that returns 0 sends the entry down the
`SparkStageModuleLoadDeviceRegion` path. Neither the hook nor `LoadEntry`
logs why:

- **Without weightd.** The family's lazy open treats `BUSY` as success and
  leaves `lazy_pack` null. The lazy opens are the shared `LazyOpen` in
  `spark_module_lazy_open.h` for laguna, `SparkQwen38MaxModuleLazyOpen` and
  `SparkQwen4FlashModuleLazyOpen`. The load then fails with `UNSUPPORTED`
  at the first loaded entry, inside `SparkStageModuleLoadDeviceRegion`,
  instead of in the lazy open.
- **With weightd, when a slice fails.** The entry switches from the lazy
  spine slice to the whole-pack mapping. Only the generic `ERRSITE` line of
  the failed slice reaches stderr; the hook cannot say why it declined.

This second load path is a fallback, which the firmware contract forbids
(`sparkpipe_invariants.md`, preamble and I03). Mapping "weightd not
configured" to `BUSY` also conflicts with I17. The fix is for the hook to
return a status and for the loader to fail on any error. `TECHDEBT.md`
tracks the `BUSY` mapping and the deployments that omit weightd under "Model
residency and storage".

### Planned (no code yet)

- **M-4, common serving frame.** A shared deployment-config handler
  skeleton. It does not exist.
- **M-6, common pack load/bind.** `spark_pack_load_common.h` exists. gemma4,
  laguna, ling, qwen38_27b, qwen38_max and qwen4_flash include it. No
  wrapper module over it exists.
- **M-13, common validation oracle and harness.** The shared validation
  templates are under `include/sparkpipe/family/validation/`. No common
  oracle module exists.
- **M-15, common deployment generator.** Deployment generators are still
  per family (`tools/*_gen_deployment.py`, `tools/k3_gen_deployment.sh`).
- **Weight acquisition.** qwen38_27b's module sources reference no weightd
  API; this is a grep over its source and include directories. Moving it
  onto weightd attach is open.
- **k3 migrates last and separately.** It builds on `inference/llms/kimi_k3/`
  and still runs the hidden transport (§5).

## 4. `llm_defines.h` and the derivation law

**Derivation law.** This is the constant audit's rule, now held here:

1. Every derived constant is computed in code from the one `#define` that
   generates it.
2. Values from different domains are tied by `_Static_assert` or `#error`
   checks.
3. Scripts parse the header rather than restating its values;
   `tools/gen_geometry_header.py` is the generator.
4. A restated literal or an `#ifndef` default is a defect.

`model-families/common/include/sparkpipe/spark_driver_defines.h` produces
the derived `SPARK_LLM_*` keys once. `tests/test_driver_defines.py` checks
them against ling's `llm_defines.h`, with negative controls.

**Family coverage.** These counts come from the tree on 2026-09-28:

- 13 families have `llm_defines.h`: dsv41_flash, gemma4, glm52, glm5_next,
  hy4, k3, laguna, ling, minimax, muse_glimmer, qwen38_27b, qwen38_max and
  qwen4_flash.
- dsv4, mimo25 and mimo26 have none. minimax has one but no model header.
- These model headers include it: gemma4, hy4, laguna, ling, qwen38_max
  and qwen4_flash.
- These restate geometry as literals and do not include it: dsv41_flash,
  glm52, glm5_next, muse_glimmer and qwen38_27b.
  - Example: glm5_next states hidden 4096 and 288 experts both in
    `spark_glm5_next_model.h` and in its `llm_defines.h`.
  - No test or `_Static_assert` ties the two copies.
- k3's model header includes `spark_k3_llm_defines.h`, a separate
  `SPARK_K3_MODEL_*` file, and no k3 code includes k3's generic
  `llm_defines.h`.
- Until these headers become shims, a value edited in one copy is not
  checked against the other.

### Generated headers

These headers are rendered from contract JSON files. Each generator's
`--check` compares the rendered text with the tracked file byte for byte, so
a hand edit fails the check. For the first three rows, edit the generator or
the contract, never the header.

| Header | Generator | Source | Who runs `--check` |
| --- | --- | --- | --- |
| `inference/llms/kimi_k3/generated_config.h` | `tools/generate_k3_contract.py`, which also writes `model_contracts/k3.json` and the k3 model description | `model_contracts/k3_authoritative.json` | `tests/test_k3_checkpoint_contract.py` (in `make test`) and `.github/workflows/cuda13-sm121a-compile.yml` |
| `model-families/dsv4/include/sparkpipe/spark_dsv4_model.h` and `spark_dsv4_pro_model.h` | `tools/generate_dsv4_contracts.py`, which also writes `model_contracts/dsv4_flash.json`, `dsv4_pro.json` and the flash model descriptions | `model_contracts/dsv4_flash_authoritative.json` and `dsv4_pro_authoritative.json` | `tests/test_dsv4_contracts.py` (in `make test`), `tools/cuda13_sm121a_compile_gate.sh` and the workflow |
| `model-families/glm52/include/sparkpipe/spark_glm52_model.h` | `tools/glm52_model_contract.py`, which also writes the glm52 model descriptions | `model_contracts/glm52.json` | `tests/test_glm52_model_identity.py` (in `make test`) and `tools/cuda13_sm121a_compile_gate.sh` |
| `spark_qwen38_27b_model.h`, `spark_qwen38_27b_serving_constants.h` (`--emit-adapter-constants`), `spark_glm5_next_model.h` and `spark_qwen4_flash_model.h`, each under `model-families/<family>/include/sparkpipe/` | `tools/gen_geometry_header.py --family <family>` | `model_contracts/qwen38_27b_authoritative.json`, `glm53_flash_authoritative.json` and `qwen4_flash_authoritative.json` | nothing |

No gate runs `tools/gen_geometry_header.py --check`:

- **qwen38_27b** (model header and serving constants): apart from comments,
  the generated text matches the tracked files.
- **glm5_next**: the generator omits `SPARK_GLM5_NEXT_REPLAY_ROWS_MAX` and
  the `SPARK_GLM5_NEXT_MODEL_MISS_*` ring macros, which the tracked header
  defines.
- **qwen4_flash**: the tracked header is a shim over `llm_defines.h`, but
  the generator emits literal values.

## 5. Mesh-kernel adoption status

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

## 6. New-driver recipe

1. Copy `llm_specifics.h` to the new family's include path as
   `llm_defines.h`, and fill every `SET_ME_*` from the model card.
2. Make the family model header include it (§4).
3. Consume the implemented modules in §3 and the family templates. Write
   only the math that no common module covers.
4. Pass the module tests, then the driver's T1 decode receipt.

## 7. Annexes

- SEAM-1: branch `lane/wave-seam1-survey` (six families).
- SEAM-2: branch `lane/wave-seam2-survey` (seven families, including the
  minimax branch survey).
- DRY-1: #976 (packbuilder and verifier consolidation).
- Constant audit: #979,
  [`archive/CONSTANT_AUDIT.md`](archive/CONSTANT_AUDIT.md).
