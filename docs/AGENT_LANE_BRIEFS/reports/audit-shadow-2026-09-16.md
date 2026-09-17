# AUDIT-SHADOW report — silent-override defect sweep — 2026-09-16

Lane: lane/audit-shadow. Base: origin/main 763ae03. Workspace:
/Users/mac/auditshadow (clean clone).

Detector: `tools/audit_shadow.py` (committed, CI-able, stdlib-only Python).
Allowlist: `tools/audit_shadow_allowlist.txt` (currently zero entries).
Machine-readable output: `python3 tools/audit_shadow.py --json <path>`.

Run on this tree:

```
$ python3 tools/audit_shadow.py --json audit.json
audit_shadow: class1=45 class2=1 class3=19 class4=0 class5=209 class6=390 gate_failures=455 link_built=True
```

Class definitions:

- C1 — duplicate-signature copy: a private-scope definition with the exact
  normalized name (family tokens replaced by `<fam>`) or the family-prefixed
  variant of a definition in a COMMON header (include/sparkpipe,
  model-families/common/include/sparkpipe, runtime).
- C2 — header shadowing: an `#include` whose string resolves to multiple real
  headers, where the winner depends on include-path order (family-local beats
  common, or 4 family-local headers collide on one name).
- C3 — feature drift: a C1 copy whose parameter list or body tokens diverge
  from the common counterpart on suspicious markers (abort/deadline, slot
  tail/publish, diag/epoch, format stamp) or similarity < 0.5. This is the
  convicted shape: an older signature that silently misses new contract
  behavior.
- C4 — link resolution: `nm` over built objects; a symbol defined both in a
  common object and a private object (duplicate strong definition), or a
  common-API symbol resolved privately.
- C5 — `#ifndef` guard violations (pragma-once law) and getenv sites with
  silent defaults (the 2^N law surface).
- C6 — cross-family twins: private definitions with the same normalized name
  in 2+ distinct families (no common counterpart — the pre-common design
  surface), with per-twin constant-divergence fingerprints.

Blame ages: `stale_days` = last change on the common-side definition minus
last change on the private copy (git blame over the definition span). Positive
= the common side moved after the copy was written (copy missed it). Negative
= the family fixed its copy and the fix never landed in common.

## The convicted example, mapped (mesh kernels)

The live defect (glm5_next publish never wrote the slot tail the waits poll;
ops 1-5 passed on warmup leftovers, op 6+ deadlocked; no abort flag, no diag)
is present on main as a structural risk:

- Common implementation:
  `model-families/common/include/sparkpipe/spark_tp_mesh_kernels.cuh` —
  kernels at lines 9, 27, 41; launchers at 272, 281, 293.
- Private copy: `modules/glm5_next_resident_decode_stage/source/`
  `spark_glm5_next_resident_decode_stage_cuda.cu` lines 132-481 — 19
  definitions duplicating the entire common mesh set (C1, stale 0-17 days,
  sim 0.778-1.0). The module does NOT include the common header; the copies
  are hand-synced (last restore: merge 65bf4d8).
- Binding: `ring/transport/tp_device_collective.c:27-32` declares the
  family-named launchers `extern` and calls them at 589, 596, 854;
  `model-families/common/include/sparkpipe/spark_tp_mesh_register.h:3-8`
  declares the same family-named symbols behind generic
  `SparkTpMeshCombine*` wrappers. Whoever links the common transport or mesh
  register silently resolves these to WHICHEVER object defines
  `SparkGlm5NextLaunchMesh*` — the common header TU or the module's private
  copies. Nothing fails if they diverge.
- Fix class: unify-on-common. Module includes
  `sparkpipe/spark_tp_mesh_kernels.cuh`; delete lines 132-481. Same fix for
  the glm52/laguna/ling/dsv4 copies below.

Risk verdict: currently text-identical only because a human restored it.
The next common-side change (abort flag, slot tail, diag param — exactly the
#997-era evolution) will silently stop applying to the module build. Deadlock
and hang class.

## C1 — duplicate-signature copies (45)

### Mesh kernel set copies (37)

All against `model-families/common/include/sparkpipe/spark_tp_mesh_kernels.cuh`:

| Private copy (file:line) | Symbols | stale_days | sim | Risk / fix class |
| --- | --- | --- | --- | --- |
| modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_cuda.cu:132-481 | 19 defs: LoadBf16Pair 132, AccumAddKernel 148, SumRanksF32Kernel 172, LaunchSumRanksF32 195, SeedF32Kernel 217, AddF32Kernel 234, RoundF32Kernel 249, LaunchSeedF32 264, LaunchAddF32 272, LaunchRoundF32 280, AccumU64MaxKernel 288, LaunchAccumAdd 333, MeshPublishKernel 368, MeshGuardKernel 386, MeshWaitKernel 400, LaunchMeshGuard 439, LaunchMeshPublish 448, LaunchMeshWait 460, LaunchAccumU64Max 475 | 0-17 | 0.778-1.0 | deadlock/hang class. unify-on-common |
| modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_cuda.cu:126,142,161,188,196 | LoadBf16Pair, AccumAddKernel, AccumU64MaxKernel, LaunchAccumAdd, LaunchAccumU64Max | 28-29 | 1.0 | drift already present (StoreBf16Pair, see C3). unify-on-common |
| modules/laguna_resident_decode_stage/source/spark_laguna_resident_decode_stage_cuda.cu:132,148,167,208,216 | LoadBf16Pair, AccumAddKernel, AccumU64MaxKernel, LaunchAccumAdd, LaunchAccumU64Max | 0-4 | 0.889-1.0 | same class. unify-on-common |
| modules/ling_resident_decode_stage/source/spark_ling_resident_decode_stage_cuda.cu:129,145,164,191,199 | LoadBf16Pair, AccumAddKernel, AccumU64MaxKernel, LaunchAccumAdd, LaunchAccumU64Max | 6 | 1.0 | same class. unify-on-common |
| modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_cuda.cu:117,1489,2572,3379 | AccumU64MaxKernel, AccumAddKernel, LaunchAccumU64Max, LaunchAccumAdd | 31-40 | 0.545-1.0 | worst sim in class (0.545): behavior already diverged. unify-on-common |

### Pack-synthesize tool copies (8)

Against `model-families/common/include/sparkpipe/spark_pack_synthesize_common.h`:

| Private copy | Symbols | stale_days | sim | Risk / fix class |
| --- | --- | --- | --- | --- |
| modules/muse_glimmer_resident_decode_stage/tools/muse_glimmer_pack_synthesize.c:42,52,60,127,145,177 | SparkSynthNext, SparkSynthTensorSeed, SparkSynthAlign, SparkSynthWriteRegion, SparkSynthWrite, main (template main not used) | -11 | 0.577-1.0 | private fork NEWER than common: family fixes starve the common template. delete, use common header |
| modules/gemma4_resident_decode_stage/tools/gemma4_pack_synthesize.c:33,43 | SparkGemma4SynthNext, SparkGemma4SynthAlign (family-prefixed twins of common SparkSynth*) | -11 | 1.0 | same. delete, use common header |

## C3 — feature drift, the convicted shape (19)

| Private copy (file:line) | Common counterpart | stale_days | Divergence fingerprint | Risk / fix class |
| --- | --- | --- | --- | --- |
| modules/dsv4_resident_decode_stage/source/spark_dsv4_serving_adapter.c:1412 SparkModelServingAdapterGetInterface | model-families/common/include/sparkpipe/spark_qwen38_pp_serving_adapter_common.h:752 (template export) | 28 | hand-rolled interface struct; template machinery (SPARK_QWEN38_SERVING_ADAPTER_FN, ServingInterface) absent | required-op omission class (I01): template gains a required op, hand-rolled struct silently lacks it. unify-on-common (template) |
| modules/glm52_resident_decode_stage/source/spark_glm52_serving_adapter.c:986 (same symbol) | same | -9 | same | same |
| modules/glm5_next_resident_decode_stage/source/spark_glm5_next_serving_adapter.c:1419 | same | 5 | same | same |
| modules/k3_resident_decode_stage/source/spark_k3_serving_adapter.c:653 | same | 5 | same | same |
| modules/laguna_resident_decode_stage/source/spark_laguna_serving_adapter.c:1222 | same | -8 | same | same |
| modules/ling_resident_decode_stage/source/spark_ling_serving_adapter.c:1244 | same | -6 | same | same |
| modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_serving_adapter.c:2319 | same | 7 | same | same |
| modules/qwen4_flash_resident_decode_stage/source/spark_qwen4_flash_serving_adapter.c:285 | same | -3 | same | same |
| modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_cuda.cu:135 SparkGlm52StoreBf16Pair | spark_tp_mesh_kernels.cuh:90 | 28 | `__float_as_uint` + mask form vs common `__float_as_int` form | bf16 pack-bit semantics drift (numerics). unify-on-common |
| modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_cuda.cu:141 SparkGlm5NextStoreBf16Pair | same | 17 | same | same |
| modules/laguna_resident_decode_stage/source/spark_laguna_resident_decode_stage_cuda.cu:141 SparkLagunaStoreBf16Pair | same | 4 | same | same |
| modules/ling_resident_decode_stage/source/spark_ling_resident_decode_stage_cuda.cu:138 SparkLingStoreBf16Pair | same | 6 | same | same |
| modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_cuda.cu:361 SparkGlm5NextGlobalTimerNs | spark_tp_mesh_kernels.cuh:97 | 1 | missing `void` param | cosmetic today; marker of hand-sync. unify-on-common |
| modules/muse_glimmer_resident_decode_stage/tools/muse_glimmer_pack_synthesize.c:65 SparkSynthFillPayload | spark_pack_synthesize_common.h:55 | -11 | missing `const SPARK_SYNTH_ENTRY_T *entry`; missing SPARK_SYNTH_F32_FORMAT/PACKED_FORMAT/PACKED_NAN_MASK handling | pack payload layout drift: writes packs the common loader may misread. delete, use common |
| modules/muse_glimmer_resident_decode_stage/tools/muse_glimmer_pack_synthesize.c:80 SparkSynthAppend | spark_pack_synthesize_common.h:147 | -11 | missing `uint32_t quantize` param | stale-signature class. delete, use common |
| modules/muse_glimmer_resident_decode_stage/tools/muse_glimmer_pack_synthesize.c:101 SparkSynthBuildDirectory | spark_pack_synthesize_common.h:209 | 0 | missing `uint32_t quantize`; missing SPARK_SYNTH_EMIT_MTP_TAIL handling | MTP-tail drift: directory built without the tail the parity design expects. delete, use common |
| modules/gemma4_resident_decode_stage/tools/gemma4_pack_synthesize.c:60 SparkGemma4SynthAppend | spark_pack_synthesize_common.h:147 | -11 | missing 5 params (context, tensor_kind, layer_index, is_global, quantize) | stale-signature class. delete, use common |
| modules/gemma4_resident_decode_stage/tools/gemma4_pack_synthesize.c:84 SparkGemma4SynthWrite | spark_pack_synthesize_common.h:273 | -11 | missing context/path/header params; no chunked writer | stale-signature class. delete, use common |
| modules/laguna_resident_decode_stage/source/cuda/layer.cuh:687 | model-families/common/include/sparkpipe/spark_lm_kernels.cuh:881 | -1 | missing the FP8_E4M3_UE8M0 codec arm (LmActivationFp8QdqFloatRow) | silent activation-codec fallback: FP8 path absent where common has it. unify-on-common |

The 9-way `SparkModelServingAdapterGetInterface` hand-roll is the single
largest cluster. Common template at
`model-families/common/include/sparkpipe/spark_qwen38_pp_serving_adapter_common.h`
generates the interface struct and the visibility-default export; every
hand-rolled copy must be re-audited field-by-field against it on every
template change, silently, forever, unless unified.

## C2 — header shadowing (1 live collision, 4 collision candidates)

| Include site | Include string | Candidate set (binding = include-path order) | Risk / fix class |
| --- | --- | --- | --- |
| model-families/gemma4/include/sparkpipe/spark_gemma4_model.h:2 | `sparkpipe/llm_defines.h` | model-families/{gemma4,glm5_next,k3,qwen38_max}/include/sparkpipe/llm_defines.h — 4 files, same macro names, different values (SPARK_LLM_HIDDEN_DIMENSION 2816/4096/7168/8192; SPARK_LLM_MTP_ENABLED 0/1; ROPE_THETA 1e4/1e7) | geometry corruption class: binding is per-includer `-I` order; the winner is never visible at the include site. The gemma4 model header is force-included (`-include $(MODEL_HEADER)`) into every gemma4 module host object (modules/resident_decode_stage_rules.mk:65). Fix class: rename per family (`SPARK_LLM_*` -> `SPARK_<FAM>_LLM_*`) or fold into each family's `spark_<fam>_model.h`; the glm5_next/k3/qwen38_max copies are otherwise orphaned definition sets kept alive only by Makefile dependencies |

Verified: glm5_next/k3/qwen38_max model headers define family-prefixed
macros directly and do not include llm_defines.h; their llm_defines.h copies
are unbound landmines (any future bare `sparkpipe/llm_defines.h` include
binds silently per -I order; every module Makefile puts several family
include dirs on one command line, e.g. modules/glm5_next_resident_decode_stage/Makefile:71-76).

## C4 — link resolution (EMPTY under partial coverage)

The nm detector runs (`--objects-glob` or auto-discovery of
`build/**/*.o|*.a|*.so`) and executed over the only artifacts buildable on
this host: `build/libsparkpipe_core.a`, `build/libsparkpipe_model_common.a`
and their objects. Result: 0 duplicate strong symbols. Coverage is PARTIAL:
no nvcc on this host and module objects require the MODEL_REVISION contract
inputs, so module archives and CUDA objects were not link-checked. Proof:
`link_built=True` in the JSON; the gate must run this check post-build in CI
where full artifacts exist. Any C1 copy left un-unified becomes a C4 finding
there (both objects define the same symbol).

## C5 — guards and env silent defaults (209)

- 67 header files use `#ifndef SPARKPIPE_*_H` guards instead of `#pragma
  once` (violates the header law). Examples: include/sparkpipe/spark_stagepack_format.h,
  model-families/{glm52,glm5_next,k3,laguna,ling}/include/sparkpipe/spark_*_kv_geometry.h,
  all four family llm_defines.h. Full list in the JSON (`kind:
  header-guard`).
- 142 getenv sites across common+private code, 101 distinct variable names,
  16 sites where the statement selects a default with no warning/log on the
  fallback path (flagged `silent`). Inventory (var, default, file:line) in
  the JSON (`kind: env-default`). This is the audit surface for the 2^N law:
  101 knobs times compile-time defines is the untested-config explosion; the
  silent 16 are where required behavior can be quietly reduced.

## C6 — cross-family twins (390 groups, 1344 definitions, 31 with constant divergence)

No common counterpart exists; each family carries its own copy and they have
already diverged in 31 groups. The detector output lists every group with
per-twin file:line and constant fingerprints. The most dangerous:

| Group (normalized) | Twins | Divergence | Risk |
| --- | --- | --- | --- |
| Spark`<fam>`ValE4m3Decode | glm52 vs glm5_next vs ling validators | glm52 uses 8.0f; glm5_next/ling use 0.125f/512.0f | a family's validation golden decodes e4m3 with different constants: the validator can pass wrong numerics |
| Spark`<fam>`ModuleInitializeTpCollective | 10 copies | timeout constant 1000000ull present in only one twin | collective init timeout asymmetry: hang in some families, fast-fail in others |
| Spark`<fam>`OrderedHeadScore | 6 copies | 0.0f present only in dsv4 | scoring semantics drift |
| Spark`<fam>`ModuleBindGlobal / ModuleBindLayer | 5 copies each | sim 0.099-0.302 | bind protocol copies with ~10% shared text: each is already a fork |
| Spark`<fam>`ServingSubmit | 5 copies | sim 0.182-0.302 | submit-path forks: policy drift class (I02) |
| Spark`<fam>`ModuleExpectedMtpBits | 3 copies | expected bits 0ull vs 1ull | MTP expectation differs by family: one is wrong |
| Spark`<fam>`ModuleOpenKvTier | 4 copies | FNV hash constants present in 2 of 4 | kv tier open forks |
| LagunaLayerAttention (`<fam>`LayerAttention) | 4 copies | 1.0f scale only in laguna | attention scale drift |

Fix class: promote the shared algorithm to model-families/common (I06/I07
narrow hooks) or declare the exception in the allowlist with a review.

## Top 10 most dangerous shadows (behavioral risk ranked)

1. glm5_next private mesh publish/wait/guard set
   (spark_glm5_next_resident_decode_stage_cuda.cu:368-473) — deadlock/hang
   class; the convicted defect; hand-synced to common as of today, no
   mechanism prevents the next divergence; common transport binds these exact
   symbols.
2. 9-way hand-rolled SparkModelServingAdapterGetInterface (per-module
   serving adapters, e.g. spark_glm5_next_serving_adapter.c:1419) — a
   template-side required-op addition silently misses 9 drivers (I01).
3. llm_defines.h 4-family macro collision (model-families/*/include/sparkpipe/
   llm_defines.h) bound by -I order and force-included via the module host
   rule — wrong-geometry compilation with zero diagnostics.
4. muse_glimmer/gemma4 pack-synthesize forks with stale signatures
   (SparkSynthWrite/Append/BuildDirectory missing quantize/MTP-tail
   handling) — writes packs that diverge from the common pack contract.
5. StoreBf16Pair expression drift in glm52/glm5_next/laguna/ling — bf16
   pack-bit semantics diverging from the common kernel set (numerics).
6. ValE4m3Decode twin divergence between family validators — a validator
   with different decode constants can certify wrong numerics.
7. ModuleInitializeTpCollective 10-way twin with timeout constant in one
   twin only — hang vs fast-fail asymmetry across families.
8. laguna layer.cuh:687 missing the FP8_E4M3_UE8M0 activation arm — silent
   fallback to the non-FP8 path where common code has the codec.
9. dsv4 AccumAdd/LaunchAccumAdd copies at sim 0.545-0.8 vs common —
   reduction numerics already diverged, oldest copies in the class (stale
   31-40 days).
10. glm5_next private stagepack format stamp table
    (spark_glm5_next_stagepack_format.h:20-23: PAYLOAD_BF16=1/F32=2/U32=3/
    PACKED_WEIGHT=4) vs common spark_stagepack_format.h weight codes
    (BF16=0/F32=1/FP8=4/NVFP4=8): two numbering systems for "the" pack
    format stamps; six modules include the common header, glm5_next/glm52/
    dsv4/k3 families carry private ones. Misread class at any common-tool
    boundary. (Enum-level: detected via C2/C6 adjacency and verified
    manually; the enum-stamp crosscheck is a proposed detector extension.)

## Gate proposal (CI check)

`python3 tools/audit_shadow.py --gate` — exit 2 when any C1/C2/C3/C6 finding
is not in `tools/audit_shadow_allowlist.txt`. Wire as a CI step after the
build, plus the nm pass over all built objects (`--objects-glob 'build/**/*.o'
build/**/*.a'`) so C4 covers the full artifact set. Gate failures print
`GATE <class>|<path>|<symbol>` lines ready to paste as allowlist review
candidates; allowlist entries require a review reference in this lane's
report directory. Current state: the gate fails with 455 findings — that is
the cleanup backlog, not a detector defect.

## Method notes and limits

- Normalization replaces 18 family-token variants (snake/UPPER/Pascal/camel,
  boundary-aware) with `<fam>`; `main` excluded (program entry point, not a
  shadow class).
- The definition extractor is a paren/brace matcher over comment- and
  string-masked sources; bodies with unbalanced macros (line-continuation
  `#define` blocks emitting braces) can confuse span ends — every finding in
  this report was spot-verified against the cited lines.
- Blame ages are per-definition-span last-change dates; negative values mean
  the family copy is newer than common (common starved of family fixes).
- Not fabricated: C4 is reported EMPTY under partial coverage with the exact
  build proof; the enum-stamp item (top-10 #10) is labeled manual/adjacent
  rather than detector output.

## Blockers

- No nvcc and no module contract inputs (MODEL_REVISION) on this host: C4
  full-artifact link check must run in the GPU CI where modules build.
- The 455 gate failures require per-family unification work (the fix classes
  above); this lane ships the detector, the ledger, and the gate, not the
  unification.
