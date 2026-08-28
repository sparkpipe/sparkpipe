# W4 redundancy + automation lane — report 2026-08-27

Workstream W4 of docs/HOUSECLEANING_PLAN.md. Branch `lane/redundancy`,
worktree `/tmp/lane-redundancy`, base c18eabe. **Scoreboard context: this
lane is why the operator's "auditor finds nothing" directive is enforceable —
every duplication known and decided, dead code deleted with receipts,
complexity hotspots each carrying a disposition, and the first token-saver
generator landed with a byte-identity proof.** It costs 0 spark-hours and
unblocks W2 with a measured hit list.

Lane rules honored: no edits under `modules/` (W2's write set this sprint);
deletions cite evidence; generators prove byte-identity where a hand-written
original exists; never merged own PR (branch pushed, PR left to coordinator).

## 1. Duplicate-code detector — tools/dup_report.py (item 1)

Method: every function in `modules/ runtime/ node/ cache/ ring/`
(`*.c *.h *.cu *.cuh`) is extracted by a paren/brace state machine, its body
normalized to identifier/number/string-free token lines (renamed clones
match), cloned via 8-line window hashes, and matching windows extended into
maximal diagonal runs. Report threshold: >30 matched normalized lines.

Raw result (command: `python3 tools/dup_report.py --json`):

- functions parsed: **3168** across 148 files (external audit counted 3,330 production functions — consistent)
- hits at threshold: **93**, totaling **~4,660** duplicated lines
- **runtime/, node/, cache/, ring/: ZERO hits** — the common trees are clean; every hit is per-family code under modules/

### Triage legend

| code | decision |
|------|----------|
| C1 | consolidate via W2 item 1 — module-lifecycle library behind the ABI |
| C2 | consolidate via W2 item 2 — serving-adapter template (lifecycle + ONE TP-collective config parser) |
| C3 | consolidate via W2 item 3 — stagepack format/verifier library |
| C4 | kernel-template consolidation candidate — PARKED until W2 lands (one refactor per file per sprint; same math, sibling geometry; recorded, decided) |
| J1 | justified in-tree: per-family validation/reference independence (DRY plan "deliberately not consolidated": control-vs-candidate must stay independent implementations) |
| J2 | reviewed: family-specific logic; no shared extraction identified |

### Duplication triage (every hit decided)

| hit | matched | pair | decision |
|-----|---------|------|----------|
| 1 | 219 | `SparkQwen38MaxModuleKvPrepareFrame` x `SparkQwen4FlashModuleKvPrepareFrame` (qwen38_max_resident_decode_stage_module / qwen4_flash_resident_decode_stage_module) | **C1** consolidate -> W2 item 1 (module-lifecycle library behind the ABI) |
| 2 | 187 | `Glm52LayerMoe` x `Glm5NextLayerMoe` (layer.cuh / layer.cuh) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 3 | 168 | `SparkQwen38AttnDecodeKernel` x `SparkQwen4FlashAttnDecodeKernel` (qwen38_max_resident_decode_stage_cudau / qwen4_flash_resident_decode_stage_cudau) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 4 | 118 | `SparkGlm52ModuleInitializeTpCollective` x `SparkGlm5NextModuleInitializeTpCollective` (glm52_resident_decode_stage_module / glm5_next_resident_decode_stage_module) | **C2** consolidate -> W2 item 2 (serving-adapter template: lifecycle + TP-collective config parser) |
| 5 | 118 | `SparkGlm52ServingLoadTpCollective` x `SparkGlm5NextServingLoadTpCollective` (glm52_serving_adapter / glm5_next_serving_adapter) | **C2** consolidate -> W2 item 2 (serving-adapter template: lifecycle + TP-collective config parser) |
| 6 | 114 | `SparkGlm52TpChainAdvance` x `SparkGlm5NextTpChainAdvance` (glm52_resident_decode_stage_module / glm5_next_resident_decode_stage_module) | **C1** consolidate -> W2 item 1 (module-lifecycle library behind the ABI) |
| 7 | 107 | `SparkQwen38MaxServingBuildFrame` x `SparkQwen4FlashServingBuildFrame` (qwen38_max_serving_adapter / qwen4_flash_serving_adapter) | **C2** consolidate -> W2 item 2 (serving-adapter template: lifecycle + TP-collective config parser) |
| 8 | 91 | `SparkQwen38AttnDecodeKernel` x `SparkQwen4FlashAttnDecodeKernel` (qwen38_max_resident_decode_stage_cudau / qwen4_flash_resident_decode_stage_cudau) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 9 | 83 | `SparkQwen38_27bAttnDecodeKernel` x `SparkQwen38AttnDecodeKernel` (qwen38_27b_resident_decode_stage_cudau / qwen38_max_resident_decode_stage_cudau) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 10 | 83 | `SparkQwen38_27bAttnDecodeKernel` x `SparkQwen4FlashAttnDecodeKernel` (qwen38_27b_resident_decode_stage_cudau / qwen4_flash_resident_decode_stage_cudau) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 11 | 74 | `SparkQwen38MaxModuleAllocateSlot` x `SparkQwen4FlashModuleAllocateSlot` (qwen38_max_resident_decode_stage_module / qwen4_flash_resident_decode_stage_module) | **C1** consolidate -> W2 item 1 (module-lifecycle library behind the ABI) |
| 12 | 70 | `SparkGlm52ExecuteBatch` x `SparkGlm5NextExecuteBatch` (glm52_resident_decode_stage_module / glm5_next_resident_decode_stage_module) | **C1** consolidate -> W2 item 1 (module-lifecycle library behind the ABI) |
| 13 | 69 | `SparkQwen38MaxModuleInitializeTpCollective` x `SparkQwen4FlashModuleInitializeTpCollective` (qwen38_max_resident_decode_stage_module / qwen4_flash_resident_decode_stage_module) | **C2** consolidate -> W2 item 2 (serving-adapter template: lifecycle + TP-collective config parser) |
| 14 | 68 | `SparkQwen38_27bValCheckAttention` x `SparkQwen4FlashValCheckAttention` (qwen38_27b_resident_decode_stage_cuda_validation / qwen4_flash_resident_decode_stage_cuda_validation) | **J1** per-family validation/reference independence (DRY plan: deliberately not consolidated) |
| 15 | 68 | `SparkQwen38_27bValModuleExecute` x `SparkQwen4FlashValModuleExecute` (qwen38_27b_resident_decode_stage_cuda_validation / qwen4_flash_resident_decode_stage_cuda_validation) | **J1** per-family validation/reference independence (DRY plan: deliberately not consolidated) |
| 16 | 64 | `SparkQwen38_27bValCheckModule` x `SparkQwen4FlashValCheckModule` (qwen38_27b_resident_decode_stage_cuda_validation / qwen4_flash_resident_decode_stage_cuda_validation) | **J1** per-family validation/reference independence (DRY plan: deliberately not consolidated) |
| 17 | 61 | `SparkQwen38MaxModuleConfigure` x `SparkQwen4FlashModuleConfigure` (qwen38_max_resident_decode_stage_module / qwen4_flash_resident_decode_stage_module) | **C1** consolidate -> W2 item 1 (module-lifecycle library behind the ABI) |
| 18 | 60 | `Glm52LayerAttention` x `Glm5NextLayerAttention` (layer.cuh / layer.cuh) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 19 | 58 | `main` x `main` (qwen38_27b_pack_synthesize / qwen38_max_pack_synthesize) | **J1** per-family validation/reference independence (DRY plan: deliberately not consolidated) |
| 20 | 58 | `main` x `main` (qwen38_27b_pack_synthesize / qwen4_flash_pack_synthesize) | **J1** per-family validation/reference independence (DRY plan: deliberately not consolidated) |
| 21 | 58 | `main` x `main` (qwen38_max_pack_synthesize / qwen4_flash_pack_synthesize) | **J1** per-family validation/reference independence (DRY plan: deliberately not consolidated) |
| 22 | 57 | `SparkGlm52ServingInitialize` x `SparkGlm5NextServingInitialize` (glm52_serving_adapter / glm5_next_serving_adapter) | **C2** consolidate -> W2 item 2 (serving-adapter template: lifecycle + TP-collective config parser) |
| 23 | 55 | `Glm52LayerAttentionBf16Graphed` x `Glm5NextLayerAttentionBf16Graphed` (unityu / unityu) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 24 | 54 | `Glm52LayerDenseMlp` x `Glm5NextLayerDenseMlp` (layer.cuh / layer.cuh) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 25 | 52 | `SparkQwen38_27bAttnDecodeKernel` x `SparkQwen38AttnDecodeKernel` (qwen38_27b_resident_decode_stage_cudau / qwen38_max_resident_decode_stage_cudau) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 26 | 52 | `SparkQwen38_27bAttnDecodeKernel` x `SparkQwen4FlashAttnDecodeKernel` (qwen38_27b_resident_decode_stage_cudau / qwen4_flash_resident_decode_stage_cudau) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 27 | 52 | `SparkQwen38MaxServingInitialize` x `SparkQwen4FlashServingInitialize` (qwen38_max_serving_adapter / qwen4_flash_serving_adapter) | **C2** consolidate -> W2 item 2 (serving-adapter template: lifecycle + TP-collective config parser) |
| 28 | 50 | `SparkGlm52BuildWave` x `SparkGlm5NextBuildWave` (glm52_resident_decode_stage_module / glm5_next_resident_decode_stage_module) | **C1** consolidate -> W2 item 1 (module-lifecycle library behind the ABI) |
| 29 | 50 | `SparkQwen38MaxModuleOpenKvTier` x `SparkQwen4FlashModuleOpenKvTier` (qwen38_max_resident_decode_stage_module / qwen4_flash_resident_decode_stage_module) | **C1** consolidate -> W2 item 1 (module-lifecycle library behind the ABI) |
| 30 | 46 | `SparkDsv4ServingLoadTpCollective` x `SparkGlm52ServingLoadTpCollective` (dsv4_serving_adapter / glm52_serving_adapter) | **C2** consolidate -> W2 item 2 (serving-adapter template: lifecycle + TP-collective config parser) |
| 31 | 46 | `SparkDsv4ServingLoadTpCollective` x `SparkGlm5NextServingLoadTpCollective` (dsv4_serving_adapter / glm5_next_serving_adapter) | **C2** consolidate -> W2 item 2 (serving-adapter template: lifecycle + TP-collective config parser) |
| 32 | 46 | `SparkQwen38_27bValModuleInitialize` x `SparkQwen4FlashValModuleInitialize` (qwen38_27b_resident_decode_stage_cuda_validation / qwen4_flash_resident_decode_stage_cuda_validation) | **J1** per-family validation/reference independence (DRY plan: deliberately not consolidated) |
| 33 | 45 | `SparkGlm52ServingBuildFrame` x `SparkGlm5NextServingBuildFrame` (glm52_serving_adapter / glm5_next_serving_adapter) | **C2** consolidate -> W2 item 2 (serving-adapter template: lifecycle + TP-collective config parser) |
| 34 | 45 | `Glm52LayerAttention` x `Glm5NextLayerAttention` (layer.cuh / layer.cuh) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 35 | 44 | `SparkQwen38GateSelectKernel` x `SparkQwen4FlashGateSelectKernel` (qwen38_max_resident_decode_stage_cudau / qwen4_flash_resident_decode_stage_cudau) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 36 | 44 | `SparkDsv4ServingLoadTpCollective` x `SparkGlm52ServingLoadTpCollective` (dsv4_serving_adapter / glm52_serving_adapter) | **C2** consolidate -> W2 item 2 (serving-adapter template: lifecycle + TP-collective config parser) |
| 37 | 44 | `SparkDsv4ServingLoadTpCollective` x `SparkGlm5NextServingLoadTpCollective` (dsv4_serving_adapter / glm5_next_serving_adapter) | **C2** consolidate -> W2 item 2 (serving-adapter template: lifecycle + TP-collective config parser) |
| 38 | 44 | `SparkGlm52ServingLoadConfiguration` x `SparkGlm5NextServingLoadConfiguration` (glm52_serving_adapter / glm5_next_serving_adapter) | **C2** consolidate -> W2 item 2 (serving-adapter template: lifecycle + TP-collective config parser) |
| 39 | 44 | `SparkQwen38ChunkPrepareKernel` x `SparkQwen4FlashChunkPrepareKernel` (qwen38_max_resident_decode_stage_cudau / qwen4_flash_resident_decode_stage_cudau) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 40 | 43 | `SparkQwen38_27bValDeviceSetup` x `SparkQwen4FlashValDeviceSetup` (qwen38_27b_resident_decode_stage_cuda_validation / qwen4_flash_resident_decode_stage_cuda_validation) | **J1** per-family validation/reference independence (DRY plan: deliberately not consolidated) |
| 41 | 42 | `SparkGlm52InitializeState` x `SparkGlm5NextInitializeState` (glm52_resident_decode_stage_module / glm5_next_resident_decode_stage_module) | **C1** consolidate -> W2 item 1 (module-lifecycle library behind the ABI) |
| 42 | 40 | `SparkQwen38MaxServingSubmit` x `SparkQwen4FlashServingSubmit` (qwen38_max_serving_adapter / qwen4_flash_serving_adapter) | **C2** consolidate -> W2 item 2 (serving-adapter template: lifecycle + TP-collective config parser) |
| 43 | 40 | `SparkQwen38_27bSynthesizeBuildDirectory` x `SparkQwen38MaxSynthesizeBuildDirectory` (qwen38_27b_pack_synthesize / qwen38_max_pack_synthesize) | **J1** per-family validation/reference independence (DRY plan: deliberately not consolidated) |
| 44 | 40 | `SparkQwen38_27bSynthesizeBuildDirectory` x `SparkQwen4FlashSynthesizeBuildDirectory` (qwen38_27b_pack_synthesize / qwen4_flash_pack_synthesize) | **J1** per-family validation/reference independence (DRY plan: deliberately not consolidated) |
| 45 | 40 | `SparkQwen38MaxSynthesizeBuildDirectory` x `SparkQwen4FlashSynthesizeBuildDirectory` (qwen38_max_pack_synthesize / qwen4_flash_pack_synthesize) | **J1** per-family validation/reference independence (DRY plan: deliberately not consolidated) |
| 46 | 39 | `SparkGlm52CompleteAsync` x `SparkGlm5NextCompleteAsync` (glm52_resident_decode_stage_module / glm5_next_resident_decode_stage_module) | **C1** consolidate -> W2 item 1 (module-lifecycle library behind the ABI) |
| 47 | 39 | `SparkQwen38_27bValCheckConv` x `SparkQwen4FlashValCheckConv` (qwen38_27b_resident_decode_stage_cuda_validation / qwen4_flash_resident_decode_stage_cuda_validation) | **J1** per-family validation/reference independence (DRY plan: deliberately not consolidated) |
| 48 | 38 | `SparkGlm52ValidateSequenceContinuity` x `SparkGlm5NextValidateSequenceContinuity` (glm52_resident_decode_stage_module / glm5_next_resident_decode_stage_module) | **C1** consolidate -> W2 item 1 (module-lifecycle library behind the ABI) |
| 49 | 38 | `SparkGlm52ValidateFrame` x `SparkGlm5NextValidateFrame` (glm52_resident_decode_stage_module / glm5_next_resident_decode_stage_module) | **C1** consolidate -> W2 item 1 (module-lifecycle library behind the ABI) |
| 50 | 38 | `SparkQwen38_27bValCheckGdnStep` x `SparkQwen4FlashValCheckGdnStep` (qwen38_27b_resident_decode_stage_cuda_validation / qwen4_flash_resident_decode_stage_cuda_validation) | **J1** per-family validation/reference independence (DRY plan: deliberately not consolidated) |
| 51 | 37 | `Glm52LayerIndexer` x `Glm5NextLayerIndexer` (layer.cuh / layer.cuh) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 52 | 37 | `SparkQwen38MaxStagePackShapeGdn` x `SparkQwen4FlashStagePackShapeGdn` (qwen38_max_stagepack_format / qwen4_flash_stagepack_format) | **C3** consolidate -> W2 item 3 (stagepack format/verifier library) |
| 53 | 37 | `SparkQwen38_27bRefAttention` x `SparkQwen4FlashRefAttention` (qwen38_27b_reference / qwen4_flash_reference) | **J1** per-family validation/reference independence (DRY plan: deliberately not consolidated) |
| 54 | 36 | `SparkQwen38AttnPrepareKernel` x `SparkQwen4FlashAttnPrepareKernel` (qwen38_max_resident_decode_stage_cudau / qwen4_flash_resident_decode_stage_cudau) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 55 | 36 | `SparkDsv4ServingLoadTpRailHosts` x `SparkGlm52ServingLoadTpRailHosts` (dsv4_serving_adapter / glm52_serving_adapter) | **C2** consolidate -> W2 item 2 (serving-adapter template: lifecycle + TP-collective config parser) |
| 56 | 36 | `SparkDsv4ServingLoadTpRailHosts` x `SparkGlm5NextServingLoadTpRailHosts` (dsv4_serving_adapter / glm5_next_serving_adapter) | **C2** consolidate -> W2 item 2 (serving-adapter template: lifecycle + TP-collective config parser) |
| 57 | 36 | `SparkGlm52ServingLoadTpRailHosts` x `SparkGlm5NextServingLoadTpRailHosts` (glm52_serving_adapter / glm5_next_serving_adapter) | **C2** consolidate -> W2 item 2 (serving-adapter template: lifecycle + TP-collective config parser) |
| 58 | 36 | `SparkQwen38MaxServingValidateRowOrder` x `SparkQwen4FlashServingValidateRowOrder` (qwen38_max_serving_adapter / qwen4_flash_serving_adapter) | **C2** consolidate -> W2 item 2 (serving-adapter template: lifecycle + TP-collective config parser) |
| 59 | 36 | `SparkQwen38_27bChunkQkDecayKernel` x `SparkQwen38ChunkQkDecayKernel` (qwen38_27b_resident_decode_stage_cudau / qwen38_max_resident_decode_stage_cudau) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 60 | 36 | `SparkQwen38_27bChunkQkDecayKernel` x `SparkQwen4FlashChunkQkDecayKernel` (qwen38_27b_resident_decode_stage_cudau / qwen4_flash_resident_decode_stage_cudau) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 61 | 36 | `SparkQwen38ChunkQkDecayKernel` x `SparkQwen4FlashChunkQkDecayKernel` (qwen38_max_resident_decode_stage_cudau / qwen4_flash_resident_decode_stage_cudau) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 62 | 36 | `SparkQwen38MaxStagePackShapeEveryLayer` x `SparkQwen4FlashStagePackShapeEveryLayer` (qwen38_max_stagepack_format / qwen4_flash_stagepack_format) | **C3** consolidate -> W2 item 3 (stagepack format/verifier library) |
| 63 | 35 | `SparkQwen38_27bServingReservePending` x `SparkQwen38MaxServingReservePending` (qwen38_27b_serving_adapter / qwen38_max_serving_adapter) | **C2** consolidate -> W2 item 2 (serving-adapter template: lifecycle + TP-collective config parser) |
| 64 | 35 | `SparkQwen38_27bServingReservePending` x `SparkQwen4FlashServingReservePending` (qwen38_27b_serving_adapter / qwen4_flash_serving_adapter) | **C2** consolidate -> W2 item 2 (serving-adapter template: lifecycle + TP-collective config parser) |
| 65 | 35 | `SparkQwen38MaxServingReservePending` x `SparkQwen4FlashServingReservePending` (qwen38_max_serving_adapter / qwen4_flash_serving_adapter) | **C2** consolidate -> W2 item 2 (serving-adapter template: lifecycle + TP-collective config parser) |
| 66 | 35 | `SparkGlm52ServingLoadDriver` x `SparkGlm5NextServingLoadDriver` (glm52_serving_adapter / glm5_next_serving_adapter) | **C2** consolidate -> W2 item 2 (serving-adapter template: lifecycle + TP-collective config parser) |
| 67 | 35 | `Glm52Head` x `Glm5NextHead` (layer.cuh / layer.cuh) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 68 | 35 | `SparkQwen38_27bRefChunkStep` x `SparkQwen4FlashRefChunkStep` (qwen38_27b_reference / qwen4_flash_reference) | **J1** per-family validation/reference independence (DRY plan: deliberately not consolidated) |
| 69 | 35 | `SparkQwen38_27bValCheckModule` x `SparkQwen4FlashValCheckModule` (qwen38_27b_resident_decode_stage_cuda_validation / qwen4_flash_resident_decode_stage_cuda_validation) | **J1** per-family validation/reference independence (DRY plan: deliberately not consolidated) |
| 70 | 34 | `SparkQwen38MaxServingLoadConfiguration` x `SparkQwen4FlashServingLoadConfiguration` (qwen38_max_serving_adapter / qwen4_flash_serving_adapter) | **C2** consolidate -> W2 item 2 (serving-adapter template: lifecycle + TP-collective config parser) |
| 71 | 34 | `SparkGlm52PackLoad` x `SparkGlm5NextPackLoad` (glm52_resident_decode_stage_module / glm5_next_resident_decode_stage_module) | **C1** consolidate -> W2 item 1 (module-lifecycle library behind the ABI) |
| 72 | 34 | `SparkQwen38GdnStepKernel` x `SparkQwen4FlashGdnStepKernel` (qwen38_max_resident_decode_stage_cudau / qwen4_flash_resident_decode_stage_cudau) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 73 | 34 | `SparkQwen38_27bChunkStepKernel` x `SparkQwen38ChunkStepKernel` (qwen38_27b_resident_decode_stage_cudau / qwen38_max_resident_decode_stage_cudau) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 74 | 34 | `SparkQwen38_27bChunkStepKernel` x `SparkQwen4FlashChunkStepKernel` (qwen38_27b_resident_decode_stage_cudau / qwen4_flash_resident_decode_stage_cudau) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 75 | 34 | `SparkQwen38ChunkStepKernel` x `SparkQwen4FlashChunkStepKernel` (qwen38_max_resident_decode_stage_cudau / qwen4_flash_resident_decode_stage_cudau) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 76 | 34 | `SparkQwen38_27bValCheckGdnChunk` x `SparkQwen4FlashValCheckGdnChunk` (qwen38_27b_resident_decode_stage_cuda_validation / qwen4_flash_resident_decode_stage_cuda_validation) | **J1** per-family validation/reference independence (DRY plan: deliberately not consolidated) |
| 77 | 33 | `SparkQwen38_27bServingLoadDriver` x `SparkQwen38MaxServingLoadDriver` (qwen38_27b_serving_adapter / qwen38_max_serving_adapter) | **C2** consolidate -> W2 item 2 (serving-adapter template: lifecycle + TP-collective config parser) |
| 78 | 33 | `SparkQwen38_27bServingLoadDriver` x `SparkQwen4FlashServingLoadDriver` (qwen38_27b_serving_adapter / qwen4_flash_serving_adapter) | **C2** consolidate -> W2 item 2 (serving-adapter template: lifecycle + TP-collective config parser) |
| 79 | 33 | `SparkQwen38MaxServingLoadDriver` x `SparkQwen4FlashServingLoadDriver` (qwen38_max_serving_adapter / qwen4_flash_serving_adapter) | **C2** consolidate -> W2 item 2 (serving-adapter template: lifecycle + TP-collective config parser) |
| 80 | 33 | `SparkQwen38MaxModuleVerifyCoverage` x `SparkQwen4FlashModuleVerifyCoverage` (qwen38_max_resident_decode_stage_module / qwen4_flash_resident_decode_stage_module) | **C1** consolidate -> W2 item 1 (module-lifecycle library behind the ABI) |
| 81 | 33 | `SparkQwen38_27bValAttention` x `SparkQwen4FlashValAttention` (qwen38_27b_resident_decode_stage_cuda_validation / qwen4_flash_resident_decode_stage_cuda_validation) | **J1** per-family validation/reference independence (DRY plan: deliberately not consolidated) |
| 82 | 32 | `SparkQwen38_27bAttnDecodeKernel` x `SparkQwen38AttnDecodeKernel` (qwen38_27b_resident_decode_stage_cudau / qwen38_max_resident_decode_stage_cudau) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 83 | 32 | `SparkQwen38_27bAttnDecodeKernel` x `SparkQwen4FlashAttnDecodeKernel` (qwen38_27b_resident_decode_stage_cudau / qwen4_flash_resident_decode_stage_cudau) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 84 | 32 | `SparkGlm52ServingReservePending` x `SparkGlm5NextServingReservePending` (glm52_serving_adapter / glm5_next_serving_adapter) | **C2** consolidate -> W2 item 2 (serving-adapter template: lifecycle + TP-collective config parser) |
| 85 | 32 | `SparkGlm52KvInitialize` x `SparkGlm5NextKvInitialize` (glm52_resident_decode_stage_module / glm5_next_resident_decode_stage_module) | **C1** consolidate -> W2 item 1 (module-lifecycle library behind the ABI) |
| 86 | 32 | `SparkQwen38_27bValCheckGdnChunk` x `SparkQwen4FlashValCheckGdnChunk` (qwen38_27b_resident_decode_stage_cuda_validation / qwen4_flash_resident_decode_stage_cuda_validation) | **J1** per-family validation/reference independence (DRY plan: deliberately not consolidated) |
| 87 | 31 | `Glm52LaunchBf16Linear` x `Glm5NextLaunchBf16Linear` (layer.cuh / layer.cuh) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 88 | 31 | `SparkQwen38_27bRefGdnRecurrence` x `SparkQwen4FlashRefGdnRecurrence` (qwen38_27b_reference / qwen4_flash_reference) | **J1** per-family validation/reference independence (DRY plan: deliberately not consolidated) |
| 89 | 31 | `SparkQwen38_27bValGdnRecurrence` x `SparkQwen4FlashValGdnRecurrence` (qwen38_27b_resident_decode_stage_cuda_validation / qwen4_flash_resident_decode_stage_cuda_validation) | **J1** per-family validation/reference independence (DRY plan: deliberately not consolidated) |
| 90 | 30 | `SparkQwen38MaxLaunchAttnDecode` x `SparkQwen4FlashLaunchAttnDecode` (qwen38_max_resident_decode_stage_cudau / qwen4_flash_resident_decode_stage_cudau) | **C4** kernel-template consolidation candidate; PARKED until W2 lands (one refactor per file per sprint); same math, sibling geometry |
| 91 | 30 | `SparkGlm52PackValidateEntryGeometry` x `SparkGlm5NextPackValidateEntryGeometry` (glm52_resident_decode_stage_module / glm5_next_resident_decode_stage_module) | **C1** consolidate -> W2 item 1 (module-lifecycle library behind the ABI) |
| 92 | 30 | `SparkQwen38_27bValCheckConv` x `SparkQwen4FlashValCheckConv` (qwen38_27b_resident_decode_stage_cuda_validation / qwen4_flash_resident_decode_stage_cuda_validation) | **J1** per-family validation/reference independence (DRY plan: deliberately not consolidated) |
| 93 | 30 | `SparkQwen38MaxResidentDecodeStageExecute` x `SparkQwen4FlashResidentDecodeStageExecute` (qwen38_max_resident_decode_stage_module / qwen4_flash_resident_decode_stage_module) | **C1** consolidate -> W2 item 1 (module-lifecycle library behind the ABI) |

Decision totals: {'C1': 16, 'C4': 27, 'C2': 26, 'J1': 22, 'C3': 2}


### W2 hand-off list (from the C1/C2/C3 triage)

The detector confirms the W2 plan's targets with exact locations — hand them
this table with the hits above:

1. **Adapter paste (C2, 26 hits)**: the qwen38_max<->qwen4_flash adapter pair
   is the same file modulo family prefix (BuildFrame at *identical line
   numbers* 732-844); glm52<->glm5_next re-paste LoadTpCollective/RailHosts/
   LoadConfiguration/Initialize/ReservePending/LoadDriver/BuildFrame; dsv4
   shares the TP-config trio. This is the ~3,500-line template kill plus the
   ~500-line TP-config parser (DRY plan queued item 4).
2. **Module lifecycle (C1, 16 hits)**: ModuleKvPrepareFrame (219 matched
   lines — the single largest clone), ModuleConfigure, AllocateSlot,
   OpenKvTier, InitializeTpCollective, ExecuteBatch, TpChainAdvance,
   BuildWave, CompleteAsync, ValidateFrame, ValidateSequenceContinuity,
   PackLoad, KvInitialize, InitializeState, VerifyCoverage.
3. **Stagepack shape fns (C3, 2 hits)**: StagePackShapeGdn/ShapeEveryLayer
   qwen38_max<->qwen4_flash identical at same lines.
4. **Kernel template follow-on (C4, 27 hits, PARKED)**: GDN/attention/chunk
   kernels + glm52/glm5_next layer.cuh donor copies (LayerMoe 187 matched
   lines). Same math across sibling geometries; the house already proved the
   pattern (shared Linear kernel, TILE_K=32 fallback). Deliberately parked so
   W2's lifecycle/template pass owns those files first; C4 dies at
   kernel-template consolidation, not before.

## 2. Dead code (item 2) — receipted

Method (compile-and-grep, all five dirs):

1. zero-reference `static` functions in `.c`/`.cu` (internal linkage,
   own-file grep = the only possible callers),
2. extern functions defined in runtime/node/cache/ring with no repo-wide
   user (all file types incl. JSON/Makefile — catches dlsym/string uses),
3. struct fields in runtime/cache/ring/node headers with zero `->field` /
   `.field` accesses repo-wide,
4. tests: AST-parse all `tests/*.py` (syntax rot), cross-check referenced
   repo paths, list unregistered tests,
5. python tools referenced nowhere.

Results — the honest headline is mostly **negative results** (the fleet's
hygiene discipline held):

| finding | evidence | action |
|---------|----------|--------|
| `LmTapPlan.hidden_elements` (ring/sideband.h:89) dead field | zero accesses repo-wide (only candidate of 87 swept fields); added by 54b2ff0 (sideband harvest), never consumed | **DELETED** (commit 498eb41); tests/test_sideband.c builds, 28 checks PASS 0-failing |
| 8 zero-reference statics in `modules/glm5_next_resident_decode_stage/validation/spark_glm5_next_resident_decode_stage_cuda_validation.cu` (ValE2m1Decode, ValPayloadCode, ValCodecCodeMin/Max, ValCodecUsesSignedIntGrid, ValFreeMatrix, ValRunTier1AttentionOnly, ValScaleBufferBytes) | own-file internal-linkage grep = 0 calls; orphaned by 5d8e079 (M3 oracle landing) / 40e885a (M3 GPU tier rework) | **W2 hand-off** (modules/ is their write set); ~70 lines |
| zero-reference statics in runtime/node/cache/ring/src/scheduler/inference | sweep returned none | none needed |
| extern dead functions in runtime/node/cache/ring | sweep returned none (after fixing two parse artifacts) | none needed |
| `tests/test_no_python_in_production.py` is RED on main: `model_contracts/references/modeling_qwen4_exp.py` not whitelisted, while test_code_size.py already excludes `model_contracts/references/` as sanctioned vendored ground truth | run output: `FAIL model_contracts/references/modeling_qwen4_exp.py` | **W3 hand-off** (red python gates are their lane); the whitelist needs the same references/ carve-out |
| 9 tests exist outside Makefile/gates wiring (test_ds4_spark_brickproof, test_dsv4_hc_residual_fusion_source, test_dsv4_indexer_post_fusion_source, test_dsv4_pro_exact32k_stage, test_glm53_contract, test_glm5_next_cuda_validator_tier2_oracle, test_glm5_next_geometry, test_head_host, test_no_python_in_production, plus hand-runnable tests/test_sideband.c) | Makefile/gates.sh grep; all AST-parse clean; glm53/glm5_next/brickproof/exact32k run PASS by hand | coordinator wiring decision (see INTEGRATION REQUEST); not bitrot — deliberate tombstone guards verified present and correct |
| 21 `tools/qwen36_*` files still carry the deprecated qwen36 prefix after the "complete" 169-file rename (33c69b6 renamed the imports but left these filenames) | git show 33c69b6: files Modified, not Renamed; contents are the live DFlash2/W-sweep toolchain for Qwen 3.8 27B | 27B-session owner decision (they are actively used; renaming is cosmetic + import churn) |

## 3. Cyclomatic hotspots (item 3) — tools/complexity_report.py

Lizard-compatible counting (1 + if/case/for/while/catch/&&/||/?:) over the
same extraction. **Validation of the counter: max CCN = 157 =
`SparkQwen38_27bModuleRunDsparkBlockForward`, exactly the external audit's
max-157 figure** (audit commit c635ee8; audit said 625 lines, we measure 619
at ccn 157 — same function, same complexity).

Distribution: 3168 functions, mean 7.99 (audit baseline 7.33 over its
production-only subset — ours includes validation harnesses), P90/P95 in line
with the audit; **151 functions > 25**. Every one carries a disposition:

- **C1/C2/C3** — dies naturally with the W2 consolidation of its file (not executed here: dual-edit of W2's write set)
- **J2** — validation-tier walkers: deliberate per-family independence; per docs/CODEBASE_CLEANUP_PLAN.md these split opportunistically when their family agent touches them
- **J3** — CUDA kernels/launchers: branch structure is the math; the cleanup plan's measured-perf exception class; re-evaluated at kernel-template consolidation
- **P** — named plan below (39 functions in runtime/node/cache/ring + non-W2 paths)

### Cyclomatic hotspots (every function > 25 decided)

| ccn | lines | function | disposition |
|-----|-------|----------|-------------|
| 157 | 619 | `modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_resident_decode_stage_module.c:SparkQwen38_27bModuleRunDsparkBlockForward` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 90 | 178 | `modules/glm52_resident_decode_stage/validation/spark_glm52_resident_decode_stage_cuda_validation.cu:SparkGlm52ValFixtureSetup` | **J2** validation tier walker: deliberate per-family independence; split opportunistically when its family agent touches it (cleanup-plan rule) |
| 87 | 137 | `runtime/pipeline_runtime.c:SparkPipelineRuntimeValidateRankPlan` | **P** named plan (below) |
| 84 | 183 | `modules/glm5_next_resident_decode_stage/source/spark_glm5_next_stagepack_format.h:SparkGlm5NextStagePackExpectedShape` | **C3** dies with W2 item 3 (stagepack library) |
| 80 | 120 | `runtime/model_serving_adapter.c:SparkModelServingAdapterValidateDescriptor` | **C2** dies with W2 item 2 (adapter template) |
| 78 | 350 | `modules/glm52_resident_decode_stage/validation/spark_glm52_resident_decode_stage_cuda_validation.cu:SparkGlm52ValRunDsaTier` | **J2** validation tier walker: deliberate per-family independence; split opportunistically when its family agent touches it (cleanup-plan rule) |
| 75 | 103 | `modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c:SparkDsv4ModuleConfigure` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 73 | 374 | `modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_serving_adapter.c:SparkQwen38_27bServingSubmitSpeculativeDecode` | **C2** dies with W2 item 2 (adapter template) |
| 67 | 279 | `runtime/gemm.cuh:LmGemmLaunchAsymmetric` | **J3** CUDA kernel/dispatch: branch structure is the math (cleanup-plan measured-perf exception class); re-evaluate at kernel-template consolidation |
| 66 | 266 | `modules/qwen38_max_resident_decode_stage/source/spark_qwen38_max_resident_decode_stage_module.c:SparkQwen38MaxModuleKvPrepareFrame` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 66 | 266 | `modules/qwen4_flash_resident_decode_stage/source/spark_qwen4_flash_resident_decode_stage_module.c:SparkQwen4FlashModuleKvPrepareFrame` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 65 | 182 | `modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_resident_decode_stage_module.c:SparkQwen38_27bModuleValidateFrame` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 64 | 202 | `modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_resident_decode_stage_module.c:SparkQwen38_27bModuleRunFrame` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 60 | 40 | `modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_module.c:SparkGlm52ValidateFrame` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 60 | 40 | `modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_module.c:SparkGlm5NextValidateFrame` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 60 | 159 | `runtime/pipeline_runtime.c:SparkPipelineRuntimeBuildTransportedRankPlan` | **P** named plan (below) |
| 58 | 260 | `modules/glm52_resident_decode_stage/validation/spark_glm52_resident_decode_stage_cuda_validation.cu:SparkGlm52ValRunRoutedTier` | **J2** validation tier walker: deliberate per-family independence; split opportunistically when its family agent touches it (cleanup-plan rule) |
| 56 | 52 | `runtime/model_serving_adapter.c:SparkModelServingAdapterValidateSubmission` | **C2** dies with W2 item 2 (adapter template) |
| 54 | 230 | `modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_cuda.cu:SparkDsv4TopKKernel` | **J3** CUDA kernel/dispatch: branch structure is the math (cleanup-plan measured-perf exception class); re-evaluate at kernel-template consolidation |
| 54 | 50 | `runtime/model_resident_deployment.c:SparkModelResidentDeploymentValidateStructure` | **P** named plan (below) |
| 53 | 129 | `modules/glm52_resident_decode_stage/source/spark_glm52_serving_adapter.c:SparkGlm52ServingLoadTpCollective` | **C2** dies with W2 item 2 (adapter template) |
| 53 | 129 | `modules/glm5_next_resident_decode_stage/source/spark_glm5_next_serving_adapter.c:SparkGlm5NextServingLoadTpCollective` | **C2** dies with W2 item 2 (adapter template) |
| 52 | 258 | `modules/glm5_next_resident_decode_stage/validation/spark_glm5_next_resident_decode_stage_cuda_validation.cu:SparkGlm5NextValOracleSelftest` | **J2** validation tier walker: deliberate per-family independence; split opportunistically when its family agent touches it (cleanup-plan rule) |
| 51 | 165 | `modules/glm5_next_resident_decode_stage/tools/glm5_next_pack_synthesize.c:main` | **P** named plan (below) |
| 50 | 219 | `modules/glm5_next_resident_decode_stage/validation/spark_glm5_next_resident_decode_stage_cuda_validation.cu:SparkGlm5NextValKdaToken` | **J2** validation tier walker: deliberate per-family independence; split opportunistically when its family agent touches it (cleanup-plan rule) |
| 49 | 65 | `modules/glm52_resident_decode_stage/source/spark_glm52_stagepack_format.h:SparkGlm52StagePackExpectedShape` | **C3** dies with W2 item 3 (stagepack library) |
| 49 | 52 | `runtime/model_serving_adapter.c:SparkModelServingAdapterValidateRows` | **C2** dies with W2 item 2 (adapter template) |
| 48 | 176 | `modules/glm5_next_resident_decode_stage/validation/spark_glm5_next_resident_decode_stage_cuda_validation.cu:SparkGlm5NextValFixtureComplete` | **J2** validation tier walker: deliberate per-family independence; split opportunistically when its family agent touches it (cleanup-plan rule) |
| 47 | 56 | `modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_module.c:SparkGlm5NextPackAssignLayer` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 47 | 130 | `modules/qwen38_27b_resident_decode_stage/validation/spark_qwen38_27b_resident_decode_stage_cuda_validation.cu:SparkQwen38_27bValCheckAttention` | **J2** validation tier walker: deliberate per-family independence; split opportunistically when its family agent touches it (cleanup-plan rule) |
| 47 | 130 | `modules/qwen4_flash_resident_decode_stage/validation/spark_qwen4_flash_resident_decode_stage_cuda_validation.cu:SparkQwen4FlashValCheckAttention` | **J2** validation tier walker: deliberate per-family independence; split opportunistically when its family agent touches it (cleanup-plan rule) |
| 46 | 54 | `ring/transport/rdma_control.c:SparkHiddenTransportRdmaV4ValidatePeerIdentity` | **P** named plan (below) |
| 45 | 378 | `modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_cuda.cu:SparkDsv4SparseAttnKernel` | **J3** CUDA kernel/dispatch: branch structure is the math (cleanup-plan measured-perf exception class); re-evaluate at kernel-template consolidation |
| 45 | 95 | `modules/dsv4_resident_decode_stage/source/spark_dsv4_serving_adapter.c:SparkDsv4ServingLoadTpCollective` | **C2** dies with W2 item 2 (adapter template) |
| 45 | 110 | `modules/qwen4_flash_resident_decode_stage/source/spark_qwen4_flash_resident_decode_stage_module.c:SparkQwen4FlashModuleAllocateSlot` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 44 | 130 | `modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_module.c:SparkGlm52ModuleInitializeTpCollective` | **C2** dies with W2 item 2 (adapter template) |
| 44 | 130 | `modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_module.c:SparkGlm5NextModuleInitializeTpCollective` | **C2** dies with W2 item 2 (adapter template) |
| 44 | 308 | `runtime/pack/module_library.c:SparkPublishValidatedModule` | **P** named plan (below) |
| 43 | 208 | `modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_tp.c:SparkQwen38_27bTpInitialize` | **P** named plan (below) |
| 43 | 136 | `modules/qwen4_flash_resident_decode_stage/source/spark_qwen4_flash_resident_decode_stage_module.c:SparkQwen4FlashModuleConfigure` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 43 | 207 | `cache/kv_cache.c:SparkKvCacheCalculateJitStageBudget` | **P** named plan (below) |
| 42 | 174 | `modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c:SparkDsv4ModuleInitializeTpCollective` | **C2** dies with W2 item 2 (adapter template) |
| 42 | 186 | `modules/glm52_resident_decode_stage/validation/spark_glm52_resident_decode_stage_cuda_validation.cu:SparkGlm52ValSelftestCodecRoundTrip` | **J2** validation tier walker: deliberate per-family independence; split opportunistically when its family agent touches it (cleanup-plan rule) |
| 42 | 198 | `modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_native_ws.cuh:__launch_bounds__` | **J3** CUDA kernel/dispatch: branch structure is the math (cleanup-plan measured-perf exception class); re-evaluate at kernel-template consolidation |
| 42 | 90 | `modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_resident_decode_stage_module.c:SparkQwen38_27bModuleLoadDsparkPack` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 42 | 99 | `modules/qwen38_max_resident_decode_stage/source/spark_qwen38_max_resident_decode_stage_module.c:SparkQwen38MaxModuleAllocateSlot` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 41 | 61 | `modules/dsv4_resident_decode_stage/source/spark_dsv4_stage_runner.c:SparkDsv4StageRunnerValidateConfiguration` | **P** named plan (below) |
| 41 | 42 | `modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_module.c:SparkGlm52ModuleConfigure` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 41 | 42 | `modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_module.c:SparkGlm5NextModuleConfigure` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 41 | 104 | `modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_resident_decode_stage_module.c:SparkQwen38_27bModuleValidateSpeculation` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 41 | 114 | `runtime/pack/module_library.c:SparkModuleValidatePublishRequest` | **P** named plan (below) |
| 41 | 117 | `node/model_residentd.c:SparkModelResidentdProcessSubmission` | **P** named plan (below) |
| 41 | 219 | `ring/transport/rdma.cu:SparkHiddenSparkHostRdmaInitialize` | **J3** CUDA kernel/dispatch: branch structure is the math (cleanup-plan measured-perf exception class); re-evaluate at kernel-template consolidation |
| 40 | 121 | `modules/qwen38_max_resident_decode_stage/source/spark_qwen38_max_resident_decode_stage_module.c:SparkQwen38MaxModuleConfigure` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 40 | 187 | `ring/transport/tp_device_collective.c:SparkTpDeviceCollectiveSubmitHidden` | **P** named plan (below) |
| 39 | 170 | `modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c:SparkDsv4ModuleContinueHeadMax` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 39 | 92 | `ring/transport/hidden_transport.c:SparkHiddenTransportValidateEndpoint` | **P** named plan (below) |
| 39 | 163 | `ring/transport/rdma.cu:SparkHiddenSparkHostRdmaConnectControl` | **J3** CUDA kernel/dispatch: branch structure is the math (cleanup-plan measured-perf exception class); re-evaluate at kernel-template consolidation |
| 38 | 25 | `modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_module.c:SparkGlm52PackValidateHeader` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 38 | 25 | `modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_module.c:SparkGlm5NextPackValidateHeader` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 38 | 41 | `modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_resident_decode_stage_module.c:SparkQwen38_27bModuleValidateEntry` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 38 | 256 | `runtime/pack/module_library.c:SparkLoadModuleArtifactRecord` | **P** named plan (below) |
| 37 | 164 | `modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c:SparkDsv4ModuleContinueLayers` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 37 | 245 | `modules/glm52_resident_decode_stage/source/cuda/layer.cuh:Glm52LayerAttention` | **J3** CUDA kernel/dispatch: branch structure is the math (cleanup-plan measured-perf exception class); re-evaluate at kernel-template consolidation |
| 37 | 270 | `modules/glm52_resident_decode_stage/source/cuda/layer.cuh:Glm52LayerMoe` | **J3** CUDA kernel/dispatch: branch structure is the math (cleanup-plan measured-perf exception class); re-evaluate at kernel-template consolidation |
| 37 | 270 | `modules/glm5_next_resident_decode_stage/source/cuda/layer.cuh:Glm5NextLayerMoe` | **J3** CUDA kernel/dispatch: branch structure is the math (cleanup-plan measured-perf exception class); re-evaluate at kernel-template consolidation |
| 36 | 193 | `modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c:SparkDsv4ModuleRunDsparkDraft` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 36 | 52 | `modules/dsv4_resident_decode_stage/source/spark_dsv4_stage_runner.c:SparkDsv4StageRunnerValidateDispatchShape` | **P** named plan (below) |
| 36 | 18 | `runtime/model_resident_ipc.c:SparkModelResidentIpcValidateHelloAck` | **P** named plan (below) |
| 36 | 36 | `runtime/model_serving_adapter.c:SparkModelServingAdapterFindLastRows` | **C2** dies with W2 item 2 (adapter template) |
| 35 | 40 | `modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c:SparkDsv4ModuleValidateFrameShape` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 35 | 226 | `modules/glm5_next_resident_decode_stage/source/cuda/layer.cuh:Glm5NextLayerAttention` | **J3** CUDA kernel/dispatch: branch structure is the math (cleanup-plan measured-perf exception class); re-evaluate at kernel-template consolidation |
| 35 | 315 | `modules/glm5_next_resident_decode_stage/source/cuda/layer.cuh:Glm5NextLayerKda` | **J3** CUDA kernel/dispatch: branch structure is the math (cleanup-plan measured-perf exception class); re-evaluate at kernel-template consolidation |
| 35 | 205 | `modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_runner.cu:SparkK3StageRunnerInitialize` | **J3** CUDA kernel/dispatch: branch structure is the math (cleanup-plan measured-perf exception class); re-evaluate at kernel-template consolidation |
| 35 | 191 | `modules/k3_resident_decode_stage/source/spark_k3_serving_adapter.c:K3ServingLoadConfiguration` | **C2** dies with W2 item 2 (adapter template) |
| 35 | 48 | `modules/qwen4_flash_resident_decode_stage/source/spark_qwen4_flash_resident_decode_stage_module.c:SparkQwen4FlashModuleValidateEntry` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 34 | 199 | `modules/glm5_next_resident_decode_stage/validation/spark_glm5_next_resident_decode_stage_cuda_validation.cu:main` | **J2** validation tier walker: deliberate per-family independence; split opportunistically when its family agent touches it (cleanup-plan rule) |
| 34 | 361 | `modules/qwen4_flash_resident_decode_stage/source/spark_qwen4_flash_resident_decode_stage_cuda.cu:SparkQwen4FlashAttnDecodeKernel` | **J3** CUDA kernel/dispatch: branch structure is the math (cleanup-plan measured-perf exception class); re-evaluate at kernel-template consolidation |
| 34 | 149 | `modules/qwen4_flash_resident_decode_stage/validation/spark_qwen4_flash_resident_decode_stage_cuda_validation.cu:SparkQwen4FlashValCheckModule` | **J2** validation tier walker: deliberate per-family independence; split opportunistically when its family agent touches it (cleanup-plan rule) |
| 34 | 38 | `runtime/model_batch_engine.c:SparkModelBatchSchedulerChooseWorkKind` | **P** named plan (below) |
| 34 | 82 | `node/model_batch.c:main` | **P** named plan (below) |
| 34 | 123 | `ring/transport/hidden_transport.c:SparkHiddenTransportValidatePacket` | **P** named plan (below) |
| 34 | 37 | `ring/transport/tp_device_collective_nccl.c:SparkTpNcclValidateConfig` | **P** named plan (below) |
| 33 | 68 | `modules/dsv4_resident_decode_stage/source/spark_dsv4_serving_adapter.c:SparkDsv4ServingLoadConfiguration` | **C2** dies with W2 item 2 (adapter template) |
| 33 | 42 | `modules/dsv4_resident_decode_stage/source/spark_dsv4_stagepack_format.h:SparkDsv4StagePackShapeOfLayer` | **C3** dies with W2 item 3 (stagepack library) |
| 33 | 47 | `modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_module.c:SparkGlm5NextAllocateSlotHidden` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 33 | 102 | `modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_resident_decode_stage_cuda.cu:SparkQwen38_27bLaunchLinear` | **J3** CUDA kernel/dispatch: branch structure is the math (cleanup-plan measured-perf exception class); re-evaluate at kernel-template consolidation |
| 33 | 56 | `modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_resident_decode_stage_module.c:SparkQwen38_27bModuleRunGdnLayer` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 33 | 100 | `modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_serving_adapter.c:SparkQwen38_27bServingSubmit` | **C2** dies with W2 item 2 (adapter template) |
| 33 | 42 | `runtime/model_resident_ipc.c:SparkModelResidentIpcEncodeSubmissionKind` | **P** named plan (below) |
| 33 | 127 | `cache/nvme_tier.c:SparkNvmeTierInitialize` | **P** named plan (below) |
| 33 | 108 | `cache/nvme_tier.c:SparkNvmeTierPump` | **P** named plan (below) |
| 32 | 120 | `modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_cuda.cu:SparkK3DispatchBindWeights` | **J3** CUDA kernel/dispatch: branch structure is the math (cleanup-plan measured-perf exception class); re-evaluate at kernel-template consolidation |
| 32 | 339 | `modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_resident_decode_stage_cuda.cu:SparkQwen38_27bAttnDecodeKernel` | **J3** CUDA kernel/dispatch: branch structure is the math (cleanup-plan measured-perf exception class); re-evaluate at kernel-template consolidation |
| 32 | 96 | `modules/qwen38_27b_resident_decode_stage/validation/spark_qwen38_27b_resident_decode_stage_cuda_validation.cu:SparkQwen38_27bValCheckGdnChunk` | **J2** validation tier walker: deliberate per-family independence; split opportunistically when its family agent touches it (cleanup-plan rule) |
| 32 | 139 | `modules/qwen38_27b_resident_decode_stage/validation/spark_qwen38_27b_resident_decode_stage_cuda_validation.cu:SparkQwen38_27bValCheckModule` | **J2** validation tier walker: deliberate per-family independence; split opportunistically when its family agent touches it (cleanup-plan rule) |
| 32 | 356 | `modules/qwen38_max_resident_decode_stage/source/spark_qwen38_max_resident_decode_stage_cuda.cu:SparkQwen38AttnDecodeKernel` | **J3** CUDA kernel/dispatch: branch structure is the math (cleanup-plan measured-perf exception class); re-evaluate at kernel-template consolidation |
| 32 | 96 | `modules/qwen4_flash_resident_decode_stage/validation/spark_qwen4_flash_resident_decode_stage_cuda_validation.cu:SparkQwen4FlashValCheckGdnChunk` | **J2** validation tier walker: deliberate per-family independence; split opportunistically when its family agent touches it (cleanup-plan rule) |
| 32 | 71 | `runtime/model_runtime.c:SparkModelRuntimeValidateProvider` | **P** named plan (below) |
| 32 | 110 | `node/model_residentd.c:SparkModelResidentdProgressRoute` | **P** named plan (below) |
| 32 | 69 | `cache/kv_cache.c:SparkKvCacheAsyncPrefetchBackendValidateConfiguration` | **P** named plan (below) |
| 31 | 61 | `runtime/json.c:SparkJsonValidateNumber` | **P** named plan (below) |
| 30 | 61 | `modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c:SparkDsv4ModuleAllocateSlotTail` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 30 | 85 | `modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c:SparkDsv4ModuleRunFrame` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 30 | 148 | `modules/glm52_resident_decode_stage/validation/spark_glm52_resident_decode_stage_cuda_validation.cu:SparkGlm52ValOracleAttentionChunk` | **J2** validation tier walker: deliberate per-family independence; split opportunistically when its family agent touches it (cleanup-plan rule) |
| 30 | 40 | `modules/glm5_next_resident_decode_stage/validation/spark_glm5_next_resident_decode_stage_cuda_validation.cu:SparkGlm5NextValFixtureBuild` | **J2** validation tier walker: deliberate per-family independence; split opportunistically when its family agent touches it (cleanup-plan rule) |
| 30 | 144 | `modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_resident_decode_stage_cuda.cu:__launch_bounds__` | **J3** CUDA kernel/dispatch: branch structure is the math (cleanup-plan measured-perf exception class); re-evaluate at kernel-template consolidation |
| 30 | 58 | `modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_resident_decode_stage_module.c:SparkQwen38_27bModuleAllocateSlotControl` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 30 | 57 | `node/model_residentd.c:SparkModelResidentdCompletion` | **P** named plan (below) |
| 30 | 67 | `ring/transport/tp_device_collective.c:SparkTpDeviceCollectiveValidateAlgorithms` | **P** named plan (below) |
| 29 | 206 | `modules/glm5_next_resident_decode_stage/source/cuda/layer.cuh:Glm5NextLayerIndexer` | **J3** CUDA kernel/dispatch: branch structure is the math (cleanup-plan measured-perf exception class); re-evaluate at kernel-template consolidation |
| 29 | 142 | `modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_serving_adapter.c:SparkQwen38_27bServingBuildFrame` | **C2** dies with W2 item 2 (adapter template) |
| 29 | 42 | `runtime/model_serving_adapter.c:SparkModelServingAdapterValidateStageCompletion` | **C2** dies with W2 item 2 (adapter template) |
| 28 | 46 | `modules/glm52_resident_decode_stage/source/spark_glm52_serving_adapter.c:SparkGlm52ServingLoadConfiguration` | **C2** dies with W2 item 2 (adapter template) |
| 28 | 46 | `modules/glm5_next_resident_decode_stage/source/spark_glm5_next_serving_adapter.c:SparkGlm5NextServingLoadConfiguration` | **C2** dies with W2 item 2 (adapter template) |
| 28 | 91 | `modules/qwen38_27b_resident_decode_stage/validation/spark_qwen38_27b_resident_decode_stage_cuda_validation.cu:SparkQwen38_27bValCheckConv` | **J2** validation tier walker: deliberate per-family independence; split opportunistically when its family agent touches it (cleanup-plan rule) |
| 28 | 30 | `modules/qwen38_max_resident_decode_stage/source/spark_qwen38_max_resident_decode_stage_module.c:SparkQwen38MaxModuleValidateEntry` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 28 | 79 | `modules/qwen4_flash_resident_decode_stage/source/spark_qwen4_flash_resident_decode_stage_module.c:SparkQwen4FlashModuleRunDecode` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 28 | 91 | `modules/qwen4_flash_resident_decode_stage/validation/spark_qwen4_flash_resident_decode_stage_cuda_validation.cu:SparkQwen4FlashValCheckConv` | **J2** validation tier walker: deliberate per-family independence; split opportunistically when its family agent touches it (cleanup-plan rule) |
| 28 | 47 | `node/model_residentd.c:SparkModelResidentdRun` | **P** named plan (below) |
| 28 | 82 | `ring/transport/rdma.cu:SparkHiddenSparkHostRdmaApplyDoorbellCompletion` | **J3** CUDA kernel/dispatch: branch structure is the math (cleanup-plan measured-perf exception class); re-evaluate at kernel-template consolidation |
| 27 | 183 | `modules/dsv4_resident_decode_stage/source/spark_dsv4_dspark_kernels.cuh:SparkDsv4DsparkAttentionKernel` | **J3** CUDA kernel/dispatch: branch structure is the math (cleanup-plan measured-perf exception class); re-evaluate at kernel-template consolidation |
| 27 | 66 | `modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c:SparkDsv4ModuleResolvedShape` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 27 | 92 | `modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c:SparkDsv4ModuleExecuteFrame` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 27 | 133 | `modules/glm52_dspark_draft_backend/source/spark_glm52_dspark_draft_backend.cu:SparkGlm52DsparkAllocateWorkspaces` | **J3** CUDA kernel/dispatch: branch structure is the math (cleanup-plan measured-perf exception class); re-evaluate at kernel-template consolidation |
| 27 | 182 | `modules/glm52_resident_decode_stage/source/cuda/layer.cuh:Glm52LayerIndexer` | **J3** CUDA kernel/dispatch: branch structure is the math (cleanup-plan measured-perf exception class); re-evaluate at kernel-template consolidation |
| 27 | 80 | `modules/glm5_next_resident_decode_stage/validation/spark_glm5_next_resident_decode_stage_cuda_validation.cu:SparkGlm5NextValHcSite` | **J2** validation tier walker: deliberate per-family independence; split opportunistically when its family agent touches it (cleanup-plan rule) |
| 27 | 194 | `modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_resident_decode_stage_module.c:SparkQwen38_27bResidentDecodeStageExecute` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 27 | 147 | `modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_resident_decode_stage_module.c:SparkQwen38_27bResidentDecodeStageInitialize` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 27 | 55 | `modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_stagepack_format.h:SparkQwen38_27bStagePackCompareGeometry` | **C3** dies with W2 item 3 (stagepack library) |
| 27 | 13 | `modules/qwen38_max_resident_decode_stage/source/spark_qwen38_max_stagepack_format.h:SparkQwen38MaxStagePackHeaderMatches` | **C3** dies with W2 item 3 (stagepack library) |
| 27 | 13 | `modules/qwen4_flash_resident_decode_stage/source/spark_qwen4_flash_stagepack_format.h:SparkQwen4FlashStagePackHeaderMatches` | **C3** dies with W2 item 3 (stagepack library) |
| 27 | 18 | `runtime/model_serving_adapter.c:SparkModelServingAdapterValidateCompletion` | **C2** dies with W2 item 2 (adapter template) |
| 26 | 73 | `modules/dsv4_resident_decode_stage/source/spark_dsv4_paged_cache.c:SparkDsv4PagedCachePrepareLane` | **P** named plan (below) |
| 26 | 35 | `modules/dsv4_resident_decode_stage/source/spark_dsv4_stagepack_format.h:SparkDsv4StagePackResolvedShape` | **C3** dies with W2 item 3 (stagepack library) |
| 26 | 38 | `modules/dsv4_resident_decode_stage/validation/spark_dsv4_resident_decode_stage_cuda_validation.cu:SparkDsv4ValidationHeadRunCase` | **J2** validation tier walker: deliberate per-family independence; split opportunistically when its family agent touches it (cleanup-plan rule) |
| 26 | 32 | `modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_module.c:SparkGlm52PackValidateEntryGeometry` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 26 | 125 | `modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_module.c:SparkGlm52TpChainAdvance` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 26 | 32 | `modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_module.c:SparkGlm5NextPackValidateEntryGeometry` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 26 | 125 | `modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_module.c:SparkGlm5NextTpChainAdvance` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 26 | 54 | `modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_resident_decode_stage_module.c:SparkQwen38_27bModuleConfigure` | **C1** dies with / reshaped by W2 item 1 (module-lifecycle library) |
| 26 | 54 | `modules/qwen38_27b_resident_decode_stage/validation/spark_qwen38_27b_resident_decode_stage_cuda_validation.cu:SparkQwen38_27bValDeviceSetup` | **J2** validation tier walker: deliberate per-family independence; split opportunistically when its family agent touches it (cleanup-plan rule) |
| 26 | 54 | `modules/qwen4_flash_resident_decode_stage/validation/spark_qwen4_flash_resident_decode_stage_cuda_validation.cu:SparkQwen4FlashValDeviceSetup` | **J2** validation tier walker: deliberate per-family independence; split opportunistically when its family agent touches it (cleanup-plan rule) |
| 26 | 159 | `runtime/pack/driver_compiler.c:SparkCompileModelPackage` | **P** named plan (below) |
| 26 | 58 | `node/model_residentd.c:SparkModelResidentdBindRoute` | **P** named plan (below) |
| 26 | 57 | `node/model_residentd.c:SparkModelResidentdProcessDecision` | **P** named plan (below) |
| 26 | 68 | `cache/prefix_cache.c:SparkPrefixCacheFindEntry` | **P** named plan (below) |
| 26 | 96 | `cache/prefix_cache.c:SparkPrefixCacheInitialize` | **P** named plan (below) |
| 26 | 175 | `cache/prefix_cache.c:SparkPrefixCacheReservePromptInternal` | **P** named plan (below) |
| 26 | 51 | `ring/transport/tp_device_collective.c:SparkTpDeviceCollectiveValidateConfig` | **P** named plan (below) |
| 26 | 213 | `ring/transport/tp_device_collective.c:SparkTpDeviceCollectiveCreate` | **P** named plan (below) |


### Named simplification plans (P1-P12; filed, not executed — none are in
W2's dying path)

- **P1 — SparkQwen38_27bModuleRunDsparkBlockForward (CCN 157, 619 lines)** —
  the max-157 entry, its own plan. The DFlash2 block forward runs 5 sublayers
  x 8 rows inline in one function: conv prepare, attention q/k/v, attention
  out + head-select, Markov/argmax selection, tap store. Plan: extract one
  `SparkQwen38_27bDsparkRunSublayer()` per sublayer (each owning its row loop
  and asserts), leaving the forward as a ~60-line sequencer. Each piece lands
  under CCN 30. Guard: no numerical change — the parity oracles
  (tools/qwen36_dflash2_* reference) pin every sublayer. Sequenced with W2's
  pass over that file (their write set), not before.
- **P2 — runtime/pipeline_runtime.c SparkPipelineRuntimeValidateRankPlan (87)
  + BuildTransportedRankPlan (60)**: replace the if-ladders with a
  {field, predicate, error} table walked by one loop; builder splits into
  per-rank slice/peer/transport phases.
- **P3 — runtime/pack/module_library.c SparkPublishValidatedModule (44),
  SparkModuleValidatePublishRequest (41), SparkLoadModuleArtifactRecord (38)**:
  one record-schema descriptor (field list + validators) shared by all three.
- **P4 — ring/transport config/identity validators:
  SparkHiddenTransportRdmaV4ValidatePeerIdentity (46),
  SparkHiddenTransportValidateEndpoint (39), ValidatePacket (34),
  SparkTpNcclValidateConfig (34), TpDeviceCollectiveValidateAlgorithms (30),
  ValidateConfig (26)**: shared field-walk validator (the runtime-side twin
  of W2's TP-collective config parser).
- **P5 — node/model_residentd.c ProcessSubmission (41), ProgressRoute (32),
  Completion (30), Run (28), BindRoute (26), ProcessDecision (26);
  node/model_batch.c main (34); runtime/model_batch_engine.c
  ChooseWorkKind (34)**: submission-kind dispatch table + phase-split main.
- **P6 — cache/kv_cache.c SparkKvCacheCalculateJitStageBudget (43),
  AsyncPrefetchBackendValidateConfiguration (32); cache/nvme_tier.c
  Initialize (33), Pump (33)**: tier-budget policy struct + one budget
  function per tier; pump split into complete/issue/advance phases.
- **P7 — cache/prefix_cache.c FindEntry (26), Initialize (26),
  ReservePromptInternal (26)**: entry-lifecycle state machine split
  (probe/insert/evict).
- **P8 — runtime/model_resident_deployment.c ValidateStructure (54),
  runtime/model_resident_ipc.c ValidateHelloAck (36),
  EncodeSubmissionKind (33), runtime/model_runtime.c ValidateProvider (32)**:
  schema-field tables.
- **P9 — runtime/json.c SparkJsonValidateNumber (31)**: split by grammar
  production (int/frac/exp) — three small pure functions.
- **P10 — runtime/pack/driver_compiler.c SparkCompileModelPackage (26)**:
  option-parse / generate / compile / link phase functions.
- **P11 — ring/transport/tp_device_collective.c SubmitHidden (40),
  Create (26)**: op-descriptor table (ties into P4's validator).
- **P12 — family tools**: modules/qwen38_27b .../spark_qwen38_27b_tp.c
  TpInitialize (43), glm5_next_pack_synthesize main (51),
  dsv4_stage_runner ValidateConfiguration (41)/ValidateDispatchShape (36),
  dsv4 paged_cache PrepareLane (26): config-schema split mirroring P2;
  dsv4 entries sequenced after W2's dsv4 pass.

No new function >25 without in-commit justification: enforced mechanically by
`python3 tools/complexity_report.py --threshold 25` at merge time
(coordinator gate wiring in the INTEGRATION REQUEST).

## 4. Token-saver generators (item 4) — tools/gen_geometry_header.py

Contract -> geometry header, house pattern (generate_dsv4_contracts/
generate_k3_contract): every geometry number read from the contract by strict
indexing (missing key = error, never a default); prose + derived-macro
structure template-owned; non-geometry constants recorded in FAMILY_POLICY
with owners.

| family | contract | proof | gate |
|--------|----------|-------|------|
| qwen38_27b | qwen38_27b_authoritative.json | **`--check` byte-identical to the hand-written header** (the required proof; no cutover needed — the contract fully drives the file) | test_qwen38_27b_bf16_contract PASS |
| glm5_next | glm53_flash_authoritative.json | cut over after diff review — 3 diffs: +generated banner (5), **removed a duplicated KDA_LOW_RANK_GATE_BOTTLENECK define**, INDEX_HEAD_WEIGHT_SCALE corrected to the exact `32**-0.5` float (hand literal was the 1-ulp-off `1/sqrt` path; the macro's own comment names `32**-0.5`) | test_glm5_next_geometry PASS |
| qwen4_flash | qwen4_flash_authoritative.json | cut over after diff review — banner only; values/prose/macro set identical | test_qwen4_flash_model_header PASS (36 bindings + 8 composed) |
| qwen38_27b adapter-descriptor blob | same | `--emit-adapter-constants` -> model-families/qwen38_27b/.../spark_qwen38_27b_serving_constants.h (37 lines): values identical to the constants pasted in the family serving adapter (only cosmetic diff: `", "` separators vs the adapter's packed `{64u,64u,...}` list); revision string now contract-fed | --check byte-stable; cc -fsyntax-only PASS |

All four `--check` runs byte-stable post-cutover; all generated headers
compile together (`cc -fsyntax-only`). The adapter include-cutover (dropping
the pasted block in modules/) is W2's edit — the blob sits ready in
model-families/.

Token math for the operator ask: a new family's geometry header + adapter
constants blob is now `python3 tools/gen_geometry_header.py --family X`
instead of ~300 hand-written lines; contract edits propagate by regeneration,
not by four hand-edits that drift (the qwen38_27b byte-identity is the
no-drift proof).

## 5. Gates rerun (raw)

- `python3 tests/test_code_size.py` -> "the authored codebase did not grow" (ceiling 210335 exact; reconciliation comment in the file: +1232 detection/generation tooling, +10 banners, -1 dead field, -1 duplicate define; base c18eabe measured exactly 209095 — clean)
- `python3 tests/test_glm5_next_geometry.py` -> PASS (geometry + name mapping round-trip)
- `python3 tests/test_qwen4_flash_model_header.py` -> PASS (36 bindings + 8 composed)
- `python3 tests/test_qwen38_27b_bf16_contract.py` -> PASS
- `python3 tests/test_glm53_contract.py` -> PASS (53 checks)
- `python3 tests/test_model_families.py` -> 5 families checked, 0 problems
- `cc -o /tmp/test_sideband tests/test_sideband.c && run` -> PASS (0 failing)
- note: origin/main has advanced past this lane's base (c18eabe is an
  ancestor of e8d4770; coordinator's main checkout already measures a higher
  ceiling). Merge accordingly; the ratchet must be re-run after any conflict
  resolution per the merge gates.

## 6. INTEGRATION REQUEST

1. **Gate wiring (Makefile / tools/gates.sh)** — not edited per lane rules:
   register `tools/dup_report.py --min-lines 30` (non-zero hits = fail or
   warn at merge), `tools/complexity_report.py --threshold 25` (changed-file
   CCN gate), and `tools/gen_geometry_header.py --family {qwen38_27b,glm5_next,qwen4_flash} --check`
   + `--emit-adapter-constants --check` (drift gate) — plus the 9 unregistered
   tests of section 2 if the coordinator wants them enforced.
2. **W2 (modules/ write set)**: the hand-off list in section 1; the 8 dead
   validator statics in section 2; the serving-constants include cutover.
3. **W3 (red gates)**: test_no_python_in_production whitelist needs the
   model_contracts/references/ carve-out (mirrors test_code_size's
   EXCLUDED_COMPONENTS decision).
4. **27B session**: qwen36_* filename rename completion (21 files; contents
   live; imports already point at qwen38_27b_*).

## 7. Next (if the lane continues)

- Extend the generator: dsv4_flash/dsv4_pro/k3/glm52 geometry headers onto
  the same contract->header path (dsv4 already generates
  inference-side config; unify at the family header), then adapter-constants
  blobs for the remaining four families.
- After W2 lands: re-run dup_report — C1/C2/C3 clusters must show ZERO hits;
  C4 (kernel template) becomes the next consolidation with the same
  detector as its gate.
- Value-metric projection: W2's consolidation of the C1+C2 clusters retires
  ~3,500 pasted adapter lines + ~500 TP-config lines against this lane's
  +1,232 detection/generation lines — the amortizer pays for itself at the
  first family port.
