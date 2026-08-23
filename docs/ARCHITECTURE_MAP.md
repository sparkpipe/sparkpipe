# SparkPipe Architecture Map (living document)

Built per coordinator directive: understand ALL aspects to find redundant/conflicting
modules and shared-library opportunities (compile-time-parameterized so every
instantiation stays fully optimal). Status: SURVEY PASS 1 (structure + first overlaps).
Last updated: 2026-08-23 ~05:00 UTC.

## 1. Model drivers (modules/*)

| Driver | Lines | Serves | State |
|---|---|---|---|
| qwen36_resident_decode_stage | 17,975 | Qwen3.8-27B (dense, DFlash2 spec) | most active: full-sequence N+8-key context, prefix-cache port in progress; name is LEGACY (dir says qwen36, model is Qwen3.8) |
| dsv4_resident_decode_stage | 16,017 | DeepSeek V4 Flash/Pro (MoE) | flash recovery in progress; pro audit round |
| glm52_resident_decode_stage (+dspark backend) | 6,532+2,986 | GLM5.2 (MoE, MLA) | diamond-pass round; PP7 target |
| qwen38_resident_decode_stage | 6,169 | Qwen3.8-Max | never hardware-tested; bring-up plan delivered |
| k3_resident_decode_stage | 3,394 | K3 (TP16 MXFP4) | #667 closeout |

## 2. Shared layers

### runtime/ (~13.8k lines)
model_batch_engine.c 2153 (the submission engine), stage_module_common.c 1137
(the module-host contract: alloc ledger, env parsing, device allocate/release),
runtime_completion.c 961, model_resident_client.c 956, json.c 931,
model_pipeline_client.c 924, prefix_cache.c+h 704+249 (NEW general core:
block tables/LCP/eviction - model-independent), work_transaction.c 659,
model_serving_adapter.c 655 (generic parts), model_resident_ipc.c 654,
pipeline_runtime.c 569, filesystem.c 525, model_resident_deployment.c 491,
arena.h 388, launch.h 317, workspace.h 297, gemm_descriptor_cache.h,
spark_stagepack_reader.h 200 (NEW shared reader), spark_hybrid_stagepack_core.h 132,
net.h 119, tensor_map.h 116.

### cache/
kv_cache.c, kv_model_table.c, kv_page_cache.c, kv_page_store.c, nvme_tier.c,
prefix_cache.c (KV-BYTE-level cache+hash layer), store/.

### node/
model_batch.c, model_residentd.c, memlink_tool.c (5,119 lines total).

## 3. First overlap finding: TWO prefix-cache layers
runtime/prefix_cache (NEW: model-independent indices/token-ids core, never
touches KV bytes) vs cache/prefix_cache (PRE-EXISTING: KV-byte-level cache +
hash). Both are included by qwen36's adapter/paged_kv. OPEN QUESTION for the
map: do they interlock cleanly (core=indices, cache-layer=bytes) or does LCP
logic exist in both? Owner: pccore dev to confirm the layering; auditor to verify.

## 4. Known cross-driver duplication (from audits so far)
1. Stagepack format/validator family x4 (dsv4 464, qwen36 550, qwen38 432,
   glm52 282 lines): one shared reader + geometry tables removes ~1200-1500.
   Shared reader landed (runtime/spark_stagepack_reader.h); family deletion pending.
2. Adapter speculation policy getenv patterns: collapsed in qwen36 (env-cache);
   other drivers TBD.
3. Bisect-dump boilerplate x9 copies in qwen36: collapsed (-102 ln); other drivers TBD.
4. KV backing fallback shims (glm52 named): deletion queued.
5. VA_TRACE debug fprintf sites x70 uncommitted in runtime/model_serving_adapter.c:
   ownership assigned to dsv4-flash dev.

## 5. Open interface questions
- hwiface v1 frozen (islands E0/L1-L5/F1/F2, R1-R7 rules, C1-C3 classes).
  Qwen27b agent's three naming extensions (dense-FFN row, GDN recurrence as
  L2/L3 state channel, drafter second-pack under F1-stays-whole) granted/adjudicated.
- MERGE-NOTE: main's bonus-fold/multi-block draft-count convention vs unified's
  all-7-drafts + credited-ceiling accounting - unified wins default path;
  main's fold machinery opt-in behind SPARK_QWEN36_DFLASH2_CACHE_PATH=1.

## PASS 2: runtime/stage_module_common.c contract inventory (1137 ln)
The module-host contract EVERY resident driver links. Exported families:
1. **Env/config**: EnvironmentText/Unsigned/Unsigned64 - per-module env parsing with [min,max] clamping.
2. **Device allocation ledger**: DeviceAllocate/Zeroed, RecordAllocation, LedgerRollback, LedgerRelease (cudaMalloc tracking; release frees all). THE allocation path for all drivers' device buffers.
3. **CUDA fork**: CudaForkInitialize/Begin/Join/Destroy (aux-stream parallel legs - e.g., glm52/qwen36 shared-expert overlap).
4. **Read-ahead**: ReadAheadInitialize/Arm/Join/Destroy.
5. **Pack I/O**: PackRead, LoadDeviceRegion (pack -> device regions).
6. **Admission**: AdmissionDecisionInitialize/Accept/Reject (+ validity).
7. **Runtime snapshot**: RuntimeSnapshotInitialize.
8. **Slot/index claims**: SlotClaim, IndexSetClaim(+Ordinal,+Prepare), IndexSetRelease, SlotAvailableCount, SlotCountFree, WaitForSlots, SlotRelease, CompleteAndReleaseClaims.
9. **Diagnostics**: CudaStatus (error wrapper w/ module tag).

CONSUMERS: all six drivers link this. ANY change here affects every model simultaneously - changes require per-driver regression runs (the completeness matrix per affected driver).

AUDIT NOTES (pass 2):
- stage_module_common has NO model-specific code - clean shared layer. ✓
- qwen36's pccore prefix_cache.c lives in runtime/ alongside - correct placement per layering.
- OPEN: cache/prefix_cache.c vs runtime/prefix_cache.c layering question (see §3) - pccore dev to confirm interlock.
