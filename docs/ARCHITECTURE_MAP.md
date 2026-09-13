# SparkPipe Architecture & Roadmap Map (v1 - goal/state/delta format)

Restructured per coordinator directive: not an inventory catalog, but
GOAL -> CURRENT STATE -> DELTA -> WORK QUEUE with critical-path tracking.
Tasks can be aborted/changed with good reason - simpler beats complex.

## 1. THE EVENTUAL GOAL (what makes SparkPipe useful)
- On-prem serving of frontier models on DGX Spark fleets, scaling 4 -> 16
  boxes WITHOUT changing the client API or model packages.
- ONE OpenAI-compatible endpoint for all models; automatic batch formation;
  sub-minute model promotion; resident working sets.
- SOTA performance PROVEN across the completeness matrix (batch x context x
  topology x weight-format x features), measured vs bandwidth-arithmetic
  bounds and external references (vLLM et al).
- New models: adding a driver should reuse the shared layers maximally
  (DFlash2 spec, paged KV + prefix cache, stagepack reader, adapter stack).
- New hardware: the hardware-agnostic interface (hwiface v1 frozen, DSV4-first)
  so AMD MI350P-class support reuses the model cores.
- Minimal code: Solutions / (production-codesize SQUARED). Deprecate freely.

## 2. WHERE WE ARE (honest, per front)
| Front | State | Evidence |
|---|---|---|
| Qwen3.8-27B DFlash2 | WORKING, very close to SOTA acceptance; consolidated tree 229526d; glm dev paused 1 week | bench matrix receipts; lossless 11/11 |
| GLM5.2 | Working B8/B16 serving (43.5/75.55 tok/s aggregates); driver audit delivered: GPU validator missing (prereq #1), ~1200-1500 collapsible lines via shared stagepack reader, compat shims named for deletion; DFlash2 adoption queued (second model, drafter backend in-tree) | glm52_report + exec result |
| vLLM reference (spark3:8124) | MEASURED baseline: prefill ~740 flat; decode 16->226 scaling by batch; cache ingestion works | docs/benchmarks receipts |
| DSV4 Flash | BROKEN serving: IPC break at 1cfea8f (last-good fd5797f); k7 completion stall; HEAD lease regression | bisect report r1 |
| GLM5.2 | Working B8/B16 serving; driver audit delivered; validator missing | glm52_report |
| K3 | TP4xPP4 deployed; TP16 pack production + e2e open | k3_report |
| Qwen38-Max | Driver complete, never hardware-tested; bring-up runbook ready | max_report |
| Prefix caching | GENERAL CORE implemented (runtime/prefix_cache.c); model port pending; production ring bit-exact | pccore_result |
| hw-interface | v1 FROZEN (DSV4-first); audited; ROCm probes started | hwiface_v1_freeze |
| OpenAI endpoint | vLLM-mode only via glm dev; GENERIC gateway refactored but decode-step broken (lease handshake suspect) | gateway_result |
| Fleet | spark0 build/ref; spark2+3 glm dev until stable; spark4-7 dsv4-flash; 9 dark boxes; all 16 return Tuesday | sysadmin_report |

## 3. THE DELTA -> WORK QUEUE (critical path first)

### CRITICAL PATH A: every model serving via the API (user: "all models work via the API")
A1. Fix gateway decode-step INVALID_ARGUMENT (lease handshake) -> qwen38 backend end-to-end [gateway-dev, IN FLIGHT]
A2. Deploy consolidated driver to spark2; verify against runbook hashes [coordinator]
A3. COMPSEC-17 through the API for qwen38 (bar 15+/17) [after A1]
A4. Same API path per remaining model as drivers stabilize

### CRITICAL PATH B: SOTA performance proof
B1. Benchmark matrix on SparkPipe-native serving (vs vLLM reference already measured) [after A2]
B2. Prefill batching gap: cross-request batching (measured flat 740 across B) - design + implement
B3. Tensor-core decode attention (decode -21-25% at ctx2048)
B4. Per-model matrix cells to SOTA bar

### FRONT QUEUES (see coord/queues/*.md - iteration 3 live)
- dsv4-flash: lease-series consolidation + BF16 crash fix
- glm52: validator restore + stagepack collapse (pure deletion round) + PP7 prep
- k3/max/pccore/qwen36prefix: iteration 3 in flight

### IDEAS / SIMPLIFICATIONS (welcome anytime)
- Adopting the glm dev's produced=accepted+1 commit shape could SIMPLIFY our
  replay machinery (their flow has no replay emission) - evaluate vs our
  lossless requirement before any change.
- Stagepack collapse is the template: shared core + geometry tables beats
  four private copies everywhere.

## PASS 4: Adapter layer comparison (drift analysis)

Adapter sizes: qwen36=2984, dsv4=1515, glm52=1489, qwen38=1471, k3=502.
Model-specific symbols: qwen36=311, glm52=159, k3=18.

qwen36's adapter is 2x the others because DFlash2 added speculation state,
draft view construction, accept-loop logic, replay handling, and per-lane
snapshot management. The OTHER drivers will need the same features when they
adopt DFlash2 - which means either (a) each driver duplicates qwen36's approach,
or (b) the shared speculation machinery moves into stage_module_common or a new
speculation_common module.

DRIFT RISK: the serving_adapter.c files share the same fundamental contract
(daemon -> module lifecycle: init, submit, complete, destroy) but each has grown
model-specific extensions independently. The hw-interface definer's v1 freeze
addresses the DEVICE side; the ADAPTER side needs equivalent treatment.

RECOMMENDATION: extract the common adapter skeleton (env parsing, slot claiming,
completion routing, residency bookkeeping) into runtime/adapter_common.h so new
drivers get DFlash2 + prefix caching for free instead of copy-pasting from qwen36.

## PASS 5: Cross-driver duplication audit

### CONFIRMED DUPLICATION
| Pattern | Copies | Lines | Drivers affected |
|---|---|---|---|
| Paged KV implementation | 3 | ~1942 | qwen36 (716), qwen38 (725), dsv4 (501) |
| Stagepack format/validator | 4 | ~1728 | dsv4 (464), qwen36 (550), qwen38 (432), glm52 (282) |
| Adapter env/config approach | inconsistent | varies | qwen36=13 getenv, glm52=1, dsv4=0, k3=0 |

### THE FIX PATH
- Paged KV: extract runtime/paged_kv_common.c with per-model geometry callbacks
- Stagepack: shared reader already landed (runtime/spark_stagepack_reader.h); DELETE the four private families
- Adapter: extract runtime/adapter_common.h skeleton; each driver provides only model-specific hooks

## PASS 6: K3 driver audit findings

### P1 DEFECTS (blocking TP16 boot)
- F1: runner.cu hard-requires host tp_collective when tp_degree>1; adapter only supplies device_collective at degree 16; host TCP tier caps at 4 ranks. INVALID_ARGUMENT at init.
- F2: bind.c layer-92 bind divergence — pure period-4 rule misses trailing-layer exception (layer 92 = MLA per checkpoint/pool-sizing/packer). Tests only cover layers 0/1/3.
- Head exchange must stay f32 (bf16 NCCL can't carry vocab-scale token ids).

### P2 DEFECTS
- NULL-lane deref in 2-rank hidden-transport combine
- ~20 unchecked cudaMallocs + init-failure leaks (munmap of registered region)
- kind-blind expert_interleave
- graph capture × NCCL completions never proven with live device tier
- accepted_token_count over-reports copied tokens

### DRY VIOLATIONS
- Layer-kind truth ×3 sources
- Slice geometry literals ×4
- gate|up sizing ×3
- Twin device-tier submission blocks
- Weight-name tables ×3

### CLOSURE PLAN
Step A (workstation): D1/F2 fix + bind-test extension to {0,1,2,3,92} + stage-3 slice; F1 fix; ride-alongs F3/F4/F7. Step B: T1 gates with device-tier legs. Steps C/D: tile_k-32 pack, deploy, window boot→smoke→capture→B1. Step E: close #667.

### DISPOSITION (updated as fixes land)
- F1 FIXED (2026-08-24, patch `.agents/coord/k3_f1_device_tier.patch`, in-tree at k3's runner): the init gate now rejects sharded configs only when NO self-sufficient tier exists — `tp_collective` is optional whenever the device collective is NCCL (the only backend that carries every exchange: bf16 hidden sums + the head argmax). Device-tier head exchange = ONE u64-max all-reduce per step: `[63:32]` the score under the IEEE-754 total-order transform (exact, no arithmetic), `[31:24]` inverted rank (MAX then breaks score ties toward the LOWEST rank = the host slot-sum scan's first-max rule), `[23:0]` token id. The audit's "stay f32" requirement (bf16 can't carry vocab-scale ids) is met exactly — one exact 64-bit word instead of an f32 sum plus a second exchange for the ids, reusing the existing shared SubmitU64Max op on both backends instead of growing the collective ABI mid-window. Evidence (spark0 GB10 CUDA 13): device-only init gate PASS incl. negative controls (`tests/test_k3_device_only_init.cu`; nccl-only status != INVALID_ARGUMENT where old code returned INVALID_ARGUMENT pre-create); head-key contract probe PASS (edge roundtrip bit-exact incl ±inf/±0/denormals + NaN-loses; 200k randomized 16-rank reductions == host first-max rule; `tests/test_k3_head_key_probe.cu`); tp_degree-1 leg byte-identical before/after (7168 bf16 hidden dump, pristine vs patched binaries); source pins live in `tests/test_k3_driver_contracts.py` contract 4. OWED to the ring: a live multi-rank NCCL exchange proof - the script has LANDED (`tests/test_k3_nccl_multirank_proof.cu` + `tools/k3_nccl_multirank_gate.sh`: WORLD real processes bootstrap the production NCCL transport via SparkTpDeviceCollectiveCreate and drive the runner's two device-tier exchanges live - SubmitBf16 at the K3 7168 width bit-exact under any reduction order plus the reserved0-narrowed override's tail-integrity check, the SubmitU64Max head argmax over engineered cross-rank cases vs the packed-key oracle AND the host first-max rule on finite rows (clear / all-tie / partial-tie / NaN-on-rank-0 documented-divergence / +-inf / denormal-vs-zero / negatives / vocab-max tokens), and four back-to-back submits proving the strict per-process ordinal chain + stream ordering live; worlds {2,4} default, {8,16} opt-in; SKIPs cleanly where nvcc/libnccl are absent), still OWED until it runs GREEN in the T1 window boot. Code size: authored production delta +171 lines (runner.cu); test/tool files excluded from the counter.
- F2 FIXED (2026-08-24, same session; patch `.agents/coord/k3_f2_bind_layer92.patch`): bind.c folded onto `SparkK3LayerIsMla` (pool_sizing.h) — one kind truth, no private period-only re-derivation. test_k3_bind.c extended to layers {0,1,2,3,90,91,92}, the PP4 stage-3 slice 70..92 (all 23 layers bind, kinds == predicate, 7 MLA / 16 KDA matching SparkK3MlaLayersInSlice), config-relative shape pins, and the past-the-end negative. New tests/test_k3_bind_fixture.py builds a 93-layer synthetic backbone through the REAL packer CLI (canonical layer_types map + true dense layer 0 via the parameterized mini_checkpoint) and runs the C gate on cc/clang — GREEN on the workstation AND spark0. Mutation-proven: restoring the period-only rule REDs exactly the layer-92 legs (bind VALIDATION_FAILED — the boot-time shape the real checkpoint would have hit).
- CONTEXT/BATCH AXES FIXED (2026-08-25, K3 worst-driver scorecard): the context axis died on a pinned page count — `"kv_pages": 2` is ONE 128-token MLA cache (K3_KV_PAGE_SLOTS=64), so every cell past ctx128 was unreachable. Deployments now speak model units: the serving adapter derives pages from `"context_tokens"` over K3_KV_PAGE_SLOTS (explicit `"kv_pages"` still overrides; default covers ctx2048), all 16 rank configs + example request ctx4096 → 64 pages (~113 MiB/rank, trivial). The batch axis died on row-ceiling pins: adapter `max_rows` and deployment `runtime_limits.max_input_rows` at 16 refused every row-width cell above b16 — B1024 (prefill-convention rows, per K3_PERF) unreachable. Both lifted to 1024 (scratch/staging scale ~192 MiB at B1024; decode concurrency stays max_sequences=16 — fp32 KDA state bounds it, the BF16-state lever is the separate known lever). tools/k3_gen_adapter_configs.sh emits the new shape; source pins in tests/test_k3_driver_contracts.py RED on any revert to pinned pages or a 16-row ceiling.
- TP16 BOOT-CHAIN FIXED (2026-08-25, config-deploy + boot-verify turn): five defects each killed residentd init before any collective ever formed, all fixed and pinned in tests/test_k3_driver_contracts.py — (1) tools/k3_gen_deployment.sh emitted duplicate stage_index (i/TP) but ValidateStructure demands a permutation of 0..N-1, plus nonzero kv page caps without JIT_KV and a 16-row ceiling → schema_error/INVALID_ARGUMENT on every boot attempt, pre-existing since staging; (2) libk3_serving_adapter.so never exported the family-neutral SparkModelServingAdapterGetInterface the loader dlsym's → NOT_FOUND after dlopen; (3) descriptor capability_flags were zeroed and row-capped at 16 → loader INVALID_ARGUMENT + limits rejection; artifact_sha256 pinned to the bucketed module identity; (4) Makefile K3 adapter link lacked -lcuda so cuTensorMapEncodeTiled stayed unresolved (the original not_found); (5) runner.cu combine kernel wrote through a const destination (compile breaker). Live evidence on spark0 with fresh binaries + corrected configs: init now traverses adapter_load → deployment_validation → runtime_limits → transport_contract → rank_plan → transport_load into ADAPTER_INITIALIZE. STILL OWED for green 16-rank: provision bin/lib/packs on ranks {1,8,9,a,b,d,e,f} (tools/k3_stage_runtime_tp16.sh + k3_deploy_tp16.sh), recover sparkc (unreachable), regenerate/repair tp16 rank packs failing loader VALIDATION_FAILED, then tmp/k3_tp16_rollout.sh boot && status. Configs deployed: 7 ranks full push+preflight PASS, 8 pre-staged, backups *.bak-<stamp>. FOLLOW-UP (same day): the tp16 rank packs NOW OPEN CLEAN via the real loader - packcheck harness on spark0 shows format v2, 99.5 GB, embed 10240x7168 (= vocab/16), layer-92 MLA present - so that owed item is RESOLVED by the salvage-stream republish. And a FULL-WORLD TP4xPP4 boot on the corrected stack brought 12/15 reachable ranks up with GROUPS 1+2 COMPLETELY SERVING-READY ("model_residentd ready rank=5 stage=5 ... rows=1024 resident=16 adapter=k3-tp4pp4 tcp=spark5:21480" - B1024 admission proven live, control endpoints :21480 x8) while group 3 was blocked solely by dead sparkc and group 0 by a stale first-attempt process; both swept after evidence capture and the fleet returned to idle. GATE GREEN (same day, spark0): tools/k3_nccl_multirank_gate.sh PASS world=2 (2/2 ranks clean) and PASS overall for worlds 2+4 - bf16 hidden sums bit-exact at 7168 width incl narrowed-tail integrity; u64-max head argmax equals the packed-key oracle plus the host first-max rule on all engineered cross-rank cases; strict back-to-back ordinal chain. Fixes en route: NCCL_MULTI_RANK_GPU_ENABLE=1 is REQUIRED on NCCL 2.30.7 for multi-rank-per-GPU proof runs (production TP16 is one rank per GPU and unaffected); leg C redesigned to four fresh per-op buffer pairs (in-place accumulation scales by world size per op and breaks bf16 exactness at w16); pre-destroy stream drain so no early-exit path tears down a communicator with in-flight collectives; ordinal assertion corrected to seven counted completions (leg B unpacks via its own callback). Logs: spark0 /home/spark0/k3gate/tmp/gate_tmp/k3_nccl_multirank_20260825T183822Z.
