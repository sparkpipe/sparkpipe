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
