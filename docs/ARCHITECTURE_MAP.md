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
