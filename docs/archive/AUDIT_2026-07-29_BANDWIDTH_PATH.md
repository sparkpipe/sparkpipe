# Fresh-Eyes Audit: Bandwidth Saturation and the Queue→Token Path
2026-07-29, pre-hardware. Everything below is grounded in file:line reads
from this session; items not read are marked UNVERIFIED and belong on
the hardware-week checklist rather than in anyone's mental model.

## Part 1 — Memory bandwidth: where it can and cannot idle

The decode economics on GB10 are simple: tokens/s is streamed-weight
bytes/s divided by bytes-per-token-per-stage, so every microsecond the
LPDDR bus is not streaming weights is throughput lost. The audit walked
the full submission machinery looking for serialization points.

### What is architecturally RIGHT (verified)
- **Async cohort pipeline, depth 13.** The serving backend's decode is
  fire-and-register: `SparkRingServiceBackendDecodeInner` returns
  `SPARK_STATUS_PENDING` after `RegisterPendingDecode` + forward
  (node/backend.c, DecodeInner tail), the pump treats PENDING as
  in-flight (api/serving_engine.c:2131,2181,2206), and completion
  arrives asynchronously through the result path into
  `SparkServingEngineCompleteDecodeDispatch` (backend.c:3499).
  Cohort capacity is exactly the spark count
  (`SPARK_RING_SERVICE_BACKEND_PIPELINE_COHORT_CAPACITY =
  SPARK_STAGE_PLAN_CURRENT_SPARK_COUNT`, backend.c:38) plus one
  reserved prefill slot — thirteen decode cohorts can occupy the
  thirteen ring stages simultaneously, which is the condition for the
  bus never waiting on the ring.
- **Submit credits gate the ring, not a mutex.**
  `RequireResidentSubmitCredits` before forward means backpressure is
  a counted resource, not a blocking wait.
- **Device completion is on-stream.** dispatch.cu completes via
  `cudaLaunchHostFunc`; the only `cudaStreamSynchronize` calls are
  env-gated debug (`GLM52_STAGE_SLICE_DEBUG_SYNC`) or error paths.
  Nothing synchronizes the stream in the hot path.
- **The scheduler admits by measured cost against the cohort window**
  (Admit -> BuildMeasuredPlanAndCosts -> per-stage critical path), so
  admission is bandwidth-aware by construction.

### Where the bus CAN still idle (findings, ranked)
- **F1. Multi-chunk decode is fail-closed, and that caps batch.**
  `DecodeInner` returns `MODULE_NOT_VALIDATED` when
  `chunk_count != 1` (backend.c, DecodeInner). Rows =
  lanes x rows_per_lane, and speculative verify multiplies rows by
  (spec+1) or the tree row count — so B x 8 rows must fit
  `execution_row_capacity` or the dispatch is refused outright. At the
  batch sizes the bandwidth ledger wants (B64+ per cohort with
  speculation), this ceiling either throttles admission or silently
  forces smaller cohorts. `SubmitDecodeChunksToResident` exists but
  the single-chunk gate in front of it means the chunked path has
  never run. **Action: validate the chunk path (it is written, gated
  off) as the first hardware task; until then know the real row
  ceiling per node and size cohorts to it.**
- **F2. Per-layer launch gaps — CUDA graphs remain the S5 item.**
  93 (K3) / 78 (glm) layers x several kernels x launch overhead is
  hidden only while the host stays ahead of the device. The NodeContext
  already carries `cuda_graph_exec_cache` slots and specialization
  signatures — the plumbing exists, capture does not. UNVERIFIED how
  far ahead residentd's IPC drain keeps the stream under load; the
  `phase_clock_cycles` per pipeline slot is the built-in instrument —
  **first measurement on real hardware: phase-clock gaps between
  slice completion and next slice launch, per stage.**
- **F3. Hidden-transport receive vs compute overlap.** The frame
  carries pre/post session functions and packets; whether the recv for
  cohort N+1's activation is posted while N computes is a property of
  the memlink/RDMA session implementation. UNVERIFIED in this pass —
  the 29 us/hop software floor was measured, the overlap discipline
  was not. Hardware checklist item.
- **F4. Prefill/decode interleave.** One reserved prefill slot in the
  queue depth means a long prefill occupies one cohort slot while
  twelve decode cohorts continue — good. But a prefill that needs the
  full ring (bulk prefill) drains decode; the scheduler's admission
  sees cost, not bus occupancy during the transition. Acceptable, but
  measure the transition gap on hardware before trusting it.
- **F5. MoE expert gather locality (glm).** Routed-expert weight reads
  are contiguous per expert but the *set* of experts per token batch
  scatters across the weight arena. The router cross-layer prefetch
  map (`glm52_router_crosslayer_map.npz`) exists from the CPU study;
  nothing consumes it yet. Post-bring-up optimization, not a
  correctness gap.

## Part 2 — Queue to decoded token: the path, stage by stage

Legend: OK = code read and coherent this session. GATED = behavioral
test executes it on every suite run. GAP = does not exist / cannot work
today. UNVERIFIED = not read this pass.

| Stage | glm 5.2 | K3 |
|---|---|---|
| HTTP ingress -> compat parse | OK (gateway lib; compat tests are template-data keepers) | GAP-K1: no K3 chat template / compat surface; K3-first endpoint README exists, parser does not |
| Submit -> request slot | OK + GATED (request api suite) | OK — slot tier is model-free (proven by null-seam link binary) |
| Speculation provider | glm module provider, parity-gated | null provider, GATED (test_null_seam_link runs) |
| Scheduler admit (geometry, cost) | OK + exercised by behavior suite; memset bug fixed and gated | GAP-K2: geometry values exist (93 layers) but **cost profiles are glm measurements** — `measured_profile_id = 20260701` tables answer glm shapes; K3 admission needs either K3 measured profiles or the estimated-profile path wired for K3 shapes |
| Dispatch -> serving engine -> PENDING | OK + GATED (service + pump suites) | Same machinery, model-free — OK once K2 admits |
| Ring forward -> rank daemons -> residentd IPC | OK (fire-and-register verified) | GAP-K3: residentd/builder are glm-bound — `builder_library.builder_interface.decode` builds glm frames; the K3 stage-execute path (the board's big rung) does not exist |
| Frame -> module.c -> doorway validation | OK; seam real, -Werror, GATED (firmware suite) | Doorway validates geometry honestly (k3 validation.c) but nothing calls it in a live daemon yet — consequence of K3 |
| Device slices -> tokens -> completion | OK (host-func completion; final-token copy path read) | GAP-K3 continuation |
| Result -> CompleteDecodeDispatch -> token out | OK + GATED end-to-end at api tier | Blocked behind K2+K3 |
| Detokenize -> stream | UNVERIFIED this pass for both (text tier not walked) — add to checklist |

### The K3 bring-up list, in dependency order
1. **K2 first — it is cheap and unblocks admission**: give the
   scheduler a K3 cost source. Either land estimated profiles keyed by
   K3 geometry (the `LoadEstimatedLargeBatchCostProfile` shape already
   exists) or measure on hardware day one and add
   `MEASURED_PROFILE_<date>_K3` tables. Until then no K3 dispatch can
   be admitted no matter what else lands.
2. **K3-A: the builder/execute rung** (known board item): K3 frame
   build + stage execute through the doorway, riding the same
   PENDING/cohort machinery — the machinery itself needs zero changes,
   which this audit confirms is the payoff of the seam campaign.
3. **K1 last**: K3 chat template + compat surface. Serving raw token
   ids works for bring-up; the template is a product step.
4. Loader blockers (`A_log[:96]`, `transpose_state_layout`) land with
   bind as already planned.

### One-line verdicts
- Bandwidth architecture: **sound** — async, credit-gated, cohort=13;
  the risks are the unvalidated chunk path (F1), launch gaps until
  graphs (F2), and unmeasured transport overlap (F3). All three are
  measurable with instruments already in the tree.
- glm path: **complete and gate-protected** queue -> token at the host
  tier; device tier awaits hardware only.
- K3 path: **three named gaps (K1 ingress, K2 cost tables, K3 execute)**
  — K2 is the cheap surprise this audit adds to the known board, and
  it comes first.

## Addendum (same day): Qwen 3.6 joins debug distance, K2 closed
The uniform-estimated cost profile (SPARK_STAGE_PLAN_PROFILE_UNIFORM_
ESTIMATED = 0, explicit and validated, never silently upgraded) gives
any measurement-less family an admissible scheduler: LoadUniformCost
Profile fills geometry->layer_count entries from a wiring-supplied
estimate and the balanced builder does the rest. test_uniform_profile_
admit proves it with qwen's dense geometry {64,64} - decode ADMITTED,
thirteen balanced stages. K2 is closed for K3 and qwen alike; both now
wait only on their execute rungs. Speed model for qwen BF16 lives in
docs/QWEN36_BF16_SPEED.md. Dense geometry convention: first_routed ==
layer_count means no routed layers; the scheduler validates <= now.
