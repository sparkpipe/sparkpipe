# Continuous batching lane report — 2026-08-27

Worktree /tmp/lane-contbatch, branch `lane/continuous-batching`. Node used:
spark3 only (27B dense FP8 TP1 deployment `~/sparkdata/qwen38.fp8.tp1`).
spark2 untouched. No GPU resets, no leaked processes (verified at exit).

## CB1 — the current admission flow (traced)

`runtime/model_batch_engine.c`, submission → residentd:

1. `SparkModelBatchEngineSubmit` only enqueues: request state
   `QUEUED_PREFILL`, free-list slot, no sizing, no waiting.
2. `SparkModelBatchEngineProgress` is the pump. Per call it loops
   `ChooseWorkKind → DispatchKind → BuildSubmission → SelectRequests`
   up to `maximum_new_submission_count`, while
   `inflight_submission_count < submission_capacity`.
3. `SparkModelBatchChooseWorkKind` computed the **first fixed gate**: work
   kinds got `minimum_by_kind` floors from
   `SparkModelBatchTargetRowFloor` (= adapter descriptor's
   `minimum_efficient_submission_row_count`; DSV4 non-TP16 = 16, K3 = 1,
   Qwen adapters = 0/1). `SparkModelBatchSchedulerChooseWorkKind`
   round-robins PREFILL→DECODE→RELEASE but *skips* any kind whose queue is
   below its floor while other work is inflight, until the request ages
   out (`scheduling_bypass_count >= submission_capacity`). That skip is
   the "wait to fill a B16 bucket" behavior.
4. `SparkModelBatchAssignPrefillCounts` applied the **second fixed gate**:
   `SparkModelBatchSchedulerPlanGroupSize(total, max_prefill_rows, floor)`
   reserved floor-rows for future fixed-size groups (its unit test pinned
   it: 136 queued rows → submit 120, hold 16 back to seed the next group).
5. The remaining caps are real resources, not buckets, and were kept:
   `max_active_sequence_count` (MAX_ACTIVE),
   `max_prefill_rows_per_submission`, and the KV-page bound
   (`SparkModelBatchSchedulerPlanCacheBoundLaneCount` + the cache-demand
   reservation in the selection passes).

Empirical confirmation of the old gate: with 2 requests queued while one
submission was inflight, the second prefill was NOT dispatched (floor 16 >
2 queued, inflight > 0) — `inflight_submission_count` stayed 1. The new
admission dispatches it (`inflight == 2`); this is now asserted in the e2e
suite and fails on the old code (verified both ways, see CB3/CB5).

## CB2 — the fixed gate removed

Commit `012e882` "continuous batching: admit the whole ready set, drop the
floor gate". Net −21 authored lines.

- `SparkModelBatchChooseWorkKind` now passes floors of 1 for every kind:
  any nonzero ready set dispatches on the next Progress. The bypass/aging
  machinery stays as the starvation escape at depth (it is engine-side
  aging in `SparkModelBatchUpdateRequestAges`, independent of the floors).
- `SparkModelBatchSchedulerPlanGroupSize` deleted (header + engine + its
  unit asserts). `SparkModelBatchAssignPrefillCounts` gives the ready set
  the whole row budget up front: `min(total_span, max_prefill_rows)`, the
  remainder chunk-prefills on the next pass. Note (honest finding): in the
  e2e fixture the old reservation was already shadowed by the block-span
  cap (each lane takes ≤ cache-block rows per pass and the lane selection
  consumes the span budget first, so total selected span never exceeded
  the row budget) — the planner's ladder path was dead code in practice.
  Its removal is real simplification; the observable win is the floor
  removal below.
- `minimum_efficient_submission_row_count` field dropped from the engine
  (the adapter descriptor still carries it; only the engine's use is gone
  — no ABI change).
- Kept: MAX_ACTIVE bucketing, prefill row cap, KV-page bound, priority
  passes, resident-slot binding, prefix cache — all resource or correctness
  logic, none of it batch-shaping.

## CB3 — existing e2e verification

`make build/test_model_pipeline_client && ./build/test_model_pipeline_client`
(host-only: 3 fixture residentd ranks + fixture adapter, no GPU needed).

- Baseline before the change: PASS (exit 0).
- After the change: PASS 6/6 consecutive runs.
- Differentiation check: with the runtime change reverted and the new tests
  kept, the suite FAILS at
  `view.inflight_submission_count == 2u` (TestModelBatchEngineContinuous)
  — old admission deferred the second prefill, new admission dispatches it.
- The full-suite contract scanners pass:
  `tests/test_model_serving_architecture.py` PASS,
  `tests/test_dsv4_driver_source_contracts.py` PASS.
- Code-size ratchet: `non-test authored lines: 186479 (ceiling 186500)`
  — net shrink, no ratchet move needed.

Pre-existing flake found (NOT from this lane): `TestModelPipelineStopResidents`
exit-status assert (test line 426) failed 2/5 baseline runs and 2/7 new-code
runs, always in the pipeline-client phase BEFORE any batch-engine scenario
runs. It reproduces on the unmodified runtime, so it predates this lane.
Exact assert: `(uint32_t)WEXITSTATUS(child_status) == expected_exit_status`
in the first `TestModelPipelineStopResidents(children,paths,1u)` call.

## CB5 — continuous batching test (staggered arrival, oldest-first)

New `TestModelBatchEngineContinuous` in tests/test_model_pipeline_client.c,
three scenarios (all deterministic; the suite's established timing model —
a just-dispatched submission cannot complete within the same Progress — is
relied on exactly once, mirroring the existing Shutdown test):

1. Chunked-prefill continuity: three 15-token prompts prefill as repeating
   block-span passes (3 lanes x 4 rows, then 9-row tail): exactly 4 prefill
   submissions, then completion. "Submit, process, repeat" with no gating
   between passes.
2. Staggered arrival (the core proof): r1/r2 submitted, one Progress (one
   prefill inflight), then r3/r4 submitted and one more Progress —
   `inflight_submission_count == 2` (both prefills in flight; the old floor
   kept this at 1). Drain: 4 completed, 8 tokens, decode batches carry
   exactly the ready set (2 lanes per wave or one merged 4-lane pass;
   decode lane sum == 4), oldest-first token order
   (`token_request_ids[0..3] == 1601..1604`).
3. Deep queue, oldest-first: 20 requests at MAX_ACTIVE 16 — first window
   takes 16 lanes in arrival order, the tail drains in a smaller variable
   batch, token emission order is exactly arrival order for all 20.

## CB4 — B* measurement on spark3

### INCIDENT FIRST — spark3 GPU wedged (report immediately)

spark3's GPU is DOWN for CUDA clients as of the end of this lane's CB4
sweep and needs host intervention (reboot) — I did NOT reboot it.

Sequence: the sweep at B=8 stalled hard (868/1024 tokens in 28.8 min,
~2.0 s per generated token vs 0.11 s/token aggregate at B=4). The
deployment's residentd was TERM'd; it IGNORED TERM (spun in R state, 39
CPU-minutes, old stack does not handle client disconnect) and was
SIGKILLed per the last remaining option. The lane rules' predicted leak
followed, and worse — GB10 has NO reset path:

```
$ sudo nvidia-smi -r
The following GPUs could not be reset:
  GPU 0000000F:01:00.0: Not Supported

$ dmesg | tail
NVRM: nvAssertFailedNoLog: Assertion failed: (status == NV_OK) ||
  (status == NV_ERR_GPU_IN_FULLCHIP_RESET) @ gpu_user_shared_data.c:373
arm-smmu-v3 arm-smmu-v3.1.auto: CMD_SYNC timeout at 0x00017d73 ...  (repeating ~1/s)

$ ./memprobe   (cudaMemGetInfo + 1 GiB cudaMalloc)
devices=0 err=no CUDA-capable device is detected
```

The killed process's `cuda-EvtHandlr` kernel thread remains in R state
inside a Zl process (un-reapable), the ARM SMMU CMD_SYNC queue never
completes, and no new CUDA context can attach. A GPU reset would clear
this but is "Not Supported" on GB10; a driver reload is unsafe with the
SMMU in this state. The node needs a reboot by whoever owns it.

Lesson recorded: on this old deployment's residentd, client disconnect
leaves the process spinning and TERM is ignored. Never SIGKILL a
residentd mid-CUDA on GB10 — there is no reset path to clean up after
it. Better to leave a user-space-spinning daemon than a kernel-wedged
GPU. The current-stack residentd (this lane's worktree) is a different
binary; its disconnect behavior is untested for this hazard.

Node state left behind: config restored byte-identical to pre-lane
(`model_resident.json.before-contbatch` backup kept), sweep and probe
temp files removed, no live daemons (one un-reapable zombie remains, and
the GPU it holds is unusable until reboot).

### Partial measurement (before the wedge)

Method: `tests/test_batch_knee_sweep.sh` (new, parameterized —
`SPARK_HOST`, `DEPLOY`, `BATCHES`, `BUDGETS`, `PROMPT_TOKENS`, `CONFIG`;
no hardcoded nodes). Per batch size B the deployment's own
`sparkpipe_model_batch` runs twice (output budgets 128 and 256, 64-token
prompts, distinct per request); the decode-only rate comes from the
budget difference so connect + prefill overhead cancels:

    decode_tok_s = B * (tokens_hi - tokens_lo) * 1000 / (wall_hi_ms - wall_lo_ms)

Deployment: Qwen3.8-27B dense (48 GDN + 16 attention, no MoE), FP8, TP1
single rank, `/home/spark3/sparkdata/qwen38.fp8.tp1`,
`max_active_sequences=64`, `kv_physical_page_capacity=64`,
`max_inflight_submissions=2`, KV block = 64 tokens. Residentd fresh, GPU
idle before (nvidia-smi: no processes), 96% GPU utilization observed
mid-sweep. Raw rows:

```
B,wall(128tok),wall(256tok)     [seconds]
1,26.387,52.417
2,52.724,104.675
4,58.402,114.960
8,~1726+,aborted (see incident)
```

Derived rates (budget-difference method):

| B | decode tok/s | scaling vs B=1 | per-decode-step |
|---|---|---|---|
| 1 | 4.92 | 1.00x | 203 ms |
| 2 | 9.86 | 2.00x | 203 ms (2-wide) |
| 4 | 36.22 | 7.36x | 110 ms (4-wide, faster than B=1) |
| 8 | ~0.5 | collapse | ~2 s — pathological |

Findings, honestly framed:

1. **The knee was not reached.** The hardware law (TOPOLOGY_GUIDE) puts
   B* ≈ 106 for FP8 on GB10. This deployment cannot sit there even in
   principle: MAX_ACTIVE=64, 64 physical KV pages (3 pages per request at
   64-token prompt + 256 budget → ≤21 concurrent), 2 inflight submissions.
   The deployment caps are exactly the "MAX_ACTIVE_SEQUENCE_COUNT: raise"
   and buffer-sizing rows of the EXPERT doc's change table.
2. **The old deployed stack's batch path is non-monotonic.** Scaling is
   perfect at B=2 (2.00x), super-linear at B=4 (7.36x — per-step decode
   time DROPS below B=1, the weight-amortization effect the doc
   predicts), then collapses at B=8 (2 s/step, ~60x worse per token than
   B=4). A queue-depth-to-throughput function that dips like that cannot
   be operated at a knee by any client; the continuous engine's
   ready-set admission is necessary but the resident-side pathology at
   B=8 needs its own investigation (candidates: the 2-submission
   pipeline cap interleaving with chunked prefill, or the old cache-demand
   model stalling admission — `BuildInflightCacheDemand` failure freezes
   dispatch via UINT32_MAX pages in the current code's ancestry).
3. The sweep used the deployment's pre-lane engine (self-consistent
   packs/binaries). The knee itself is engine-independent; the
   continuous-admission engine is what lets a deployment SIT at the
   knee once the caps above are lifted.

Re-running the full sweep needs the node recovered (reboot) and should
additionally raise `kv_physical_page_capacity`/`kv_logical_page_capacity`
with the stage's KV pool budget in view (the stage reported 71.1 GiB
device at 256 logical pages / 8192 blocks — verify pool scaling before
raising), then sweep B in powers of two up to MAX_ACTIVE with
`tests/test_batch_knee_sweep.sh`.

## INTEGRATION REQUEST

Files this lane changed that are OUTSIDE the strict lane write-set
(granted explicitly by the lane brief, but these are shared runtime files
the coordinator owns):

1. `runtime/model_batch_engine.c` — admission change (commit 012e882).
2. `runtime/model_batch_scheduler.h` — `PlanGroupSize` declaration removed,
   continuous-admission contract documented on the header.
3. `tests/test_batch_knee_sweep.sh` — new (lane write-set).

No changes to: kernels, modules, packs, serving adapter, module ABI,
Makefile, sources.mk. `tests/test_model_pipeline_client.c` is in the lane
write-set.

If the coordinator prefers the engine change to land as a policy flag
rather than a behavior change, the minimal seam is the three
`minimum_by_kind[...] = 1u` lines in `SparkModelBatchChooseWorkKind` —
everything else stands alone.

## Next experiment

- **First: recover spark3 (reboot) and re-verify the GPU** — the wedge
  blocks every other lane's GPU plan for this node.
- Re-run `tests/test_batch_knee_sweep.sh` with the caps lifted (KV pages,
  then MAX_ACTIVE with the stage pool budget verified) and find the real
  B=8 collapse point on the current stack: sweep B = 4..16 at 1-token
  budget steps to bracket it.
- Sweep a MoE deployment (k3.mxfp4.tp4pp4 sits on spark3) where the
  weight-amortization knee is sharper (expert stream is constant per
  step); the dense 27B knee is the gentlest case.
- Re-run the sweep with the continuous engine driving (needs a rebuilt
  batch tool against the deployment's runtime — coordinator call).
