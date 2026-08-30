# kimi-k3 — P1 chain+async port plan (design before code, 2026-08-30)

Operator directive this session: ledger item 3 (the D2/P1 step-loop
program). Status found on arrival: the HOST half of D2 already landed
(`lane/p1d2-steploop`, merged as part of fa337b7's program; receipts in
reports/p1d2-steploop-2026-08-29.md) — the serving loop now drains
async adapters to the adapter's own declared backpressure. What remains
of P1 is the module half, and PERF_PROGRAM parks it explicitly:
"FLEET-GATED (needs live modules; after closeout)". Every k3 module
file is nvcc-gated (no host build on the mac lane checkout), and the
runner's completion semantics need GPU exactness receipts — so the
landable unit today is THIS plan: the design, the oracle list, and the
discriminating receipts, ready to execute the moment the window opens.

## The defect, precisely (k3 today)

The landed loop treats a family as async ONLY if its adapter declares
`SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_ASYNC_COMPLETION`
(include/sparkpipe/spark_model_serving_adapter.h:54). The k3 adapter
does not (spark_k3_serving_adapter.c:689-694 — the flag list has
PREFILL/DECODE/FANOUT/HIDDEN_TRANSPORT/SPECULATION/HYBRID_TP_PP only),
so the resident loop still bounds k3 to ONE adapter op per Progress
pass — the p1d2 report's own "PATTERN B verbatim" bound.

Why k3 never declared it — because it would be a lie today:

1. `K3ServingSubmit` (spark_k3_serving_adapter.c:519) ends with the
   runner having completed the step synchronously; the adapter then
   D2H-copies the tokens (lines 602-610) and fires the serving
   completion INLINE inside submit (line 611). The comment at line
   583 says it outright: "The runner completed the step
   synchronously".
2. `SparkK3StageRunnerSubmit` (runner.cu:917) carries 5
   cudaStreamSynchronize calls on the step path (lines 420, 569, and
   the tail around 1035/1049) vs ZERO in dsv4's stage runner
   (spark_dsv4_stage_runner.c — the count is the diagnosis).
3. Per-submit heap churn on the serialized path: positions_host64
   malloc/free (adapter line 533/548) and tokens_host malloc/free
   (line 588/610) every token — against the engine's own contract
   ("no request or dispatch allocation occurs on the hot path",
   spark_model_batch_engine.h:118-119).

This is the exact shape PERF_PROGRAM measured on the 27B ("chain
depth 1, synchronous submit ending cudaStreamSynchronize ... ~30%
serialized host bubble") and it is why the TP16 roofline (20.2 tok/s
at 49.5 ms; K3_PERF.md) is unreachable at chain depth 1: the ~55 ms
host enqueue sits INSIDE every step's wall time.

## The reference (dsv4, the Rosetta Stone)

- Runner: dispatch carries completion_function/completion_context
  (spark_dsv4_stage_runner.c:353-354) into the driver frame; the
  runner never syncs.
- Module: `cudaLaunchHostFunc(stream, SparkDsv4ModuleCompleteAsync,
  &completions[slot_index])` (module.c:5712) — a stream-ordered host
  callback fires the adapter completion when the device work drains;
  per-slot completion state (module.c:3423's event records feed the
  fork/join milestones).
- Adapter: pending-slot pool (`SparkDsv4ServingPending`), submit =
  validate -> reserve pending -> enqueue dispatch -> return
  (spark_dsv4_serving_adapter.c:1093-1175); completion publishes from
  the driver callback (SparkDsv4ServingDriverCompletion:576); quiesce
  drains by deadline.
- Capabilities: composed via SPARK_SERVING_ADAPTER_CAPABILITY_CHAIN
  (spark_serving_adapter_template.h:43) so async can never silently
  go missing from a family paste.

## The port, staged (each stage independently landable + gated)

### Stage A — adapter pending slots + async declaration (host C)

- Reserve a `K3ServingPending` slot per submission (pool sized by the
  existing pipeline slot count, dsv4 shape); pass
  `dispatch.completion_function/context` (the fields ALREADY EXIST in
  SparkK3StageRunnerDispatch — the adapter currently writes 0 at
  lines 578-579) pointing at the slot.
- Stop the inline tokens D2H + completion fire; the completion moves
  to the runner callback (Stage B) or, until Stage B lands, a
  sync-tier shim calls it at the end of submit (behavior-identical to
  today, but through the callback path — so the flag flips ONLY with
  Stage B).
- Adopt SPARK_SERVING_ADAPTER_CAPABILITY_CHAIN; declare
  ASYNC_COMPLETION only when Stage B is in.
- Kill the per-submit malloc/free: fold positions/tokens staging into
  the per-slot scratch (allocated at connect, like dsv4's pending).

### Stage B — runner async completion (nvcc, node-built on sparke)

- Replace the tail sync at the device-collective tier: the head
  exchange already rides stream-ordered completions
  (SparkTpDeviceCollective completion hooks, runner.cu:42-64,
  K3RunnerTpCompletion) — extend the same pattern to the step end:
  `cudaLaunchHostFunc(stream, K3RunnerCompleteAsync, slot)` (dsv4
  module.c:5712 shape), the slot's host staging does the tokens D2H,
  then fires the adapter completion.
- The TP4 host-collective fallback tier KEEPS the sync (it round-trips
  host TCP by design); async is the device-tier path — matching the
  TP16-first plan (the host tier tops out at 4 ranks anyway).
- Quiesce: poll the slot pool to drain (dsv4's deadline shape); the
  "runner is synchronous: nothing can be in flight" comment at adapter
  line 646 dies with it.

### Stage C — chain depth > 1 (on-device token feedback, 8 tok/submission)

Separate landing AFTER Stage A+B exactness: feed the sampled token
back on-device (dsv4's RESIDENT_DECODE_CHAIN proper), which K3_PERF's
capture work (item 1) also wants — graph capture and the async loop
compose (graphs must WRAP device programs, per PERF_PROGRAM P2's
lesson).

## Oracle + receipts (before touching, per PATTERN C discipline)

1. HOST SOURCE-CONTRACT (lands with Stage A, offline-green): extend
   the k3 driver-contract test to pin "submit does not fire the
   serving completion inline" and "the capability list composes via
   the CHAIN macro" — the test_k3_driver_contracts.py pattern.
2. NODE BUILD receipt (sparke, k3tp16-src tree): the adapter .so +
   runner compile with the LaunchHostFunc path; symbol check per the
   build-environment rule.
3. WINDOW EXACTNESS: TP16 equivalence gates unchanged (mismatch = RED
   stop) — async completion must be bit-identical on the drained
   tokens; then the p1d2 receipt surface discriminates:
   `model_residentd_exit ... max_ops_per_pass>1` on the k3 route (the
   same receipt p1d2 used; today k3 pins it at exactly 1).
4. Then timing: first TP16 number vs the 18 tok/s TP4xPP4 baseline /
   20.2 roofline — and ONLY then any claim.

## Sequencing note for the coordinator

P1 module half is parked behind the fleet closeout by the program's
own text; this plan does not jump that gate. It exists so the k3 lane
can execute Stages A+B in the first window after the closeout (or
sooner if the coordinator re-orders), with the oracle already green
and the receipts named.
