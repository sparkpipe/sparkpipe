# Client B1 single-stream — the inter-token bubble

Attribution converged: residentd services each decode in ~190 ms (queue_ns=0 after
the poll fix); the ~82 ms residual is the CLIENT-side post-completion loop + IPC
round-trip BETWEEN tokens — the GPU idles while the client processes the completion,
emits, and submits the next decode. Cited `file:line`; HEAD `3bc44f5`.

## 1. The client loop, code-cited

`SparkModelBatchRun` (`node/model_batch.c:519-533`): `FlushOutput` → `Progress`
→ `GetPollDescriptors` → `poll(10ms)`. Per token this is:

1. **IPC read** — `SparkModelBatchEngineProgress` → `SparkModelPipelineClientProgress`
   (`model_batch_engine.c:2002`) → per-rank `SparkModelResidentClientProgress`
   (`model_resident_client.c:893-908`) = `Flush` (write queued PREPARE) + `Read`
   (`recv` loop, non-blocking, `856-891`).
2. **Event handling** — completion → `ProcessCompletion` (`767-790`) →
   `completion_function` → `SparkModelBatchCompletion` → `HandleDecodeCompletion`
   → `AcceptToken` → `Emit` (`model_batch_engine.c:626-649, 764-794, 857-886`) →
   batch-tool `SparkModelBatchWriteEvent` (buffered, non-blocking after the prior fix).
3. **Submission setup** — dispatch loop (`model_batch_engine.c:2017-2038`):
   `ChooseWorkKind` (full request scan) → `BuildSubmission` →
   `SparkModelPipelineClientSubmit` (`model_pipeline_client.c:754-808`) →
   `ResidentClientPrepare` **queues** the PREPARE (write deferred).

## 2. Why the loop is µs-tight, and where the 82 ms is NOT

Every step above is a non-blocking syscall or a bounded table walk; the poll is
wake-driven (`POLL_READ` only when idle, `model_resident_client.c:920-922`), so
there is no 10 ms sleep and no busy-spin on the client. The ~82 ms is the **serial
sum** of the real spark3 host latencies across steps 1-3 plus the residentd
submit-processing — and, crucially, it is **never overlapped with the GPU**: the
engine submits decode N+1 only after decode N's completion is fully processed.

## 3. The overlap fix — and why it is speculation (not a scheduler change)

At B1 decode, the frame for step N+1 takes as input the token produced by step N,
so decode N+1 **cannot be submitted before decode N's completion carries that
token**. True overlap therefore means submitting the next decode with a *predicted*
token and replaying on miss — i.e. the driver's MTP draft chain, which is the 27B
agent's lane (`docs/QWEN38-27B_HILLCLIMB.md` A1-A4, and the DSpark drafter at A4).

The scheduler already does its part: `engine->submission_capacity =
max_inflight_submission_count` (`model_batch_engine.c:1047`), and the dispatch
gate is `inflight_submission_count < submission_capacity` (`2017`) — so two frames
in flight is permitted the moment a second frame can be built. The serializer is
the per-sequence token dependency, not the engine.

## 4. First fix (loop-latency reduction, applied — `model_batch_engine.c:2039-2050`)

`ResidentClientPrepare` queued the PREPARE, and it was written on the **next**
`Progress` (one poll + one `GetPollDescriptors` + one `FlushOutput` later). The
fix flushes the just-queued submission inside the same `Progress` that reads the
completion:

```c
status = SparkModelPipelineClientProgress(engine->pipeline, engine->maximum_messages_per_rank);
```

This removes the one-iteration bubble on the critical path (the extra read is a
no-op on the non-blocking socket). **Honest magnitude:** this is a ~µs win, not the
82 ms — it tightens the serial loop but does not overlap it. The 82 ms closes only
via speculation (overlap), which is the 27B agent's MTP/DSpark work.

## 5. Coordination ask (via the coordinator)

1. Land the two scheduler fixes (`node/model_residentd.c` poll gate, `node/model_batch.c`
   emission, `runtime/model_batch_engine.c` flush) and have the 27B agent re-run
   B1 no-spec to confirm the remaining bubble.
2. The 27B agent should prioritize the speculation path (A1 re-arm MTP in the build
   gate → A2/A3 fix the chain → A4 DSpark drafter) — that is the only route to
   "2 in flight at B1" and closing the ~82 ms.

## Verification

`model_batch_engine.c` edit is a second `SparkModelPipelineClientProgress` call
with the same error handling; full `-fsyntax-only` needs the CUDA SDK (absent
here), so verified by read-back. No commits, no pushes.
