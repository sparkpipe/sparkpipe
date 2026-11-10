# Asynchronous KV finish: design notes for audit 2.6

Audit finding 2.6 (`docs/CODEBASE_AUDIT_2026-11-09.md`) says a publish frame
blocks the wave while the KV side writes 9.3 MB (GLM 5 Next at TP16) to 28 MB
(K3) of recurrent state per publishing lane, and that
`SparkStageKvBindingFinishAsync` exists unused. Target step 2 asks that routes
complete from the KV completion callback, so residentd serves its socket during
the frame. This page records how the code behaves at `bc2ffc8b4`, mapped by
three independent readers on 2026-11-10, and the rules a change must keep so
that no token, logprob, admission, cancellation or error result moves. Every
line number is at `bc2ffc8b4`.

## Where the time goes

There are two finish paths, and the audit's bytes are on the second.

- **Row frames** (prefill, decode chains, verify, tree). After the runner
  returns, the adapter enqueues the recurrent capture (device state into the
  pinned `lane_state` image) and the token copy on the execution stream and
  waits for it (`inference/runner/stage_serving_adapter.c:1852-1863`). It then
  calls `StageServingFinish`, which is `SparkStageKvBindingFinishWait`: it
  queues a completion record with `FinishAsync` and blocks on a stack waiter
  until the binding's completion thread has run `WriteStates`,
  `ApplyCompletion` and `PersistCheckpoints` (`runtime/stage_kv_binding.c:2670-2700,
  2880-2937`). On the Qwen3.8-27B lane this phase is `finish_ms` in
  `STAGE-CHAIN-HOST` and `STAGE-WAVE-HOST`: 0.2-0.4 ms per decode chain and
  0.3-1.2 ms per prefill wave.
- **Cache publications** (`CACHE_PUBLISH` frames). `StageServingSubmitNonRow`
  calls `SparkStageKvBindingPublishFrame` (`stage_serving_adapter.c:1411-1431`;
  `stage_kv_binding.c:2783-2819`). It copies each lane's live device state on
  the legacy default stream and synchronizes it, then writes one state record
  per lane through the state store (memcpy, SHA-256 and pwrite in the store's
  worker, which the caller waits for), finishes the lanes and calls the
  completion inline. All of it runs on residentd's only thread, with no
  completion record, so `FinishAsync` does not reach it.
  `STAGE-PUBLISH-HOST` (`ef8536924`) reports its cost per frame and per lane.

## What residentd already supports

residentd is built for a completion that arrives after submit returns, from
another thread (`node/model_residentd.c`):

- submit runs on the main thread with `runtime->mutex` dropped (2882-2887); an
  OK return with the route still `WAIT_ADAPTER` is accepted (2903), and
  `ProgressRoute` skips such routes (3104-3108);
- the callback `SparkModelResidentdCompletion` (1304-1337) takes only
  `runtime->mutex`, matches the route by submission id and `WAIT_ADAPTER`,
  validates identity and residency, copies the completion by value and writes
  the wake pipe;
- an adapter that declares `ASYNC_COMPLETION` gets the committed FIFO heads
  until it answers BUSY; BUSY keeps the head and ends the pass (2890-2894,
  3184-3188); any other error means no callback will come;
- claims on resident slots are released only when the completion is queued
  (`QueueCompletionLocked` -> `CompleteResidentSlotsLocked`, 1244, 1037), so a
  route waiting on the adapter keeps its slots, and reset waits for active
  routes.

The stage serving adapter does not declare `ASYNC_COMPLETION`
(`stage_serving_adapter.c:2072-2082`) and completes every frame inline.

## Rules an asynchronous finish must keep

1. **Deliver residentd's completion only after `ApplyCompletion`.** Delivering
   earlier frees the slot claims, so the coordinator's next frame for the lane
   could reach PREPARE while the lane is still `EXECUTING` (BUSY instead of
   admission), read stale `lane_next_positions` (continuity fails), or reach a
   RELEASE whose `CommitStaged` runs before the lane-phase check and writes a
   later state as an earlier page's checkpoint (`stage_kv_binding.c:1856-1861,
   2435-2479`), which changes tokens on a later prefix hit.
2. **Release the adapter's `lane_states` claims after the KV completion and
   before the delivery.** Releasing them when submit returns lets a same-slot
   frame or publication capture or restore into the `lane_state` image that
   `WriteStates` is still reading (`stage_serving_adapter.c:1822-1825, 1908`).
3. **Build the completion's content before the next submit can run.** The
   tokens, scores, logprobs, sideband and service time are read from host
   buffers (`tokens_host`, `runs_host`, `order_host`, the tree tables,
   `path_rows`, the distribution rules) and stack arrays that the next submit
   overwrites or that die when submit returns (`stage_serving_adapter.c:1701-1744,
   1756-1760`). Copy the finished completion into the record's own storage
   before `FinishAsync`, and set only its status from the KV result.
4. **A KV-side failure must give the same completion as today.** Today a
   failed finish makes submit return the error and residentd synthesizes a
   failure with the identity, residency and zero tokens, flags and accepted
   count (`model_residentd.c:3189-3227`). After `FinishAsync`, the adapter must
   deliver that same failure through the callback and never return an error
   from submit; a completion after an error return is dropped as
   `COMPLETION-STRAY` and the route's FIFO entry is left behind (1260-1264,
   3296-3350).
5. **One record means BUSY, never INTERNAL_ERROR.** The binding has
   `pipeline_slot_count = 1` and the adapter always finishes on slot 0
   (`stage_serving_adapter.c:968, 1486`). A second frame that reaches
   `FinishAsync` while the record is queued gets INTERNAL_ERROR and its lanes
   stay `EXECUTING` for good. Submit must check the record and answer BUSY
   before it admits or claims anything; the callback's wake reruns the route.
   More records need explicit slot ownership, since
   `request_id % pipeline_slot_count` can map two frames to one slot.
6. **Move the deferred-prefill bookkeeping into the callback.** `saves_held`
   and `streamed_failed` are decided from the finish status
   (`stage_serving_adapter.c:1890-1900`) and read by the next submit, reset and
   snapshot; they need atomics or a lock once the callback sets them.
7. **Quiesce waits for the binding.** `StageServingQuiesce` returns OK at once
   (`stage_serving_adapter.c:1965-1970`) and residentd then closes transports
   and destroys the adapter. It must return `SparkStageKvBindingQuiesce`'s BUSY
   while records are pending. The callback must never wait on the completion
   thread itself (`FinishWait`, `Quiesce`, `Stop` or `Destroy` from inside it
   deadlock or join themselves).
8. **Expect the interleavings that are new.** With residentd's thread free,
   PREPARE, DECISION, HINT, RELEASE and status snapshots run while the
   completion thread writes states. `WriteStates` and `RecurrentAdmit` drop
   the binding mutex around state-store I/O, which has one transfer job and one
   staging buffer (`stage_kv_binding.c:2500-2506, 2318-2324`;
   `cache/kv_page_store.c:701-726`). A PREPARE that restores a lane waits
   behind the completion thread's writes, and under KV pressure a PREPARE sees
   the frame's pages still pinned. These change timing; a test must show they
   do not change admission results.
9. **Cache publications need their own asynchronous form.** Their device copy
   runs on the legacy default stream, unordered with the non-blocking
   execution stream (`stage_kv_binding.c:2733`; `stage_runner.cu:3136`). An
   asynchronous publication copies on the execution stream behind an event,
   lets the completion thread wait for the event, write the states and finish
   the lanes, and then follows rules 1 to 7.
10. **The device serving runner shares the adapter.** Metal's runner refuses
    DEFER, reports no capture time and copies recurrent state on its device
    queue (`inference/runner/device_serving_runner.c:230-309, 367-371`); its
    end-of-frame waits must be checked before relying on the CUDA ones.

## Order of work

1. Measure: `STAGE-PUBLISH-HOST` on a recurrent lane under publishing traffic
   gives the cost per publication and per lane, next to `finish_ms` for row
   frames. Rank the two paths by the time they hold residentd's thread.
2. Cache publications first if they carry the bytes: the event-ordered copy,
   the completion-thread write and finish, the BUSY rule and the completion
   storage (rules 1-5, 9).
3. Row frames on the same machinery (rules 1-8), then declare
   `ASYNC_COMPLETION`.
4. Tests: the binding (a frame that meets an occupied record gets BUSY and no
   lane changes; a failed `WriteStates` delivers today's failure; quiesce
   waits), the adapter against a recording callback (tokens, logprobs and
   sideband equal to the inline path on every frame kind, including verify,
   tree and deferred prefill), and residentd with the asynchronous adapter
   (`tests/test_model_resident_adapter_budget.c` already pins the contract).
5. Fleet: a recurrent lane (Qwen3.8-27B) through the release kit and a reset
   and cancellation run, compared with the previous head on every path, and
   the frame and publication costs before and after.
