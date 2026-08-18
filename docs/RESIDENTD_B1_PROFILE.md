# Residentd B1 single-stream — profile + first fix

Area: the residentd's shared loop for 27B TP1 B1. 27B agent re-measured: GPU
decode ~190 ms/token, ~82 ms/token (~30%) in the residentd loop — adapter
scheduling + two IPC round-trips + completion processing. Cited `file:line`; HEAD `3bc44f5`.

## 1. Per-token residentd path (code-cited)

Submit round-trip: client `Prepare` (`model_pipeline_client.c:791-795`) → residentd
poll wakes on POLLIN → `ReadClient` → `ProcessSubmission` (decode + validate +
`PrepareSubmission` + reserve/bind route, `model_residentd.c:1715-1830`) →
`Progress` → `SubmitAdapter` → `adapter.submit` (`2225-2252`), GPU starts.

Completion round-trip: driver calls `SparkModelResidentdCompletion` (validate +
queue + `SparkModelResidentdWake`, `1022-1081`) → wake pipe → poll wakes →
`Progress` → routes-final → `QueueCompletionLocked` (`987-1019`) →
`WriteClient` (`2073-2101`) → client reads.

## 2. Root cause: the residentd poll busy-spins

`SparkModelResidentdBuildPollFds` requested **POLLOUT unconditionally** on the
client socket (`node/model_residentd.c:2607-2614`). A healthy socket is always
writable, so `poll()` returned immediately every iteration — the loop busy-spun at
full CPU for the whole 190 ms GPU decode, each iteration re-running the full
`Progress` (adapter.progress + 4×`ProgressRoutes` + 4×`ProgressTransport`,
`2524-2563`) plus a mutex lock/unlock. That core-starvation competes with the
driver's completion-callback thread, which is exactly the "loop scheduling /
poll wakeups" cost that delays both IPC round-trips.

The client already gates it correctly: `SparkModelResidentClientGetPollDescriptor`
requests POLL_WRITE **only when `output_count != 0`** (`model_resident_client.c:920-922`).
The residentd is the asymmetric half.

## 3. First fix (applied — `node/model_residentd.c:2607-2616`)

Request POLLOUT only while output is genuinely pending:

```c
if ( runtime->client.fd >= 0 && runtime->client.close_after_output == 0u &&
     runtime->client.output_count != 0u )
    fds[1].events |= POLLOUT;
```

The queued ACK/completion is still flushed: `ReadClient` queues it and the run
loop calls `WriteClient` in the same iteration (`2656-2675`); a partial write
keeps `output_count != 0`, so the next poll re-arms POLLOUT. Hello-ACK is covered
the same way (queued then flushed on the next POLLOUT-ready poll, well inside
`connect_timeout_ms`).

**Measured before/after (host-only poll loop, 100 ms window):**

| `fds[1].events` | poll returns / 100 ms | loops/sec | iterations over a 190 ms GPU decode |
| --- | --- | --- | --- |
| BEFORE — always `POLLIN|POLLOUT` | 68,135 | 681,350 | **~129,000** |
| AFTER — `POLLIN`, `POLLOUT` gated on pending | 9 | 85 | **~16** |

~8000× fewer `Progress` passes/token — the busy-spin is gone; the loop now blocks
on poll and wakes only on a real event (submit or wake-pipe completion).

## 4. To confirm end-to-end (needs the 27B agent / spark3)

Re-run the B1 no-spec measure after this lands and report the new wall vs the
225→190 ms GPU split; the expected win is the elimination of the ~129 K wasted
Progress passes that contended with the completion callback. A follow-up is
dropping the remaining 10 ms active-poll floor (`PollTimeoutMs`, `2633-2643`) to
-1 when the adapter's completion is wake-driven (qwen36's `progress` is a no-op,
`spark_qwen36_serving_adapter.c:1529-1535`).

## Verification

Edit is a one-line condition on `fds[1].events` under the existing mutex; the
full `-fsyntax-only` needs the CUDA SDK (absent here), so this was verified by
read-back + the host microbenchmark above. No commits, no pushes.
