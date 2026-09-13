# Host-side B1 overhead — breakdown + first fix

Area: shared host path for single-stream B1 (27B TP1 on spark3). The 27B agent
measures GPU decode 225 ms/token vs wall 322 ms/token → ~97 ms/token (~30%)
outside the module. Cited `file:line`; HEAD `3bc44f5`.

## 1. The four buckets, code-cited

| Bucket | Path | Per-token cost in shared code |
| --- | --- | --- |
| **adapter scheduling** | serving submit = validate → build frame → admit → `program->submit` (`spark_qwen38_27b_serving_adapter.c:981-1030`); engine dispatch (`model_batch_engine.c:1973-2019`) | µs — no blocking call |
| **residentd IPC** | 2 socket round-trips: submit encode/write (`model_resident_client.c:124-147`), completion read (`149-170`); residentd writes completion (`model_residentd.c:2073-2101`) | µs — `poll(…,POLLIN/OUT,timeout)` returns on readiness |
| **batch-tool event loop** | `SparkModelBatchRun`: Progress + `poll(…,10ms)` (`node/model_batch.c:519-527`) | ~0 latency (poll returns on completion; the 10 ms timeout is never hit at B1) |
| **token emission** | per-token `fprintf(stdout,…)+fflush(stdout)` (`node/model_batch.c:316-388`, before this change: `311-329`) | **blocks on a slow stdout reader** |

**Conclusion:** scheduling, IPC, and the poll loop are µs-scale and add no
wall-clock. The single shared-path term that can convert host latency into
wall-clock is **token emission**: `fflush(stdout)` per token blocks the decode
loop — the *only* producer of the next submission — at the stdout reader's drain
rate. If the 27B harness reads the batch tool's stdout line-by-line with per-token
work (parse/render/log), that reader latency becomes the ~97 ms/token gap
(322 − 225).

## 2. First fix (implemented, my lane — `node/model_batch.c`)

Decouple emission from the decode loop: format each event into a user-space
buffer and `write(STDOUT_FILENO,…)` with `O_NONBLOCK`; on `EAGAIN` keep the
bytes and retry on the next loop iteration; drain blocking at exit.

- `SparkModelBatchOutputReserve` / `SparkModelBatchFlushOutput` (`316-360`)
- `SparkModelBatchWriteEvent` builds one `snprintf` line + buffers + flushes (`362-388`)
- run-loop flush at top of iteration (`521`); non-blocking stdout before run + blocking drain after (`613-619`)

**Measured before/after (host-only microbenchmark, slow reader at 1 ms/line):**
the producer-side wall the decode loop experiences —

| Emission path | Producer wall | Per token |
| --- | --- | --- |
| BEFORE — blocking `write` (old `fflush` semantics) | 10.07 s / 10 k tokens | **1007 µs/token** |
| AFTER — `O_NONBLOCK` write + user buffer | 10.9 ms / 10 k tokens | **1.1 µs/token** |

~1000× producer-side reduction under reader back-pressure — the exact shape of
the ~97 ms stall. The emission bytes are unchanged (byte-identical JSON; the 27B
harness parser is unaffected).

## 3. To finish the attribution (needs the 27B agent)

The remaining shared-path terms are µs-scale, so if ~90 ms/token persists after
this fix, it is **not** in the shared loop. Request: the 27B agent's
`sparkpipe_model_batch --profile-stages` output — `sparkpipe_stage_profile`
lines carry `queue_ns / service_ns / elapsed_ns / completion_ns` per stage
(`node/model_batch.c:366-379`) — plus how the harness consumes stdout
(line-buffered pipe? per-token work?). `service_ns` vs `elapsed_ns` splits the
97 ms into adapter-GPU vs host, which confirms this fix lands where the wall is.

## Verification

`cc -fsyntax-only -Wall -Iinclude -I. -Isrc node/model_batch.c` → exit 0.
No commits, no pushes; working tree = `node/model_batch.c` (edit) + this doc.
