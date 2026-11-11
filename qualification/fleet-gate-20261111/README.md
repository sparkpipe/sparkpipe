# Fleet gate of 2026-11-11: the drafter split and the wave boundary's epoch

Model: Qwen3.8-27B, BF16 pack, TP16, 8 sequences, 1,024-row waves. A test
lane on mesh lane 5 beside production (qwen3.8-max stayed active on lane
12), started with `tools/stage_lane.sh` in model mode from its own root,
with its own API instance on the hub (port 8490). Prompts: the release
kit's seven (`tools/release_capture.py`), compared with
`tools/release_compare.py` on every path (one step at a time with top-4
logprobs, decode chains, speculation on, seeded sampled, all at once,
sampled all at once).

| Build | Commit | What it is |
| --- | --- | --- |
| previous head | `9170998ff` | the capture every run is compared with (code equal to `b2c838010`) |
| first try | `ab701281f` | the drafter split, before the epoch correction |
| gated | `7ceba5b55` | the drafter split and the epoch correction, firmware and API |
| final API | `28503c91f` | the same firmware; the API's seed lands on the ranks' floor |
| blocking API | not merged | `7ceba5b55` with the drafter call blocking again, for the arrival probe |
| pushed head | `3a1cff2da` | firmware and API: the unused speculation code removed, the engine's session check, the release kit's API restart |

## What the first try found

With `ab701281f`, the capture on a fresh lane matched the previous head on
seven of seven prompts. The gate then restarted the API on the serving lane,
which no gate had done before, and every request of the new API failed with
HTTP 500: each rank logged
`MESH-DEFERRED-ROUNDS-FAILED ... seq=1 error=<new epoch, 1> diag=<old epoch, 1303>`.
The old API binary failed in the same way, so the defect was on the ranks:
the wave boundary of 2026-11-10 let a wave's epoch fall when a new API
session numbered its submissions from 1,000,000 again, and weightd never
ships a tag below the last one of its cell. `docs/CODEBASE_AUDIT_2026-11-09.md`
has the cause and the correction.

## Outputs on the gated build

| Run | Against | Result |
| --- | --- | --- |
| no draft server | previous head | tokens and logprobs equal, 7 of 7, every path |
| replay drafter that knows nothing | no draft server | equal, 7 of 7, every path |
| replay drafter replaying the first run | no draft server | equal, 7 of 7, every path; 23 to 146 model drafts verified per prompt |
| final API, fresh lane | gated build | equal, 7 of 7, every path |
| final API after an API restart | the same API before it | equal, 2 of 2 prompts, every path |
| after rank 5's residentd restarted | before it | equal, 3 of 3 prompts, every path |
| after rank 0's residentd restarted | before it | equal, 3 of 3 prompts, every path |

The speculating run of a 32 to 48 token reply took 0.31 to 0.39 s with the
replayed drafts and 0.69 to 1.07 s without. Replayed drafts are always right,
so this shows the path, not a drafter.

## Restarts

Six API sessions ran on one start of the lane (the gated API with and without
a draft server, and the two APIs of the arrival probe at two drafter delays).
Every request was served; the ranks logged no `COLLECTIVE-EPOCH` refusal, no
failed round and no adapter error. The final API, restarted under the serving
lane, logged
`submission_id_seeded next=1000738 ... (above the highest submission the ranks have accepted)`
after a session of 737 submissions.

A rank's residentd was stopped and started through its node's `lane.sh`
(`stop`, then `start` with a new run id) while the API served, rank 5 and
then rank 0. The API reported fifteen connected ranks and sixteen again
within seconds; the first prefill after each restart was answered `PENDING`
once by the restarted rank, which was restoring the cached prefix, and the
engine retried it; the 849-token prompt resumed 848 tokens from the cache.

## Arrival probe

`arrival_probe.py PORT TAG ROUNDS TOKENS` runs one greedy stream twice (the
replay drafter learns it, then drafts it), measures a one-token request with
a prompt of its own thirty times alone, and then sends such requests back to
back while the stream runs, for ROUNDS streams. The draft server was
`tools/draft_replay_server.py --delay-ms 10` or `100`, with `depth` 15 and
`wait_ms` 30 in the API's `draft_server` block; the stream was 384 tokens and
ROUNDS was 4.

| Drafter delay | Measure | Blocking call | Split |
| --- | --- | --- | --- |
| 100 ms | arrival, median | 283.3 ms | 92.6 ms |
| 100 ms | arrival, 90th percentile | 346.7 ms | 108.5 ms |
| 100 ms | arrival, slowest | 394.5 ms | 132.9 ms |
| 100 ms | arrivals completed in four streams | 44 | 141 |
| 100 ms | stream with arrivals | 100-102 tok/s | 91-95 tok/s |
| 100 ms | stream alone | 101.2 tok/s | 100.8 tok/s |
| 10 ms | arrival, median | 130.4 ms | 131.9 ms |
| 10 ms | arrival, 90th percentile | 168.4 ms | 162.6 ms |
| 10 ms | stream with arrivals | 155-159 tok/s | 175-190 tok/s |
| 10 ms | stream alone | 188.5 tok/s | 196.9 tok/s |
| either | arrival alone, median | 48.1-48.3 ms | 48.9-49.4 ms |

The engine's mean wait for a draft read 10.35 and 100.41 ms with the blocking
call and 10.38 and 100.40 ms with the split. With the 10 ms drafter the
stream keeps more of its rate because an arrival that is dispatched at once
is no longer a waiting prefill when the stream's next frame is sized (three
quarters of its frames carry 14 to 16 rows, where 128 of 132 carried 12). With the 100 ms drafter the stream gives up 8% to
three times as many arrivals.

## The pushed head

`3a1cff2da` was built again for the ranks and the hub and gated on a fresh
start of the same lane (run `gh1111`).

| Run | Against | Result |
| --- | --- | --- |
| fresh lane, new firmware and API | gated build `7ceba5b55` | tokens and logprobs equal, 7 of 7, every path |
| after an API restart | the same API before it | equal, 2 of 2 prompts, every path |
| after rank 5's residentd restarted | before it | equal, 3 of 3 prompts, every path |
| after rank 0's residentd restarted | before it | equal, 3 of 3 prompts, every path |

The restarted API numbered from 1,000,737, above the 736 submissions of the
session before it. Each rank restart ended the session and began a new one:
the API log has two self-heals and two session changes, none with a
submission in flight, and one `PENDING` prefix restore that the engine
retried. Ranks 0, 5 and 9 logged no `COLLECTIVE-EPOCH` refusal, no failed
round and no adapter error.

The session check is what this head changes on that path. On the build host
the loopback fuzz had once hung a request (seed 1, round 21, a stopped rank
killed and restarted under a request, at `28503c91f`): the last rank
reconnected during the engine's own pipeline progress, the engine dispatched
on the new session before comparing it, and from then on refused to
invalidate with a submission in flight and never progressed the pipeline
again. The engine now compares the session after pipeline progress as well.
`tests/test_model_batch_engine_mock.c` brings a rank back at that point and
fails without the change.

## Host suite

On the build host at `3a1cff2da`: the whole tree builds, and of 538 C,
Python and shell tests 504 pass. The other 34 cannot run there: 29 import
the `tokenizers` module, which the build host lacks, and five list files
with git, which an archive tree is not. All 34 pass in the workstation's
checkout at the same commit. The loopback fuzz ran its 200 rounds.
