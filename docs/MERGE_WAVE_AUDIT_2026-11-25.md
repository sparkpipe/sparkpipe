# Audit of the changes merged on 24 and 25 November 2026

This audit covers the 113 pull requests merged to `main` between
2026-11-24 00:11 UTC and 2026-11-25 06:54 UTC (numbers 2129 to 2231, and
twelve older ones merged in the same window), the eleven releases rolled to
the sixteen-node fleet in that time, and the benchmark receipts published for
them. It is written for the author of those changes: each finding says what
was observed, where the evidence is, and what would close it.

The short version:

- **Two regressions introduced in the window were found and fixed by their
  author within hours,** and the benchmark's own gates caught a third fault.
  Section 2 records what went well, so it is not reported as a finding.
- **The largest open problems are not in any one kernel.** They are that
  published benchmark receipts do not show what a release did (F1), that
  replies changed between builds with no record of whether they should have
  (F2, F3), and that the sixteen ranks of the two Kimi lanes stopped running
  in step while Kimi K3's long-context decode became 48% slower a step (F4).
- **Nothing here was measured by running the fleet.** Every number comes from
  logs and records the fleet and CI already keep. Section 6 says how to read
  them, so each finding can be checked, and section 7 says what was not read.

All times are UTC. Builds are named by the commit their release was cut from.
The tree was read at `ceafafd1e`.

## 1. What to do first

1. Publish receipts that hold every attempt of the named build and nothing
   else, and let the scorer refuse anything else (F1).
2. Before the next rollout, run the rollout's own comparison steps (`old`,
   `qualify`, `gx` in `scripts/release_rollout.sh`) and keep their output in
   the release root (F2).
3. Find why cached and cold replies differ on GLM 5.3, A.X K2 and Kimi K3
   (F3). It is intermittent, so it needs repeated runs, not one.
4. Find why the ranks of the Kimi lanes are out of step since `99fc2a2c1`
   (F4). Give the fast collective launcher an off switch first, so one run can
   settle whether it is the cause.
5. Bound the captured-graph cache and give the Kimi lanes memory headroom
   (F5).
6. Get `main` green and keep it green on a small core before merging (F6).

## 2. What went well

- **The repository rules held.** At `ceafafd1e` the no-comments, shared-code
  naming, device-neutral, test-registration and code-size checks pass.
- **Tests were added with the code.** No test file was deleted in the window.
- **Two regressions were diagnosed by their author from production shapes.**
  Pull request 2209 names the same two causes for the Kimi K3.1 Pro slowdown
  that this audit isolated independently (F7), and the fleet confirms the
  fix. Pull request 2208 repairs the NVIDIA-free launcher, and CI now
  confirms it (F8).
- **Two kernel faults were fixed within about two hours of merging:** the
  shared-pointer atomic in `LmIndexSelectMergeKernel` (2182, fixed by 2185)
  and `__ldg` on shared offsets in `LmSkinnyCellGroup` (2184, fixed by 2186).
- **A wrong benchmark run was repeated rather than published.** The first run
  of `dff7b5a96` had speculation on for plain requests; it was run again
  without (F1 uses it only to show what the gates do not catch).
- **A speculation fault caught by the benchmark's own gate was fixed.** Qwen
  3.8 27B failed `prose_exact` at `454edc951` and passes at `dff7b5a96`.

## 3. Findings

Each finding has a status as of `ceafafd1e` and release `dff7b5a96`.

| | Finding | Status |
|---|---|---|
| F1 | Published receipts do not show what the release did | open |
| F2 | Replies change between builds and nothing compares them | open |
| F3 | Cached and cold replies differ, intermittently | open |
| F4 | The ranks of the Kimi lanes are out of step | open |
| F5 | Kimi lanes run nodes close to out of memory; the graph cache is unbounded | open |
| F6 | `main` has been red throughout; merges did not wait for CI | open; the compile gate's hidden error is fixed in 2228 |
| F7 | Kimi K3.1 Pro multi-row regression | fixed by 2209, confirmed on the fleet |
| F8 | NVIDIA-free build break | fixed by 2208, confirmed by the compile job of 2228; one hazard left |
| F9 | The benchmark cannot carry the comparisons made on it | open |
| F10 | Smaller points | open |

### F1. Published receipts do not show what the release did

A receipt under `qualification/releases/<build>/` is read as "this build,
measured". Three things break that.

**A receipt mixes builds.** `qualification/releases/07a6e6bcd/rais-fleet-16x-gb10-1x100g.jsonl`
has 21 rows. By their own `engine` field, 3 were measured on `07a6e6bcd`, 14
on `4fc4a10e3`, 3 on `98386babe` and 1 on `b8df69a2e`, between 13:45 and
19:29 on 24 November. Pull request 2189 published it as the fleet's scores.

**Failing rows were replaced.** On `07a6e6bcd` Kimi K3 was measured twice
(19:12:59 and 19:46:59). Both rows have `prefix_exact: false` and
`errors: 2`. The receipt carries Kimi K3's row from `4fc4a10e3` instead. On
`dff7b5a96` GLM 5.3 (05:49:57) and A.X K2 (06:10:04) failed `prefix_exact`;
both were measured again (06:33:00, 06:41:39) and passed; the receipt holds
only the passing rows, and pull request 2230 states 19 of 19 models valid.
The failing rows are in the benchmark files on the hub
(`hillclimb/rais-07a6e6bcd-*.jsonl`, `hillclimb/rais-dff7b5a96-fleet21-exact.jsonl`).

**A run can be wrong with every gate true.** The first full run of
`dff7b5a96` (04:21 to 05:11) passed every gate on all 21 lanes with no
errors. Its long-context decode figures were two to four times those of
every other run (Kimi K3.1 Pro 58.6 against 14.8 tokens a second, Kimi K3
49.5 against 11.8, Qwen 3.8 27B 106.2 against 45.1), and batches of 32 and 64
were recorded as unsupported on every lane. The Kimi K3.1 Pro API log shows
why: 247 of that run's decode submissions were verify steps and 5 were plain
steps, against 88 and 103 in the repeat. Plain requests were being
speculated, so `prose_exact` compared a speculative reply with another
speculative reply.

Why it matters: the site and the score are built from these files. A reader
cannot tell a clean release from one that failed a gate and was measured
again, and a row from an older build can stand in for a build that failed.

What would close it:

- A receipt holds every attempt made on the named build, failures included,
  and only that build. A lane that failed once and passed once is shown as
  such.
- `tools/rais_score.py` refuses a receipt whose rows carry more than one
  `engine` value, or an `engine` that differs from the release named.
- The build identity comes from the server, not from `--engine` on the
  command line (F9).
- The benchmark records, for each row, the lane settings it ran against
  (speculation default, sequence limit), and `prose_exact` fails when the
  "plain" request was served with drafts.

### F2. Replies change between builds and nothing compares them

The benchmark sends the same three short prompts at temperature 0 on every
run, and the API log keeps the token ids of every reply
(`request_measurements`, keyed by `prompt_sha256`). Comparing the 128 reply
tokens of each prompt across builds, a letter standing for one distinct
reply:

| Lane | Prompt | `b8df69a2e` | `4fc4a10e3` | `07a6e6bcd` | `d99d5cc81` | `99fc2a2c1` | `454edc951` | `dff7b5a96` |
|---|---|---|---|---|---|---|---|---|
| Kimi K3.1 Pro | prose | A | A | B | B | C | D | D |
| Kimi K3.1 Pro | code | A | A | B | B | C | D | D |
| Kimi K3.1 Pro | edit | A | A | A | A | B | A | A |
| Kimi K3 | prose | n/a | n/a | B | B | C | D | D |
| Kimi K3 | code | n/a | n/a | B | B | B | A | A |

`n/a` is a run this audit could not tie to a build. `d99d5cc81` and
`dff7b5a96` were each run twice and gave the same reply both times on every
lane compared, so the changes come from the builds, not from noise. The
other lanes show the same picture over their last nine runs: A.X K2 gave
five distinct prose replies and five code replies, GLM 5.3 four and four
(one differs from the first at reply token 0), DeepSeek V4.1 Flash four and
three, GLM 5.3 Flash three and three, Qwen 3.8 Max three and two, Qwen 3.8
27B two prose replies.

A changed reply is not by itself a fault: a kernel that sums in another
order changes low bits, and some changes are corrections. The fault is that
no rollout recorded that replies changed, which change did it, or whether
the new reply is the right one. `scripts/release_rollout.sh` has steps for
exactly this (`old` captures the outgoing build, `qualify` compares old with
new and runs the golden comparison and the reasoning check when they differ,
`gx` runs the cached-prefix gate). The release roots of earlier releases on
the hub hold their per-model captures and comparisons. Those of
`07a6e6bcd`, `d99d5cc81`, `99fc2a2c1`, `454edc951` and `dff7b5a96` hold only
`kit` and `tree`.

The two kernel faults listed in section 2 were on the fleet for one to two
hours each (releases `c8e2ce7cc` and `98386babe`) and were found when ranks
failed with CUDA error 717, not by a comparison.

What would close it: run `old`, `qualify` and `gx` on every rollout and keep
their output in the release root; when replies differ, the pull request that
changed them says so and says why the new reply is the right one.

### F3. Cached and cold replies differ, intermittently

`prefix_exact` asks whether a reply after a cached prefix equals the reply
to the same prompt cold. It failed on:

| Build | Lane | Attempts | Failed |
|---|---|---|---|
| `07a6e6bcd` | Kimi K3 | 2 | 2, each with 2 request errors |
| `dff7b5a96` | GLM 5.3 | 2 | 1 |
| `dff7b5a96` | A.X K2 | 2 | 1 |

On `dff7b5a96` the same build failed and then passed on the same lane within
an hour, so the cached path does not give one answer. This is the property
the product promises (a cached prompt answers as a cold one does), and a
single passing run does not show it holds.

Separately, at `454edc951` Qwen 3.8 27B answered differently with and
without speculation (`prose_exact: false`; the replies part at token 12).
That build was rolled to the fleet. `dff7b5a96` passes.

What would close it: run the cached-prefix comparison ten times a lane on
GLM 5.3, A.X K2 and Kimi K3 and find the first differing token and the step
that produced it; keep every attempt in the receipt (F1); do not roll a
build whose gate is false.

### F4. The ranks of the Kimi lanes are out of step

Every rank logs, about every 64 steps, how long a step took and how much of
it was waiting for peers (`STAGE-STEP-TIMING`). Subtracting the waits leaves
each rank's own compute time. With tensor parallelism inside every layer,
all sixteen ranks do identically shaped work, so their compute times should
agree. Until `d99d5cc81` they did, to within 1% in the median interval.
Since `99fc2a2c1` they do not, and every collective waits for the slowest
rank.

Fifteen ranks were read (rank 8's log could not be read for this audit, so
where rank 8 is the slow one the spread below is an underestimate).
Intervals are classed by step time: under 80 ms a step as single-row, the
rest as multi-row.

| Lane | Measure | `d99d5cc81` | `99fc2a2c1` | `454edc951` | `dff7b5a96` |
|---|---|---|---|---|---|
| Kimi K3.1 Pro | rank 0 time waiting for peers, share of a step | 33% | 45% | 48% | 43% |
| Kimi K3.1 Pro | slowest rank over the median, multi-row intervals (median / worst) | 0.7% / 12% | 13% / 25% | 18% / 95% | 7.5% / 40% |
| Kimi K3 | rank 0 time waiting for peers, share of a step | 35% | 36% | 40% | 41% |
| Kimi K3 | slowest rank over the median, single-row intervals (median / worst) | 0.8% / 1.2% | 0.7% / 0.8% | 1.5% / 35% | 5.0% / 8.3% |
| Kimi K3 | slowest rank over the median, multi-row intervals (median / worst) | 0.5% / 0.9% | 4.3% / 16% | 8.6% / 31% | 6.0% / 23% |

In Kimi K3's single-row decode intervals the median rank's compute time
stayed within a millisecond of `d99d5cc81`, so the kernels are not slower
there; one or two ranks are, and which ones changes from interval to
interval.

What it costs, from the API log's per-step service times (`batch_step`):

| Kimi K3, one request | `d99d5cc81` | `99fc2a2c1` | `454edc951` | `dff7b5a96` |
|---|---|---|---|---|
| Decode step, short context | 59.6 ms | 59.5 ms | 60.1 ms | 61.5 ms |
| Decode step at 22,700 tokens of context | 59.7 ms | 59.4 ms | 87.8 ms | 88.1 ms |
| `b1_long_tok_s` in the benchmark | 16.4 | 16.7 | 12.5 | 11.8 |

Kimi K3's long-context decode is 48% slower per step since `454edc951` and
still is in the certified release. Pull request 2204 rewrote the attention
kernel this path uses and measured it 1.78 times faster on one GPU; on the
fleet the step got slower. The rank logs average 64 steps a line, so this
audit could not separate the long-context steps from their neighbours, and
it leaves open how much of the 28 ms is ranks out of step and how much is
the new kernel on the fleet's shapes.

Six pull requests separate the last in-step build from the first one out
of step (2197 to 2202). A kernel that computes more slowly slows every rank
alike, which is not what the logs show, so the candidates are the changes
that can make single ranks late, and they are not exclusive:

- **The collective path (2199, 2200).** 2199 adds the fast collective
  launcher and 2200 changes the release order in the mesh kernels. The
  launcher is installed by a constructor (`SparkTpMeshFastLaunchersInit` in
  `inference/runner/mesh_fast_kernels.cuh`) with no way to turn it off, so
  it cannot be ruled in or out by a run.
- **Memory pressure on the nodes (F5).** The two Kimi lanes leave the least
  free memory of all lanes. Seen from one node, GLM 5.3, Qwen 3.8 Max, A.X
  K2 and Solar Open 2, which keep 11 GiB or more free, showed no new spread
  at `99fc2a2c1` or `454edc951`; K-EXAONE 2.0, with 18 GiB free, did at
  `454edc951`, so memory is not the whole story. The NVIDIA driver logged
  bursts of
  `Out of memory [NV_ERR_NO_MEMORY]` on each of the three nodes checked when
  the Kimi K3.1 Pro run of `99fc2a2c1` started and on one of them when that
  of `454edc951` started, and none when the in-step run of `d99d5cc81`
  started.

What would close it:

- Add a lane setting or environment switch that keeps the reference
  launcher, and run Kimi K3 once each way on one build.
- Have the benchmark read the rank logs and report, for each run, rank 0's
  waiting share and the slowest rank over the median. A run with ranks out
  of step should not be certified as the lane's speed.
- Measure kernel changes on the fleet's shapes with all sixteen ranks before
  recording a gain (F9).

### F5. Kimi lanes run nodes close to out of memory; the graph cache is unbounded

- **Headroom.** During a four-minute benchmark the Kimi K3 lane's lowest
  available memory per node was 6.0 to 6.4 GiB on the three earlier builds
  and 5.1 GiB on `dff7b5a96`; its lowest free GPU memory was 3.0 GiB at the
  median and 0.7 GiB on one rank (`STAGE-MEMORY`). Kimi K3.1 Pro went from
  7.1 to 3.8 GiB available on rank 0 in one run.
- **What uses it.** The runner keeps one captured program for every
  (key, row count) it has served and never drops one
  (`inference/runner/stage_runner.cu`). Pull request 2150 raised the keys
  from 3 to 10 (`STAGE_RUNNER_GRAPH_KEYS`), so the table went from 768 to
  2,560 entries. On the Kimi lanes a multi-row program costs 76 to 81 MiB,
  and one benchmark run captures 46 to 64 of them. About fifty more shapes
  exhaust a node.
- **What happens at the limit.** A capture that fails prints
  `STAGE-PROGRAM-RECORD-FAILED ... fatal` and calls `abort()`, taking the
  rank and with it the lane.
- **Concurrency went up at the same time.** Pull request 2176 raised every
  lane's `sequences` in `deployment/lanes.json` from 4, 8 or 16 to 32 or 64,
  and 2178 set the DeepSeek V4.1 Flash state budget to 32 GiB. More
  sequences means more row counts served, so more programs.
- **The nodes show it.** Fourteen of the sixteen nodes count two to four
  processes killed for lack of memory between 23 and 25 November
  (`oom_kill` in `/proc/vmstat`), one of them the DeepSeek V4.1 Flash lane's
  model process (24 November 21:48:27), and 0.5 to 3.1 million pages
  swapped out a node.

A benchmark run of four minutes does not reach the limit. A lane serving
varied batch sizes for hours will.

What would close it: evict programs (least recently used, within a byte
budget taken from the lane's memory plan) or capture only the row counts a
lane is configured to serve; refuse a request rather than abort when a
program cannot be recorded; state each lane's free memory at full
concurrency in its qualification and fail it below a floor.

### F6. `main` has been red throughout; merges did not wait for CI

- **No green run.** The last successful CI run on `main` is `867895e32`, 22
  November 20:30. `main` was already red when this work began (an argument
  mismatch from pull request 2127, not part of it). Of the 118 runs since,
  40 failed, 76 were cancelled by the next merge and 2 were still running
  when this was read.
- **Merges did not wait.** Of the 101 pull requests opened in the window, 92
  merged within five minutes of opening (median one minute). None carries a
  review.
- **The compile job hid its own error.** Since pull request 2181 it has
  stopped in the gate's "stage-runner binaries" phase, first on spill-audit
  refusals in two drafter modules, then, since 2194, on
  `tools/dsv41_flash_decode_replay.cu` launching a kernel that 2194 folded
  into another. The job page showed neither: the gate printed the first
  twenty log lines containing "error", which are source lines of warnings.
  Pull request 2228 makes every phase print the real error and repairs the
  tool; it is in review.
- **What had not run because of that.** The gate's later phases
  (dependent-wait audit, NVIDIA-free launch arguments, the NVIDIA-free
  release root, the spill audit over the adapters, the staged roots) have
  not run on `main` since 2181, and ten test binaries in
  `STAGE_RUNNER_BINARIES` (the verify-frames tests, the fused-exact test,
  the tap-plan test) were not built.
- **What they hold.** The compile job of pull request 2228 gets past the
  stop on `main` at `21bdba99a`. All 53 stage-runner binaries build, and the
  dependent-wait audit, the NVIDIA-free launch arguments and the NVIDIA-free
  release root pass. The spill audit then refuses nine attention kernels
  that keep a stack frame: two window-fold kernels that 2194 added
  (`D41WindowFoldSpreadKernel<4, 2>`, 32 bytes;
  `D41WindowFoldSpreadClusterKernel<4>`, 48 bytes) and seven
  `LmLatentShardPrefillKernel` instantiations of the Kimi family (48 to 72
  bytes). Six of the seven are frames the audit's pending list already
  records under the view type `LmKvShardGatherView`; 2182 moved the family
  to `LmKvShardGatherAllView` and the list was not updated, and one of them
  grew from 64 to 72 bytes. The staged-roots phase has still not run.
- **Four repository checks fail on `main` itself:**
  `tests/test_complexity_ceiling.py`
  (`SparkModelBatchPlanBatchDrafts` in `runtime/model_batch_engine.c` at
  complexity 101, ceiling 73), `tests/test_memory_contracts.py` (eight new
  entries: `tools/mesh_round_bench.cu`, `tools/rais_score.py`, one drafter
  module) and `tests/test_upload_ordering_sites.py` (copy sites in
  `inference/runner/stage_runner.cu` and
  `ring/transport/tp_device_collective.c` that 2199 changed and did not
  classify in `docs/UPLOAD_ORDERING.md`), and, since 2229,
  `tests/test_no_build_defaults.py` (a default for `GLM53_FLASH_PROGRAM` in
  the GLM 5.3 Flash serving module).
- **One failure hides the rest.** The light and host test jobs stop at the
  first failing test, so every test after the complexity check in
  `PYTHON_TESTS` has not run in CI.
- **The size budget no longer bounds anything.** The ceiling is the base
  plus every grant file under `tests/code_size_budget/`. The window added 59
  grant files worth 37,003 lines; grants total 64,507 lines over a base of
  380,539. One commit after the tree this audit read, 2231 put `main` 35
  lines over even that, so `tests/test_code_size.py` is a fifth failing
  check.

Why it matters during fast-moving work: the checks that would have caught
the replay tool, the spill refusals and the unrecorded copy sites existed
and were red, so they told no one. A small core that is always green (the
compile gate, the repository checks, the host tests) is the only signal
that survives a hundred merges a day.

What would close it: merge 2228; remove the nine frames (the pending list
may only shrink); fix the four checks; then merge only on a
green core, and let runs on `main` finish (76 of 118 were cancelled),
merging in batches if the queue cannot keep up.

### F7. Kimi K3.1 Pro multi-row regression (fixed)

Recorded because of what it shows about measuring, not because it is open.

| Kimi K3.1 Pro | `d99d5cc81` | `99fc2a2c1` | `454edc951` | `dff7b5a96` |
|---|---|---|---|---|
| 2,048-row prefill wave | 3,570 ms | 4,482 ms | 4,258 ms | 3,694 ms |
| Prefill of the 8K prompt, tokens a second | 531 | 452 | 479 | 570 |
| Batch of 8, tokens a second | 56.4 | 45.9 | 52.8 | 53.1 |
| Decode step, one request, short context | 63.7 ms | 65.1 ms | 67.7 ms | 64.7 ms |

- **Cause.** Pull request 2201 dropped the `k_tiles < 32` condition in
  `LmCellMmaLaunchKernel` (`inference/kernels/cell_mma.cuh`), so the wide
  first expert projection of this model (112 K-tiles, 896 groups, 37 rows a
  group in a 2,048-row wave) moved from the kernel that covers 128 rows a
  pass to one that covers 64 (32 on sparser waves), and a busy expert read
  its weights two to four times as often. Pull request 2198 changed the
  small-batch cell path. Kimi K3 loads the older expert format and was not
  affected.
- **Fix.** Pull request 2209 restores both, and the fleet confirms it.
- **Lesson.** The shape 2201 measured as "k31" was 24 groups of 256 rows
  with K 7,168 and N 512. Production is 896 groups of 37 rows with K 3,584
  and N 384, which is the branch the change altered. 2209 measured the
  production shape, on one GPU; the fleet run that confirmed it came with
  the next release.

### F8. NVIDIA-free build break (fixed, one hazard left)

- **Cause.** Kernel clean-ups gave shared kernels default parameters (22
  launch sites) and overloads. The NVIDIA-free flavour of `LM_LAUNCH` takes
  the kernel as a function pointer, which drops defaults and cannot choose
  an overload, so 14 of 27 adapters stopped compiling in that flavour.
- **Fix.** Pull request 2208 reworks `runtime/device/nvrm/launch_nvrm.h`.
  The compile job of pull request 2228 is the first to reach the
  NVIDIA-free phases since 2181, and they pass there (F6).
- **Hazard left.** For a kernel with default parameters the launch
  arguments are evaluated twice: once when bound to the dispatch call and
  once inside the stub call that recovers the converted arguments
  (`LM_LAUNCH_CALL`, the `__lm_fb` lambda and the dispatch both expand
  `__VA_ARGS__`). None of the 838 launch sites scanned at `e22005456` has a
  side effect in its arguments. One that gains `++`, an assignment or a
  call with an effect will run it twice, in this flavour only.

What would close it: bind the arguments once (forward the captured tuple to
the stub), or add a repository check that refuses a side effect in a launch
argument.

### F9. The benchmark cannot carry the comparisons made on it

- **One sample.** Every figure except the short single-request decode is one
  measurement of 64 to 128 tokens. Between two runs of one build fifty
  minutes apart, Kimi K3.1 Pro's batch-of-64 figure moved from 101 to 111
  tokens a second and its tokens per verify step on the edit prompt
  (`alpha`) from 9.1 to 10.6.
- **The prompts changed under one version.** Pull requests 2187, 2188 and
  2197 changed the long prompts and 2200 changed how the batch speculation
  figure is computed; every row still says `rais_version: "1.0"`. The
  `07a6e6bcd` receipt holds rows of both prompt versions (5,800 and 5,793
  prompt tokens).
- **Long-context decode moves with what the model writes.** On one build
  (`d99d5cc81`) Kimi K3.1 Pro's decode step at 22,700 tokens of context took
  67 ms with one prompt wording and 95 ms with the other; on `dff7b5a96` the
  second wording takes 67 ms. Replies differ between these runs (F2), so
  `b1_long_tok_s` cannot be compared across builds as it stands.
- **The build is a typed label.** `--engine` is free text, and the request
  records in the API log carry the model contract's hash, not the build's
  (this predates the work audited here).
- **Gates compare a run with itself.** Nothing compares a build's replies
  with the previous build's or with a reference.
- **Gains were measured where they do not apply.** Kernel pull requests
  report one-GPU microbenchmarks. One moved the fleet the other way (F7),
  and another did not show on the fleet what it measured (F4).

What would close it: three or more samples and a median for every figure; a
version bump whenever a prompt or a formula changes, and no comparison
across versions; the build identity and lane settings read from the server;
a fixed continuation (or a token-forced one) for long-context decode; fleet
figures, with ranks in step, before a gain is recorded.

### F10. Smaller points

- **A title that understates its reach.** Pull request 2199 is titled for
  single-request decode. Its fused collective kernels also run in multi-row
  programs: the transport passes one logical row with the rows folded into
  the element count, and captured programs lose 72 kernels at every row
  count.
- **A receipt is not a qualification.** `qualification/releases/` holds
  benchmark rows only. The exactness captures the rollout script produces
  belong beside them.

## 4. Suggested order of work

| Step | Closes | Check that it worked |
|---|---|---|
| Receipts hold all attempts of one build; scorer refuses mixed or relabelled rows | F1 | Scoring the `07a6e6bcd` receipt fails; a new receipt shows any failed attempt |
| Run `old`, `qualify`, `gx` on the next rollout | F2 | The release root holds per-model captures and comparisons |
| Ten cached-prefix runs a lane on GLM 5.3, A.X K2, Kimi K3 | F3 | Ten of ten equal, or the differing step is named |
| Switch for the reference collective launcher; one Kimi K3 run each way | F4 | Rank 0's waiting share is back near 35%, or the launcher is cleared |
| Rank spread in the benchmark's output | F4 | Each row carries the waiting share and the slowest rank over the median |
| Bound the graph cache; no abort on a failed capture | F5 | A lane serving every row count from 1 to its limit keeps a stated memory floor |
| Merge 2228; remove the nine kernel frames; fix the four checks; merge on green | F6 | A green run on `main`, and one for every commit after it |
| Bind launch arguments once in the NVIDIA-free flavour | F8 | A launch with `i++` in its arguments increments once in both flavours |
| Samples, version bumps, server-reported build | F9 | Two runs of one build agree within a stated band |

## 5. Numbers for the window

| | |
|---|---|
| Pull requests merged | 113 (101 opened in the window) |
| Lines | 93,734 added, 17,768 removed, 872 file changes |
| Time from opening to merge | median 1 minute; 92 of 101 within 5 minutes |
| CI runs on `main` since the last green one | 118: 40 failed, 76 cancelled, 2 running |
| Releases rolled to the fleet | 11: `7aa056bf8`, `78e1d298f`, `b8df69a2e`, `4fc4a10e3`, `c8e2ce7cc`, `98386babe`, `07a6e6bcd`, `d99d5cc81`, `99fc2a2c1`, `454edc951`, `dff7b5a96` |
| Test files | 17 added, none deleted |

## 6. How to check these findings

Everything above comes from four places.

- **Benchmark rows.** `qualification/releases/<build>/` in the tree, and
  every run, kept or not, in `hillclimb/rais-<build>-*.jsonl` on the hub.
  Each row has `engine`, `utc`, `gates` and `metrics`.
- **The API log of each lane** (`<lane>-kv-api.log` on the hub).
  `api_start pid=` begins a start of the lane. `batch_step kind= rows=
  steps= service_ms= context=` gives the service time of every submission,
  by kind (`prefill`, `chain`, `verify`) and row count.
  `request_measurements` records keep `prompt_sha256` and the token ids of
  the reply, so one prompt can be compared across starts.
- **The rank logs** (`logs/residentd-<utc>.log` under each lane's root on
  every node, one file a lane start). `STAGE-STEP-TIMING` gives, about every
  64 steps, `submit_us`, `peer_wait_us`, `copy_us`, `combine_us` and
  `source_wait_us` per step. A rank's own compute time is the first minus
  the other four; compare it across ranks line by line. `STAGE-MEMORY` gives
  available memory and free GPU memory at each captured program.
- **CI.** The failing job's log, and the `cuda13-sm121a-receipts` artifact
  of the compile job (`logs/stage_runner_binaries.txt`, `logs/adapters.txt`).

## 7. What this audit did not cover

- The speed gains the pull requests report. This audit did not try to
  confirm them; F4, F7 and F9 say why one-GPU figures and single benchmark
  samples do not settle them.
- The model-family stage kernels changed in the window (about 11,600
  lines), the Metal changes, the site, and most of `tools/`, including the
  scoring arithmetic of `tools/rais_score.py`.
- The drafter modules and model additions of 25 November (pull requests
  2215 to 2223 and 2229), the collective changes after `dff7b5a96` (2224,
  2225) and the opt-in wire compression for prefill collectives (2231, off
  by default).
- `ring/transport/draft_bridge.c`, `node/weightd_mesh.c`, and most of the
  batch engine's planner changes.
- Whether any rewritten kernel computes the right values. That needs a GPU
  and a reference; nothing was run for this audit.
- The claim in 2199 that its exchange cannot overwrite a slot a peer has not
  read.
- Rank 8 of the fleet in F4.
