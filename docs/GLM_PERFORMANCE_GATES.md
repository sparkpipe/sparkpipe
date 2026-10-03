# GLM 5.3 Flash acceptance gates

These are the checks a GLM 5.3 Flash build passes before anyone reports its
output quality or throughput. Measured performance and the roofline live in
[GLM5_NEXT_ROOFLINE.md](GLM5_NEXT_ROOFLINE.md). The contract is
[sparkpipe_invariants.md](../sparkpipe_invariants.md).

Report the functional, quality and performance verdicts separately (I39). A
correct driver can still miss its performance target. When it does, use the
measured gap to choose the next optimization.

## Quality: COMPSEC-17

Run `tools/glm5_next_compsec17.py --thinking off` against the served API.
The protocol is:

- the 17 COMPSEC cases (`compsec-076` to `compsec-092`) from
  `qualification/ds4_eval/quality-fixtures-glm5.3-flash.json`;
- each question decoded to text and wrapped in the GLM chat template,
  `[gMASK]<sop><|user|>\n{question}<|assistant|>\n<think></think>\n`;
- temperature 0, at most 512 tokens, one request at a time;
- grading by `qualification/ds4_eval/compare_runs.py`. It takes the last
  `Answer:` line after `</think>`. A case passes if its line set is a
  non-empty subset of the expected lines.

The baseline is 14/17, measured on engines built from `dd3526b` (PR #1243).
The receipt is
`qualification/ds4_eval/runs/glm5-next-tp16-20260928-dd3526b-thinkoff/`.
A build that scores below 14/17 under this protocol fails the gate.

The same run carries two more checks.

- **Graph against eager.** The 17 sequential completions must be
  byte-identical with `SPARK_GLM5_NEXT_GRAPH_PATH=1` and pinned experts and
  with `SPARK_GLM5_NEXT_GRAPH_PATH=0`. The 2026-09-28 receipt records this for
  graph with hardware waits against eager with spin waits.
- **Batch invariance, which is not met yet.** With `--concurrency 17`, the
  same build scored 13/17 to 14/17 and reproduced 15 or 16 of the 17
  sequential completions. The differing cases changed from run to run (same
  receipt). Report concurrent scores as a separate result and never in place
  of the sequential score. README promises that the batch a request lands in
  does not change its tokens; TECHDEBT ("Dynamic batching") tracks the batched
  numerics that break that promise.

## Test the acceptance test

Before trusting a gate, inject bad values and prove that the gate rejects
them.

- Before PR #1256, the COMPSEC protocol sent raw token ids without the chat
  template and graded the first line of a 256-token completion. It scored a
  coherent model 0/17 (see the receipt above), so it could not tell a good
  build from a broken one.
- PR #863 found three numerical report results cast to void, so they could
  print FAIL without failing the binary. A projection probe also skipped its
  comparison when readback failed. The error norm lost small errors to
  cancellation, and nonfinite references could pass through zero-norm
  branches.

Numerical comparisons use the shared `include/sparkpipe/spark_numerical_metrics.h`.
It accumulates squared errors directly and rejects nonfinite inputs; its tests
are in `tests/test_numerical_metrics.c`. A component PASS does not stand in
for full-model, distributed or cache-restore results (I40, I46).

## Token parity across execution paths

Receipts at merged main `93c8f0d` compared resident and lazy loading. That
comparison can no longer run. `c67be23` deleted direct pack loading, so a
module started without `SPARK_WEIGHTD_SOCKET` fails with `UNSUPPORTED`.

Instead, greedy tokens must match across the paths the module can take for
the same prompts:

| Paths compared | Settings |
| --- | --- |
| Graph replay against eager | `SPARK_GLM5_NEXT_GRAPH_PATH=1` against `0`, experts pinned |
| Linear chain against chain state machine | `SPARK_GLM5_NEXT_GRAPH_PATH=0` with `SPARK_GLM5_NEXT_PIN_EXPERTS=1` against `0` |
| Resident decode chain against single-step frames | one stream, against a build without decode chains ([roofline](GLM5_NEXT_ROOFLINE.md), "Resident decode chains") |
| Prefix-cache hit against cold run | see the next section |

`SPARK_GLM5_NEXT_GRAPH_PATH` has no default. The module fails with
`INVALID_ARGUMENT` unless it is `0` or `1`.

For a single rank with collectives disabled,
[GLM_REAL_DRIVER_PROBE.md](GLM_REAL_DRIVER_PROBE.md) runs the published
driver lazily at B1, B3 and B5 and checks prefix restore exactly. That is
component evidence only (I38).

## Prefix cache

GLM advertises `SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CACHE_PUBLISH`
(`spark_glm5_next_serving_adapter.c`). A checkpoint holds KV, index KV, the
KDA recurrent state, the Q/K/V convolution windows and the sequence position
(`SparkGlm5NextRecurrentCopy` and the state capture in the module). The engine
publishes a checkpoint when a chain ends at a 64-token block boundary or
finishes its request (`SparkModelBatchChainCheckpoint`,
`runtime/model_batch_engine.c`).

Invariant I27 applies: a run that hits the prefix cache must emit the same
tokens as the cold run. In #1230, cold runs and cache-hit runs disagreed. #1230
removed three choices that depended on the execution path; see "One answer on
every path (#1230)" in the roofline. No GPU test yet compares a graph replay's
logits with an eager wave's.

Never label an uncached run a cache hit, and never silently recompute a
missing benchmark entry (I41).

## Throughput receipts

Before reporting a full-system throughput result:

1. Build every participating binary from one clean merged commit (I33). Record
   the engine commit and whether it matches main. Also record the package,
   driver and pack hashes, the topology, GPU and driver versions, context,
   occupancy and precision. Record the serving environment too:
   `G5_GRAPH_PATH`, `G5_PIN_EXPERTS`, `SPARK_TP_WAIT_MODE` and the prefill row
   limit.
2. Pass the quality gate and token parity on the topology being measured.
   Qualify and measure each topology on its own (I38). Do not derive one
   topology's speedup from another topology's numbers.
3. Run memory checking outside the timing cell. Turn off diagnostic probes
   for timing, and report cold loading separately.
4. Measure B1 and other occupancies such as 3, 7 and 15, then continuous
   arrivals and completions. Report aggregate output tok/s separately from
   per-sequence latency and TTFT. Repeat comparable unprofiled runs and record
   the spread. Do not infer scaling to large batches without memory and
   compute measurements.

Wrap the real `sparkpipe_model_batch` client in
`tools/glm5_next_bench_wrap.py --timeout-seconds 600 -- COMMAND ...`. The
wrapper reads the client's newline-delimited JSON events and checks each
request's token order and terminal completion. It drains stderr to a
temporary file and kills its own process group at the deadline. Failed,
cancelled, incomplete or malformed token streams get no throughput field, and
the wrapper exits nonzero for invalid receipts.

For batched decode, use `all_sequences_decode_window`. Its clock starts
after every sequence has emitted its first token and stops at the earliest
sequence's last token. This excludes the remaining prefill and the
shrinking-batch tail. If those bounds do not overlap, no decode-window rate is
reported. Per-sequence token IDs and arrival times are kept for parity checks
and for independent timing analysis. The first-token boundary includes
scheduling, prefill and first-token work, so it does not measure prefill
compute alone.

Without help, that window still mixes decode steps with the prefill waves of
later sequences. To measure decode alone, run the DEBUG client
`build/debug/sparkpipe_model_batch` (the GLM Full lane build installs it as
`sparkpipe_model_batch_debug`) with `SPARK_MODEL_BATCH_DECODE_AFTER_PREFILL=1`.
The barrier changes execution shape, so it exists only in DEBUG builds (I22): a
release client refuses the variable and a release engine refuses the flag. The
engine logs when the barrier is armed and when it opens. It holds every decode
wave until all submitted prompts have finished prefill, then schedules
normally.
Decode runs on the KV cache that the real prefill wrote; nothing is copied or
replayed. Wave composition differs from an unbarriered run, so compare tokens
against a barriered reference. The window from the last
first token to the earliest last token of a sequence that ran its whole budget
holds decode waves only. Sequences that stop early on EOS leave the batch, so
report the median rows per step with the rate. The engine refuses (`CAPACITY_EXCEEDED`) a request beyond the resident
sequence capacity while the barrier is closed: an unbound prompt could never
prefill, so the barrier would never open.

`decode_tokens_per_second` counts all output tokens after the first global
token and divides by the interval from first to last arrival. Tokens stay in
arrival order; request-local indices never reorder different requests. TTFT
is measured from client command launch, not from each request's own admission
time. Single-sequence inter-token statistics are omitted for concurrent
streams. These are client-observed rates, so stdout buffering and client
scheduling are part of the measurement. Use device and transport profiling
separately to explain the critical path. A single token has no measurable
decode rate.
