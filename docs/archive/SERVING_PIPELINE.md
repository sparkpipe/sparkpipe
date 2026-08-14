# Serving pipeline audit

api -> queue -> JIT cache -> micro-batch -> speculation -> emit, read
end-to-end in the code (node/backend.c pump, spark_serving_engine,
spark_request_api, scheduler/, spark_prefix_cache, dspark), judged against
the requirement: every slot of B* fully used, a completed lane's slot
refilled by the next queued request at the next step, and only the true
long tail paying for its own length. Verdicts first, mechanisms after.

## The requirement, as the code meets it

**Slot refill is per-step re-formation, and it works.** The dispatch is not
a persistent cohort: every pump, the request API rebuilds the decode batch
from the ACTIVE set. A lane that hits a stop token is marked done inside
SparkServingEngineCompleteDecodeDispatch (AUTO_RELEASE_COMPLETED_REQUESTS),
its KV release is queued, and the next dispatch simply does not contain it.
A queued request enters through the prefill reserve (one prefill wave rides
alongside the decode cohorts by construction:
QUEUE_DEPTH_PER_SPARK = cohort capacity + 1 prefill reserve, statically
asserted), then joins the next decode batch. Refill latency = the
newcomer's own prefill plus one pump - the floor.

**Multiple cohorts stay in flight.** pending_decodes capacity equals the
stage count: while one cohort's final events are in transit, the next
dispatches. The ring does not drain between steps; service_can_dispatch
gates on a free pending slot, not on quiescence.

**Batch shapes bucket with padding for graph capture.** Decode packs to
bucketed widths (adaptive_decode_pack, graph_sequence_padding_count).
Padding rows burn lane-compute only - weights amortise regardless - which
is the right trade on a bus-bound machine and the precondition for S5's
CUDA-graph capture.

**Lockstep completion is batch-granular and that is correct.** A cohort's
CompleteDecodeDispatch fires at done_count == request_count. The slowest
lane's final event bounds the step for all lanes - inherent to lockstep
batching, since the ring computed every lane together anyway. Not a defect;
listed so nobody "fixes" it.

**Speculation rides the same round-trip.** DSpark drafts arrive ON the
decode final event (tap capture), are stashed per lane, and the draft
function serves the next verify dispatch from the stash - zero extra
round-trips. Verify batches are a first-class dispatch kind
(SPECULATIVE_VERIFY_BATCH) with row-capacity chunking. Emit: final events
carry the accepted token IDs (multi-token per step under verify), through
the serving event ring to the client session.

## Findings - correct code that leaves speed on the table

**P1. Cross-sequence prefix reuse is DISABLED.** The ring backend
initialises the scheduler with
`DEFAULT_FLAGS & ~CROSS_SEQUENCE_PREFIX_REUSE`: the radix cache exists,
NVMe JIT modes are contract-checked at resident hello, and then the one
property that pays for an API serving centaur - many requests sharing a
system prompt, prefill skipped for all but the first - is switched off.
Presumably a distributed-KV-directory correctness caution. DO NOT flip
blind: the verification is a hardware-week test (two requests, shared
prefix, assert block-table aliasing and byte-identical logits vs cold
prefill). If it holds, this is the single largest serving-throughput win
in this file.

**P2. Sequence release awaits the resident synchronously.**
SubmitReleaseToResident submits with EXPECT_RESULT and then blocks the pump
in ResidentAwaitSubmitResult until the round-trip returns. On a healthy
resident that is sub-millisecond; it is still the only synchronous stall
in the serving loop, once per release batch, on the thread that forms the
next dispatch. The await exists for reuse safety - a freed block must not
be re-issued before the resident processed the free. But the resident's
work queue is FIFO and the KV directory is resident-internal: a release
submitted BEFORE any dispatch that could reuse its blocks is processed
first by construction, making the await redundant IF the FIFO holds
end-to-end. Hardware-week: verify the ordering claim on the wire, then
make release fire-and-forget under the same credit accounting as decode.

**P3. The K3 engine duplicates this stack, and must dissolve into it.**
inference/llms/kimi_k3/engine.h was built bottom-up as a step planner with
a verify lane - and the serving stack read here already speaks everything
it speaks: continuous batching, chunked prefill, verify dispatch kinds,
draft plumbing, slot reuse. The A4 adapter's job is therefore NOT to mount
K3Engine behind the backend; it is to make the K3 driver consume
SparkServingDecodeDispatch directly, exactly as the glm driver does, and
let K3Engine shrink to what only it does: the model-side run contract
(rows-per-run, fold-begin-one-past-head, aux capture) that the host gates
exercise. One queue, one scheduler, one request API, N drivers. Recorded
in docs/DRY_LEDGER.md as the A4 acceptance criterion.

## Minor notes

Final-event lookup scans pending cohorts linearly - bounded by stage count
times lane count, noise. Early final events (arriving before their
dispatch registers) are stashed and replayed - the race is handled, with an
eviction counter when the stash overflows. The non-resident builder path
refuses multi-chunk decode loudly rather than half-supporting it.

## Standing rule

A change to admission, completion, or release lands with its line here:
what the slot-refill latency is after the change, and why the long tail is
still the only thing that waits.

## Fast by default (2026-07-28)

The findings above stopped being findings: cross-sequence prefix reuse is
ON (the backend takes the scheduler's own default flags), sequence release
is FIRE-AND-FORGET (EXPECT_RESULT stays set, so credits and rejections
flow through the async pump the loop already runs), and DSpark speculation
is ON at the gateway. A disabled speed booster is a fallback wearing a
configuration's clothes; the doctrine now enforced by
tests/test_fast_defaults.py is: fast is the default, slow is a NAMED
kill-switch for a sparkdev bisecting a suspicion, and every mode announces
itself in the ring_effective_config banner at startup.

| kill-switch | restores |
|---|---|
| SPARKPIPE_DISABLE_PREFIX_REUSE | per-sequence-only prefix cache |
| SPARKPIPE_RELEASE_SYNC_AWAIT | blocking release round-trip |
| SPARKPIPE_DISABLE_DSPARK / --no-dspark | plain decode, no speculation |

MTP remains opt-in (--mtp): two simultaneous speculators is a mode choice,
not a speed default. The P1 hardware validation (shared-prefix aliasing,
byte-identical logits) is still owed on first ring bring-up - the default
changed because the code path is complete and the library already believed
in it; the test confirms, it does not enable.
