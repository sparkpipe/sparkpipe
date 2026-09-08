# Driver acceptance

Required behavior is mandatory. No descriptor bit, configuration option,
environment variable, stub or documentation exception can disable it. Report
functional qualification and hardware-normalized performance qualification
separately. An incomplete driver fails functional qualification; a correct but
slow driver remains an explicit performance optimization target.

## One implementation of common policy

Use one optimized algorithm with operation callbacks and an opaque context,
as qsort separates sorting from comparison. Call hooks at stage/batch
boundaries, not per tensor element. Extend the existing implementation rather
than copying lifecycle, admission or cache policy into another model family.

Reuse has overlapping boundaries, not a rigid family inheritance tree:

1. Universal policy shared by every driver, device and topology: scheduling,
   admission, cache ownership, transactions and completion/lifetime rules.
2. Topology algorithms shared by TP, PP and hybrid deployments: partitioning,
   communication dependencies, credits and progress, with device operations
   supplied through backend hooks.
3. Mathematical operations and state-layout algorithms shared across families:
   DSA, recurrent attention, routing, normalization and dense/expert execution.
   Parameterize geometry, precision and layout; use callbacks when behavior
   differs. A model family composes these operations rather than owning a copy.
4. Device implementations shared by all users of CUDA, Metal or another
   backend: kernels, allocation, copies, events and submission mechanisms.

Precision/weight codecs and transport fabrics are further independent reuse
boundaries. Place each implementation at the broadest boundary its semantics
permit. Do not move CUDA code into universal policy merely to call it common,
or force cross-family DSA behind a single-family abstraction. Preserve direct
specialization of hot math while sharing its dispatch and scheduling policy.

| Responsibility | Existing common implementation | Model/backend contribution |
| --- | --- | --- |
| Continuous batching, prefix lookup, fair admission | `runtime/model_batch_engine.c`, `runtime/model_batch_scheduler.h` | Model capacity and state geometry; no alternate scheduler |
| Row ordering and wave selection | `spark_row_layout.h` | Lane-ordinal callback/context |
| Submission/cache transactions | `runtime/model_serving_adapter.c`, `node/model_residentd.c`, `spark_admission.h` | Model state preparation and commit/abort operations |
| Page lifetime, residency, immutable prefix sharing | `cache/kv_page_cache.c`, `cache/kv_cache.c`, `cache/prefix_cache.c` | Complete model payload layout, including recurrent state |
| Page movement | `cache/kv_page_store.c` | Copy callback and opaque device context |
| Collective orchestration and progress | `ring/transport/tp_device_collective.c` | Device combine/transfer/completion operations |
| Slot ownership, completion lifetime | `runtime/stage_module_common.c`, `runtime/work_transaction.c` | Model stage work and final state publication |
| Deployment and hardware reservations | `tools/spark_queue.py` and its existing runners | Assigned-node configuration, immutable source/build receipts |

Keep model geometry, state layout and math in model hooks. Keep device
allocation, execution, copies and events in backend hooks. Shared policy
must remain portable to the eight Mac Studios arriving in October 2026;
CUDA/NCCL-specific assumptions do not belong in scheduler or cache policy.
Existing backend abstractions still need audit and Metal qualification; this
document does not claim that implementation is already complete.

## Mandatory interface and behavior

Serving ABI 21 retires the PREFILL, DECODE, RELEASE, PREFETCH, RESET,
DRIVER_OWNS_KV and JIT_KV capability bits. Their old numeric bits are rejected.
Every interface must provide initialize, destroy, validate_submission,
submit, prefetch, resolve_prefetch, progress, quiesce, snapshot and reset.
Cache geometry, positive runtime page capacities and slot reuse are required.
Clearing every remaining capability bit does not skip any of these checks.
The remaining descriptors describe transport/topology and specialized
execution modes; they cannot waive required externally observable behavior.

Compile with the current headers and use the common loader. In particular,
`make build/probe_adapter` builds the probe against the real ABI; do not copy
struct definitions or bit values into a probing utility. Interface validation
only establishes structural completeness. It cannot establish correctness of
a callback or qualify a driver by itself.

Functional evidence must cover:

- Numerical agreement with a pinned reference for prefill and decode,
  short/long contexts, ragged mixed-position batches and all supported
  topologies. Consistently wrong token streams are failures.
- Continuous arrivals and completions at arbitrary occupancies, including
  odd sizes. Prove ready rows execute together; output counts alone do not
  detect a hidden one-row loop. Exercise admission pressure without starvation.
- Prefix hit, miss, eviction, residency transfer, copy-on-write and transaction
  abort. Compare restored execution with uninterrupted execution. The state
  payload includes all model-dependent history, not just attention KV.
- Required-hit benchmarks reject a miss; report cold/prefill work separately.
- Cancellation, failure cleanup, release, slot reuse and reset without stale
  sequence identity or recycling buffers still owned by device work.
- Strict lazy expert loading, missing/corrupt manifest errors, bounded
  residency and concurrent independently reserved driver consumers.

Run host policy tests, real driver tests and distributed numerical/lifecycle
tests. Keep behavioral tests for incomplete drivers failing until the actual
implementation works; do not replace them with tests accepting rejection.
No-op callbacks that return success are not implementations. Diagnostic
behavior clamps require #ifdef DEBUG and a visible diagnostic; DEBUG does not
waive mandatory operations.

## Performance evidence

Record raw per-request latency and aggregate output rate, prefill throughput,
TTFT, precision/source, context, batch occupancy, topology/node count,
speculation, effective bytes, memory budget and immutable build provenance.
Compare separate cold and required-hit warm-cache runs. Use matched workloads
and repetitions; source the community baseline and inspect its actual launch
configuration before labeling it comparable.

Normalize hardware count and precision transparently. A bandwidth-bound FP8
TP4 rate of 25 token/s can be comparable to a 4-bit TP4 rate of 50 token/s
under a two-times-weight-traffic assumption. Show that assumption separately
from measurement; refine it using actual dense/expert/cache traffic and
quantization costs. A 25 token/s TP16 result consumes four times the nodes
and does not pass that same comparison.

Measure TP4 bandwidth efficiency first, then TP16 speedup (target 3.5x on
matched workloads), and approximately 4x TP4 capacity with a full TP4xPP4
pipeline. Queued request count is not serving capacity. Report how much
allreduce time remains exposed after compute overlap, the bytes/weight reuse
lost to microbatch splitting, GPU launch gaps and rank imbalance. Use these
measurements to select the next optimization while preserving correctness.

## Current enforcement change is not driver completion

The ABI enforcement change intentionally exposes missing integrations. GLM
Flash lacks complete cache preparation/restoration; production adapters lack
reset callbacks, and Qwen had no-op prefetch/resolve callbacks. Do not fill
these holes with dummy functions or invented geometry. Complete the common
machinery and real model state integration, then rerun the behavioral gates.
