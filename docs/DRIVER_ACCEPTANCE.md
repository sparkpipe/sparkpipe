# Driver acceptance

The normative contract is [SparkPipe invariants](../sparkpipe_invariants.md).

Required behavior is mandatory. No descriptor bit, configuration option,
environment variable, stub or documentation exception can disable it. Report
functional qualification and hardware-normalized performance qualification
separately. An incomplete driver fails functional qualification; a correct but
slow driver remains an explicit performance optimization target. The one
recorded deviation, pinned GLM graph serving, is listed under "Known
exception" with what it does not qualify.

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
| Development hardware reservations | `tools/spark_queue.py` and its existing runners | Assigned-node configuration, immutable source/build receipts |
| Production deployment | fleet-agent (`tools/fleet_node_agent.sh`) pulling release roots from the hub over HTTP :8802 ([`FLEET_RELEASE_RUNBOOK.md`](FLEET_RELEASE_RUNBOOK.md)) | A coherent release root per rank |

Keep model geometry, state layout and math in model hooks. Keep device
allocation, execution, copies and events in backend hooks. Shared policy
must remain portable to the eight Mac Studios on order;
CUDA/NCCL-specific assumptions do not belong in scheduler or cache policy.
Existing backend abstractions still need audit and Metal qualification; this
document does not claim that implementation is already complete.

## Mandatory interface and behavior

The serving ABI is 23 (`SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION` in
`include/sparkpipe/spark_model_serving_adapter.h`). ABI 21 retired the
PREFILL, DECODE, RELEASE, PREFETCH, RESET, DRIVER_OWNS_KV and JIT_KV
capability bits; their old numeric bits are rejected. Every interface must
provide initialize, destroy, validate_submission, submit, prefetch,
resolve_prefetch, progress, quiesce, snapshot and reset;
`SparkModelServingAdapterValidateInterface` (`runtime/model_serving_adapter.c`)
rejects a missing one. ABI 22 removed the slot-reuse policy field. Common code
always requires release before another sequence can own a bound slot,
including at position zero. ABI 23 (a2fbb5f) added per-lane sampling rules and
the SAMPLING capability; a sampled lane on an adapter without the capability
is rejected. Cache geometry and positive runtime page capacities are required.
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

## Driver identity

A serving adapter loads a driver only when five descriptor fields equal the
ones the adapter sends; otherwise initialize fails with `TARGET_MISMATCH`
(`runtime/serving_adapter_template.c`). The driver compiler takes them from
the firmware description it compiles:

| Descriptor field | Firmware description | Adapter side |
| --- | --- | --- |
| `model_id` | `model.id` | its driver model id constant |
| `model_revision` | `model.revision` | the checkpoint revision the build passes, the contract's `source_revision` |
| `stage_name` | the stage's `name` | its stage name constant |
| `target` | the stage's `target`, which must be the module's `MODULE_TARGET` | its target constant, and each node's `node_target` |
| `model_description_sha256` | the description file's SHA-256 | the hash its build computes from the same file |

The stage's operations name the module by its `MODULE_IDENTIFIER`.
`tests/test_adapter_description_identity.py` expands the adapter of every
family that loads a compiled driver, with the flags its own build passes:
dsv4 (flash), gemma4 31B and 26B, glm52 at each of its seven codecs,
glm5_next, laguna, ling, minimax, muse_glimmer, qwen38_27b, qwen38_max and
qwen4_flash. It checks what each sends against its description, and the
description's target and module against the module Makefile. Where the
build invocation supplies the revision (glm52, glm5_next, qwen38_max), the
test cannot check it. glm5_next, laguna and ling do not compare the
description hash, and dsv4 leaves the target to the loader, which compares
it with the node target.

Module Makefiles build their adapter with the one `adapter` rule in
`modules/resident_decode_stage_rules.mk`. A module names `ADAPTER_SOURCE`,
`ADAPTER_FLAGS` and, when it links the CUDA runtime, `ADAPTER_LINK`; the
library lands in `$(BUILD_DIRECTORY)/lib<family>_serving_adapter_<codec>`,
where `tools/module_build_release.sh` looks for it.

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

## Quality gate

COMPSEC-17 (`qualification/ds4_eval` fixtures) passes only when the request
applies the model's chat template, disables thinking, allows at least 512
output tokens and grades the last `Answer:` line
(`qualification/ds4_eval/compare_runs.py`). `tools/glm5_next_compsec17.py`
does this for GLM (default 512 tokens, pass threshold 14 of 17); GLM 5.3
Flash scored 14/17 on TP16 engines from dd3526b
(`qualification/ds4_eval/runs/glm5-next-tp16-20260928-dd3526b-thinkoff`). The K3
fixtures are raw text with no chat template, so a K3 gate needs the same
treatment before it can pass or fail a model. Batched serving is not
batch-invariant, so compare concurrent runs by grade, not by token stream.

## Known exception: pinned GLM graph serving

Production GLM 5.3 Flash does not use strict lazy, bounded expert residency.
fleet-agent's `20-serving.conf` drop-in sets `G5_PIN_EXPERTS=1` and
`G5_GRAPH_PATH=1`; `tools/fleet_node_agent.sh` passes them to residentd as
`SPARK_GLM5_NEXT_PIN_EXPERTS` and `SPARK_GLM5_NEXT_GRAPH_PATH`, and
`SparkGlm5NextPinAllExperts` leases every routed expert (12096 = 288 experts
x 42 routed layers, arithmetic from `model_contracts/glm53_flash_authoritative.json`).
The graph path requires this since 78c2c21. It is an explicitly selected,
reported resident mode (I28): it does not qualify the lazy path under I29 or
I30, and it remains an exception until graphs can relocate expert pointers
(TECHDEBT).

## Current per-driver gaps

Code state on 2026-09-28. Do not fill these holes with dummy functions or
invented geometry; complete the real integration and rerun the gates.

- k3: `K3ServingPrefetch`, `K3ServingResolvePrefetch`, `K3ServingProgress`,
  `K3ServingQuiesce` and `K3ServingReset` return success without doing
  anything (`spark_k3_serving_adapter.c`).
- qwen4_flash: its interface table (`ServingSeamInterface` in
  `spark_qwen4_flash_serving_adapter.c`) sets no prefetch,
  resolve_prefetch or reset, so `SparkModelServingAdapterValidateInterface`
  rejects it.
- dsv41_flash, hy4 and mimo26: no serving adapter source exists under
  `modules/<family>_resident_decode_stage/source/` (mimo26 holds only its
  stagepack format header). MiMo 2.5 has no module directory; its engine is
  `inference/llms/mimo_2_5`.
- The families built on `include/sparkpipe/family/serving/` (dsv4, gemma4,
  glm52, glm5_next, laguna, ling, minimax, muse_glimmer, qwen38_27b) and
  qwen38_max route reset through a driver RESET admission. A present
  callback is structural evidence only; the functional evidence above still
  qualifies each driver.
