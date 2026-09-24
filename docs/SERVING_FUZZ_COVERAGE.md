# Serving fuzz coverage and open qualification

This is a coverage map, not a claim that all possible schedules or model
implementations are qualified. A stable system needs several independent
oracles at real module boundaries. More random rounds cannot repair an oracle
that accepts permanent BUSY, ignores an error or never injects its named fault.

Run the host campaign from a fresh checkout:

```
bash tools/test_serving_reliability_host.sh --seeds 1,7,73 --rounds 128 --loopback-rounds 24
```

An exported source archive additionally requires `--source-commit FULL_SHA`.
The runner refuses an existing object tree, builds with explicit host CUDA
stubs, records the source hash, exact commands, seeds, logs and verdicts in
`build/reliability/results.json`, and continues independent checks after a
failure. Failed builds produce SETUP_FAIL for missing binaries; missing targets
are never skipped as success. Each process group has a deadline and is killed
on timeout, including its owned child processes. PASS means that executable's
assertions passed, not that every behavior of its linked modules was reviewed.
Unselected registered C targets and Python files remain enumerated as unrun.

B2+ selects the common tree with FP32 accumulation; its host qualification gate
passes. Real GPU overlap, cancellation and topology performance remain open.
The aggregate campaign stays red if any required gate fails.

## Shared serving boundaries

| Boundary and contract | Required scenarios and oracle | Executable evidence | Remaining boundary |
| --- | --- | --- | --- |
| HTTP queue, token output, disconnect | Head/middle/tail completion permutations preserve every request, token and queue link; fragmented requests; malformed body; early disconnect; concurrent requests; complete response after recovery | `test_model_api_queue_lifetime.py`, `test_system_loopback`, `test_model_api_text` | Host fixture tokens; no real model numerical claim |
| Continuous batching and cache admission | Odd/concurrent arrivals, finite capacity, cache hits/misses, cancellation, partial output, error exactly once, no replay after emitted tokens, digest reset after reconnect | `test_model_batch_engine_mock`, `test_continuous_batch`, `test_serving_cache_admission`, `test_steploop_admission` | High-occupancy fairness, EOS from real deployment and GPU reuse need hardware workload evidence |
| Pipeline transaction protocol | Required rank/phase fault matrix before seeded schedules; actual dropped results/decisions/completions; delayed/reordered/duplicate events; rejected or BUSY work; concurrent pressure; matching identity/status and exactly-once terminal callbacks; bounded recovery | `test_serving_fault_fuzz`, pipeline mock and socket integration tests | Resident client boundary is mocked in the state fuzzer; integration tests cover real sockets separately |
| Resident sessions and IPC | Partial frames/writes, stale generation, disconnect, abort/drain/reset before new admission, reset BUSY/error, deadline and slot ownership, process replacement | Resident session/deadline/IPC/reconnect tests; loopback rank/API replacement | GPU drain and in-flight DMA lifetime require real device injection |
| Collective lifecycle | Mandatory fault cases plus seeds across 2/4/8/16 ranks; SUM/MAX/GATHER, multiple rows, output canaries; exact accepted/rejected callback identity; missing peers, stale payload, daemon loss, combine/transport errors, capacity and recreation | `test_tp_allreduce_fuzz`, collective mock, TP config tests | Host memory/shipper emulation; GPU progress and RDMA ordering remain unqualified |
| Collective algorithm | Full logical batch survives row splitting; B1 direct and B2+ tree policy, bounded large payloads, numerical results and demonstrated overlap | `test_tp_allreduce_fuzz --qualify-tree` | Host FP32 tree selection/numerics/transfer bounds pass; GPU overlap, RDMA ordering and measured topology performance remain open |
| Cache lane transactions | Prepare/commit/claim/finish/abort/release/reset; stale identity, occupied lane, duplicate owner, aged deadlines, executing reset; every lane, sequence, pin/refcount and resident mapping checked after each operation | `test_kv_lane_fuzz`, `test_kv_cache` | Real KV/index/recurrent payload equivalence is separate from ownership correctness |
| Cache storage and transport | Layout, writeback/restore, corruption, eviction, copy-on-write, page movement, failed transfers and protocol rejection | KV store/layout, NVMe, JIT slice/wire/C3C4/C5W2 and topology-switch tests | Their existing assertions run as regression evidence; full semantic audit and model-state oracle remain open |
| Expert residency and leases | Two clients, duplicate keys, pin union under budget, live bytes preserved, stale/foreign releases and exports, impossible sets, corrupt span rollback; no outstanding ownership after release | Seeded `test_weightd_working_set`, lease/expert/churn/stress tests | Host CUDA memory; real GPU mappings and daemon replacement need device tests |
| Concurrent mesh namespaces | Two daemon IPC instances, reversed 2/3/4-job startup across 24 seeded lane permutations; exact occupied-lane rejection; duplicate/capacity limits; disconnect/reuse cannot release neighboring lanes; undrained orphan fences new reservations | `test_weightd_mesh_mock`, GLM stage-context environment fixture | Host CUDA/verbs; real concurrent resident and working-set budget runs require a separate fleet receipt |
| Daemon worker/control | Cold operation blocked while HELLO/mesh progresses; repeated connection churn; interrupted RPC poisons connection; configured allocation/read/record failure rolls back | fd frames, worker, attach, map, mesh tests | Eager map fixture passes with 65 real 64-MiB chunks across descriptor batches; real CUDA mapping and daemon replacement remain unqualified |
| Warm-up and supervision | Required finite configuration; malformed/missing manifests; atomic working-set publication; checked acquire/release; readiness/ownership failures; dependent startup gating | Supervision/supervised/manifest/lazy-pair tests | Fixture syscall/process boundaries; real replacement and resource-ledger reconciliation still required |
| Memory and core ownership | Embedded descriptor lifetime, matching allocator, transaction/completion/release ordering, arenas and runtime ABI | Memory/arena/work-transaction/completion/release/runtime tests | Host backends and declared fixture boundaries |
| Module/deployment/serialization | Module ABI/load/compile, deployment metadata, tokenizer/JSON validation, numerical error metrics and codec contracts | Model-description/module-library/compiler/stage-common/LLM/tokenizer/JSON tests; deployment generation/drift/queue tests | Compilation/metadata checks do not qualify inference or fleet behavior |
| Module teardown ownership | Quiesce/unregister/lazy-release failures retain internal state and ledger; zero-slot partial initialization cleans up; retries release ownership once; GLM52 retired leases and K3 acquired/begun/recorded leases remain recoverable | `test_stage_module_teardown.py` executes actual Qwen4 Flash, Qwen38 Max, Gemma4 and Muse callbacks, common lifecycle, GLM52 recovery, and extracted actual K3 acquire/release bodies | CUDA/weightd boundaries are injected; K3 full destructor is not executed here. Public void module/driver teardown still cannot propagate retained cleanup status to its caller |
| GLM graph and lazy integration | Sticky collective failure, invalid token blocked, daemon loss fencing, stage context, embedding collective, lazy dispatch, explicit geometry/configuration | GLM graph-failure/stage-context/embedding/config/driver-probe/geometry/shard-math/lazy tests | Production bodies with external boundaries mocked; GPU graph replay, cancel, recurrent restore and numerics remain open |

## Every repository module

The campaign records the discovered `modules/` directory names. Each module
below has a distinct qualification boundary; passing common serving tests does
not silently qualify a different model. Modules without executable model
coverage remain explicit gaps for the subsequent review and hardware campaign.

| Module | Host campaign coverage | Model execution qualification |
| --- | --- | --- |
| `glm5_next_resident_decode_stage` | Common fuzzers plus GLM integration harnesses listed above | OPEN: current GLM5.3 focus; real GPU/fleet gates required |
| `dsv4_resident_decode_stage` | DSV4, TP16 and TP4xPP4 adapters; cache, lane, pool, shape, stage runner and loader fixtures | OPEN: actual model math/state and each topology |
| `qwen38_27b_resident_decode_stage` | Serving adapter, host work-control fixture and common contracts | OPEN: actual model math/state |
| `gemma4_resident_decode_stage` | Serving adapter; dense/MoE geometry and negative controls; actual teardown callback | OPEN: actual model math/state, dense/MoE variants |
| `muse_glimmer_resident_decode_stage` | Serving adapter fixture, actual teardown callback and common contracts | OPEN: actual model math/state |
| `ling_resident_decode_stage` | Serving adapter fixture and common contracts | OPEN: actual model math/state |
| `hy4_resident_decode_stage` | Lifecycle fixture and common contracts | OPEN: actual model math/state |
| `k3_resident_decode_stage` | Attach/KV/defines contracts and host kernel-run equivalence | OPEN: adapter lifecycle fuzz and real model math/state |
| `glm52_dspark_draft_backend` | Draft dispatch/MTP policy tests | OPEN: model execution and speculative accept/reject numerical oracle |
| `glm52_resident_decode_stage` | Common/module contract tests; retired/current expert-lease recovery | OPEN: adapter lifecycle fuzz and real model math/state |
| `kv_mooncake` | Host client fixture | OPEN: external service failure/recovery and actual payload residency |
| `laguna_resident_decode_stage` | Model-header contract test | OPEN: executable adapter fuzz and real model math/state |
| `qwen4_flash_resident_decode_stage` | Model-header contract test and actual teardown callback | OPEN: executable adapter fuzz and real model math/state |
| `qwen38_max_resident_decode_stage` | Host work-control fixture, actual teardown callback and common contracts | OPEN: executable adapter fuzz and real model math/state |
| `dsv41_flash_resident_decode_stage` | Common contract discovery only | OPEN: executable adapter fuzz and real model math/state |

## Reproducing and extending failures

Fuzzer diagnostics include the initial seed and operation/round. Re-run the
exact command in the receipt; minimize to the named deterministic case before
changing production code. The cache fuzzer also accepts an optional scenario
index. Required scenarios run before random exploration, so seed selection
cannot accidentally omit a failure class. Preserve negative controls proving
that the old implementation or a broken oracle fails the new check.

New behavior needs an invariant and an observable outcome at its owning module,
not a success count added after a symptom. Check synchronous rejection versus
accepted completion separately. Check all resources and unaffected owners after
failed transitions. Time passing never proves device completion or permission
to steal a pinned lane. Recovery must terminate in successful new work or an
explicit fenced failure; permanent BUSY is a failed liveness check.

## Review following stabilization

Review from request admission through terminal device completion and resource
release, using the failure receipts as starting evidence. First inspect duplicated
ownership/reset/cache policy, callback reentrancy and post-callback access,
ignored errors, allocation arithmetic, hidden mode/configuration choices and
blocking work on progress threads. Compare all model adapters with the existing
common serving/lifecycle primitives. Consolidate only demonstrably equivalent
policy, keeping model math direct. The removed deadline takeover and duplicate
force-cleanup path are examples of reducing states instead of stacking repairs.

The review must separately account for unrun driver paths, source-only tests,
legacy tests for removed implementations, eager-map fixture geometry and real
hardware gates. Do not replace those gaps with a blanket “all modules covered.”

## Historical Linux campaign receipt, 2026-09-22

A fresh exported checkout of `6a4e905d60b4a228ca25f948bac23d2f9245c59c`
ran on Spark0/Linux aarch64 with explicit host CUDA stubs, seeds 1,7,73,
128 random rounds and 24 loopback rounds per seed. Result: 114 PASS, 8 FAIL,
4 SETUP_FAIL, 0 TIMEOUT. These are execution counts; the matrix above defines
the narrower semantic and hardware boundaries. No GPU inference or service
restart was performed.

All four seeded lifecycle fuzzers, the API queue fuzzer, real-socket loopback,
tokenizer/text API tests, and continuous-batch tests passed. The collective
fuzzer exercised ranks 2,4,8,16 for every seed. The expanded pipeline oracle
also rejects final callbacks while any matching rank still owns work.

Historical FAIL at `6a4e905d`: aggregate build; RDMA hello-control fixture; weightd subprocess
startup; eager-map chunk geometry; DSV4 TP4xPP4 adapter ABI loading; deployment
generator and checked-in deployment schema drift; required B2+ tree selection.
Historical SETUP_FAIL at `6a4e905d`: mesh mock CUDA/verbs declarations, LLM contract math
linkage, Gemma4 model defines, Ling attention-period define. These failures
were acceptance gates at that source revision; subsequent results below supersede
their current status without rewriting this receipt.

The campaign found a tokenizer regression inherited from PR1077: replacing
the allocated vocabulary-entry count with maximum token ID plus one caused
a heap overrun in reverse-table construction and destruction. Keep entry
count unchanged; validate deployment vocabulary span using maximum token ID.
Existing JSON/compiled roundtrip fixtures cover sparse IDs and added tokens.
The original Linux ASan trace identifies `SparkTokenizerBuildReverseVocabulary`;
the repaired tokenizer, sidecar ground truth and actual text API tests pass.

## Later Linux receipt and host inventory

The fresh `ec340d3a167a570c6fed6d4becb842603ae171a7` campaign recorded
128 PASS, 1 FAIL, 0 SETUP_FAIL and 0 TIMEOUT. The failure was
`test_jit_kv_c3c4`: its asynchronous worker outlived stack fixture storage.
The ASan diagnosis led to `5fa65244`, which drains both fixture workers before
scope exit and makes their shared gate atomic. Targeted checks pass after that
repair; the full updated campaign requires a new receipt.

Review of the 34 C targets absent from that receipt selected 33 additional host
gates: resident end-to-end and orchestration; draft/speculation ownership;
DSV4 cache, lane, pool, shape, runner and loader behavior; Gemma dense/MoE
geometry and negative controls; K3 host emulation; Qwen work control; and common
cache/model/kernel metadata. Their transitive build recipes use host compilers,
fixture modules or explicit CUDA stubs with `CUDA_HOME=/nonexistent`. This is
registration, not an execution verdict or a complete semantic audit.
`test_qwen38_math_kernels` is separate from that host campaign because it uses
NVCC and the real CUDA runtime. On 2026-09-22, exact source
`d8cc2a906c1d3325899ebe601b525e65bbfaca91` compiled with CUDA 13.0.88 for
`sm_121a` and passed on Spark0: exit 0 in 0.314 seconds under a 30-second
process limit. The three cases check router softmax, shared gating and the
weighted reduction with all ten model experts, nonidentity mapping and distinct
weights. The fixture checks CUDA allocation/copy/free failures; the owned PID
exited and disappeared from the CUDA process list. The executable SHA256 is
`54cf5a9c4fbdfd7cbf2c0031c3f8d8a3c9564cb1fa36d0b526d9432c636b4300`.
This is primitive numerical evidence, not Qwen or GLM model parity, serving
throughput, distributed inference or RDMA qualification. Unselected Python
files remain listed separately in each receipt; host reference results retain
their narrower scope.

The W1 loader gate now verifies real weightd attachment, shared mapped-region
bytes and pointer offsets, short-pack bounds, and missing-attach rejection.
It also rejects the obsolete direct-copy pipeline with a configured daemon:
setting an environment variable is not proof that an allocation belongs to
weightd. This gate makes no pipeline-overlap claim. The unused direct-copy
worker and loader benchmark remain a concrete removal item for the subsequent
code review; their public region operation now returns `UNSUPPORTED`.


## Historical host and GPU checkpoint

The immutable Linux campaign at
`52d2944262eb643d901df08f7662e5272fa7351b` recorded **163 PASS, 0 FAIL,
0 SETUP_FAIL and 0 TIMEOUT**, using seeds `1,7,73`, 128 fuzz rounds,
24 loopback rounds, two build jobs and a 180-second per-test limit. It selected
114 of the 115 registered C targets; `test_qwen38_math_kernels` retains its
separate GPU receipt above. The receipt also lists 164 unselected Python files.
Its non-document source digest is
`e01d7ad867eb0067415b992148e9d3a564087e9222b00233e77f7c3a0846268d`.
These are executed host fixtures and fault schedules, not a complete semantic
review or proof of every model's GPU behavior.

At the same source, the real CUDA `tp_mesh_hardware_probe` passed 70 required
cases on Spark0 with both CUDA function and data loading explicitly `LAZY`:
66 numerical schedules over TP2/3/4/8/16, direct/tree and split payloads,
BF16 sum, unsigned 64-bit maximum and rank-major gather, eager execution and
graph replay; three stale-readiness/cancellation/timeout schedules; and one
91-round graph timing schedule. It calls the shared production GPU kernels and
the daemon's actual readiness processor. One 128 MiB mapped region supplies
private rank bands, with CPU copies acting as transport. The future-tag
acceptance mutant failed the no-early-phase assertion. The prior first-round
failure under lazy loading disappeared after explicit kernel preload during
hardware preparation; no eager-loading fallback is required. See
[the hardware receipts and limits](TP_STREAM_MEMOP_QUALIFICATION.md).

This device evidence covers those kernels, mapped-memory gates, guarded errors,
drain and replay recovery. It does not establish distributed serving throughput,
model parity, complete GLM cache restoration or the outer driver unload contract.
The generated public void destructor still discards a module's retained-cleanup
status; that ownership boundary remains a concrete item for the deeper review.

## HC site and skinny GEMV GPU regressions

`make test-glm-hc-mix` runs the fused HC site kernel (`Glm5NextHcSiteKernel`:
mix, Sinkhorn and pre-reduce in one eight-CTA cluster launch). Mixes are
checked against an f64 host reference, bounded by 1e-5 of the L1 magnitude
of the dot product, because the split over eight CTAs reassociates the sum.
Sinkhorn, pre-reduce, collapsed and snapshot outputs must be bitwise equal to
the frozen pre-fusion kernels fed the fused kernel's own mixes. It also times
90 B1 sites against the frozen three-kernel sequence, with one reused weight
matrix and with 90 distinct matrices.

`make test-skinny-gemv` checks the decode GEMV (`inference/kernels/skinny.cuh`)
against f64 references for every GLM 5.3 Flash TP16 dense BF16 projection
shape (rows 1-4) and for the routed FP8 W1/W2 experts, including two routes
sharing one expert. It confirms the fallback cases (more than four rows or
routed tokens, misaligned weights) and prints effective GB/s for the skinny
kernel and the tensor-core GEMM on each shape.

Both are GPU targets outside the host campaign. The pre-fusion HC receipt at
`012f16a4f25a89cf8ff1745b4bcdfed56b6bebcf` is retained in
[the HC receipt](receipts/glm5-next-hc-mix-012f16a4.json) and
[the performance report](TP16_HARDWARE_PROFILE_20260922.md).

## Shared serving release host campaign

Source `040f8eec545e052b38b9212a27ffa7d62ae92d6d` completed **166 PASS,
0 FAIL, 0 SETUP_FAIL, 0 TIMEOUT** in a fresh Linux archive: seeds 1/7/73,
128 rounds, 24 loopback rounds, two build jobs and 180 seconds per test. It
selected 114/115 registered C targets; the Qwen GPU target and 164 unselected
Python files remain separate. The non-document source digest is
`d64118a9cedc2233064b7882a8de4a7b04b1996783b64a8375411551885d979f`. The retained receipt is
`/private/tmp/sparkpipe-pr1082-receipts/host-040f8eec/reliability/results.json`.

The preceding 44fe4af7 attempt exposed a GCC fixture-indentation error and
was killed at its 8 GiB cgroup limit after 54 successful checks; it is a failed,
partial campaign. The corrected fresh run used an explicit 32 GiB host budget.
The queue regression also reproduces fetched-only commits in a real shallow
Git repository and proves sync preserves the requested source.

See [parallel resident qualification](PARALLEL_RESIDENT_QUALIFICATION.md) for
actual four/eight-instance GLM fleet results and the separate mixed-topology
initialization failure. These host and device receipts retain different scopes.

The final runtime candidate `852e01a2c9ee8b7b273370a793761dc70db5b005` repeated
the same fresh Linux campaign: **166 PASS, 0 FAIL, 0 SETUP_FAIL, 0 TIMEOUT**.
It includes the forced-alignment shared-file mapping regression. Non-documentation
source digest: `c20ff6ac61fd91170fddb4c41854fc3b47a5cb8980449cefce9cc0862b770b94`.
The complete inventory, unrun list and logs are retained in the release qualification
archive; local copy: `/private/tmp/sparkpipe-pr1082-receipts/host-852e01a2/reliability/`.
