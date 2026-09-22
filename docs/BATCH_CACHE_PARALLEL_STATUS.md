# Batch, cache and parallel inference implementation status

PR1082 targets `main`; PR1077 and PR1081 are already merged. The shared runtime
has passed four- and eight-instance GLM inference across all sixteen Sparks,
including exact output, overlapping decode and terminal cleanup. The latest
broad host campaign passed 166 checks. See [parallel qualification](PARALLEL_RESIDENT_QUALIFICATION.md)
and [multideveloper setup](MULTIDEV_QUICKSTART.md) for precise results and limits.
Different-model numerical parity, partial-pool GPU eviction and sustained
continuous-batch performance remain open.

Implemented and host-tested:

- GLM uses configured logical and physical cache capacities, dirty-page writeback,
  complete KV/index/recurrent/window state capture, and nonidentity restoration.
  Physical capacity reaches the CUDA view. Common partial-prefix indexing and
  copy-on-write preserve immutable shared prefixes across divergent branches,
  failures, eviction and rollback. No alternate per-driver prefix manager.
- Temporal prefill traverses each layer with the admitted row batch together.
  Recurrent state follows each sequence's row indices; DSA visibility is causal
  per row. Tests exercise widths 1 through 101, odd and unequal lengths, final-row
  capture and synthetic numerical KDA recurrence. This is not real-model parity.
- Completed multi-token chains publish their actual final processed context with
  a prepared, zero-row CACHE_PUBLISH operation before release. Publication captures
  paired GLM recurrent state, retains the binding and invents no intermediate
  checkpoints. Early EOS releases cleanly when the earlier state was not retained.
- Logical B1 uses direct contribution broadcast. B2+ uses shared masked binomial
  tree reduction/broadcast with FP32 SUM partials, U64 MAX and rank-major gather.
  Host tests cover TP2 through TP16 and the exact 2(N-1) payload bound. Oversized
  transfers are chunked. Real compute overlap remains unqualified.
- Mesh source ownership lasts through every posted transfer completion, with
  finite send credits, full epoch/sequence tags, stale completion rejection and
  source-copy gating. Explicit participant masks isolate TP groups; readiness
  requires all configured peers with matching ABI/group identity.
- The existing queue accounts for persistent processes, device reservations and
  ports, and retains ownership through verified cleanup. The real inference
  smoke runner isolates sockets, listeners, mesh records, cache and logs; pins
  source, executables, driver/configuration and model inputs; compares exact
  reference tokens and requires successful owned-process shutdown before PASS.
- Common collective activity owns the weightd interval through terminal stream
  completion. The daemon sleeps only after active owners, queued doorbells and
  pending NIC writes drain. Lost owners fail explicitly. CUDA graph completion
  uses bounded callback receipts instead of a host spin loop; GLM eager waits
  use the same primitive. MTP callbacks hand CUDA work to the existing worker.
- The API and CLI wake on sockets, queued work and explicit retry deadlines.
  They no longer depend on a fixed 5/10 ms progress cadence. Token events flush
  before waiting. HTTP cancellation is serialized through the engine worker.
  See [event-driven progress](EVENT_DRIVEN_PROGRESS.md).

Selected evidence:

- The hardware wait prototype passed 24 two-Spark NIC-to-GPU trials using shared
  memfd registration, plus three local GPU cases. Bypassing the wait fails the
  assertion. The [receipt](receipts/tp-hardware-wait-cd344a64.json) pins source,
  binaries, commands and results; it does not qualify model throughput.

- Queue: 48 tests; legacy receipt rejection: 3; real smoke runner: 27.
- Cache fuzz: 13 scenarios, seed73/2000 produced 659645 checks; additional
  seed1337, sanitizer runs and rollback/copy failure negative controls passed.
- Collectives: all TP2..TP16 host configurations pass the direct/tree mandatory cases (46 after initialization/registration coverage);
  TP4 sanitizer and mesh sanitizer pass. FP32-rounding and premature source
  release mutations are rejected. These simulate transport/device behavior.
- GLM temporal module and CUDA sources compile with real CUDA 13.0.88 for sm121a.
  K3 and Qwen3.6/3.8 shared-kernel consumers also compile. No GPU launch implied.
- CACHE_PUBLISH: real TCP/UNIX three-rank fixture passes prepare/commit,
  publication, retained ownership, decode and release. GLM actual source-body
  capture/restore tests and sanitizers pass; batch tests cover final-prefix reuse,
  early EOS and retry deadlines (90 checks after event-driven additions).
- Real API text/token-ID and system loopback tests pass. An idle API responds
  correctly to a queue wake; removing that wake fails the regression.
- Gemma/Ling family-local headers, DSV4 topology slicing/full algorithm parsing,
  and deployment tokenizer generation now pass their focused host gates.

Still required:

- Finish qualification of GPU cancellation/drain and event-driven mesh activity.
  GPU wait rearming and normal completion cannot cancel unrelated rank work.
- Complete lazy segmented graphs. Whole-chain graph capture now rejects missing
  expert leases explicitly; a mapped arena or eager warm-up does not prove that
  all experts are owned. The existing explicit resident pin mode now has correct
  Acquire/BeginUse/completion/release lifecycle and PP layer enumeration.
- Enforce complete device allocation budgets before shared GPU rollout. GB10
  MemoryMax does not contain all CUDA allocations. The driver ledger omits some
  direct allocations and CUDA graph/context overhead; declarations alone do not
  establish a hard device memory bound.
- Real GPU numerical parity, graph replay/failure, TP4/TP16/TP4xPP4 serving,
  simultaneous different-model inference and sustained matched performance.
- Retain the earlier failed campaigns. PR1081's historical
  receipt remains 114 PASS, 8 FAIL, 4 SETUP_FAIL. The PR1082 checkpoint 816160d2
  recorded 120 PASS, 8 FAIL, 1 SETUP_FAIL, 0 TIMEOUT. Checkpoint ec340d3a recorded
  128 PASS, 1 FAIL, 0 SETUP_FAIL, 0 TIMEOUT. Its sole failure exposed two cache
  fixture workers outliving stack storage; ASan reproduced the use-after-scope,
  and checked shutdown plus an atomic gate fixed it (74 checks, clean ASan/UBSan).
  Earlier build/fixture failures, API fuzzer setup and configuration drift passed;
  those fixes do not rewrite either complete-campaign receipt. K3/GLM committed
  deployments now match the current generators, including mesh session tables,
  GLM pack names, EOS tokens and configured cache geometry.

The old fleet-agent inference services were stopped for the shared qualification.
The authoritative queue owns the new jobs; unknown or unbounded consumers block
shared admission. Record functional,
setup, unrun hardware and performance results separately.
