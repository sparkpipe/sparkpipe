# spark_weightd — the weight-residency daemon (operator design, 2026-08-30)

THE PROBLEM: every process start re-reads and re-uploads every pack
byte per rank (fopen + sequential per-tensor reads → device memory).
With the operator's constraint — NO budget for 2× RAM during
transitions — and the dev-cycle pain (code redeploy paying a full
reload), the fix is a residency daemon.

## The design (as specified)

OWNERSHIP: one spark_weightd per node owns the weight arenas. VMM API
(cuMemCreate + CU_MEM_HANDLE_TYPE_POSIX_FD), loads the stagepack
once, verifies content hash + geometry fingerprint once, exports
shareable handles. Consumers: cuMemImportFromShareableHandle +
cuMemMap at startup.

IDENTITY-KEYED ATTACH: (model, revision, topology, pack SHA-256,
geometry fingerprint, ABI version). Code bump w/o pack change
attaches in ms; pack change misses → daemon-side reload. The daemon
is the module library's runtime twin — content-addressing throughout.

READ-ONLY EXPORT: VMM access flags map consumers read-only (the
marketplace tenant-scribble protection for free).

## The perf notes (preserved verbatim from the analysis)

- Consumers' kernels read the same physical DRAM pages — zero copies
  after mapping; no IPC-per-access; the pointer IS the weight.
- GB10 unified memory: no PCIe boundary — not the discrete-GPU
  zero-copy trap. Essentially free.
- 2 MB VMM pages (a 25-100 GB arena must not drown the TLB in 4 KB).
- CUDA graphs re-captured after attach (process-local; imported
  addresses stable for process lifetime = the existing prewarm path).
- Cold load unchanged (~20s/100GB NVMe); WARM CODE REDEPLOY < 1s
  attach — the dev-cycle win.
- MODEL UPDATE: background load while old serves, then re-attach —
  needs TRANSIENT 2× the shard footprint. OPERATOR CONSTRAINT: NO
  2× BUDGET → the background-load variant is DEFERRED: updates go
  through stop-attach-start (the fleet is dark-briefly, per the
  registrar's cold wave — seconds). Revisit only with explicit
  budget.
- CRASH SEMANTICS: daemon death invalidates all consumers (detect +
  refuse, fail-closed — never chase stale pointers); consumer death
  drops a refcount.

## Loader fixes regardless (ride the same lane)

L1 parallelize the per-tensor sequential fread+upload (single-thread
today; the daemon's cold path pays it too).
L2 parallel hash (SHA-256 of 25-100 GB single-threaded is a double-
digit-seconds tax).
L3 zero-copy cold: if the pack is already exact runtime layout,
mmap the pack AS the arena backing (no copy-then-fill) — VERIFY the
layout claim per family before adopting.

## Not debug-only (the rationale)

Exact identity + fail-closed = production-safe by construction: an
attach-by-hash consumer loses nothing vs loading the bytes itself;
the determinism receipts stay valid (same bytes). The risk is
lifecycle (refcounts, orphaned arenas, version skew) — gated by the
existing promotion/qualification chain. THE OPERATIONAL WIN: the
per-node multi-topology layout becomes cheap — sixteen topologies'
packs daemon-managed; switching stops being a reload.

## Implementation order (deliberately incremental)

W1 loader fixes L1+L2 (pure win, no daemon needed, measure first).
W2 the daemon core: arena alloc + identity table + export; ONE
  family (dsv4 — its loader is the reference), consumer attach
  path, the crash semantics, 2 MB pages, graph re-capture.
W3 fleet integration: the registrar's GO gains weightd-healthy;
  the wave tools attach instead of load; the qualification gates
  re-run on attached-arena serving (determinism must be identical).
W4 multi-family + the multi-topology operational win.
