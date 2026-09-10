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

## The lazy expert arena (2026-09, shipped)

ATTACH CONTRACT (CONFIGURED LAZY ARENA): when SPARK_WEIGHTD_ATTACH_LAZY
is set the pack MUST attach through KIND_ATTACH_LAZY (VMM reserve, no
read; per-expert demand acquisition and reclaim) and requires a valid
per-pack .experts manifest. A lazy failure is terminal for that attach:
the caller's configured lazy load must never silently degrade to a
whole-pack resident arena. Modules that do not implement the
acquisition protocol must leave the env unset.

BUDGETS: the per-pack manifest entry cap (SPARK_WEIGHTD_EXPERT_COUNT_MAX
40960) covers glm53full's 75 routed layers x 256 experts x 2 kinds =
38400 entries (glm5.3-flash needs 24192); the manifest load is a calloc
of count * ~48B (~2MB at the cap). SPARK_WEIGHTD_LAZY_POOL_BYTES_DEFAULT
(8 GiB) is the materialized-expert budget the lazy arena reclaims
against; SPARK_WEIGHTD_EXPERT_BYTES_MAX (64 MiB) caps one expert tensor.

LEASE CONTRACT: SparkWeightdManifestIdentity is the canonical identity
of a successfully loaded, grouped manifest. On
SparkWeightdClientRelease the caller must have established GPU
completion and unmapped first. Export-lease batches index batch_offset
over the sorted union of the lease's physical chunks; chunk_count stays
the arena's total virtual chunk count.

GLM52 CONSUMER MECHANICS (modules/glm52_resident_decode_stage): the
lazy pack qualifies the FP8 codec only — BF16 keeps the resident eager
load. Routed experts stay in the arena's sparse address space and
materialize through per-wave acquisition; the module retains pack
offsets for lease binding while non-expert tensors come from the
compact spine. Kernel-side, expert pointers are consumer-local leased
VMM addresses — the weightd map exposes only acquired extents. Route
results publish to host storage (event + pinned host mirror) only for
slots wired for lazy acquisition; resident and validator slots skip it.
Retained lazy chains awaiting lease recovery are keyed by pipeline slot
in the module state.

## glm52 serving adapter geometry (2026-09)

FLAT RANKS: the adapter exposes FLAT_RANKS flat ranks, one per TP rank,
single PP stage; residentd fans each submission out to every rank
(PARALLEL_FANOUT) and the firmware stage stays STAGE_COUNT=1; the
adapter maps flat rank -> tp_rank and pins the firmware stage to 0. The
5.2 serving band was TP8; the glm53full fleet deploys TP16. The rank
count is a per-deployment environment selection
(SPARK_GLM52_SERVING_FLAT_RANKS, 8 or 16) so one adapter artifact
serves both topologies while each keeps its own adapter identity —
ValidateForAdapter pins deployment node_count == stage_count and the
stage configs' tp_degree == TP_DEGREE. Unset, empty, or nonsense values
leave the descriptor unconfigured and the host's adapter-load
validation fails closed.

LAUNCHER PASSTHROUGH: tools/fleet_serve.sh forwards
SPARK_GLM52_SERVING_FLAT_RANKS to residentd verbatim (no default — an
unset value fails closed at load with the adapter's diagnostic). The
glm53full TP16 window-respawn must export 16 before relaunch; the 5.2
TP8 band exports 8.

DRIVER MODEL ID: the expected DRIVER model id must equal the model.id
of the firmware the driver was compiled from
(ServingAdapterTemplateLoadDriver strcmps them). The bf16 arm's
firmware pins the 5.3-full identity (native publisher precision arm,
per-source firmware pins); every other codec's firmware keeps the 5.2
identity. GLM52_EXPERT_WEIGHT_CODEC is a numeric define.
