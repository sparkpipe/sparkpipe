# W2 dry-template continuation — items 2-5 report (2026-08-28)

Branch `lane/dry-template2` (worktree /tmp/lane-dry2, from main
de94765). Four commits, per item. spark5 reserved (holder
lane-dry-template2); full `make test` + gates queued as
`dry2-full-validation` before any merge. NOT merged (never merge own
PR).

## Item 2 — serving-adapter template (commits 1)

`include/sparkpipe/spark_serving_adapter_template.h` +
`runtime/serving_adapter_template.c`:

- Descriptor identity macro (`SPARK_SERVING_ADAPTER_DESCRIPTOR_IDENTITY`)
  and the shared capability chain (`..._CAPABILITY_CHAIN`, base = PREFILL
  | DECODE | DRIVER_OWNS_KV — the exact intersection of all five
  adapters; extras stay family-side).
- ONE tp_collective config parser. Family policy is data:
  `SparkTpCollectiveConfigPolicy` {peer_count,
  allow_zero_collective_identifier, require_contiguous_peer_ports,
  algorithm flavor, threshold flavor}. The dsv4 contiguous-port check
  (with its overflow guard) moved verbatim from the adapter Initialize
  block into the parser; glm52/glm5_next adopt it at their cutover —
  note: the template's overflow-guarded check REJECTS wrapped ports
  their current code can mis-accept; their cells must be re-read at
  cutover.
- ReservePending spine: template fills the common submission view +
  last_row_by_lane (family layout via offsetof); the FAMILY sets the
  active flag after its own fill steps — a failed cache-lane/emit-row
  fill leaves the slot free, exactly as pasted.
- LoadDriver spine: reset/load/identity (target pin optional)/program
  lookup/admit+submit presence/create-request/null-instance guard. The
  flag+profile acceptance is a family CALLBACK: the 27b inline condition
  and the dsv4 SparkModelDriverProgramSupportsRuntimeLimits call are NOT
  equivalent (the helper checks profile->max_inflight and
  max_resident_sequences too) — a shared approximation would have
  changed accept/reject behavior.

Cut over: **qwen38_27b** (reference) and **dsv4** (all four topology
builds: TP4, TP16, PP13, TP4xPP4 — compile-checked per build on the Mac,
full cells on spark5). glm52/glm5_next/k3 + the qmax<->q4f pair follow
after review (mid-flight code, per the brief). dup_report 93 -> 84: every
27b/dsv4 adapter-side row and all dsv4<->glm52/glm5_next TP-config rows
died.

## Item 3 — stagepack format library (commit 2)

`include/sparkpipe/spark_stagepack_format.h` +
`runtime/stagepack_format.c`. The two W4 C3 targets (qwen38_max <->
qwen4_flash `StagePackShapeGdn`/`StagePackEveryLayer`, plus the 13-line
`HeaderMatches` pair) consolidated: shared per-layer tensor axis (kind
numbers 3..21 both v1 inventories already agree on), geometry as data
(`SparkStagePackGeometryTable`), neutral weight-format/layer-class codes
pinned per family with _Static_assert, and
`SPARK_STAGEPACK_HEADER_LAYOUT_PROOF` — an assertion struct giving the
byte-order/layout bug classes compile-time teeth.

Receipts: old vs new implementations compiled side-by-side per family
over the FULL tensor-kind space — every (rows, columns, natural_format,
layer_class) tuple identical, header match/drift group codes identical
(qwen38_max: 32 kinds; qwen4_flash: 55 kinds incl. the family-only HC/
indexer/mixer/PLE cases left family-side). Python packers untouched.

## Item 4 — memory-M1 rides the template (commit 1)

`include/sparkpipe/spark_memory_buffer.h` + `runtime/memory_buffer.c`:
`SparkMemoryBuffer {pointer, space, bytes}` — HOST_COHERENT/HOST_PINNED/
DEVICE_PRIVATE allocate+free, FILE_BACKED declared for M2 and refusing
(SPARK_STATUS_UNSUPPORTED) until then; `SparkMemoryBufferCopy` resolves
direction FROM THE SPACE TAGS, refuses undefined pairings/overruns
(INVALID_ARGUMENT) and maps cuda errors to IO_ERROR as pasted; view
macro for memory we don't own. The comment rule is now a type: 27b's
five pool allocations + block-table upload carry handles; dsv4's state
allocation is typed (self-allocation handle). Same allocators, same
failure statuses.

## Item 5 — speculation-provider slot (commit 3)

`include/sparkpipe/spark_speculation_provider.h` +
`runtime/speculation_provider.c` +
`tests/test_speculation_provider_slot.c` (in `make test`). Interface:
kind enum, capability query with NAMED refusal, draft lifecycle
(inner loops stay provider-owned — zero hot-path indirection), ONE
verify contract (accepted_token_count/chain_width/tokens_per_sequence —
the lease-advance bug class structurally dead at migration), KV
interaction flags, canonical env schema. BOTH binding shapes proven
through the identical adapter-side calls: module-provider (resolved like
a firmware-module unit — the dspark dispatch-policy backend shape) and
embedded-provider (static ops table in the adapter — the block-drafter
shape). The two-shape mapping is recorded in
docs/SPECULATION_PROVIDER_DESIGN.md ("Slot landed" section). NO family
migration this sprint.

## Gates

- dry-law: PASS (all new shared headers/files model-neutral).
- cyclomatic: every NEW function CCN <= 23 (the shared HeaderMatches was
  transplanted at 27, split into five group compares instead of
  justified). Changed pre-existing functions: no CCN growth.
- dup_report: 93 -> 84 (27b/dsv4 adapter rows + both C3 rows dead; the
  remaining rows are the families queued for cutover after review).
- code-size (ratchet): 216444 exact, ceiling moved +1123 with the
  accounting in-commit — libraries paid once, two of five families cut
  over; the remaining cutovers (glm52/glm5_next/k3, qmax<->q4f) are PURE
  DELETION and return the counter under the old ceiling. Honest note:
  this is the one gate where the landing is not yet net-negative; the
  alternative was cutting the template's contract documentation.
- Cells (spark5, queued): full make test incl. test_qwen38_27b_serving_
  adapter, test_dsv4_{tp16,tp4,tp4_pp4}_serving_adapter (the tp_collective
  schema-error fixtures), both stagepack families' module archives +
  python pack tests, test_speculation_provider_slot.

## Coordinator decisions needed

1. Merge order: items 2+4 land together (one seam); 3 and 5 independent.
2. glm52/glm5_next/k3 adapter cutovers next sprint: the template's
   contiguous-port check is overflow-guarded (stricter than glm52's
   current parse) — expect their tp_collective error fixtures to need a
   re-read, not a revert.
3. The +1123 ratchet spend recovers at the next family cutover
   (glm52+glm5_next alone are ~-800 of pasted TP-config parse).
