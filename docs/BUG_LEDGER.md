# The bug ledger — one coherent plan, not monkey patches (2026-08-30)

THE OPERATOR'S DIRECTIVE: dozens of bugs; fix them coherently; redesign
where a redesign is cheaper than the sum of patches. This ledger is
the single list (every kimi finding + every lane receipt), each with a
FIX STRATEGY class — because the right response to "the same mistake
at every stage" is a structural fix, not twelve local ones.

## The three systemic patterns → three structural programs

PATTERN A "staged path exists but isn't wired" (R1 screened head,
JIT-KV's 85%, prefill width, the collapse of 3 spill mechanisms):
the fix class is WIRING + an ADOPTION GATE (already live: new
families must consume). No new abstractions — wire the certified
ones. OWNER: the perf programs' lanes, in kimi's ranked order.

PATTERN B "per-token work on the serialized path" (validation
fanout, block-table uploads, host bubbles, PP lockstep): the fix
class is the STEP-LOOP REDESIGN — P1's chain+async (the dsv4
Rosetta Stone), THEN step-level pipelining, THEN the fanout hoists.
Sequenced as one program (PERF_PROGRAM P1+R5+R7), not isolated
patches — the hoists only pay once the loop is async.

PATTERN C "latency-bound kernels filling too little machine"
(attention family 30-60x, WS pipelining, scalar GEMV): the fix
class is KERNEL PROJECTS with the oracle-first discipline (host
oracle + validator entries BEFORE touching, per kimi's warning).
R3 flash-decode is the flagship; R4 amortization second.

## The discrete correctness bugs (each: root-cause → gate, not patch)

| Bug | Strategy | Status |
|---|---|---|
| EOS unwired | wire engine stop set + request field | FIXED (4e2f19f) |
| Cancel unwired | disconnect→engine Cancel (existing API) | queued, mine |
| O(n²) token parse | dies with JSON edge / one-pass if kept | queued |
| trap-on-corruption (experts) | fail-frame semantics, never context death | lane: kernel-crew |
| sparse-attn bounds | bounds check at the index consumer | lane: kernel-crew |
| UE8M0 round-down | round-to-nearest in the shared encoder | lane: kernel-crew |
| rANS smem unbounded | bounded window + fail-loud | lane: kernel-crew |
| HC-width collectives | the twin (done) + GENERALIZE: collective takes a width param, not a second instance | redesign queued |
| template offset-0 | the common_offset fix (done) + layout-as-data everywhere | done + gate |
| fused (start,count) slicing | fixed (done) + a packer SHAPE CONTRACT test so section-slicing bugs die at build | add to lane |
| prefix-cache collision | SHA-256 digest (done) — pattern reused for tier (done) | done |
| write-back wedge | degrade-not-stall (done) + the SAME policy audited in every retry loop | audit lane |
| KDA state persistence | lane-closeout's active hunt (state reset on acquire) | in flight |
| registrar flap | two-phase startup (in flight) — the structural fix | in flight |
| memory-contracts 285 | parked-ratchet (shrinks only) | live gate |

## The redesigns deliberately chosen (not patches)

D1 COLLECTIVE WIDTH-PARAM (replaces per-family twins): the submit API
takes element_width; RegisterCredits prices from it. Kills the
HC-twin pattern before it multiplies across 7 families.
D2 STEP-LOOP (P1+pipelining): one loop redesign replaces the bubble
patches.
D3 SPILL COLLAPSE: one spill mechanism (the pager) + Mooncake
boundary, replacing three parallel ones.
D4 VALIDATION ONCE: immutable inputs validated at connect; the
per-frame path carries only the mutable delta.

SEQUENCING: correctness bugs first (they gate the measurements), then
R1/R5 (cheap wins, receipts), then D1 before any new family hits the
collective, then P1/D2 (the loop), then R3/R4 kernel projects, then
D3. The ledger is THE checklist for the next kimi pass.
