# The bug ledger — one coherent plan, not monkey patches (2026-08-30)

THE OPERATOR'S DIRECTIVE: dozens of bugs; fix them coherently; redesign
where a redesign is cheaper than the sum of patches. This ledger is
the single list (every kimi finding + every lane receipt), each with a
FIX STRATEGY class — because the right response to "the same mistake
at every stage" is a structural fix, not twelve local ones.

## The three systemic patterns → three structural programs

PATTERN A "staged path exists but isn't wired" (R1 screened head,
JIT-KV's 85%, prefill width [FLASH WIRED 2026-08-30: the 16-row cap
decoupled + 1024-row chunks, main 1554464; qwen template + dsv4
bulk-prefill kernel remain], the collapse of 3 spill mechanisms):
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
| Cancel unwired | disconnect→engine Cancel (existing API) | FIXED (40ca88c + queued-orphan window closed, lane/kimik3-dev 2026-08-30: orphaned requests are never submitted, worker/connector mutex handshake cancels exactly once) |
| O(n²) token parse | dies with JSON edge / one-pass if kept | FIXED (f175099: sequential array accessors; request-scale loops one-pass) |
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
   patches. HOST HALF LANDED (lane/p1d2-steploop, in fa337b7's program:
   async adapters drain to adapter-declared backpressure, write-through
   client, bubble patch deleted — receipts in
   reports/p1d2-steploop-2026-08-29.md). MODULE HALF (per-family
   chain+async ports) remains fleet-gated; the k3 port design is filed
   (reports/kimi-k3-p1-port-plan-2026-08-30.md: adapter pending slots,
   LaunchHostFunc stream-ordered completion, staged A/B/C with the
   oracle + max_ops_per_pass>1 receipt).
D3 SPILL COLLAPSE: one spill mechanism (the pager) + Mooncake
boundary, replacing three parallel ones.
D4 VALIDATION ONCE: immutable inputs validated at connect; the
per-frame path carries only the mutable delta.

SEQUENCING: correctness bugs first (they gate the measurements), then
R1/R5 (cheap wins, receipts), then D1 before any new family hits the
collective, then P1/D2 (the loop), then R3/R4 kernel projects, then
D3. The ledger is THE checklist for the next kimi pass.

## The salvage list (from the PR triage 2026-08-30 — value to re-land, each re-pointed)

S1 16-BYTE VECTORIZED GEMV loads (ex-#651 MXFP4/FP8, ex-#652 BF16):
   R4-class kernel filling; re-measure vs CURRENT baseline (the
   B1=B2 correction is why), then land through the oracle gates.
S2 DSPARK SPEC_STEP PRO ALIAS (ex-#672): small real mapping — folds
   into the Pro driver-rebuild item.
S3 EXACT-32K ADMISSION (ex-#719): the math was right, the KV budget
   wasn't — re-opens with the JIT-KV tier or sized-down lanes.
S4 P1A COMPLETING POINTS (ex-#735): B48/32/16 + B24x2 behind the
   fleet window (queued).
19 PRs triaged total: 11 superseded/rejected closed, 8 value-carrying
re-pointed (7 closed into programs, #740's lane continues live).
