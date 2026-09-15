# GOALS.md — T1-G53

Fleet goals G1–G6, verbatim short form: G1 fast+correct ≥14 tok/s; G2 accuracy
binding; G3 production legality; G4 fleet stability; G5 multi-dev common
fleet; G6 the architecture runway.

Driver goals:

D1. Driver correctness: T1 PASS on the committed glm5_next fixtures —
    ids/routing exact + bf16 streams within banded rel 0.02 / abs 1e-3 on all
    canonical prompts, EXACT match rates reported separately, raw decode logs
    + comparison output committed as receipts.
D2. Driver performance: MEASURED B1 decode tok/s on the placed FP8 arm as the
    honest serving number; per-op collective latency if the engine exposes
    it; then at most ONE hillclimb iteration on the top roofline lever chosen
    from wave-r measured reports, delta re-measured and reported.
D3. Driver accuracy: the T1 reference comparison status vs
    qualification/t1_reference/glm5_next/ (fixture SHAs host-pinned in
    MANIFEST); any divergence reported as FIRST DIVERGENCE evidence, never
    papered over.
D4. (secondary, standing) Common-code contribution: adopt the shared modules
    (the #997 engine tree) for glm5_next; propose upstream anything generic
    this lane was forced to write family-locally.

## Significance vector

| Result | G1 | G2 | G3 | G4 | G5 | D1-D4 | Verdict |
|---|---|---|---|---|---|---|---|
| Offline gates: verify-manifest PASS + negative control convicts |  | + | + |  | + | D3+ | instrument sound |
| Live fp8.tp16 decode vs fixture: ids DIVERGE (engine fluent, fixture mojibake) | − | + | + |  | + | D1−,D3+ | T1 FAIL as committed; adjudication vs bf16 arm directed |
| Engine wedge (1 request/boot, silent spin) reproduced 2x, routed to coredev | − |  | + | − |  | D1− | production serving blocker, not mine to fix |
| T1 dump hook (G5N-T1) built + GPU-validator PASS | + | + | + |  | + | D1+,D3+ | serving-gated half now exists |
| main adapter-vs-header tap drift + hub stage_count=16 found (TP8 parked) |  |  | + |  | + | D4+ | findings for PR |
| Offline adjudication: reference CLEAN, fixture ckpt != served ckpt (packs = bf16-official-class, router fingerprint 0.984/0.925) | + | + | + | + | + | D3+ | fixture invalidated as-committed; operator canonical-snapshot ruling requested |
| (pending) one-snapshot T1 rerun (decode + streams/routes compare) | + | + | + | + | + | D1,D3 | staged, blocked on incident + ruling |
| (pending) MEASURED B1 tok/s + hillclimb delta | + |  | + |  |  | D2 | pending run |

## Update rule

Every significant event appends a vector row and touches MISSION.md line one.
