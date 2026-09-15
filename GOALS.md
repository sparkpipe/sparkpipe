# GOALS.md — T1-GEMMA4 (26b retarget)

Fleet goals G1–G6, verbatim short form: G1 fast+correct ≥14 tok/s; G2 accuracy
binding; G3 production legality; G4 fleet stability; G5 multi-dev common
fleet; G6 the architecture runway.

Driver goals:

D1. Driver correctness: gemma4-26b A4B serving bring-up on TP4 (replica law
    rank = node % 4), weightsd config-only attach, ACC-2-verified packs;
    both fixture prompts decoded (per-boot if the converged-engine wedge
    fires), raw logs committed as receipts.
D2. Driver performance: ONE measured B1 decode tok/s on the placed bf16/TP4
    arm, graded MEASURED, stated against the 26b roofline ceiling 230.4
    tok/s (DERIVED, R-wave); no hillclimb, no transport tuning this wave
    (accuracy-phase directive).
D3. Driver accuracy: 26b fixtures regenerated from the 26b checkpoint via
    the same REF-PORTS-2 reference engine (31b fixtures pinned 31b — do not
    compare across variants); t1_reference_compare.py PASS on both prompts —
    routing ids + token ids exact, streams banded (rel 0.02 / abs 1e-3);
    exact-id match rates reported separately; any divergence = FIRST
    DIVERGENCE evidence, no loosening.
D4. (secondary, standing) Common-code contribution: consume PR #998's shared
    gemma4 modules (defines/hybrid_state/rope_plan — already on main); fetch
    origin lane/t1-g53 as the template for the module-side env-gated T1 dump
    and propose upstream anything generic forced family-local.

## Significance vector

| Result | G1 | G2 | G3 | G4 | G5 | D1-D4 | Verdict |
|---|---|---|---|---|---|---|---|
| Retarget 31b→26b absorbed (operator correction); mission files updated |  | + | + |  | + | D3+ | brief corrected before work |
| 31b fixtures + reference decoder staged from wave-refs2 onto lane/gemma4-t1 |  | + | + |  | + | D3+ | instruments in place |
| (pending) 26b packs located/placed per TP4 replica law | + | + | + | + | + | D1 | pending |
| (pending) 26b fixtures regenerated via reference engine |  | + | + |  | + | D3 | pending |
| (pending) serving bring-up TP4 | + | + | + | + | + | D1 | pending |
| (pending) T1 compare both prompts | + | + | + |  | + | D1,D3 | pending |
| (pending) PR lane/gemma4-t1 + receipts |  |  | + |  | + | D4+ | pending |
| (pending) ONE measured B1 tok/s | + |  | + |  |  | D2 | pending |

## Update rule

Every significant event appends a vector row and touches MISSION.md line one.
