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
| 31b fixtures + reference decoder staged; offline gates PASS + negative control convicts (exit codes receipted) |  | + | + |  | + | D3+ | instrument sound |
| Reference engine extended for 26b MoE from module-kernel trace; 26b header+prompts pinned |  | + | + |  | + | D3+ | fixture-grade path prepared |
| 2 ACC-2-verified stage2 packs located + placed per replica law (spark8/sparkb, dest sha == source sha) | + | + | + | + | + | D1+ | reuse-first honored |
| 3 main-compile gaps fixed (shared combine kernels, mesh-kernels timer, register-header C-guard, validator rewire to shared launchers) |  | + | + | + | + | D4+ | family publishable again; donor-symbol class flagged fleet-wide |
| GPU validator dataflow tier 28/28 PASS (receipt) incl. MoE router + rope tables | + | + | + |  | + | D1+,D3+ | kernel tier green on the new archive |
| Chain-tier SIGSEGV inside libcuda 4x under 3-tenant co-residency (gdb signature receipted) — pre-existing r3 quiet-GPU class, one in-lane diagnostic attempt (store clean) | − |  | + | − |  | D1− | publish gate BLOCKED; node reboot/lane drain = operator class |
| Serving bring-up, T1 compare, B1 tok/s: NOT RUN (blocked above); warm batch queued behind HY4-T1 lease | − |  | + |  |  | D1−,D2−,D3− | honest blocked state, resume path scripted (tools/gemma4_t1_lease_batch.sh + node trees) |

## Update rule

Every significant event appends a vector row and touches MISSION.md line one.
