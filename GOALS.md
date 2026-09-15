# GOALS — GFULL-T1

Fleet goals G1–G6 (word for word from the fleet GOALS.md): G1 fast+correct
≥14 tok/s; G2 accuracy binding; G3 production legality; G4 fleet stability;
G5 multi-dev common fleet; G6 the architecture runway.

Driver goals:

D1. Driver correctness: T1 gate on glm53full.fp8.tp16 — reference fixtures
    (T1R1, MANIFEST, negative controls convicting by name, determinism
    byte-identical reruns) + serving-gated decode via the shared weightsd
    compared with t1_reference_compare.py (ids/routing exact, in-band streams).
D2. Driver performance: first measured B1 tok/s after the T1 verdict; the
    levers beyond it live in the glm52 routed-path (A-0058 double-route class)
    and the lazy expert lease path; per-kernel GPU-time capture (#924 facility)
    is the shared instrument when the hill climb opens.
D3. Driver accuracy: this wave = first T1 for glm53full (family uncovered
    before it); gap after PASS = T2 B1 measurement; gap after FAIL = FIRST
    DIVERGENCE evidence, no loosening, no rerun-until-pass.
D4. (secondary, standing) Common-code contribution: consume the T1 instrument
    as-found (t1_reference_common/decoder/compare); upstream anything the
    glm52 port forced family-locally (driver tap hook pattern is the
    candidate); adopt the ACC-1 packer plan re-derivation instead of
    duplicating fp8 slice/scale math.

## Significance vector

| Result | G1 | G2 | G3 | G4 | G5 | D1-D4 | Verdict |
|---|---|---|---|---|---|---|---|
| (pending: fixture port committed) | | | | | | | |
| (pending: serving bring-up verdict) | | | | | | | |
| (pending: T1 verdict + receipt PR) | | | | | | | |
| (pending: measured B1 tok/s + WS) | | | | | | | |

Update rule: every significant event appends a vector row and touches
MISSION.md line one. The manager reads both at every dispatch and report.
