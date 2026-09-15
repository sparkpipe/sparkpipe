# MISSION.md — T1-G53 (glm53flash / glm5_next)

glm53flash serves ≥14 tok/s correct tokens under production law. Its critical
path: (1) T1 PASS — the two committed canonical prompts decoded through the
real driver on the placed fp8.tp16 packs over weightsd, ids/routing exact and
streams in-band vs the committed fixtures, receipted with raw logs; (2) T2
baseline — MEASURED B1 decode tok/s on that arm with raw timing logs; (3) one
roofline hillclimb iteration picked by measured data, delta reported.

Measured anchors: T1 instrument on lane/wave-t1-offline @ cc5d5f8 (fixtures +
MANIFEST + compare tool) MEASURED present; verify-manifest PASS + corrupt-fixture
negative control convicts MEASURED 09-15; live fp8.tp16 serving arm decodes
capital_of_france -> [12089,13,758,8584] (" Paris. In French", fluent) vs
committed fixture [3837,271,271,12] (mojibake) MEASURED 09-15 — token-id gate
FAILS as committed, adjudication vs the bf16 arm directed by mgr2; production
engine wedges after ~1 request per residentd boot (main-thread 100% spin, zero
log errors; api then loops engine_connect status=4) MEASURED, reproduced 2x,
routed to coredev; T1 dump hook (G5N-T1 log lines) built into the glm5_next
module, GPU component validator PASS MEASURED 09-15; main's serving-adapter
target does not compile vs main's firmware header (tap-define drift; hub core
is ahead of main) MEASURED 09-15; hub adapter descriptor hardcodes serving
stage_count=16 MEASURED (TP8 lane parked); T2 number: NOT YET MEASURED —
anything else is ASSUMED and says so.

Behavioral rules (binding):
1. T1 accurate-inference receipt is the default work; nothing precedes it
   except the mesh lease and fleet recon that make it possible.
2. A slow-but-accurate state is a diagnostic reference, never a resting state.
3. A fast method proven accurate gets integration effort, not suspicion.
4. A window that doesn't advance the critical path says so explicitly.
5. Every claim graded MEASURED/DERIVED/ASSUMED; ungraded = not a fact.
6. Update MISSION.md + GOALS.md on every significant event — no exceptions.
7. Operator directive binds this wave: ACCURATE INFERENCE FIRST, then
   hillclimb; no cheating, lying or fraud; actual real results. A failed T1 is
   reported as a failure with FIRST DIVERGENCE evidence; tolerances are never
   loosened; no rerun-until-pass.
