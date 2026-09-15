# MISSION.md — T1-GEMMA4 (gemma4-26b A4B bf16/TP4; retargeted from 31b 09-15)

gemma4-26b A4B serves correct tokens under production law; the fleet tok/s
target is NOT this wave's gate — the operator directive for this wave is
ACCURATE INFERENCE ONLY, slow-but-reliable allreduce acceptable, no perf
work. RETARGET (coordinator, operator correction): this lane is the 26b
A4B variant (the smoke-selected family, 0.93 GB/tok working set), NOT 31b.
Its critical path: (1) bring-up — the family has NEVER been served; locate
the ACC-2-verified 26b TP4 packs (rank0 stage2 FULL 134/134, rank3 66/134),
reconcile placement vs the TP4 replica law (rank = node % 4); (2) fixtures
— REF-PORTS-2 #1017 fixtures are 31b-pinned (MANIFEST checkpoint path
gemma-4-31b-it): port the 26b config through the same reference engine
(family llm_defines carries both variants) and regenerate 26b fixtures
before the compare; (3) T1 verdict — decode the two canonical prompts
through the real serving arm, ids/routing exact + streams in-band, FIRST
DIVERGENCE evidence on any mismatch, exact rates reported separately
(31b greedy chains split at position 2 on a near-tie — verify the same
for 26b); (4) ONE measured B1 tok/s vs the 26b ceiling, then STOP.

Measured anchors: 31b fixtures at lane/wave-refs2 (PR #1017 OPEN)
MEASURED present, MANIFEST pin: checkpoint gemma-4-31b-it, config
e967dd38, fixture SHAs cb80dfaa / 831d0252; reference engine validated
bitwise 0-ulp vs committed anchor kit + publisher HF top-1 exact at
decode positions 0-1 (REF-PORTS-2 receipt) — 3 engine bugs caught before
fixture cut (KV u16 readback, missing post_attention_layernorm, k_eq_v
full-layers-only) MEASURED; 26b packs ACC-2 checkpoint-faithful MEASURED
(134/134 + 66/134 walks, bitwise f32 rope tables, layer_scalar real);
gemma4-26b roofline ceiling 230.4 tok/s DERIVED (R-wave ledger); gemma4-31b
ceiling 15.5 tok/s DERIVED (superseded for this lane); common modules PR
#998 MERGED (9c7ee3f) MEASURED; weightsd LIVE 16 nodes config-only
MEASURED per dispatch; SMOKE-MD audit: gemma4-26b NOT found on any
sparkdata root — path-convention mismatch suspected MEASURED; B1 tok/s:
NOT YET MEASURED — anything else is ASSUMED and says so.

Behavioral rules (binding):
1. T1 accurate-inference receipt is the default work; nothing precedes it
   except the co-residency constraints (3 other accuracy lanes live: glm53
   flash TP8, qwen38max TP16, ling TP16 — touch nothing of theirs).
2. A slow-but-accurate state is a diagnostic reference, never a resting
   state — but THIS wave ends at T1 + one B1 measurement by directive.
3. A fast method proven accurate gets integration effort, not suspicion.
4. A window that doesn't advance the critical path says so explicitly.
5. Every claim graded MEASURED/DERIVED/ASSUMED; ungraded = not a fact.
6. Update MISSION.md + GOALS.md on every significant event — no exceptions.
7. A clean bring-up failure reported with evidence is an acceptable
   outcome: capture the exact defect, ONE in-lane fix attempt, then report
   either way. FAIL is honest; no tolerance loosening; no rerun-until-pass.
8. The known converged-engine defect (wedges after ~1 request) is worked
   around per-boot; it is coredev's to fix, reported with evidence.
