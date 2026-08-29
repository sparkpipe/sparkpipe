# Cleanup program — the three-audit response (operator priority, 2026-08-30)

The audits' cross-cutting diagnosis is accepted as the headline: THE
RIGHT ABSTRACTIONS EXIST — the failure is ENFORCEMENT at the module
layer. This program closes that loop and takes the ranked wins.

## The enforcement gate (the loop-closer — lands first)

A TEMPLATE-ADOPTION gate, mechanical like dry-law: families must
CONSUME the shared pattern, not fork it. Checks (tests/test_template_adoption.py):
- serving adapters: the family's adapter consumes the template's
  tp_collective parser + capability chain (grep the template symbols;
  a family-local parser copy = FAIL with the paste lineage shown)
- pack_synthesize: tools/<family>_pack_synthesize.c imports the shared
  core; a standalone quartet member = FAIL
- work_control / batch_tuning: shared include consumed, not a
  byte-sibling
NEW families hit the gate at merge; existing offenders get named
deadlines in the failure message (adoption in flight below).

## The DRY takes (the audit's table, sequenced by its own risk calls)

WAVE 1 (low risk, ~3.9k lines — parallel agent NOW):
- pack_synthesize quartet → one parameterized generator (~1,090)
- work_control trio: max↔flash share 337/337 (~890)
- batch_tuning quartet — byte-identical but 2 ids (~615)
- validate_*.sh sextet → one parameterized driver (~590)
- python residual incl. the DORMANT glm52_resident_pack_common.py
  (zero importers — delete with receipt) (~700)
WAVE 2 (low risk, adoption precedent x3, ~1.4k): qwen38_max +
qwen4_flash + glm5_next adapters onto the template.
WAVE 3 (medium — design review first): qwen38_max↔qwen4_flash module
core re-parameterization (~2,800; the fork the lane brief forbade);
stagepack_format geometry-as-data (~1,000); firmware headers (~950).
GATED (do NOT touch): glm52 deletion (~14k) — AFTER glm5_next
qualification, one shot. Validation .cu harnesses: deliberately
independent, never merge (control-vs-candidate doctrine).

## Cyclomatic (complexity lane)

1. SCOPE HONESTLY: */validation/* out of the merge-gate metric into
   its own budget (the audit's own recommendation — scoping, not
   gaming; the 8.01-vs-7.33 gap was largely this).
2. STAGE ZERO (zero risk): evict the 5 getenv flags + 55-line /tmp
   debug dump from the 158-CCN DsparkBlockForward — before the P1
   decomposition the plan already names.
3. CONJUNCTION SOUP → predicate/field tables: ValidateRankPlan (87),
   ValidateDescriptor (80), Dsv4ModuleConfigure's 30-term ||, and
   StagePackExpectedShape (84 → ~6 as a static table).
4. The max must MOVE DOWN each merge window while the plan exists —
   a plan that doesn't bind is the audit's sharpest line.

## Hardware independence (device-typedef lane, next wave)

The single highest-leverage move per the audit: a device typedef
layer (SparkDeviceStream/Event/Graph/Error) retyping the launch seam,
cascading all nine *_module.c at once — the four pollution patterns
(KV staging memcpy batches, mapped-host credit setup x4 clones,
graph capture in orchestration, cudaStream/cudaEvent in the SHARED
stage_module_common.h) all cross it. Seams already present:
spark_memory_buffer.h, runtime/launch.h, the ring control plane.
CUDA-graphs-on-Metal gets the re-record fallback behind the
abstraction. This is the inference-OS ladder's launch leg with the
audit's concrete shape.
