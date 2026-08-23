# QUALITY LAW + PROJECT CONTEXT (all agents)

## What SparkPipe IS (read README.md + SPEC.md before any audit or design)
A private on-premises serving engine for open-source frontier models on DGX
Spark fleets (GB10), scaling 4 -> 8 -> 16 boxes WITHOUT changing the serving
API or model packages. Resident model working sets; other configured models
promote in <=1 minute; ONE OpenAI-compatible endpoint. The scheduler chooses
placement, batch width, stage microbatch geometry, speculation policy and
collective algorithm to maximize useful hardware occupancy under caller
priority/deadline. Batch formation is automatic - callers never pick a B-number.
Measured results are recorded separately from projections (PERFORMANCE_STATUS.md).

## Metric: maximize Solutions / (production-codesize SQUARED)


## The KVcache incident (user directive 2026-08-22, canonical example)
A prior cleanup DELETED working JIT KV-cache functionality because a debug
setting had it disabled - "dead code that could be deleted." That was wrong:
config-disabled is NOT deletable when the feature is a project goal. The
completeness matrix outranks size accounting. Never let a size push delete a
main-goal feature.

## Naming
The 27B driver directory/modules named qwen36_* actually serve QWEN3.8-27B.
A dedicated rename (qwen36 -> qwen38) is queued - coordinate timing with the
active dev on those files. Until then: reports should say Qwen3.8-27B; do not
add NEW qwen36-branded user-facing names.
