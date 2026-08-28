# SparkPipe goals (operator-ratified 2026-08-29)

Scope source: the platform plan (Phase 0-5) + docs/MODEL_SUPPORT.md.
This file is the durable statement agents inherit; the scoreboard and
lane rules carry the tactics.

## HARD CONSTRAINT: quantization policy (operator directive)

We do NOT produce our own quantizations. Model weights arrive
ALREADY-quantized from (a) the publisher's official releases (GLM 5.3
Flash FP8, DSV4 FP8, K3 MXFP4, Qwen Max FP8) or (b) community
quantizations that are VETTED before use (e.g. AMD Quark's MXFP4 —
recepit-verified shard-by-shard; vetting = provenance pinned +
full-receipt hash + quality gate on first serve). Our packers
REPACKAGE — format conversion into stagepacks, TP/PP sharding, scale
plane re-laying — they never quantize a master. Where no acceptable
quantization exists, serve the publisher's native precision (BF16
packs fit the fleet: Qwen Flash bf16 = 84G/rank at TP4) rather than
inventing one. Ideas like "our own NVFP4/FP8 requant" are void; a
future precision change means adopting a NEW official/vetted source
and re qualifying.

## OPERATING PRINCIPLE: the jigsaw (operator directive)

Solve corners and edges first — the well-defined, independently
completable pieces (front-door integrations, staging/manifests,
quality gates, scoreboard, infra rules) — each completion gives the
internal work a solid reference and removes a worry. Prioritize
completions over open-ended explorations when choosing next work.

## SLOP GATES (operator directive)

Incoming code is audited at merge: (a) cyclomatic complexity audit
(mean and max; the historical baseline is mean 7.33 / max 157 —
regressions need in-commit justification); (b) the value metric
Solutions/(Codesize^2) MAXIMIZED — a change that adds lines must
add disproportionately more solution; (c) NO high-level DRY
violations: the ~3,500 pasted adapter-lifecycle lines are the known
debt and the DRY template is its fix; new pasted-lifecycle code is
refused at review. AI slop (plausible filler, unjustified
abstraction, silent fallbacks) gets rejected at merge, not admired.

## Near-term (days)

1. glm5.3 Flash first tokens -> M5 exact-32K B1 + COMPSEC-17.
2. Qwen Flash live 4-node cell + COMPSEC-17 — under the quantization
   policy: serve the bf16 source (84G/rank) unless a vetted community
   quant is verified; NO self-made MX-FP8.
3. K3 first fleet number; P1a retest verdict.
4. Staging complete + tools/staging_manifest.py in the test gate.
5. Every first cell quality-gated (COMPSEC-17 before "usable", full
   92x before "not horrible").
6. liteLLM front door live (controller-side proxy routing every
   deployment; one OpenAI-compatible endpoint for clients).
7. Qwen Flash planned as TP4xPP4 16-rank bf16 (~21G/rank) per the
   multi-topology fleet layout; internal NVMe kept clear of
   non-essentials so topology variants coexist.

## SPECULATION: all providers, one contract (operator directive)

Support every speculation type — MTP, DFlash, DSpark, DFlash2, the
coming DSpark2, and whatever follows — because not every model gets
today's best. Design + sequencing: docs/SPECULATION_PROVIDER_DESIGN.md
(provider = capability unit behind the adapter; lifecycle+contract
abstracted, inner loops stay provider-owned for zero hot-path cost;
DSpark2 = a new provider module, not five family edits).

## Medium-term (weeks)

1. All product-set models serving honest perf+quality cells; the
   scoreboard gaps closed (Flash x2, Pro, then Max).
2. Qwen Max served from the official FP8 or vetted MXFP4 source.
3. Phase-1 PLATFORM work resumed as first-class lanes, not backlog:
   DRY adapter template (one lifecycle, not seven), the device API
   (memory-first per docs/INFERENCE_OS_DESIGN.md), and the RECIPE
   COMPILER (contract -> family header/module/adapter/packer/bench,
   the amortizer that makes model N+1 days not weeks).
4. Co-residency proven (110 GiB ceiling) -> multiple models serving
   simultaneously; utilization story per node.
5. Speculation re-qualified per model (env-gated DSpark/DFlash2) —
   the 2-3x class toward the 110%-of-public policy.
6. Serving completion: B-ladder honest re-measure (P1b cliff fixed),
   prefill regressions closed, prefix caching + JIT-KV tiers live.

## SOTA NORMALIZATION (operator directive 2026-08-29)

4x faster on 4x sparks is NOT 4x SOTA — it is 1x. All claims
normalize to SPARK-EQUIVALENT throughput: single-spark cells are the
head-to-head SOTA++ baseline FIRST; topology stacking (TP/PP/EP) is a
SEPARATE, second-axis gain reported alongside, never blended into the
per-spark number. The scoreboard carries both axes explicitly.

## QUALITY STUDY (queued, AFTER the basics work)

Quantized-experts-with-FULL-RESOLUTION spines now available for some
models — enables a three-way quality comparison per model: (a) full
resolution everywhere, (b) quantized experts + full-res spine, (c)
fully quantized. The ds4_eval 92x suite is the instrument (quality,
not vibes). Priority: after first tokens + COMPSEC-17 on the fleet —
the format-quality map feeds the island catalog's precision-variant
recommendations.

## INCOMING SOURCES (sysadmin, ~1 day)

GLM 5.3 (full) official BF16 + FP8 + radixark NVFP4 (community —
VETTING REQUIRED per policy); Qwen Max 4-bit found. Implication:
multiple expert precisions per model, FULL-SIZED spines — the island
catalog gains a precision-variant axis; the quant policy's
official/vetted rule applies per variant.

## SPECULATOR PORTFOLIO (operator)

A dozen speculators incoming for head-to-head testing. The provider
abstraction (SPECULATION_PROVIDER_DESIGN.md) is the harness for it.
The operator's multi-speculator-live idea is assessed as the
TOURNAMENT PROVIDER: see the design doc's addendum.

## MARKETPLACE TRACK (docs/MARKETPLACE_PLAN.md — the commercial frame)

The long-term platform goal has a business model: a two-sided compute
marketplace (SparkPipe-required supply, liteLLM demand door, 10%-
in-compute take, anti-cheating via off-node replay of real traffic).
BUILD HOOKS in priority order (§7): (1) liteLLM door DONE; (2)
REQUEST-LOGGING PIPELINE keyed (driver hash, contract hash, request) —
the audit substrate, first real build item; (3) audit service; (4)
provider onboarding; (5) fee metering. TECHNICAL GATES from the
determinism appendix: API logprob RETURN support (the temp-independent
audit signal — today the API returns tokens only); seed field passthru
+ logging when sampling lands (counter-based RNG, no runtime-varying
state); receipts language notes greedy-only (temp=0 proofs). Provider
contract: no forcing temp>0, no refusing seed logs.

## The island-catalog manifest (operator-ratified abstraction)

An ISLAND = compute nodes + fabric + a CATALOG of pre-vetted model
descriptions. The catalog entry = contract (pinned source, quant-policy
compliant) + topology for that hardware + resource envelope (vs the
110 GiB ceiling) + qualification receipts — one per model PER HARDWARE
CONFIGURATION. It is the recipe compiler's output format, the object
liteLLM islands register against, and the single source of truth any
future UI (operator + request console) reads — never the UI's own
state. Multi-island federation routes through liteLLM (priority/
health/load in its config, not our runtime). Named dependency for
text-in island routing: the Phase-4 tokenizer sidecar at the island
edge (today's token-ID contract works via LiteLLM's /vllm/ passthrough
— proven, merged).

## Long-term

The platform plan's end state, restated: SparkPipe as the inference
OS for this fleet — every product-set model (MODEL_SUPPORT.md catalog,
open to new ones via the same contract) quality-gated and beating
110% of the best public comparable per recorded cell; multiple models
co-resident; and the stack PROVABLY hardware-independent below the
module boundary: cuda (production), host oracle (CI), Metal (the
controller Mac), ROCm (rented, budget-approved) — one memory model,
one comms model, recipes not hand-built drivers.
