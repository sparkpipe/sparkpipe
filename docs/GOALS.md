# SparkPipe goals (operator-ratified 2026-08-29)

Scope source: the platform plan (Phase 0-5) + docs/MODEL_SUPPORT.md.
This file is the durable statement agents inherit; the scoreboard and
lane rules carry the tactics.

## Active GLM 5.3 Flash goal (operator directive, 2026-09-08)

Deliver correct, fully functional, non-speculative serving on the 16 Sparks.
TP4xPP4 is the default topology; TP16 is the speed topology. Target TP16
throughput of 3.5 times a qualified TP4 baseline at matched precision,
context, batch occupancy and timing boundaries. Establish the TP4 memory
roofline using measured sustainable bandwidth and actual weight/cache bytes,
then close the measured gap. Speculative or differently quantized public
results are separately identified comparisons, not the baseline.

Report two acceptance results for each driver: functional qualification and
hardware-normalized SOTA performance qualification. Passing the former while
lagging in the latter is a visible optimization backlog, not a waiver or a
claim of completed performance work. Show raw throughput, node count, source
precision, topology, context, occupancy and speculation first. For
bandwidth-bound comparisons, a 25 token/s FP8 TP4 result can be comparable to
a 50 token/s 4-bit TP4 result under a stated two-times-weight-traffic
assumption; refine that assumption with measured bytes, dense weights, cache
traffic and quantization overhead. The same 25 token/s on TP16 is not the same
hardware-normalized result. TP4xPP4 must demonstrate approximately four times
TP4 serving capacity with a full pipeline, not just accept four times as many
queued requests. Use both normalized comparisons and measured efficiency to
select the next hill-climbing step.

Required serving behavior cannot be waived by driver capability flags,
environment variables, stubs or qualification paperwork. Missing required
operations fail explicitly and prevent production acceptance. Prove behavior,
not just callback presence: arbitrary-size continuous batching with arrivals
and completions; bounded, fair chunked prefill; JIT KV residency and complete
prefix-state restore; cancellation, release, slot reuse and clean restart.
Cache restoration includes model recurrent and convolution state as well as
KV/index state. A benchmark requiring a warm cache hit must reject a miss.

Refactor reusable work into the existing common runtime, admission, row
layout, cache, execution, kernels and transport code. The dozens of planned
drivers must not each implement scheduling, cache transactions, collective
policy, progress, completion lifetimes or deployment. Keep model geometry,
state layout and mathematical stages in the model driver. Consolidate real
duplicated paths; do not introduce another parallel framework.

Use the qsort pattern: one optimized common algorithm parameterized by narrow
operation callbacks and an opaque context. Call at stage/batch boundaries,
not once per tensor element. Keep scheduling and cache policy independent of
hardware; allocation, execution, events, copies and collectives belong behind
the existing device/backend interfaces. Eight Mac Studios arrive in October
2026: CUDA streams, CUDA pointers and NCCL semantics must not leak into common
serving policy. Qualify the common algorithms with the host backend now and
the Metal backend on that hardware when available.

Execute distinct ready lanes together and reuse dense weight tiles across
rows; group routed MoE rows by expert for weight reuse. Qualify odd batch sizes
and dynamic occupancies through roughly 100 requests, measuring the memory
and compute crossover. No hidden serialization, power-of-two admission rule,
or production clamp may substitute for implementing the required behavior.
Behavior-changing diagnostic clamps belong behind #ifdef DEBUG and must be
reported; DEBUG does not excuse missing mandatory implementations.

Implement B1 contribution broadcast with local reduction and B2+ tree
reduction overlapped with independent compute. Preserve logical batch identity
across microbatches, collective ordering, independent workspaces and GPU
completion lifetimes. Measure exposed communication, launch gaps and the
weight-reuse cost of splitting. Use topology-specific rank balance and
verified data-plane routes, not management-network assumptions.

Iterate on correct merged-main builds: numerical/lifecycle gate, cached warm
decode and separate prefill measurement, critical-path profile, focused shared
code/kernel fix, repeat. Report per-sequence latency, aggregate output rate,
TTFT, bandwidth efficiency and exposed communication with reproducible
SHA/command/hardware receipts. A component pass or merged incremental PR is
not completion. Continue until functional acceptance and performance targets
are demonstrated; any remaining physical limit requires measured evidence.

Keep strict lazy expert loading and assigned-node queue/rsync workflows
reliable for parallel driver debugging. Missing or corrupt .experts data is
an error, never an eager-load fallback. Give pending driver PRs actionable
feedback naming the common primitives to use and the required tests to pass.

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
Status 2026-08-29: NVFP4 radixark COMPLETE and admitted warm
(433G, DOWNLOAD-RECEIPT pinned, Z.AI commercial grant); official BF16
and FP8 staging on warm (glm-5.3-bf16-304b8051, glm-5.3-935644c0).

## GLM 5.2 FULLY DEPRECATED (operator directive, 2026-08-29)

GLM 5.2's kernel-donor role is superseded by GLM 5.3 Full, which runs
the SAME glm52 module (GlmMoeDsaForCausalLM geometry identical) with
new weights — the module is the 5.3-full module; 5.2 is a pure
repack away from obsolete. The 5.2 TP8 donor packs stay on disk
DEPRECATED-NOT-DELETED pending 5.3-full validation; they are removed
from the support catalog (MODEL_SUPPORT.md) and are not a serving
target. No new work builds against 5.2 sources.

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

## AGENT FLEET POLICY (operator, 2026-08-30)

CAP 5 — with the standing expectation they are ALWAYS productive and
on the critical path. The 15-min cycle's duty 3 enforces this: count,
think, spawn to cap, never make-work. TEMPORARY SPRINTS: when a phase
warrants more (e.g. a change applied to all 8 models at once, plus
the normal working lanes), the coordinator PROPOSES the boost with a
reason and an end condition; the operator approves. Budget reality:
TEAM ACCOUNT live — 2 independent 5h windows now (~2.5h combined
drain), a THIRD when the weekly renews in a few days. Bursts to 8-10
become cheap; the cap stays 5 between bursts.

## STRATEGIC PIVOT: SERVE OURSELVES (operator, 2026-08-30)

The moment glm-5.3 (full) + glm-5.3-flash serve on the sparks, the
DEV FLEET runs ON THEM — agents' own inference (drafting, analysis,
doc summarization) stops burning API credits and starts burning
sparks we already own. Priority ordering implications: the glm5_next
closeout chain and the glm53full 3-res packs are not just scoreboard
items; they are the API-cost-elimination path. The liteLLM front
door is the seam: agents point at it, the fleet serves.
