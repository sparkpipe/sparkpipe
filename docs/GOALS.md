# SparkPipe goals

This file holds the operator directives that agents inherit. It was last
revised on 2026-09-28. The order of work is in [`ROADMAP.md`](ROADMAP.md),
unfinished work in [`../TECHDEBT.md`](../TECHDEBT.md), measurements in
[`../PERFORMANCE_STATUS.md`](../PERFORMANCE_STATUS.md), and the model catalog
in [`MODEL_SUPPORT.md`](MODEL_SUPPORT.md).

## Active GLM 5.3 Flash goal (operator, 2026-09-08, revised 2026-09-28)

Deliver correct, fully functional serving on the 16 Sparks. Users get tokens
per second and latency, so no topology is fixed in advance. Splitting each
model to 1/16 of its size per Spark is what makes frontier models fit. Which
layout serves them best is a measurement: hill-climb TP16, PP16 and TP4 x PP4
on measured tok/s and latency for each batch range, and keep the winner for
each range.

Today GLM 5.3 Flash runs only at TP16. The serving adapter rejects every
other TP degree (`SPARK_GLM5_NEXT_SERVING_TP_DEGREE 16u`,
`spark_glm5_next_serving_adapter.c:37` and `:669`). GLM TP4 x PP4 is ROADMAP
M5.

- **B1 without speculation.** The target is 3 to 3.5 times the best public
  4-Spark B1 result for the same model without speculation, at stated
  precision. For GLM 5.3 Flash FP8 that public result is 23.2 tok/s, so the
  target is 70-81 tok/s at TP16 (arithmetic). The fleet measured 36 tok/s on
  2026-09-28. The roofline floor is 8.0 ms per token, a 125 tok/s ceiling
  ([`GLM5_NEXT_ROOFLINE.md`](GLM5_NEXT_ROOFLINE.md)).
- **Aggregate throughput.** The operator reads the earlier PP results as
  output that kept scaling up to B1024. TP16 has not been measured at that
  batch. Measure PP16 for the most aggregate throughput. An earlier analysis
  put TP4 x PP4 almost level with PP16, but it came before the current
  measurements. If TP16 turns compute-bound at B128, that bound sets the
  largest batch TP16 serves.

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
the existing device/backend interfaces. Eight Mac Studios (M5 Ultra,
256 GB) are on order: CUDA streams, CUDA pointers and NCCL semantics must not
leak into common serving policy. Qualify the common algorithms with the host
backend now and the Metal backend on that hardware when available.

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

Production GLM graph decode runs in a deliberately pinned resident mode. The
serving drop-in sets `G5_PIN_EXPERTS=1`, and the module then leases every
routed expert (`SparkGlm5NextPinAllExperts`).
`SparkGlm5NextGraphClaimExperts` refuses the whole-chain graph unless all of
them are held (commit `78c2c21`). Pinning still acquires experts through the
lazy weightd map, so missing or corrupt `.experts` data is still an error and
never falls back to an eager load (I28).

Pinning does give up bounded residency (I29). That lasts until relocatable
expert graphs land: capture once, record which kernel arguments are expert
pointers, and patch them on load. The messages of `8adebc6`, `360c0ee` and
`1a674be` describe that design, but nothing implements it yet (ROADMAP M6).
Do not turn pinning off to satisfy I29. Without it, chains run through the
state machine instead of the graph path (`GLM5_NEXT_ROOFLINE.md`).

Keep strict lazy expert loading and the assigned-node queue/rsync workflows
reliable for parallel driver debugging. Give pending driver PRs actionable
feedback naming the common primitives to use and the required tests to pass.

## MODEL DIRECTION (operator, 2026-09-28)

- MiMo 2.6 is a target. Contracts `mimo26_flash` and `mimo26_pro` exist. The
  `mimo26` module is so far a 137-line stage-pack header.
- DeepSeek: DSV4.1 Flash outranks DSV4 Pro 0813. DSV4 Pro 0813 is the last
  driver in the order. DSV4.1 Pro will be supported when it is released.
  How much it will differ is unknown, which is why 0813 Pro goes last.
- Kimi K3 moves up into the slot DSV4 Pro 0813 held.
- Ling: Ling 3.0 and its finance fine-tune (contracts `ling` and `lingfin`).
  Ling 2.x is dropped.
- GLM: GLM 5.2 weights are not used for anything (see below).
- Licensing: revenue will not come near $20M, so only the Qwen license is an
  issue. Qwen models stay off the external API service. Internal use is fine,
  and Qwen 3.8 27B is a good internal model.

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
quality gates, infra rules) — each completion gives the internal work
a solid reference and removes a worry. Prioritize completions over
open-ended explorations when choosing next work.

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

## QUALITY GATE: COMPSEC-17 before "usable"

Every first cell is quality-gated: COMPSEC-17 before it is called usable,
and the full ds4_eval 92x suite before it is called "not horrible". For GLM
5.3 Flash the gate is `tools/glm5_next_compsec17.py --thinking off`. It uses
the GLM chat template, temperature 0, at most 512 tokens and one request at a
time. `qualification/ds4_eval/compare_runs.py` grades it by the last
`Answer:` line, and a run passes at 14 of 17. GLM 5.3 Flash at TP16 passed
14/17 on 2026-09-28
([run report](../qualification/ds4_eval/runs/glm5-next-tp16-20260928-dd3526b-thinkoff/REPORT.md)).

The harness before #1256 sent no chat template and graded the first line.
That scored 0/17 on a coherent model, so its earlier results cannot tell a
broken model from a working one.

Run the gate sequentially. Concurrent runs are not batch-invariant yet: two
of the 17 completions change with batch composition.

## SPECULATION: all providers, one contract (operator directive)

Support every speculation type — MTP, DFlash, DSpark, DFlash2, the
coming DSpark2, and whatever follows — because not every model gets
today's best. Design + sequencing: docs/SPECULATION_PROVIDER_DESIGN.md
(provider = capability unit behind the adapter; lifecycle+contract
abstracted, inner loops stay provider-owned for zero hot-path cost;
DSpark2 = a new provider module, not five family edits).

Direction (operator, 2026-09-28): the fleet has more compute than bandwidth
below the batch size where decode turns compute-bound. Speculation can spend
that compute on a tree fed by several drafters, cutting branches short or
extending them conditionally. The condition itself is still open. The
operator notes that TensorFold appears to reach nearly 4x. The aim is a gain
of that order at B8, and if possible at B64, not only at B1. Today glm5_next
has no DFlash2 source, and its MTP drafts skip multi-step decode chains
(TECHDEBT.md, Speculation).

## Medium-term (weeks)

1. All product-set models serving honest perf+quality cells, each
   recorded in PERFORMANCE_STATUS.md (Flash x2, Pro, then Max).
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
per-spark number. PERFORMANCE_STATUS.md reports both axes explicitly.

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

The operator confirmed this on 2026-09-28: GLM 5.2 weights are not
considered for anything. Five of the glm52 firmware descriptions (fp8,
int6, int7, int8, mxfp4) name GLM 5.3 revision `935644c0`. The generators
and packer still pass GLM 5.2 revision `b4734de4`
(`tools/glm52_gen_deployment.py`, `tools/glm53full_gen_deployment.py`,
`tools/glm52_resident_stagepack.py`), and they have to move to GLM 5.3.

## SPECULATOR PORTFOLIO (operator)

A dozen speculators incoming for head-to-head testing. The provider
abstraction (SPECULATION_PROVIDER_DESIGN.md) is the harness for it.
The operator's multi-speculator-live idea is assessed as the
TOURNAMENT PROVIDER: see the design doc's addendum.

## MARKETPLACE TRACK

The commercial frame is [`MARKETPLACE_PLAN.md`](MARKETPLACE_PLAN.md): the
fee, the build hooks and the determinism gates. ROADMAP M9 orders the work.

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

The end state: SparkPipe as the inference
OS for this fleet — every product-set model (MODEL_SUPPORT.md catalog,
open to new ones via the same contract) quality-gated and beating
110% of the best public comparable per recorded cell; multiple models
co-resident; and the stack PROVABLY hardware-independent below the
module boundary: cuda (production), host oracle (CI), Metal (the
controller Mac), ROCm (rented, budget-approved) — one memory model,
one comms model, recipes not hand-built drivers.

## HOW WORK IS RUN (2026-09-28)

Claude is lead dev and the operator merges PRs by hand; PRs may stack.
Drivers are developed in parallel while common code improves, and the
operator sets how many agents run. The coordinator regime of 2026-08-30 no
longer applies: the agent cap, the 15-minute cycle and burst approval.

## SERVE OURSELVES (operator, 2026-08-30)

GLM 5.3 Flash now serves on the Sparks. Agents' own inference (drafting,
analysis, doc summarization) should run on the fleet rather than on paid
APIs. The live GLM API is the `g53-api` user unit on the rtx5090 hub, port
8433. It is an x86 build of the engines' source commit, and every Spark runs
with `G5_API_DISABLED=1` (`tools/fleet_node_agent.sh`, `ensure_api`). The
LiteLLM front door is the seam. [`LITELLM_FRONTEND.md`](LITELLM_FRONTEND.md)
still routes `glm-5.3-flash` to spark0:8433 and has to be repointed. The
pivot also covers GLM 5.3 Full (the glm52 module) once it serves.
