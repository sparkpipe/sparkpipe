# Merged improvement backlog (all 12 agent assessments)

Source: the twelve TOP10_*.md agent assessments (landed in the agent
workspaces; this file is the coordinator merge). Sort key: Solutions /
(code size * 2) with the value ladder from .agents/METRIC.md. DRY wins
first (negative code), then level buys ranked by value-per-line.

## Human decisions needed (blockers - nothing moves until these are made)

1. **Qwen 3.8 27B checkpoint pin** - the exact checkpoint id+revision is not
   in the tree; the whole qwen38-27b lane (contract, geometry, perf targets)
   re-bases from it. Owner: you + the qwen session. (Directive PR #664.)
2. **GLM52 DSpark draft weights** - the ~3.2K-line speculation path buys 0
   tok/s until a draft model is trained/published ("draft weights must be
   trained first"). Owner: you (training/hosting decision).
3. **The 8 dead nodes** - wait-for-fsck is exhausted (14h+). The remaining
   lever is a smart-plug power-cycle on ONE host (spark8) as a lottery
   ticket, plus the permanent fastboot fix on first login. Owner: you
   (trigger the plug) + sysadmin agent (landing). See
   docs/INCIDENT_RECOVERY_PLAYBOOK.md.

## Top picks (highest value-per-line, cross-agent consensus)

| Rank | Item | Kind | Delta | Owner |
| --- | --- | --- | --- | --- |
| 1 | Measure DSV4 Flash DSpark end-to-end (wired, never measured) | level 4->5 | ~0 | dsv4-flash + cuda-kernels cards |
| 2 | Land admission core (in flight: admission-core task agent) | DRY | -400 | admission-core -> coordinator |
| 3 | Land SparkKvModelTable + SparkKvBackendInitialize (signed) | DRY | ~0, unblocks 4 models | kv-cache -> coordinator |
| 4 | K3 onto the neutral DSpark backend (block-7/GQA-16 table exists) | DRY + level | -1000 | k3 + coordinator |
| 5 | Pro 0813 drafter adoption (fill the 4 zero fields first) | DRY + level | ~0 | dsv4-pro |
| 6 | DSV4 tree adoption (spec landed; 7-candidate/8-row) | DRY | +140 | dsv4 sessions |
| 7 | K3 dead-code/bitrotted-test fix (half-removed decay|gate fusion) | DRY | -30 | k3 + coordinator |
| 8 | Sysadmin fastboot fix on every host as it returns | infra | 0 | sysadmin |
| 9 | qwen38-max activation chain (admit stub, 4x KV, GDN record, TP collective) | level 1->2 | +350 | qwen38-max |
| 10 | DSV4 Pro GA DSpark native pass (the one big level buy) | level 1->4/5 | +900-1400 | dsv4-pro |

## The DRY block (do in this order; all negative code)

1. glm52 dead/dormant deletions: glm52_resident_pack_common.py (-99),
   glm52_stagepack.py migration onto spark_pack_common.py (last packer).
2. KV demotions: cache/cache.h -> spec (-600), EstimateCapacity single
   authority (-100), DSv4 cache_plan vs paged_cache reconcile (-300).
3. Scheduler DRY set: reject-map fix, ServingAdmit collapse, module-admit
   tables, B1-B1024 ladder extraction, generated-merge promote (-204).
4. cuda-kernels DRY: fold GLM52 dspark norm/swiglu/add onto shared
   primitives, delete dead LmGatherRowsKernel, one shared block-reduction.
5. qwen38-27b identity fix: four conflicting adapter ids + registry
   pp16/TP4 mismatch (after the checkpoint pin).

## The level-buy block (ranked by value per line)

- NAMESPACE SWEEP (post-DFlash2-adoption, mechanical, ~0 lines): rename the
  qwen36 namespace to qwen38 across the tree (module dir, headers, tools,
  gate entries, validation scripts, agent clones). The qwen36 name is the
  Qwen3.6-family lineage; the served model is Qwen3.8-27B (pinned in
  model_contracts/qwen38_27b_authoritative.json) which shares the same
  geometry. Deferred until the DFlash2 adoption lands so in-flight patches
  and agent clones are not invalidated mid-sequence.

- Enable-and-measure speculation where it exists: DSV4 Flash (done-wired),
  then K3/Pro drafters.
- GLM52: global confidence-scheduled verifier (+200-400), B1 reduce-path
  squeeze, B1 GEMV expert kernel (+150), wire mxfp4/int codecs (+50),
  CUDA-graph decode step (+100).
- qwen38-max: head-parallel attention activation (+250-350), prefill
  kernels (+260-360), B=1 tensor-core expert path (+150-250), serve MTP
  (+500).
- qwen38-27b: lossless weight codec (90->match SOTA), sharded-delta B64
  collective (80->90), B1 fusions (92->98 ceiling).
- K3: BF16 KDA state, TP16 TILE_K=64 repack, reduce-scatter AR, 16-rank
  measurement + PERFORMANCE_STATUS promotion.
- DSV4 Pro: FP8 KV + FP8 expert kernels (codec selectability), weight
  read-ahead, 2-token chains.
- DSV4 Flash: predeclared collective program (-150-300 + the no-spec
  ceiling unlock), greedy->sampled verify (+60-80), TP4xPP4 stage-3 draft.
- ENGINE (coordinator-owned, deferred to post-merge): b1 first-decode wedge
  at ctx=129. Signature (fork driver dd75a1cb, TP4 BUCKET=1, uniform tps):
  all ranks end at lease_advance ctx=129 accepted=1 tps=1 completed=1
  next=129, prefill completes, first decode submission never completes,
  no error lines, client wedged in dispatch loop. The fork's runtime diff
  (node/runtime files, incl. POLLOUT gating + continuation-lease emitted
  count + 1cfea8f anchor-double-count fix) must land first so the stall
  reproduces on the merged tree; the kernel fix verifies via the in-binary
  1-row reference and does NOT wait on this.
- KV: Qwen38 TP 1/N head-sharding (+300), GLM52 JIT-KV onto common core
  (+400), NVMe tier into common path (+200), Mooncake rank-sharded backend
  (+150).

## Standing rules

- A proposal must name its level (or DRY) and its code-size delta.
- Shared-code changes land through the coordinator; model changes through
  the model agents; kernel work through contract cards (cuda-kernels).
- Re-rank this backlog whenever a new measurement lands (the ladder
  re-sorts automatically).
