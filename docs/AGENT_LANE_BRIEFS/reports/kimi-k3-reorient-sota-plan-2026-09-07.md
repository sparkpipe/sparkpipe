# kimi-k3 reorientation + the SOTA plan (2026-09-07)

Written at the operator's ask: "a lot of changes have been made, not all
completed. TP16 allreduce is nearly ready, so before resuming hill
climbing, lets get reoriented to where things are and make a plan on how
we will get SOTA perf."

## What changed while this lane was paused (09-03 -> 09-07)

1. **The collective world was replaced** (#816, merged): d2a one-round
   direct all-to-all (78us sync at 8KiB, 57us in-engine median) + ack
   flow control + the nonce-offset root-cause fix (the old 0.46 tok/s
   serving and 1-17ms stalls were an 8MiB write per 8KiB op). glm5_next
   TP16 serving now measures **13.6 tok/s B1 vs the NCCL 12.69 baseline**.
   HONEST VERDICT in that PR: **TP16-at-B1 is marginal** (91
   collectives/token x ~57us = ~5.2ms/token of collective floor);
   **TP16 pays at B8+** (147us/row at B1 -> 18us/row at B16); the
   TP4xPP4-vs-TP16 comparison is the remaining verdict leg.
2. **The stagepack matrix is COMPLETE** (firing 245): every family x
   quant x topology placed MTP-free, receipted, locked, audited clean —
   including **k3.mxfp4 x tp16 + tp4pp4** (canonical packs; the lane's
   own TP16 base-pack work was folded into this program).
3. **Modules are publishing green** (firing 246-247): flash + 27B
   resident_decode_stage modules GPU-validated. K3's module is NOT
   published yet.
4. **The lane's work landed via harvest** (5ccdb8d): the KDA run port
   (multi-row prefill), the S1-S4 run-equivalence gate, the spec-tree
   plan form — all on main. PRs #772/#773 closed as harvested.
5. **Shared-kernel health**: tile-Mloop nvfp4 prefill fixed+validated
   (firing 253), GDN single-token prefill routed through the decode core
   (b475e8a), multi-row module lineage landed (#815), all six drivers
   wired (#788).
6. **Serving is resident on the fleet**: glm53flash.fp8.tp16 residentd
   on the nodes (idle at check time). fleet_serve.sh is the launch path.
7. **Zombies**: spark5's 44-process leak was cleaned (now 2 = the
   original pair); sparkc still holds the 09-01 TERM-immune pair
   (2 processes, ~34.8GB GPU). Operator-only cleanup (KILL
   lane-forbidden).

## Where K3 stands (the five facts)

- **Packs: DONE.** k3.mxfp4.tp16 + tp4pp4 canonical on all 16, audited.
- **Correctness: DONE on main.** The prefill bug fix (KDA run chaining)
  and the certified-FP8 screened head are both in the tree; the S1-S4
  gate proves the kernel contract on CPU.
- **Runtime trees: STALE.** The 16-node k3.mxfp4.tp16 staging is from
  09-02 — it predates the tree engine, d2a, the multi-row lineage, and
  the harvest. Must be re-staged from current main before any wave.
- **Module: UNPUBLISHED.** No K3 resident_decode_stage module in the
  published library (flash + 27B are).
- **First number: NEVER PRODUCED.** B1 decode has never run. This is
  still the operator's unanswered question.

## The SOTA plan (in dependency order)

**Phase 0 — unblock (parallel, mostly CPU):**
- P0.1 Operator clears the sparkc zombie pair (spark5's pair too for a
  clean envelope). Only operator action; everything else is ready.
- P0.2 Re-stage the K3 TP16 runtime trees from CURRENT main: build
  residentd + libk3_serving_adapter.so + hidden transport on a build
  node (the k3_single_spark_step.sh / k3_runner_compile_gate.sh recipe),
  then k3_stage_runtime.sh <host> <build> 16 <configs>. The trees then
  carry tree-engine + d2a + harvested prefill port + certified head.
- P0.3 Module publication (the firing-246 recipe: validator vs the
  placed rank0 pack). Needed for the module-gated serving path.

**Phase 1 — the first number (one wave, ~30 min):**
- P1.1 fleet_serve.sh k3.mxfp4.tp16 full -> ready line -> B1 decode
  receipt (dashboard + nvidia-smi pair). ANSWERS THE OPERATOR.
- P1.2 The two equivalence cells (binaries + run scripts already staged
  on spark8/k3cell; pack tasks k3-*-slice-pack2 need dispatch): prefill
  run-vs-sequential (the prefill fix's on-GPU proof) and head
  certified-vs-full (token-exact).

**Phase 2 — hill climb in the d2a-informed order:**
- P2.1 B1 vs roofline: the ledger target is ~20 tok/s (weights-bound).
  With d2a the collectives are no longer the wall at B1 (glm5_next
  receipt: in-serving op time hides behind layer compute). If B1 lands
  well under roofline, the gap is the K3 per-step overheads, not the
  collective — profile before touching anything.
- P2.2 **B8 decode is where TP16 wins** (per #816's curve). 8 resident
  sequences, aggregate tok/s vs B1. This is the throughput SOTA claim.
- P2.3 TP4xPP4 B1 (packs + trees exist) — the verdict leg #816 names:
  pipelined 18 tok/s estimate vs TP16 measured. Pick the serving
  topology on measurement, not design.
- P2.4 P1 async port (K3 declares no ASYNC_COMPLETION -> 1 adapter
  op/Progress pass; the port plan is filed: kimi-k3-p1-port-plan). This
  removes the host bubble per step — likely the biggest B1 lever after
  the collective fix.
- P2.5 Prefill: width bump 16->1024 (now unblocked end-to-end: run port
  landed, Mloop fixed for nvfp4 tile prefill) then prefill tok/s vs the
  92-1537 tok/s estimate curve.

**SOTA definition for K3 (measurable):** B1 decode >= 20 tok/s (the
weights-bound roofline), B8 aggregate showing the TP16 amortization
curve, prefill >= 500 tok/s at B64-class widths, every number
exactness-gated (the two cells) and receipted per the ledger convention.
