# Wave K3A — K3 perf-provenance + accuracy audit, 2026-09-13

Agent: K3A (mgr2 dispatch; operator challenged the K3_PERF.md performance
numbers). Branch `lane/wave-k3a-audit` off origin/main `d1c3822`.
Scope: read-only on all packs and node artifacts (SP-4R owns mutations);
deliverables are this report, the oracle tool, and corrected arithmetic.

## 1. Verdict summary

1. **Provenance:** only ONE number in docs/K3_PERF.md claims to be measured
   (the 55.5/54.2 ms stage-0 step). Every other number is DERIVED by
   tools/k3_tp4pp4_perf_estimate.py from three constants of varying quality,
   or ASSUMED. The derivation itself carries a material error: the expert
   inventory in the estimate tool is 2.0x LOW against the placed packs
   (details in §5).
2. **Sparka forensics: UNFOUNDED as a reproducible artifact** (raw gate
   output, pack identity, build SHA, and co-residency record all absent).
   The number is *physically coherent* only under the TP4-rank-pack reading
   (see §3): 54.2 ms implies 173 GB/s effective = 70% of the measured triad.
   Verdict class: PLAUSIBLE-BUT-UNPROVEN; it is NOT "measured" in the sense
   I46 requires (no preserved command/output/pack identity).
3. **Accuracy oracle — the decisive test (mgr2 scope refinement):** the two
   questions are SEPARABLE and both are now answered with receipts.
   (a) FORMAT GENERATION: all 16 placed packs carry manifest length exactly
   262128 B (payload base 262144) — the pre-2b27e64 (08-31) sharder reserve
   — so byte-identity to the CURRENT packer fails for all 16, as expected.
   (b) CONTENT FIDELITY: the oracle is header-driven (it reads each pack
   through its own manifest, so the reserve skew cannot bias it) and
   verifies byte-exact against /mnt/model-warm/kimi-k3: the stage2 build
   generation PASSES at ALL FOUR ranks and stage3 at both completed ranks;
   every full-file digest matches the Wave-P ledger. **The placed k3
   content is checkpoint-faithful: content-fine, format-skewed,
   rebuild-on-place — not 4-way build chaos.**
4. **Corrected budget:** worst-stage B1 stream is 9.39 GB per rank (not
   8.04-8.57 GB as the tool prices it). Honest floors: TP4xPP4 26.4 tok/s
   ceiling at the measured 247.5 GB/s triad (23.4 with the 100 us AR
   scenario); TP16 27.8 tok/s ceiling (18.3 with the 100 us scenario).
   18.0 tok/s is therefore *consistent with* the real budget (generous
   reading, not impossible) — but it is not a measurement. 20.2 tok/s TP16
   is derived, not measured; it sits inside the corrected envelope.
   20.6/48.6 ms roofline lands on wrong constants that partially cancel.

## 2. Provenance table — docs/K3_PERF.md numbers

| # | Claim (doc line) | Class | Trace |
|---|---|---|---|
| 1 | Warm step 55.5 ms, stage 0 (24 layers), B1, "real rank pack", sparka | MEASURED-CLAIMED, artifact NONE | Entered repo 2026-08-16 (dfe195a `MEASURED_STAGE_MS = 55.5` + 859dd16 doc text, same session). Gate = tools/k3_single_spark_step.sh -> tests/test_k3_runner_step.cu (print-only). No raw output, pack digest, or build SHA survives (§3). |
| 2 | Graph replay 54.2 ms "(step 3 in the gate)", "bit-identical output" | MEASURED-CLAIMED, artifact NONE | step 3 = pure graph replay confirmed in the era gate source (sparka:~/sparkpipe-k3/tests/test_k3_runner_step.cu, mtime 08-16 21:06). No run log survives. |
| 3 | 2.3 ms/layer | DERIVED | 55.5/24. |
| 4 | ~35 kernels/layer, ~840/step, ~66 us host enqueue each | UNVERIFIED | Asserted outputs of the same lost run; no artifact. |
| 5 | Cold first step ~2.5 s | UNVERIFIED | Same lost run. |
| 6 | Decode 18.0 tok/s | DERIVED | 1000/55.5 (tool `measured_decode_tokens_per_s`). The "measured" label transfers from #1. |
| 7 | "MEMORY-BANDWIDTH-BOUND... each layer streams ~0.8-1 GB" | DERIVED, INCONSISTENT | Verified geometry gives 385 MB avg per layer per rank at B1 top-16 (§5); 0.8-1 GB matches neither the rank slice nor the full stage. Also "~273 GB/s => ~2.3 ms/layer" does not close (0.65 GB at 273 GB/s = 2.4 ms only if 0.65 GB were the layer stream). |
| 8 | Roofline 48.6 ms / 20.6 tok/s; stage streams 8.04-8.57 GB | DERIVED (defective constants) | Tool output reproduced exactly this audit (JSON). Expert term 2.0x low (§5.1); BW = 273x0.65. Corrected streams 8.87-9.39 GB. |
| 9 | Prefill 92 tok/s @ B8 .. 1537 @ B1024; expert saturation B=56; KDA state dominates large B; 0.35 s -> 2.66 s single-prompt | DERIVED (same defects) | Tool `prefill` table. |
| 10 | TP16 20.2 tok/s @ 49.5 ms; "~4x lower latency" | DERIVED + ASSUMED | Tool `tp16` block: assumed 15 us per 16-rank AR, defective expert bytes, convention BW. The doc itself labels the 4x a scaling assumption. |
| 11 | 273 GB/s LPDDR5X | SPEC | docs/archive/PERF_ROADMAP_2026-08-01.md:29 (aa0cb75, 2026-08-14), citing README.md:50-56 and GB10_CUDA_COST_MODEL_CALIBRATION.md. |
| 12 | 0.65 usable-BW fraction | CONVENTION, no measurement artifact | Introduced 59c7fb1 2026-08-12 tools/dsv4_tp4_decode_roofline.py `USABLE_BANDWIDTH_FRACTION = 0.65` (DSV4 roofline), copied into k3_tp4pp4_perf_estimate.py. Note the repo disagrees with itself: tools/k3_param_budget.py and tools/nvme_kv_estimate.py use eta 0.80. |
| 13 | AR latency 8 us (TP4) / 15 us (TP16), 2 ARs/layer | ASSUMED | Tool constants `AR_LATENCY_US`; no measurement artifact in repo. |
| 14 | ACTIVATION_B 88 KB, wire 2x43 KB/layer | ASSUMED | Tool constant. |
| 15 | (brief constant) 247.5 GB/s GPU triad, spark0, sparkcap'd | MEASURED, artifact MISSING | Reported by mgr2 for this wave; NO repo receipt exists yet. T2 item 6 closes this (§6). |
| 16 | Stage-layer mix 18/6, 18/5, 17/6, 16/7 (93 = 69 KDA + 24 MLA, layer 0 dense) | VERIFIED | Checkpoint config (linear_attn_config.kda_layers, first_k_dense_replace=1) + placed pack layer sets; reproduced by the byte model to 0.00-0.22% against placed pack sizes (§5.2). The estimate tool ignores the dense layer 0 entirely. |

Primary artifact refs: `git log -S` anchors — 0.65: 59c7fb1 (2026-08-12);
273 spec: aa0cb75 (2026-08-14); 55.5: dfe195a + 859dd16 (2026-08-16);
"must be RE-SLICED" note: ba9e30b (2026-08-16); w1 k-slice fix: b484605
(2026-08-16 10:22), w2 output-split fix: 5385a63 + 1f8b190 (2026-08-17) —
all BEFORE the four placed builds (2026-08-27/29), so the placed rank packs
should carry the fixed slicing; §4 proves it from content.

## 3. Sparka measurement forensics

What was searched on sparka: /tmp (k3_single_step binary gone; only later
wave dirs remain), ~ (k3repo 08-15, sparkpipe-k3 08-15/16 checkouts,
k3_adapter_gate), ~/.bash_history (empty of k3), journalctl user + system
windows 08-14..09-01 (firmware noise only), ~/.local/share/Trash, all
*.pack older than 08-20 (none outside .git objects), /mnt/model-warm
staging/packbuild (no k3 packs), fleet_agent.log (no 08-15/16 entries),
last/wtmp (sparka up continuously 08-17 01:42 onward; no reboot in the
measurement window).

Established facts:

- The gate is real and its print format matches the doc exactly:
  `step 1: ... ms`, `step 2: ... ms (graph capture+replay)`,
  `step 3: ... ms (graph replay)` — the doc's "(step 3 in the gate)" is the
  literal label of the third submit in tests/test_k3_runner_step.cu.
- The era checkout (sparka:~/sparkpipe-k3, HEAD 50cc1a6e6, dirty tree) is
  the Aug-15/16 gate-debugging session (o_proj race commits match the doc's
  first section). Its docs/K3_PERF.md already contains the 55.5/54.2 text —
  the committed form, no raw numbers beyond it.
- The pack it measured NO LONGER EXISTS in any identifiable form. The
  placed k3.stage*.rank*.pack files are from 08-27..08-29 (Wave-P ledger
  mtimes; node mtimes confirm), i.e. ~2 weeks AFTER the measurement. The
  measured pack has no recorded digest, and no P2 verifier receipts existed
  before 9eb640f (08-27 20:05).
- The gate source header in its current era form says "stage 0, TP 1" — a
  full-width TP1 stage-0 step of 24 layers would stream ~38 GB (top-16 of
  896 experts over a 393 GB stage pack) and CANNOT complete in 55.5 ms on
  one spark (273 GB/s physical ceiling => 141 ms floor). The only physically
  coherent reading of 55.5/54.2 ms is a TP4 RANK pack step: 9.24 GB stream
  / 54.2 ms = 173 GB/s = 70% of the measured 247.5 GB/s triad and 98% of
  the 0.65 convention. That reading is consistent with the doc's "real rank
  pack" wording and with 18 KDA/6 MLA x ~2.3 ms/layer.
- Co-residency on sparka during the run: NO RECORD either way (no queue
  receipts, no process census, no journal evidence). The number therefore
  cannot be certified exclusive.

**Verdict: PLAUSIBLE-BUT-UNPROVEN.** No artifact reproduces it; the
surviving consistency arithmetic (70% of triad) is exactly the kind of
agreement that also results from anchoring the estimate to the convention.
Under I46 the number cannot be cited as measured. The honest budget in §5
supersedes it.

## 4. Accuracy oracle — first checkpoint-content verification of k3 packs

Tool: `tools/k3_checkpoint_oracle.py` (this branch). Offline, no model
execution, no weightd contact. It recomputes byte-exact expectations from
the warm-storage checkpoint (/mnt/model-warm/kimi-k3, 96-shard safetensors,
497,220 tensors) under the pack law and compares a deterministic sample of
every tensor class: direct moves (norms, router, KDA convs/dt_bias/A_log/
o_norm, routed/shared/dense, vocab rows), gamma folds, the fused KDA
q|k|v|beta sections, the MLA q-fold per head (rope rows included), and the
MXFP4 expert payload+E8M0 planes under the interleave grid (w1 k-tile
slice, w2 cell slice, gate=w1/up=w3 order). Mismatches name tensor, byte
offset, expected/got hex. `--digest` anchors the verdict to the full placed
file sha256.

Validation before fleet use: a synthetic mini checkpoint (TP2-splittable
geometry, dense + KDA + MLA + MoE layers) packed with tools/k3_pack.py and
sharded with tools/k3_shard.py verifies PASS on both rank packs and the
stage pack; a single flipped payload byte in a compared tensor FAILS with
the exact tensor/offset/hex and PASSES again after restore (rc=1/rc=0).

Fleet runs: one process per node under
`sudo -n systemd-run --scope -q -p MemoryMax=4096M -p MemoryHigh=2900M
--uid=1000`, then `sudo -n sync; sudo -n sh -c 'echo 3 >
/proc/sys/vm/drop_caches'`, per the heavy-read law. Raw per-node JSON in
/tmp/k3a-tools/result_<node>.json on each node.

| node | pack (build generation) | ledger sha256 (Wave-P 6637980) | --digest this audit | verdict | checks / bytes compared |
|---|---|---|---|---|---|
| spark8 | k3.stage2.rank00.pack (08-29 18:38Z build) | c2ee5a33d89f72f9 | c2ee5a33d89f72f9 | PASS | 1107 / 107.2 MB |
| spark9 | k3.stage2.rank01.pack (same build) | 8ce3ba3b758c83a2 | 8ce3ba3b758c83a2 | PASS | 1107 / 107.2 MB |
| sparka | k3.stage2.rank02.pack (same build) | 25e376c06d37b861 | 25e376c06d37b861 | PASS | 1107 / 107.2 MB |
| sparkb | k3.stage2.rank03.pack (same build) | fe455ac7db5dbd11 | fe455ac7db5dbd11 | PASS | 1107 / 107.2 MB |
| sparkc | k3.stage3.rank00.pack (08-27 21:34Z build) | 23df8aa64e02cbc6 | 23df8aa64e02cbc6 | PASS | 1104 / 108.7 MB |
| sparkf | k3.stage3.rank03.pack (same build) | 02882b942cfc9321 | 02882b942cfc9321 | PASS | 1104 / 108.7 MB |
| sparkd, sparke (stage3 r1, r2) | same build | d9deda6a8e6d70d4 / b1727beca1789860 | (in flight) | IN FLIGHT | D-state on the ceph mount at commit time |
| spark0 (stage0 rank00, 08-29 14:56Z build) | bef86d7ea918619f | (in flight) | IN FLIGHT | D-state |
| spark4 (stage1 rank00, 08-29 13:09Z build) | 2fd46f282f006205 | (in flight) | IN FLIGHT | D-state |

Every completed digest re-anchored in this audit matches Wave-P's ledger
digest exactly — the bytes verified are the bytes Wave-P recorded. The
stage2 build generation is checkpoint-faithful at ALL FOUR ranks; stage3 is
faithful at both completed ranks. Remaining ranks were still executing at
commit time (three concurrent verification waves share the ceph warm-store
mount; the processes were in uninterruptible ceph reads with IO progressing,
not dead); their JSON receipts land in /tmp/k3a-tools/result_<node>.json.
The wave that inherits this lane runs the same command to finish the
in-flight rows and the six untested nodes (spark1-3 stage0 r1-3, spark5-7
stage1 r1-3); nothing else is pending.

### 4.1 Content fidelity vs format generation — two separable verdicts

mgr2 scope refinement (SP-4R ground truth): the 16 packs have no replicas
(16 distinct stage x rank shards — confirmed in this table's file map), and
ALL placed packs predate k3_shard 2b27e64 (2026-08-31), which raised the
rank manifest reserve 262128 -> 1048560 and thereby shifts every payload
offset. This audit anchored that per pack: the manifest length field is
EXACTLY 262128 on all 16 placed packs (payload base 262144 on every node).
So:

- **Byte-identity to the CURRENT packer: FAIL 16/16** — by format
  generation, not by content. A re-slice with the current sharder produces
  identical per-tensor bytes at different offsets (different file sha256).
- **Content fidelity to the checkpoint: PASS on every pack verified** —
  the oracle reads each pack through its OWN manifest, so the offset skew
  cannot bias the comparison. The failed expectation is pack==packer; the
  pack==checkpoint half of the law HOLDS for the placed generation.

Risk statement for the lane: k3 flips from "4-way build chaos" to
"**content-fine, format-skewed, rebuild-on-place**". The placed packs are
content-trustworthy for serving; SP-4R's fresh rebuilds are needed for
format uniformity (and fresh receipts), not to fix content.

Per-class coverage on a rank pack (e.g. spark8): 81 norm, 46 gamma-fold,
261 expert_w1 + 261 expert_w2 cell samples (payload + scale lanes),
68 fused KDA section samples, 51 conv + 51 f32 scalar tensors, 68 KDA + 58
MLA projection row/col samples, 29-30 MLA q-fold heads, 46 router, 46
shared, 46 routed checks — ~107 MB compared per pack against the
checkpoint, plus the full-file sha256.

**Canonical answer for SP-4R:** the **stage2 build generation
(2026-08-29 18:3xZ, packs k3.stage2.rank00-03, digests c2ee5a33/8ce3ba3b/
25e376c0/fe455ac7) is CHECKPOINT-FAITHFUL at all four ranks** — the first
content-verified k3 placement. stage3 is faithful at both completed ranks.
No content defect was found in any verified pack. Framing correction for
the repair wave: the 16 placed packs are 16 DISTINCT stage x rank shards by
deployment contract (PP stage = rank/4, TP rank = rank%4) — s0/s4/s8/sC
hold stage0/1/2/3 rank00, NOT replicas of one another, so distinct digests
across quartiles are the CONTRACT, not a violation. The lawful repair is
therefore not byte-unification of "quartile replicas": it is (1)
content-proving every build against the checkpoint (this oracle), (2)
re-anchoring receipts to the surviving digests, and (3) if any build FAILED,
re-packing that stage from the checkpoint with the current packer. No pack
needs re-placement on grounds of content.

## 5. Step-time budget rebuild (measured constants only)

Constants: 247.5 GB/s measured GPU triad (spark0, sparkcap'd — receipt
owed, §6 item 6); 4.7-5.1 GB/s measured NVMe cold; AR treated as labeled
SCENARIOS (8 us = the tool's assumed NCCL 4-rank; 15 us = its 16-rank
assumption; 100 us = the mandated pessimistic scenario). Nothing else.

### 5.1 Two defects in the existing derivation

1. **Expert inventory 2.0x low.** The tool's "ground truth inventory"
   prices w1 as [896 experts x 1536 out] and w2 as [896 x 3584 x 768] — it
   halved moe_intermediate_size (3072, verified in the checkpoint config
   and in the placed manifests). The true per-rank per-layer expert set is
   3.931 GB (w1 2.620 + w2 1.311), so the B1 top-16 stream is 70.2 MB per
   layer, not 35.1.
2. **The dense layer 0 is ignored** (all 93 layers priced as MoE;
   checkpoint first_k_dense_replace=1 makes layer 0 dense).

### 5.2 Byte model validated against the placed packs

Per-rank pack bytes recomputed from geometry (TP4 slices of the checkpoint
config) vs placed sizes: stage0 98.55 vs 98.77 GB (-0.22%, the dense
layer's true intermediate is slightly larger than assumed), stage1 97.65 vs
97.65 (+0.00%), stage2 97.59 vs 97.59 (+0.00%), stage3 98.12 vs 98.12
(+0.00%). The model is exact to the rounding of one layer's dense
intermediate.

### 5.3 Corrected B1 decode streams and floors

| stage | layers (kda/mla) | B1 stream per rank | at 247.5 GB/s | at 177.45 (0.65 conv) |
|---|---|---|---|---|
| 0 (0-23) | 18/6 | 9.24 GB | 37.3 ms | 52.1 ms |
| 1 (24-46) | 18/5 | 8.93 GB | 36.1 ms | 50.3 ms |
| 2 (47-69) | 17/6 | 8.87 GB | 35.8 ms | 50.0 ms |
| 3 (70-92) | 16/7 + lm_head | 9.39 GB | 37.9 ms | 52.9 ms |

TP4xPP4 pipelined (worst stage governs): **37.9 ms -> 26.4 tok/s ceiling**
at the measured triad; +2 AR/layer at 8 us = 38.3 ms (26.1 tok/s); at the
100 us SCENARIO = 42.7 ms (23.4 tok/s). At the 0.65 convention: 52.9 ms
(18.9 tok/s).

TP16 (PP1): 8.90 GB/token/rank (93 layers at ~97/85 MB + head slice) —
**35.9 ms -> 27.8 tok/s ceiling**; +2 AR/layer at 15 us = 38.7 ms (25.8);
at 100 us = 54.5 ms (18.3). At the convention: 50.1 ms (19.9).

Cold-end effects: per-rank pack 97.6-98.8 GB at 4.7-5.1 GB/s = 19-21 s
cold load (amortised once the page cache is warm; the KDA memory envelope
caps how much of it stays resident).

### 5.4 What the 54.2 ms stage implies

54.2 ms on the corrected 9.24 GB stage-0 stream = 173 GB/s effective =
70% of the measured triad and 98% of the 0.65 convention; 2.26 ms/layer
average. Two readings: (a) it was a real rank-pack step running at 70% of
triad — a plausible GEMM weight-streaming efficiency, which would put the
real fleet ceiling near 26 tok/s once the pipeline is filled; or (b) the
anchor was absorbed from the convention it "anchors" (98% agreement is not
independent evidence). The artifacts cannot distinguish (a) from (b) —
that is exactly why T2 receipts are mandatory.

### 5.5 Verdicts on the published numbers

- **18.0 tok/s (TP4xPP4)**: NOT measured (§3). As a magnitude it is
  CONSISTENT WITH and GENEROUS TO nothing — it sits below the corrected
  ceiling (23.4-26.4 tok/s), i.e. the real budget makes it look
  conservative, but as evidence it is unfounded and must be re-labeled.
- **20.2 tok/s (TP16)**: DERIVED/ASSUMED, not measured; inside the
  corrected envelope (18.3-27.8) but resting on the 2x-low expert bytes,
  an assumed 15 us AR, and the convention BW.
- **48.6 ms / 20.6 tok/s roofline**: DERIVED from wrong constants that
  partially cancel (expert bytes 2x low push down ~9%, convention BW 28%
  low pushes up); corrected roofline at the same convention = 52.9 ms.
- **Prefill family (92..1537 tok/s)**: DERIVED, same defects; do not cite
  until re-derived (the top-16 saturation batch B=56 is geometry, not
  measurement, and survives).

## 6. T2 plan — minimal receipt set for real K3 numbers

1. **Build identity (I33/I46).** One merged-main SHA; per-node binary
   sha256 (residentd, libk3_serving_adapter.so, hidden_transport, NCCL
   build); per-node pack sha256 = the canonical digests from §4; deployment
   JSON digest. All in the PR description before any timing is cited.
2. **Exclusive window census (I31/I32).** Before and after each timed run:
   process/cgroup census + device memory census on all 16 nodes, queue
   reservation receipts (spark_queue), K3's ~96-97 GiB envelope leaves no
   room for co-residents.
3. **Arms.**
   a. Single-rank stage-0 gate rebuilt on merged main (the gate test was
      deleted from main at a29ea53 — restore or supersede): B1 warm steps,
      graph-capture + pure-replay step, direct-vs-replay fidelity — the
      exact arm that produced the 55.5/54.2 anchor, this time with a
      recorded pack digest and stdout log.
   b. TP4xPP4 fleet: 16-rank B1 warm decode on the persistent engine,
      >=100 steps, all four stages filled (I38: a TP16 or single-rank pass
      does not qualify PP boundaries).
   c. TP16 arm once the TILE_K=32 INTERLEAVED_B GEMM lands.
   d. Prefill at B8/B56/B128/B1024 for the corrected prefill curve.
4. **GPU-time capture (I41/I45).** cudaEvent boundaries around whole steps
   AND per-phase: load, prefill, TTFT, warm decode, batch tails reported
   separately; a 100-step steady-state trace (nsys or event pairs) giving
   per-layer wall time vs summed kernel time; host enqueue vs device gap
   per step. HTTP wall time is not decode time.
5. **Numerics in the same runs (I39/I40/I27).** Reference comparison vs
   the checkpoint-faithful path per topology + COMPSEC-17 through /v1
   before any number leaves the lane.
6. **Constant receipts.** Persist the 247.5 GB/s triad command + log in
   docs (it currently exists nowhere); add a GEMM weight-streaming
   microbench (persistent GEMMs reading the actual pack layout) — the
   decode path's real ceiling, which triad only upper-bounds; measure AR
   latency at 14 KB/43 KB payloads for 4-rank and 16-rank to replace the
   8/15 us assumptions; keep the 100 us figure as the labeled pessimistic
   scenario.
7. **Raw receipts.** Per-run JSON (config, digests, per-step timings,
   census outputs) committed alongside the lane report; failures and
   aborts recorded verbatim (fail loud).

## 7. Blockers and handoff

- At final commit, four oracle runs (spark0, spark4 — stage0/stage1 rank0;
  sparkd, sparke — stage3 rank1/2) were still in uninterruptible ceph reads
  with IO progressing (spark0: 192 MB read and climbing). Root cause: three
  concurrent waves share the ceph warm-store mount — SP-4R's k3 stage
  rebuild scopes (sp4r_k3_stage.sh -> tools/k3_pack.py on
  /mnt/model-warm/kimi-k3, confirmed live on sparkd) and the acc1 glm53full
  verify both read the same store. The runs were left alive on purpose;
  their verdicts land in /tmp/k3a-tools/result_<node>.json on each node.
  The same one-line command finishes the six untested nodes (spark1-3 =
  stage0 ranks 1-3, spark5-7 = stage1 ranks 1-3).
- The 247.5 GB/s triad has no repo artifact yet (T2 item 6).
- tests/test_k3_runner_step.cu no longer exists on main (a29ea53 stripped
  it); the T2 single-rank gate arm needs it restored or superseded.
- SP-4R handoff: §4/§4.1 are the canonicalization input. The placed
  generation is CONTENT-FAITHFUL where verified (stage2 4/4, stage3 2/2)
  and format-skewed 16/16 (manifest 262128 / base 262144 receipts taken).
  Nothing verified needs re-placement for content; the rebuilds are for
  format uniformity and fresh receipts. If any in-flight row returns
  MISMATCH, that stage alone gets re-packed from /mnt/model-warm/kimi-k3
  with current tools/k3_pack.py before placement.

Run identity: audit clone /Users/mac/k3a @ lane/wave-k3a-audit; oracle
sha256 recorded in the branch commit; node-side oracle copies at
/tmp/k3a-tools/k3_checkpoint_oracle.py on spark0-9,a-f; results at
/tmp/k3a-tools/result_<node>.json.
