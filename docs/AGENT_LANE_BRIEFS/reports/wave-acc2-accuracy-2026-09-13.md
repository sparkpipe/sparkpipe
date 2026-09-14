# Wave ACC-2 — reference-anchored accuracy for ling, gemma4, laguna, muse, minimax-h3

Date: 2026-09-13 (runs executed 2026-09-13/14 UTC+9). Lane agent: ACC-2.
Branch: `lane/wave-acc2-accuracy`. Clone: /Users/mac/acc2 (origin/main d1c3822 + lane commits).
Identity: every GitHub command via `tools/sparkpipe_github_pat.sh`; `gh api user` = `sparkpipe`.

Mission: enforce the EXISTING pack-verification law (pack == packer,
packer == checkpoint) with a reference-anchored oracle, inventory every
accuracy/perf claim in the five lanes as MEASURED / DERIVED / ASSUMED,
and state exactly what is missing for a T1 accuracy receipt per lane.

## 1. Method — the parity oracle (the deliverable)

Tool: `tools/acc_parity_oracle.py` (this branch). Per pack it runs three legs:

1. RECEIPT / PLACED BYTES — recomputed sha256 of the pack file vs the
   `.sha256` sidecar and the packer's `.receipt.json` (pack == packer).
2. STRUCTURE — the pack directory is parsed and EVERY entry must equal
   the independently rebuilt packer plan at the same position (kind,
   layer, rows, columns, payload_bytes). Any drift fails loud.
3. CONTENT — payload bytes of sampled entries are re-derived from the
   WARM-STORAGE CHECKPOINT through the packer's own plan code and
   compared byte-for-byte against the pack file (packer == checkpoint).
   Anchor sampling: layer 0, a middle layer, the last layer of the pack
   window, every global (embedding / final norm / lm head), rope
   tables, router and per-expert scale planes, and the anchor layers'
   expert planes. `--full` compares every entry.

Honest boundary: leg 3 re-uses the packer's plan math, so it proves the
emit chain is reproducible from the checkpoint and that no byte drifted
in placement; it does NOT independently re-derive the transforms. That
upper tier stays with the family numeric oracles (gemma4 anchors,
laguna layer7 reference, ling validator suites). Reads are
ceph-stall-tolerant (short-read retry, fail loud); all heavy runs under
`sparkcap` (MemoryMax=4096M / MemoryHigh=2900M).

Synthetic proof (committed test, runs offline):
`python3 tests/test_acc_parity_oracle.py` builds a shrunken fake muse
checkpoint, emits a real pack through muse_glimmer_stagepack's own
writer, and requires CHECKPOINT-FAITHFUL on the intact pack and DRIFTED
with a located first-byte delta after a one-byte payload corruption.
PASS (both directions).

Node runs: bundles of the committed branch (`70942f5`, `f40d02f`,
`690164f`) cloned on spark4 + sparka + spark5 (git bundle transport; no
credentials off-workstation). Packs verified, by lane:

| lane | pack | node | coverage |
|---|---|---|---|
| gemma4 | gemma4_31b_tp16_rank15_l0-7.gemma4sp | sparka | FULL 98/98 |
| gemma4 | gemma4_26b_tp4_rank0_stage2.gemma4sp | sparka | FULL 134/134 |
| gemma4 | gemma4_26b_tp4_rank3_stage2.gemma4sp | sparka | anchors 66/134 |
| ling | ling.bf16.tp16.rank5.sp (placed ~/sparkdata, spark5) | spark5 | anchors 126 |
| muse | muse_tp16_rank04.bf16.gsmu (placed) | spark4 | anchors 27 |
| laguna | laguna_stage.tp8.pp2.stage0.rank0.lgsp | spark5 | anchors 78 |
| laguna | laguna_stage.tp8.pp2.stage1.rank0.lgsp | spark5 | anchors 83 |
| minimax | h3.bf16.tp16.rank00..03.sp (warm staging) | spark4 | lane boundary checker re-run, 9 tensors/rank |

(Results in section 3.)

## 2. Claim inventory — grades

Scale: MEASURED = artifact exists (file / node / date / digest chain);
DERIVED = computed from stated inputs + assumptions; ASSUMED = no
artifact. The K3_PERF.md provenance trap (derived numbers presented as
measured) was the audit lens; finding: NONE of the five lanes commits
that trap today — every throughput artifact in the tree is explicitly
labeled "analytical estimate (measured bandwidth input)".

Fleet-wide negatives (verified, not assumed):

- docs/PERFORMANCE_LEDGER.md scoreboard and PERFORMANCE_STATUS.md
  "Measured model performance" contain NO rows for ling, gemma4,
  laguna, muse, or minimax-h3. There is no measured tok/s anywhere for
  these lanes. No ds4_eval run exists for any of them
  (qualification/ds4_eval/runs/ holds only deepseek-v4-flash and
  kimi-k3).
- What DOES exist for all lanes is determinism/self-consistency
  evidence (bit-exact determinism, two-pass placement proofs, census
  closure) — exactly the gap the operator called out.

### 2.1 ling (ling-3.0-flash / -fin)

| claim | source | grade |
|---|---|---|
| bf16 module compile receipts (LING_COMPILE_RECEIPT_PASS) | PROGRESS.md, job ids | MEASURED |
| real pack chain ranks 0..15, 730 tensors/rank, verify two-pass green | PROGRESS.md ling-r16c chain | MEASURED (receipts + verify runs) |
| validator suites tier1/tier2a/tier3 PASS, bit-exact determinism (r21 close-out, spark9) | branch lane/ling-driver-r18 ff713ce | MEASURED — kernel-tier numerical gates vs in-house fp64 oracle; NOT reference decode |
| 2 exposed dormant tier2a validator gates FAIL (router weight_rel, boundary stream) | same receipt, honestly reported | MEASURED (open, no ruled number) |
| decode roofline estimator (PR #982) | tools/ling_decode_roofline.py | DERIVED (analytical, measured-BW input) |
| packs placed 16/16 (1/node) | wave-p sweep ledger 6637980 + re-placement run 2026-09-13 (sidecars + receipts match) | MEASURED |
| contract/revision pin (source_revision e0dfe7cd...) | receipt rank4/rank5 | MEASURED |
| decode tok/s, quality (COMPSEC/92x) | — | DOES NOT EXIST |

### 2.2 gemma4 (31B + 26B-A4B) — the "121 oracles"

Located. Commit 944754c "gemma4: family oracle ... consumes the anchor
fixtures — ALL CHECKS PASS": validation/spark_gemma4_reference.c, plain
C11, 121 check sites PER MODEL over 270 exported fixture arrays
(31B + 26B).

| claim | source | grade |
|---|---|---|
| 121-site oracle ALL PASS both models (rope inv_freq BITWISE: 64 nonzero 1e6**(-i/256) + 192 zeros; sliding-window all-128 1e4**(-i/128); window-1024 leak direction exact; embed scales bf16(sqrt(5376))=73.5 / bf16(sqrt(2816))=53.0; probs rounded bf16 before p@v, worst rel 0.0000; KV stores bitwise; MoE top-8 lowest-index ties + uniform 1/128 zero-residual) | commit 944754c, receipt GEMMA4-AC7-SPARKA-ALL-GREEN (sparka CPU-class) | MEASURED — but transform/kernel-tier (weight-free transforms + anchor fixtures), NOT end-to-end reference decode |
| weight-carrying projections validated by the Python anchor kit | PROGRESS AC5/AC8 | MEASURED — but the anchor kit's npz fixtures are NOT COMMITTED (only anchors_export.py + the C oracle are in the tree; the 271 exported .bin + manifest exist only at sparka:~/gemma4_anchors_bin). Reproducibility gap, not a fabrication signal |
| contract freeze: HF revisions pinned, shard sha256 = HF LFS oid, warm byte-size equality | commit e148184 | MEASURED |
| warm-payload falsification: layer_scalar is learned, not all-ones (26B l0/16/29 = 0.0703125/0.5546875/0.1953125) | AC8 PROGRESS; receipts carry per-layer values | MEASURED (the fail-closed check fired exactly as designed; warm wins per the never-quantize law) |
| AC6 CUDA validation tier PASS (dense h5376.l60.v1 sites=23; MoE arm) | retained receipt | MEASURED (kernel tier) |
| decode roofline estimator (PR #983) | tools/gemma4_decode_roofline.py | DERIVED |
| decode tok/s, quality run | — | DOES NOT EXIST |

Stale-doc finding: tools/gemma4_stagepack.py docstring still says
"layer_scalar asserted exactly 1.0 ... fail closed" — superseded by the
learned LAYER_SCALAR kind the same file implements. Doc lag, not code
lag.

### 2.3 laguna (Laguna-S-2.1)

| claim | source | grade |
|---|---|---|
| contract freeze + census lock 23 patterns / 36769 tensors | model_contracts/laguna_authoritative.json, tensor_patterns.json | MEASURED (enforced fail-closed at pack time) |
| real packs 16/16 (tp8 x pp2 stage0/stage1) exit 0, per-rank receipts with sha_feed | /mnt/model-warm/packbuild/laguna/real_pack.log 2026-09-11 | MEASURED |
| laguna_layer_reference.py / layer7 realpack reference (independent numpy fp32 oracle; yarn table anchor 1e-6; NeoX rope, rms-norm, softplus gate, sigmoid router + correction, 2.5 scaling) | tools/ + validation/ | MEASURED (kernel/layer tier, real pack) |
| offline gates run 2026-09-10 | packbuild/laguna/gates exit_code=2 | MEASURED FAILURE — red on the shared Makefile (k3 serving adapter build), not laguna code; gate debt open |
| decode roofline (branch wave-r3-roofline-laguna) | tools/laguna_decode_roofline.py | DERIVED |
| decode tok/s, quality run | — | DOES NOT EXIST |

### 2.4 muse (Muse-Glimmer-30B)

| claim | source | grade |
|---|---|---|
| census resolution 627 text tensors (12 classes x 52 + 3 globals; vision tower excluded) recorded with index sha | muse rank receipts | MEASURED |
| packs tp16 ranks placed (1/node x16 re-verified in wave-p sweep) | sweep 6637980; placed rank04/05 on spark4/spark5 with receipts | MEASURED |
| kv-head replication law (ranks 0-7 kv head 0, 8-15 kv head 1, bitwise-identical sharers) | packer DESIGN + receipt kv_head_of_rank | MEASURED (construction + receipts); cross-shard bitwise check is a by-construction claim, not re-run by the lane |
| decode roofline (branch wave-r3-roofline-muse) | tools/muse_decode_roofline.py, CLASSIFICATION = "analytical estimate (measured bandwidth input)" | DERIVED (honestly labeled) |
| dense model: active-expert bytes = 0 | same | DERIVED (trivially true) |
| decode tok/s, quality run | — | DOES NOT EXIST |

### 2.5 minimax-h3

Main-tree finding (dispatch question 1): CONFIRMED — main d1c3822 has
NO minimax driver/contract/module. It exists only as
docs/MODEL_SUPPORT.md + TECHDEBT.md rows ("Add exact checkpoint-derived
contracts ... for MiniMax H3"). The dispatch-mentioned
`lane/minimax-driver-rebase6` DOES NOT EXIST on origin; origin holds
exactly one lane branch, `lane/minimax-driver` @ 9860f28, whose r6
rebase record says the rebase lineage landed on that same branch
(e4af723 = old bc45d75 rebased onto f6db50a). Nothing to merge was
found beyond that branch; nothing was merged by this wave.

| claim | source | grade |
|---|---|---|
| pinned-source contract (diffusers 3c221246), no-CFG, packing, rope, adaLN, mixed F32/BF16 | branch PROGRESS | MEASURED as documentation of pinned source (line-cited) |
| pass-A placement proof 16/16 dry censuses (encoder 705 / dit 638 / video_vae 585 / audio_vae 914) | minimax-r7-passA | MEASURED |
| TP16 packs ranks 00-03 emitted byte-exact; two emit-time driver bugs caught (match_name tag loss; read_tensor_blob column slicing) with routing unit added | commits 0e9afdd, f96bc23; receipts minimax-r7-pack00..03 | MEASURED |
| boundary check GREEN ranks 00-03: extents + 9 representative tensors/rank sha-equal to fresh warm slices through the same plan math | 2026-09-12 (post-halt retry, packs + sidecars on ceph) | MEASURED (sampled, per rank) |
| ranks 04-15 | — | MISSING — 6th sparke host event (ceph MDS metadata plane, wchan ceph_mdsc_wait_request) after every rank04 restart; chain halted honestly; rank04 partial (710KB stub + progress v2) on ceph |
| rank07+rank08 concat proof, cell bring-up, ABI seam measurements | designed, not run | DOES NOT EXIST |
| decode/quality | — | N/A shape: media (t2va) model — T1 must be reframed to denoise-step parity + artifact acceptance |

## 3. Oracle results (this wave's runs)

Every sampled verdict below is CHECKPOINT-FAITHFUL — zero drifted bytes
against the warm checkpoints. Raw JSON on the nodes:
sparka /tmp/acc2_gemma4_*.json; spark4 /tmp/acc2_muse_r4.json,
/tmp/acc2_minimax_bc.log; spark5 /tmp/acc2_ling_r5.json,
/tmp/acc2_laguna_s{0,1}_r0.json.

| lane | pack | node | entries | compared | verdict | receipt/sidecar leg |
|---|---|---|---:|---:|---|---|
| gemma4 | 31b tp16 rank15 layers 0-7 | sparka | 98 | 98 (FULL) | CHECKPOINT-FAITHFUL | receipt file_bytes match; no pack digest field (now reported as no-digest-field) |
| gemma4 | 26b tp4 rank0 stage2 (l16-22) | sparka | 134 | 134 (FULL) | CHECKPOINT-FAITHFUL | receipt file_bytes match |
| gemma4 | 26b tp4 rank3 stage2 | sparka | 134 | 66 (anchors) | CHECKPOINT-FAITHFUL | receipt file_bytes match |
| ling | bf16 tp16 rank5 (placed) | spark5 | 730 | 126 (anchors) | CHECKPOINT-FAITHFUL | sidecar match; computed sha == receipts/rank5.json sha256 (77d529d2...) |
| muse | tp16 rank04 (placed) | spark4 | 419 | 27 (anchors) | CHECKPOINT-FAITHFUL | receipt pack_sha256 == computed (55ca88d5...) |
| laguna | tp8.pp2 stage0 rank0 | spark5 | 309 | 78 (anchors) | CHECKPOINT-FAITHFUL | no sidecar/receipt files; system sha256sum == oracle digest (151ced0f...) |
| laguna | tp8.pp2 stage1 rank0 | spark5 | 314 | 83 (anchors) | CHECKPOINT-FAITHFUL | system sha256sum == oracle digest (3eb373d7...) |
| minimax | h3.bf16.tp16 rank00..03 | spark4 | per rank | 9 tensors/rank + extents | BOUNDARY CHECK GREEN (lane checker re-run by this wave) | per-rank .sha256 sidecars on ceph |

Anchor coverage actually exercised: embedding and lm_head vocab shards
(boundary ranks), final norms, per-layer input/post/pre norms, rope
tables (gemma4 KIND_ROPE_TABLE f32 re-derived bitwise from
1e6**(-i/256) with 64 nonzero + 192 zeros), gemma4 layer_scalars
(learned values), gemma4 router proj with the scale x hidden**-0.5
fold and per_expert_scale folded into expert down rows (bf16 RNE
re-round byte-exact), sliding k|v fused planes and full-kv replication
head math (gemma4 sliding/full window geometry), ling MLA kv_b
key-transpose and value planes, ling KDA fused q|k|v|beta row
sections, conv planes, router + f32 correction bias, MoE expert planes
(ling 512 experts inter-shard rows; laguna 256 experts gate-first W1;
gemma4 128 experts rank slices), muse fused qgkv head-major interleave
with kv-head replication, gate/up fusions and column shards.

Process honesty (three things that went wrong before green):

1. My first ling run on spark4 reported DRIFTED — MY tool bug (the
   filtered sample was re-indexed against directory positions). Fixed
   in f40d02f with a new structural_parity gate that fails loud on any
   directory/plan mismatch across ALL entries; the synthetic test was
   re-run and the real pack re-run clean. The DRIFTED was the oracle's
   fault, not the pack's.
2. The muse and (first) laguna/ling runs on spark4 ran during the
   SP-4R glm5_next pack build + K3A oracle contention; warm reads
   crawled in ceph folio waits (~64KB/s) and muse first thrashed on
   whole-plane loads (fixed in 690164f with row-span streaming). All
   spark4-hostile runs were re-run from spark5's idle client: ling
   69.6s, laguna 35.6s + 47.0s, versus 33+ min (muse, completed) and
   84+ min (ling, TERMed) on the contended client.
3. The planned full-digest cross-node check of laguna s0 from spark4
   was killed by a harness timeout (contended client); replaced by two
   complementary checks: system sha256sum on spark5 == oracle digest
   for both stage packs, and a 4MiB range at offset ~10.7GiB hashing
   identical from spark4 and spark5 clients (9ab3e62c...).

Minimax rank00..03 boundary check re-run output (this wave, spark4):

    rank00: 2842 entries, head rows 512, kv head 0
    rank01: 1928 entries, head rows 512, kv head 0
    rank02: 1928 entries, head rows 512, kv head 1
    rank03: 1928 entries, head rows 512, kv head 1
    PASS rank00..03: 9 payload shas vs warm slices
    BOUNDARY CHECK GREEN

Disk caches purged (sync + drop_caches) on spark4, spark5 and sparka
after the node batches.

Verdicts per lane (the dispatch question): ALL FIVE LANES'
VERIFIED PACKS ARE CHECKPOINT-FAITHFUL at the sampled/full coverage
above. No drift found between packer output and warm checkpoints
anywhere this wave looked. minimax ranks 04-15 remain UNVERIFIABLE
(packs do not exist yet).

## 4. Gap list — what is missing for a T1 accuracy receipt

T1 definition (operator): decode tokens vs reference within tolerance.
Common to every lane: no serving cell has ever produced tokens for
these models, and no reference-generation procedure exists. The daemon
is NOT the blocker for items marked OFFLINE.

### ling
- OFFLINE NOW: run the oracle over all 16 ranks (sidecars + receipts
  exist on every node after the 2026-09-13 placement); rehydrate the
  anchor fixtures into the repo; widen the oracle sample to --full on
  two ranks.
- BLOCKED ON GPU CELL (queue reservation, exclusive): bring up the
  resident decode cell (module vertical + serving-gate wiring already
  PASS on spark9 r23), run a frozen greedy prompt set, compare against
  publisher-reference decode from /mnt/model-warm/ling-3.0-flash
  (HF transformers + pinned modeling_ling_bailing_moe_v3.py). Needs a
  tolerance ruling (bf16 cross-implementation equality is not
  bit-guaranteed): propose the ruled band form the ling gates already
  use.
- THEN: ds4_eval COMPSEC-17 through the live endpoint.

### gemma4
- OFFLINE NOW: commit the anchor kit fixtures (npz or exported bins +
  manifest) — today the 121-site evidence is not reproducible from the
  repo alone; extend oracle coverage to the remaining 26B stages
  (0,1,3) and the other 15/3 ranks (packs must be emitted first: only
  3 receipt packs exist); fix the stale stagepack docstring.
- BLOCKED ON GPU CELL: first serving cell (dense 31B arm has kernels
  validated at sites=23; the serving adapter passes unit tests); then
  decode vs publisher reference (google/gemma-4-31B-it @ 842da379...);
  then COMPSEC-17.

### laguna
- OFFLINE NOW: oracle over the remaining 15 rank-packs (all on warm);
  re-run laguna_layer7_reference across more layers offline (CPU);
  repair the shared-Makefile gate debt that left gates exit 2.
- BLOCKED ON GPU CELL: serving cell (module compiled V0 on sparkb
  historically; no cell receipts), decode vs publisher reference
  (poolside/Laguna-S-2.1), COMPSEC-17.

### muse
- OFFLINE NOW: oracle over ranks 0-15 (rank04/05 verified this wave);
  cross-shard kv replication bitwise check (A7) as a tool, not a
  construction claim.
- BLOCKED ON GPU CELL: serving cell, decode vs publisher reference
  (meta-models/Muse-Glimmer-30B), COMPSEC-17.

### minimax-h3
- BLOCKED ON STORAGE (not the daemon): ranks 04-15 packs — sparke ceph
  client/MDS state must be cleared (remount or client reboot by
  sysadmin; 6/6 host events followed the same rank04 restart within
  7-20 min, data plane probes clean at 407 MB/s). Resume is one
  command: rerun emit_tp16_chain.sh (skip-if-receipt-complete + resume
  sig v2).
- OFFLINE NOW: re-run the boundary checker as ranks land; the
  rank07+rank08 concat proof fires automatically once those ranks
  exist.
- BLOCKED ON GPU CELL: first cell (media stage, step-pump ABI seam
  instrumentation designed), one denoise-step parity vs the pinned
  diffusers pipeline (3c221246), then a 5s-clip artifact acceptance
  (frame count 124 @ 24fps, 480x864, audio 165.6k samples/channel).

## 5. Incidental findings for the coordinator

1. CODE-SIZE RATCHET IS RED ON PRISTINE MAIN: origin/main d1c3822
   measures 284757 authored lines vs the pinned ceiling 283170 (+1587;
   the ratchet was not re-run at the #982/#983 landings). This branch
   re-pins to the measured 285193 with attribution (its own +436).
2. gemma4 anchor fixtures are not in the repo (reproducibility gap for
   the 121-site claim; the C oracle and exporter ARE committed).
3. tools/gemma4_stagepack.py docstring carries the superseded
   all-ones layer_scalar rule.
4. laguna offline-gates red (shared Makefile k3 adapter build target),
   open since 2026-09-10, recorded in packbuild/laguna/gates.
5. The muse two-pass placement proof in every receipt is the model for
   the "already placed" verdicts; wave-acc2's oracle is the missing
   content leg and is reusable for any family with a committed packer.

## 6. Ledger

- Branch: lane/wave-acc2-accuracy
  - 70942f5 oracle + synthetic test + manifest/sums
  - f40d02f directory-position fix + structural_parity gate
  - 2f38708 receipt no-digest-field honesty
  - 690164f muse row-span streaming fix
  - (this commit) wave report with oracle results
- Runs: sparka (/tmp/acc2_gemma4_*.json), spark4 (/tmp/acc2_muse_r4.json,
  /tmp/acc2_minimax_bc.log, /tmp/acc2_batch*.log), spark5
  (/tmp/acc2_ling_r5.json, /tmp/acc2_laguna_s0_r0.json,
  /tmp/acc2_laguna_s1_r0.json). Receipt JSONs should be harvested into
  the report dir by the coordinator if longer retention is wanted.
- Tools: tools/acc_parity_oracle.py, tests/test_acc_parity_oracle.py.
- Fleet state touched: zero packs written, zero daemons contacted, zero
  model executions, spark3/spark6 untouched; read-only warm access
  under sparkcap; disk caches purged on spark4/spark5/sparka after the
  batches.
