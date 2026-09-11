# PROGRESS.md — laguna lane (lane/laguna-driver)

Coder stage log. DESIGN.md is the contract; this file records what landed,
what is blocked, and what the manager must serialize.

## Ruling status

- A (muse LmHeadRmsNormKernel): origin/lane/muse-driver DOES NOT EXIST on
  origin (ls-remote empty) and muse's kernel is not on main. Fallback per
  ruling: copied the kernel EXACTLY from the local batch-muse clone
  (lane/muse-driver, inference/kernels/norm.cuh) into our norm.cuh —
  separate in/out pointers, grid (head=x,row=y), optional weight, bf16
  round-trip before the `head_multiply` epilogue. Laguna passes
  head_multiply=1.0. Text is byte-identical to muse's; dedupes at merge.
- B (LmRopePerHeadKernel + 3 trailing defaulted params): landed
  (project.cuh + the attention_scale thread through attn.cuh
  LmRopePair/LmRopeRotate). Existing-family verification: glm52 layer host,
  dsv4 layer host, gqa host, k3 layer host all build and pass on the mac
  (exit 0) after the edit.
- C (LmHeadGateBroadcastKernel<SIGMOID|SOFTPLUS>): landed in norm.cuh.
  ling's LmHeadWiseGateKernel is NOT on main (PR #830 open) — per the
  ruling, laguna does NOT convert ling's call sites. MANAGER: whoever
  merges second converts and deletes the older variant.
- D: gqa.cuh untouched (git diff clean).
- E: tools/dev/port_family.py copied from batch-ling, byte-identical.
- F: TP8xPP2, stages {24,24}, session port base 64800.

## Criteria

1. DONE — port executed, 22 files, all per-file counts nonzero (commit 201d100).
2. in progress
...

## Criteria status at handoff (2026-09-09, coder handoff)

1. DONE — port executed, 22 files, all counts nonzero (201d100).
2. DONE — family complete: model header (tables+asserts), GQA kv_geometry,
   census 23/36769, firmware description (a7687e2, 40074e2, 25b06a9).
3. DONE (pre-freeze) — contract + header bind test green on mac; reference
   .py byte copies + revision/shard shas PIN AT FREEZE on sparkb.
4. PARTIAL — all mac compile checks green under -Werror (module.c, adapter,
   synthesize tool, format/kv/config headers, host tests glm52/layer/gqa/k3);
   `make offline-gates` on sparkb NOT RUN (no sparkb queue access this
   session) — manager must fire it.
5. DONE — residue grep clean (remaining hits are KV_BITS/LmTpBf16 false
   positives); MTP parity + flash-decode validation files deleted.
6. DONE — layer.cuh defines ZERO private attention/router/norm kernels; the
   path is gqa.cuh + project.cuh rope + split + norm.cuh kernels only
   (qk-norm = muse's LmHeadRmsNormKernel head_multiply=1.0; gate =
   LmHeadGateBroadcastKernel<SOFTPLUS>).
7. BLOCKED (mac) — tests/host_cuda/laguna_layer_host.cu + python oracle
   harness not yet written; oracle exists (tools/laguna_layer_reference.py).
   The kernels themselves are verified by the existing host tests after the
   Flag B/C edits, and the V0 validator dumps fixtures for the deep compare.
8. MECHANISM DONE — oracle --dump-yarn (independent HF formula) vs the
   validator's yarn dump; the <=1e-6 comparison runs at V0 on sparkb.
9. MOSTLY — packer rewritten (census lock fail-closed, per-section whole-head
   q|k|v rows, gate-first W1, expert intermediate slicing, receipts, .lgsp);
   synthetic-fixture end-to-end pack load NOT exercised (needs a fixture
   safetensors dir — sparkb task); real-pack BLOCKED on warm download.
10. DONE — gen_deployment emits TP8xPP2 (16 rank-stages, stage_layer_counts
    {24,24}, eos {2,24}); session base REQUIRED env (SPARK_LAGUNA_SESSION_BASE),
    no frozen default — PENDING FLEET RENUMBER (64800+ unsafe with route-kind
    offsets; the concrete 64800 must come from the renumber, not this lane).
11. DONE — synthesize --dflash emits flags 0x2 + one ignored kind-19 section
    (pack-level proof); module loader accepts/records/skips (code path
    SparkLagunaPackValidateEntryGeometry/PackLoadEntry); end-to-end load
    proof runs at V0 on sparkb.
12. PENDING — PACKAGE_MANIFEST.json + SHA256SUMS regenerate LAST after the
    sparkb gates; NOT done in this session (any earlier regen would go stale
    with the pending sparkb runs).

## Manager notes

- Ruling A: muse branch absent on origin; kernel copied byte-exact from the
  local batch-muse clone (lane/muse-driver). Dedupes at merge.
- Ruling C: ling's LmHeadWiseGateKernel NOT on main → laguna did not convert
  its call sites; whoever merges second converts + deletes the older variant.
- Topology: TP8xPP2 default (operator exemption granted mid-session);
  TP4xPP4 macros + stage validator support kept as the standard alternative.
- Deviation from DESIGN §4 wording: "experts 32/rank" is implemented as the
  donor grouped-GEMM slicing (all 256 experts resident per rank, W1 rows /
  W2 cols sharded 8-way) — whole-expert partition would need cross-rank
  dispatch machinery that does not exist in the tree; per-rank expert bytes
  identical (211.5 GiB / 8).
- BLOCKED on sparkb: offline-gates, V0 synth validation, fixture-pack load,
  real-pack boundary-rank checks (warm download must complete; poll the
  marker, do not blind-sleep), contract freeze shas, manifest+sums regen.

## Validation session (2026-09-10/11, validation-debugger)

REBASE VERDICT (target recorded per coordinator): rebased lane/laguna-driver
onto origin/main 50bd0d3 (PR #913, the E2E-proven mesh platform). Path:
suspension tip 97f1aff -> coredev-aligned origin/lane/laguna-driver 5acf486
(= main 8f3a6f2 + 24 laguna family + coredev mesh commits) -> 50bd0d3.
- Patch-equivalent coredev mesh commits auto-skipped onto main's wave
  (5acf486->aba22b2, ef6fcd0->feda24c, 78b07e1->99f42c0, ...); three
  shared-file-only commits resolved toward origin (f6ce20a, 8b19353 + the
  Makefile arm - content already on main, skipped as subsumed; bf85caf
  skipped: shared-only, superseded by the E2E-proven wave). 6b9ffcc's
  family hunk (laguna module.c credit-binding strip + mesh receive hook)
  kept; its shared surface merged to main's (32MB slots, no duplicates).
- Family files byte-kept (packer hash b782c8c4 identical pre/post rebase -
  the in-flight sparkd pack is exactly what the rebased branch produces).
- Ceiling re-measure: merged tree = 240779 authored lines, under main's
  241052 pin (main's wave deleted the old-engine/nccl test surface); the
  lane's interim 246010 pin removed as dead code (bc82860).
- Post-rebase mac checks green: header bind test, make contract (module +
  host sources vs main's transport), make adapter (dylib), synthesize tool
  under real flags, code-size test.

FINDINGS fixed en route (each fail-loud, exact-site):
1. make contract was RED since 5c14a6f: SparkLagunaPackAssignLayer took
   state it never used (-Wextra -Werror); dead parameter dropped (ae67d88).
2. Packer never ran end-to-end; first real run exposed four breaks
   (6d659b0): census regex re.escape ate the {layer}/{expert} braces so
   EVERY checkpoint tensor was rejected; receipt() read nonexistent Entry
   fields; pack header REVISION/CONTRACT_SHA256 were undefined names (now
   required --revision/--contract-sha256 CLI threaded to assemble_header);
   donor docstring replaced with actual behavior.
3. Generator/deploy mismatch (6d659b0): pack template named packs the
   packer never emits (laguna-s-2.1.bf16.tp8pp2.stage%d.rank%d vs the
   packer's laguna_stage.tp8.pp2.stage%d.rank%d) and numbered stage-1
   packs by global rank; fixed with rank%%TP.
4. Adapter tp_rank check compared config tp_rank (0..7) against the GLOBAL
   stage index (0..15): ranks 8..15 unservable - module validate rejects
   tp_rank>=8; now stage_index %% TP_DEGREE (6d659b0).
5. Expert W2 producer used tp_shard_range's (start, count) as
   [c0, c1): rank 0 worked by accident, rank 1 emitted an empty region
   (db84461) - found by the real pack at stage0/rank1.
6. sparkcap over non-interactive ssh: systemd transient scope needs root
   authorization - sudo -n sparkcap is the working form (worker fixed).

REAL-PACK (sparkd, sparkcap --mem 4096, resumable per-shard markers):
warm source verified complete (241G, 46 shards, HF tree id
0f573140834b11cfac0c2af97a101a7a69a13e22 == the worker's --revision;
--contract-sha256 == sha256 of model_contracts/laguna_authoritative.json
354f559d...). census lock green on all 36769 real tensors. stage0 rank0-1
packed (~14G/rank-stage, ~4min each), remainder running; receipts appended
to /mnt/model-warm/packbuild/laguna/real_pack.log.
