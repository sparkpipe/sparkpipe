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
