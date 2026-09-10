# qwen-flash: coexistence smoke PASS — bf16 + nvfp4 arms resident and validated concurrently — 2026-09-10

## Result

`qwen-flash-coexist-5` (queue v2, spark5, gpu, 10240 MiB, ttl 15, wall 141 s):
both placed arms validated CONCURRENTLY on one GPU with honest per-arm rc
files — **bf16 rc=0, nvfp4 rc=0**.

- bf16 arm: placed `qwen3flash.bf16.tp8` rank5 pack (1246 entries,
  MTP-carrying), module compiled MTP=1 → ladder 17/17 PASS.
- nvfp4 arm: placed `qwen3flash.nvfp4.tp8` rank5 pack (1215 entries,
  MTP-free, sha256 3cbad2bc…, matches its packer receipt), module compiled
  MTP=0 → ladder PASS (16 checks; `mtp_draft` is MTP=1-only by design).
  decode_vs_prefill bit_exact=1, module_determinism bit_exact=1,
  ple_hash_gather bit_exact=1.
- Residency proof: nvidia-smi sampler captured BOTH compute pids
  simultaneously (13905 MiB + 13891 MiB) mid-overlap on the same GPU.

This is the first demonstration that the standalone tier consumes the
coordinator's MTP-free nvfp4 arm — the gate #828 opens.

## Three defects the cell caught (each fixed in this PR)

The prior session's rerun plan assumed the ladder-gate relaxation
(a8130a6) alone would turn arm B green. The cell said otherwise — three
distinct defects, found in three successive runs (coexist-2/3/4), each
fixed and re-proven:

1. **Expected tensor count had no MTP term** (583b81f).
   `SparkQwen4FlashStagePackExpectedTensorCount` was derived from the
   MTP-carrying layout: head-owned slices counted the 31 draft-chain
   entries (2 MTP mixers + 4 MTP norms + 25 draft pseudo-layer tensors)
   unconditionally, so a module compiled MTP=0 expected 1246 while the
   MTP-free pack carries 1215 → `pack_geometry_mismatch`. The count is now
   MTP-conditional. `SparkQwen4FlashModuleExpectedGlobalBits` likewise
   required MTP_MIXER_DOWN/UP unconditionally; now MTP-conditional.

2. **NVFP4-packed experts rejected on fp8-natural shapes** (5a4e447).
   The shared `SparkStagePackShapeMoE` declares routed-expert shapes
   natural_format FP8_E4M3_F32B128, but the module's nvfp4-release-arm
   clause accepted NVFP4_PACKED only on MXFP4-natural shapes → the placed
   nvfp4 pack's experts failed entry validation (`pack_entry_invalid
   kind=6 layer=0`). Every other check (payload elements/2, per-16 e4m3
   plane + per-expert F32 input/weight globals, scale_group_size 16)
   already matched the module's own byte formulas. The clause now accepts
   NVFP4_PACKED on MXFP4- or FP8-natural shapes — in this family those
   are exactly the routed experts.

3. **Validator harness compiled without the MTP define** (245a7e8).
   `make validate` runs the validator SCRIPT, and the script's nvcc line
   for the harness .cu never forwarded the MTP knob — the harness always
   compiled with the family default MTP=1, ran its `mtp_draft` check, and
   the MTP=0 module's new fail-closed draft guard returned
   UNSUPPORTED (ERRSITE module.c:2354). The script now forwards
   `-DSPARK_QWEN4_FLASH_MODEL_MTP_LAYER_COUNT=$SPARK_QWEN4_FLASH_STAGE_MTP`.

Also fixed in this PR's cell tooling: the coexistence sentinel composed
`wait`'s status (a subshell echo, always 0) instead of make's rc — the
rerun reads per-arm rc files as ground truth, and arms build in SEPARATE
checkouts so the two geometries (MTP=1/MTP=0) never share objects (make
ignores -D changes).

## Fail-closed preserved

The MTP=0 module does NOT silently skip a draft request: any consumer
sending FLAG_MTP_DRAFT_AFTER to a no-MTP module gets
SPARK_STATUS_UNSUPPORTED with an ERBSITE line naming the birth site
(SparkQwen4FlashModuleRunMtpDraftChain compile-out + guard). The shared
synthesize header gained an overridable SPARK_SYNTH_EMIT_MTP_TAIL
(default 1 = unchanged behavior for the other three families).

## Run identity

- queue job qwen-flash-coexist-5 (attempt 800eed59 lineage), submitted by
  qwen38flash, node spark5, resources gpu, budget 10240 MiB, ttl 15 min.
- Sources: lane/qwen38flash-dev @ 583b81f + 5a4e447 + 245a7e8 overlaid on
  the queue-synced clean main 8f3a6f2 checkout (qwen-flash-cuda5b) and its
  arm-B copy (qwen-flash-cuda5b-armb).
- Packs: bf16.tp8 rank5 + .experts sidecar; nvfp4.tp8 rank5 + .sha256
  (coordinator-placed, not lane-owned).

## Next (completion sequence)

3. Multi-rank TP8 functional via --per-node over spark0-7 (verify every
   rank pack + .experts sidecar first).
4. TP4xPP4 wave + B1 (determinism double-smoke).
5. Lazy-qualified gate: two real consumers x cold/hit/eviction/reload +
   cancellation + numerics inside declared budgets.
