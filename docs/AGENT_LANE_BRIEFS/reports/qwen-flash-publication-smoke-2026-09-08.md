# qwen-flash: publication smoke PASS — module consumes placed fleet arms — 2026-09-08

## Result

`qwen4_flash_validation PASS` — rc=0, 17/17 checks, wall 123s (incl. eager
46GB pack load) on spark4, against the COORDINATOR-PLACED
`qwen3flash.bf16.tp8` rank0 pack (Q4SP v2, 1246 entries, full 48 layers,
MTP-carrying). The qwen4_flash module publishes against the fleet's placed
arms: no lane-owned packs are needed.

## Checks (selected)

- gdn_chunk_output rel_l2 0.00166 / cosine 0.99999863; gdn_chunk_state
  cosine 1.0
- hc_residual rel_l2 0.00242 / cosine 0.99999711
- indexer_pooled rel_l2 0.00497 / cosine 0.99998764
- indexer_select boundary_flips=0 mask_mismatches=0
- ple_hash_gather bit_exact=1 nonzero=10240/10240
- decode-vs-prefill, determinism, head/MTP ladder: PASS (17/17)

## Path to green (each step a distinct finding)

1. Stale S5-era deploy binaries rejected fresh deployment.json (schema 2) —
   serving stack rebuilt from main (PR #781 lineage).
2. Eager 46GB load OOM'd beside the resident glm53 weightd (42GB, NOT ours
   to kill). Fix: weightd-attach lazy load (SPARK_WEIGHTD_SOCKET +
   SPARK_WEIGHTD_ATTACH=1 → pack-arena slices, zero eager copies). NOTE: on
   this run weightd held glm53 identity → graceful fallback to eager; the
   lazy arena path for qwen3flash identity is the weightd registry phase.
3. Ladder geometry: standalone tier requires a FULL 48-layer TP-rank-0 pack —
   tp4pp4 slices are not standalone-validatable; bf16.tp8 rank0 is the right
   arm.
4. MTP geometry: placed packs are MTP-carrying (mtp=1) while the ladder
   demanded a strict match — fixed in this PR: the standalone tier accepts
   MTP-free packs too (the operator's separate-MTP-sidecar design).
5. OPERATIONAL LAW (re-learned): detached GPU work has NO queue hold — a
   concurrent mb_doorbell bench killed the first detached attempt silently.
   The passing run held spark4 exclusively through the queue.

## Lazy-load engagement (operator directive, resolved)

The lazy expert load requires ALL of:
1. `<pack>.experts` sidecar (WEPX per-expert manifest) beside the pack —
   weightd's per-expert residency index (spark0's tp8 rank0 pack lacked it;
   spark4's rank4 pair has both files).
2. `SPARK_WEIGHTD_PACK_SHA256` — NOTE the exact name (SPARK_WEIGHTD_SHA256 is
   NOT a real knob; the silent no_identity was this).
3. `SPARK_WEIGHTD_IDENTITY_MODEL` = Qwen/Qwen3.8-Flash-Next + socket/attach.
Result: fallback lines GONE, ladder PASS rc=0 wall=70s (vs 84-123s eager), no
46GB eager copy — pages fault through weightd's mapping.

## Next

- Coexistence smoke (operator directive): two arms resident on one node —
  bf16.tp8 rank0 + nvfp4/fp8 arm — both ready, both smoke, nvidia-smi as
  residency proof.
- Wave10 (16-rank baseline): deployment regenerated from merged main,
  placed bf16 arms, then B1 smoke ×2 (determinism hash) + exact-32K cell.
- Follow-up PR (common code, coordinator review): load-and-ignore MTP —
  module compiled MTP=0 accepts MTP-carrying packs (skips the entries),
  removing the need to compile the draft chain when unused.
