# Qwen 3.8 Flash M5 kernel lane report — 2026-08-28

Worktree /tmp/lane-qwenflash, branch lane/qwen-flash. Build/GPU node sparka
(GB10, idle, CUDA 13.0); whole-stack evidence additionally on spark4 (idle
when used; the deployed rank packs live there). The deployed packs from the
P1-P4 session (format-4 experts, NARROWED router gate) were the test
artifacts; a v2 pack generation (replicated gate) is building on spark4 as
this is written.

Mission framing corrections, with evidence:

  * "Port the hyper-connection readout from the qwen38_max module which
    handles the same pattern" - qwen38_max contains NO hyper-connection code
    (grep hyper|hc_|input_mix|block_inject|sinkhorn over
    modules/qwen38_max_resident_decode_stage: zero hits). The only in-repo
    hc implementation is dsv4's, a different parameterization (Sinkhorn
    mixing from fn/base/scale vectors, 24 scalar mixes/row) than the Flash
    checkpoint's low-rank mixers. There is no porting source. See M5.2.
  * M5.2/M5.4/M5.5 are blocked on MODELING SEMANTICS, not kernel code: no
    modeling reference for qwen3.8-flash-next exists anywhere reachable
    (repo grep: input_mix_weight appears only in the flash model header;
    `import transformers` fails on the nodes; `find / -name 'modeling_qwen*'`
    on spark4 returns nothing; the warm mount ships weights + config only;
    the CPU oracle has no hc/indexer/ple). Details per milestone below.

## Milestones

### M5.1 TP_STANDALONE bypass + format-6 acceptance — DONE

TP_STANDALONE (the 27b escape hatch): `SPARK_QWEN4_FLASH_TP_STANDALONE`
skips the collective entirely and turns every reduce into a no-op, so ONE
rank exercises the whole-stack sharded paths without a peer group.
Embedding gathers stay rank-partial and the head argmax covers only the
rank's vocab shard - validation semantics, never serving. The retained
validation script already required this env (it was planted for exactly
this port); the Makefile already plumbs it.

Format-6 (FP8_E4M3_E8M0B128, `--expert-format fp8-e8m0b128`):

  * The packer's old format-6 scale plane was per-128x128-TILE e8m0 bytes -
    INCOMPATIBLE with the common kernels' E8M0B128 decode, which is per-ROW
    (SparkLmDotRowFp8E8m0: scale index = row * (columns/128) + k/128, and
    SparkLmGroupedScalarLinearKernel has NO format-6 branch at all - an
    unknown format falls through to the MXFP4 decode). Fixed the packer to
    emit the per-row plane (finer scales; exponent rounded UP so |v|/scale
    never exceeds the E4M3 max), with the verifier's byte formula and
    dequant trace updated to the same layout.
  * Module format-6 grouped expert paths: rows >= 8 instantiate the shared
    mloop tile kernel template at GROUP_SIZE 128 (its E8M0 else-branch then
    indexes the per-row plane exactly); rows < 8 run a family-local scalar
    grouped kernel (same task decomposition as the common one, decoding each
    neuron through the shared SparkLmDotRowFp8E8m0).

**Exit evidence** (sparka, real weights):

```
python3 tools/qwen4_flash_stagepack.py --checkpoint /mnt/model-warm/qwen3.8-flash-next \
  --output build/stagepacks/qwen4_flash_slice0p4.tp1.e8m0.qwen4_flashsp \
  --first-layer 0 --layer-count 4 --expert-format fp8-e8m0b128
qwen4_flash_stagepack slice=0+4 tp=1/0 tensors=74 file_gib=11.10 wrote ...

python3 tools/qwen4_flash_pack_verify.py --pack ...e8m0.qwen4_flashsp \
  --checkpoint /mnt/model-warm/qwen3.8-flash-next --sample 10
trace kind=6 layer=2 fp8 relative_l2=0.02659
trace kind=8 layer=3 fp8 relative_l2=0.02657
PASS ... 74 directory entries (tp 1/0), 10 byte-traced samples receipt=verified

make -C modules/qwen4_flash_resident_decode_stage validate ... (mid-pipeline tier, format-6 pack)
qwen4_flash_validation check=module_decode_vs_prefill elements=2560 relative_l2=0 cosine=1 max_abs=0
qwen4_flash_validation check=module_determinism bit_exact=1
qwen4_flash_validation PASS
```

CPU round-trip of the new quantizer (verifier dequant path): format-6
relative_l2 0.02644 vs format-4's 0.02646 on the same expert-shaped matrix;
byte counts match the module's formulas exactly (payload rows*cols, scale
rows*(cols/128)).

### M5.3 whole-stack TP4 enablement (argmax/embedding AND the sharded layer
machinery the guard was hiding) — DONE, plus three real bugs found

Ported from 27b: sharded head screening (HeadScreenedArgmaxScore with the
rank's candidate offset) + maxloc u64 pack/unpack + the u64-max collective
(SparkTpDeviceCollectiveSubmitU64Max with a combine callback), vocab-sharded
embedding gather (in-block rows copy, else zero; the bf16 all-reduce
completes it), MTP draft argmax through the same sharded flow. The
`tp_whole_stack_pending` fail-closed guard is retired.

Whole-stack TP4 first exposed that the guard had been hiding FOUR untested
breakages, each fixed and verified:

  1. GDN kernels were FULL-WIDTH at tp>1. The packs shard the GDN
     projections AND conv/A_log/dt_bias in a stitched rank-local layout
     (2*(K/tp*128) + V/tp*128 channels); the conv update, decay/beta, gdn
     step, gated norm and all chunk kernels read the full-width layout and
     48-head parameter slabs - garbage from layer 0 (the first whole-stack
     run emitted argmax sentinel 0xffffffff). All kernels now take
     tp_degree and derive rank-local dims; the state pool keeps full-width
     strides (each rank uses its head shard's slots). At degree 1 the port
     is numerically a no-op (mid-pipeline regression bit-identical).
  2. mtp_draft_ids was used but NEVER allocated (NULL deref on the MTP
     chain's first-ever exercise; mid-pipeline tiers never ran MTP).
  3. The MoE route-offset rank shift was wrong for the deployed packs: the
     route is built over the NARROWED gate (rank-local top-k), so
     group_row_offset[0..experts_per_rank] is already the valid prefix -
     shifting by rank*experts_per_rank read stale rows. Now derived from
     the gate width (see the gate note below). rows_per_expert also divided
     by the full expert count (wrong stride at tp>1 on rank-local views).
     The shared-expert SwiGlu width is the rank's narrowed intermediate.
  4. TWO latent module bugs the whole-stack decode-vs-prefill gate caught
     (found with an env-gated per-layer hidden dump,
     SPARK_QWEN4_FLASH_STAGE_DEBUG_DUMP_HIDDEN):
       a. The internal (adapter-less) staging computed slot_mapping as
          position % 64 - LANE-BLIND, always block 0 - while attention
          reads resolve blocks through the lane's table. Lane 1 wrote
          block 0 and read block 1 (never written). Fixed: the smoke path
          now resolves the PHYSICAL slot through the table, same contract
          as the serving adapter. (Two identical prefills on different
          lanes are now bit-identical through all 48 layers.)
       b. Prefill continuation reset the GDN state: host_row_cold keyed on
          the FRAME-LOCAL token index, so every 1-token continuation
          prefill restarted the recurrence while the decode stayed warm.
          Cold is now keyed on the sequence position. This was ALSO the
          source of the long-standing 1.4e-3 decode-vs-prefill drift: the
          mid-pipeline gate is now BIT-EXACT (relative_l2=0, cosine=1).
  Plus: the MTP attention ran at cache ordinal attn_layer_count in a cache
  sized attn_layer_count - a past-the-end write into adjacent device
  memory. The cache now allocates one more layer when the MTP tensors load
  (the MTP drafts changed after the fix: it had been reading partially
  corrupted cache).

**Exit evidence** (sparka AND spark4, DEPLOYED rank0 pack ef8bd0c9..., same
command, whole-stack tier TP4/0 standalone):

```
qwen4_flash_stage tp_standalone degree=4 rank=0 (collective skipped; ...)
qwen4_flash_stage initialize ok slice=0+48 gdn=36 attn=12 owns_embedding=1 owns_head=1
qwen4_flash_validation check=decay_gate   relative_l2=7.56e-08 cosine=1
... (all nine kernel checks at the M3 thresholds)
qwen4_flash_validation check=module_admit_snapshot admit=ok snapshot=ok
qwen4_flash_validation check=module_decode_vs_prefill decode_token=37853 prefill_token=37853 bit_exact=1
qwen4_flash_validation check=module_mtp_draft in_vocab=1 drafts=[17991,23116]
qwen4_flash_validation check=module_determinism bit_exact=1
qwen4_flash_validation PASS
```

sparka and spark4 produce the IDENTICAL token stream (37853; drafts
17991/23116 after the MTP cache fix) from the same pack - cross-node
bit-reproducibility. Mid-pipeline TP1 regression after every change:
PASS, decode-vs-prefill now relative_l2=0 (was 1.42688586e-03).

Router gate architecture note (deployed packs carry a narrowed gate): a
rank-local top-10 is NOT the model's top-10 - the union across ranks is
wrong for serving. Correct plan = REPLICATED gate (every rank scores all
512, routes the same global top-k, executes only its shard's pairs; the
all-reduce completes the mixture). Implemented: the packer now replicates
KIND_MOE_GATE; the module accepts BOTH gate widths and derives the route
group base from the loaded width (narrowed packs keep running, self-
consistent, for validation). v2 packs building on spark4 (below).

### M5.6 whole-stack TP4 smoke — VALIDATOR GATE DONE (standalone)

spark4, the deployed pack, the retained recipe:
`make ... validate STAGE_COUNT=1 STAGE_LAYER_COUNT=48 MTP=1 TP_DEGREE=4
TP_RANK=0 TP_STANDALONE=1` → full ladder PASS (above). This is the
milestone's stated gate. The LIVE 4-node collective run needs the
residentd/api deployment (driver compile, M6): the module-side collective
paths (embedding all-reduce, u64 maxloc, per-layer residual reduce) are
coded and the collective env/backend .so are deployed from the packs
session, but no qwen4_flash residentd binary exists yet.

### M5.2 hyper-connection / M5.4 attention indexer / M5.5 PLE — BLOCKED on
modeling semantics (not on kernel capacity)

  * M5.2 hc: checkpoint facts (safetensors headers, read directly):
    per-sublayer attn/mlp_hyper_connection {input_mix_weight_up [10240,320],
    input_mix_weight_down [320,10240], block_inject_weight [4,10240],
    hc_norm [10240]} x 2 x 48 layers + MTP, plus the global
    hyper_connection_mixer {up, down, hc_norm} readout (NO final norm
    exists). Budget if ported: ~1.3 GB bf16 replicated - cheap. BLOCKED:
    no modeling source for the low-rank chain -> 4-stream mixing dataflow
    (several wirings fit the shapes; dsv4's Sinkhorn hc is NOT this form).
  * M5.4 indexer: index_qk_proj [640,2560] (= (4 q heads + 1 kv head) x
    128) + q/k layernorms [128] on the 12 full-attn layers + MTP, budget
    2048, compress ratio 4. dsv4's indexer (compressor wkv/wgate/norm/ape
    + p-cache scoring) is a different, heavier design; the Flash selection
    semantics are not derivable from shapes. ~43 MB - trivially packable.
  * M5.5 PLE at layer 1 (weights truth; config says 2 - M1 drift note):
    137 tensors. The ngram embedding is 128 shards x [2500012, 160] BF16 =
    95.4 GiB - about the size of the entire current 4-rack pack fleet
    (123 GiB). Plus checkpoint-only composition metadata
    (layer_multipliers I64[3], ngram_heads_offsets I64[16],
    ngram_heads_vocab_sizes I64[16], vocab base 20M, 8 heads/ngram,
    16x160=2560 value_proj in). TWO blockers: semantics source AND a
    precision/budget decision (bf16 vocab-sharded = 23.9 GiB/rank; fp8
    halves it but needs a numerics baseline call).

Honest negatives from the debugging window: the first whole-stack runs
failed with argmax sentinel ids and then a decode-vs-prefill token flip;
both were real module bugs (GDN widths; slot mapping; cold reset; the MTP
OOB), not validator artifacts. The skip_gdn debug probe also proved
misleading (it manufactures stale-delta reads) - documented here so nobody
re-runs that bisect expecting signal.

## v2 pack generation (rank 0 verified + validated; 1-3 building)

packs_v2 on spark4: format-4 experts (byte-comparable to the deployed
class) + REPLICATED router gates. Rank 0:

```
qwen4_flash_stagepack slice=0+48 tp=4/0 tensors=899 file_gib=30.89 wrote ...packs_v2/...rank0...
PASS qwen4_flash_full.tp4-rank0.qwen4_flashsp: header geometry, 899 directory
     entries (tp 4/0), 10 byte-traced samples receipt=verified

whole-stack TP4 standalone, v2 pack (dual-mode replicated-gate path):
qwen4_flash_validation check=module_decode_vs_prefill decode_token=17776 prefill_token=17776 bit_exact=1
qwen4_flash_validation check=module_mtp_draft in_vocab=1 drafts=[377,20656]
qwen4_flash_validation check=module_determinism bit_exact=1
qwen4_flash_validation PASS
```

Tokens differ from the v1 pack run (17776 vs 37853) exactly as expected:
the global top-10 over 512 experts is a different mixture than rank 0's
local top-10 over 128. Ranks 1-3 build+verify with the same loop
(/tmp/q4f_v2_build.log on spark4); deploy = rsync packs_v2 to spark5-7 and
point the launchers at it (next session; the deployed v1 packs remain
valid for standalone validation).

## INTEGRATION REQUEST

  1. (coordinator) Modeling reference for qwen3.8-flash-next
     (modeling_qwen3_8_flash or equivalent) from the publisher - the ONLY
     blocker for M5.2/M5.4/M5.5. Shapes and budgets are ready (contract
     census + this report).
  2. (coordinator) PLE precision/budget decision (95.4 GiB bf16): fp8
     ngram tables (~48 GiB, needs a numerics baseline call) vs vocab
     sharding across ranks (23.9 GiB/rank bf16).
  3. (driver lane / M6) residentd + api deployment for the LIVE 4-node
     collective smoke: module-side live-collective paths are coded
     (embedding all-reduce + u64 maxloc + per-layer residual reduce), the
     collective env and backend .so are deployed from the packs session,
     the packs are on all four nodes. What is missing is the compiled
     serving stack.
  4. (coordinator, FYI) The v2 packs (replicated gate) supersede the
     deployed generation for SERVING once verified; the deployed packs
     remain valid for standalone validation. Old packs still load and run
     (dual-width gate acceptance in the module).
  5. tests wiring for tests/test_qwen4_flash_model_header.py (Makefile
     TEST list + sources.mk) - unchanged from the previous reports.

## Commits (this session)

  * module Makefile: the adapter contract-sha provenance pin compiled
    EMPTY (REPOSITORY_ROOT undefined at immediate-expansion time) - now
    CURDIR-relative; the pin builds as
    d5227ed0792eda1c24272ce549538a793d0f861000f7057db99afbccb79ed09c.
  * module: TP_STANDALONE bypass, whole-stack TP4 enablement (sharded
    embedding + head argmax + MTP chain, rank-local GDN kernels, dual-width
    router gate + route base, format-6 expert paths incl. the family-local
    scalar kernel), fixes: mtp_draft_ids allocation, physical slot-mapping
    resolution in the smoke path, prefill continuation cold semantics, MTP
    cache layer allocation.
  * packer/verifier: format-6 per-row E8M0 plane (byte formulas + trace),
    replicated router gate.
  * validation: tp_degree plumbing for the GDN launcher oracle checks,
    env-gated token-parity warn mode + the debug hidden-dump hook.
  * ratchet 186345 -> 187008 (M5 kernel port window; same commit).

## Next

  1. v2 packs: finish build+verify, rank-0 whole-stack standalone on spark4
     (dual-mode gate), deploy to spark5-7.
  2. M6 driver compile -> residentd/api on spark4-7 -> LIVE 4-node
     collective smoke (the collective env is ready).
  3. On the modeling reference landing: hc readout first (cheapest,
     ~1.3 GB), then the indexer data path, then PLE after the budget call.
