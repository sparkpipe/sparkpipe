# Qwen 3.8 Flash lane report — 2026-08-27

Worktree /tmp/lane-qwenflash, branch lane/qwen-flash. Nodes used: spark4
(freeze + module compile), spark5 (GPU validation + packs). spark4 and spark5
each host a DSV4-lane residentd (~40 GB device each, left untouched); all my
GPU work ran beside them with no resets, no kills.

Source: /mnt/model-warm/qwen3.8-flash-next, revision
f5d08274bafd880402bd16f5e3e6c514136ec06c, 144 model files +
ARCHIVE-RECEIPT.json/PUBLISHED/SHA256SUMS (147 files on warm; brief said
"149 files" — actual count is 147), 360,023,351,155 model bytes.

## Milestones

### M1 Contract freeze — IN PROGRESS (tool done, digests hashing)

Tool: `tools/qwen4_flash_verify_source.py` (freeze/verify/census modes).
Sampling policy: all 18 files < 256 MiB fully hashed + shards on stride 17
from 1 (9 of 131: 1,18,35,52,62(largest),69,86,103,120) + publisher
SHA256SUMS + ARCHIVE-RECEIPT.json hashed, transitively pinning the rest.

The warm Ceph mount measured ~4 MiB/s single-stream, ~30 MiB/s aggregate
(dd/head timings in the session log), so the initial stride-5 plan (~85 GiB)
was widened to stride-17 (~27 GiB). The parallel (8-way) freeze started
18:12 KST on spark4 and is still IO-bound at report time. The contract JSON
lands when it completes; the M2 header conformance test
(tests/test_qwen4_flash_model_header.py) is written and waits on it.

Census (from index.json, pinned for the contract): 1658 entries, 227 pattern
classes. Checkpoint facts that DIFFER from the brief's "sibling" framing:

  * PLE tensors live at `layers.1.ple.*` (137 tensors) although config says
    `ple_layer_ids: [2]` — the weights win; config drift recorded.
  * q_proj is [12288, 2560] = query(6144) + sigmoid gate(6144) — matches the
    max sibling's fused query|gate layout; config num_attention_heads 24.
  * NO `model.language_model.norm.weight` exists: the final readout is the
    hyper_connection_mixer (hc_norm [10240] + lowrank 320 mixers).
  * lm_head [248320, 2560]; embed [248320, 2560].

### M2 Geometry header — DONE (commit 2ecb9c5)

`model-families/qwen4_flash/include/sparkpipe/spark_qwen4_flash_model.h`
mirrors the qwen38_max header shape with the Flash geometry (h2560 l48
gdn36/full12, gdn k16/v48 x128 → grouped-value ratio 3 (NOT the max
sibling's 8), conv 10240, attn q24/kv2 x256 rope64 theta 1e7, MoE 512/10/640
+1 shared, mtp 1, hc 4x320, indexer 4+1 x128 budget 2048 compress 4, PLE
ngram3 8-per 128 shards layer 1). Conformance test
`tests/test_qwen4_flash_model_header.py` binds 36 header constants + 8
composed invariants to the authoritative contract (runs once the contract
JSON lands with M1).

### M3 Module skeleton + validator PASS — DONE (commit 2ecb9c5)

`modules/qwen4_flash_resident_decode_stage/` copied from
qwen38_max_resident_decode_stage and re-parameterized. The max module DOES
NOT BUILD on main — my port fixed, with evidence:

  1. `include $(REPOSITORY_ROOT)/modules/resident_decode_stage_rules.mk`
     expands REPOSITORY_ROOT before rules.mk defines it → "No rule to make
     target '/modules/resident_decode_stage_rules.mk'" (verified by building
     the pristine module on spark4; same failure).
  2. `.cu` references `SPARK_QWEN38_RESIDENT_DECODE_STAGE_*` macros defined
     nowhere in the repo (stale pre-rename names).
  3. `SparkLmLinearKernel<32u,SPARK_ACTIVATION_CODEC_NONE>` — the shared
     template grew a third parameter (CTA_WARPS); the 27b module's form is
     the fix.
  4. Admit/Snapshot were UNSUPPORTED stubs; Execute rejected all prefill
     frames. Ported from the proven qwen38_27b unit: the admission table +
     slot machinery, the runtime snapshot, and a sequential-walk prefill
     path (token t runs the same per-row machinery as a decode row on one
     lane, warm after the first token — bit-equivalent to the chunk walk by
     construction; only the final token emits head/hidden).

Validation harness ported to `modules/qwen4_flash_resident_decode_stage/
validation/` from qwen38_27b (oracle + CUDA harness + validate.sh), with:
attn GQA group macro-derived (27b hardcoded 6; Flash is 12 — this was the
one real oracle bug found by the run), TP-arg launch signatures, 48-layer
tier bounds, `head / 3u` derived from the grouped-value macro.

**Exit evidence** (spark5, GB10, mid-pipeline tier, synthetic pack):

```
make -C modules/qwen4_flash_resident_decode_stage validate \
  NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a \
  STAGE_PACK_PATH=build/stagepacks/qwen4_flash_midPipeline.qwen4_flashsp \
  STAGE_COUNT=2 STAGE_INDEX=0 STAGE_FIRST_LAYER=0 STAGE_LAYER_COUNT=4 \
  MTP_LAYER_COUNT=0 MAX_ACTIVE_SEQUENCES=8 KV_BLOCK_COUNT=8 \
  ALLOW_UNQUALIFIED_EXECUTION=1            → exit 0

qwen4_flash_validation check=decay_gate     relative_l2=7.6e-08 cosine=1
qwen4_flash_validation check=write_gate     relative_l2=4.1e-08 cosine=1
qwen4_flash_validation check=conv_update    relative_l2=1.65e-3 cosine=0.999998634
qwen4_flash_validation check=gdn_step_output relative_l2=1.69e-3 cosine=0.999998578
qwen4_flash_validation check=gdn_step_state relative_l2=6.1e-08 cosine=1
qwen4_flash_validation check=gated_norm     relative_l2=1.66e-3 cosine=0.999998615
qwen4_flash_validation check=attn_decode    relative_l2=1.68e-3 cosine=0.999998589
qwen4_flash_validation check=gdn_chunk_output relative_l2=1.66e-3 cosine=0.99999863
qwen4_flash_validation check=gdn_chunk_state  relative_l2=1.1e-07 cosine=1
qwen4_flash_validation check=module_admit_snapshot admit=ok snapshot=ok
qwen4_flash_validation check=module_decode_vs_prefill elements=2560 relative_l2=1.66e-3 cosine=0.99999863
qwen4_flash_validation check=module_determinism bit_exact=1
qwen4_flash_validation PASS
```

Synthetic-pack tool fix included: the max-lineage synthesizer wrote FP8
payloads without the F32B128 scale plane (scale_group_size 0) — the loader
correctly rejected it; fixed to emit group-128 scales near 1.0.

### M4 Packer + verifier — DONE for the 4-layer real pack; full TP4 building (commit fef7335)

`tools/qwen4_flash_stagepack.py`: BF16 warm checkpoint → qwen4_flash packs.
Routed experts quantized at pack time to FP8 E4M3 block-128 (F32 scale
plane, format 4 — the codec the module validates; `--expert-format
fp8-e8m0b128` emits the format-6 MX plane for the module format bump).
Fused gate_up [512,1280,2560] split w1/w3; down flattened expert-major.
TP4 rank plan: q by fused head-blocks, k/v replicated (kv_heads 2 < 4),
o/gdn-out columns, gdn qkv/conv triple-sliced by head groups, beta/decay/
A_log/dt_bias by value heads, experts 128/rank, shared expert by
intermediate, vocab-blocks for embed/lm_head, norms + MTP globals
replicated, MTP layer kinds slicing like main layers per the family rule.

The Flash checkpoint has no plain per-sublayer layernorms and no final
model.norm (hyper-connections carry that role); the [1,H] slots are filled
from the corresponding hc_norm stream-0 sections and MTP_FC from
column-concat(fc_embedding, fc_hidden) — all enumerated in the receipt under
`hc_approximations`, and the unmapped classes (per-layer hyper-connection
mixers, attention indexer, layer-1 PLE, vision tower) under
`unmapped_checkpoint_tensors`.

E4M3 encoder: the first cut had a binade off-by-one (power used the field
exponent), no zero case, and broken subnormals; rewritten and edge-tested
(0 → 0, 2^-9 exact, 2^-10 → 0 RNE, ±448 saturation).

`tools/qwen4_flash_pack_verify.py`: header geometry vs pinned constants,
directory vs the TP-narrowed inventory, byte formulas, offsets/alignment,
receipt sha256, and byte-trace sampling (BF16 bit-exact vs checkpoint
slices; FP8 dequant-with-own-scales within 0.2 relative L2).

**Exit evidence** (spark5, real weights):

```
python3 tools/qwen4_flash_pack_verify.py \
  --pack build/stagepacks/qwen4_flash_slice0p4.tp1.qwen4_flashsp \
  --checkpoint /mnt/model-warm/qwen3.8-flash-next --sample 12
  trace kind=8 layer=0 fp8 relative_l2=0.02645
  trace kind=7 layer=1 fp8 relative_l2=0.02646
  trace kind=6 layer=2 fp8 relative_l2=0.02646
  PASS qwen4_flash_slice0p4.tp1.qwen4_flashsp: header geometry, 74 directory
       entries (tp 1/0), 12 byte-traced samples receipt=verified
```

Dry runs: whole-stack TP1 = 899 tensors, 122.93 GiB; TP4 = 4 × 30.80 GiB
(30.8 GiB/rank ≈ the ~42GB/rank class once the unmapped hc/indexer/ple
tensors eventually land). The 4-rank build+verify loop is running in
background on spark5 (/tmp/q4f_build_packs.log); results to be appended.

### M5–M7 — NOT STARTED (blockers below)

## Honest negatives / open items

  * The architecture is NOT the pure sibling the brief describes. Beyond
    the 3:1 hybrid + MoE shape, the checkpoint adds: hyper-connection
    residual mixing on every sublayer and at the stack readout, an
    attention indexer on all 12 full-attention layers + MTP, and a PLE
    n-gram embedding block on layer 1. The module validates and serves the
    qwen38_max-shaped subset; the extra classes are receipted and carried
    as integration requests.
  * The module's whole-stack MTP draft chain (MTP_DRAFT_AFTER frames) is
    not implemented in the max lineage (RunDecode ignores mtp_draft);
    porting the 27b's RunMtpDraftChain is the M5 prerequisite.
  * The module's TP launcher requires KV_HEAD_COUNT % tp_degree == 0;
    Flash kv=2 forces the replicated-KV plan the packer emits.
  * qwen38_max module on main does not build (see M3 list) — worth a
    coordinator fix or removal notice.
  * Code-size ratchet moved 175507 → 184913 with a justification comment
    (new family port) in the same commit.

## INTEGRATION REQUEST

  1. tests wiring: tests/test_qwen4_flash_model_header.py is standalone
     (`python3 tests/test_qwen4_flash_model_header.py`); the root Makefile
     TEST list + sources.mk would register it (files outside my write set).
  2. Hyper-connection residual path: kernels + module plumbing for the
     4-stream hc mixers (per-sublayer up/down [10240,320], block_inject
     [4,10240], global mixer readout). Blocks faithful M5+ numerics.
  3. Attention indexer (DSV4-flash family, indexer_* config) for the 12
     full-attention layers + MTP: cache layout + budget-2048 selection.
  4. PLE n-gram embedding block at layer 1 (137 tensors).
  5. Module format bump for format-6 (FP8_E4M3_E8M0B128) expert packs:
     packer support exists (`--expert-format fp8-e8m0b128`); the module's
     natural format + entry validation currently pin F32B128.
  6. TP pack acceptance: the module's pack loader validates full-width
     shapes only; rank-local packs need the 27b-style TPD shard plan
     (my verifier validates the rank packs against the documented plan).

## Next

  1. Finish M1 when the freeze lands (contract JSON + verify PASS + M2 test
     green), commit.
  2. Append the 4-rank pack verify results (M4 full exit).
  3. Port the 27b MTP draft chain for M5; build the whole-stack TP1 real
     pack (122.93 GiB, ~2 h at observed throughput) and run the
     whole-stack validator on spark5.
