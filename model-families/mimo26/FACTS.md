# MiMo 2.6 family facts (lane 7)

Measured, not inferred: every number below cites the shard-header census of
the warm checkpoints (`census/pro_census.json`, `census/flash_census.json`,
`census/distill_census.json`, captured 2026-09-22 on spark3 via queue jobs
mimo26-census-run1/run2/run5; the walk reads safetensors headers only, peak
RSS 155 MiB, and cross-checks `model.safetensors.index.json` exactly:
160,040 = 160,040 tensors pro, 73,081 = 73,081 flash).

## Checkpoints

| arm | checkpoint | shards | tensors | payload | provenance |
| --- | --- | --- | --- | --- | --- |
| pro (primary) | mimo-v2.6-pro-rl | 130 | 160,040 | 527.155 GiB | DOWNLOAD-RECEIPT 573,492,067,324 B, 137/155 file shas pinned |
| flash | mimo-v2.6-flash-rl | 65 | 73,081 | 161.047 GiB | same receipt scheme |
| distill (reference) | mimo-v2.6-distill-qwen-9b | 4 | 760 (all BF16) | 17.526 GiB | Qwen3_5 dense hybrid, priority 3 |

Sharding: `model_pp0_ep{N}_shard{S}.safetensors` - one pp stage; pro ep file
N (1..127) carries experts {3N, 3N+1, 3N+2} for every MoE layer (128 files x
3 = 384), flash ep file N (1..63) carries {4N..4N+3} (64 files x 4 = 256);
pro ep0 = shard0 (spine + experts 0-2) + shard1 (vision/audio/speech towers),
flash ep0 = a single shard0 (spine + towers + experts 0-3); `model_mtp.safetensors`
= MTP head; `dflash/` = a 5.5 GB draft model (speculation is the parallel
project, not this lane).

## Geometry (text tower; mimo25 is the precedent family)

| fact | pro | flash |
| --- | --- | --- |
| hidden / layers / vocab / ctx | 6144 / 70 / 152576 / 1M | 4096 / 48 / 152576 / 1M |
| rms eps | 1e-5 | **1e-6** (mimo25 base was 1e-5) |
| heads x head_dim / v_dim | 128 x 192 / 128 | 64 x 192 / 128 |
| KV heads full / SWA | 8 / 8 | 4 / 8 |
| full-attention layers | 10: 0,7,15,23,31,39,47,55,**62,69** | 9: 0,5,11,...,47 |
| qkv rows full / SWA | 27136 / 27136 | 13568 / **14848** |
| qkv scale_inv rows | 216 / 216 | 108 / 116 |
| rope | partial 0.334 -> 64 of 192, half-split | same |
| value scale (folded pre-cache) | 0.612 | 0.707 |
| SWA window / sink bias | 128 / SWA-only `[heads]` | 128 / SWA-only |
| dense layer 0 intermediate | 16384 | 16384 |
| MoE | 384 experts top-8, inter 2048, no shared | 256 experts top-8, inter 2048 |
| router | bf16 sigmoid + noaux_tc e_score_correction_bias, norm_topk, eps 1e-20 | same |
| MTP head | 3 layers, F32 storage, 2.294 GiB | 3 layers, 1.108 GiB |

The pro full-attention set breaks the every-8 rhythm at the tail (62 and 69,
not 63-only) and flash keeps mimo25's per-kind KV split - both facts are
pinned by `tests/test_mimo26_census.py`, which binds the headers, contracts
and must-work registration to the census byte counts.

## Codecs (the quality law, as measured)

- Routed experts: **MXFP4** - e2m1 weights (U8, two 4-bit elements per byte)
  + e8m0 scales (U8, one byte per 32-element block, unpacked): gate/up
  `[I, K/2]` + `[I, K/32]`, down `[H, I/2]` + `[H, I/32]`. pro 465.750 +
  29.109 = 494.859 GiB; flash 141.000 + 8.812 = 149.812 GiB. Never
  requantized.
- Spine linear: fp8 e4m3 + F32 `weight_scale_inv` per [128,128] block
  (qkv_proj, layer-0 dense MLP); qkv scale grids carry upstream row padding
  (measured constants above; not a clean closed form).
- **BF16, byte-identical to source**: every `o_proj` (the checkpoint's
  `ignored_layers` is exactly all layer o_proj + the MTP decoder o_proj),
  embeddings, lm_head, all norms, sink biases, router gates.
- MTP head tensors are stored F32 (weights and scale-grids alike); eh_proj
  and norms BF16.

## Scope ruling (v1 driver)

v1 = non-speculative **text tower** decode. Text-only tokens never route
through the vision/audio towers, so the towers are out of scope for v1
serving; the MTP head is speculation territory (parallel project). None of
these regions are stripped from packs silently: the census records their
per-tensor byte ranges (`out_of_scope_regions`), and the stagepack emits the
served set by explicit inclusion, so a later MTP/multimodal decision is a
delta against recorded facts, not a rediscovery.

## Topology (tools/mimo26_param_budget.py, census-driven)

Per-node budgets TOTAL 9792 MiB / DEVICE 6400 MiB; lane envelope 8 Sparks
(78,336 MiB); shared weightd arena 28.5 GiB device per node fleet-wide
(bounded working sets only - a whole rank pack never attaches).

- **pro -> TP8**: rank pack 65.6 GiB NVMe (spine 3.78 GiB resident device +
  61.9 GiB experts), 8 nodes = exactly the lane envelope; spine+KV32k leaves
  ~2.3 GiB device headroom. TP16 fits device but doubles the envelope; TP4
  overflows DEVICE_MIB with spine+KV.
- **flash -> TP4**: rank pack 39.6 GiB NVMe, device spine 2.14 GiB, half the
  envelope; TP8 stays open as a co-residency fallback.
- Decode floor at B1 (native codecs, 218 GB/s effective): pro 38.4
  GiB/token -> ~5.3 tok/s; flash 13.0 GiB/token -> ~15.7 tok/s. Batch
  amortises the expert read; the spine read is the fixed cost.

## Stagepack wire (M2, tools/mimo26_stagepack.py)

Arms: `mimo26pro.mxfp4.tp8` and `mimo26flash.mxfp4.tp4` under
`~/sparkdata/<arm>/packs/`. Wire: 120-byte 26I2Q header + 56-byte 6I4Q
entries (kind, layer, format, rows, cols, reserved, payload_offset,
payload_bytes, scale_offset, scale_bytes) + 256-aligned payload/scale planes,
magic 'M26P'. Weight codes: BF16/F32 shared codes, fp8 e4m3 block-128 = 4,
**mxfp4 e2m1+e8m0 g32 = 9** (added to include/sparkpipe/spark_stagepack_format.h).
Slicing: q row-sliced by head groups, k/v sections replicated whole (kv-head
granules cannot cut the fp8 grid; ~16 MB/layer cost), o_proj col-sliced,
embed/lm_head vocab-row-sliced, router/norms replicated, sink head-sliced,
experts per-rank disjoint slabs expert-major. The fused scale grids' padding
rows are measured (216/216, 108/116), sliced on block boundaries and dropped
where they are not data. Emission is staged/resumable (`--emit`/`--assemble`
/`--verify`, `--layer-window FIRST:COUNT` for TTL-bounded fanout); verify
byte-compares every plane against the checkpoint.

### Tensor kinds, header fields and per-rank census

`modules/mimo26_resident_decode_stage/source/spark_mimo26_stagepack_format.h`
is the C side of the M26P wire. The values of `SparkMimo26StagePackTensorKind`
equal the `KIND_*` constants in `tools/mimo26_stagepack.py`.

**Tensor kinds.**

- **Kinds 0-2** (embedding, final norm, LM head) are the global tensors
  (`SparkMimo26StagePackIsGlobal`). They are not part of the shared
  `SparkStagePackCommonTensorKind`, but they use the same values as
  qwen4_flash.
- **Kinds 3-5** (attention norm, MLP norm, MoE router gate) have the values
  of the shared `SparkStagePackCommonTensorKind`. mimo26 does not use the
  shared kinds 6-21.
- **Kinds 22-33** are the family block: sink bias, router gate bias, q, k,
  v, o_proj, the three dense-MLP tensors, and the three expert slabs at
  31-33 (`SparkMimo26StagePackIsExpert`).
  `SPARK_MIMO26_STAGEPACK_TENSOR_KIND_COUNT` is 34.

**Header and entry fields.**

- `SparkMimo26StagePackHeader` matches `SparkStagePackHeaderCommon` field
  for field, which `SPARK_STAGEPACK_HEADER_LAYOUT_PROOF` checks.
- The full/SWA layer pattern is irregular, so the tool writes 0 for
  `attention_period`, `full_attention_phase` and the GDN fields.
- Per-layer kinds come from compile-time tables in the family model headers:
  `SPARK_MIMO26_PRO_MODEL_LAYER_KIND` and `SPARK_MIMO26_MODEL_LAYER_KIND`,
  plus the matching `_LAYER_IS_MOE` tables. `tests/test_mimo26_census.py`
  checks each table against the census (`hybrid_layer_pattern`,
  `moe_layer_freq`).
- The sixth entry field is `scale_group_size` in the C struct, and the tool
  writes 0 there. The MXFP4 group size (32) travels in the header's
  `mxfp4_group_size`.

**Census.** `SparkMimo26StagePackExpectedTensorCount(layer_count,
swa_layer_count, moe_layer_count)` gives the number of directory entries in
one rank pack. The count does not depend on the TP degree:

- 3 globals;
- plus 6 per layer (attention norm, MLP norm, q, k, v, o_proj);
- plus 1 per SWA layer (sink bias);
- plus 5 per MoE layer (router gate, router bias, three expert slabs);
- plus 3 per dense layer (dense-MLP gate, up and down).

This gives 831 for pro (70 layers: 60 SWA, 69 MoE) and 568 for flash (48
layers: 39 SWA, 47 MoE). `tests/test_mimo26_stagepack_format.py` pins both
values.

No mimo26 loader exists, so nothing enforces the count at load time.
`tools/mimo26_stagepack.py --verify` compares the pack's `tensor_count` with
the tool's own plan.

## Registration

- Geometry headers: `include/sparkpipe/spark_mimo26_pro_model.h`,
  `spark_mimo26_model.h` (flash).
- Contracts: `model_contracts/mimo26_pro_authoritative.json`,
  `mimo26_flash_authoritative.json` (census-pinned).
- Must-work targets: `mimo26_pro_mxfp4_experts_fp8_spine`,
  `mimo26_flash_mxfp4_experts_fp8_spine` (tests updated in the same PR).
- Binding test: `tests/test_mimo26_census.py`.

## First correct tokens (2026-09-28, single GPU, partial residency)

`modules/mimo26_resident_decode_stage/validation/mimo26_model_cuda.cu` decodes the whole
Flash text stack on one GB10 using the shared kernels. It uses skinny fp8 for qkv and the
layer-0 dense MLP, skinny bf16 for o_proj, the router and lm_head, skinny MXFP4 for the
experts, `LmGqaKvStoreKernel`, `LmGqaAttentionDecodeKernel` on the full layers and
`LmGqaSinkAttentionDecodeKernel` on the SWA layers. Its own small kernels handle the
per-rank qkv de-interleave with partial rope and v scale, SwiGLU, and the weighted
combine. Routing runs on the host.

The resident expert set is the one the CPU reference routed through for the prompt
(35-45 GiB). Any other expert the GPU routes to is demand-loaded from local NVMe and
counted.

`validate_mimo26_model_cuda.sh <checkpoint> <nvme work dir> capital code science` builds
the binary, stages the inputs (`tools/mimo26_model_inputs.py`, about 160 GB on the first
prompt, then symlinks for the others) and runs each prompt.

Routing uses the shared `LmTopkSmallKernel<..., 8, renormalise, 1, 1, LM_TOPK_SCORE_SIGMOID>`
with the correction bias, which is the noaux_tc rule. A second pass replays the same
positions with no host synchronisation inside the layer loop; any non-resident expert
fails it through a device counter.

Measured on sparkf, with production GLM resident, non-speculative, B1, greedy:

| prompt | tokens equal to the CPU reference | demand-loaded experts | wall (load + 20 positions) |
| --- | --- | --- | --- |
| capital | 16 / 16 | 40 | about 60 s |
| code | 16 / 16 | 70 | about 80 s |
| science | 16 / 16 | 50 | about 55 s |

Device-routed replay: 71.6 / 72.9 / 73.1 ms per position (about 13.8 tok/s at B1 on one
GB10 with every routed expert resident). The logits come back to the host for the argmax
once per token. The replay tokens equal the first pass in all three prompts.

`--tp4-slices` runs the TP4 decomposition the resident module needs, on the same GPU:
- per-rank fused QKV segment (the v2 pack's QKV entry) with 16 q heads and a rank-local
  KV cache (1 full / 2 SWA heads);
- per-rank o_proj column slice;
- per-rank partial sums of the routed experts each rank owns (64 per rank);
- bf16 partials summed in f32, the all-reduce.

It gives 16/16 tokens equal to the reference for all three prompts, with the
device-routed replay at 77.1 / 78.2 / 77.3 ms per position. The layer-0 dense MLP is
still computed unsliced.

Numerics (capital prompt, `--teacher-forced` feeds every layer the reference's input
stream, and route decisions are compared with the fixture):

- Teacher-forced, every layer matches the reference to 1e-3 to 5e-3 relative L2 at
  positions 1-5. Only 3 of 235 top-8 sets differ, each by one near-tie expert (8th vs
  9th choice margin 5e-6 to 2e-4). The GPU layers are faithful.
- Free-running, 122 of 940 sets differ by one expert (margins 1e-5 to 2e-2) and the
  layer-47 error grows to 1.5e-2 to 5e-2. The tokens still agree 16/16.
- Position 0 is the massive-activation token. Its relative error is 3e-5 on layers
  17-40 and 7e-2 on the full-attention layer 41 even when teacher-forced. That is
  cancellation of the huge dims, not a layout error. The free-running 0.39 at
  position 0 layer 47 comes from this.

The 2 percent layer-47 band of `docs/T1_REFERENCE_COMPARE.md` therefore needs either
teacher-forced comparison or a route-aware band for this family.
