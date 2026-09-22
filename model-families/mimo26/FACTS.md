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

## Registration

- Geometry headers: `include/sparkpipe/spark_mimo26_pro_model.h`,
  `spark_mimo26_model.h` (flash).
- Contracts: `model_contracts/mimo26_pro_authoritative.json`,
  `mimo26_flash_authoritative.json` (census-pinned).
- Must-work targets: `mimo26_pro_mxfp4_experts_fp8_spine`,
  `mimo26_flash_mxfp4_experts_fp8_spine` (tests updated in the same PR).
- Binding test: `tests/test_mimo26_census.py`.
