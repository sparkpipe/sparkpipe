# Pack-emission wave 3 — per-layer codecs (laguna) + nvidia nvfp4 convention (dsv41) — 2026-09-21

Mission: remove the two BLOCKED-MODULE codec limitations left by waves 1-2 so
the last unplaced quants can emit, then emit + place + verify them.

Branch: `lane/module-codec-wave3`. CPU only; emissions ran on sparkc under
`~/codec-wave3/` (read-only over `/mnt/model-warm`); the CUDA gate ran on
sparkb with output + logs under `~/codec-wave3/build` (outside any tree).

## Blocker 1 — laguna mixed-dtype routed experts: FIXED

The releases quantize routed experts only on layers 1-43 (fp8 arm) / 1-39
(nvfp4 arm); later routed layers stay BF16 (sensitivity-preserved by the
publisher — as important as the spine, so NO requantization to uniform). The
wave-2 census variants existed, but the module pinned ONE expert codec per
pack. This wave:

- module (`spark_laguna_resident_decode_stage_*`): mixed header sentinel 8;
  per-layer codec table recorded entry-by-entry at load
  (`SparkLagunaModuleEntryExpertCodec` — first expert entry pins the layer's
  codec, a disagreeing entry is a SCHEMA_ERROR); shape/byte expectations and
  manifest planes resolve through the recorded per-layer codec; runtime MoE
  dispatch under mixed selects bf16/fp8/nvfp4 instantiations per layer
  (unity.cu static-asserts all three decode to the BF16 MMA geometry).
- packer (`tools/laguna_stagepack.py`): `--expert-codec mixed` resolves each
  routed layer's NATIVE wire from the checkpoint dtypes (U8 `*_packed` ->
  nvfp4, F8_E4M3 -> fp8, BF16 -> bf16 passthrough) and emits it verbatim;
  per-layer codec receipt (`expert_codec_by_layer`, `expert_codec_classes`);
  `--verify PACK` re-opens an emitted pack and elementwise-pins it against
  the warm source PER LAYER CLASS (BF16 spine + BF16 routed layers
  byte-exact; fp8/nvfp4 payload+scale+global bytes verbatim).
- NVFP4 per-expert scale slab pinned precisely (this wave's correction):
  `[gate plane][up plane][gate weight_global_scale][up weight_global_scale]`
  — one 4-byte F32 global PER PROJECTION (the fused gate_up slab ends in 8
  global bytes, the down slab in 4). `SparkLagunaStagePackExpectedScaleBytesForKind`
  now counts globals per projection; the first real nvfp4 emission crashed
  on the +4/+8 mismatch and the twins + module now pin the same math.
- CPU twins (`make test-module-codec-wave3`): the synthesized mixed laguna
  pack is accepted layer-by-layer by the mixed build; 30/30 mutated expert
  entries rejected; coverage rejects a removed expert entry; the uniform
  fp8 build rejects the mixed header (code 6). Synthesized nvfp4 dsv41 pack
  accepted entry-by-entry by the nvfp4 build; the mxfp4 build rejects all
  12 expert entries; the mxfp4 wire is unchanged.

## Blocker 2 — dsv41 NVIDIA nvfp4 convention: FIXED

Wave 2 proved the reader (`validate-nvidia`: 184,320 tensors = 40L x 384
experts x 3 proj x 4 tensors; U8 e2m1 payload + F8_E4M3 per-16 plane + two
F32 scalars) but the module accepted mxfp4/fp8 only. This wave:

- module (`spark_dsv41_flash_stagepack_format.h` +
  `spark_dsv41_flash_resident_decode_stage_module.c`):
  `SparkDsv41FlashStagePackShapeNvfp4` (codec 6,
  `UE4M3_F32_GLOBAL` scale encoding, the family's packed-column convention)
  in the EXPERT_W1/W2/W3 branches of `SparkDsv41FlashStagePackExpectedShape`;
  the expert-manifest codec whitelist gains NVFP4_E2M1; the module Makefile
  accepts `EXPERT_CODEC=nvfp4` (codec id 6). Per-expert scale slab =
  `rows*(cols*2/16)` e4m3 bytes + 4-byte `weight_scale_2` global
  (`SparkDsv41FlashStagePackNvfp4ScaleBytesPerGroup`), mirroring the
  t1-measured dequant `e2m1(payload, low nibble first) x e4m3(plane) x
  weight_scale_2` (identical to the proven gemma4 nvfp4 reader;
  `input_scale` is activation-side and never enters the weight pack).
- packer (`tools/dsv41_flash_stagepack.py`): `plan ... nvfp4` emits the
  nvidia wire with payload/plane/global copied VERBATIM; the checkpoint
  accounting closes with a new `activation_scales_out_of_scope` bucket
  (40x384x3 `input_scale` tensors).
- byte verifier (`tools/dsv41_flash_pack_verify.py`): re-derives the nvfp4
  layout independently and byte-compares EVERY payload/plane/global byte;
  the mechanical verifier (`tools/dsv41_verify_pack.py`) accepts the codec-6
  header. The python self-test runs BOTH wires end-to-end on tiny synthetic
  checkpoints and proves corruption is localized in spine, expert payload,
  scale plane, and global.

## Compile gate (sparkb, CUDA 13.0, sm_121a)

- `PATH=/usr/local/cuda/bin:$PATH SPARK_CUDA_GATE_OUTPUT_DIRECTORY=$HOME/codec-wave3/build tools/cuda13_sm121a_compile_gate.sh`
  -> `PASS CUDA 13 exact sm_121a compile gate` (log
  `sparkb:~/codec-wave3/gate.log`, receipt dir `sparkb:~/codec-wave3/build`
  with per-object PTX/ELF/resource logs + SHA256SUMS).
- Wave-3 module variants compiled on top of the gate, same toolchain
  (rc=0, logs under `sparkb:~/codec-wave3/build/module-variants/`):
  - `make -C modules/laguna_resident_decode_stage archive EXPERT_CODEC=mixed
    MODEL_REVISION=0f573140... CONTRACT_SHA256=8e9e82f9...` — the per-layer
    dispatch build (bf16/fp8/nvfp4 instantiations) compiles.
  - `make -C modules/dsv41_flash_resident_decode_stage archive
    EXPERT_CODEC=nvfp4 MODEL_REVISION=050c4552...
    CONTRACT_SHA256=44fcba0b...`.

## Arms

| # | warm quant | arm | verdict |
|---|---|---|---|
| A | laguna-s-2.1-fp8 | laguna-s-2.1.fp8-mixed.tp8pp2 | see below |
| B | laguna-s-2.1-nvfp4 | laguna-s-2.1.nvfp4-mixed.tp8pp2 | see below |
| C | deepseek-v4.1-flash-nvfp4-nvidia | dsv41flash.nvfp4.tp8 | see below |

Per-rank shas, verify receipts, and placement receipts are appended below as
measured (the ledger is regenerated by the wave; "verify match" =
`tools/verify_package_manifest.py` PASS over the final tree).

## EXECUTION-GATED runtime items (honesty law)

The dsv41_flash module's Execute path is host-contract only today (load
validation + lazy-pack manifest; `Execute` returns UNSUPPORTED), and the
laguna mixed runtime dispatch compiles but has no device golden on this
wave. A GPU run must verify:

1. laguna mixed: per-layer MoE outputs for one bf16-preserved layer (44-47)
   and one quantized layer match `t1_reference_laguna`'s golden decode
   (BF16 passthrough is byte-exact by construction; fp8/nvfp4 dequant
   against the released payload/scale/global bytes).
2. dsv41 nvfp4: expert matvec vs `e2m1(low-nibble-first) x e4m3(per-16
   plane) x weight_scale_2` golden (the gemma4 nvfp4 reader's math), and
   the activation path consumes `input_scale` from the sidecar source.
