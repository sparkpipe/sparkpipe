# TICKET: flash module NVFP4 resident-decode enablement (owner: coordinator)

## Goal
The flash fp8 TP8 + TP4xPP4 arms are placed (16/16 each, wire code 8 =
SPARK_STAGEPACK_FORMAT_WEIGHT_NVFP4_PACKED, scale_group 16, receipts,
chattr ring). The qwen4_flash module accepts the wire at entry validation
(bdc20cf) but the RESIDENT DECODE (bind + kernels) must be enabled for
serving. This ticket is the complete implementation spec.

## Facts
- Source of record: glm-5.3-flash-nvfp4-redhatai (modelopt NVFP4,
  group-16, per-row scales; experts SPLIT per-expert: gate/up/down_proj
  .weight (U8 packed 2/byte) + .weight_scale (F8_E4M3 [rows, cols/16]) +
  .input_global_scale (F32) + .weight_global_scale (F32)).
- Packer: tools/qwen4_flash_stagepack.py --expert-format nvfp4-official
  (dry-run GREEN: 1246 tensors, 22.5G/rank @ TP8). Scale region layout
  per tensor: [rows x cols/16 e4m3 bytes] + input_global(F32) +
  weight_global(F32) appended.
- Shared kernel machinery EXISTS: inference/kernels/weight_codec.cuh:29
  declares the NVFP4 trait (LmNvfp4, UE4M3_F32_GLOBAL); the glm5_next MoE
  kernel is codec-templated (glm5_next .../cuda/layer.cuh:2055
  Glm5NextLayerMoe<ExpertCodec>) and SERVES GLM-5.2-NVFP4 (the precedent).
- The flash module validates wire 8 (bdc20cf) but its MoE bind + kernels
  hardcode MXFP4_E2M1 + e8m0-per-32 (cuda.cu:2199,2437,2509).

## Work
1. module.c BindMoe: the nvfp4 wire cases bind the payload + the e4m3
   scale plane + the 2 F32 globals into the linear views.
2. cuda.cu: the MoE path branches on the wire: nvfp4 = the per-16 e4m3
   scale decode (UE4M3_F32_GLOBAL) in the grouped-expert kernels.
3. Rebuild the flash module with the nvfp4 codec enabled (the glm5_next
   Makefile EXPERT_CODEC=nvfp4 pattern).
4. GATE: smoke load a wire-8 rank; the decode output must match the
   numpy reference dequant (tools/qwen4_flash_stagepack.py LUT path is
   the reference). Mismatch = scope the delta; do NOT serve.

## Non-goals
- No BF16-wire fallback serving for the nvfp4 arm (the wire exists; the
  decode must be real).
- The 27B nvfp4a16 arm follows this ticket verbatim (the same modelopt
  format; the a16 spine = BF16 activations).
