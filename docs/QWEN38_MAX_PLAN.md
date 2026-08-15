# Qwen3.8-2.4T-A95B (Qwen 3.8 Max) TP4xPP4 bring-up plan

Branch: `qwen38-max` (single-writer for all Qwen 3.8 Max files).
Owner: this session. Common-code policy: consume shared kernels unchanged;
any common-layer edit is a flagged, separately-reviewable commit.

## Facts

- Checkpoint: Hugging Face `Qwen/Qwen3.8-2.4T-A95B`, revision
  `207bd685a7e3696cfaff12ded7c6a7ea0f88c996`, bf16, 4.89 TB, 213 shards.
- Geometry: 92 layers = 23 x (3 GatedDeltaNet+MoE, 1 GatedAttention+MoE);
  hidden 8192; vocab 248320; ctx 262144 native (1M extensible).
- GDN: 16 QK / 128 V heads, dim 128, conv kernel 4. Attention: 64 Q / 4 KV
  heads, dim 256, rope 64, output gate. MoE: 512 routed experts, top-10,
  one shared expert (learned `shared_expert_gate`), intermediate 2048,
  fused `gate_up_proj` in the checkpoint. MTP: one decoder layer.
- Pack policy (quality-first, mirrors DSV4 Flash): routed experts
  MXFP4-E2M1, non-expert weights/activations/KV/head BF16, accumulation
  and GDN state FP32. `model_contracts/qwen38_authoritative.json` pins it.

## State

- [x] Contract written (model_contracts/qwen38_authoritative.json).
- [x] Checkpoint source: the LOCAL FP8 release at
      /home/spark9/extnvme/models/hf/Qwen/Qwen3.8-2.4T-A95B-FP8 (2.3 TB,
      224/224 files hash-verified, vendor block-128 FP8 experts + BF16
      spine). Redistributed from spark9 to each spark's INTERNAL NVMe at
      /home/<user>/sparkdata/qwen38_2.4t_a95b/checkpoint by PP-stage layer
      slices (54/53/52/54 shards). The earlier HF bf16 download was stopped
      and its partials removed.
- [x] qwen38 stage packer (tools/qwen38_stagepack.py): whole-PP-stage slices,
      routed experts now packed in the VENDOR FP8 form (FP8_E4M3 payload +
      F32 block-128 scales, codec code 4) with BF16 non-expert; inventory and
      shapes verified against the pinned FP8 index. TP4 rank slicing still
      comes with the TP4xPP4 deployment work.
- [x] qwen38 resident decode stage module + serving adapter + firmware JSON.
      Module compiles clean (Makefile archive target). Real FP8 packs execute
      a two-step decode on a single spark, compute-sanitizer clean: the
      1-layer GDN pack (layer 1, 24.9 GiB) and the 2-layer GDN+attention
      pack (layers 2-3, 49.8 GiB) - both layer kinds and both MoE paths on
      real weights. Adapter identifiers now tp4-pp4 / fp8 / 16 ranks; the
      TP-rank frame mapping inside the adapter is flagged outstanding.
- [x] inference/llms/qwen_3_8 driver family (config.h / layer.cuh / bind.cu /
      unity.cu): GDN + gated attention + routed FP8 MoE + shared expert,
      three family-local kernels (per-head q/k RMSNorm, gated head norm,
      shared-expert gate add); nvcc compile-checked and the layer-kinds gate
      lists it green.
- [x] TP4xPP4 deployment configs: examples/deployments/qwen38_fp8_tp4_pp4_
      {host_rdma.spec,stage}.json (port block 22480/66620, revision
      d2dc3565, ctx 262144) and examples/release/qwen38_tp4_pp4_b1_template/.
- [ ] Torch/HF reference harness + validation gate; exact end-to-end run.
- [ ] TP4 rank-local packs (the packer slices column-parallel inputs and
      shards the hidden dimension per rank; the module dispatches the
      rank-local view). Today the pack is whole-stage; a 16-rank run needs
      four rank-local packs per stage.

Single-spark validation done (no ring needed): pack-load test and the
module execute test both pass on the real FP8 pack; the full TP4xPP4
end-to-end run waits for a fleet window. Build recipe:
`make archive REPOSITORY_ROOT=/home/spark4/sparkpipe-qwen38` in
modules/qwen38_resident_decode_stage, then link
tests/test_qwen38_execute.c against libqwen38_resident_decode_stage.a
with -lcudart -lcuda -lstdc++ -lpthread -lm. The test runs
`test_qwen38_execute PACK` and expects execute[0]/execute[1] status 0.

## Packer plan

Input: the 213 shards gathered per PP stage (intra-stage copies over the
fabric; ~1.2 TB per stage slice, fits one internal NVMe). Output: four
rank-local packs per stage (TP4 sharding), MTP packed but not served.

Tensor map (from the pinned index):
- GDN layer: input_layernorm, linear_attn.{in_proj_qkv (fused q2048|k2048|v16384),
  in_proj_z, in_proj_a, in_proj_b, conv1d, A_log, dt_bias, norm, out_proj}.
- Attention layer: input_layernorm, self_attn.{q_proj,k_proj,v_proj,o_proj,
  q_norm,k_norm}, post_attention_layernorm.
- Every layer: mlp.experts.{gate_up_proj (fused [512,4096,8192], split to
  w1/w3 at pack time), down_proj}, mlp.gate.weight, mlp.shared_expert.*,
  mlp.shared_expert_gate.weight.
- Head: model.norm.weight, lm_head.weight. MTP: mtp.layers.0.*, mtp.fc.weight.

Codec work reuses the existing packers: MXFP4-E2M1 expert packing from the
DSV4 stagepack path (per-group-32 E8M0 scales), bf16 non-expert verbatim
(qwen36 packer convention), router gate bf16 (small).

## Module plan

Clone the qwen36 module (same GDN/attention family, ~7.8k LOC) and:
1. Geometry: hidden 8192, 92 layers, 3:1 phase, 16/128 GDN heads, 64/4
   attention heads.
2. Swap the dense FFN (gate/up/down + SwiGlu) for the routed-MoE path:
   route (gate GEMM + top-10 + shared gate) -> grouped expert W13/W2 using
   the COMMON `SparkLmSm121B1Expert*` and grouped-MoE kernels already in
   model-families/common (no common-code changes expected) -> shared expert
   scaled by shared_expert_gate -> pair reduce into the residual.
3. TP4xPP4: adopt the DSV4 TP4xPP4 execution pattern (collective boundary
   chain, rank-local packs, stage-local graphs).
4. Serving adapter + firmware description per the existing family seam.

## Verification

- Torch reference (HF modeling_qwen3_5 on a spark with GPU) vs module
  outputs for the exact checkpoint, GDN state trajectories, MoE routing,
  and head logits; then the exact end-to-end token gate on the 16-node
  deployment. No approximate parity claim.

## Fleet port registry and measurement exclusivity (operational policy)

Qwen 3.8 Max TP4xPP4 occupies the full-16 fleet slot: control TCP 20480,
collective ports 66620-66623, transport control port 63700, on all sixteen (the next free block; dsv4-pro took 20480/64620/61700 in the fleet registry)
sparks. Ports per rank follow the per-model port block registry; no model
may claim another slot's block.

Coexistence rule: models may be dev-active together (residentd up, idle)
anytime, but a measured window is exclusive on its hosts - pause sibling
residentds there, measure, restore. Because this model's TP4xPP4 window
covers the whole fleet, its measurements are fleet-exclusive slices; the
scheduler owns the timeslice. DSV4 Flash TP4 (spark4-7) and Qwen 27B PP16
(spark0-3) stay resident and pause during my windows.

My workspace is the dedicated worktree /Users/cem/Documents/dsh/qwen38-worktree
(branch qwen38-max-tp4pp4); the shared checkout at ~/Documents/dsh/sparkpipe
is the DSV4 Pro session's tree and is not mine to switch.
