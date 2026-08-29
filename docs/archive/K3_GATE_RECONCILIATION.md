# K3 gate reconciliation — released checkpoint vs pack V2 + driver

Pinned 2026-08-15 against moonshotai/Kimi-K3-MXFP4 (revision on disk in
/home/<user>/srcdata/kimi_k3.mxfp4.pp13), shard headers read directly.

## The drift

The packer (tools/k3_pack.py), the V2 format doc (docs/K3_PACK_FORMAT_V2.md)
and the driver gate paths were written for the LOW-RANK output gate
(g_a_proj to a 128-wide latent, g_b_proj up). The RELEASED checkpoint ships
a FULL-RANK gate only:

| Tensor | Released checkpoint shape | Old expectation |
| --- | --- | --- |
| KDA self_attn.g_proj.weight | [12288, 7168] = heads*head_dim x hidden | g_a_proj [128, hidden] + g_b_proj [12288, 128] (absent) |
| MLA self_attn.g_proj.weight | [12288, 7168] = heads*v_head x hidden | g_a_proj [v_head, hidden] + g_b_proj (absent) |

config.json text_config: use_full_rank_gate: true; the contract's
kda.full_rank_output_gate: true matches the released checkpoint, not the
old packer.

Everything ELSE about the KDA/MLA mapping reconciles cleanly: q/k/v/b_proj,
f_a_proj [128, hidden], f_b_proj [12288, 128], qkv_conv1d, A_log, dt_bias,
o_norm, o_proj [7168, 12288], the MLA q/kv low-rank tensors and the
language_model.model. prefix. The layer-0 pack reached the gate tensor
before failing, so the prefix/config/slice fixes are validated.

## Required changes

1. **Packer** (tools/k3_pack.py):
   - KDA: keep the decay tensors as their own pair (kda_decay_down_weight =
     f_a [128, hidden], replicated class; kda_decay_up_weight = f_b
     [12288, 128], output_dim_heads class). Emit kda_gate_weight = g_proj
     [12288, 7168] (output_dim_heads class). Remove the
     kda_decay_gate_down fusion and kda_gate_up.
   - MLA: emit mla_gate_weight = g_proj [12288, 7168] (output_dim_heads).
     Remove mla_gate_down/up.
2. **Driver** (inference/llms/kimi_k3/layer.cuh): replace the two-stage
   latent gate with one full-rank projection from normed_bf16 for KDA and
   from attention_out (v-space) for MLA; K3_KDA_FULL_RANK_GATE is already
   1 in config.h but the layer paths still consume the low-rank tensors.
3. **Format doc** (docs/K3_PACK_FORMAT_V2.md): update the kda_fused
   sections table (decay_gate_down fusion is gone; single full-rank gate
   tensors) and the manifest shard classes.
4. **k3_shard.py**: the new gate tensors take the output_dim_heads class
   (same split as qkv_beta), so the shard table needs the two new names.
