# qwen-image-2.1 — architecture record (driver-class: FLOW-MATCHING DIFFUSION)

Source: /mnt/model-warm/qwen-image-2.1 (PUBLISHED marker, configs + index landed
2026-09-21; transformer 14G/2 shards, vae 1.3G, text_encoder 17G). This is the
fleet's FIRST non-autoregressive family: the serving loop is the scheduler's
denoise iteration over full latent frames — no KV cache, no token decode. The
resident_decode_stage pattern applies to PACK LOADING and kernel residency; the
runtime loop belongs to the scheduler (FlowMatchEulerDiscreteScheduler).

## Components

| component | class | geometry |
|---|---|---|
| transformer | diffusers.QwenImage21Transformer2DModel | 32 dual-stream blocks, 32 heads x 128 (=4096 hidden), in/out 64 latent channels, patch 1, mlp_ratio 3, rope axes [16,56,56], context_in 4096, eps 1e-6, causal_condition true; ~297 tensors, ~7B BF16 |
| text_encoder | transformers.Qwen3VLForConditionalGeneration | 17G — the same Qwen3VL tower class as minimax-h3's text encoder |
| vae | diffusers.AutoencoderKLQwenImage21 | base 96, decoder 144, z_dim 64, in/out 4, spatial scale 16, residual |
| scheduler | diffusers.FlowMatchEulerDiscreteScheduler | flow-matching Euler |
| processor | transformers.Qwen3VLProcessor | prompts |

## Tensor census (transformer)

Emitted by tools/qwen_image_census.py from the shard index at build time.
Observed patterns: per-block img_attn (norm_q/norm_k/to_q/k/v/out) +
img_mlp (proj/gate_layer/out) + modulation + txt stream + txt_in + norm_out +
proj_out + time_text_embed. 297 tensors total.

## Driver-class decision (recorded)

The minimax-h3 video stack is the SAME class (Qwen3VL text encoder + DiT +
VAE + scheduler): ONE diffusion resident stage architecture serves both
families. The autoregressive resident_decode_stage pattern contributes pack
loading and kernel residency; the denoise loop, scheduler integration, and
VAE boundary are new stage shapes defined in the driver design (to follow).
