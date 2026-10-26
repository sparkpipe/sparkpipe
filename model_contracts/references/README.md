# Pinned modeling references (semantics ground truth for kernel ports)

## modeling_axk2.py, modular_axk2.py, configuration_axk2.py, modeling_deepseek_v32.py
- Source: huggingface/transformers @ a672eb5f474b8b30173e9f96eab66adf292da635 (src/transformers/models/axk2/ and models/deepseek_v32/)
- sha256 prefixes: 2f218206b5526d1e (modeling), 6dc3a4509d468852 (modular), 37c78827e2eb6290 (configuration), 214168ee8214f99e (deepseek_v32 base); pinned by model_contracts/axk2_authoritative.json
- Purpose: A.X K2 semantics ground truth for the axk2 driver and tools/axk2_reference.py. One departure: the expert router masks unselected groups with -inf, where these files fill 0.0 (TECHDEBT, A.X K2).

## modeling_solar_open2.py, modular_solar_open2.py, configuration_solar_open2.py, modeling_solar_open.py, fla_kda_gate.py
- Source: upstageAI/transformers @ ef5c37485091057070b36dd0620798f74988eb01 (branch v5.14.1-solar-open2; models/solar_open2/ and models/solar_open/) and fla-org/flash-linear-attention @ 3a59c2ac74b708d5774346cf784f03239f8ab534 (fla/ops/kda/gate.py)
- sha256 pinned by model_contracts/solar_open2_authoritative.json
- Purpose: Solar Open 2 semantics ground truth for the solar_open2 driver and tools/solar_open2_reference.py. The KDA decay follows the fla gate the publisher's fast path calls (lower bound times sigmoid), not the torch fallback (softplus then clamp), which is a different function (TECHDEBT, Solar Open 2).

## modeling_exaone_moe.py, configuration_exaone_moe.py
- Source: nuxlear/transformers @ 969f8bdfc73d7d8bb0df1b426e057d80a1adf501 (LG AI Research fork named by the K-EXAONE 2.0 model card)
- sha256 prefixes: e2b107896dd690aa (modeling), a3218188e7d8e462 (configuration); pinned by model_contracts/kexaone2_authoritative.json
- Purpose: K-EXAONE 2.0 semantics ground truth for the kexaone2 driver and tools/kexaone2_reference.py.

## modeling_ling_bailing_moe_v3.py
- Source: inclusionAI/Ling-3.0-flash @ e0dfe7cd0f6e3b572bbbc0a8a84947469e428cc3 (modeling_bailing_moe_v3.py, repo remote code)
- Fetched: 2026-09-07 from huggingface.co (resolve/main)
- sha256 prefix: c2509bf7ac580c26
- Purpose: ling M1 semantics ground truth — BailingMoeV3 layer dispatch
  ((layer+1)%6==0 -> MultiLatentAttention else KimiDeltaAttention;
  7 MLA + 35 KDA of 42), MLA q_pass(nope 128)/q_rot(rope 64) split with
  interleave rope and head_wise sigmoid g_proj gate, KDA full-rank
  f_proj/g_proj + dt_bias + A_log + sigmoid b_proj + FusedRMSNormGated,
  MTP layer 42 = full MLA+MoE layer with enorm/hnorm/eh_proj and its own
  final_layernorm. The checkpoint census (63,783 tensors) agrees: 8
  kv_b_proj (7 stack + MTP), 43 g_proj (all layers), 41 routers.

## modeling_qwen4_exp.py
- Source: huggingface/transformers main, src/transformers/models/qwen4_exp/modeling_qwen4_exp.py
- Fetched: 2026-08-28 from the controller (raw.githubusercontent.com)
- sha256 prefix: 77fec77d87f2a0eb
- Purpose: qwen-flash M5.2/M5.4/M5.5 semantics — Qwen4ExpTextGatedResidual
  (the "hyper-connection" class, GR parameterization: low-rank up/down +
  block_inject mixers), Qwen4ExpTextQSAIndexer, Qwen4ExpTextNGramEmbedding
  (the PLE ngram table), plus GatedDeltaNet/QSA/experts/router references.
  Same pattern as glm53's modeling_glm5_next.py pin.

## hy4_preview_config.json
- Source: tencent/Hy4-preview-FP8 @4215ec29de873a998e849cee902654490c7ff4d1 (config.json)
- Fetched: 2026-09-01 from the warm copy (hy4-preview-fp8-official, DOWNLOAD-RECEIPT.json)
- sha256 prefix: see shasum below (committed bytes are the pin)
- Purpose: hy4 M1 semantics ground truth — HYV4ForCausalLM geometry (gated MLA
  kv_lora 512/q_lora 2048, DSA indexer 32x128 top-k 2048 full-every-4th,
  256+1 MoE elementwise 8/tok swiglu_limit 10, hc_mult 4 magnitude 2.0,
  learnable sink, 1M ctx rope theta 1e7, num_nextn_predict_layers=1).
  The PACK source of truth is the AngelSlim UD-IQ1_M GGUF (operator ruling);
  this config is the publisher-geometry reference both artifacts share.
