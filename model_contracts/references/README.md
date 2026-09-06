# Pinned modeling references (semantics ground truth for kernel ports)

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
