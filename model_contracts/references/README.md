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
